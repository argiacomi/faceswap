#!/usr/bin/env python3
"""Geometry-risk alignment resolver for extract alignment (#78).

The resolver routes each face through one of three paths based on
geometry-risk evidence, **not** by predicting whether a face is "profile" or
"frontal":

* **low_risk**   — candidates agree closely, the geometry signals look healthy,
  so the general (config-driven) strategy runs.
* **high_risk**  — candidates disagree, hard-case signals fire, or the bbox is
  unusual; the resolver swaps in the hard-case strategy.
* **invalid**    — at least one candidate is degenerate (cloud collapse, eye-
  mouth flip, hull-mismatched). The resolver drops invalid candidates and
  falls back to ``fallback_strategy``; if no candidate remains, it raises a
  resolver-specific error so callers can decide whether to skip or hard-fail.

The resolver is intentionally rule-based for v1: each signal threshold is
inspectable and documented, no learned model is involved. Promotion of a
signal into the runtime path needs validation evidence (#80) before the
threshold is tuned.

This module is the offline scaffold. The runtime Ensemble plugin (#66) can
delegate to :func:`resolve_alignment_geometry` behind a config flag once
signal thresholds are calibrated; the function itself is pure and works on
cached predictions too.
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.eval.geometry_signals import (
    alignment_summary,
    cloud_collapse,
    eye_mouth_flip,
    points_outside_bbox,
)
from lib.landmarks.eval.roi_diagnostics import DEFAULT_LANDMARK_COVERAGE_FLOOR
from lib.landmarks.fusion import (
    FusionResult,
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.rejection import weighted_median
from lib.landmarks.schema import CANONICAL_SCHEMA, LandmarkPrediction

#: Inter-model disagreement (mean pairwise per-landmark distance, in pixels)
#: above which the resolver routes to the hard-case path.
DEFAULT_HIGH_DISAGREEMENT_PX: float = 12.0

#: Fraction of bbox-aspect deviation from 1.0 above which we consider the
#: detector bbox unusual (very wide or very tall) — hard-case signal.
DEFAULT_UNUSUAL_ASPECT_DELTA: float = 0.6

#: ROI / face-bbox ratio outside [min, max] is considered a crop-risk signal.
#: AlignedFace pads the ROI for any non-zero coverage, so the floor must
#: stay loose; only extreme mismatches (e.g. tiny or runaway crops) should
#: trip this signal.
DEFAULT_ROI_RATIO_MIN: float = 0.4
DEFAULT_ROI_RATIO_MAX: float = 3.0


class AlignmentResolverError(RuntimeError):
    """Raised when no valid candidate survives the resolver."""


@dataclass(frozen=True)
class AlignmentResolverConfig:
    """Configuration for :func:`resolve_alignment_geometry`."""

    general_strategy: str = "static_weighted"
    hard_case_strategy: str = "static_weighted_downweight"
    fallback_strategy: str = "plain_average"
    outlier_threshold: float = 3.5
    weights: T.Mapping[str, T.Sequence[float]] | None = None  # for weighted strategies

    # Risk-routing thresholds (all documented above).
    high_disagreement_px: float = DEFAULT_HIGH_DISAGREEMENT_PX
    unusual_bbox_aspect_delta: float = DEFAULT_UNUSUAL_ASPECT_DELTA
    roi_ratio_min: float = DEFAULT_ROI_RATIO_MIN
    roi_ratio_max: float = DEFAULT_ROI_RATIO_MAX
    landmark_coverage_floor: float = DEFAULT_LANDMARK_COVERAGE_FLOOR
    min_models_after_rejection: int = 1


@dataclass(frozen=True)
class CandidateInput:
    """One candidate landmark prediction plus its model name."""

    model: str
    landmarks: np.ndarray  # (68, 2) canonical


@dataclass(frozen=True)
class AlignmentGeometryResult:
    """Resolver decision plus the resulting fused landmark geometry."""

    chosen_strategy: str
    risk_route: str  # "low_risk" / "high_risk" / "invalid"
    risk_score: float
    geometry_confidence: float
    geometry_flags: tuple[str, ...]
    rejected_models: tuple[str, ...]
    active_models: tuple[str, ...]
    alignment_landmarks: np.ndarray
    canonical_landmarks: np.ndarray
    visible_mask_landmarks: np.ndarray
    debug_metadata: dict[str, T.Any] = field(default_factory=dict)


def _pairwise_disagreement(stack: np.ndarray) -> dict[int, float]:
    """Mean pairwise per-landmark distance (in pixel units) per model."""
    if stack.shape[0] < 2:
        return {0: 0.0}
    diffs = np.linalg.norm(stack[:, None] - stack[None, :], axis=-1)
    mean_per_model = diffs.mean(axis=1).mean(axis=1)
    return {idx: float(value) for idx, value in enumerate(mean_per_model)}


def _evaluate_validity(
    candidate: CandidateInput,
    bbox: T.Sequence[float] | None,
    *,
    reference_extent: float,
    landmark_count: int,
    outside_bbox_fraction_floor: float = 0.35,
) -> tuple[bool, tuple[str, ...]]:
    """Return ``(is_valid, geometry_flags)`` for a candidate.

    Invalid candidates trip cloud-collapse against the reference extent (the
    largest candidate cloud), eye-mouth flip in normalized space, or have an
    unusually large fraction of landmarks outside the (inflated) face bbox.
    """
    flags: list[str] = []
    candidate_extent = _bbox_diagonal(candidate.landmarks)
    if candidate_extent <= 1e-6:
        flags.append("cloud_collapse")
    elif reference_extent > 0 and cloud_collapse(
        candidate.landmarks,
        truth_extent=reference_extent,
    ):
        flags.append("cloud_collapse")

    # Avoid running AlignedFace on a collapsed landmark cloud. Umeyama cannot
    # estimate scale from zero-variance points and emits a divide-by-zero
    # RuntimeWarning before the resolver rejects the candidate anyway.
    if "cloud_collapse" in flags:
        return (False, tuple(flags))

    summary = alignment_summary(candidate.landmarks)
    if eye_mouth_flip(summary):
        flags.append("eye_mouth_flip")
    if bbox is not None and landmark_count > 0:
        outside = points_outside_bbox(candidate.landmarks, bbox=bbox)
        if outside / landmark_count > outside_bbox_fraction_floor:
            flags.append("points_outside_bbox")
    return (not flags, tuple(flags))


def _bbox_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return math.hypot(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]))


def _detect_unusual_bbox(
    bbox: T.Sequence[float] | None,
    config: AlignmentResolverConfig,
) -> bool:
    """Return True when the detector bbox aspect ratio drifts far from 1.0."""
    if bbox is None:
        return False
    left, top, right, bottom = (float(value) for value in bbox)
    width = max(right - left, 1e-6)
    height = max(bottom - top, 1e-6)
    aspect = width / height
    return abs(aspect - 1.0) > config.unusual_bbox_aspect_delta


def _detect_roi_ratio_outlier(
    candidate: CandidateInput,
    bbox: T.Sequence[float] | None,
    config: AlignmentResolverConfig,
) -> bool:
    """Return True when the candidate's ROI is dramatically smaller/larger than the bbox."""
    if bbox is None:
        return False
    bbox_diag = _bbox_diagonal(np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype="float64"))
    if bbox_diag <= 0:
        return False
    summary = alignment_summary(candidate.landmarks)
    roi_diag = _bbox_diagonal(summary.roi)
    if roi_diag <= 0:
        return False
    ratio = roi_diag / bbox_diag
    return ratio < config.roi_ratio_min or ratio > config.roi_ratio_max


def _fuse(
    strategy: str,
    candidates: T.Sequence[CandidateInput],
    config: AlignmentResolverConfig,
) -> FusionResult:
    """Run the chosen strategy over the active candidates."""
    canonical = canonical_strategy(strategy)
    items = [
        LandmarkPrediction(candidate.landmarks.astype("float32"), source=candidate.model)
        for candidate in candidates
    ]
    method = strategy_outlier_method(canonical)
    threshold = config.outlier_threshold if strategy_uses_threshold(canonical) else 3.5
    if not strategy_requires_weights(canonical):
        return plain_average(items, outlier_method=method, outlier_threshold=threshold)
    if config.weights is None:
        # Fall back to equal weights when no static weights are configured.
        weights = {candidate.model: [1.0] * 68 for candidate in candidates}
    else:
        weights = {model: config.weights[model] for model in (c.model for c in candidates)}
    matrix = np.array([weights[model] for model in (c.model for c in candidates)], dtype="float32")
    if canonical == "weighted_median":
        stack = np.stack([item.canonical_68().points for item in items], axis=0)
        normalized = normalize_weight_matrix(
            matrix, model_count=stack.shape[0], landmark_count=stack.shape[1]
        )
        return FusionResult(
            points=weighted_median(stack, normalized),
            schema=CANONICAL_SCHEMA,
            strategy=canonical,
            weights=normalized,
            sources=tuple(c.model for c in candidates),
            kept_indices=tuple(range(len(candidates))),
        )
    return static_weighted(
        items,
        matrix,
        outlier_method=method,
        outlier_threshold=threshold,
    )


def resolve_alignment_geometry(
    candidates: T.Sequence[CandidateInput],
    *,
    config: AlignmentResolverConfig,
    detector_bbox: T.Sequence[float] | None = None,
    image_shape: tuple[int, int] | None = None,  # reserved for future use
) -> AlignmentGeometryResult:
    """Route a face through low_risk / high_risk / invalid paths.

    The function is pure: it consumes already-computed candidate landmarks and
    returns a fully described decision. The runtime plugin can call it on every
    face once the thresholds are validated; offline tools can call it on cached
    predictions to inspect what the resolver would have done.
    """
    if not candidates:
        raise AlignmentResolverError("no candidates supplied to resolve_alignment_geometry")

    stack = np.stack([candidate.landmarks for candidate in candidates], axis=0)
    # Use the largest candidate cloud as the reference so a single collapsed
    # candidate is detectable even when other candidates are healthy. When
    # every candidate collapses to the same tiny cloud, fall back to the
    # detector bbox diagonal.
    per_candidate_extent = [_bbox_diagonal(candidate.landmarks) for candidate in candidates]
    reference_extent = max(per_candidate_extent) if per_candidate_extent else 0.0
    if reference_extent <= 0 and detector_bbox is not None:
        bbox_arr = np.array(
            [[detector_bbox[0], detector_bbox[1]], [detector_bbox[2], detector_bbox[3]]],
            dtype="float64",
        )
        reference_extent = _bbox_diagonal(bbox_arr)
    landmark_count = int(stack.shape[1]) if stack.ndim == 3 else 0
    flags: list[str] = []
    rejected: list[str] = []

    # Per-candidate validity check.
    valid_indices: list[int] = []
    per_candidate_flags: dict[str, tuple[str, ...]] = {}
    for idx, candidate in enumerate(candidates):
        is_valid, candidate_flags = _evaluate_validity(
            candidate,
            detector_bbox,
            reference_extent=reference_extent,
            landmark_count=landmark_count,
        )
        per_candidate_flags[candidate.model] = candidate_flags
        if is_valid:
            valid_indices.append(idx)
        else:
            rejected.append(candidate.model)
    if not valid_indices:
        raise AlignmentResolverError(
            "all candidates failed validity checks: "
            + ", ".join(f"{model}={flags}" for model, flags in per_candidate_flags.items())
        )

    active_candidates = [candidates[idx] for idx in valid_indices]
    active_stack = np.stack([candidate.landmarks for candidate in active_candidates], axis=0)

    # Disagreement-based risk score (mean pairwise per-landmark distance).
    disagreement = _pairwise_disagreement(active_stack)
    max_disagreement_px = max(disagreement.values()) if disagreement else 0.0

    risk_signals: list[str] = []
    if max_disagreement_px > config.high_disagreement_px:
        risk_signals.append("high_disagreement")
    if _detect_unusual_bbox(detector_bbox, config):
        risk_signals.append("unusual_bbox_aspect")
    if detector_bbox is not None and any(
        _detect_roi_ratio_outlier(candidate, detector_bbox, config)
        for candidate in active_candidates
    ):
        risk_signals.append("roi_ratio_outlier")
    if len(active_candidates) < len(candidates):
        risk_signals.append("partial_candidate_rejection")

    if len(active_candidates) < config.min_models_after_rejection:
        chosen_strategy = config.fallback_strategy
        risk_route = "invalid"
    elif risk_signals:
        chosen_strategy = config.hard_case_strategy
        risk_route = "high_risk"
    else:
        chosen_strategy = config.general_strategy
        risk_route = "low_risk"

    # Risk score: bounded by max_disagreement / threshold (0..N), with each
    # extra qualitative signal adding 1.0. Mostly useful as a diagnostic.
    raw_score = max_disagreement_px / max(config.high_disagreement_px, 1e-6)
    risk_score = float(raw_score + len(risk_signals))

    confidence_components = [
        1.0 if len(rejected) == 0 else 0.5,
        max(0.0, 1.0 - raw_score),
    ]
    geometry_confidence = float(max(0.0, min(1.0, np.mean(confidence_components))))

    try:
        fused = _fuse(chosen_strategy, active_candidates, config)
    except ValueError as err:
        # Fall back when the chosen strategy cannot run (e.g. weights missing).
        flags.append(f"fusion_fallback:{err.__class__.__name__}")
        fused = _fuse(config.fallback_strategy, active_candidates, config)
        chosen_strategy = config.fallback_strategy
        risk_route = "invalid" if risk_route != "low_risk" else risk_route

    flags.extend(risk_signals)
    for candidate_flags in per_candidate_flags.values():
        flags.extend(candidate_flags)

    debug: dict[str, T.Any] = {
        "max_disagreement_px": max_disagreement_px,
        "per_model_disagreement_px": {
            active_candidates[idx].model: value for idx, value in disagreement.items()
        },
        "per_candidate_flags": {
            model: list(values) for model, values in per_candidate_flags.items()
        },
        "rejected_models": list(rejected),
        "risk_signals": list(risk_signals),
        "chosen_strategy": chosen_strategy,
    }

    return AlignmentGeometryResult(
        chosen_strategy=chosen_strategy,
        risk_route=risk_route,
        risk_score=risk_score,
        geometry_confidence=geometry_confidence,
        geometry_flags=tuple(dict.fromkeys(flags)),
        rejected_models=tuple(rejected),
        active_models=tuple(candidate.model for candidate in active_candidates),
        alignment_landmarks=fused.points.astype("float32"),
        canonical_landmarks=fused.points.astype("float32"),
        visible_mask_landmarks=fused.points.astype("float32"),
        debug_metadata=debug,
    )


__all__ = [
    "AlignmentGeometryResult",
    "AlignmentResolverConfig",
    "AlignmentResolverError",
    "CandidateInput",
    "DEFAULT_HIGH_DISAGREEMENT_PX",
    "DEFAULT_ROI_RATIO_MAX",
    "DEFAULT_ROI_RATIO_MIN",
    "DEFAULT_UNUSUAL_ASPECT_DELTA",
    "resolve_alignment_geometry",
]
