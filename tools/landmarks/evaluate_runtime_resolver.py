#!/usr/bin/env python3
"""Evaluate a runtime resolver policy against cached candidate predictions.

For each manifest sample this tool reads single-model predictions from
``--cache-dir``, optionally fuses ensemble variants (``static_weighted``,
``static_weighted_downweight``, …) using ``--weights``, applies the named
``--policy`` to pick a winning candidate, and compares the resolver's
choice against the per-sample oracle (the cohort candidate with the
lowest NME against ground truth).

The runtime resolver itself does not see the GT; this harness consumes
the cached predictions plus the manifest's GT landmarks and asks
"how often does the policy match the oracle, and by how much when it
doesn't?". That's the offline calibration step before promoting a
policy into ``plugins/extract/align/ensemble.py``.

Outputs (written under ``--output-dir``):

    resolver_policy_report.json   aggregate stats (policy + per candidate)
    resolver_policy_report.csv    flat per-sample row
    resolver_failures.csv         resolver picked NME > failure threshold
    resolver_worst_samples.json   top-N worst gaps vs oracle
    resolver_overlays/            PNG overlays for failures + worst-N
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.fusion_variants import fuse_variant
from lib.landmarks.core.schema import LandmarkPrediction, normalize_landmarks
from lib.landmarks.datasets.manifest_io import (
    LandmarkSample,
    bbox_from_truth_fallback,
    load_manifest,
)
from lib.landmarks.ensemble.strategies import canonical_strategy
from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.evaluation.geometry_signals import AlignmentSummary, alignment_summary
from lib.landmarks.evaluation.nme_metrics import evaluate_prediction

logger = logging.getLogger("evaluate_runtime_resolver")

#: Roll deviation (in degrees) above which a candidate is vetoed for the
#: ``roll_aware_veto`` policy.
DEFAULT_ROLL_VETO_THRESHOLD_DEG: float = 15.0

#: NME above which a sample is counted as a resolver failure.
DEFAULT_FAILURE_THRESHOLD: float = 0.08

#: Number of worst samples surfaced in ``resolver_worst_samples.json`` and overlays.
DEFAULT_WORST_COUNT: int = 20

# Prediction-only geometry veto thresholds. These are intentionally broad:
# they should catch malformed landmark clouds / flipped face geometry without
# becoming another noisy oracle selector.
DEFAULT_MIN_CLOUD_AREA_RATIO: float = 0.08
DEFAULT_MAX_CLOUD_AREA_RATIO: float = 3.00
DEFAULT_MIN_HULL_AREA_RATIO: float = 0.035
DEFAULT_MAX_HULL_AREA_RATIO: float = 2.25
DEFAULT_MAX_POINTS_OUTSIDE_EXPANDED_BBOX_FRACTION: float = 0.35
DEFAULT_MAX_ROI_CENTER_CONSENSUS_DISTANCE: float = 0.22
DEFAULT_MAX_LANDMARK_CONSENSUS_DISTANCE: float = 0.16


@dataclass(frozen=True)
class CandidateRecord:
    """A single resolver candidate plus its derived diagnostics."""

    name: str
    landmarks: np.ndarray  # (68, 2) canonical
    is_fusion: bool
    contributing_models: tuple[str, ...]


@dataclass
class CandidateMetrics:
    """NME-side metrics plus prediction-only geometry diagnostics."""

    nme: float
    failure: bool
    roll_degrees: float | None
    yaw_degrees: float | None
    pitch_degrees: float | None
    cloud_area_ratio: float | None = None
    hull_area_ratio: float | None = None
    points_outside_expanded_bbox_fraction: float | None = None
    eye_mouth_order_valid_after_deroll: bool | None = None
    roi_center_consensus_distance: float | None = None
    landmark_consensus_distance: float | None = None
    geometry_veto_reasons: tuple[str, ...] = ()


@dataclass
class PolicyDecision:
    """Output of a resolver policy for a single sample."""

    policy: str
    chosen: str
    vetoed: tuple[str, ...]
    consensus_roll_deg: float | None
    diagnostics: dict[str, T.Any] = field(default_factory=dict)


@dataclass
class SampleReport:
    """All resolver diagnostics for one manifest sample."""

    sample_id: str
    dataset: str
    condition: str
    candidates: tuple[str, ...]
    metrics: dict[str, CandidateMetrics]
    decision: PolicyDecision
    oracle: str
    image_path: str
    truth: np.ndarray
    landmarks_by_candidate: dict[str, np.ndarray]


# --------------------------------------------------------------------------- #
# Candidate construction
# --------------------------------------------------------------------------- #


def _single_model_predictions(
    cache: DiskPredictionCache,
    sample: LandmarkSample,
    model_names: T.Sequence[str],
) -> dict[str, LandmarkPrediction]:
    """Return cached single-model predictions for one sample, keyed by name."""
    available = set(cache.available_models(sample.sample_id))
    missing = [name for name in model_names if name not in available]
    if missing:
        raise FileNotFoundError(f"sample {sample.sample_id!r} missing cached models: {missing}")
    return {name: cache.read(sample.sample_id, name) for name in model_names}


def _build_candidates(
    sample: LandmarkSample,
    cache: DiskPredictionCache,
    requested: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]] | None,
    *,
    outlier_threshold: float,
) -> list[CandidateRecord]:
    """Build the candidate list (single + fused) for one sample."""
    single_names = []
    fusion_names = []
    for name in requested:
        try:
            canonical_strategy(name)
            fusion_names.append(name)
        except (KeyError, ValueError):
            single_names.append(name)

    fusion_models = tuple(weights or ())
    needed = tuple(sorted(set(single_names) | set(fusion_models)))
    if not needed:
        raise ValueError("no single or fusion models requested — nothing to evaluate")
    predictions = _single_model_predictions(cache, sample, needed)

    candidates: list[CandidateRecord] = []
    for name in single_names:
        candidates.append(
            CandidateRecord(
                name=name,
                landmarks=predictions[name].landmarks.astype("float32"),
                is_fusion=False,
                contributing_models=(name,),
            )
        )
    if fusion_names and not fusion_models:
        raise ValueError("fusion variants requested but --weights resolved to an empty model set")
    for variant in fusion_names:
        fused = fuse_variant(
            variant,
            [predictions[m] for m in fusion_models],
            models=fusion_models,
            weights=weights,
            outlier_threshold=outlier_threshold,
        )
        candidates.append(
            CandidateRecord(
                name=variant,
                landmarks=fused.astype("float32"),
                is_fusion=True,
                contributing_models=fusion_models,
            )
        )
    return candidates


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _safe_alignment_summary(landmarks: np.ndarray) -> AlignmentSummary | None:
    """Run :func:`alignment_summary` and swallow recoverable failures."""
    try:
        return alignment_summary(normalize_landmarks(landmarks))
    except Exception as err:  # noqa: BLE001 — defensive at the boundary
        logger.debug("alignment_summary failed: %s", err)
        return None


def _landmark_bbox(points: np.ndarray) -> tuple[float, float, float, float] | None:
    arr = np.asarray(points, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0 or not np.all(np.isfinite(arr[:, :2])):
        return None
    left, top = np.min(arr[:, :2], axis=0)
    right, bottom = np.max(arr[:, :2], axis=0)
    if right <= left or bottom <= top:
        return None
    return (float(left), float(top), float(right), float(bottom))


def _bbox_area(bbox: tuple[float, float, float, float] | None) -> float | None:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    return float(width * height)


def _bbox_center(bbox: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return None
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def _bbox_diag(bbox: tuple[float, float, float, float] | None) -> float | None:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    diag = math.hypot(right - left, bottom - top)
    return float(diag) if diag > 0 else None


def _expanded_bbox(
    bbox: tuple[float, float, float, float] | None,
    *,
    margin_ratio: float = 0.25,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    margin_x = margin_ratio * width
    margin_y = margin_ratio * height
    return (left - margin_x, top - margin_y, right + margin_x, bottom + margin_y)


def _points_outside_bbox_fraction(
    points: np.ndarray, bbox: tuple[float, float, float, float] | None
) -> float | None:
    if bbox is None:
        return None
    arr = np.asarray(points, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0:
        return None
    left, top, right, bottom = bbox
    outside = (arr[:, 0] < left) | (arr[:, 0] > right) | (arr[:, 1] < top) | (arr[:, 1] > bottom)
    return float(np.mean(outside))


def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    unique = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype="float64")[:, :2]})
    if len(unique) <= 1:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_area(points: T.Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    arr = np.asarray(points, dtype="float64")
    x = arr[:, 0]
    y = arr[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _convex_hull_area(points: np.ndarray) -> float | None:
    try:
        return _polygon_area(_convex_hull(points))
    except (TypeError, ValueError, IndexError):
        return None


def _eye_mouth_order_valid_after_deroll(points: np.ndarray) -> bool | None:
    arr = np.asarray(points, dtype="float64")
    if arr.shape[0] < 68 or arr.shape[1] < 2 or not np.all(np.isfinite(arr[:, :2])):
        return None
    left_eye = arr[36:42, :2].mean(axis=0)
    right_eye = arr[42:48, :2].mean(axis=0)
    mouth = arr[48:68, :2].mean(axis=0)
    eye_mid = (left_eye + right_eye) / 2.0
    eye_vector = right_eye - left_eye
    if float(np.linalg.norm(eye_vector)) <= 1e-6:
        return None
    angle = math.atan2(float(eye_vector[1]), float(eye_vector[0]))
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype="float64")
    derolled_eye = (eye_mid - eye_mid) @ rotation.T
    derolled_mouth = (mouth - eye_mid) @ rotation.T
    # Image coordinates have y increasing downward, so the mouth should remain
    # below the eye midpoint after de-rolling.
    return bool(derolled_mouth[1] > derolled_eye[1])


def _geometry_veto_reasons(metric: CandidateMetrics) -> tuple[str, ...]:
    reasons: list[str] = []
    if metric.cloud_area_ratio is None:
        reasons.append("missing_cloud_area_ratio")
    elif metric.cloud_area_ratio < DEFAULT_MIN_CLOUD_AREA_RATIO:
        reasons.append("cloud_area_too_small")
    elif metric.cloud_area_ratio > DEFAULT_MAX_CLOUD_AREA_RATIO:
        reasons.append("cloud_area_too_large")

    if metric.hull_area_ratio is None:
        reasons.append("missing_hull_area_ratio")
    elif metric.hull_area_ratio < DEFAULT_MIN_HULL_AREA_RATIO:
        reasons.append("hull_area_too_small")
    elif metric.hull_area_ratio > DEFAULT_MAX_HULL_AREA_RATIO:
        reasons.append("hull_area_too_large")

    if (
        metric.points_outside_expanded_bbox_fraction is not None
        and metric.points_outside_expanded_bbox_fraction
        > DEFAULT_MAX_POINTS_OUTSIDE_EXPANDED_BBOX_FRACTION
    ):
        reasons.append("too_many_points_outside_expanded_bbox")

    if metric.eye_mouth_order_valid_after_deroll is False:
        reasons.append("eye_mouth_order_invalid_after_deroll")

    if (
        metric.roi_center_consensus_distance is not None
        and metric.roi_center_consensus_distance > DEFAULT_MAX_ROI_CENTER_CONSENSUS_DISTANCE
    ):
        reasons.append("roi_center_consensus_distance_high")

    if (
        metric.landmark_consensus_distance is not None
        and metric.landmark_consensus_distance > DEFAULT_MAX_LANDMARK_CONSENSUS_DISTANCE
    ):
        reasons.append("landmark_consensus_distance_high")
    return tuple(reasons)


def _evaluate_candidate(
    landmarks: np.ndarray,
    truth: np.ndarray,
    *,
    normalizer: float | None,
    visibility: T.Sequence[bool] | None,
    failure_threshold: float,
    face_bbox: tuple[float, float, float, float] | None = None,
) -> CandidateMetrics:
    """Compute per-candidate NME, pose, and standalone geometry diagnostics."""
    metrics = evaluate_prediction(
        landmarks,
        truth,
        normalizer=normalizer,
        failure_threshold=failure_threshold,
        visibility=visibility,
    )
    summary = _safe_alignment_summary(landmarks)
    reference_bbox = face_bbox or bbox_from_truth_fallback(truth)
    reference_area = _bbox_area(reference_bbox)
    candidate_bbox = _landmark_bbox(landmarks)
    candidate_area = _bbox_area(candidate_bbox)
    hull_area = _convex_hull_area(landmarks)
    expanded_reference = _expanded_bbox(reference_bbox)
    return CandidateMetrics(
        nme=float(metrics["nme"]),
        failure=bool(metrics["failure"]),
        roll_degrees=None if summary is None else float(summary.roll),
        yaw_degrees=None if summary is None else float(summary.yaw),
        pitch_degrees=None if summary is None else float(summary.pitch),
        cloud_area_ratio=(
            None
            if reference_area is None or candidate_area is None
            else candidate_area / reference_area
        ),
        hull_area_ratio=(
            None if reference_area is None or hull_area is None else hull_area / reference_area
        ),
        points_outside_expanded_bbox_fraction=_points_outside_bbox_fraction(
            landmarks, expanded_reference
        ),
        eye_mouth_order_valid_after_deroll=_eye_mouth_order_valid_after_deroll(landmarks),
    )


def _populate_consensus_geometry(
    candidates: T.Sequence[CandidateRecord],
    metrics: T.MutableMapping[str, CandidateMetrics],
    *,
    reference_bbox: tuple[float, float, float, float] | None,
) -> None:
    if not candidates:
        return
    diag = _bbox_diag(reference_bbox)
    if diag is None:
        return
    stack = np.stack([candidate.landmarks.astype("float64") for candidate in candidates], axis=0)
    consensus_points = np.median(stack, axis=0)
    consensus_bbox = _landmark_bbox(consensus_points)
    consensus_center = _bbox_center(consensus_bbox)
    for candidate in candidates:
        metric = metrics[candidate.name]
        candidate_bbox = _landmark_bbox(candidate.landmarks)
        candidate_center = _bbox_center(candidate_bbox)
        if consensus_center is not None and candidate_center is not None:
            metric.roi_center_consensus_distance = float(
                math.hypot(
                    candidate_center[0] - consensus_center[0],
                    candidate_center[1] - consensus_center[1],
                )
                / diag
            )
        metric.landmark_consensus_distance = float(
            np.mean(
                np.linalg.norm(candidate.landmarks.astype("float64") - consensus_points, axis=1)
            )
            / diag
        )
        metric.geometry_veto_reasons = _geometry_veto_reasons(metric)


# --------------------------------------------------------------------------- #
# Resolver policies
# --------------------------------------------------------------------------- #


def _circular_median(values: T.Sequence[float]) -> float:
    """Median of angle values in degrees, wrapping at ±180°."""
    radians = np.deg2rad(np.asarray(list(values), dtype="float64"))
    mean_cos = float(np.mean(np.cos(radians)))
    mean_sin = float(np.mean(np.sin(radians)))
    centre = np.arctan2(mean_sin, mean_cos)
    wrapped = np.mod(radians - centre + np.pi, 2 * np.pi) - np.pi
    return float(np.rad2deg(centre + np.median(wrapped)))


def resolve_roll_aware_veto(
    candidates: T.Sequence[CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    *,
    veto_threshold_deg: float = DEFAULT_ROLL_VETO_THRESHOLD_DEG,
) -> PolicyDecision:
    """Veto candidates whose roll diverges from the cohort consensus."""
    if not candidates:
        raise ValueError("roll_aware_veto requires at least one candidate")
    rolls = [metrics[c.name].roll_degrees for c in candidates]
    valid_rolls = [r for r in rolls if r is not None]
    if not valid_rolls:
        return PolicyDecision(
            policy="roll_aware_veto",
            chosen=candidates[0].name,
            vetoed=tuple(c.name for c in candidates[1:]),
            consensus_roll_deg=None,
            diagnostics={
                "fallback_reason": "no_candidate_pose_estimate",
                "rolls": {c.name: rolls[i] for i, c in enumerate(candidates)},
            },
        )
    consensus = _circular_median(valid_rolls)
    survivors: list[tuple[CandidateRecord, float]] = []
    vetoed: list[str] = []
    for candidate, roll in zip(candidates, rolls, strict=False):
        if roll is None:
            vetoed.append(candidate.name)
            continue
        delta = abs(_signed_degree_delta(roll, consensus))
        if delta > veto_threshold_deg:
            vetoed.append(candidate.name)
        else:
            survivors.append((candidate, delta))
    diagnostics: dict[str, T.Any] = {
        "rolls": {c.name: rolls[i] for i, c in enumerate(candidates)},
        "veto_threshold_deg": veto_threshold_deg,
    }
    if not survivors:
        diagnostics["fallback_reason"] = "all_candidates_vetoed"
        deltas = [
            (candidate, abs(_signed_degree_delta(roll, consensus)))
            for candidate, roll in zip(candidates, rolls, strict=False)
            if roll is not None
        ]
        chosen = min(deltas, key=lambda item: item[1])[0]
    else:
        chosen = min(survivors, key=lambda item: item[1])[0]
    return PolicyDecision(
        policy="roll_aware_veto",
        chosen=chosen.name,
        vetoed=tuple(vetoed),
        consensus_roll_deg=consensus,
        diagnostics=diagnostics,
    )


def _signed_degree_delta(a: float, b: float) -> float:
    """Return the signed angular delta ``a - b`` wrapped to (-180, 180]."""
    return float(((a - b) + 180.0) % 360.0 - 180.0)


POLICY_REGISTRY: dict[
    str,
    T.Callable[[T.Sequence[CandidateRecord], T.Mapping[str, CandidateMetrics]], PolicyDecision],
] = {
    "roll_aware_veto": resolve_roll_aware_veto,
}


def apply_policy(
    name: str,
    candidates: T.Sequence[CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
) -> PolicyDecision:
    """Dispatch to the named policy with a clear error for unknown names."""
    if name not in POLICY_REGISTRY:
        raise ValueError(f"unknown resolver policy {name!r}; available: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name](candidates, metrics)


# --------------------------------------------------------------------------- #
# Sample loop
# --------------------------------------------------------------------------- #


def _load_truth(sample: LandmarkSample) -> np.ndarray:
    return np.load(sample.landmarks).astype("float32")


def evaluate_sample(
    sample: LandmarkSample,
    *,
    cache: DiskPredictionCache,
    requested: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]] | None,
    policy: str,
    outlier_threshold: float,
    failure_threshold: float,
) -> SampleReport:
    """Build candidates, score them against GT, apply the policy."""
    truth = _load_truth(sample)
    candidates = _build_candidates(
        sample,
        cache,
        requested,
        weights,
        outlier_threshold=outlier_threshold,
    )
    reference_bbox = sample.face_bbox or bbox_from_truth_fallback(truth)
    metrics: dict[str, CandidateMetrics] = {}
    for candidate in candidates:
        metrics[candidate.name] = _evaluate_candidate(
            candidate.landmarks,
            truth,
            normalizer=sample.normalizer,
            visibility=sample.visibility,
            failure_threshold=failure_threshold,
            face_bbox=reference_bbox,
        )
    _populate_consensus_geometry(candidates, metrics, reference_bbox=reference_bbox)
    decision = apply_policy(policy, candidates, metrics)
    oracle = min(metrics.items(), key=lambda item: item[1].nme)[0]
    return SampleReport(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        condition=sample.condition,
        candidates=tuple(c.name for c in candidates),
        metrics=metrics,
        decision=decision,
        oracle=oracle,
        image_path=sample.image,
        truth=truth,
        landmarks_by_candidate={c.name: c.landmarks for c in candidates},
    )


# --------------------------------------------------------------------------- #
# Aggregation + output
# --------------------------------------------------------------------------- #


def _summarise_nme(values: T.Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def aggregate_reports(
    reports: T.Sequence[SampleReport],
    *,
    policy: str,
    failure_threshold: float,
) -> dict[str, T.Any]:
    """Produce the JSON aggregate from per-sample reports."""
    if not reports:
        return {"policy": policy, "sample_count": 0}
    candidate_names = reports[0].candidates
    per_candidate_nme: dict[str, list[float]] = {name: [] for name in candidate_names}
    per_candidate_failures: dict[str, int] = {name: 0 for name in candidate_names}
    geometry_veto_counts: dict[str, int] = {name: 0 for name in candidate_names}
    chosen_nme: list[float] = []
    oracle_nme: list[float] = []
    gap_vs_oracle: list[float] = []
    oracle_match = 0
    chosen_failures = 0
    chosen_pick_counts: dict[str, int] = {name: 0 for name in candidate_names}
    veto_counts: dict[str, int] = {name: 0 for name in candidate_names}
    fallback_counts: dict[str, int] = {}

    for report in reports:
        for name, metric in report.metrics.items():
            per_candidate_nme[name].append(metric.nme)
            if metric.failure:
                per_candidate_failures[name] += 1
            if metric.geometry_veto_reasons:
                geometry_veto_counts[name] += 1
        chosen_metric = report.metrics[report.decision.chosen]
        chosen_nme.append(chosen_metric.nme)
        oracle_nme.append(report.metrics[report.oracle].nme)
        gap_vs_oracle.append(chosen_metric.nme - report.metrics[report.oracle].nme)
        if chosen_metric.failure:
            chosen_failures += 1
        if report.decision.chosen == report.oracle:
            oracle_match += 1
        chosen_pick_counts[report.decision.chosen] = (
            chosen_pick_counts.get(report.decision.chosen, 0) + 1
        )
        for vetoed in report.decision.vetoed:
            veto_counts[vetoed] = veto_counts.get(vetoed, 0) + 1
        fallback_reason = report.decision.diagnostics.get("fallback_reason")
        if fallback_reason:
            fallback_counts[fallback_reason] = fallback_counts.get(fallback_reason, 0) + 1

    sample_count = len(reports)
    return {
        "policy": policy,
        "failure_threshold": failure_threshold,
        "sample_count": sample_count,
        "candidates": list(candidate_names),
        "chosen": {
            "nme": _summarise_nme(chosen_nme),
            "failure_rate": chosen_failures / sample_count,
            "oracle_match_rate": oracle_match / sample_count,
            "mean_gap_vs_oracle": float(np.mean(gap_vs_oracle)),
            "p90_gap_vs_oracle": float(np.percentile(gap_vs_oracle, 90)),
            "pick_counts": chosen_pick_counts,
            "fallback_counts": fallback_counts,
        },
        "oracle": {"nme": _summarise_nme(oracle_nme)},
        "per_candidate": {
            name: {
                "nme": _summarise_nme(per_candidate_nme[name]),
                "failure_rate": per_candidate_failures[name] / sample_count,
                "veto_count": veto_counts.get(name, 0),
                "geometry_veto_count": geometry_veto_counts.get(name, 0),
            }
            for name in candidate_names
        },
    }


def _candidate_diagnostic_fields(name: str) -> list[str]:
    return [
        f"{name}_cloud_area_ratio",
        f"{name}_hull_area_ratio",
        f"{name}_points_outside_expanded_bbox_fraction",
        f"{name}_eye_mouth_order_valid_after_deroll",
        f"{name}_roi_center_consensus_distance",
        f"{name}_landmark_consensus_distance",
        f"{name}_geometry_veto_reasons",
    ]


def write_csv_report(reports: T.Sequence[SampleReport], path: Path) -> None:
    """Write the flat per-sample CSV with all candidate NMEs and diagnostics."""
    if not reports:
        path.write_text("", encoding="utf-8")
        return
    candidate_names = reports[0].candidates
    fieldnames = [
        "sample_id",
        "dataset",
        "condition",
        "chosen",
        "chosen_nme",
        "chosen_failure",
        "oracle",
        "oracle_nme",
        "gap_vs_oracle",
        "consensus_roll_deg",
        "vetoed",
        "fallback_reason",
    ]
    for name in candidate_names:
        fieldnames.extend([f"{name}_nme", f"{name}_failure", f"{name}_roll_deg"])
        fieldnames.extend(_candidate_diagnostic_fields(name))
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            chosen = report.decision.chosen
            chosen_nme = report.metrics[chosen].nme
            oracle_nme = report.metrics[report.oracle].nme
            row: dict[str, T.Any] = {
                "sample_id": report.sample_id,
                "dataset": report.dataset,
                "condition": report.condition,
                "chosen": chosen,
                "chosen_nme": chosen_nme,
                "chosen_failure": int(report.metrics[chosen].failure),
                "oracle": report.oracle,
                "oracle_nme": oracle_nme,
                "gap_vs_oracle": chosen_nme - oracle_nme,
                "consensus_roll_deg": report.decision.consensus_roll_deg,
                "vetoed": "|".join(report.decision.vetoed),
                "fallback_reason": report.decision.diagnostics.get("fallback_reason", ""),
            }
            for name in candidate_names:
                metric = report.metrics[name]
                row[f"{name}_nme"] = metric.nme
                row[f"{name}_failure"] = int(metric.failure)
                row[f"{name}_roll_deg"] = (
                    "" if metric.roll_degrees is None else metric.roll_degrees
                )
                row[f"{name}_cloud_area_ratio"] = metric.cloud_area_ratio
                row[f"{name}_hull_area_ratio"] = metric.hull_area_ratio
                row[f"{name}_points_outside_expanded_bbox_fraction"] = (
                    metric.points_outside_expanded_bbox_fraction
                )
                row[f"{name}_eye_mouth_order_valid_after_deroll"] = (
                    ""
                    if metric.eye_mouth_order_valid_after_deroll is None
                    else int(metric.eye_mouth_order_valid_after_deroll)
                )
                row[f"{name}_roi_center_consensus_distance"] = metric.roi_center_consensus_distance
                row[f"{name}_landmark_consensus_distance"] = metric.landmark_consensus_distance
                row[f"{name}_geometry_veto_reasons"] = "|".join(metric.geometry_veto_reasons)
            writer.writerow(row)


def write_failures_csv(
    reports: T.Sequence[SampleReport], path: Path, *, failure_threshold: float
) -> None:
    """Write only rows where the resolver's chosen candidate failed."""
    fieldnames = [
        "sample_id",
        "dataset",
        "condition",
        "chosen",
        "chosen_nme",
        "oracle",
        "oracle_nme",
        "gap_vs_oracle",
        "consensus_roll_deg",
        "vetoed",
        "chosen_geometry_veto_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            chosen_metric = report.metrics[report.decision.chosen]
            if chosen_metric.nme <= failure_threshold:
                continue
            oracle_metric = report.metrics[report.oracle]
            writer.writerow(
                {
                    "sample_id": report.sample_id,
                    "dataset": report.dataset,
                    "condition": report.condition,
                    "chosen": report.decision.chosen,
                    "chosen_nme": chosen_metric.nme,
                    "oracle": report.oracle,
                    "oracle_nme": oracle_metric.nme,
                    "gap_vs_oracle": chosen_metric.nme - oracle_metric.nme,
                    "consensus_roll_deg": report.decision.consensus_roll_deg,
                    "vetoed": "|".join(report.decision.vetoed),
                    "chosen_geometry_veto_reasons": "|".join(chosen_metric.geometry_veto_reasons),
                }
            )


def select_worst_samples(reports: T.Sequence[SampleReport], *, count: int) -> list[SampleReport]:
    """Return the samples with the largest gap between chosen and oracle NME."""
    ranked = sorted(
        reports,
        key=lambda r: r.metrics[r.decision.chosen].nme - r.metrics[r.oracle].nme,
        reverse=True,
    )
    return ranked[:count]


def _metric_diagnostics(metric: CandidateMetrics) -> dict[str, T.Any]:
    return {
        "cloud_area_ratio": metric.cloud_area_ratio,
        "hull_area_ratio": metric.hull_area_ratio,
        "points_outside_expanded_bbox_fraction": metric.points_outside_expanded_bbox_fraction,
        "eye_mouth_order_valid_after_deroll": metric.eye_mouth_order_valid_after_deroll,
        "roi_center_consensus_distance": metric.roi_center_consensus_distance,
        "landmark_consensus_distance": metric.landmark_consensus_distance,
        "geometry_veto_reasons": list(metric.geometry_veto_reasons),
    }


def write_worst_samples_json(worst: T.Sequence[SampleReport], path: Path) -> None:
    payload = {
        "samples": [
            {
                "sample_id": r.sample_id,
                "dataset": r.dataset,
                "condition": r.condition,
                "chosen": r.decision.chosen,
                "chosen_nme": r.metrics[r.decision.chosen].nme,
                "oracle": r.oracle,
                "oracle_nme": r.metrics[r.oracle].nme,
                "gap_vs_oracle": r.metrics[r.decision.chosen].nme - r.metrics[r.oracle].nme,
                "consensus_roll_deg": r.decision.consensus_roll_deg,
                "vetoed": list(r.decision.vetoed),
                "rolls": {name: metric.roll_degrees for name, metric in r.metrics.items()},
                "diagnostics": {
                    name: _metric_diagnostics(metric) for name, metric in r.metrics.items()
                },
            }
            for r in worst
        ]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def render_overlay(report: SampleReport, output_path: Path) -> bool:
    """Render a GT / chosen / oracle landmark overlay onto the source image."""
    try:
        import cv2
    except ImportError:  # pragma: no cover
        logger.warning("cv2 unavailable; skipping overlay rendering")
        return False
    image = cv2.imread(report.image_path, cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("could not read image for overlay: %s", report.image_path)
        return False
    canvas = image.copy()
    palette = {
        "truth": (255, 255, 255),
        "chosen": (0, 255, 0),
        "oracle": (255, 0, 0),
    }
    layers = [
        ("truth", report.truth),
        ("chosen", report.landmarks_by_candidate[report.decision.chosen]),
        ("oracle", report.landmarks_by_candidate[report.oracle]),
    ]
    for label, points in layers:
        if points is None:
            continue
        for x, y in points:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, palette[label], -1)
    chosen_nme = report.metrics[report.decision.chosen].nme
    oracle_nme = report.metrics[report.oracle].nme
    legend = (
        f"chosen={report.decision.chosen} nme={chosen_nme:.4f}  "
        f"oracle={report.oracle} nme={oracle_nme:.4f}"
    )
    cv2.putText(
        canvas, legend, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), canvas))


def render_overlays(
    reports: T.Sequence[SampleReport],
    output_dir: Path,
    *,
    failure_threshold: float,
    worst_count: int,
) -> dict[str, int]:
    """Render overlays for failures + the top-N worst gaps."""
    targets: dict[str, SampleReport] = {}
    for report in reports:
        if report.metrics[report.decision.chosen].nme > failure_threshold:
            targets[report.sample_id] = report
    for report in select_worst_samples(reports, count=worst_count):
        targets.setdefault(report.sample_id, report)
    counts = {"requested": len(targets), "written": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, report in targets.items():
        safe = sample_id.replace("/", "_").replace("\\", "_")
        if render_overlay(report, output_dir / f"{safe}.png"):
            counts["written"] += 1
    return counts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a runtime resolver policy against cached candidate predictions."
    )
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    parser.add_argument("--cache-dir", required=True, help="Prediction cache root directory")
    parser.add_argument("--weights", required=True, help="Static weights JSON for fusion variants")
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate names")
    parser.add_argument(
        "--policy",
        default="roll_aware_veto",
        choices=sorted(POLICY_REGISTRY),
        help="Resolver policy to evaluate.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory where reports + overlays are written."
    )
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--worst-count", type=int, default=DEFAULT_WORST_COUNT)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    """Run the resolver evaluation harness from CLI arguments."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    requested = tuple(item.strip() for item in args.candidates.split(",") if item.strip())
    if not requested:
        raise SystemExit("--candidates must list at least one name")
    weights = load_weights(args.weights) if args.weights else None
    cache = DiskPredictionCache(args.cache_dir)
    samples = load_manifest(args.manifest)
    logger.info("evaluating %d samples with policy=%s", len(samples), args.policy)
    reports: list[SampleReport] = []
    for sample in samples:
        try:
            reports.append(
                evaluate_sample(
                    sample,
                    cache=cache,
                    requested=requested,
                    weights=weights,
                    policy=args.policy,
                    outlier_threshold=args.outlier_threshold,
                    failure_threshold=args.failure_threshold,
                )
            )
        except FileNotFoundError as err:
            logger.warning("skipping sample %s: %s", sample.sample_id, err)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_reports(
        reports, policy=args.policy, failure_threshold=args.failure_threshold
    )
    (output_dir / "resolver_policy_report.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    write_csv_report(reports, output_dir / "resolver_policy_report.csv")
    write_failures_csv(
        reports, output_dir / "resolver_failures.csv", failure_threshold=args.failure_threshold
    )
    write_worst_samples_json(
        select_worst_samples(reports, count=args.worst_count),
        output_dir / "resolver_worst_samples.json",
    )
    overlay_counts = render_overlays(
        reports,
        output_dir / "resolver_overlays",
        failure_threshold=args.failure_threshold,
        worst_count=args.worst_count,
    )
    logger.info(
        "wrote %d reports; overlays requested=%d written=%d",
        len(reports),
        overlay_counts["requested"],
        overlay_counts["written"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
