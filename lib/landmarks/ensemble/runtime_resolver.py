#!/usr/bin/env python3
"""Production runtime resolver for landmark ensemble candidates.

This module promotes the v7 bucket-aware resolver path from the offline
evaluation harness into runtime code. It builds single-model and fusion
candidates, derives pose/shape diagnostics from prediction geometry only, maps
the face into a roll/yaw bucket, applies bucket priority plus vetoes, and
returns both the selected landmarks and serializable debug metadata.
"""

from __future__ import annotations

import logging
import math
import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.core.fusion import normalize_weight_matrix, plain_average, static_weighted
from lib.landmarks.core.rejection import weighted_median
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.evaluation.geometry_signals import AlignmentSummary, alignment_summary
from lib.landmarks.evaluation.hard_slices import HardSliceThresholds, hard_slice_label

logger = logging.getLogger(__name__)

DEFAULT_MIN_CLOUD_AREA_RATIO: float = 0.08
DEFAULT_MAX_CLOUD_AREA_RATIO: float = 3.00
DEFAULT_MIN_HULL_AREA_RATIO: float = 0.035
DEFAULT_MAX_HULL_AREA_RATIO: float = 2.25
DEFAULT_MAX_POINTS_OUTSIDE_EXPANDED_BBOX_FRACTION: float = 0.35
DEFAULT_MAX_ROI_CENTER_CONSENSUS_DISTANCE: float = 0.22
DEFAULT_MAX_LANDMARK_CONSENSUS_DISTANCE: float = 0.16

DEFAULT_PRIORITY: tuple[str, ...] = (
    "static_weighted_downweight",
    "static_weighted",
    "static_weighted_hard_drop",
    "weighted_median",
    "spiga",
    "hrnet",
    "orformer",
)

BUCKET_PRIORITIES: dict[str, tuple[str, ...]] = {
    "large_roll": (
        "static_weighted_downweight",
        "static_weighted",
        "weighted_median",
        "spiga",
        "orformer",
        "hrnet",
    ),
    "extreme_roll": (
        "hrnet",
        "spiga",
        "orformer",
        "static_weighted_downweight",
        "static_weighted",
    ),
    "large_yaw_left": (
        "spiga",
        "static_weighted_downweight",
        "static_weighted",
        "hrnet",
        "orformer",
    ),
    "large_yaw_right": (
        "spiga",
        "static_weighted_downweight",
        "static_weighted",
        "hrnet",
        "orformer",
    ),
    "profile_left": (
        "static_weighted_downweight",
        "static_weighted",
        "spiga",
        "hrnet",
        "orformer",
    ),
    "profile_right": (
        "static_weighted_downweight",
        "static_weighted",
        "hrnet",
        "spiga",
        "orformer",
    ),
    "rolled_large_yaw_left": ("spiga", "hrnet", "static_weighted_downweight", "orformer"),
    "rolled_large_yaw_right": ("spiga", "hrnet", "orformer", "static_weighted_downweight"),
    "rolled_profile_left": ("hrnet", "spiga", "static_weighted_downweight", "orformer"),
    "rolled_profile_right": (
        "spiga",
        "hrnet",
        "static_weighted_downweight",
        "orformer",
        "static_weighted_hard_drop",
    ),
}


@dataclass(frozen=True)
class ModelPrediction:
    """One single-model prediction entering the runtime resolver."""

    model: str
    landmarks: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True)
class CandidateRecord:
    """A single or fused runtime candidate."""

    name: str
    landmarks: np.ndarray
    is_fusion: bool
    contributing_models: tuple[str, ...]


@dataclass
class CandidateMetrics:
    """Prediction-only geometry diagnostics for a candidate."""

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


@dataclass(frozen=True)
class RuntimeResolverConfig:
    """Configuration for the production runtime resolver."""

    policy: str = "roll_aware_veto"
    general_strategy: str = "static_weighted"
    hard_case_strategy: str = "static_weighted_downweight"
    secondary_hard_case_strategy: str = "static_weighted_hard_drop"
    fallback_strategy: str = "plain_average"
    fallback_model: str = "orformer"
    outlier_threshold: float = 3.5
    weights: T.Mapping[str, T.Sequence[float]] | None = None
    adapter_weights: T.Mapping[str, float] = field(default_factory=dict)
    hard_disagreement_px: float = 12.0
    roll_veto_degrees: float = 15.0
    hard_roll_degrees: float = 30.0
    strict: bool = False


@dataclass(frozen=True)
class RuntimeResolverResult:
    """Selected candidate and debug payload."""

    selected_candidate: str
    landmarks: np.ndarray
    metadata: dict[str, T.Any]


class RuntimeResolverError(RuntimeError):
    """Raised when no runtime candidate can be selected."""


def _safe_alignment_summary(landmarks: np.ndarray) -> AlignmentSummary | None:
    try:
        return alignment_summary(landmarks.astype("float32", copy=False))
    except Exception as err:  # noqa: BLE001
        logger.debug("alignment_summary failed for runtime resolver candidate: %s", err)
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
    return (
        left - margin_ratio * width,
        top - margin_ratio * height,
        right + margin_ratio * width,
        bottom + margin_ratio * height,
    )


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
    return bool(derolled_mouth[1] > derolled_eye[1])


def _signed_degree_delta(a: float, b: float) -> float:
    return float(((a - b) + 180.0) % 360.0 - 180.0)


def _circular_median(values: T.Sequence[float]) -> float | None:
    if not values:
        return None
    radians = np.deg2rad(np.asarray(list(values), dtype="float64"))
    mean_cos = float(np.mean(np.cos(radians)))
    mean_sin = float(np.mean(np.sin(radians)))
    centre = np.arctan2(mean_sin, mean_cos)
    wrapped = np.mod(radians - centre + np.pi, 2 * np.pi) - np.pi
    return float(np.rad2deg(centre + np.median(wrapped)))


def _reference_bbox(
    candidates: T.Sequence[CandidateRecord],
    detector_bbox: T.Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    if detector_bbox is not None:
        left, top, right, bottom = (float(value) for value in detector_bbox)
        if right > left and bottom > top:
            return (left, top, right, bottom)
    if not candidates:
        return None
    stack = np.stack([candidate.landmarks.astype("float64") for candidate in candidates], axis=0)
    return _landmark_bbox(np.median(stack, axis=0))


def _metric_for_candidate(
    candidate: CandidateRecord,
    *,
    reference_bbox: tuple[float, float, float, float] | None,
) -> CandidateMetrics:
    summary = _safe_alignment_summary(candidate.landmarks)
    reference_area = _bbox_area(reference_bbox)
    candidate_bbox = _landmark_bbox(candidate.landmarks)
    candidate_area = _bbox_area(candidate_bbox)
    hull_area = _convex_hull_area(candidate.landmarks)
    return CandidateMetrics(
        roll_degrees=None if summary is None else float(summary.roll),
        yaw_degrees=None if summary is None else float(summary.yaw),
        pitch_degrees=None if summary is None else float(summary.pitch),
        cloud_area_ratio=(
            None
            if reference_area is None or candidate_area is None
            else candidate_area / reference_area
        ),
        hull_area_ratio=None
        if reference_area is None or hull_area is None
        else hull_area / reference_area,
        points_outside_expanded_bbox_fraction=_points_outside_bbox_fraction(
            candidate.landmarks, _expanded_bbox(reference_bbox)
        ),
        eye_mouth_order_valid_after_deroll=_eye_mouth_order_valid_after_deroll(
            candidate.landmarks
        ),
    )


def _populate_consensus_geometry(
    candidates: T.Sequence[CandidateRecord],
    metrics: T.MutableMapping[str, CandidateMetrics],
    *,
    reference_bbox: tuple[float, float, float, float] | None,
) -> None:
    diag = _bbox_diag(reference_bbox)
    if diag is None:
        return
    stack = np.stack([candidate.landmarks.astype("float64") for candidate in candidates], axis=0)
    consensus_points = np.median(stack, axis=0)
    consensus_center = _bbox_center(_landmark_bbox(consensus_points))
    for candidate in candidates:
        metric = metrics[candidate.name]
        candidate_center = _bbox_center(_landmark_bbox(candidate.landmarks))
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


def _shape_reasons(bucket: str, name: str, metric: CandidateMetrics) -> tuple[str, ...]:
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
    # Keep eye/mouth ordering and consensus-distance diagnostics in metadata only.
    # Offline v5 showed these are unsafe as hard vetoes: when a bad majority forms
    # the consensus, the good single-model candidate becomes the apparent outlier.
    if (
        bucket == "rolled_large_yaw_left"
        and name == "spiga"
        and metric.cloud_area_ratio is not None
        and metric.cloud_area_ratio < 0.55
    ):
        reasons.append("rolled_left_spiga_cloud_area_low")
    if (
        bucket == "rolled_large_yaw_right"
        and name == "spiga"
        and metric.cloud_area_ratio is not None
        and metric.cloud_area_ratio < 0.60
    ):
        reasons.append("rolled_right_spiga_cloud_area_low")
    return tuple(reasons)


def _strategy_candidates(config: RuntimeResolverConfig) -> tuple[str, ...]:
    requested = [
        config.hard_case_strategy,
        config.general_strategy,
        config.secondary_hard_case_strategy,
        config.fallback_strategy,
        *DEFAULT_PRIORITY,
    ]
    retval: list[str] = []
    for name in requested:
        try:
            canonical = canonical_strategy(name)
        except (KeyError, ValueError):
            continue
        if canonical not in retval:
            retval.append(canonical)
    return tuple(retval)


def _fuse_strategy(
    strategy: str,
    singles: T.Sequence[CandidateRecord],
    config: RuntimeResolverConfig,
) -> np.ndarray:
    canonical = canonical_strategy(strategy)
    items = [
        LandmarkPrediction(candidate.landmarks.astype("float32"), source=candidate.name)
        for candidate in singles
    ]
    method = strategy_outlier_method(canonical)
    threshold = config.outlier_threshold if strategy_uses_threshold(canonical) else 3.5
    if not strategy_requires_weights(canonical):
        return plain_average(items, outlier_method=method, outlier_threshold=threshold).points
    models = tuple(candidate.name for candidate in singles)
    if config.weights is None:
        matrix = np.array(
            [[float(config.adapter_weights.get(model, 1.0))] * 68 for model in models],
            dtype="float32",
        )
    else:
        matrix = np.array(
            [config.weights.get(model, [1.0] * 68) for model in models],
            dtype="float32",
        )
    if canonical == "weighted_median":
        stack = np.stack([item.canonical_68().points for item in items], axis=0)
        normalized = normalize_weight_matrix(
            matrix,
            model_count=stack.shape[0],
            landmark_count=stack.shape[1],
        )
        return weighted_median(stack, normalized).astype("float32", copy=False)
    return static_weighted(
        items,
        matrix,
        outlier_method=method,
        outlier_threshold=threshold,
    ).points


def build_candidates(
    predictions: T.Sequence[ModelPrediction],
    config: RuntimeResolverConfig,
) -> list[CandidateRecord]:
    """Build single-model and fusion candidates for the runtime resolver."""
    singles = [
        CandidateRecord(
            name=prediction.model,
            landmarks=np.asarray(prediction.landmarks, dtype="float32"),
            is_fusion=False,
            contributing_models=(prediction.model,),
        )
        for prediction in predictions
    ]
    candidates = list(singles)
    if len(singles) < 2:
        return candidates
    for strategy in _strategy_candidates(config):
        try:
            landmarks = _fuse_strategy(strategy, singles, config)
        except Exception as err:  # noqa: BLE001
            if config.strict:
                raise RuntimeResolverError(f"fusion candidate {strategy!r} failed: {err}") from err
            logger.debug("Skipping runtime fusion candidate %s: %s", strategy, err)
            continue
        candidates.append(
            CandidateRecord(
                name=strategy,
                landmarks=landmarks.astype("float32", copy=False),
                is_fusion=True,
                contributing_models=tuple(candidate.name for candidate in singles),
            )
        )
    return candidates


def _priority_for_bucket(bucket: str, available: T.AbstractSet[str]) -> list[str]:
    priority = list(BUCKET_PRIORITIES.get(bucket, DEFAULT_PRIORITY))
    priority.extend(name for name in DEFAULT_PRIORITY if name not in priority)
    priority.extend(sorted(name for name in available if name not in priority))
    return [name for name in priority if name in available]


def _available_by_priority(priority: T.Sequence[str], available: T.AbstractSet[str]) -> str:
    for name in priority:
        if name in available:
            return name
    if not available:
        raise RuntimeResolverError("runtime resolver received no selectable candidates")
    return sorted(available)[0]


def _roll_vetoes(
    candidates: T.Sequence[CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    *,
    threshold_deg: float,
    consensus_roll: float | None,
) -> set[str]:
    if consensus_roll is None:
        return set()
    fusion_names = {candidate.name for candidate in candidates if candidate.is_fusion}
    vetoed: set[str] = set()
    for name in fusion_names:
        roll = metrics[name].roll_degrees
        if roll is None or abs(_signed_degree_delta(roll, consensus_roll)) > threshold_deg:
            vetoed.add(name)
    return vetoed


def _metrics_payload(
    metrics: T.Mapping[str, CandidateMetrics],
    attr: str,
) -> dict[str, float | bool | None]:
    return {name: getattr(metric, attr) for name, metric in metrics.items()}


def _max_landmark_consensus_px(candidates: T.Sequence[CandidateRecord]) -> float:
    if not candidates:
        return 0.0
    stack = np.stack([candidate.landmarks.astype("float64") for candidate in candidates], axis=0)
    consensus = np.median(stack, axis=0)
    per_candidate = np.mean(np.linalg.norm(stack - consensus[None], axis=2), axis=1)
    return float(per_candidate.max()) if per_candidate.size else 0.0


def resolve_runtime(
    predictions: T.Sequence[ModelPrediction],
    config: RuntimeResolverConfig,
    *,
    detector_bbox: T.Sequence[float] | None = None,
) -> RuntimeResolverResult:
    """Resolve one face using v7 runtime candidate priority and vetoes."""
    if config.policy != "roll_aware_veto":
        raise RuntimeResolverError(f"unsupported runtime resolver policy {config.policy!r}")
    candidates = build_candidates(predictions, config)
    if not candidates:
        raise RuntimeResolverError("no runtime resolver candidates were provided")

    reference_bbox = _reference_bbox(candidates, detector_bbox)
    metrics = {
        candidate.name: _metric_for_candidate(candidate, reference_bbox=reference_bbox)
        for candidate in candidates
    }
    _populate_consensus_geometry(candidates, metrics, reference_bbox=reference_bbox)

    roll_estimate = _circular_median(
        [metric.roll_degrees for metric in metrics.values() if metric.roll_degrees is not None]
    )
    yaw_estimate = _circular_median(
        [metric.yaw_degrees for metric in metrics.values() if metric.yaw_degrees is not None]
    )
    bucket = hard_slice_label(
        yaw_estimate,
        roll_deg=roll_estimate,
        thresholds=HardSliceThresholds(roll_degrees=config.hard_roll_degrees),
    )

    for name, metric in metrics.items():
        metric.geometry_veto_reasons = _shape_reasons(bucket, name, metric)

    available = {candidate.name for candidate in candidates}
    priority = _priority_for_bucket(bucket, available)
    roll_vetoed = _roll_vetoes(
        candidates,
        metrics,
        threshold_deg=config.roll_veto_degrees,
        consensus_roll=roll_estimate,
    )
    geometry_vetoed = {name for name, metric in metrics.items() if metric.geometry_veto_reasons}
    vetoed = roll_vetoed | geometry_vetoed
    survivors = available - vetoed
    max_disagreement_px = _max_landmark_consensus_px(candidates)
    hard_bucket = bucket not in {"frontal", "intermediate", "no_pose"}
    risk_route = (
        "high_risk"
        if hard_bucket or max_disagreement_px > config.hard_disagreement_px or vetoed
        else "low_risk"
    )
    selected = (
        _available_by_priority(priority, survivors)
        if survivors
        else _available_by_priority(priority, available)
    )
    if not survivors:
        fallback_reason: str | None = "all_candidates_vetoed"
    else:
        fallback_reason = None

    by_name = {candidate.name: candidate for candidate in candidates}
    metadata: dict[str, T.Any] = {
        "selected_candidate": selected,
        "bucket": bucket,
        "candidate_priority": priority,
        "vetoed": sorted(vetoed & available),
        "veto_reasons": {
            name: list(metric.geometry_veto_reasons)
            for name, metric in metrics.items()
            if metric.geometry_veto_reasons
        },
        "roll_estimate": roll_estimate,
        "yaw_estimate": yaw_estimate,
        "cloud_area_ratio": _metrics_payload(metrics, "cloud_area_ratio"),
        "hull_area_ratio": _metrics_payload(metrics, "hull_area_ratio"),
        "points_outside_expanded_bbox_fraction": _metrics_payload(
            metrics, "points_outside_expanded_bbox_fraction"
        ),
        "eye_mouth_order_valid_after_deroll": _metrics_payload(
            metrics, "eye_mouth_order_valid_after_deroll"
        ),
        "landmark_consensus_distance": _metrics_payload(metrics, "landmark_consensus_distance"),
        "roi_center_consensus_distance": _metrics_payload(
            metrics, "roi_center_consensus_distance"
        ),
        "model_predictions_available": {prediction.model: True for prediction in predictions},
        "roll_vetoed": sorted(roll_vetoed & available),
        "geometry_vetoed": sorted(geometry_vetoed & available),
        "rolls": _metrics_payload(metrics, "roll_degrees"),
        "yaws": _metrics_payload(metrics, "yaw_degrees"),
        "max_disagreement_px": max_disagreement_px,
        "risk_route": risk_route,
        "fallback_reason": fallback_reason,
        "policy": config.policy,
    }
    return RuntimeResolverResult(
        selected_candidate=selected,
        landmarks=by_name[selected].landmarks.astype("float32", copy=False),
        metadata=metadata,
    )


__all__ = [
    "BUCKET_PRIORITIES",
    "CandidateMetrics",
    "CandidateRecord",
    "DEFAULT_PRIORITY",
    "ModelPrediction",
    "RuntimeResolverConfig",
    "RuntimeResolverError",
    "RuntimeResolverResult",
    "build_candidates",
    "resolve_runtime",
]