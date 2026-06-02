#!/usr/bin/env python3
"""Production runtime resolver for landmark ensemble candidates.

This module promotes the v7 bucket-aware resolver path from the offline
evaluation harness into runtime code. It builds single-model and fusion
candidates, derives pose/shape diagnostics from prediction geometry plus image
crop evidence, maps the face into a runtime bucket, applies bucket priority plus
vetoes, and returns both the selected landmarks and serializable debug metadata.
"""

from __future__ import annotations

import logging
import math
import typing as T
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.core.fusion import normalize_weight_matrix, plain_average, static_weighted
from lib.landmarks.core.rejection import weighted_median
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.production_artifacts import LEARNED_POLICIES
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    candidate_scores as score_runtime_candidates,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    load_runtime_resolver_scorer,
)
from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.evaluation.geometry_signals import AlignmentSummary, alignment_summary
from lib.landmarks.evaluation.shape_plausibility import evaluate_shape_plausibility

logger = logging.getLogger(__name__)

DEFAULT_MIN_CLOUD_AREA_RATIO: float = 0.08
DEFAULT_MAX_CLOUD_AREA_RATIO: float = 3.00
DEFAULT_MIN_HULL_AREA_RATIO: float = 0.035
DEFAULT_MAX_HULL_AREA_RATIO: float = 2.25
DEFAULT_MAX_POINTS_OUTSIDE_EXPANDED_BBOX_FRACTION: float = 0.35
DEFAULT_MAX_SHAPE_PLAUSIBILITY_SCORE: float = 1.10
HARD_POSE_MAX_SHAPE_PLAUSIBILITY_SCORE: float = 1.35
DEFAULT_MAX_ROI_CENTER_CONSENSUS_DISTANCE: float = 0.22
DEFAULT_MAX_LANDMARK_CONSENSUS_DISTANCE: float = 0.16
IMAGE_YAW_SIDE_THRESHOLD: float = 0.08
LANDMARK_YAW_SIDE_THRESHOLD: float = 3.0
NOSE_SIDE_THRESHOLD: float = 0.08
JAW_SIDE_THRESHOLD: float = 0.20
LARGE_YAW_IMAGE_SIGNAL_THRESHOLD: float = 0.28
LARGE_YAW_LANDMARK_THRESHOLD: float = 35.0
PROFILE_IMAGE_SIGNAL_THRESHOLD: float = 0.50
PROFILE_LANDMARK_YAW_THRESHOLD: float = 55.0
PROFILE_CANDIDATE_YAW_DISAGREEMENT_THRESHOLD: float = 120.0
PROFILE_MAX_DISAGREEMENT_BBOX_FRACTION_THRESHOLD: float = 0.28
PROFILE_JAW_STRUCTURE_THRESHOLD: float = 0.45
PROFILE_NOSE_STRUCTURE_THRESHOLD: float = 0.35
PROFILE_CANDIDATE_SIDE_STRUCTURE_THRESHOLD: float = 0.30
PROFILE_NOSE_OFFSET_THRESHOLD: float = 0.12
PROFILE_MULTI_MODEL_YAW_THRESHOLD: float = 60.0
PROFILE_CANDIDATE_TRUSTED_YAW_THRESHOLD: float = 40.0
PROFILE_VISUAL_SUPPORT_IMAGE_THRESHOLD: float = 0.12
PROFILE_VISUAL_SUPPORT_STRUCTURE_THRESHOLD: float = 0.18
WEAK_YAW_CAP_IMAGE_THRESHOLD: float = 0.12
WEAK_YAW_CAP_STRUCTURE_THRESHOLD: float = 0.18
SIDE_YAW_TRUSTED_MODELS: tuple[str, ...] = ("hrnet", "spiga", "orformer")
SIDE_YAW_PRIMARY_MODEL: str = "hrnet"
SIDE_YAW_MIN_ABS_DEGREES: float = 20.0
SIDE_YAW_FULL_CONFIDENCE_DEGREES: float = 75.0
ROLL_BUCKET_AGREEMENT_DEGREES: float = 15.0
ROLL_BUCKET_MIN_SUPPORTING_CANDIDATES: int = 2
HARD_SLICE_SAFE_SINGLE_BUCKETS: frozenset[str] = frozenset(
    (
        "extreme_roll",
        "rolled_large_yaw_left",
        "rolled_large_yaw_right",
        "rolled_profile_left",
        "rolled_profile_right",
    )
)
HARD_POSE_PLAIN_AVERAGE_GUARD_BUCKET_PREFIXES: tuple[str, ...] = (
    "profile_",
    "large_yaw_",
    "rolled_large_yaw_",
)
HARD_POSE_PLAIN_AVERAGE_SAFE_MODELS: tuple[str, ...] = ("hrnet", "spiga")
HARD_SLICE_CONSENSUS_DISTANCE_THRESHOLD: float = 0.04
HARD_SLICE_SINGLE_MODEL_DISAGREEMENT_PX_THRESHOLD: float = 35.0
HARD_SLICE_DISTANCE_TO_HRNET_THRESHOLD: float = 0.08
SAFE_SINGLE_TIE_BREAKER: tuple[str, ...] = ("hrnet", "spiga", "orformer")
DEFAULT_SAFE_FALLBACK_MIN_DELTA: float = 0.05

DEFAULT_PRIORITY: tuple[str, ...] = (
    "weighted_median",
    "spiga",
    "static_weighted_downweight",
    "static_weighted_hard_drop",
    "static_weighted",
    "orformer",
    "hrnet",
    "fan",
)

RUNTIME_BUCKETS: frozenset[str] = frozenset(
    (
        "frontal",
        "intermediate",
        "large_yaw_left",
        "large_yaw_right",
        "profile_left",
        "profile_right",
        "large_roll",
        "extreme_roll",
        "rolled_large_yaw_left",
        "rolled_large_yaw_right",
        "rolled_profile_left",
        "rolled_profile_right",
    )
)

BUCKET_PRIORITIES: dict[str, tuple[str, ...]] = {
    # No GT runtime metrics in uploaded file, left unchanged.
    "large_roll": (
        "weighted_median",
        "static_weighted_downweight",
        "spiga",
        "static_weighted_hard_drop",
        "static_weighted",
        "plain_average",
        "orformer",
        "hrnet",
        "fan",
    ),
    "extreme_roll": (
        "spiga",
        "static_weighted",
        "plain_average",
        "static_weighted_downweight",
        "weighted_median",
        "static_weighted_hard_drop",
        "hrnet",
        "fan",
        "orformer",
    ),
    "large_yaw_left": (
        "weighted_median",
        "static_weighted_downweight",
        "static_weighted_hard_drop",
        "spiga",
        "static_weighted",
        "plain_average",
        "orformer",
        "hrnet",
        "fan",
    ),
    "large_yaw_right": (
        "weighted_median",
        "static_weighted_downweight",
        "static_weighted_hard_drop",
        "orformer",
        "static_weighted",
        "plain_average",
        "spiga",
        "hrnet",
        "fan",
    ),
    "profile_left": (
        "static_weighted_downweight",
        "static_weighted",
        "plain_average",
        "static_weighted_hard_drop",
        "weighted_median",
        "spiga",
        "orformer",
        "hrnet",
        "fan",
    ),
    "profile_right": (
        "static_weighted_downweight",
        "static_weighted_hard_drop",
        "weighted_median",
        "static_weighted",
        "orformer",
        "spiga",
        "plain_average",
        "hrnet",
        "fan",
    ),
    "rolled_large_yaw_left": (
        "spiga",
        "static_weighted_downweight",
        "weighted_median",
        "static_weighted_hard_drop",
        "static_weighted",
        "plain_average",
        "orformer",
        "hrnet",
        "fan",
    ),
    "rolled_large_yaw_right": (
        "spiga",
        "weighted_median",
        "static_weighted_downweight",
        "static_weighted_hard_drop",
        "static_weighted",
        "plain_average",
        "orformer",
        "hrnet",
        "fan",
    ),
    "rolled_profile_left": (
        "spiga",
        "static_weighted",
        "plain_average",
        "static_weighted_downweight",
        "weighted_median",
        "static_weighted_hard_drop",
        "hrnet",
        "fan",
        "orformer",
    ),
    "rolled_profile_right": (
        "spiga",
        "weighted_median",
        "static_weighted_downweight",
        "static_weighted_hard_drop",
        "static_weighted",
        "plain_average",
        "hrnet",
        "fan",
        "orformer",
    ),
}


def _trace(message: str, *args: T.Any) -> None:
    """Log at Faceswap TRACE level when available."""
    trace = getattr(logger, "trace", None)
    if callable(trace):
        trace(message, *args)


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
    shape_plausibility_score: float | None = None
    shape_veto_reasons: tuple[str, ...] = ()
    max_edge_length_ratio: float | None = None
    mean_shape_fit_error: float | None = None
    topology_violation_count: int = 0


@dataclass(frozen=True)
class RuntimeResolverConfig:
    """Configuration for the production runtime resolver."""

    policy: str = "roll_aware_veto"
    scorer_path: str = ""
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
    risk_floor_for_safe_fallback: float = 0.50
    safe_fallback_min_delta: float = DEFAULT_SAFE_FALLBACK_MIN_DELTA
    strict: bool = False


@dataclass(frozen=True)
class RuntimeResolverResult:
    """Selected candidate and debug payload."""

    selected_candidate: str
    landmarks: np.ndarray
    metadata: dict[str, T.Any]


class RuntimeResolverError(RuntimeError):
    """Raised when no runtime candidate can be selected."""


@dataclass(frozen=True)
class RuntimeBucketResult:
    """Image-aware runtime bucket plus diagnostic features."""

    bucket: str
    features: dict[str, T.Any]


def _assert_frame_space_candidates(
    candidates: T.Sequence[CandidateRecord],
    detector_bbox: T.Sequence[float] | None,
) -> None:
    """Reject normalized landmarks before frame-space geometry validation."""
    if detector_bbox is None:
        return
    left, top, right, bottom = (float(value) for value in detector_bbox)
    bbox_w = right - left
    bbox_h = bottom - top
    if bbox_w <= 0 or bbox_h <= 0 or max(bbox_w, bbox_h) <= 100.0:
        return

    leaked: list[str] = []
    for candidate in candidates:
        landmarks = np.asarray(candidate.landmarks, dtype="float64")
        if (
            landmarks.ndim != 2
            or landmarks.shape[1] < 2
            or landmarks.size == 0
            or not np.all(np.isfinite(landmarks[:, :2]))
        ):
            continue
        extent_x = float(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]))
        extent_y = float(np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]))
        if extent_x < 2.0 and extent_y < 2.0:
            leaked.append(f"{candidate.name}(extent_x={extent_x:.6g}, extent_y={extent_y:.6g})")
    if leaked:
        raise RuntimeResolverError(
            "Runtime resolver received non-frame-space landmarks for "
            f"detector_bbox={(left, top, right, bottom)}: {', '.join(leaked)}"
        )


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
    plausibility = evaluate_shape_plausibility(candidate.landmarks)
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
        shape_plausibility_score=plausibility.score,
        shape_veto_reasons=plausibility.reasons if plausibility.severe else (),
        max_edge_length_ratio=plausibility.metrics.get("max_edge_length_ratio"),
        mean_shape_fit_error=plausibility.metrics.get("mean_shape_fit_error"),
        topology_violation_count=int(plausibility.metrics.get("topology_violation_count", 0.0)),
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


def _mean_landmark_distance_px(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(left.astype("float64") - right.astype("float64"), axis=1)))


def _candidate_extra_features(
    candidates: T.Sequence[CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    *,
    reference_bbox: tuple[float, float, float, float] | None,
    condition: str = "",
    runtime_bucket: str = "",
    runtime_bucket_source: str = "",
) -> dict[str, dict[str, float]]:
    """Return full-candidate-set features used by learned quality scoring."""
    diag = _bbox_diag(reference_bbox)
    by_name = {candidate.name: candidate for candidate in candidates}
    single_candidates = [
        candidate for candidate in candidates if not _is_canonical_strategy_name(candidate.name)
    ]
    single_stack = (
        np.stack(
            [candidate.landmarks.astype("float64") for candidate in single_candidates], axis=0
        )
        if single_candidates
        else None
    )
    single_cluster = np.median(single_stack, axis=0) if single_stack is not None else None
    hrnet = by_name.get("hrnet")
    single_disagreement_px = 0.0
    for left_index, left in enumerate(single_candidates):
        for right in single_candidates[left_index + 1 :]:
            single_disagreement_px = max(
                single_disagreement_px,
                _mean_landmark_distance_px(left.landmarks, right.landmarks),
            )
    geometry_valid = {
        name: float(name in metrics and not metrics[name].geometry_veto_reasons)
        for name in ("hrnet", "spiga", "orformer")
    }
    condition_or_bucket = condition or runtime_bucket
    condition_is_gt_roll_hard = (
        condition_or_bucket in HARD_SLICE_SAFE_SINGLE_BUCKETS
        and runtime_bucket_source == "derived_no_image_evidence"
    )
    source_is_derived_no_image = runtime_bucket_source == "derived_no_image_evidence"

    payload: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        metric = metrics.get(candidate.name)
        consensus_distance = (
            float(metric.landmark_consensus_distance or 0.0) if metric is not None else 0.0
        )
        distance_to_hrnet_px = (
            _mean_landmark_distance_px(candidate.landmarks, hrnet.landmarks)
            if hrnet is not None
            else 0.0
        )
        distance_to_cluster_px = (
            _mean_landmark_distance_px(candidate.landmarks, single_cluster)
            if single_cluster is not None
            else 0.0
        )
        fusion_vs_single_px = 0.0
        if candidate.is_fusion and single_candidates:
            fusion_vs_single_px = min(
                _mean_landmark_distance_px(candidate.landmarks, single.landmarks)
                for single in single_candidates
            )
        payload[candidate.name] = {
            "candidate_is_consensus_like": float(
                consensus_distance <= HARD_SLICE_CONSENSUS_DISTANCE_THRESHOLD
            ),
            "candidate_distance_to_hrnet": (
                distance_to_hrnet_px / diag if diag is not None else 0.0
            ),
            "candidate_distance_to_best_single_cluster": (
                distance_to_cluster_px / diag if diag is not None else 0.0
            ),
            "single_model_disagreement_px": single_disagreement_px,
            "fusion_vs_single_disagreement_px": fusion_vs_single_px,
            "hrnet_geometry_valid": geometry_valid["hrnet"],
            "spiga_geometry_valid": geometry_valid["spiga"],
            "orformer_geometry_valid": geometry_valid["orformer"],
            "condition_is_gt_roll_hard": float(condition_is_gt_roll_hard),
            "runtime_bucket_source_is_derived_no_image_evidence": float(
                source_is_derived_no_image
            ),
        }
    return payload


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
    shape_threshold = (
        HARD_POSE_MAX_SHAPE_PLAUSIBILITY_SCORE
        if bucket not in {"frontal", "intermediate", "no_pose"}
        else DEFAULT_MAX_SHAPE_PLAUSIBILITY_SCORE
    )
    if (
        metric.shape_plausibility_score is not None
        and metric.shape_plausibility_score > shape_threshold
    ):
        reasons.append("face_shape_plausibility_low")
    for reason in metric.shape_veto_reasons:
        reasons.append(f"shape_{reason}")
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
        return T.cast(
            np.ndarray,
            plain_average(items, outlier_method=method, outlier_threshold=threshold).points,
        )
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
        return T.cast(np.ndarray, weighted_median(stack, normalized).astype("float32", copy=False))
    return T.cast(
        np.ndarray,
        static_weighted(
            items,
            matrix,
            outlier_method=method,
            outlier_threshold=threshold,
        ).points,
    )


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


def _finite_candidate_names(candidates: T.Sequence[CandidateRecord]) -> set[str]:
    return {
        candidate.name
        for candidate in candidates
        if np.all(np.isfinite(np.asarray(candidate.landmarks, dtype="float64")))
    }


def _all_candidates_vetoed_fallback(
    candidates: T.Sequence[CandidateRecord],
    config: RuntimeResolverConfig,
    *,
    fatal_shape_vetoed: T.AbstractSet[str] | None = None,
) -> str:
    """Return deterministic safe fallback when every candidate was vetoed.

    Fatal shape failures are excluded on the first pass. If every finite candidate
    is a fatal shape failure, keep the historical finite-candidate ordering as the
    last-resort emergency behavior.
    """
    finite_available = _finite_candidate_names(candidates)
    fatal_names = set(fatal_shape_vetoed or ())
    fallback_order = (
        config.hard_case_strategy,
        config.secondary_hard_case_strategy,
        "static_weighted_downweight",
        "static_weighted_hard_drop",
        "plain_average",
        "static_weighted",
        "weighted_median",
        "hrnet",
        "spiga",
        "orformer",
    )
    for pass_name, available in (
        ("nonfatal_shape", finite_available - fatal_names),
        ("finite", finite_available),
    ):
        for name in dict.fromkeys(fallback_order):
            if name not in available:
                continue
            logger.debug(
                "[RuntimeResolver] all candidates vetoed; emergency fallback selected %s pass=%s",
                name,
                pass_name,
            )
            _trace(
                "[RuntimeResolver] all-veto fallback order=%s finite_candidates=%s "
                "fatal_shape_vetoed=%s pass=%s",
                fallback_order,
                sorted(finite_available),
                sorted(fatal_names),
                pass_name,
            )
            return name
    raise RuntimeResolverError("all runtime candidates were vetoed and none had finite landmarks")


def _candidate_veto_reasons(
    candidates: T.Sequence[CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    *,
    roll_vetoed: T.AbstractSet[str],
    vetoed: T.AbstractSet[str],
) -> dict[str, list[str]]:
    """Return per-candidate veto reasons for runtime resolver metadata."""
    payload: dict[str, list[str]] = {}
    by_name = {candidate.name: candidate for candidate in candidates}
    for name in sorted(vetoed & set(by_name)):
        reasons = list(metrics.get(name, CandidateMetrics(None, None, None)).geometry_veto_reasons)
        if name in roll_vetoed:
            reasons.append("roll_veto")
        if not np.all(np.isfinite(np.asarray(by_name[name].landmarks, dtype="float64"))):
            reasons.append("nonfinite_landmarks")
        payload[name] = reasons or ["vetoed_without_specific_reason"]
    return payload


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
) -> dict[str, T.Any]:
    return {name: getattr(metric, attr) for name, metric in metrics.items()}


def _max_landmark_consensus_px(candidates: T.Sequence[CandidateRecord]) -> float:
    if not candidates:
        return 0.0
    stack = np.stack([candidate.landmarks.astype("float64") for candidate in candidates], axis=0)
    consensus = np.median(stack, axis=0)
    per_candidate = np.mean(np.linalg.norm(stack - consensus[None], axis=2), axis=1)
    return float(per_candidate.max()) if per_candidate.size else 0.0


def _frame_to_crop_pixels(
    points: np.ndarray,
    crop_to_frame_matrix: np.ndarray,
    image_crop: np.ndarray,
) -> np.ndarray:
    """Transform frame-space landmarks to image-crop pixel coordinates."""
    matrix = np.asarray(crop_to_frame_matrix, dtype="float64")
    if matrix.shape != (3, 3):
        raise ValueError(f"crop_to_frame_matrix must have shape (3, 3), got {matrix.shape}")
    height, width = image_crop.shape[:2]
    pts = np.asarray(points, dtype="float64")
    ones = np.ones((pts.shape[0], 1), dtype="float64")
    normalized = np.concatenate([pts[:, :2], ones], axis=1) @ np.linalg.inv(matrix).T
    pixels = normalized[:, :2].copy()
    pixels[:, 0] *= float(width)
    pixels[:, 1] *= float(height)
    return pixels.astype("float32", copy=False)  # type: ignore[no-any-return]


def _crop_gray(image_crop: np.ndarray) -> np.ndarray:
    """Return a float grayscale crop in 0..1 range."""
    crop = np.asarray(image_crop)
    if crop.ndim == 2:
        gray = crop.astype("float32", copy=False)
    elif crop.ndim == 3 and crop.shape[2] >= 3:
        arr = crop[..., :3].astype("float32", copy=False)
        gray = (0.299 * arr[..., 0]) + (0.587 * arr[..., 1]) + (0.114 * arr[..., 2])
    else:
        raise ValueError(f"image_crop must have shape (H, W) or (H, W, C), got {crop.shape}")
    if gray.size and float(np.nanmax(gray)) > 2.0:
        gray = gray / 255.0
    return np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)  # type: ignore[no-any-return]


def _eye_visual_score(gray: np.ndarray, eye_points: np.ndarray) -> float:
    """Return a rough visual evidence score for an eye landmark region."""
    points = np.asarray(eye_points, dtype="float32")
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return 0.0
    height, width = gray.shape[:2]
    if width <= 1 or height <= 1:
        return 0.0
    finite = points[np.all(np.isfinite(points[:, :2]), axis=1), :2]
    if finite.size == 0:
        return 0.0
    left, top = np.min(finite, axis=0)
    right, bottom = np.max(finite, axis=0)
    eye_w = max(float(right - left), 4.0)
    eye_h = max(float(bottom - top), 4.0)
    margin_x = max(eye_w * 0.9, 6.0)
    margin_y = max(eye_h * 1.2, 6.0)
    x0 = max(0, int(math.floor(left - margin_x)))
    y0 = max(0, int(math.floor(top - margin_y)))
    x1 = min(width, int(math.ceil(right + margin_x)))
    y1 = min(height, int(math.ceil(bottom + margin_y)))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    if patch.size < 9:
        return 0.0
    contrast = float(np.std(patch))
    grad_y, grad_x = np.gradient(patch.astype("float32", copy=False))
    edge = float(np.mean(np.hypot(grad_x, grad_y)))
    darkness = max(0.0, 0.55 - float(np.percentile(patch, 20)))
    score = (2.4 * contrast) + (3.2 * edge) + (0.7 * darkness)
    return float(max(0.0, min(score, 1.0)))


def _signed_candidate_yaw_disagreement(metrics: T.Mapping[str, CandidateMetrics]) -> float:
    """Return yaw spread across candidates, ignoring missing estimates."""
    yaws = [metric.yaw_degrees for metric in metrics.values() if metric.yaw_degrees is not None]
    if len(yaws) < 2:
        return 0.0
    return float(max(yaws) - min(yaws))


def _dominant_candidate_yaw(metrics: T.Mapping[str, CandidateMetrics]) -> float:
    """Return the candidate yaw with the strongest absolute side evidence."""
    yaws = [
        float(metric.yaw_degrees)
        for metric in metrics.values()
        if metric.yaw_degrees is not None and math.isfinite(float(metric.yaw_degrees))
    ]
    if not yaws:
        return 0.0
    return max(yaws, key=abs)


def _trusted_single_model_yaw(metrics: T.Mapping[str, CandidateMetrics]) -> float:
    """Return the strongest finite yaw from trusted single-model pose estimates."""
    yaws: list[float] = []
    for model in SIDE_YAW_TRUSTED_MODELS:
        metric = metrics.get(model)
        yaw = None if metric is None else metric.yaw_degrees
        if yaw is not None and math.isfinite(float(yaw)):
            yaws.append(float(yaw))
    if not yaws:
        return 0.0
    return max(yaws, key=abs)


def _is_canonical_strategy_name(name: str) -> bool:
    try:
        canonical_strategy(name)
    except (KeyError, ValueError):
        return False
    return True


def _single_model_yaw_side_agreement(
    metrics: T.Mapping[str, CandidateMetrics],
    *,
    min_abs_degrees: float,
) -> tuple[bool, str | None, int]:
    """Return whether at least two non-fusion models agree on yaw side."""
    side_counts: Counter[str] = Counter()
    for name, metric in metrics.items():
        if _is_canonical_strategy_name(name):
            continue
        yaw = metric.yaw_degrees
        if yaw is None or not math.isfinite(float(yaw)):
            continue
        yaw_value = float(yaw)
        if abs(yaw_value) < min_abs_degrees:
            continue
        side_counts[_side_from_model_yaw(yaw_value)] += 1
    if not side_counts:
        return False, None, 0
    side, count = side_counts.most_common(1)[0]
    return count >= 2, side, int(count)


def _side_from_model_yaw(yaw_degrees: float) -> str:
    """Map raw single-model yaw to image-facing side."""
    return "left" if yaw_degrees > 0.0 else "right"


def _side_yaw_confidence(yaw_degrees: float) -> float:
    """Normalize absolute yaw into a 0..1 side confidence score."""
    return float(min(abs(yaw_degrees) / SIDE_YAW_FULL_CONFIDENCE_DEGREES, 1.0))


def _runtime_side_from_single_model_yaws(
    metrics: T.Mapping[str, CandidateMetrics],
) -> tuple[str | None, str, float, dict[str, dict[str, T.Any]]]:
    """Infer side from trusted single-model yaw only.

    Fused candidates and consensus-derived geometry are deliberately excluded:
    their side signs can flip when the 68-point profile topology is completed or
    hallucinated. HRNet is treated as the primary side source when available;
    the other single-model yaws are only a fallback when HRNet is missing or too
    weak.
    """
    votes: dict[str, dict[str, T.Any]] = {}
    for model in SIDE_YAW_TRUSTED_MODELS:
        metric = metrics.get(model)
        yaw = None if metric is None else metric.yaw_degrees
        if yaw is None or not math.isfinite(float(yaw)):
            continue
        yaw_value = float(yaw)
        usable = abs(yaw_value) >= SIDE_YAW_MIN_ABS_DEGREES
        votes[model] = {
            "yaw": yaw_value,
            "side": _side_from_model_yaw(yaw_value),
            "usable": usable,
            "confidence": _side_yaw_confidence(yaw_value) if usable else 0.0,
        }

    primary = votes.get(SIDE_YAW_PRIMARY_MODEL)
    if primary is not None and bool(primary["usable"]):
        return (
            T.cast(str, primary["side"]),
            f"{SIDE_YAW_PRIMARY_MODEL}_yaw",
            float(primary["confidence"]),
            votes,
        )

    usable_votes = [vote for vote in votes.values() if bool(vote["usable"])]
    if usable_votes:
        score = sum(
            float(vote["confidence"]) * (1.0 if vote["side"] == "left" else -1.0)
            for vote in usable_votes
        )
        total = sum(float(vote["confidence"]) for vote in usable_votes)
        side = "left" if score >= 0.0 else "right"
        confidence = 0.0 if total <= 0.0 else abs(score) / total
        return side, "single_model_yaw_vote", float(confidence), votes

    return None, "single_model_yaw_unavailable", 0.0, votes


def _runtime_side_from_signals(
    *,
    image_geometry_yaw_signal: float,
    nose_offset_from_face_center: float,
    mouth_nose_jaw_asymmetry: float,
    landmark_pose_yaw: float | None,
    dominant_candidate_yaw: float,
) -> tuple[str, str]:
    """Return landmark-geometry side for diagnostics and fallback only.

    This signal is not trusted for normal profile/large-yaw side routing because
    it is derived from consensus/completed 68-point geometry rather than a raw
    image-native side source.
    """
    if abs(nose_offset_from_face_center) >= NOSE_SIDE_THRESHOLD:
        return ("left" if nose_offset_from_face_center < 0 else "right"), "nose_offset"
    if abs(mouth_nose_jaw_asymmetry) >= JAW_SIDE_THRESHOLD:
        return ("left" if mouth_nose_jaw_asymmetry > 0 else "right"), "jaw_asymmetry"
    if abs(image_geometry_yaw_signal) >= IMAGE_YAW_SIDE_THRESHOLD:
        return ("left" if image_geometry_yaw_signal < 0 else "right"), "image_geometry"
    if (
        landmark_pose_yaw is not None
        and abs(float(landmark_pose_yaw)) >= LANDMARK_YAW_SIDE_THRESHOLD
    ):
        return ("left" if float(landmark_pose_yaw) < 0 else "right"), "landmark_pose_yaw"
    if abs(dominant_candidate_yaw) >= LANDMARK_YAW_SIDE_THRESHOLD:
        return ("left" if dominant_candidate_yaw < 0 else "right"), "dominant_candidate_yaw"
    return ("left" if image_geometry_yaw_signal < 0 else "right"), "weak_image_geometry"


def _normalized_max_disagreement(
    max_disagreement_px: float,
    detector_bbox: T.Sequence[float] | None,
    consensus_landmarks: np.ndarray,
) -> float:
    """Return max candidate disagreement normalized by detector or landmark size."""
    bbox: tuple[float, float, float, float] | None = None
    if detector_bbox is not None:
        left, top, right, bottom = (float(value) for value in detector_bbox)
        if right > left and bottom > top:
            bbox = (left, top, right, bottom)
    if bbox is None:
        bbox = _landmark_bbox(consensus_landmarks)
    diag = _bbox_diag(bbox)
    if diag is None:
        return 0.0
    return float(max_disagreement_px / diag)


def _runtime_yaw_severity(
    *,
    image_geometry_yaw_signal: float,
    landmark_pose_yaw: float | None,
    trusted_single_model_yaw: float,
    profile_yaw_agreement: bool,
    candidate_profile_yaw_agreement: bool,
    candidate_yaw_disagreement: float,
    max_disagreement_bbox_fraction: float,
    nose_offset_from_face_center: float,
    mouth_nose_jaw_asymmetry: float,
) -> tuple[str, str]:
    """Return yaw severity and evidence source.

    Visual definitions:
    ``intermediate`` is a moderate turn without strong profile evidence.
    ``large_yaw`` is a strongly turned face where both sides still have
    meaningful visual/geometric support. ``profile`` is a near side-on view:
    one side dominates, the far-side landmarks are mostly inferred or unstable,
    or the nose/front is close to a silhouette.
    """
    abs_image_yaw = abs(image_geometry_yaw_signal)
    abs_landmark_yaw = 0.0 if landmark_pose_yaw is None else abs(float(landmark_pose_yaw))
    abs_trusted_yaw = abs(float(trusted_single_model_yaw))
    nose_structure = abs(nose_offset_from_face_center)
    jaw_structure = abs(mouth_nose_jaw_asymmetry)
    side_structure = max(nose_structure, jaw_structure)
    profile_allowed = (
        abs_image_yaw >= PROFILE_IMAGE_SIGNAL_THRESHOLD
        or abs_trusted_yaw >= PROFILE_MULTI_MODEL_YAW_THRESHOLD
    )
    strong_profile_shape = (
        nose_structure >= PROFILE_NOSE_STRUCTURE_THRESHOLD
        and jaw_structure >= PROFILE_JAW_STRUCTURE_THRESHOLD
    )
    if (
        profile_allowed
        and abs_image_yaw >= PROFILE_IMAGE_SIGNAL_THRESHOLD
        and strong_profile_shape
    ):
        return "profile", "image_geometry"
    if (
        profile_allowed
        and profile_yaw_agreement
        and (
            abs_image_yaw >= PROFILE_VISUAL_SUPPORT_IMAGE_THRESHOLD
            or side_structure >= PROFILE_VISUAL_SUPPORT_STRUCTURE_THRESHOLD
        )
    ):
        return "profile", "multi_model_yaw_agreement"
    if (
        profile_allowed
        and candidate_yaw_disagreement >= PROFILE_CANDIDATE_YAW_DISAGREEMENT_THRESHOLD
        and max_disagreement_bbox_fraction >= PROFILE_MAX_DISAGREEMENT_BBOX_FRACTION_THRESHOLD
        and side_structure >= PROFILE_CANDIDATE_SIDE_STRUCTURE_THRESHOLD
        and candidate_profile_yaw_agreement
    ):
        return "profile", "candidate_instability"
    if (
        abs_image_yaw < WEAK_YAW_CAP_IMAGE_THRESHOLD
        and side_structure < WEAK_YAW_CAP_STRUCTURE_THRESHOLD
        and abs_landmark_yaw < LARGE_YAW_LANDMARK_THRESHOLD
    ):
        if abs_landmark_yaw <= 15.0:
            return "frontal", "low_yaw"
        return "intermediate", "weak_visual_shape_cap"
    if (
        abs_image_yaw >= LARGE_YAW_IMAGE_SIGNAL_THRESHOLD
        or abs_landmark_yaw >= LARGE_YAW_LANDMARK_THRESHOLD
        or abs_trusted_yaw >= LARGE_YAW_LANDMARK_THRESHOLD
    ):
        return "large_yaw", "yaw_evidence"
    if abs_landmark_yaw <= 15.0:
        return "frontal", "low_yaw"
    return "intermediate", "moderate_yaw"


def _consensus_landmarks(candidates: T.Sequence[CandidateRecord]) -> np.ndarray:
    """Return median candidate landmarks in frame coordinates."""
    stack = np.stack([candidate.landmarks.astype("float64") for candidate in candidates], axis=0)
    return np.median(stack, axis=0).astype("float32", copy=False)  # type: ignore[no-any-return]


def _image_geometry_yaw_signal(
    landmarks: np.ndarray,
    detector_bbox: T.Sequence[float] | None,
) -> tuple[float, float, float]:
    """Return signed image-aware yaw signal and supporting asymmetry terms."""
    points = np.asarray(landmarks, dtype="float64")
    if points.shape[0] < 68:
        return 0.0, 0.0, 0.0
    if detector_bbox is not None:
        left, _top, right, _bottom = (float(value) for value in detector_bbox)
    else:
        bbox = _landmark_bbox(points)
        if bbox is None:
            return 0.0, 0.0, 0.0
        left, _top, right, _bottom = bbox
    width = max(right - left, 1.0)
    face_center_x = (left + right) * 0.5
    nose_tip_x = float(points[30, 0])
    mouth_center_x = float(points[48:68, 0].mean())
    jaw_left_x = float(points[0, 0])
    jaw_right_x = float(points[16, 0])
    nose_offset = (nose_tip_x - face_center_x) / (width * 0.5)
    mouth_offset = (mouth_center_x - face_center_x) / (width * 0.5)
    left_span = max(nose_tip_x - jaw_left_x, 1e-6)
    right_span = max(jaw_right_x - nose_tip_x, 1e-6)
    jaw_asymmetry = (right_span - left_span) / max(right_span + left_span, 1e-6)
    signal = (0.55 * nose_offset) + (0.25 * mouth_offset) + (0.20 * jaw_asymmetry)
    return float(signal), float(nose_offset), float(jaw_asymmetry)


def infer_runtime_bucket(
    *,
    image_crop: np.ndarray | None,
    crop_to_frame_matrix: np.ndarray | None,
    detector_bbox: T.Sequence[float] | None,
    candidates: T.Sequence[CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    yaw_estimate: float | None,
    roll_estimate: float | None,
    max_disagreement_px: float,
    hard_roll_degrees: float,
) -> RuntimeBucketResult:
    """Infer an image-aware runtime bucket for resolver routing and metadata."""
    consensus = _consensus_landmarks(candidates)
    yaw_signal, nose_offset, jaw_asymmetry = _image_geometry_yaw_signal(consensus, detector_bbox)
    candidate_yaw_disagreement = _signed_candidate_yaw_disagreement(metrics)
    dominant_candidate_yaw = _dominant_candidate_yaw(metrics)
    trusted_single_model_yaw = _trusted_single_model_yaw(metrics)
    profile_yaw_agreement, profile_yaw_side, profile_yaw_agreement_count = (
        _single_model_yaw_side_agreement(
            metrics,
            min_abs_degrees=PROFILE_MULTI_MODEL_YAW_THRESHOLD,
        )
    )
    (
        candidate_profile_yaw_agreement,
        candidate_profile_yaw_side,
        candidate_profile_yaw_agreement_count,
    ) = _single_model_yaw_side_agreement(
        metrics,
        min_abs_degrees=PROFILE_CANDIDATE_TRUSTED_YAW_THRESHOLD,
    )
    max_disagreement_bbox_fraction = _normalized_max_disagreement(
        max_disagreement_px,
        detector_bbox,
        consensus,
    )
    geometry_side, geometry_side_source = _runtime_side_from_signals(
        image_geometry_yaw_signal=yaw_signal,
        nose_offset_from_face_center=nose_offset,
        mouth_nose_jaw_asymmetry=jaw_asymmetry,
        landmark_pose_yaw=yaw_estimate,
        dominant_candidate_yaw=dominant_candidate_yaw,
    )
    trusted_side, trusted_side_source, side_confidence, side_votes = (
        _runtime_side_from_single_model_yaws(metrics)
    )
    if trusted_side is None:
        yaw_side = geometry_side
        yaw_side_source = f"landmark_geometry_fallback:{geometry_side_source}"
        side_confidence = 0.0
    else:
        yaw_side = trusted_side
        yaw_side_source = trusted_side_source
    side_conflict = bool(geometry_side != yaw_side)
    yaw_severity, yaw_severity_source = _runtime_yaw_severity(
        image_geometry_yaw_signal=yaw_signal,
        landmark_pose_yaw=yaw_estimate,
        trusted_single_model_yaw=trusted_single_model_yaw,
        profile_yaw_agreement=profile_yaw_agreement,
        candidate_profile_yaw_agreement=candidate_profile_yaw_agreement,
        candidate_yaw_disagreement=candidate_yaw_disagreement,
        max_disagreement_bbox_fraction=max_disagreement_bbox_fraction,
        nose_offset_from_face_center=nose_offset,
        mouth_nose_jaw_asymmetry=jaw_asymmetry,
    )
    abs_roll = 0.0 if roll_estimate is None else abs(float(roll_estimate))
    hard_roll_support_count = _roll_support_count(
        metrics,
        consensus_roll=roll_estimate,
        threshold_degrees=hard_roll_degrees,
    )
    hard_roll = (
        abs_roll >= hard_roll_degrees
        and hard_roll_support_count >= ROLL_BUCKET_MIN_SUPPORTING_CANDIDATES
    )
    extreme_roll_threshold = max(hard_roll_degrees * 1.8, 55.0)
    extreme_roll_support_count = _roll_support_count(
        metrics,
        consensus_roll=roll_estimate,
        threshold_degrees=extreme_roll_threshold,
    )
    extreme_roll = (
        abs_roll >= extreme_roll_threshold
        and extreme_roll_support_count >= ROLL_BUCKET_MIN_SUPPORTING_CANDIDATES
    )
    features: dict[str, T.Any] = {
        "left_eye_visual_score": None,
        "right_eye_visual_score": None,
        "eye_visibility_asymmetry": None,
        "nose_offset_from_face_center": nose_offset,
        "mouth_nose_jaw_asymmetry": jaw_asymmetry,
        "candidate_yaw_disagreement": candidate_yaw_disagreement,
        "max_disagreement_px": max_disagreement_px,
        "max_disagreement_bbox_fraction": max_disagreement_bbox_fraction,
        "landmark_pose_yaw": yaw_estimate,
        "landmark_pose_roll": roll_estimate,
        "trusted_single_model_yaw": trusted_single_model_yaw,
        "profile_yaw_agreement": profile_yaw_agreement,
        "profile_yaw_agreement_side": profile_yaw_side,
        "profile_yaw_agreement_count": profile_yaw_agreement_count,
        "candidate_profile_yaw_agreement": candidate_profile_yaw_agreement,
        "candidate_profile_yaw_agreement_side": candidate_profile_yaw_side,
        "candidate_profile_yaw_agreement_count": candidate_profile_yaw_agreement_count,
        "image_geometry_yaw_signal": yaw_signal,
        "dominant_candidate_yaw": dominant_candidate_yaw,
        "runtime_bucket_side": yaw_side,
        "runtime_bucket_side_source": yaw_side_source,
        "runtime_bucket_side_confidence": side_confidence,
        "runtime_bucket_side_votes": side_votes,
        "runtime_bucket_side_conflict": side_conflict,
        "runtime_bucket_geometry_side": geometry_side,
        "runtime_bucket_geometry_side_source": geometry_side_source,
        "runtime_bucket_severity": yaw_severity,
        "runtime_bucket_severity_source": yaw_severity_source,
        "runtime_bucket_hard_roll_supported": hard_roll,
        "runtime_bucket_hard_roll_support_count": hard_roll_support_count,
        "runtime_bucket_extreme_roll_supported": extreme_roll,
        "runtime_bucket_extreme_roll_support_count": extreme_roll_support_count,
    }
    if image_crop is not None and crop_to_frame_matrix is not None:
        try:
            crop_points = _frame_to_crop_pixels(consensus, crop_to_frame_matrix, image_crop)
            gray = _crop_gray(image_crop)
            left_eye_score = _eye_visual_score(gray, crop_points[36:42])
            right_eye_score = _eye_visual_score(gray, crop_points[42:48])
            eye_asymmetry = left_eye_score - right_eye_score
            features.update(
                {
                    "left_eye_visual_score": left_eye_score,
                    "right_eye_visual_score": right_eye_score,
                    "eye_visibility_asymmetry": eye_asymmetry,
                }
            )
            if max(left_eye_score, right_eye_score) >= 0.18 and abs(eye_asymmetry) >= 0.12:
                eye_side = "left" if eye_asymmetry < 0 else "right"
                features["runtime_bucket_eye_side"] = eye_side
                features["runtime_bucket_eye_side_conflict"] = bool(eye_side != yaw_side)
        except Exception as err:  # noqa: BLE001
            logger.debug("image-aware runtime bucket eye scoring failed: %s", err)

    if yaw_severity == "profile" and hard_roll:
        return RuntimeBucketResult(bucket=f"rolled_profile_{yaw_side}", features=features)
    if yaw_severity == "large_yaw" and hard_roll:
        return RuntimeBucketResult(bucket=f"rolled_large_yaw_{yaw_side}", features=features)
    if yaw_severity == "profile":
        return RuntimeBucketResult(bucket=f"profile_{yaw_side}", features=features)
    if yaw_severity == "large_yaw":
        return RuntimeBucketResult(bucket=f"large_yaw_{yaw_side}", features=features)

    if extreme_roll:
        return RuntimeBucketResult(bucket="extreme_roll", features=features)
    if hard_roll:
        return RuntimeBucketResult(bucket="large_roll", features=features)

    if yaw_severity == "frontal" and abs_roll <= max(hard_roll_degrees * 0.5, 12.0):
        return RuntimeBucketResult(bucket="frontal", features=features)
    return RuntimeBucketResult(bucket="intermediate", features=features)


def _roll_support_count(
    metrics: T.Mapping[str, CandidateMetrics],
    *,
    consensus_roll: float | None,
    threshold_degrees: float,
    agreement_degrees: float = ROLL_BUCKET_AGREEMENT_DEGREES,
) -> int:
    """Return count of roll estimates that independently support a hard roll bucket."""
    if consensus_roll is None or abs(float(consensus_roll)) < threshold_degrees:
        return 0
    support = 0
    for name, metric in metrics.items():
        if _is_canonical_strategy_name(name):
            continue
        roll = metric.roll_degrees
        if roll is None or not math.isfinite(float(roll)):
            continue
        if abs(float(roll)) < threshold_degrees:
            continue
        if abs(_signed_degree_delta(float(roll), float(consensus_roll))) <= agreement_degrees:
            support += 1
    return support


def _score_delta_passes(
    *,
    replacement: str,
    selected: str,
    scores: T.Mapping[str, float],
    min_delta: float,
) -> bool:
    """Return whether replacement is materially lower risk than selected."""
    replacement_score = scores.get(replacement)
    selected_score = scores.get(selected)
    if replacement_score is None or selected_score is None:
        return False
    if not math.isfinite(float(replacement_score)) or not math.isfinite(float(selected_score)):
        return False
    return float(replacement_score) < float(selected_score) - min_delta


def _high_risk_safe_fallback_candidate(
    *,
    scores: T.Mapping[str, float],
    selectable: T.AbstractSet[str],
    candidates: T.Mapping[str, CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    risk_floor: float,
) -> str | None:
    """Return the lowest-risk valid single model when every selectable risk is high."""
    if risk_floor < 0.0:
        return None
    selectable_scores = [
        float(scores[name])
        for name in selectable
        if name in scores and math.isfinite(float(scores[name]))
    ]
    if not selectable_scores or min(selectable_scores) <= risk_floor:
        return None
    return _lowest_risk_valid_single(
        scores=scores,
        selectable=selectable,
        candidates=candidates,
        metrics=metrics,
    )


def _lowest_risk_valid_single(
    *,
    scores: T.Mapping[str, float],
    selectable: T.AbstractSet[str],
    candidates: T.Mapping[str, CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
) -> str | None:
    """Return the lowest-risk geometry-valid single model using stable tie-breaks."""
    tie_order = {name: index for index, name in enumerate(SAFE_SINGLE_TIE_BREAKER)}
    valid_singles = [
        name
        for name, candidate in candidates.items()
        if name in selectable
        and not candidate.is_fusion
        and name in metrics
        and not metrics[name].geometry_veto_reasons
    ]
    if not valid_singles:
        return None
    return min(
        valid_singles,
        key=lambda name: (
            scores.get(name, float("inf")),
            tie_order.get(name, len(tie_order)),
            name,
        ),
    )


def _hard_slice_safe_single_candidate(
    *,
    selected: str,
    candidates: T.Mapping[str, CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    candidate_extra_features: T.Mapping[str, T.Mapping[str, float]],
    condition: str,
    runtime_bucket: str,
    runtime_bucket_source: str,
    scores: T.Mapping[str, float],
    selectable: T.AbstractSet[str],
) -> str | None:
    """Reject consensus-collapse fusion on derived rolled/profile hard slices."""
    hard_label = condition or runtime_bucket
    if (
        hard_label not in HARD_SLICE_SAFE_SINGLE_BUCKETS
        or runtime_bucket_source != "derived_no_image_evidence"
    ):
        return None
    selected_candidate = candidates.get(selected)
    if selected_candidate is None or not selected_candidate.is_fusion:
        return None
    selected_features = candidate_extra_features.get(selected, {})
    consensus_like = bool(selected_features.get("candidate_is_consensus_like", 0.0) >= 1.0)
    single_disagreement_px = float(selected_features.get("single_model_disagreement_px", 0.0))
    distance_to_hrnet = float(selected_features.get("candidate_distance_to_hrnet", 0.0))
    consensus_collapse = (
        consensus_like
        and single_disagreement_px >= HARD_SLICE_SINGLE_MODEL_DISAGREEMENT_PX_THRESHOLD
        and distance_to_hrnet >= HARD_SLICE_DISTANCE_TO_HRNET_THRESHOLD
    )
    if not consensus_collapse:
        return None
    return _lowest_risk_valid_single(
        scores=scores,
        selectable=selectable,
        candidates=candidates,
        metrics=metrics,
    )


def _hard_pose_plain_average_replacement(
    *,
    selected: str,
    bucket: str,
    selectable: T.AbstractSet[str],
    candidates: T.Mapping[str, CandidateRecord],
    metrics: T.Mapping[str, CandidateMetrics],
    scores: T.Mapping[str, float] | None,
    min_delta: float,
) -> str | None:
    """Avoid plain-average in hard-pose buckets unless it beats HRNet/SPIGA by margin."""
    if selected != "plain_average" or not bucket.startswith(
        HARD_POSE_PLAIN_AVERAGE_GUARD_BUCKET_PREFIXES
    ):
        return None
    safe_models = [
        model
        for model in HARD_POSE_PLAIN_AVERAGE_SAFE_MODELS
        if model in selectable
        and model in candidates
        and not candidates[model].is_fusion
        and model in metrics
        and not metrics[model].geometry_veto_reasons
        and np.all(np.isfinite(np.asarray(candidates[model].landmarks, dtype="float64")))
    ]
    if not safe_models:
        return None
    if scores is None:
        return safe_models[0]

    selected_score = scores.get(selected)
    scored_safe_models = [
        model for model in safe_models if model in scores and math.isfinite(float(scores[model]))
    ]
    if (
        selected_score is None
        or not math.isfinite(float(selected_score))
        or not scored_safe_models
    ):
        return safe_models[0]

    order = {model: index for index, model in enumerate(HARD_POSE_PLAIN_AVERAGE_SAFE_MODELS)}
    replacement = min(scored_safe_models, key=lambda model: (scores[model], order[model]))
    if float(selected_score) < float(scores[replacement]) - min_delta:
        return None
    return replacement


def resolve_runtime(
    predictions: T.Sequence[ModelPrediction],
    config: RuntimeResolverConfig,
    *,
    detector_bbox: T.Sequence[float] | None = None,
    image_crop: np.ndarray | None = None,
    crop_to_frame_matrix: np.ndarray | None = None,
    preloaded_scorer: T.Any = None,
) -> RuntimeResolverResult:
    """Resolve one face using runtime candidate diagnostics and policy selection."""
    learned_policies = set(LEARNED_POLICIES)
    if config.policy not in {"roll_aware_veto", *learned_policies}:
        raise RuntimeResolverError(f"unsupported runtime resolver policy {config.policy!r}")
    logger.debug(
        "[RuntimeResolver] resolving face policy=%s scorer=%s bbox=%s image_evidence=%s "
        "general=%s hard=%s secondary=%s fallback=%s",
        config.policy,
        bool(config.scorer_path),
        detector_bbox,
        image_crop is not None and crop_to_frame_matrix is not None,
        config.general_strategy,
        config.hard_case_strategy,
        config.secondary_hard_case_strategy,
        config.fallback_strategy,
    )
    candidates = build_candidates(predictions, config)
    if not candidates:
        raise RuntimeResolverError("no runtime resolver candidates were provided")
    logger.debug(
        "[RuntimeResolver] built candidates=%s from models=%s",
        [candidate.name for candidate in candidates],
        [prediction.model for prediction in predictions],
    )
    _assert_frame_space_candidates(candidates, detector_bbox)

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
    max_disagreement_px = _max_landmark_consensus_px(candidates)
    runtime_bucket = infer_runtime_bucket(
        image_crop=image_crop,
        crop_to_frame_matrix=crop_to_frame_matrix,
        detector_bbox=detector_bbox,
        candidates=candidates,
        metrics=metrics,
        yaw_estimate=yaw_estimate,
        roll_estimate=roll_estimate,
        max_disagreement_px=max_disagreement_px,
        hard_roll_degrees=config.hard_roll_degrees,
    )
    bucket = runtime_bucket.bucket
    logger.debug(
        "[RuntimeResolver] bucket=%s roll=%s yaw=%s max_disagreement_px=%.3f features=%s",
        bucket,
        roll_estimate,
        yaw_estimate,
        max_disagreement_px,
        {
            key: runtime_bucket.features.get(key)
            for key in (
                "runtime_bucket_severity",
                "runtime_bucket_severity_source",
                "runtime_bucket_side",
                "runtime_bucket_side_source",
                "candidate_yaw_disagreement",
                "max_disagreement_bbox_fraction",
            )
        },
    )
    _trace("[RuntimeResolver] bucket features: %s", runtime_bucket.features)

    for name, metric in metrics.items():
        metric.geometry_veto_reasons = _shape_reasons(bucket, name, metric)
    runtime_bucket_source = (
        "runtime_image_evidence" if image_crop is not None else "derived_no_image_evidence"
    )
    candidate_extra_features = _candidate_extra_features(
        candidates,
        metrics,
        reference_bbox=reference_bbox,
        condition=bucket,
        runtime_bucket=bucket,
        runtime_bucket_source=runtime_bucket_source,
    )

    available = {candidate.name for candidate in candidates}
    priority = _priority_for_bucket(bucket, available)
    roll_vetoed = _roll_vetoes(
        candidates,
        metrics,
        threshold_deg=config.roll_veto_degrees,
        consensus_roll=roll_estimate,
    )
    geometry_vetoed = {name for name, metric in metrics.items() if metric.geometry_veto_reasons}
    fatal_shape_vetoed = {name for name, metric in metrics.items() if metric.shape_veto_reasons}
    vetoed = roll_vetoed | geometry_vetoed
    normal_vetoed = vetoed - fatal_shape_vetoed
    survivors = available - vetoed
    logger.debug(
        "[RuntimeResolver] veto summary bucket=%s survivors=%s roll_vetoed=%s "
        "normal_vetoed=%s fatal_shape_vetoed=%s geometry_vetoed=%s",
        bucket,
        sorted(survivors),
        sorted(roll_vetoed & available),
        sorted(normal_vetoed & available),
        sorted(fatal_shape_vetoed & available),
        {
            name: list(metrics[name].geometry_veto_reasons)
            for name in sorted(geometry_vetoed & available)
        },
    )
    _trace(
        "[RuntimeResolver] candidate metrics: %s",
        {
            name: {
                "roll": metric.roll_degrees,
                "yaw": metric.yaw_degrees,
                "cloud_area_ratio": metric.cloud_area_ratio,
                "hull_area_ratio": metric.hull_area_ratio,
                "points_outside_expanded_bbox_fraction": (
                    metric.points_outside_expanded_bbox_fraction
                ),
                "landmark_consensus_distance": metric.landmark_consensus_distance,
                "roi_center_consensus_distance": metric.roi_center_consensus_distance,
                "shape_plausibility_score": metric.shape_plausibility_score,
                "shape_veto_reasons": metric.shape_veto_reasons,
                "max_edge_length_ratio": metric.max_edge_length_ratio,
                "mean_shape_fit_error": metric.mean_shape_fit_error,
                "topology_violation_count": metric.topology_violation_count,
                "geometry_veto_reasons": metric.geometry_veto_reasons,
            }
            for name, metric in metrics.items()
        },
    )
    hard_bucket = bucket not in {"frontal", "intermediate", "no_pose"}
    risk_route = (
        "high_risk"
        if hard_bucket or max_disagreement_px > config.hard_disagreement_px or vetoed
        else "low_risk"
    )
    if not survivors:
        fallback_reason: str | None = "all_candidates_vetoed"
    else:
        fallback_reason = None
    hard_pose_plain_average_guard_used = False
    hard_pose_plain_average_rejected_candidate = ""
    scorer_metadata: dict[str, T.Any] = {}
    if config.policy in learned_policies:
        if preloaded_scorer is not None:
            logger.debug(
                "[RuntimeResolver] using preloaded scorer type=%s", type(preloaded_scorer).__name__
            )
            scorer = preloaded_scorer
        else:
            if not config.scorer_path:
                raise RuntimeResolverError(f"{config.policy} requires resolver_scorer_path")
            logger.debug("[RuntimeResolver] loading scorer path=%s", config.scorer_path)
            scorer = load_runtime_resolver_scorer(config.scorer_path)
        logger.debug(
            "[RuntimeResolver] loaded scorer type=%s version=%s policy=%s",
            type(scorer).__name__,
            getattr(scorer, "version", ""),
            getattr(scorer, "runtime_policy", ""),
        )
        scorer_policy = getattr(scorer, "runtime_policy", "")
        if scorer_policy and scorer_policy != config.policy:
            # The artifact was trained for a different runtime policy than the
            # one currently configured. Refuse to proceed: the production
            # bundle's manifest mapped this file under the wrong slot, or the
            # operator pointed resolver_policy at a scorer it does not match.
            # Either way, predictions would be silently mis-ranked.
            raise RuntimeResolverError(
                f"scorer runtime_policy={scorer_policy!r} does not match "
                f"config policy={config.policy!r}; re-promote the artifact "
                "or align resolver_policy with the installed scorer"
            )
        model_available = {prediction.model: True for prediction in predictions}
        logger.debug("[RuntimeResolver] scoring candidates count=%d", len(candidates))
        scores = score_runtime_candidates(
            scorer,
            candidates,
            metrics,
            runtime_bucket=bucket,
            risk_route=risk_route,
            model_predictions_available=model_available,
            roll_estimate=roll_estimate,
            yaw_estimate=yaw_estimate,
            candidate_yaw_disagreement=runtime_bucket.features.get("candidate_yaw_disagreement"),
            max_disagreement_px=max_disagreement_px,
            runtime_bucket_source=runtime_bucket_source,
            candidate_extra_features=candidate_extra_features,
        )
        logger.debug("[RuntimeResolver] scored candidates=%s", scores)
        logger.debug(
            "[RuntimeResolver] scorer ranked candidates=%s",
            [name for name, _score in sorted(scores.items(), key=lambda item: (item[1], item[0]))],
        )
        _trace("[RuntimeResolver] scorer candidate scores: %s", scores)
        by_name = {candidate.name: candidate for candidate in candidates}
        hard_slice_fallback_used = False
        safe_fallback_used = False
        if not survivors:
            selected = _all_candidates_vetoed_fallback(
                candidates,
                config,
                fatal_shape_vetoed=fatal_shape_vetoed,
            )
            rejected_candidate = ""
        else:
            selectable = survivors
            selected = min(
                selectable,
                key=lambda name: (scores.get(name, float("inf")), priority.index(name), name),
            )
            hard_slice_fallback = _hard_slice_safe_single_candidate(
                selected=selected,
                candidates=by_name,
                metrics=metrics,
                candidate_extra_features=candidate_extra_features,
                condition=bucket,
                runtime_bucket=bucket,
                runtime_bucket_source=runtime_bucket_source,
                scores=scores,
                selectable=selectable,
            )
            hard_slice_fallback_used = (
                hard_slice_fallback is not None and hard_slice_fallback != selected
            )
            if hard_slice_fallback_used:
                rejected_candidate = selected
                selected = hard_slice_fallback  # type: ignore[assignment]
                fallback_reason = "consensus_collapse_fusion_rejected"
            else:
                rejected_candidate = ""
            safe_fallback = _high_risk_safe_fallback_candidate(
                scores=scores,
                selectable=selectable,
                candidates=by_name,
                metrics=metrics,
                risk_floor=config.risk_floor_for_safe_fallback,
            )
            safe_fallback_used = (
                safe_fallback is not None
                and safe_fallback != selected
                and _score_delta_passes(
                    replacement=safe_fallback,
                    selected=selected,
                    scores=scores,
                    min_delta=config.safe_fallback_min_delta,
                )
            )
            if safe_fallback_used:
                rejected_candidate = selected
                selected = safe_fallback  # type: ignore[assignment]
                fallback_reason = "scorer_high_risk_safe_fallback"
            guard_replacement = _hard_pose_plain_average_replacement(
                selected=selected,
                bucket=bucket,
                selectable=selectable,
                candidates=by_name,
                metrics=metrics,
                scores=scores,
                min_delta=config.safe_fallback_min_delta,
            )
            hard_pose_plain_average_guard_used = (
                guard_replacement is not None and guard_replacement != selected
            )
            if hard_pose_plain_average_guard_used:
                rejected_candidate = selected
                hard_pose_plain_average_rejected_candidate = selected
                selected = guard_replacement  # type: ignore[assignment]
                fallback_reason = fallback_reason or "hard_pose_plain_average_guard"
        logger.debug(
            "[RuntimeResolver] learned selection=%s fallback_reason=%s rejected=%s",
            selected,
            fallback_reason,
            rejected_candidate,
        )
        candidate_rank = [
            name for name, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))
        ]
        scorer_metadata = {
            "selected_candidate_score": scores.get(selected),
            "candidate_scores": dict(sorted(scores.items())),
            "candidate_rank": candidate_rank,
            "candidate_risk_rank": candidate_rank,
            "scorer_path": str(config.scorer_path),
            "scorer_version": scorer.version,
            "scorer_model_type": scorer.model_type,
            "scorer_score_semantics": scorer.score_semantics,
            "scorer_safe_fallback_tie_breaker": SAFE_SINGLE_TIE_BREAKER,
            "scorer_safe_fallback_floor": config.risk_floor_for_safe_fallback,
            "scorer_safe_fallback_min_delta": config.safe_fallback_min_delta,
            "scorer_safe_fallback_used": safe_fallback_used,
            "hard_slice_safe_fallback_used": hard_slice_fallback_used,
            "hard_pose_plain_average_guard_used": hard_pose_plain_average_guard_used,
            "hard_pose_plain_average_guard_margin": config.safe_fallback_min_delta,
            "rejected_candidate": rejected_candidate,
            "replacement_candidate": (
                selected
                if fallback_reason
                in {"all_candidates_vetoed", "consensus_collapse_fusion_rejected"}
                or rejected_candidate
                else ""
            ),
        }
    else:
        selected = (
            _available_by_priority(priority, survivors)
            if survivors
            else _all_candidates_vetoed_fallback(
                candidates,
                config,
                fatal_shape_vetoed=fatal_shape_vetoed,
            )
        )
        by_name = {candidate.name: candidate for candidate in candidates}
        selectable = survivors if survivors else _finite_candidate_names(candidates)
        guard_replacement = _hard_pose_plain_average_replacement(
            selected=selected,
            bucket=bucket,
            selectable=selectable,
            candidates=by_name,
            metrics=metrics,
            scores=None,
            min_delta=config.safe_fallback_min_delta,
        )
        hard_pose_plain_average_guard_used = (
            guard_replacement is not None and guard_replacement != selected
        )
        if hard_pose_plain_average_guard_used:
            hard_pose_plain_average_rejected_candidate = selected
            selected = guard_replacement  # type: ignore[assignment]
            fallback_reason = fallback_reason or "hard_pose_plain_average_guard"
        logger.debug(
            "[RuntimeResolver] bucket-priority selection=%s fallback_reason=%s",
            selected,
            fallback_reason,
        )

    by_name = {candidate.name: candidate for candidate in candidates}
    candidate_veto_reasons = _candidate_veto_reasons(
        candidates,
        metrics,
        roll_vetoed=roll_vetoed,
        vetoed=vetoed,
    )
    metadata: dict[str, T.Any] = {
        "selected_candidate": selected,
        "runtime_bucket": bucket,
        "bucket": bucket,
        "runtime_bucket_source": runtime_bucket_source,
        "runtime_bucket_features": runtime_bucket.features,
        **runtime_bucket.features,
        "candidate_priority": priority,
        "vetoed": sorted(vetoed & available),
        "normal_vetoed": sorted(normal_vetoed & available),
        "fatal_shape_vetoed": sorted(fatal_shape_vetoed & available),
        "veto_reasons": {
            name: list(metric.geometry_veto_reasons)
            for name, metric in metrics.items()
            if metric.geometry_veto_reasons
        },
        "vetoed_candidates": candidate_veto_reasons,
        "candidate_veto_reasons": candidate_veto_reasons,
        "all_candidates_vetoed_count": len(vetoed & available) if not survivors else 0,
        "replacement_candidate": selected if fallback_reason == "all_candidates_vetoed" else "",
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
        "shape_plausibility_score": _metrics_payload(metrics, "shape_plausibility_score"),
        "shape_veto_reasons": _metrics_payload(metrics, "shape_veto_reasons"),
        "max_edge_length_ratio": _metrics_payload(metrics, "max_edge_length_ratio"),
        "mean_shape_fit_error": _metrics_payload(metrics, "mean_shape_fit_error"),
        "topology_violation_count": _metrics_payload(metrics, "topology_violation_count"),
        "model_predictions_available": {prediction.model: True for prediction in predictions},
        "roll_vetoed": sorted(roll_vetoed & available),
        "geometry_vetoed": sorted(geometry_vetoed & available),
        "rolls": _metrics_payload(metrics, "roll_degrees"),
        "yaws": _metrics_payload(metrics, "yaw_degrees"),
        "max_disagreement_px": max_disagreement_px,
        "risk_route": risk_route,
        "fallback_reason": fallback_reason,
        "fallback_used": fallback_reason is not None,
        "hard_pose_plain_average_guard_used": hard_pose_plain_average_guard_used,
        "hard_pose_plain_average_guard_margin": config.safe_fallback_min_delta,
        "hard_pose_plain_average_rejected_candidate": (hard_pose_plain_average_rejected_candidate),
        "policy": config.policy,
        **scorer_metadata,
    }
    logger.debug(
        "[RuntimeResolver] final selected=%s bucket=%s fallback=%s vetoed_count=%d",
        selected,
        bucket,
        fallback_reason,
        len(vetoed & available),
    )
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
    "RuntimeBucketResult",
    "RUNTIME_BUCKETS",
    "build_candidates",
    "resolve_runtime",
]
