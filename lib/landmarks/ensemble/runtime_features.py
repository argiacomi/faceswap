#!/usr/bin/env python3
"""Runtime-visible feature extraction for learned landmark candidate scoring.

The v3 label may use GT-derived transform regret, but these features may not.
This module is the single source of truth for features available at extraction
runtime. Training, evaluation, and the extract/runtime resolver path should all
call this module rather than rebuilding feature dictionaries independently.
"""

from __future__ import annotations

import math
import typing as T
from collections.abc import Mapping

CandidateLike = T.Any
MetricLike = T.Any

RUNTIME_FEATURE_CONTRACT_VERSION = "runtime_features_v1"

RUNTIME_PREFERRED_FEATURE_ORDER: tuple[str, ...] = (
    # Candidate structure.
    "candidate_is_single_model",
    "candidate_is_fusion",
    # Geometry / structural plausibility.
    "cloud_area_ratio",
    "hull_area_ratio",
    "points_outside_expanded_bbox_fraction",
    "eye_mouth_order_valid_after_deroll",
    "shape_plausibility_score",
    "max_edge_length_ratio",
    "mean_shape_fit_error",
    "topology_violation_count",
    "shape_veto_reason_count",
    # Consensus distances.
    "roi_center_consensus_distance",
    "landmark_consensus_distance",
    # Pose and pose-to-consensus deltas.
    "roll_degrees",
    "yaw_degrees",
    "roll_delta_to_consensus",
    "yaw_delta_to_consensus",
    # Disagreement features.
    "candidate_yaw_disagreement",
    "max_disagreement_px",
    # Veto / validity features.
    "has_geometry_veto",
    # Profile/occlusion visible-side features (#218). Emitted only for
    # profile/occlusion contexts; frontal anchors default these to 0.0.
    "profile_yaw_abs",
    "profile_yaw_signed",
    "profile_roll_abs",
    "profile_is_left",
    "profile_is_right",
    "profile_is_large_yaw",
    "profile_is_rolled",
    "profile_has_occlusion",
    "profile_has_single_eye_visible",
    "profile_has_mouth_or_jaw_occluded",
    "visible_side_consensus_distance",
    "visible_side_candidate_spread",
    "visible_eye_consensus_distance",
    "visible_brow_consensus_distance",
    "visible_mouth_corner_consensus_distance",
    "nose_bridge_consistency",
    "mouth_corner_asymmetry",
    "occluded_side_spread",
    "occluded_side_outlier_rate",
    "candidate_profile_validity_score",
    # Profile repair candidate features (#219).
    "candidate_is_profile_repaired",
    "profile_repair_source_rank",
    "profile_repair_visible_side_left",
    "profile_repair_visible_side_right",
    "profile_repair_candidate_shape_score",
)


def _float(value: T.Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool(value: T.Any) -> float:
    return 1.0 if bool(value) else 0.0


def candidate_feature_map(
    candidate: CandidateLike,
    metric: MetricLike,
    *,
    runtime_bucket: str = "",
    risk_route: str = "",
    model_predictions_available: T.Mapping[str, bool] | T.Iterable[str] | None = None,
    roll_estimate: float | None = None,
    yaw_estimate: float | None = None,
    candidate_yaw_disagreement: float | None = None,
    max_disagreement_px: float | None = None,
    runtime_bucket_source: str = "",
    hard_case_tags: T.Sequence[str] = (),
    candidate_extra_features: T.Mapping[str, T.Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Return runtime-visible numeric and one-hot features for one candidate.

    Do not add GT, NME, oracle, transform-regret, or label-only fields here.
    Feature names are intentionally stable because scorer artifacts persist the
    ordered feature list.
    """
    name = str(getattr(candidate, "name", ""))
    is_fusion = bool(getattr(candidate, "is_fusion", False))
    veto_reasons = tuple(getattr(metric, "geometry_veto_reasons", ()) or ())
    shape_veto_reasons = tuple(getattr(metric, "shape_veto_reasons", ()) or ())

    model_available: dict[str, bool] = {}
    if isinstance(model_predictions_available, Mapping):
        model_available = {
            str(model): bool(available) for model, available in model_predictions_available.items()
        }
    elif model_predictions_available is not None:
        model_available = {str(model): True for model in model_predictions_available}

    roll = getattr(metric, "roll_degrees", None)
    yaw = getattr(metric, "yaw_degrees", None)

    features: dict[str, float] = {
        "candidate_is_single_model": 0.0 if is_fusion else 1.0,
        "candidate_is_fusion": 1.0 if is_fusion else 0.0,
        "cloud_area_ratio": _float(getattr(metric, "cloud_area_ratio", None)),
        "hull_area_ratio": _float(getattr(metric, "hull_area_ratio", None)),
        "points_outside_expanded_bbox_fraction": _float(
            getattr(metric, "points_outside_expanded_bbox_fraction", None)
        ),
        "eye_mouth_order_valid_after_deroll": _bool(
            getattr(metric, "eye_mouth_order_valid_after_deroll", False)
        ),
        "roi_center_consensus_distance": _float(
            getattr(metric, "roi_center_consensus_distance", None)
        ),
        "landmark_consensus_distance": _float(
            getattr(metric, "landmark_consensus_distance", None)
        ),
        "shape_plausibility_score": _float(getattr(metric, "shape_plausibility_score", None)),
        "max_edge_length_ratio": _float(getattr(metric, "max_edge_length_ratio", None)),
        "mean_shape_fit_error": _float(getattr(metric, "mean_shape_fit_error", None)),
        "topology_violation_count": _float(getattr(metric, "topology_violation_count", None)),
        "shape_veto_reason_count": float(len(shape_veto_reasons)),
        "roll_degrees": _float(roll),
        "yaw_degrees": _float(yaw),
        "roll_delta_to_consensus": abs(_float(roll) - _float(roll_estimate)),
        "yaw_delta_to_consensus": abs(_float(yaw) - _float(yaw_estimate)),
        "candidate_yaw_disagreement": _float(candidate_yaw_disagreement),
        "max_disagreement_px": _float(max_disagreement_px),
        "has_geometry_veto": 1.0 if veto_reasons else 0.0,
    }

    features[f"candidate_name={name}"] = 1.0

    if runtime_bucket:
        features[f"runtime_bucket={runtime_bucket}"] = 1.0
    if risk_route:
        features[f"risk_route={risk_route}"] = 1.0
    if runtime_bucket_source:
        features[f"runtime_bucket_source={runtime_bucket_source}"] = 1.0

    for tag in hard_case_tags:
        features[f"hard_case_tag={tag}"] = 1.0
    for model, available in model_available.items():
        if available:
            features[f"model_predictions_available={model}"] = 1.0
    for reason in veto_reasons:
        features[f"geometry_veto_reason={reason}"] = 1.0
    for reason in shape_veto_reasons:
        features[f"shape_veto_reason={reason}"] = 1.0

    if candidate_extra_features is not None:
        features.update(candidate_extra_features.get(name, {}))

    return features


def runtime_candidate_feature_maps(
    candidates: T.Sequence[CandidateLike],
    metrics: T.Mapping[str, MetricLike],
    **context: T.Any,
) -> list[dict[str, float]]:
    """Return feature maps for candidates, in input order, using runtime-visible inputs only."""
    return [
        candidate_feature_map(candidate, metrics[str(candidate.name)], **context)
        for candidate in candidates
        if str(candidate.name) in metrics
    ]


def runtime_feature_order(
    feature_maps: T.Iterable[T.Mapping[str, float] | T.Any],
) -> tuple[str, ...]:
    """Return stable artifact feature order from runtime feature maps or row-like objects."""
    names: set[str] = set()
    for item in feature_maps:
        if isinstance(item, Mapping):
            names.update(str(name) for name in item)
            continue
        feature_values = getattr(item, "feature_values", None)
        if isinstance(feature_values, Mapping):
            names.update(str(name) for name in feature_values)

    ordered = [name for name in RUNTIME_PREFERRED_FEATURE_ORDER if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return tuple(ordered)


__all__ = [
    "RUNTIME_FEATURE_CONTRACT_VERSION",
    "RUNTIME_PREFERRED_FEATURE_ORDER",
    "candidate_feature_map",
    "runtime_candidate_feature_maps",
    "runtime_feature_order",
]
