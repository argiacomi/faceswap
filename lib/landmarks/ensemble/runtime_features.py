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

import numpy as np

CandidateLike = T.Any
MetricLike = T.Any

RUNTIME_FEATURE_CONTRACT_VERSION = "runtime_features_v1"

#: Substrings that must never appear in a runtime feature name. These mark
#: GT-derived, oracle, label-only, or transform-regret fields that are visible
#: only offline. The runtime feature builders and the stacked regressor training
#: path both assert against this list so a leaked label can never enter an
#: artifact's feature contract.
FORBIDDEN_RUNTIME_FEATURE_TOKENS: tuple[str, ...] = (
    "gt_",
    "_gt",
    "ground_truth",
    "truth",
    "nme",
    "oracle",
    "regret",
    "label",
    "is_selected",
    "was_selected",
)

#: Canonical adapter order for stacked-regression model-derived features. Missing
#: models contribute zeros plus an availability flag, so the feature vector is a
#: fixed width regardless of which adapters produced a prediction for a face.
STACKED_REGRESSION_FEATURE_MODELS: tuple[str, ...] = ("fan", "hrnet", "orformer", "spiga")

#: 5-region partition used for stacked-regression model-disagreement features.
#: Mirrors ``stacked_regressor.STACKED_REGION_INDICES`` (kept here to avoid an
#: import cycle; a drift-guard test asserts the two stay identical).
STACKED_REGRESSION_REGION_INDICES: dict[str, tuple[int, ...]] = {
    "jaw": tuple(range(0, 17)),
    "brows": tuple(range(17, 27)),
    "nose": tuple(range(27, 36)),
    "eyes": tuple(range(36, 48)),
    "mouth": tuple(range(48, 68)),
}
STACKED_REGRESSION_REGION_NAMES: tuple[str, ...] = tuple(STACKED_REGRESSION_REGION_INDICES)

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
    # Stacked residual candidate features (#223). Describe the regressor
    # correction so the learned scorer can rank/penalize it; emitted only for
    # the stacked_residual candidate, defaulting to 0.0 elsewhere.
    "candidate_is_stacked_residual",
    "stacked_residual_norm_max",
    "stacked_residual_norm_mean",
    "stacked_clip_applied",
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


def forbidden_runtime_features(names: T.Iterable[str]) -> tuple[str, ...]:
    """Return feature names that leak GT/NME/oracle/regret/label-only fields.

    The stacked regressor and scorer feature contracts must contain only
    runtime-visible inputs. Training paths call this to fail fast before
    persisting an artifact whose feature list references an offline label.
    """
    leaked: list[str] = []
    for name in names:
        lowered = str(name).lower()
        if any(token in lowered for token in FORBIDDEN_RUNTIME_FEATURE_TOKENS):
            leaked.append(str(name))
    return tuple(leaked)


def _bbox_bounds(
    reference_bbox: T.Sequence[float] | None,
    landmarks: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (x0, y0, size_x, size_y) from a bbox or landmark extent."""
    if reference_bbox is not None and len(reference_bbox) == 4:
        x0, y0, x1, y1 = (float(value) for value in reference_bbox)
        size_x = x1 - x0
        size_y = y1 - y0
    else:
        mins = landmarks.min(axis=0)
        maxs = landmarks.max(axis=0)
        x0, y0 = float(mins[0]), float(mins[1])
        size_x = float(maxs[0] - mins[0])
        size_y = float(maxs[1] - mins[1])
    size_x = size_x if abs(size_x) > 1e-6 else 1.0
    size_y = size_y if abs(size_y) > 1e-6 else 1.0
    return x0, y0, size_x, size_y


def stacked_regression_feature_map(
    *,
    base_landmarks: np.ndarray,
    model_landmarks: T.Mapping[str, np.ndarray],
    reference_bbox: T.Sequence[float] | None = None,
    runtime_bucket: str = "",
    risk_route: str = "",
    runtime_bucket_source: str = "",
    roll_estimate: float | None = None,
    yaw_estimate: float | None = None,
    candidate_yaw_disagreement: float | None = None,
    max_disagreement_px: float | None = None,
    hard_case_tags: T.Sequence[str] = (),
    model_predictions_available: T.Mapping[str, bool] | T.Iterable[str] | None = None,
) -> dict[str, float]:
    """Build the runtime-visible feature row for the stacked residual regressor.

    Inputs are the base candidate, the per-model single predictions (frame
    space), and the runtime pose/bucket context. The feature vector is a fixed
    width keyed on :data:`STACKED_REGRESSION_FEATURE_MODELS`, so missing models
    contribute zeros plus an availability flag. All features are runtime-visible;
    none derive from GT, NME, oracle, regret, or labels.
    """
    base = np.asarray(base_landmarks, dtype="float64")
    x0, y0, size_x, size_y = _bbox_bounds(reference_bbox, base)
    diag = float(math.hypot(size_x, size_y)) or 1.0

    available: dict[str, np.ndarray] = {
        str(model): np.asarray(points, dtype="float64")
        for model, points in model_landmarks.items()
        if np.asarray(points, dtype="float64").shape == (68, 2)
    }
    consensus = np.mean(list(available.values()), axis=0) if available else base.copy()

    base_centroid = base.mean(axis=0)
    features: dict[str, float] = {
        "base_centroid_x_norm": (float(base_centroid[0]) - x0) / size_x,
        "base_centroid_y_norm": (float(base_centroid[1]) - y0) / size_y,
        "base_bbox_aspect": size_x / size_y,
        "roll_degrees": _float(roll_estimate),
        "yaw_degrees": _float(yaw_estimate),
        "abs_roll_degrees": abs(_float(roll_estimate)),
        "abs_yaw_degrees": abs(_float(yaw_estimate)),
        "candidate_yaw_disagreement": _float(candidate_yaw_disagreement),
        "overall_disagreement_norm": _float(max_disagreement_px) / diag,
        "available_model_count": float(len(available)),
    }

    # Per-region inter-model disagreement (mean per-point std across models).
    if len(available) >= 2:
        stack = np.stack(list(available.values()), axis=0)
        per_point_std = stack.std(axis=0).mean(axis=1)  # (68,)
    else:
        per_point_std = np.zeros(68, dtype="float64")
    for region, indices in STACKED_REGRESSION_REGION_INDICES.items():
        region_std = float(np.mean(per_point_std[list(indices)])) if indices else 0.0
        features[f"region_disagreement_{region}"] = region_std / diag

    # Per-model per-region centroid offset from consensus (directional bias).
    consensus_region_centroids = {
        region: consensus[list(indices)].mean(axis=0)
        for region, indices in STACKED_REGRESSION_REGION_INDICES.items()
    }
    for model in STACKED_REGRESSION_FEATURE_MODELS:
        points = available.get(model)
        features[f"model_available_{model}"] = 1.0 if points is not None else 0.0
        for region, indices in STACKED_REGRESSION_REGION_INDICES.items():
            if points is None:
                features[f"model_{model}_region_{region}_dx"] = 0.0
                features[f"model_{model}_region_{region}_dy"] = 0.0
                continue
            centroid = points[list(indices)].mean(axis=0)
            offset = centroid - consensus_region_centroids[region]
            features[f"model_{model}_region_{region}_dx"] = float(offset[0]) / diag
            features[f"model_{model}_region_{region}_dy"] = float(offset[1]) / diag

    if runtime_bucket:
        features[f"runtime_bucket={runtime_bucket}"] = 1.0
    if risk_route:
        features[f"risk_route={risk_route}"] = 1.0
    if runtime_bucket_source:
        features[f"runtime_bucket_source={runtime_bucket_source}"] = 1.0
    for tag in hard_case_tags:
        features[f"hard_case_tag={tag}"] = 1.0

    model_available: dict[str, bool] = {}
    if isinstance(model_predictions_available, Mapping):
        model_available = {
            str(model): bool(value) for model, value in model_predictions_available.items()
        }
    elif model_predictions_available is not None:
        model_available = {str(model): True for model in model_predictions_available}
    for model, is_available in model_available.items():
        if is_available:
            features[f"model_predictions_available={model}"] = 1.0

    return features


__all__ = [
    "FORBIDDEN_RUNTIME_FEATURE_TOKENS",
    "RUNTIME_FEATURE_CONTRACT_VERSION",
    "RUNTIME_PREFERRED_FEATURE_ORDER",
    "STACKED_REGRESSION_FEATURE_MODELS",
    "STACKED_REGRESSION_REGION_INDICES",
    "STACKED_REGRESSION_REGION_NAMES",
    "candidate_feature_map",
    "forbidden_runtime_features",
    "runtime_candidate_feature_maps",
    "runtime_feature_order",
    "stacked_regression_feature_map",
]
