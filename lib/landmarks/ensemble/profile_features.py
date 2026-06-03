#!/usr/bin/env python3
"""Profile/occlusion-aware candidate features for learned scoring (#218).

Profile/occlusion faces fail differently from frontal faces: the visible side
can be well-aligned while the occluded side / full-face topology is unstable.
These features expose visible-side vs occluded-side behaviour to the scorer so
the profile specialist (``learned_quality_v3_profile``) can prefer candidates
with good visible-side geometry rather than chasing full-face point error.

The features are only emitted for profile/occlusion contexts; frontal anchors
get no profile features (they default to 0.0 in the feature matrix), so the
normal route's feature distribution is unchanged.
"""

from __future__ import annotations

import typing as T
from typing import cast

import numpy as np

# Canonical 68-point side groups (image space). ``RIGHT_SIDE`` covers the
# subject-right jaw/brow/eye, ``LEFT_SIDE`` the subject-left; the nose bridge
# and mouth are treated as center-stable anchors.
RIGHT_SIDE_68: tuple[int, ...] = (*range(0, 8), *range(17, 22), *range(36, 42))
LEFT_SIDE_68: tuple[int, ...] = (*range(9, 17), *range(22, 27), *range(42, 48))
NOSE_68: tuple[int, ...] = tuple(range(27, 36))
NOSE_BRIDGE_68: tuple[int, ...] = (27, 28, 29, 30)
MOUTH_68: tuple[int, ...] = tuple(range(48, 68))
RIGHT_EYE_68: tuple[int, ...] = tuple(range(36, 42))
LEFT_EYE_68: tuple[int, ...] = tuple(range(42, 48))
RIGHT_BROW_68: tuple[int, ...] = tuple(range(17, 22))
LEFT_BROW_68: tuple[int, ...] = tuple(range(22, 27))
MOUTH_CORNERS_68: tuple[int, int] = (48, 54)
NOSE_TIP_68 = 30

LARGE_YAW_DEGREES = 35.0
ROLLED_DEGREES = 25.0
OUTLIER_FRACTION_OF_DIAG = 0.06

#: Stable feature names emitted for profile/occlusion contexts. Kept in module
#: scope so the runtime feature order and tests can reference them.
PROFILE_FEATURE_NAMES: tuple[str, ...] = (
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
)


def profile_side_from_context(
    *,
    runtime_bucket: str = "",
    condition: str = "",
    yaw_estimate: float | None = None,
) -> str:
    """Return ``'left'``, ``'right'``, or ``''`` for the visible profile side.

    Resolution order: explicit left/right bucket or condition labels first, then
    the yaw estimate (negative yaw -> left, positive -> right).
    """
    blob = f"{runtime_bucket} {condition}".lower()
    if "left" in blob:
        return "left"
    if "right" in blob:
        return "right"
    if yaw_estimate is not None:
        yaw = float(yaw_estimate)
        if yaw < -5.0:
            return "left"
        if yaw > 5.0:
            return "right"
    return ""


def visible_side_indices(side: str) -> tuple[int, ...]:
    """Return the visible-side landmark indices for ``side``."""
    if side == "left":
        return LEFT_SIDE_68
    if side == "right":
        return RIGHT_SIDE_68
    return ()


def occluded_side_indices(side: str) -> tuple[int, ...]:
    """Return the occluded-side landmark indices for ``side``."""
    if side == "left":
        return RIGHT_SIDE_68
    if side == "right":
        return LEFT_SIDE_68
    return ()


def _visible_eye_brow(side: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if side == "left":
        return LEFT_EYE_68, LEFT_BROW_68
    if side == "right":
        return RIGHT_EYE_68, RIGHT_BROW_68
    return (), ()


def _as_points(landmarks: T.Any) -> np.ndarray | None:
    points = np.asarray(landmarks, dtype="float64")
    if points.ndim != 2 or points.shape[0] < 68 or points.shape[1] < 2:
        return None
    if not np.all(np.isfinite(points[:68, :2])):
        return None
    return cast(np.ndarray | None, points[:68, :2])


def _region_distance(points: np.ndarray, consensus: np.ndarray, indices: T.Sequence[int]) -> float:
    if not indices:
        return 0.0
    idx = np.asarray(indices, dtype="int64")
    deltas = points[idx] - consensus[idx]
    return float(np.mean(np.linalg.norm(deltas, axis=1)))


def _robust_consensus(point_stack: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.median(point_stack, axis=0))


def profile_candidate_features(
    candidates: T.Sequence[T.Any],
    metrics: T.Mapping[str, T.Any],
    *,
    diag: float | None,
    side: str,
    yaw_estimate: float | None,
    roll_estimate: float | None,
    has_occlusion: bool,
    has_single_eye_visible: bool = False,
    has_mouth_or_jaw_occluded: bool = False,
) -> dict[str, dict[str, float]]:
    """Return profile/visible-side features keyed by candidate name.

    Returns an empty mapping when the context is neither profile (no visible
    side resolved) nor occluded, so frontal anchors are left untouched.
    """
    is_profile = bool(side)
    if not is_profile and not has_occlusion:
        return {}

    scale = float(diag) if diag and diag > 0 else 1.0
    yaw = 0.0 if yaw_estimate is None else float(yaw_estimate)
    roll = 0.0 if roll_estimate is None else float(roll_estimate)

    point_by_name: dict[str, np.ndarray] = {}
    for candidate in candidates:
        points = _as_points(getattr(candidate, "landmarks", None))
        if points is not None:
            point_by_name[str(candidate.name)] = points

    # Robust consensus from geometry-valid candidates, falling back to all.
    valid_names = [
        name
        for name in point_by_name
        if name in metrics and not getattr(metrics[name], "geometry_veto_reasons", ())
    ]
    consensus_names = valid_names or list(point_by_name)
    consensus = (
        _robust_consensus(np.stack([point_by_name[name] for name in consensus_names], axis=0))
        if consensus_names
        else None
    )

    face_level = {
        "profile_yaw_abs": abs(yaw),
        "profile_yaw_signed": yaw,
        "profile_roll_abs": abs(roll),
        "profile_is_left": 1.0 if side == "left" else 0.0,
        "profile_is_right": 1.0 if side == "right" else 0.0,
        "profile_is_large_yaw": 1.0 if abs(yaw) >= LARGE_YAW_DEGREES else 0.0,
        "profile_is_rolled": 1.0 if abs(roll) >= ROLLED_DEGREES else 0.0,
        "profile_has_occlusion": 1.0 if has_occlusion else 0.0,
        "profile_has_single_eye_visible": 1.0 if has_single_eye_visible else 0.0,
        "profile_has_mouth_or_jaw_occluded": 1.0 if has_mouth_or_jaw_occluded else 0.0,
    }

    vis_idx = visible_side_indices(side)
    occ_idx = occluded_side_indices(side)
    eye_idx, brow_idx = _visible_eye_brow(side)

    # Visible-side candidate spread is a face-level robustness signal.
    spread = 0.0
    if consensus is not None and vis_idx and len(consensus_names) > 1:
        spread = float(
            np.mean(
                [
                    _region_distance(point_by_name[name], consensus, vis_idx)
                    for name in consensus_names
                ]
            )
            / scale
        )

    payload: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        name = str(candidate.name)
        features = dict(face_level)
        features["visible_side_candidate_spread"] = spread
        points = point_by_name.get(name)
        if points is not None and consensus is not None:
            features["visible_side_consensus_distance"] = (
                _region_distance(points, consensus, vis_idx) / scale
            )
            features["visible_eye_consensus_distance"] = (
                _region_distance(points, consensus, eye_idx) / scale
            )
            features["visible_brow_consensus_distance"] = (
                _region_distance(points, consensus, brow_idx) / scale
            )
            features["visible_mouth_corner_consensus_distance"] = (
                _region_distance(points, consensus, MOUTH_CORNERS_68) / scale
            )
            features["nose_bridge_consistency"] = (
                _region_distance(points, consensus, NOSE_BRIDGE_68) / scale
            )
            features["occluded_side_spread"] = _region_distance(points, consensus, occ_idx) / scale
            features["occluded_side_outlier_rate"] = _occluded_outlier_rate(
                points, consensus, occ_idx, scale
            )
            features["mouth_corner_asymmetry"] = _mouth_corner_asymmetry(points, scale)
            features["candidate_profile_validity_score"] = (
                features["visible_side_consensus_distance"]
                + 0.5 * features["occluded_side_spread"]
            )
        payload[name] = features
    return payload


def _occluded_outlier_rate(
    points: np.ndarray, consensus: np.ndarray, occ_idx: T.Sequence[int], scale: float
) -> float:
    if not occ_idx:
        return 0.0
    idx = np.asarray(occ_idx, dtype="int64")
    distances = np.linalg.norm(points[idx] - consensus[idx], axis=1) / scale
    return float(np.mean(distances > OUTLIER_FRACTION_OF_DIAG))


def _mouth_corner_asymmetry(points: np.ndarray, scale: float) -> float:
    nose = points[NOSE_TIP_68]
    left_corner = points[MOUTH_CORNERS_68[1]]
    right_corner = points[MOUTH_CORNERS_68[0]]
    left_dist = float(np.linalg.norm(left_corner - nose))
    right_dist = float(np.linalg.norm(right_corner - nose))
    return abs(left_dist - right_dist) / scale


__all__ = [
    "LEFT_SIDE_68",
    "MOUTH_68",
    "NOSE_68",
    "PROFILE_FEATURE_NAMES",
    "RIGHT_SIDE_68",
    "occluded_side_indices",
    "profile_candidate_features",
    "profile_side_from_context",
    "visible_side_indices",
]
