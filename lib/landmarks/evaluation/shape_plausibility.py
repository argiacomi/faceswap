#!/usr/bin/env python3
"""68-point landmark topology plausibility checks."""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShapePlausibility:
    """Face-shape quality diagnostics for one 68-point landmark candidate."""

    score: float
    severe: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


_REGION_POLYLINES: tuple[tuple[int, ...], ...] = (
    tuple(range(0, 17)),
    tuple(range(17, 22)),
    tuple(range(22, 27)),
    tuple(range(27, 31)),
    tuple(range(31, 36)),
    tuple(range(36, 42)),
    tuple(range(42, 48)),
    tuple(range(48, 60)),
    tuple(range(60, 68)),
)
_CLOSED_REGIONS: tuple[tuple[int, ...], ...] = (
    tuple(range(36, 42)),
    tuple(range(42, 48)),
    tuple(range(48, 60)),
    tuple(range(60, 68)),
)
_INTERSECTION_REGIONS: tuple[tuple[int, ...], ...] = (
    tuple(range(0, 17)),
    tuple(range(17, 22)),
    tuple(range(22, 27)),
    tuple(range(27, 31)),
    tuple(range(31, 36)),
)


def _canonical_68() -> np.ndarray:
    points = np.zeros((68, 2), dtype="float64")
    points[0:17, 0] = np.linspace(0.05, 0.91, 17)
    points[0:17, 1] = 0.62 + 0.39 * np.sin(np.linspace(0.0, math.pi, 17))
    points[17:22, 0] = np.linspace(0.18, 0.54, 5)
    points[17:22, 1] = np.array([0.27, 0.22, 0.22, 0.25, 0.30])
    points[22:27, 0] = np.linspace(0.62, 0.94, 5)
    points[22:27, 1] = np.array([0.29, 0.26, 0.27, 0.32, 0.39])
    points[27:31, 0] = [0.53, 0.52, 0.51, 0.50]
    points[27:31, 1] = [0.35, 0.44, 0.52, 0.61]
    points[31:36, 0] = np.linspace(0.41, 0.61, 5)
    points[31:36, 1] = [0.66, 0.68, 0.69, 0.68, 0.66]
    points[36:42] = [(0.25, 0.40), (0.31, 0.36), (0.38, 0.36), (0.44, 0.41), (0.38, 0.44), (0.31, 0.44)]
    points[42:48] = [(0.64, 0.42), (0.71, 0.38), (0.78, 0.38), (0.84, 0.43), (0.78, 0.46), (0.71, 0.46)]
    points[48:60] = [(0.32, 0.78), (0.40, 0.73), (0.48, 0.72), (0.54, 0.74), (0.60, 0.72), (0.69, 0.74), (0.76, 0.79), (0.69, 0.85), (0.60, 0.88), (0.52, 0.89), (0.44, 0.88), (0.38, 0.84)]
    points[60:68] = [(0.38, 0.79), (0.48, 0.77), (0.54, 0.78), (0.60, 0.77), (0.70, 0.80), (0.60, 0.81), (0.53, 0.82), (0.47, 0.81)]
    return points


_CANONICAL_68 = _canonical_68()


def _polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-9
    if (
        np.linalg.norm(a - c) <= eps
        or np.linalg.norm(a - d) <= eps
        or np.linalg.norm(b - c) <= eps
        or np.linalg.norm(b - d) <= eps
    ):
        return False
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return bool(o1 * o2 < -eps and o3 * o4 < -eps)


def _polyline_intersections(points: np.ndarray, indices: tuple[int, ...], *, closed: bool) -> int:
    if len(indices) < 4:
        return 0
    region = points[list(indices), :2]
    if closed and _polygon_area(region) <= 1e-6:
        return 0
    edges = [(indices[idx], indices[idx + 1]) for idx in range(len(indices) - 1)]
    if closed:
        edges.append((indices[-1], indices[0]))
    intersections = 0
    for left_index, (a_idx, b_idx) in enumerate(edges):
        for right_index, (c_idx, d_idx) in enumerate(edges[left_index + 1 :], left_index + 1):
            if {a_idx, b_idx} & {c_idx, d_idx}:
                continue
            if closed and {left_index, right_index} == {0, len(edges) - 1}:
                continue
            if _segments_cross(
                points[a_idx, :2], points[b_idx, :2], points[c_idx, :2], points[d_idx, :2]
            ):
                intersections += 1
    return intersections


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    if polygon.shape[0] < 3 or _polygon_area(polygon) <= 1e-6:
        return True
    x, y = float(point[0]), float(point[1])
    inside = False
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if (yi > y) != (yj > y):
            denom = yj - yi
            if abs(denom) <= 1e-12:
                j = i
                continue
            x_at_y = (xj - xi) * (y - yi) / denom + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def _similarity_fit_error(points: np.ndarray) -> float:
    source = points[:, :2].astype("float64", copy=False)
    source_centered = source - np.mean(source, axis=0)
    target_centered = _CANONICAL_68 - np.mean(_CANONICAL_68, axis=0)
    source_norm = float(np.linalg.norm(source_centered))
    target_norm = float(np.linalg.norm(target_centered))
    if source_norm <= 1e-9 or target_norm <= 1e-9:
        return float("inf")
    source_unit = source_centered / source_norm
    target_unit = target_centered / target_norm
    try:
        u_matrix, _singular, vt_matrix = np.linalg.svd(source_unit.T @ target_unit)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.mean(np.linalg.norm(source_unit @ (u_matrix @ vt_matrix) - target_unit, axis=1)))


def _derolled(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    left_eye = np.mean(points[36:42, :2], axis=0)
    right_eye = np.mean(points[42:48, :2], axis=0)
    eye_vector = right_eye - left_eye
    iod = float(np.linalg.norm(eye_vector))
    if iod <= 1e-6:
        return None
    eye_mid = (left_eye + right_eye) / 2.0
    angle = math.atan2(float(eye_vector[1]), float(eye_vector[0]))
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype="float64")
    return (points[:, :2] - eye_mid) @ rotation.T, iod


def evaluate_shape_plausibility(landmarks: T.Any) -> ShapePlausibility:
    points = np.asarray(landmarks, dtype="float64")
    if points.ndim != 2 or points.shape[0] < 68 or points.shape[1] < 2:
        return ShapePlausibility(1.0, True, ("invalid_landmark_shape",), {"topology_violation_count": 1.0})
    points = points[:68, :2]
    if not np.all(np.isfinite(points)):
        return ShapePlausibility(1.0, True, ("non_finite_landmarks",), {"topology_violation_count": 1.0})

    width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    bbox_diag = math.hypot(width, height)
    derolled = _derolled(points)
    if bbox_diag <= 1e-6 or derolled is None:
        return ShapePlausibility(1.0, True, ("degenerate_face_scale",), {"topology_violation_count": 1.0})

    derolled_points, interocular = derolled
    scale = max(interocular, bbox_diag * 0.35, 1e-6)
    edge_lengths: list[float] = []
    long_edge_count = 0
    for indices in _REGION_POLYLINES:
        region = points[list(indices), :2]
        lengths = np.linalg.norm(np.diff(region, axis=0), axis=1)
        edge_lengths.extend(float(length) for length in lengths)
        long_edge_count += int(np.count_nonzero(lengths / scale > 0.95))
    max_edge_length_ratio = max(edge_lengths) / scale if edge_lengths else 0.0

    intersection_count = sum(
        _polyline_intersections(points, indices, closed=False) for indices in _INTERSECTION_REGIONS
    ) + sum(_polyline_intersections(points, indices, closed=True) for indices in _CLOSED_REGIONS)
    inner_outside_fraction = float(
        np.mean([not _point_in_polygon(point, points[48:60, :2]) for point in points[60:68, :2]])
    )

    left_eye = np.mean(derolled_points[36:42], axis=0)
    right_eye = np.mean(derolled_points[42:48], axis=0)
    mouth = np.mean(derolled_points[48:68], axis=0)
    nose_tip = np.mean(derolled_points[31:36], axis=0)
    eye_mid = (left_eye + right_eye) / 2.0
    eye_span = max(abs(float(right_eye[0] - left_eye[0])), 1e-6)
    local_order_violations = int(not bool(mouth[1] > eye_mid[1] + 0.05 * interocular))
    local_order_violations += int(
        not bool(eye_mid[1] - 0.20 * interocular <= nose_tip[1] <= mouth[1] + 0.35 * interocular)
    )
    nose_lateral_offset_ratio = abs(float(nose_tip[0] - eye_mid[0])) / eye_span
    local_order_violations += int(nose_lateral_offset_ratio > 1.15)
    local_order_violations += int(not bool(left_eye[0] < right_eye[0]))

    mean_shape_fit_error = _similarity_fit_error(points)
    topology_violation_count = (
        long_edge_count + intersection_count + int(inner_outside_fraction > 0.25) + local_order_violations
    )

    reasons: list[str] = []
    if max_edge_length_ratio > 1.35 or long_edge_count >= 3:
        reasons.append("edge_length_extreme")
    elif max_edge_length_ratio > 0.95:
        reasons.append("edge_length_high")
    if intersection_count:
        reasons.append("self_intersection")
    if inner_outside_fraction > 0.25:
        reasons.append("inner_mouth_outside_outer_mouth")
    if local_order_violations >= 2:
        reasons.append("region_order_inconsistent")
    elif local_order_violations == 1:
        reasons.append("region_order_warning")
    if math.isfinite(mean_shape_fit_error) and mean_shape_fit_error > 0.13:
        reasons.append("mean_shape_fit_high")

    score = (
        min(max_edge_length_ratio / 1.35, 2.0) * 0.35
        + min(intersection_count / 2.0, 2.0) * 0.25
        + min(inner_outside_fraction / 0.5, 2.0) * 0.15
        + min(local_order_violations / 3.0, 2.0) * 0.10
        + min((mean_shape_fit_error if math.isfinite(mean_shape_fit_error) else 1.0) / 0.13, 2.0) * 0.15
    )
    severe = bool(
        max_edge_length_ratio > 1.35
        or long_edge_count >= 3
        or intersection_count >= 2
        or inner_outside_fraction > 0.50
        or (local_order_violations >= 2 and max_edge_length_ratio > 0.95)
    )
    return ShapePlausibility(
        float(score),
        severe,
        tuple(reasons),
        {
            "max_edge_length_ratio": float(max_edge_length_ratio),
            "mean_shape_fit_error": float(mean_shape_fit_error),
            "topology_violation_count": float(topology_violation_count),
            "self_intersection_count": float(intersection_count),
            "inner_mouth_outside_fraction": float(inner_outside_fraction),
            "local_order_violation_count": float(local_order_violations),
            "long_edge_count": float(long_edge_count),
            "nose_lateral_offset_ratio": float(nose_lateral_offset_ratio),
        },
    )


__all__ = ["ShapePlausibility", "evaluate_shape_plausibility"]
