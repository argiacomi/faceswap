#!/usr/bin/env python3
"""Per-sample crop / ROI diagnostics for landmark adapters (#81).

These signals answer the question: is the *crop* — not the model — the reason
this sample produced bad extraction geometry? Profile and large-yaw faces are
the canonical case: if the detector bbox cuts off the silhouette or the model
crop misses the visible face, no fusion strategy can recover.

The diagnostics are intentionally cache-only: they consume the manifest's
face bbox, the GT landmarks, and the predicted-derived ``AlignedFace`` ROI.
They never invoke adapters or re-render images.
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

import numpy as np

from lib.landmarks.evaluation.geometry_signals import (
    AlignmentSummary,
    _convex_hull_points,
    polygon_iou,
)

#: Default fraction of GT landmarks that must lie inside the predicted aligned
#: crop polygon before we consider the crop to cover the visible face. Below
#: this threshold the sample is flagged ``aligned_crop_misses_visible_face``.
DEFAULT_LANDMARK_COVERAGE_FLOOR: float = 0.85


@dataclass(frozen=True)
class RoiDiagnostics:
    """Crop / ROI signals for one (predicted, truth) pair."""

    bbox_source: str
    bbox_width: float
    bbox_height: float
    bbox_aspect_ratio: float  # width / height
    bbox_diagonal: float
    aligned_roi_width: float
    aligned_roi_height: float
    aligned_roi_aspect_ratio: float
    aligned_crop_visible_hull_iou: float  # IoU between predicted ROI and GT-hull
    landmarks_inside_aligned_crop_fraction: float
    landmarks_outside_detector_bbox_fraction: float
    aligned_crop_misses_visible_face: bool

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "bbox_source": self.bbox_source,
            "bbox_width": float(self.bbox_width),
            "bbox_height": float(self.bbox_height),
            "bbox_aspect_ratio": float(self.bbox_aspect_ratio),
            "bbox_diagonal": float(self.bbox_diagonal),
            "aligned_roi_width": float(self.aligned_roi_width),
            "aligned_roi_height": float(self.aligned_roi_height),
            "aligned_roi_aspect_ratio": float(self.aligned_roi_aspect_ratio),
            "aligned_crop_visible_hull_iou": float(self.aligned_crop_visible_hull_iou),
            "landmarks_inside_aligned_crop_fraction": float(
                self.landmarks_inside_aligned_crop_fraction
            ),
            "landmarks_outside_detector_bbox_fraction": float(
                self.landmarks_outside_detector_bbox_fraction
            ),
            "aligned_crop_misses_visible_face": bool(self.aligned_crop_misses_visible_face),
        }


def _bbox_dimensions(bbox: T.Sequence[float]) -> tuple[float, float, float]:
    """Return ``(width, height, diagonal)`` for a ``(left, top, right, bottom)`` bbox."""
    left, top, right, bottom = (float(value) for value in bbox)
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    diagonal = math.hypot(width, height)
    return width, height, diagonal


def _polygon_dimensions(corners: np.ndarray) -> tuple[float, float]:
    """Return axis-aligned (width, height) of a polygon's bbox."""
    if corners.size == 0:
        return 0.0, 0.0
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    return float(maxs[0] - mins[0]), float(maxs[1] - mins[1])


def _ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    if polygon.shape[0] < 3:
        return polygon
    x = polygon[:, 0]
    y = polygon[:, 1]
    signed = float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return polygon[::-1] if signed < 0 else polygon


def landmarks_inside_polygon(landmarks: np.ndarray, polygon: np.ndarray) -> int:
    """Return the count of landmarks inside (or on the edge of) ``polygon``.

    Vectorized cross-product sign test for a convex polygon: a point lies
    inside iff every directed edge keeps it on its left. AlignedFace ROIs are
    always convex quads, so this is exact for our use; non-convex inputs may
    over-count. Compared to a Python ray-cast loop this is ~10x faster on
    ``(68, 2)`` landmark sets.
    """
    if polygon.shape[0] < 3:
        return 0
    poly = _ensure_ccw(polygon.astype("float64"))
    edges_start = poly  # (E, 2)
    edges_end = np.roll(poly, -1, axis=0)  # (E, 2)
    edge_vec = edges_end - edges_start  # (E, 2)
    # (E, N, 2): vector from each edge start to each landmark.
    deltas = landmarks.astype("float64")[None, :, :] - edges_start[:, None, :]
    cross = edge_vec[:, None, 0] * deltas[..., 1] - edge_vec[:, None, 1] * deltas[..., 0]
    inside = (cross >= 0).all(axis=0)  # (N,)
    return int(inside.sum())


def landmarks_outside_bbox(landmarks: np.ndarray, bbox: T.Sequence[float]) -> int:
    """Return the count of landmarks outside the (raw) bbox."""
    left, top, right, bottom = (float(value) for value in bbox)
    inside_x = (landmarks[:, 0] >= left) & (landmarks[:, 0] <= right)
    inside_y = (landmarks[:, 1] >= top) & (landmarks[:, 1] <= bottom)
    return int(np.sum(~(inside_x & inside_y)))


def evaluate_roi_diagnostics(
    *,
    predicted_summary: AlignmentSummary,
    truth_landmarks: np.ndarray,
    bbox: T.Sequence[float],
    bbox_source: str = "manifest",
    coverage_floor: float = DEFAULT_LANDMARK_COVERAGE_FLOOR,
) -> RoiDiagnostics:
    """Compute ROI diagnostics for one prediction vs the GT landmark set.

    ``predicted_summary`` is the ``AlignmentSummary`` already computed for the
    candidate; we reuse its ``roi`` directly to avoid running ``AlignedFace``
    twice.
    """
    width, height, diagonal = _bbox_dimensions(bbox)
    if width <= 0 or height <= 0:
        raise ValueError(f"bbox must have positive width and height, got {bbox!r}")
    aspect_ratio = width / height if height > 0 else 0.0

    roi_corners = np.asarray(predicted_summary.roi, dtype="float64")
    roi_width, roi_height = _polygon_dimensions(roi_corners)
    roi_aspect = roi_width / roi_height if roi_height > 0 else 0.0

    truth_hull = _convex_hull_points(truth_landmarks)
    hull_iou = 0.0 if truth_hull is None else polygon_iou(roi_corners, truth_hull)

    inside_count = landmarks_inside_polygon(truth_landmarks, roi_corners)
    coverage = float(inside_count / truth_landmarks.shape[0]) if truth_landmarks.size else 0.0
    outside_bbox_count = landmarks_outside_bbox(truth_landmarks, bbox)
    outside_fraction = (
        float(outside_bbox_count / truth_landmarks.shape[0]) if truth_landmarks.size else 0.0
    )
    misses_face = coverage < coverage_floor

    return RoiDiagnostics(
        bbox_source=str(bbox_source),
        bbox_width=width,
        bbox_height=height,
        bbox_aspect_ratio=aspect_ratio,
        bbox_diagonal=diagonal,
        aligned_roi_width=roi_width,
        aligned_roi_height=roi_height,
        aligned_roi_aspect_ratio=roi_aspect,
        aligned_crop_visible_hull_iou=hull_iou,
        landmarks_inside_aligned_crop_fraction=coverage,
        landmarks_outside_detector_bbox_fraction=outside_fraction,
        aligned_crop_misses_visible_face=misses_face,
    )


__all__ = [
    "DEFAULT_LANDMARK_COVERAGE_FLOOR",
    "RoiDiagnostics",
    "evaluate_roi_diagnostics",
    "landmarks_inside_polygon",
    "landmarks_outside_bbox",
]
