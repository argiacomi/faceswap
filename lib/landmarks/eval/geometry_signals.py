#!/usr/bin/env python3
"""Geometry signals for extract-alignment quality (alignment-geometry roadmap).

These primitives measure the *outputs* that Faceswap's extract path actually
computes from landmarks rather than landmark-point accuracy alone. They are
grounded in :class:`lib.align.aligned_face.AlignedFace`: every prediction is
fed through the same Umeyama-based alignment Faceswap uses at runtime, so the
measurements correspond to the geometry that downstream crops, masks, pose,
and swap placement consume.

For each (predicted, truth) pair we evaluate:

* alignment-matrix deltas (scale, rotation degrees, translation pixels)
* original ROI IoU and center distance — the crop quality signal
* normalized average-distance delta — Faceswap's own mean-face fit metric
* relative eye-mouth position — Faceswap's flip / catastrophic-anatomy check
* pose deltas (pitch / yaw / roll)
* convex hull IoU — visible-face hull quality for mask construction
* catastrophic failure flags: cloud collapse, anatomical flip, points outside
  the (inflated) face bbox

Visibility flags from the manifest may be supplied so the hull check only
considers landmarks that the dataset says are visible; otherwise the full 68
points are used.
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from lib.align.aligned_face import AlignedFace, _umeyama
from lib.align.constants import MEAN_FACE, LandmarkType

DEFAULT_ALIGNED_SIZE: int = 512
"""Pixel size used when instantiating :class:`AlignedFace` for ROI extraction."""

DEFAULT_BBOX_INFLATION: float = 1.5
"""Multiplier applied to the face bbox before counting out-of-bbox landmarks."""

DEFAULT_COLLAPSE_RATIO: float = 0.3
"""Threshold below which the predicted landmark cloud is considered collapsed."""

LEFT_EYE_INDICES: tuple[int, ...] = (36, 37, 38, 39, 40, 41)
RIGHT_EYE_INDICES: tuple[int, ...] = (42, 43, 44, 45, 46, 47)
MOUTH_INDICES: tuple[int, ...] = tuple(range(48, 68))


@dataclass(frozen=True)
class AlignmentSummary:
    """The slice of :class:`AlignedFace` outputs the geometry metrics rely on."""

    matrix: np.ndarray  # (2, 3) legacy alignment matrix
    roi: np.ndarray  # (4, 2) original-frame ROI corners
    aligned_landmarks: np.ndarray  # (68, 2) landmarks in aligned-face space
    normalized_landmarks: np.ndarray  # (68, 2) Umeyama-normalized landmarks
    average_distance: float  # mean-face fit (Faceswap metric)
    relative_eye_mouth_position: float  # Faceswap's eyes-above-mouth check
    pitch: float
    yaw: float
    roll: float
    scale: float  # decomposed from matrix
    rotation_degrees: float  # decomposed from matrix
    translation: tuple[float, float]


def alignment_summary(
    landmarks: np.ndarray,
    *,
    size: int = DEFAULT_ALIGNED_SIZE,
    coverage_ratio: float = 1.0,
) -> AlignmentSummary:
    """Run Faceswap's alignment over ``landmarks`` and return the geometry summary.

    Uses :class:`AlignedFace` with ``image=None`` so no face image is extracted;
    we only need the matrix, ROI, pose, aligned/normalized landmarks, and the
    canonical scalar diagnostics. The aligned landmarks let downstream metrics
    compare prediction vs GT in aligned-face space, which is exactly the
    coordinate frame Faceswap's crop/mask/swap-placement consumes.

    ``coverage_ratio`` is forwarded to :class:`AlignedFace` so callers that
    tune the runtime crop scale (``plugins/extract/align/ensemble.py`` uses
    the same value to inflate the detection bbox) can score predictions in
    the same coordinate frame the extract pipeline will actually deploy.
    """
    if landmarks.shape != (68, 2):
        raise ValueError(f"expected (68, 2) landmarks, got {landmarks.shape!r}")
    aligned = AlignedFace(
        landmarks.astype("float32"),
        image=None,
        centering="face",
        size=size,
        coverage_ratio=coverage_ratio,
    )
    matrix = aligned.matrix.astype("float64")
    roi = aligned.original_roi.astype("float64")
    pose = aligned.pose
    scale, rotation_degrees, translation = _decompose_similarity_matrix(matrix)
    return AlignmentSummary(
        matrix=matrix,
        roi=roi,
        aligned_landmarks=aligned.landmarks.astype("float64"),
        normalized_landmarks=aligned.normalized_landmarks.astype("float64"),
        average_distance=float(aligned.average_distance),
        relative_eye_mouth_position=float(aligned.relative_eye_mouth_position),
        pitch=float(pose.pitch),
        yaw=float(pose.yaw),
        roll=float(pose.roll),
        scale=float(scale),
        rotation_degrees=float(rotation_degrees),
        translation=translation,
    )


def _decompose_similarity_matrix(
    matrix: np.ndarray,
) -> tuple[float, float, tuple[float, float]]:
    """Decompose a 2×3 similarity matrix into (scale, rotation_degrees, translation)."""
    if matrix.shape != (2, 3):
        raise ValueError(f"expected (2, 3) matrix, got {matrix.shape!r}")
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    scale = math.sqrt(a * a + c * c)
    rotation = math.degrees(math.atan2(c, a))
    return scale, rotation, (float(tx), float(ty))


@dataclass(frozen=True)
class MatrixDelta:
    """Difference between two alignment matrices, decomposed into similarity components."""

    scale_delta: float
    rotation_degrees_delta: float
    translation_pixel_distance: float
    translation_normalized_distance: float


def alignment_matrix_delta(
    predicted: AlignmentSummary,
    truth: AlignmentSummary,
    *,
    normalizer: float,
) -> MatrixDelta:
    """Return the matrix-space delta between predicted and truth alignments.

    ``normalizer`` is typically the bbox diagonal so translation deltas are
    scale-invariant.
    """
    if normalizer <= 0:
        raise ValueError(f"normalizer must be > 0, got {normalizer!r}")
    scale_delta = abs(predicted.scale - truth.scale)
    rotation_delta = abs(_wrap_angle_degrees(predicted.rotation_degrees - truth.rotation_degrees))
    translation_vec = (
        predicted.translation[0] - truth.translation[0],
        predicted.translation[1] - truth.translation[1],
    )
    translation_pixels = math.hypot(translation_vec[0], translation_vec[1])
    return MatrixDelta(
        scale_delta=scale_delta,
        rotation_degrees_delta=rotation_delta,
        translation_pixel_distance=translation_pixels,
        translation_normalized_distance=translation_pixels / normalizer,
    )


def _wrap_angle_degrees(angle: float) -> float:
    """Wrap an angle to ``(-180, 180]`` so deltas stay bounded."""
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    if wrapped == -180.0:
        wrapped = 180.0
    return wrapped


@dataclass(frozen=True)
class RoiDelta:
    """Crop-quality delta between two ``AlignedFace.original_roi`` boxes."""

    iou: float
    center_pixel_distance: float
    center_normalized_distance: float


def roi_delta(
    predicted: AlignmentSummary,
    truth: AlignmentSummary,
    *,
    normalizer: float,
) -> RoiDelta:
    """Return crop-IoU and center distance between two ROIs in the source frame."""
    if normalizer <= 0:
        raise ValueError(f"normalizer must be > 0, got {normalizer!r}")
    iou = polygon_iou(predicted.roi, truth.roi)
    pred_center = predicted.roi.mean(axis=0)
    truth_center = truth.roi.mean(axis=0)
    delta = pred_center - truth_center
    distance = float(np.linalg.norm(delta))
    return RoiDelta(
        iou=iou,
        center_pixel_distance=distance,
        center_normalized_distance=distance / normalizer,
    )


@dataclass(frozen=True)
class PoseDelta:
    """Absolute pose deltas in degrees."""

    pitch_delta_degrees: float
    yaw_delta_degrees: float
    roll_delta_degrees: float


def pose_delta(predicted: AlignmentSummary, truth: AlignmentSummary) -> PoseDelta:
    """Return absolute roll / yaw / pitch deltas between two alignments."""
    return PoseDelta(
        pitch_delta_degrees=abs(_wrap_angle_degrees(predicted.pitch - truth.pitch)),
        yaw_delta_degrees=abs(_wrap_angle_degrees(predicted.yaw - truth.yaw)),
        roll_delta_degrees=abs(_wrap_angle_degrees(predicted.roll - truth.roll)),
    )


def average_distance_delta(predicted: AlignmentSummary, truth: AlignmentSummary) -> float:
    """Return ``predicted.average_distance - truth.average_distance``.

    Lower (or negative) means the predicted landmarks fit the canonical mean
    face *better* than the truth landmarks — uncommon, but useful as a
    signed signal so regressions are visible.
    """
    return float(predicted.average_distance - truth.average_distance)


def umeyama_alignment_matrix(landmarks: np.ndarray) -> np.ndarray:
    """Return the legacy 2×3 alignment matrix Faceswap uses at extract time.

    Convenience wrapper around :func:`lib.align.aligned_face._umeyama` on the
    core 51 landmarks (indices 17–67), matching ``AlignedFace._get_default_matrix``.
    """
    if landmarks.shape != (68, 2):
        raise ValueError(f"expected (68, 2) landmarks, got {landmarks.shape!r}")
    return _umeyama(landmarks[17:].astype("float64"), MEAN_FACE[LandmarkType.LM_2D_51], True)[0:2]


# ---------------------------------------------------------------------------
# Hull + catastrophic flags
# ---------------------------------------------------------------------------


def _convex_hull_points(points: np.ndarray) -> np.ndarray | None:
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return None
    return points[hull.vertices].astype("float64")


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    if polygon.shape[0] < 3:
        return polygon
    x = polygon[:, 0]
    y = polygon[:, 1]
    signed = float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return polygon[::-1] if signed < 0 else polygon


def _sutherland_hodgman(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip ``subject`` against the convex polygon ``clip`` (both CCW)."""
    output: list[T.Sequence[float]] = subject.tolist()
    for i in range(clip.shape[0]):
        if not output:
            break
        input_list = output
        output = []
        edge_start = clip[i - 1]
        edge_end = clip[i]
        edge = edge_end - edge_start

        def inside(point: T.Sequence[float], _es=edge_start, _e=edge) -> bool:
            return (_e[0] * (point[1] - _es[1]) - _e[1] * (point[0] - _es[0])) >= 0

        def intersect(
            p1: T.Sequence[float],
            p2: T.Sequence[float],
            _es=edge_start,
            _ee=edge_end,
        ) -> tuple[float, float]:
            r = (p2[0] - p1[0], p2[1] - p1[1])
            s = (_ee[0] - _es[0], _ee[1] - _es[1])
            denom = r[0] * s[1] - r[1] * s[0]
            if denom == 0:
                return (p2[0], p2[1])
            t = ((_es[0] - p1[0]) * s[1] - (_es[1] - p1[1]) * s[0]) / denom
            return (p1[0] + t * r[0], p1[1] + t * r[1])

        for j, point in enumerate(input_list):
            previous = input_list[j - 1]
            if inside(point):
                if not inside(previous):
                    output.append(intersect(previous, point))
                output.append(tuple(point))
            elif inside(previous):
                output.append(intersect(previous, point))
    if not output:
        return np.zeros((0, 2), dtype="float64")
    return np.asarray(output, dtype="float64")


def polygon_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    """IoU between two convex polygons given as ordered vertices."""
    pred = _ensure_ccw(np.asarray(predicted, dtype="float64"))
    truth_poly = _ensure_ccw(np.asarray(truth, dtype="float64"))
    intersection = _sutherland_hodgman(pred, truth_poly)
    inter_area = _polygon_area(intersection)
    union = _polygon_area(pred) + _polygon_area(truth_poly) - inter_area
    if union <= 0:
        return 0.0
    return float(max(0.0, min(1.0, inter_area / union)))


def visible_hull_iou(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    visibility: T.Sequence[bool] | None = None,
) -> float:
    """Return the IoU between the convex hulls of the (visible) landmark points.

    When ``visibility`` is provided, only flagged-visible landmarks contribute
    to both hulls. The truth hull is computed with the same visibility mask so
    we are comparing the model's visible-face geometry against the dataset's
    visible-face geometry.
    """
    pred_points = _select_visible(predicted, visibility)
    truth_points = _select_visible(truth, visibility)
    pred_hull = _convex_hull_points(pred_points)
    truth_hull = _convex_hull_points(truth_points)
    if pred_hull is None or truth_hull is None:
        return 0.0
    return polygon_iou(pred_hull, truth_hull)


def _select_visible(points: np.ndarray, visibility: T.Sequence[bool] | None) -> np.ndarray:
    if visibility is None:
        return points
    if len(visibility) != points.shape[0]:
        raise ValueError(
            f"visibility length {len(visibility)} must match landmark count {points.shape[0]}"
        )
    mask = np.array(list(visibility), dtype=bool)
    return points[mask]


@dataclass(frozen=True)
class CatastrophicFlags:
    """Boolean failure flags for one prediction."""

    cloud_collapse: bool
    eye_mouth_flip: bool
    points_outside_bbox: bool

    @property
    def any(self) -> bool:
        return self.cloud_collapse or self.eye_mouth_flip or self.points_outside_bbox


def cloud_collapse(
    predicted: np.ndarray,
    *,
    truth_extent: float,
    ratio: float = DEFAULT_COLLAPSE_RATIO,
) -> bool:
    """Return True when the predicted landmark cloud is much smaller than the truth cloud."""
    pred_diag = _bbox_diagonal_of(predicted)
    return pred_diag < ratio * truth_extent


def eye_mouth_flip(summary: AlignmentSummary, *, floor: float = 0.0) -> bool:
    """Return True when the alignment fails Faceswap's eyes-above-mouth check.

    Faceswap defines a flipped face as ``relative_eye_mouth_position <= 0``
    in normalized coordinates. ``floor`` lets callers add a small margin
    (positive values stricten the check).
    """
    return summary.relative_eye_mouth_position <= floor


def points_outside_bbox(
    predicted: np.ndarray,
    *,
    bbox: T.Sequence[float],
    inflation: float = DEFAULT_BBOX_INFLATION,
) -> int:
    """Return the count of predicted landmarks outside an inflated face bbox."""
    left, top, right, bottom = (float(value) for value in bbox)
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    half_w = 0.5 * (right - left) * inflation
    half_h = 0.5 * (bottom - top) * inflation
    box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    inside_x = (predicted[:, 0] >= box[0]) & (predicted[:, 0] <= box[2])
    inside_y = (predicted[:, 1] >= box[1]) & (predicted[:, 1] <= box[3])
    return int(np.sum(~(inside_x & inside_y)))


def evaluate_catastrophic_flags(
    predicted: np.ndarray,
    truth: np.ndarray,
    summary: AlignmentSummary,
    *,
    bbox: T.Sequence[float] | None,
    collapse_ratio: float = DEFAULT_COLLAPSE_RATIO,
    bbox_inflation: float = DEFAULT_BBOX_INFLATION,
) -> CatastrophicFlags:
    """Return the full set of catastrophic-failure flags for one prediction."""
    truth_extent = _bbox_diagonal_of(truth)
    return CatastrophicFlags(
        cloud_collapse=cloud_collapse(predicted, truth_extent=truth_extent, ratio=collapse_ratio),
        eye_mouth_flip=eye_mouth_flip(summary),
        points_outside_bbox=(
            False
            if bbox is None
            else points_outside_bbox(predicted, bbox=bbox, inflation=bbox_inflation) > 0
        ),
    )


def _bbox_diagonal_of(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    width = float(maxs[0] - mins[0])
    height = float(maxs[1] - mins[1])
    return math.hypot(width, height)


__all__ = [
    "AlignmentSummary",
    "CatastrophicFlags",
    "DEFAULT_ALIGNED_SIZE",
    "DEFAULT_BBOX_INFLATION",
    "DEFAULT_COLLAPSE_RATIO",
    "LEFT_EYE_INDICES",
    "MOUTH_INDICES",
    "MatrixDelta",
    "PoseDelta",
    "RIGHT_EYE_INDICES",
    "RoiDelta",
    "alignment_matrix_delta",
    "alignment_summary",
    "average_distance_delta",
    "cloud_collapse",
    "evaluate_catastrophic_flags",
    "eye_mouth_flip",
    "points_outside_bbox",
    "polygon_iou",
    "pose_delta",
    "roi_delta",
    "umeyama_alignment_matrix",
    "visible_hull_iou",
]
