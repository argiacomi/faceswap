#!/usr/bin/env python3
"""Direct transform-impact alignment cost for learned-quality v3.

This module intentionally does not change ``alignment_geometry_v1``.  It adds a
separate, explicit v3 scoring primitive for candidate-selection labels:

    candidate_alignment_cost =
        w_center * crop_center_delta
        + w_scale * scale_delta
        + w_roll * roll_delta_degrees
        + w_fit * fit_delta
        + soft_structural_plausibility_penalty

The deltas are computed from the same Faceswap alignment vocabulary used by the
existing geometry stack: Umeyama alignment matrices, crop/ROI deltas, roll deltas,
and average-distance fit deltas.  Unlike ``alignment_summary()``, the transform
fit here can be gated by landmark visibility, so profile/occluded GT points do
not pollute the transform used as the v3 oracle target.
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

import numpy as np

from lib.align.aligned_face import batch_umeyama
from lib.align.constants import EXTRACT_RATIOS, MEAN_FACE, CenteringType, LandmarkType
from lib.landmarks.evaluation.geometry_signals import (
    DEFAULT_ALIGNED_SIZE,
    AlignmentSummary,
    alignment_matrix_delta,
    average_distance_delta,
    roi_delta,
)

_CORE_START = 17
_CORE_COUNT = 51
_DEFAULT_MIN_VISIBLE_POINTS = 8


@dataclass(frozen=True)
class TransformCostWeightsV3:
    """Weights for the direct v3 transform-cost components.

    The defaults mirror the scale of ``alignment_geometry_v1`` where possible:
    crop-center/translation dominates, scale is unit-weighted, and roll degrees
    get a small coefficient so several degrees matter without overwhelming the
    center/scale terms.  ``fit`` is a normalized mean-face fit delta.
    """

    center: float = 5.0
    scale: float = 1.0
    roll: float = 0.02
    fit: float = 1.0


@dataclass(frozen=True)
class TransformCostV3:
    """Direct transform-impact alignment cost for one candidate.

    ``hard_invalid`` marks candidates that should be removed from rankable v3
    groups by the scorer-dataset construction step.  The cost object still
    carries the flag/reasons for diagnostics, but LambdaRank should not consume
    hard-invalid rows as ordinary finite-cost items.
    """

    center_delta: float
    scale_delta: float
    roll_delta_degrees: float
    fit_delta: float
    soft_structural_penalty: float
    total_cost: float
    hard_invalid: bool
    hard_invalid_reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, T.Any]:
        """Return a JSON-serializable diagnostic payload."""
        return {
            "center_delta": float(self.center_delta),
            "scale_delta": float(self.scale_delta),
            "roll_delta_degrees": float(self.roll_delta_degrees),
            "fit_delta": float(self.fit_delta),
            "soft_structural_penalty": float(self.soft_structural_penalty),
            "total_cost": float(self.total_cost),
            "hard_invalid": bool(self.hard_invalid),
            "hard_invalid_reasons": list(self.hard_invalid_reasons),
        }


@dataclass(frozen=True)
class VisibleSubsetAlignmentSummary:
    """Alignment summary fitted from a visibility-gated landmark subset."""

    summary: AlignmentSummary
    visible_indices: tuple[int, ...]
    coverage_ratio: float
    centering: CenteringType


def _as_68_landmarks(landmarks: np.ndarray, *, name: str) -> np.ndarray:
    """Return landmarks as finite ``float64`` with shape ``(68, 2)``."""
    points = np.asarray(landmarks, dtype="float64")
    if points.shape != (68, 2):
        raise ValueError(f"{name} expected (68, 2) landmarks, got {points.shape!r}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} landmarks contain non-finite values")
    return points


def _visibility_mask(visibility: T.Sequence[bool] | None) -> np.ndarray:
    """Return a 68-long boolean visibility mask, defaulting to all visible."""
    if visibility is None:
        return np.ones(68, dtype=bool)
    mask = np.asarray(visibility, dtype=bool)
    if mask.shape != (68,):
        raise ValueError(f"visibility expected shape (68,), got {mask.shape!r}")
    return mask


def visible_core_indices(
    visibility: T.Sequence[bool] | None,
    *,
    min_visible_points: int = _DEFAULT_MIN_VISIBLE_POINTS,
) -> tuple[int, ...]:
    """Return visible 68-point indices used by the v3 transform fit.

    Faceswap's 68-point alignment matrix is fitted from the 51 core landmarks
    (indices 17..67) against ``MEAN_FACE[LM_2D_51]``.  v3 preserves that
    convention and applies visibility only within that core subset.
    """
    if min_visible_points < 3:
        raise ValueError("min_visible_points must be at least 3 for a similarity transform")
    mask = _visibility_mask(visibility)
    core_visible = np.flatnonzero(mask[_CORE_START : _CORE_START + _CORE_COUNT])
    if core_visible.size < min_visible_points:
        raise ValueError(
            "visible-subset transform fit needs at least "
            f"{min_visible_points} visible core landmarks; got {core_visible.size}"
        )
    return tuple(int(index + _CORE_START) for index in core_visible)


def _fit_matrix_from_visible_core(
    landmarks: np.ndarray,
    visible_indices: T.Sequence[int],
) -> np.ndarray:
    """Fit a 2×3 Umeyama matrix from visible core landmarks to the mean face."""
    core_offsets = np.asarray(visible_indices, dtype=np.intp) - _CORE_START
    if core_offsets.size < 3:
        raise ValueError("at least 3 visible core landmarks are required")
    source = landmarks[np.asarray(visible_indices, dtype=np.intp)]
    destination = MEAN_FACE[LandmarkType.LM_2D_51][core_offsets]
    matrix = batch_umeyama(source[np.newaxis, ...], destination, estimate_scale=True)[0][:2]
    if not np.isfinite(matrix).all():
        raise ValueError("visible-subset Umeyama fit produced a non-finite matrix")
    return T.cast(np.ndarray, matrix.astype("float64"))


def _affine_transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2×3 affine matrix to ``(N, 2)`` points."""
    homogeneous = np.concatenate(
        [points.astype("float64"), np.ones((points.shape[0], 1), dtype="float64")],
        axis=1,
    )
    return T.cast(np.ndarray, homogeneous @ matrix.T)


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of a 2×3 affine matrix as another 2×3 matrix."""
    full = np.eye(3, dtype="float64")
    full[:2, :] = matrix
    inverse = np.linalg.inv(full)
    if not np.isfinite(inverse).all():
        raise ValueError("affine matrix inversion produced non-finite values")
    return T.cast(np.ndarray, inverse[:2, :])


def _padding_from_coverage(
    *,
    size: int,
    coverage_ratio: float,
    centering: CenteringType,
) -> int:
    """Return Faceswap's padding for ``size``/``coverage_ratio``/``centering``."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    if coverage_ratio <= 0:
        raise ValueError(f"coverage_ratio must be positive, got {coverage_ratio!r}")
    return round(size * (EXTRACT_RATIOS[centering] + coverage_ratio - 1) / (2 * coverage_ratio))


def _adjusted_matrix(
    matrix: np.ndarray,
    *,
    size: int,
    coverage_ratio: float,
    centering: CenteringType,
) -> np.ndarray:
    """Return Faceswap-style crop matrix with coverage padding applied."""
    padding = _padding_from_coverage(size=size, coverage_ratio=coverage_ratio, centering=centering)
    adjusted = matrix.copy()
    adjusted *= size - 2 * padding
    adjusted[:, 2] += padding
    return adjusted


def _roi_from_adjusted_matrix(matrix: np.ndarray, *, size: int) -> np.ndarray:
    """Return original-frame ROI corners implied by an adjusted crop matrix."""
    corners = np.asarray(
        [[0.0, 0.0], [0.0, size - 1.0], [size - 1.0, size - 1.0], [size - 1.0, 0.0]],
        dtype="float64",
    )
    return _affine_transform_points(corners, _invert_affine(matrix))


def _decompose_similarity_matrix(matrix: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """Return ``(scale, rotation_degrees, translation)`` for a 2×3 matrix."""
    if matrix.shape != (2, 3):
        raise ValueError(f"expected (2, 3) matrix, got {matrix.shape!r}")
    a, _b, tx = matrix[0]
    c, _d, ty = matrix[1]
    scale = math.sqrt(a * a + c * c)
    rotation = math.degrees(math.atan2(c, a))
    return float(scale), float(rotation), (float(tx), float(ty))


def visible_subset_alignment_summary(
    landmarks: np.ndarray,
    visibility: T.Sequence[bool] | None = None,
    *,
    size: int = DEFAULT_ALIGNED_SIZE,
    coverage_ratio: float = 1.0,
    centering: CenteringType = "face",
    min_visible_points: int = _DEFAULT_MIN_VISIBLE_POINTS,
) -> VisibleSubsetAlignmentSummary:
    """Fit a Faceswap-style alignment summary from visible core landmarks.

    The fit uses exactly the same visible landmark subset for the Umeyama source
    and mean-face destination.  Full 68-point ``aligned_landmarks`` and
    ``normalized_landmarks`` are still exposed for diagnostics, but the matrix
    and average-distance fit are visibility-gated.
    """
    points = _as_68_landmarks(landmarks, name="landmarks")
    indices = visible_core_indices(visibility, min_visible_points=min_visible_points)
    matrix = _fit_matrix_from_visible_core(points, indices)
    adjusted = _adjusted_matrix(
        matrix,
        size=size,
        coverage_ratio=coverage_ratio,
        centering=centering,
    )
    normalized = _affine_transform_points(points, matrix)
    aligned = _affine_transform_points(points, adjusted)
    roi = _roi_from_adjusted_matrix(adjusted, size=size)
    scale, rotation_degrees, translation = _decompose_similarity_matrix(matrix)
    mean_face = MEAN_FACE[LandmarkType.LM_2D_51]
    core_offsets = np.asarray(indices, dtype=np.intp) - _CORE_START
    average_distance = float(np.mean(np.abs(normalized[list(indices)] - mean_face[core_offsets])))
    summary = AlignmentSummary(
        matrix=matrix,
        roi=roi,
        aligned_landmarks=aligned,
        normalized_landmarks=normalized,
        average_distance=average_distance,
        relative_eye_mouth_position=0.0,
        pitch=0.0,
        yaw=0.0,
        roll=rotation_degrees,
        scale=scale,
        rotation_degrees=rotation_degrees,
        translation=translation,
    )
    return VisibleSubsetAlignmentSummary(
        summary=summary,
        visible_indices=indices,
        coverage_ratio=coverage_ratio,
        centering=centering,
    )


def transform_cost_v3(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    visibility: T.Sequence[bool] | None = None,
    size: int = DEFAULT_ALIGNED_SIZE,
    coverage_ratio: float = 1.0,
    centering: CenteringType = "face",
    weights: TransformCostWeightsV3 = TransformCostWeightsV3(),
    soft_structural_penalty: float = 0.0,
    hard_invalid_reasons: T.Sequence[str] = (),
    min_visible_points: int = _DEFAULT_MIN_VISIBLE_POINTS,
) -> TransformCostV3:
    """Return direct v3 transform-impact cost for one prediction vs GT.

    The function is pure and label-oriented: it may use GT and visibility.  It
    does not mutate or replace ``alignment_geometry_v1``.  Hard-invalid reasons
    are surfaced for the later group-construction step that removes invalid
    candidates before LambdaRank training.
    """
    reasons = tuple(str(reason) for reason in hard_invalid_reasons if str(reason))
    try:
        pred_summary = visible_subset_alignment_summary(
            predicted,
            visibility,
            size=size,
            coverage_ratio=coverage_ratio,
            centering=centering,
            min_visible_points=min_visible_points,
        ).summary
        truth_summary = visible_subset_alignment_summary(
            truth,
            visibility,
            size=size,
            coverage_ratio=coverage_ratio,
            centering=centering,
            min_visible_points=min_visible_points,
        ).summary
        matrix_delta = alignment_matrix_delta(pred_summary, truth_summary, normalizer=float(size))
        crop_delta = roi_delta(pred_summary, truth_summary, normalizer=float(size))
        center_delta = float(crop_delta.center_normalized_distance)
        scale_delta = abs(math.log(max(pred_summary.scale, 1e-12) / max(truth_summary.scale, 1e-12)))
        roll_delta_degrees = float(matrix_delta.rotation_degrees_delta)
        fit_delta = max(float(average_distance_delta(pred_summary, truth_summary)), 0.0)
        hard_invalid = bool(reasons)
    except ValueError as err:
        center_delta = float("inf")
        scale_delta = float("inf")
        roll_delta_degrees = float("inf")
        fit_delta = float("inf")
        reasons = (*reasons, str(err))
        hard_invalid = True

    if hard_invalid:
        total_cost = float("inf")
    else:
        total_cost = float(
            weights.center * center_delta
            + weights.scale * scale_delta
            + weights.roll * roll_delta_degrees
            + weights.fit * fit_delta
            + soft_structural_penalty
        )
    return TransformCostV3(
        center_delta=center_delta,
        scale_delta=scale_delta,
        roll_delta_degrees=roll_delta_degrees,
        fit_delta=fit_delta,
        soft_structural_penalty=float(soft_structural_penalty),
        total_cost=total_cost,
        hard_invalid=hard_invalid,
        hard_invalid_reasons=reasons,
    )


__all__ = [
    "TransformCostV3",
    "TransformCostWeightsV3",
    "VisibleSubsetAlignmentSummary",
    "transform_cost_v3",
    "visible_core_indices",
    "visible_subset_alignment_summary",
]
