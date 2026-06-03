#!/usr/bin/env python3
"""Direct transform-impact alignment cost for learned-quality v3.

This module intentionally does not change ``alignment_geometry_v1``.  It adds a
separate, explicit v3 scoring primitive for candidate-selection labels:

    candidate_alignment_cost =
        w_corner * crop_corner_rms_delta
        + w_fit * fit_delta
        + soft_structural_plausibility_penalty

``crop_corner_rms_delta`` is the single grounded geometric primitive: the RMS
displacement (in GT output-frame units) of the four aligned crop corners between
the candidate transform and the GT transform.  It subsumes translation, scale,
and rotation with their actual geometric effect, so there is one weight to reason
about instead of three competing magic numbers.  ``center``/``scale``/``roll``
deltas are still computed and reported as diagnostics but no longer contribute to
``total_cost``.

The deltas are computed from the same Faceswap alignment vocabulary used by the
existing geometry stack: Umeyama alignment matrices, crop/ROI corner deltas, and
average-distance fit deltas.  Unlike ``alignment_summary()``, the transform fit
here is gated by manifest landmark visibility, so profile/occluded GT points do
not pollute the transform used as the v3 oracle target.
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lib.align.aligned_face import batch_umeyama
from lib.align.constants import EXTRACT_RATIOS, MEAN_FACE, LandmarkType
from lib.landmarks.evaluation.geometry_signals import DEFAULT_ALIGNED_SIZE, AlignmentSummary

_CORE_START = 17
_CORE_END = 68
_DEFAULT_MIN_VISIBLE_POINTS = 8
_DEFAULT_CENTERING: T.Literal["face", "head", "legacy"] = "face"
FloatArray: T.TypeAlias = NDArray[np.float64]
DEFAULT_SOFT_STRUCTURAL_PENALTY_V3 = 0.05


@dataclass(frozen=True)
class TransformCostWeightsV3:
    """Weights for the direct v3 transform-cost components.

    The cost has a single dominant geometric knob (``corner``: the output-frame
    RMS crop-corner displacement, which already encodes translation, scale, and
    rotation) plus a small ``fit`` adjunct (the output-frame RMS fit-regret
    delta).  The legacy ``center``/``scale``/``roll`` weights were removed from
    the cost; those deltas remain available as diagnostics only.
    """

    corner: float = 1.0
    fit: float = 0.25


DEFAULT_TRANSFORM_COST_WEIGHTS_V3 = TransformCostWeightsV3()


@dataclass(frozen=True)
class StructuralValidityV3:
    """Two-tier structural validity signal for v3 scorer-label construction.

    Hard-invalid candidates are removed from LambdaRank groups before oracle
    selection.  They must not be represented as giant numeric costs.  Soft
    suspects stay rankable and contribute a finite additive penalty.
    """

    hard_invalid: bool
    hard_invalid_reasons: tuple[str, ...]
    soft_suspect: bool
    soft_suspect_reasons: tuple[str, ...]
    soft_structural_penalty: float

    def to_payload(self) -> dict[str, T.Any]:
        """Return a JSON-serializable diagnostic payload."""
        return {
            "hard_invalid": bool(self.hard_invalid),
            "hard_invalid_reasons": list(self.hard_invalid_reasons),
            "soft_suspect": bool(self.soft_suspect),
            "soft_suspect_reasons": list(self.soft_suspect_reasons),
            "soft_structural_penalty": float(self.soft_structural_penalty),
        }


@dataclass(frozen=True)
class TransformCostV3:
    """Direct transform-impact alignment cost for one candidate.

    ``hard_invalid`` marks candidates that should be removed from rankable v3
    groups by the scorer-dataset construction step.  The cost object still
    carries diagnostics, but LambdaRank should not consume hard-invalid rows as
    ordinary finite-cost items and they are never assigned an ``inf`` label.
    """

    corner_delta: float
    center_delta: float
    scale_delta: float
    roll_delta_degrees: float
    fit_delta: float
    soft_structural_penalty: float
    total_cost: float
    hard_invalid: bool
    hard_invalid_reasons: tuple[str, ...]
    soft_suspect_reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, T.Any]:
        """Return a JSON-serializable diagnostic payload.

        ``corner_delta`` is the cost-bearing geometric term; ``center_delta`` /
        ``scale_delta`` / ``roll_delta_degrees`` are retained diagnostics only.
        """
        return {
            "corner_delta": float(self.corner_delta),
            "center_delta": float(self.center_delta),
            "scale_delta": float(self.scale_delta),
            "roll_delta_degrees": float(self.roll_delta_degrees),
            "fit_delta": float(self.fit_delta),
            "soft_structural_penalty": float(self.soft_structural_penalty),
            "total_cost": float(self.total_cost),
            "hard_invalid": bool(self.hard_invalid),
            "hard_invalid_reasons": list(self.hard_invalid_reasons),
            "soft_suspect_reasons": list(self.soft_suspect_reasons),
        }


@dataclass(frozen=True)
class VisibleAlignmentSummary:
    """Alignment summary fitted from a visibility-gated landmark subset.

    ``visible_indices`` is the manifest-visible 68-point set, defaulting to all
    landmarks when no manifest visibility is available.  ``fit_indices`` is the
    usable visible subset consumed by Faceswap's 68-point Umeyama convention.
    Faceswap fits 68-point alignments from landmarks 17..67 against
    ``MEAN_FACE[LM_2D_51]``, so non-core visible landmarks are tracked for
    diagnostics but not used by the transform fit.
    """

    summary: AlignmentSummary
    adjusted_matrix: np.ndarray
    visible_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    coverage_ratio: float
    fit_rms_pixels: float


def _clean_reasons(reasons: T.Sequence[str]) -> tuple[str, ...]:
    """Return non-empty structural reason strings in stable order."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        text = str(reason).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return tuple(cleaned)


def structural_validity_v3(
    *,
    hard_invalid_reasons: T.Sequence[str] = (),
    soft_suspect_reasons: T.Sequence[str] = (),
    soft_structural_penalty: float | None = None,
) -> StructuralValidityV3:
    """Return the v3 hard-invalid / soft-suspect structural split.

    Hard-invalid examples: cloud collapse, eye/mouth flip, impossible topology,
    self-intersection, fatal pose/transform failure, or inability to fit the
    visible-subset transform.  Soft suspects stay rankable and carry a finite
    additive penalty.
    """
    hard = _clean_reasons(hard_invalid_reasons)
    soft = _clean_reasons(soft_suspect_reasons)
    if soft_structural_penalty is not None and soft_structural_penalty < 0:
        raise ValueError("soft_structural_penalty must be non-negative")
    if soft_structural_penalty is not None and soft_structural_penalty > 0 and not soft:
        soft = ("explicit_soft_structural_penalty",)
    penalty = (
        float(soft_structural_penalty)
        if soft_structural_penalty is not None
        else (DEFAULT_SOFT_STRUCTURAL_PENALTY_V3 if soft else 0.0)
    )
    return StructuralValidityV3(
        hard_invalid=bool(hard),
        hard_invalid_reasons=hard,
        soft_suspect=bool(soft),
        soft_suspect_reasons=soft,
        soft_structural_penalty=penalty,
    )


def _with_hard_invalid_reason(
    validity: StructuralValidityV3,
    reason: str,
) -> StructuralValidityV3:
    """Return ``validity`` with one additional hard-invalid reason."""
    return structural_validity_v3(
        hard_invalid_reasons=(*validity.hard_invalid_reasons, reason),
        soft_suspect_reasons=validity.soft_suspect_reasons,
        soft_structural_penalty=validity.soft_structural_penalty,
    )


def _as_68_landmarks(landmarks: np.ndarray, *, name: str) -> np.ndarray:
    """Return landmarks as finite ``float64`` with shape ``(68, 2)``."""
    points = T.cast(np.ndarray, np.asarray(landmarks, dtype=np.float64))
    if points.shape != (68, 2):
        raise ValueError(f"{name} expected (68, 2) landmarks, got {points.shape!r}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} landmarks contain non-finite values")
    return points


def _visibility_mask(visibility: T.Sequence[bool] | None) -> np.ndarray:
    """Return a 68-long boolean visibility mask, defaulting to all visible."""
    if visibility is None:
        return T.cast(np.ndarray, np.ones(68, dtype=bool))
    mask = T.cast(np.ndarray, np.asarray(visibility, dtype=bool))
    if mask.shape != (68,):
        raise ValueError(f"visibility expected shape (68,), got {mask.shape!r}")
    return mask


def visible_landmark_indices(visibility: T.Sequence[bool] | None) -> tuple[int, ...]:
    """Return manifest-visible 68-point indices, defaulting to all landmarks."""
    mask = _visibility_mask(visibility)
    return tuple(int(index) for index in np.flatnonzero(mask))


def _fit_indices_from_visible_indices(
    visible_indices: T.Sequence[int],
    *,
    min_visible_points: int = _DEFAULT_MIN_VISIBLE_POINTS,
) -> tuple[int, ...]:
    """Return the visible core subset usable by Faceswap's 68-point fit."""
    if min_visible_points < 3:
        raise ValueError("min_visible_points must be at least 3 for a similarity transform")
    visible_set = {int(index) for index in visible_indices}
    fit_indices = tuple(index for index in range(_CORE_START, _CORE_END) if index in visible_set)
    if len(fit_indices) < min_visible_points:
        raise ValueError(
            "visible-subset transform fit needs at least "
            f"{min_visible_points} visible core landmarks; got {len(fit_indices)}"
        )
    return fit_indices


def _fit_matrix_from_visible_core(
    landmarks: np.ndarray,
    fit_indices: T.Sequence[int],
) -> FloatArray:
    """Fit a 2×3 Umeyama matrix from visible core landmarks to the mean face."""
    core_offsets = np.asarray(fit_indices, dtype=np.intp) - _CORE_START
    if core_offsets.size < 3:
        raise ValueError("at least 3 visible core landmarks are required")
    source = landmarks[np.asarray(fit_indices, dtype=np.intp)]
    destination = MEAN_FACE[LandmarkType.LM_2D_51][core_offsets]
    raw_matrix = batch_umeyama(source[np.newaxis, ...], destination, estimate_scale=True)[0][:2]
    matrix = T.cast(FloatArray, raw_matrix.astype(np.float64))
    if not np.isfinite(matrix).all():
        raise ValueError("visible-subset Umeyama fit produced a non-finite matrix")
    return matrix


def _affine_transform_points(points: np.ndarray, matrix: np.ndarray) -> FloatArray:
    """Apply a 2×3 affine matrix to ``(N, 2)`` points."""
    homogeneous = T.cast(
        FloatArray,
        np.concatenate(
            [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
            axis=1,
        ),
    )
    return T.cast(FloatArray, homogeneous @ matrix.T)


def _invert_affine(matrix: np.ndarray) -> FloatArray:
    """Return the inverse of a 2×3 affine matrix as another 2×3 matrix."""
    full: FloatArray = np.eye(3, dtype=np.float64)
    full[:2, :] = matrix
    inverse = np.linalg.inv(full)
    if not np.isfinite(inverse).all():
        raise ValueError("affine matrix inversion produced non-finite values")
    return T.cast(FloatArray, inverse[:2, :])


def _padding_from_coverage(*, size: int, coverage_ratio: float) -> int:
    """Return Faceswap's face-centering padding for ``size``/``coverage_ratio``."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    if coverage_ratio <= 0:
        raise ValueError(f"coverage_ratio must be positive, got {coverage_ratio!r}")
    extract_ratio = float(EXTRACT_RATIOS[_DEFAULT_CENTERING])
    padding = round(size * (extract_ratio + coverage_ratio - 1.0) / (2.0 * coverage_ratio))
    return int(padding)


def _adjusted_matrix(matrix: np.ndarray, *, size: int, coverage_ratio: float) -> np.ndarray:
    """Return Faceswap-style face-centered crop matrix with coverage padding."""
    padding = _padding_from_coverage(size=size, coverage_ratio=coverage_ratio)
    adjusted = T.cast(np.ndarray, matrix.copy())
    adjusted *= size - 2 * padding
    adjusted[:, 2] += padding
    return adjusted


def _roi_from_adjusted_matrix(matrix: np.ndarray, *, size: int) -> FloatArray:
    """Return original-frame ROI corners implied by an adjusted crop matrix."""
    corners = T.cast(
        FloatArray,
        np.asarray(
            [[0.0, 0.0], [0.0, size - 1.0], [size - 1.0, size - 1.0], [size - 1.0, 0.0]],
            dtype=np.float64,
        ),
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


def _wrap_angle_degrees(angle: float) -> float:
    """Wrap an angle to ``(-180, 180]`` for deterministic roll deltas."""
    wrapped = ((float(angle) + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _fit_targets_output_frame(
    fit_indices: T.Sequence[int],
    *,
    size: int,
    coverage_ratio: float,
) -> FloatArray:
    """Return mean-face fit targets in the aligned output frame."""
    padding = _padding_from_coverage(size=size, coverage_ratio=coverage_ratio)
    core_offsets = np.asarray(fit_indices, dtype=np.intp) - _CORE_START
    target_normalized = MEAN_FACE[LandmarkType.LM_2D_51][core_offsets]
    return T.cast(FloatArray, target_normalized * (size - 2 * padding) + padding)


def _visible_fit_rms_pixels(
    aligned_landmarks: np.ndarray,
    fit_indices: T.Sequence[int],
    *,
    size: int,
    coverage_ratio: float,
) -> float:
    """Return output-frame RMS residual to the visible mean-face fit targets."""
    fit_points = aligned_landmarks[np.asarray(fit_indices, dtype=np.intp)]
    target_points = _fit_targets_output_frame(
        fit_indices,
        size=size,
        coverage_ratio=coverage_ratio,
    )
    residual = fit_points - target_points
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _crop_center_in_output_frame(source_roi: np.ndarray, output_matrix: np.ndarray) -> FloatArray:
    """Return a source-frame ROI center transformed into an aligned output frame."""
    center = T.cast(
        FloatArray, np.asarray(source_roi, dtype=np.float64).mean(axis=0, keepdims=True)
    )
    return T.cast(FloatArray, _affine_transform_points(center, output_matrix)[0])


def _crop_corner_rms_delta(
    candidate_roi: np.ndarray,
    truth_roi: np.ndarray,
    output_matrix: np.ndarray,
    *,
    size: int,
) -> float:
    """Return RMS crop-corner displacement in GT output-frame units.

    Both source-frame ROIs are mapped into the GT output frame by the GT
    ``output_matrix``; the RMS of the four per-corner Euclidean displacements is
    normalized by ``size`` so the term is invariant to pixel resolution. This is
    the single geometric primitive that subsumes the separate center/scale/roll
    deltas (translation shifts all corners, scale dilates them, rotation rotates
    them).
    """
    candidate_corners = _affine_transform_points(
        np.asarray(candidate_roi, dtype=np.float64), output_matrix
    )
    truth_corners = _affine_transform_points(
        np.asarray(truth_roi, dtype=np.float64), output_matrix
    )
    displacement = candidate_corners - truth_corners
    rms_pixels = float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))
    return rms_pixels / float(size)


def can_fit_visible_subset_transform(
    visibility: T.Sequence[bool] | None,
    *,
    min_visible_points: int = _DEFAULT_MIN_VISIBLE_POINTS,
) -> bool:
    """Return ``True`` when enough visible core landmarks exist to fit the transform.

    The visible-subset Umeyama fit needs at least ``min_visible_points`` visible core
    (17..67) landmarks.  Callers use this to avoid generating candidates that would be
    immediately hard-invalid with ``unable_to_fit_visible_subset_transform`` (e.g. only
    0/4/5/6/7 visible core points).
    """
    try:
        _fit_indices_from_visible_indices(
            visible_landmark_indices(visibility),
            min_visible_points=min_visible_points,
        )
    except ValueError:
        return False
    return True


def visible_subset_alignment_summary(
    landmarks: np.ndarray,
    visibility: T.Sequence[bool] | None = None,
    *,
    size: int = DEFAULT_ALIGNED_SIZE,
    coverage_ratio: float = 1.0,
    min_visible_points: int = _DEFAULT_MIN_VISIBLE_POINTS,
) -> VisibleAlignmentSummary:
    """Fit a Faceswap-style alignment summary from visible landmarks.

    ``visible_indices`` follows the manifest visibility mask exactly, defaulting
    to all 68 landmarks when no visibility is present.  The same visibility mask
    must be passed for candidate and GT summaries so both transforms are fitted
    from the same landmark subset.  For 68-point Faceswap alignment, only the
    visible core subset 17..67 is usable for the Umeyama fit because the project
    mean face for 68-point landmarks is ``MEAN_FACE[LM_2D_51]``.
    """
    points = _as_68_landmarks(landmarks, name="landmarks")
    visible_indices = visible_landmark_indices(visibility)
    fit_indices = _fit_indices_from_visible_indices(
        visible_indices,
        min_visible_points=min_visible_points,
    )
    matrix = _fit_matrix_from_visible_core(points, fit_indices)
    adjusted = _adjusted_matrix(matrix, size=size, coverage_ratio=coverage_ratio)
    normalized = _affine_transform_points(points, matrix)
    aligned = _affine_transform_points(points, adjusted)
    roi = _roi_from_adjusted_matrix(adjusted, size=size)
    scale, rotation_degrees, translation = _decompose_similarity_matrix(matrix)
    fit_rms_pixels = _visible_fit_rms_pixels(
        aligned,
        fit_indices,
        size=size,
        coverage_ratio=coverage_ratio,
    )
    summary = AlignmentSummary(
        matrix=matrix,
        roi=roi,
        aligned_landmarks=aligned,
        normalized_landmarks=normalized,
        average_distance=fit_rms_pixels / float(size),
        relative_eye_mouth_position=0.0,
        pitch=0.0,
        yaw=0.0,
        roll=rotation_degrees,
        scale=scale,
        rotation_degrees=rotation_degrees,
        translation=translation,
    )
    return VisibleAlignmentSummary(
        summary=summary,
        adjusted_matrix=adjusted,
        visible_indices=visible_indices,
        fit_indices=fit_indices,
        coverage_ratio=coverage_ratio,
        fit_rms_pixels=fit_rms_pixels,
    )


def _finite_transform_cost(
    *,
    corner_delta: float,
    fit_delta: float,
    soft_structural_penalty: float,
    weights: TransformCostWeightsV3,
) -> float:
    """Return finite weighted transform cost for rankable candidates.

    The cost is the single grounded corner-displacement primitive plus the small
    ``fit`` adjunct and the soft structural penalty.  ``center``/``scale``/
    ``roll`` are diagnostics and intentionally excluded here.
    """
    return float(weights.corner * corner_delta + weights.fit * fit_delta + soft_structural_penalty)


def transform_cost_v3(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    visibility: T.Sequence[bool] | None = None,
    size: int = DEFAULT_ALIGNED_SIZE,
    coverage_ratio: float = 1.0,
    weights: TransformCostWeightsV3 | None = None,
    hard_invalid_reasons: T.Sequence[str] = (),
    soft_suspect_reasons: T.Sequence[str] = (),
    soft_structural_penalty: float | None = None,
    min_visible_points: int = _DEFAULT_MIN_VISIBLE_POINTS,
) -> TransformCostV3:
    """Return direct v3 transform-impact cost for one prediction vs GT.

    Deterministic v3 deltas are expressed in GT-aligned output-frame units.
    Hard-invalid candidates are flagged for group exclusion and receive no giant
    numeric label; their returned ``total_cost`` is zero because it must not be
    consumed by ranking/oracle selection.
    """
    cost_weights = weights or DEFAULT_TRANSFORM_COST_WEIGHTS_V3
    validity = structural_validity_v3(
        hard_invalid_reasons=hard_invalid_reasons,
        soft_suspect_reasons=soft_suspect_reasons,
        soft_structural_penalty=soft_structural_penalty,
    )
    try:
        pred = visible_subset_alignment_summary(
            predicted,
            visibility,
            size=size,
            coverage_ratio=coverage_ratio,
            min_visible_points=min_visible_points,
        )
        truth_fit = visible_subset_alignment_summary(
            truth,
            visibility,
            size=size,
            coverage_ratio=coverage_ratio,
            min_visible_points=min_visible_points,
        )
        pred_center = _crop_center_in_output_frame(pred.summary.roi, truth_fit.adjusted_matrix)
        truth_center = _crop_center_in_output_frame(
            truth_fit.summary.roi, truth_fit.adjusted_matrix
        )
        center_delta = float(np.linalg.norm(pred_center - truth_center) / float(size))
        scale_delta = abs(
            math.log(max(pred.summary.scale, 1e-12) / max(truth_fit.summary.scale, 1e-12))
        )
        roll_delta_degrees = abs(
            _wrap_angle_degrees(pred.summary.rotation_degrees - truth_fit.summary.rotation_degrees)
        )
        fit_delta = max(pred.fit_rms_pixels - truth_fit.fit_rms_pixels, 0.0) / float(size)
        corner_delta = _crop_corner_rms_delta(
            pred.summary.roi,
            truth_fit.summary.roi,
            truth_fit.adjusted_matrix,
            size=size,
        )
    except ValueError as err:
        corner_delta = 0.0
        center_delta = 0.0
        scale_delta = 0.0
        roll_delta_degrees = 0.0
        fit_delta = 0.0
        validity = _with_hard_invalid_reason(
            validity,
            f"unable_to_fit_visible_subset_transform: {err}",
        )

    total_cost = (
        0.0
        if validity.hard_invalid
        else _finite_transform_cost(
            corner_delta=corner_delta,
            fit_delta=fit_delta,
            soft_structural_penalty=validity.soft_structural_penalty,
            weights=cost_weights,
        )
    )
    return TransformCostV3(
        corner_delta=corner_delta,
        center_delta=center_delta,
        scale_delta=scale_delta,
        roll_delta_degrees=roll_delta_degrees,
        fit_delta=fit_delta,
        soft_structural_penalty=validity.soft_structural_penalty,
        total_cost=total_cost,
        hard_invalid=validity.hard_invalid,
        hard_invalid_reasons=validity.hard_invalid_reasons,
        soft_suspect_reasons=validity.soft_suspect_reasons,
    )


__all__ = [
    "DEFAULT_SOFT_STRUCTURAL_PENALTY_V3",
    "StructuralValidityV3",
    "TransformCostV3",
    "TransformCostWeightsV3",
    "VisibleAlignmentSummary",
    "structural_validity_v3",
    "transform_cost_v3",
    "can_fit_visible_subset_transform",
    "visible_landmark_indices",
    "visible_subset_alignment_summary",
]
