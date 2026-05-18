#!/usr/bin/env python3
"""Evaluation metrics for landmark predictions.

By default, NME uses the iBUG/300W interocular convention: mean point error
divided by the ground-truth distance between outer eye corners 36 and 45 in the
canonical 68-point ordering. Datasets with a different published convention
should pass an explicit ``normalizer`` or ``interocular_indices``.
"""

from __future__ import annotations

import numpy as np

from lib.landmarks.core.schema import to_canonical_68


def _paired_points(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = to_canonical_68(predicted)
    truth = to_canonical_68(target)
    if pred.shape != truth.shape:
        raise ValueError(
            f"predicted and target shapes must match, got {pred.shape}, {truth.shape}"
        )
    return pred, truth


def per_landmark_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean error for each landmark point.

    ``visibility`` is intentionally **not** an argument here: callers may need
    the full per-landmark error array (including invisible points) for
    diagnostics. Apply masking at the aggregation site instead.
    """
    pred, truth = _paired_points(predicted, target)
    return np.linalg.norm(pred - truth, axis=1).astype("float32")


def mean_point_error(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    visibility: np.ndarray | None = None,
) -> float:
    """Return mean Euclidean landmark error, optionally restricted to visible points.

    When ``visibility`` is provided (one bool per landmark), only the points
    flagged True contribute to the mean. This matters for datasets like COFW
    (per-landmark occlusion flags) and AFLW2000-3D / WFLW under heavy yaw,
    where self-occluded points annotated in 3D space would otherwise inflate
    the error against 2D-projected predictions. ``None`` preserves the
    legacy behaviour of averaging over all landmarks.
    """
    errors = per_landmark_error(predicted, target)
    if visibility is None:
        return float(np.mean(errors))
    mask = np.asarray(visibility, dtype=bool)
    if mask.shape[0] != errors.shape[0]:
        raise ValueError(
            f"visibility length {mask.shape[0]} must match landmark count {errors.shape[0]}"
        )
    if not mask.any():
        # No visible landmarks → undefined; fall back to the all-landmark mean
        # so the metric never silently returns 0 for a sample worth flagging.
        return float(np.mean(errors))
    return float(np.mean(errors[mask]))


def normalized_mean_error(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    normalizer: float | None = None,
    interocular_indices: tuple[int, int] = (36, 45),
    visibility: np.ndarray | None = None,
) -> float:
    """Return mean point error normalized by an explicit or interocular distance.

    ``visibility`` is forwarded to :func:`mean_point_error`; the interocular
    normalizer is unaffected (eye corners 36/45 are assumed to be visible
    landmarks on any face the manifest accepted).
    """
    pred, truth = _paired_points(predicted, target)
    if normalizer is None:
        normalizer = float(
            np.linalg.norm(truth[interocular_indices[0]] - truth[interocular_indices[1]])
        )
    if normalizer <= 0:
        raise ValueError("normalizer must be greater than zero")
    return float(mean_point_error(pred, truth, visibility=visibility) / normalizer)


def failure_rate(errors: np.ndarray, *, threshold: float) -> float:
    """Return the fraction of normalized errors above a threshold."""
    values = np.asarray(errors, dtype="float32")
    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    if values.size == 0:
        return 0.0
    return float(np.mean(values > threshold))


def auc(errors: np.ndarray, *, threshold: float = 0.08, steps: int = 100) -> float:
    """Approximate cumulative error distribution AUC up to ``threshold``."""
    values = np.asarray(errors, dtype="float32")
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    if steps <= 1:
        raise ValueError("steps must be greater than one")
    if values.size == 0:
        return 0.0
    xs = np.linspace(0.0, threshold, steps, dtype="float32")
    ced = np.array([np.mean(values <= x_val) for x_val in xs], dtype="float32")
    return float(np.trapezoid(ced, xs) / threshold)
