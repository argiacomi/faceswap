#!/usr/bin/env python3
"""Evaluation metrics for landmark predictions."""

from __future__ import annotations

import numpy as np

from lib.landmarks.schema import to_canonical_68


def _paired_points(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pred = to_canonical_68(predicted)
    truth = to_canonical_68(target)
    if pred.shape != truth.shape:
        raise ValueError(
            f"predicted and target shapes must match, got {pred.shape}, {truth.shape}"
        )
    return pred, truth


def per_landmark_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean error for each landmark point."""
    pred, truth = _paired_points(predicted, target)
    return np.linalg.norm(pred - truth, axis=1).astype("float32")


def mean_point_error(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean Euclidean landmark error."""
    return float(np.mean(per_landmark_error(predicted, target)))


def normalized_mean_error(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    normalizer: float | None = None,
    interocular_indices: tuple[int, int] = (36, 45),
) -> float:
    """Return mean point error normalized by an explicit or interocular distance."""
    pred, truth = _paired_points(predicted, target)
    if normalizer is None:
        normalizer = float(
            np.linalg.norm(
                truth[interocular_indices[0]] - truth[interocular_indices[1]]
            )
        )
    if normalizer <= 0:
        raise ValueError("normalizer must be greater than zero")
    return float(mean_point_error(pred, truth) / normalizer)


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
