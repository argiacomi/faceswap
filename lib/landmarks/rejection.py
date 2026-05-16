#!/usr/bin/env python3
"""Outlier rejection for landmark ensemble predictions.

Thresholds are robust z-score units computed from per-landmark distance to the
ensemble median. A threshold of ``3.5`` therefore means roughly "more than 3.5
robust standard deviations from the median landmark position".
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import numpy as np

from lib.landmarks.schema import LandmarkPrediction, to_canonical_68


@dataclass(frozen=True)
class OutlierRejectionResult:
    """Result from a prediction-level outlier rejection pass."""

    kept: list[int]
    rejected: list[int]
    distances: np.ndarray
    threshold: float


@dataclass(frozen=True)
class PerLandmarkOutlierResult:
    """Per-landmark rejection output."""

    weights: np.ndarray
    rejected: list[dict[str, int | float | str]]
    method: str


def _as_stack(predictions: T.Sequence[LandmarkPrediction | np.ndarray]) -> np.ndarray:
    points = [
        prediction.canonical_68().points
        if isinstance(prediction, LandmarkPrediction)
        else to_canonical_68(prediction)
        for prediction in predictions
    ]
    if not points:
        raise ValueError("at least one prediction is required")
    return np.stack(points, axis=0).astype("float32", copy=False)


def _normalize_weights(
    weights: T.Sequence[float] | np.ndarray,
    *,
    model_count: int,
    landmark_count: int,
) -> np.ndarray:
    """Validate and normalize model weights independently for every landmark."""
    array = np.asarray(weights, dtype="float32")
    if array.ndim == 1:
        if array.shape != (model_count,):
            raise ValueError(f"weights must have shape {(model_count,)}, got {array.shape}")
        if np.any(array < 0):
            raise ValueError("weights cannot contain negative values")
        total = float(array.sum())
        if total <= 0:
            raise ValueError("at least one weight must be greater than zero")
        return np.repeat((array / total)[:, None], landmark_count, axis=1)
    if array.shape != (model_count, landmark_count):
        raise ValueError(
            "weights must be one per model or a per-landmark matrix with shape "
            f"{(model_count, landmark_count)}, got {array.shape}"
        )
    if np.any(array < 0):
        raise ValueError("weights cannot contain negative values")
    totals = array.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("each landmark must have at least one non-zero model weight")
    return array / totals[None, :]


def pairwise_disagreement(stack: np.ndarray) -> np.ndarray:
    """Return mean pairwise disagreement per model and landmark in pixel units."""
    points = np.asarray(stack, dtype="float32")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"stack must have shape (M, 68, 2), got {points.shape}")
    diffs = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return diffs.mean(axis=1).astype("float32")


def reject_outliers(
    stack: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    model_names: T.Sequence[str] | None = None,
    threshold: float = 3.5,
    method: str = "hard_drop",
) -> PerLandmarkOutlierResult:
    """Return adjusted weights after per-landmark outlier handling.

    ``threshold`` is measured in robust z-score units. ``hard_drop`` zeros
    rejected model/landmark weights, ``downweight`` multiplies them by ``0.25``,
    ``weighted_median`` leaves weights unchanged for the median fusion step, and
    ``none`` only normalizes the incoming weights.
    """
    points = np.asarray(stack, dtype="float32")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"stack must have shape (M, 68, 2), got {points.shape}")
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    if method not in ("none", "hard_drop", "downweight", "weighted_median"):
        raise ValueError("method must be one of: none, hard_drop, downweight, weighted_median")
    names = tuple(model_names or [f"model_{idx}" for idx in range(points.shape[0])])
    if len(names) != points.shape[0]:
        raise ValueError("model_names must match stack model count")
    base = (
        np.ones((points.shape[0], points.shape[1]), dtype="float32")
        if weights is None
        else np.asarray(weights, dtype="float32")
    )
    adjusted = _normalize_weights(
        base, model_count=points.shape[0], landmark_count=points.shape[1]
    )
    if method in ("none", "weighted_median"):
        return PerLandmarkOutlierResult(weights=adjusted, rejected=[], method=method)

    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=2)
    med = np.median(distances, axis=0)
    mad = np.median(np.abs(distances - med), axis=0)
    scale = np.where(mad > 1e-6, mad / 0.6745, 1.0)
    z_scores = distances / scale
    rejected: list[dict[str, int | float | str]] = []

    for model_idx, landmark_idx in np.argwhere(z_scores > threshold):
        rejected.append(
            {
                "model": names[int(model_idx)],
                "model_index": int(model_idx),
                "landmark": int(landmark_idx),
                "score": float(z_scores[model_idx, landmark_idx]),
            }
        )
        if method == "hard_drop":
            adjusted[model_idx, landmark_idx] = 0.0
        else:
            adjusted[model_idx, landmark_idx] *= 0.25

    empty_columns = np.where(adjusted.sum(axis=0) <= 0)[0]
    for landmark_idx in empty_columns:
        base_column = np.asarray(base[:, landmark_idx], dtype="float32")
        eligible = np.where(base_column > 0)[0]
        if eligible.size:
            closest_model = int(
                eligible[np.argmin(distances[eligible, landmark_idx])]
            )
            adjusted[closest_model, landmark_idx] = float(base_column[closest_model])
        else:
            closest_model = int(np.argmin(distances[:, landmark_idx]))
            adjusted[closest_model, landmark_idx] = 1.0
        rejected = [
            item
            for item in rejected
            if not (item["landmark"] == int(landmark_idx) and item["model_index"] == closest_model)
        ]
    adjusted = _normalize_weights(
        adjusted, model_count=points.shape[0], landmark_count=points.shape[1]
    )
    return PerLandmarkOutlierResult(weights=adjusted, rejected=rejected, method=method)


def weighted_median(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return weighted median landmarks per coordinate.

    Vectorized across landmarks and coordinates: a single argsort over the
    models axis, weights reordered with the same permutation, and a per-
    column ``argmax(cumulative >= 0.5)`` to pick the median. ``_normalize_weights``
    guarantees the last cumulative value is ``1.0`` per landmark, so the
    boolean column is never all-False.
    """
    points = np.asarray(stack, dtype="float32")
    weight_matrix = _normalize_weights(
        weights, model_count=points.shape[0], landmark_count=points.shape[1]
    )  # (M, L)
    order = np.argsort(points, axis=0)  # (M, L, 2)
    sorted_values = np.take_along_axis(points, order, axis=0)
    weights_broadcast = np.broadcast_to(weight_matrix[..., None], points.shape)
    sorted_weights = np.take_along_axis(weights_broadcast, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    median_idx = np.argmax(cumulative >= 0.5, axis=0)  # (L, 2)
    selected = np.take_along_axis(sorted_values, median_idx[None, :, :], axis=0)
    return selected[0].astype("float32", copy=False)


def reject_landmark_outliers(
    predictions: T.Sequence[LandmarkPrediction | np.ndarray],
    *,
    threshold: float = 3.5,
    min_predictions: int = 3,
) -> OutlierRejectionResult:
    """Reject predictions whose mean robust z-score distance is too large."""
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    stack = _as_stack(predictions)
    if stack.shape[0] < min_predictions:
        kept = list(range(stack.shape[0]))
        return OutlierRejectionResult(
            kept=kept,
            rejected=[],
            distances=np.zeros(stack.shape[0], dtype="float32"),
            threshold=threshold,
        )

    median = np.median(stack, axis=0)
    point_distances = np.linalg.norm(stack - median, axis=2)
    median_dist = np.median(point_distances, axis=0)
    mad = np.median(np.abs(point_distances - median_dist), axis=0)
    scale = np.where(mad > 1e-6, mad / 0.6745, 1.0)
    robust_z = point_distances / scale
    distances = np.mean(robust_z, axis=1).astype("float32")
    kept = [idx for idx, distance in enumerate(distances) if distance <= threshold]
    rejected = [idx for idx, distance in enumerate(distances) if distance > threshold]
    if not kept:
        kept = [int(np.argmin(distances))]
        rejected = [idx for idx in rejected if idx not in kept]
    return OutlierRejectionResult(
        kept=kept,
        rejected=rejected,
        distances=distances,
        threshold=threshold,
    )
