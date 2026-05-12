#!/usr/bin/env python3
"""Outlier rejection for landmark ensemble predictions."""

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


def reject_landmark_outliers(
    predictions: T.Sequence[LandmarkPrediction | np.ndarray],
    *,
    threshold: float = 3.5,
    min_predictions: int = 3,
) -> OutlierRejectionResult:
    """Reject predictions whose mean robust distance from the median is too large."""
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
