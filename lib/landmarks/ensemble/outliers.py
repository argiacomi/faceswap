#!/usr/bin/env python3
"""Per-landmark outlier rejection strategies."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import numpy as np

from lib.landmarks.fusion import normalize_weight_matrix


@dataclass(frozen=True)
class PerLandmarkOutlierResult:
    """Per-landmark rejection output."""

    weights: np.ndarray
    rejected: list[dict[str, int | float | str]]
    method: str


def pairwise_disagreement(stack: np.ndarray) -> np.ndarray:
    """Return mean pairwise disagreement per model and landmark."""
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
    """Return adjusted weights after per-landmark outlier handling."""
    points = np.asarray(stack, dtype="float32")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"stack must have shape (M, 68, 2), got {points.shape}")
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    names = tuple(model_names or [f"model_{idx}" for idx in range(points.shape[0])])
    if len(names) != points.shape[0]:
        raise ValueError("model_names must match stack model count")
    base = (
        np.ones((points.shape[0], points.shape[1]), dtype="float32")
        if weights is None
        else np.asarray(weights, dtype="float32")
    )
    adjusted = normalize_weight_matrix(
        base, model_count=points.shape[0], landmark_count=points.shape[1]
    )
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=2)
    med = np.median(distances, axis=0)
    mad = np.median(np.abs(distances - med), axis=0)
    scale = np.where(mad > 1e-6, mad / 0.6745, 1.0)
    z_scores = distances / scale
    rejected: list[dict[str, int | float | str]] = []

    if method not in ("none", "hard_drop", "downweight"):
        raise ValueError("method must be one of: none, hard_drop, downweight")
    if method == "none":
        return PerLandmarkOutlierResult(weights=adjusted, rejected=[], method=method)

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
        closest_model = int(np.argmin(distances[:, landmark_idx]))
        adjusted[closest_model, landmark_idx] = base[closest_model, landmark_idx]
        rejected = [
            item
            for item in rejected
            if not (item["landmark"] == int(landmark_idx) and item["model_index"] == closest_model)
        ]
    adjusted = normalize_weight_matrix(
        adjusted, model_count=points.shape[0], landmark_count=points.shape[1]
    )
    return PerLandmarkOutlierResult(weights=adjusted, rejected=rejected, method=method)


def weighted_median(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return weighted median landmarks per coordinate."""
    points = np.asarray(stack, dtype="float32")
    weight_matrix = normalize_weight_matrix(
        weights, model_count=points.shape[0], landmark_count=points.shape[1]
    )
    output = np.empty(points.shape[1:], dtype="float32")
    for landmark_idx in range(points.shape[1]):
        landmark_weights = weight_matrix[:, landmark_idx]
        for coord_idx in range(2):
            values = points[:, landmark_idx, coord_idx]
            order = np.argsort(values)
            cumulative = np.cumsum(landmark_weights[order])
            output[landmark_idx, coord_idx] = values[order[np.searchsorted(cumulative, 0.5)]]
    return output
