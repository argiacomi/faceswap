#!/usr/bin/env python3
"""Fusion helpers for landmark ensemble predictions."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import numpy as np

from lib.landmarks.core.rejection import reject_outliers as adjust_outlier_weights
from lib.landmarks.core.rejection import weighted_median
from lib.landmarks.core.schema import CANONICAL_SCHEMA, LandmarkPrediction, to_canonical_68


@dataclass(frozen=True)
class FusionResult:
    """Fused landmark prediction plus trace metadata."""

    points: np.ndarray
    schema: str
    strategy: str
    weights: np.ndarray
    sources: tuple[str, ...]
    kept_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...] = ()
    rejected_landmarks: int = 0


def _prediction_points(prediction: LandmarkPrediction | np.ndarray) -> np.ndarray:
    if isinstance(prediction, LandmarkPrediction):
        return prediction.canonical_68().points
    return to_canonical_68(prediction)


def _prediction_source(prediction: LandmarkPrediction | np.ndarray, index: int) -> str:
    if isinstance(prediction, LandmarkPrediction) and prediction.source:
        return prediction.source
    return f"prediction_{index}"


def normalize_weights(weights: T.Sequence[float], count: int) -> np.ndarray:
    """Validate and normalize one weight per model."""
    array = np.asarray(weights, dtype="float32")
    if array.shape != (count,):
        raise ValueError(f"weights must have shape {(count,)}, got {array.shape}")
    if np.any(array < 0):
        raise ValueError("weights cannot contain negative values")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("at least one weight must be greater than zero")
    return array / total


def normalize_weight_matrix(
    weights: T.Sequence[float] | np.ndarray,
    *,
    model_count: int,
    landmark_count: int = 68,
) -> np.ndarray:
    """Validate and normalize model weights independently for every landmark."""
    array = np.asarray(weights, dtype="float32")
    if array.ndim == 1:
        array = normalize_weights(array, model_count)[:, None]
        return np.repeat(array, landmark_count, axis=1)
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


def weights_from_sources(
    predictions: T.Sequence[LandmarkPrediction | np.ndarray],
    source_weights: T.Mapping[str, float],
    *,
    default_weight: float = 1.0,
) -> np.ndarray:
    """Build static weights from prediction source names."""
    if default_weight < 0:
        raise ValueError("default_weight cannot be negative")
    weights = [
        source_weights.get(_prediction_source(prediction, idx), default_weight)
        for idx, prediction in enumerate(predictions)
    ]
    return normalize_weights(weights, len(predictions))


def weighted_average(
    predictions: T.Sequence[LandmarkPrediction | np.ndarray],
    weights: T.Sequence[float] | None = None,
    *,
    reject_outliers: bool = False,
    outlier_threshold: float = 3.5,
    outlier_method: str | None = None,
    strategy: str = "static_weighted",
) -> FusionResult:
    """Fuse predictions with static weights and optional per-landmark rejection.

    ``outlier_threshold`` is measured in robust z-score units. ``outlier_method``
    supports ``none``, ``hard_drop``, ``downweight``, and ``weighted_median``.
    When omitted, ``reject_outliers=True`` keeps the historical ``hard_drop``
    behavior and ``False`` uses ``none``.
    """
    if not predictions:
        raise ValueError("at least one prediction is required")
    method = outlier_method or ("hard_drop" if reject_outliers else "none")
    if method not in ("none", "hard_drop", "downweight", "weighted_median"):
        raise ValueError(
            "outlier_method must be one of: none, hard_drop, downweight, weighted_median"
        )
    stack = np.stack([_prediction_points(prediction) for prediction in predictions], axis=0)
    selected_weights = (
        np.ones((len(predictions), stack.shape[1]), dtype="float32")
        if weights is None
        else np.asarray(weights, dtype="float32")
    )
    normalized = normalize_weight_matrix(
        selected_weights,
        model_count=len(predictions),
        landmark_count=stack.shape[1],
    )
    rejected_indices: list[int] = []
    rejected_landmarks = 0
    if method == "weighted_median":
        points = weighted_median(stack, normalized)
    elif method == "none":
        points = (stack * normalized[..., None]).sum(axis=0).astype("float32")
    else:
        rejection = adjust_outlier_weights(
            stack,
            normalized,
            model_names=[
                _prediction_source(prediction, idx) for idx, prediction in enumerate(predictions)
            ],
            threshold=outlier_threshold,
            method=method,
        )
        normalized = rejection.weights
        rejected_landmarks = len(rejection.rejected)
        rejected_indices = [idx for idx in range(len(predictions)) if np.all(normalized[idx] <= 0)]
        points = (stack * normalized[..., None]).sum(axis=0).astype("float32")
    kept_indices = [idx for idx in range(len(predictions)) if idx not in set(rejected_indices)]
    return FusionResult(
        points=points,
        schema=CANONICAL_SCHEMA,
        strategy=strategy,
        weights=normalized,
        sources=tuple(
            _prediction_source(prediction, idx) for idx, prediction in enumerate(predictions)
        ),
        kept_indices=tuple(kept_indices),
        rejected_indices=tuple(rejected_indices),
        rejected_landmarks=rejected_landmarks,
    )


def plain_average(
    predictions: T.Sequence[LandmarkPrediction | np.ndarray],
    *,
    reject_outliers: bool = False,
    outlier_threshold: float = 3.5,
    outlier_method: str | None = None,
) -> FusionResult:
    """Fuse predictions with equal weights."""
    return weighted_average(
        predictions,
        weights=None,
        reject_outliers=reject_outliers,
        outlier_threshold=outlier_threshold,
        outlier_method=outlier_method,
        strategy="plain_average",
    )


def static_weighted(
    predictions: T.Sequence[LandmarkPrediction | np.ndarray],
    weights: T.Sequence[float] | np.ndarray,
    *,
    reject_outliers: bool = False,
    outlier_threshold: float = 3.5,
    outlier_method: str | None = None,
) -> FusionResult:
    """Fuse predictions with static per-model or per-landmark weights."""
    return weighted_average(
        predictions,
        weights=weights,
        reject_outliers=reject_outliers,
        outlier_threshold=outlier_threshold,
        outlier_method=outlier_method,
        strategy="static_weighted",
    )
