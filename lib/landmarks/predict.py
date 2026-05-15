#!/usr/bin/env python3
"""Public prediction API for standalone landmark ensembles."""

from __future__ import annotations

import typing as T

import numpy as np

from lib.landmarks.adapters import LandmarkAdapter
from lib.landmarks.ensemble.outliers import weighted_median
from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
)
from lib.landmarks.fusion import normalize_weight_matrix, plain_average, static_weighted
from lib.landmarks.schema import LandmarkPrediction


def predict_landmarks_68(
    image: np.ndarray,
    adapters: T.Sequence[LandmarkAdapter],
    *,
    face: object | None = None,
    matrix: np.ndarray | None = None,
    strategy: str = "static_weighted",
    weights: T.Sequence[float] | np.ndarray | None = None,
    outlier_method: str | None = None,
    outlier_threshold: float = 3.5,
) -> np.ndarray:
    """Predict fused canonical 68-point landmarks in original-frame pixels.

    Adapters that emit normalized crop coordinates require ``matrix`` so their
    output can be converted to frame space before fusion. ``outlier_threshold``
    is a robust z-score in landmark coordinate space. ``outlier_method`` is
    derived from the canonical ``strategy``; passing it explicitly overrides
    the strategy default for callers that need to (e.g. tests).
    """
    if not adapters:
        raise ValueError("at least one adapter is required")
    canonical = canonical_strategy(strategy)
    method = outlier_method if outlier_method is not None else strategy_outlier_method(canonical)
    batch = image[None]
    matrices = None if matrix is None else np.asarray(matrix, dtype="float32")[None]
    predictions: list[LandmarkPrediction] = []
    for adapter in adapters:
        if not adapter.config.enabled:
            continue
        predictions.extend(adapter.predict_batch(batch, matrices=matrices, faces=[face]))
    if not predictions:
        raise ValueError("no enabled adapters produced predictions")

    if not strategy_requires_weights(canonical):
        return plain_average(
            predictions,
            outlier_method=method,
            outlier_threshold=outlier_threshold,
        ).points

    weight_matrix = (
        np.asarray(weights, dtype="float32")
        if weights is not None
        else np.asarray([adapter.config.weight for adapter in adapters if adapter.config.enabled])
    )
    if canonical == "weighted_median":
        stack = np.stack([prediction.canonical_68().points for prediction in predictions], axis=0)
        normalized = normalize_weight_matrix(
            weight_matrix,
            model_count=len(predictions),
            landmark_count=stack.shape[1],
        )
        return weighted_median(stack, normalized)
    return static_weighted(
        predictions,
        weight_matrix,
        outlier_method=method,
        outlier_threshold=outlier_threshold,
    ).points
