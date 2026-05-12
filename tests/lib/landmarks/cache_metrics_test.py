#!/usr/bin/env python3
"""Tests for cache and metric helpers."""

import numpy as np
import pytest

from lib.landmarks.cache import PredictionCache, cache_key_for_array
from lib.landmarks.metrics import (
    auc,
    failure_rate,
    mean_point_error,
    normalized_mean_error,
)
from lib.landmarks.schema import LandmarkPrediction


def _face_points(offset: float = 0.0) -> np.ndarray:
    points = np.stack(
        (
            np.linspace(0, 67, 68, dtype="float32"),
            np.linspace(10, 77, 68, dtype="float32"),
        ),
        axis=1,
    )
    return points + offset


def test_prediction_cache_lru_eviction() -> None:
    """Cache evicts least recently used entries at max size."""
    cache = PredictionCache(max_size=2)
    prediction = LandmarkPrediction(_face_points())
    cache.put("first", prediction)
    cache.put("second", prediction)
    assert cache.get("first") is prediction
    cache.put("third", prediction)
    assert cache.get("second") is None
    assert cache.get("first") is prediction
    assert cache.get("third") is prediction


def test_cache_key_for_array_uses_image_content() -> None:
    """Array cache keys change when image bytes change."""
    image = np.zeros((4, 4, 3), dtype="uint8")
    changed = image.copy()
    changed[0, 0, 0] = 1
    assert cache_key_for_array(image, "adapter") != cache_key_for_array(
        changed, "adapter"
    )


def test_metrics_for_shifted_landmarks() -> None:
    """Metric helpers return expected values for a uniform shift."""
    truth = _face_points()
    predicted = _face_points(offset=3.0)
    expected_error = float(np.sqrt(18.0))
    assert mean_point_error(predicted, truth) == pytest.approx(expected_error)
    assert normalized_mean_error(predicted, truth, normalizer=10.0) == pytest.approx(
        expected_error / 10.0
    )


def test_failure_rate_and_auc() -> None:
    """Distribution metrics handle simple normalized error arrays."""
    errors = np.array([0.01, 0.02, 0.10], dtype="float32")
    assert failure_rate(errors, threshold=0.08) == pytest.approx(1 / 3)
    assert 0.0 < auc(errors, threshold=0.10) <= 1.0
