#!/usr/bin/env python3
"""Tests for cache and metric helpers."""

import numpy as np
import pytest

from lib.landmarks.cache.runtime_cache import PredictionCache, cache_key_for_array
from lib.landmarks.core.metrics import (
    auc,
    failure_rate,
    mean_point_error,
    normalized_mean_error,
)
from lib.landmarks.core.schema import LandmarkPrediction


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
    assert cache_key_for_array(image, "adapter") != cache_key_for_array(changed, "adapter")


def test_metrics_for_shifted_landmarks() -> None:
    """Metric helpers return expected values for a uniform shift."""
    truth = _face_points()
    predicted = _face_points(offset=3.0)
    expected_error = float(np.sqrt(18.0))
    assert mean_point_error(predicted, truth) == pytest.approx(expected_error)
    assert normalized_mean_error(predicted, truth, normalizer=10.0) == pytest.approx(
        expected_error / 10.0
    )


def test_normalized_mean_error_uses_ibug_outer_eye_corners_by_default() -> None:
    """Default NME normalizes by canonical 68 outer eye corners 36 and 45."""
    truth = np.zeros((68, 2), dtype="float32")
    truth[36] = [10.0, 20.0]
    truth[45] = [60.0, 20.0]
    predicted = truth.copy()
    predicted[:, 0] += 5.0

    assert mean_point_error(predicted, truth) == pytest.approx(5.0)
    assert normalized_mean_error(predicted, truth) == pytest.approx(5.0 / 50.0)


def test_failure_rate_and_auc() -> None:
    """Distribution metrics handle simple normalized error arrays."""
    errors = np.array([0.01, 0.02, 0.10], dtype="float32")
    assert failure_rate(errors, threshold=0.08) == pytest.approx(1 / 3)
    assert 0.0 < auc(errors, threshold=0.10) <= 1.0


def test_mean_point_error_visibility_none_matches_legacy_average() -> None:
    """Passing ``visibility=None`` preserves the legacy all-landmark mean."""
    truth = _face_points()
    predicted = _face_points(offset=3.0)
    baseline = mean_point_error(predicted, truth)
    assert mean_point_error(predicted, truth, visibility=None) == pytest.approx(baseline)


def test_mean_point_error_visibility_masks_out_occluded_points() -> None:
    """A visibility mask excludes flagged-occluded landmarks from the mean."""
    truth = np.zeros((68, 2), dtype="float32")
    predicted = truth.copy()
    # Move every landmark slightly, but blow up landmarks 50..67 (occluded).
    predicted[:50, 0] = 1.0
    predicted[50:, 0] = 100.0
    visibility = np.ones(68, dtype=bool)
    visibility[50:] = False

    masked = mean_point_error(predicted, truth, visibility=visibility)
    unmasked = mean_point_error(predicted, truth)
    assert masked == pytest.approx(1.0)
    # The unmasked mean is dragged up by the occluded points; masking should
    # strictly improve (lower) it.
    assert masked < unmasked


def test_mean_point_error_visibility_all_false_falls_back_to_full_mean() -> None:
    """When no landmark is visible, return the full-landmark mean (not zero)."""
    truth = np.zeros((68, 2), dtype="float32")
    predicted = truth.copy()
    predicted[:, 0] = 7.0
    visibility = np.zeros(68, dtype=bool)
    full_mean = mean_point_error(predicted, truth)
    assert mean_point_error(predicted, truth, visibility=visibility) == pytest.approx(full_mean)


def test_mean_point_error_visibility_length_mismatch_raises() -> None:
    """A visibility array of the wrong length is a programmer error."""
    truth = np.zeros((68, 2), dtype="float32")
    predicted = truth.copy()
    with pytest.raises(ValueError):
        mean_point_error(predicted, truth, visibility=np.ones(10, dtype=bool))


def test_normalized_mean_error_forwards_visibility() -> None:
    """``normalized_mean_error`` honours the visibility mask via mean_point_error."""
    truth = np.zeros((68, 2), dtype="float32")
    truth[36] = [10.0, 20.0]
    truth[45] = [60.0, 20.0]
    predicted = truth.copy()
    predicted[:50, 0] += 5.0
    predicted[50:, 0] += 500.0
    visibility = np.ones(68, dtype=bool)
    visibility[50:] = False

    nme = normalized_mean_error(predicted, truth, visibility=visibility)
    # Visible-only mean error is 5.0; normalizer is 50.0.
    assert nme == pytest.approx(5.0 / 50.0)
