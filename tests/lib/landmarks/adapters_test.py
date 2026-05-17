#!/usr/bin/env python3
"""Tests for landmark adapter public APIs."""

import numpy as np

from lib.landmarks.adapters import LandmarkAdapterConfig, StaticLandmarkAdapter
from lib.landmarks.coordinates import roi_to_matrix
from lib.landmarks.core.schema import LandmarkPrediction


def _points(value: float = 0.5) -> np.ndarray:
    return np.full((68, 2), value, dtype="float32")


def test_static_adapter_batch_returns_frame_landmark_predictions() -> None:
    """Batch predictions are canonical LandmarkPrediction objects in frame space."""
    adapter = StaticLandmarkAdapter(
        LandmarkAdapterConfig("static", coordinate_space="normalized_crop"),
        _points(0.25),
    )
    matrices = roi_to_matrix(np.array([[10, 20, 110, 120], [50, 60, 150, 160]], dtype="int32"))

    predictions = adapter.predict_batch(
        np.zeros((2, 8, 8, 3), dtype="float32"),
        matrices=matrices,
    )

    assert len(predictions) == 2
    assert all(isinstance(prediction, LandmarkPrediction) for prediction in predictions)
    assert all(prediction.points.shape == (68, 2) for prediction in predictions)
    assert all(prediction.coordinate_space == "frame" for prediction in predictions)
    np.testing.assert_allclose(
        predictions[0].points,
        np.full((68, 2), [35.0, 45.0], dtype="float32"),
    )
    np.testing.assert_allclose(
        predictions[1].points,
        np.full((68, 2), [75.0, 85.0], dtype="float32"),
    )


def test_predict_landmarks_68_returns_canonical_frame_points() -> None:
    """The single-image public API returns only the canonical ``(68, 2)`` array."""
    adapter = StaticLandmarkAdapter(
        LandmarkAdapterConfig("static", coordinate_space="normalized_crop"),
        _points(0.5),
    )
    matrix = roi_to_matrix(np.array([10, 20, 110, 120], dtype="int32"))

    landmarks = adapter.predict_landmarks_68(
        np.zeros((8, 8, 3), dtype="float32"),
        matrix=matrix,
    )

    assert landmarks.shape == (68, 2)
    assert landmarks.dtype == np.float32
    np.testing.assert_allclose(landmarks, np.full((68, 2), [60.0, 70.0], dtype="float32"))
