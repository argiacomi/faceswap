#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.schema`."""

import numpy as np
import pytest

from lib.align.constants import MAP_2D_68, LandmarkType
from lib.landmarks.schema import (
    LandmarkPrediction,
    canonicalize_schema,
    normalize_landmark_array,
    normalize_landmarks,
    to_canonical_68,
)


def test_canonicalize_schema_aliases() -> None:
    """Schema aliases normalize to local names."""
    assert canonicalize_schema("68pt") == "2d_68"
    assert canonicalize_schema(LandmarkType.LM_2D_98) == "2d_98"


def test_normalize_landmark_array_flat_input() -> None:
    """Flat x/y pairs are reshaped and converted to float32."""
    points = normalize_landmark_array([0, 1, 2, 3, 4, 5, 6, 7], schema="2d_4")
    assert points.dtype == np.float32
    np.testing.assert_array_equal(
        points,
        np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype="float32"),
    )


def test_normalize_landmark_array_rejects_non_finite() -> None:
    """NaN or infinite values are not accepted."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        normalize_landmark_array([[0, 1], [np.nan, 3]])


def test_landmark_prediction_validates_confidence_shape() -> None:
    """Confidence must match the point count."""
    points = np.zeros((68, 2), dtype="float32")
    with pytest.raises(ValueError, match="one value per landmark"):
        LandmarkPrediction(points=points, confidence=np.zeros(67, dtype="float32"))


def test_landmark_prediction_records_adapter_metadata() -> None:
    """Predictions expose the metadata required by model adapters."""
    points = np.zeros((68, 2), dtype="float32")
    prediction = LandmarkPrediction(
        landmarks=points,
        model_name="hrnet",
        source_landmark_count=98,
        coordinate_space="frame",
        metadata={"checkpoint": "test"},
    )

    assert prediction.points is prediction.landmarks
    assert prediction.source == "hrnet"
    assert prediction.model_name == "hrnet"
    assert prediction.source_landmark_count == 98
    assert prediction.coordinate_space == "frame"
    assert prediction.metadata == {"checkpoint": "test"}


def test_to_canonical_68_from_98_point_schema() -> None:
    """98-point inputs reuse Faceswap's existing 98-to-68 mapping."""
    points = np.arange(196, dtype="float32").reshape((98, 2))
    expected = points[MAP_2D_68[LandmarkType.LM_2D_98]]
    np.testing.assert_array_equal(
        to_canonical_68(points, source_schema="2d_98"), expected
    )


def test_normalize_landmarks_public_wrapper() -> None:
    """Public normalization wrapper maps supported inputs to canonical 68."""
    points = np.arange(196, dtype="float32").reshape((98, 2))
    expected = points[MAP_2D_68[LandmarkType.LM_2D_98]]
    np.testing.assert_array_equal(
        normalize_landmarks(points, source_schema="2d_98"), expected
    )
