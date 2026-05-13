#!/usr/bin/env python3
"""Tests for the public landmark prediction API."""

from __future__ import annotations

import numpy as np

from lib.landmarks import predict_landmarks_68
from lib.landmarks.adapters import (
    FaceswapAlignerAdapter,
    LandmarkAdapterConfig,
    StaticLandmarkAdapter,
)
from lib.landmarks.coordinates import roi_to_matrix


def _points(value: float = 0.0, count: int = 68) -> np.ndarray:
    return np.full((count, 2), value, dtype="float32")


def test_predict_landmarks_68_returns_frame_space_points() -> None:
    """The public API fuses enabled adapters into canonical frame coordinates."""
    adapters = [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("hrnet", coordinate_space="normalized_crop"),
            _points(0.25),
        ),
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("spiga", coordinate_space="normalized_crop"),
            _points(0.75),
        ),
    ]
    matrix = roi_to_matrix(np.array([10, 20, 50, 60], dtype="float32"))

    points = predict_landmarks_68(
        np.zeros((32, 32, 3), dtype="float32"),
        adapters,
        matrix=matrix,
        strategy="plain_average",
    )

    assert points.shape == (68, 2)
    np.testing.assert_allclose(points, np.tile([30.0, 40.0], (68, 1)))


class _FakeAligner:
    """Tiny Faceswap-style aligner test double."""

    is_rgb = True
    dtype = "float32"
    scale = (0, 1)

    def __init__(self, value: float, count: int = 68) -> None:
        self.value = value
        self.count = count

    def process(self, images: np.ndarray) -> np.ndarray:
        """Return one fixed normalized-crop prediction per image."""
        return np.full((images.shape[0], self.count, 2), self.value, dtype="float32")

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Pass through model output."""
        return raw


def test_faceswap_aligner_adapter_fixtures_return_canonical_68() -> None:
    """HRNet/SPIGA/ORFormer wrapper path normalizes fake plugin outputs."""
    matrix = roi_to_matrix(np.array([0, 0, 100, 100], dtype="float32"))
    images = np.zeros((1, 16, 16, 3), dtype="float32")
    adapters = [
        FaceswapAlignerAdapter(
            LandmarkAdapterConfig(model, schema="2d_68", coordinate_space="normalized_crop"),
            _FakeAligner(0.5),
        )
        for model in ("hrnet", "spiga", "orformer")
    ]

    for adapter in adapters:
        prediction = adapter.predict_batch(images, matrices=matrix[None])[0]
        assert prediction.model_name in {"hrnet", "spiga", "orformer"}
        assert prediction.coordinate_space == "frame"
        assert prediction.landmarks.shape == (68, 2)
        np.testing.assert_allclose(prediction.landmarks, _points(50.0))
