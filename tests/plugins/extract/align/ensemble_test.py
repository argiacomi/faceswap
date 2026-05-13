#!/usr/bin/env python3
"""Smoke tests for the landmark ensemble aligner plugin."""

import numpy as np
import pytest

from lib.align.constants import MAP_2D_68, LandmarkType
from lib.landmarks.adapters import (
    FaceswapAlignerAdapter,
    LandmarkAdapterConfig,
    StaticLandmarkAdapter,
)
from lib.landmarks.coordinates import roi_to_matrix
from plugins.extract.align.ensemble import Ensemble


def _points(value: float) -> np.ndarray:
    points = np.full((68, 2), value, dtype="float32")
    return points


def test_ensemble_fuses_injected_static_adapters() -> None:
    """Injected adapters keep the plugin testable without model downloads."""
    adapters = [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig(
                "near",
                coordinate_space="normalized_crop",
                weight=3.0,
            ),
            _points(0.25),
        ),
        StaticLandmarkAdapter(
            LandmarkAdapterConfig(
                "far",
                coordinate_space="normalized_crop",
                weight=1.0,
            ),
            _points(0.75),
        ),
    ]
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        reject_outliers=False,
        strategy="static_weighted",
    )
    plugin.model = plugin.load_model()
    roi = plugin.pre_process(np.array([[10, 20, 50, 60]], dtype="int32"))

    result = plugin.process(np.zeros((1, 256, 256, 3), dtype="float32"))

    np.testing.assert_array_equal(roi, np.array([[10, 20, 50, 60]], dtype="int32"))
    assert result.shape == (1, 68, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], _points(0.375), rtol=1e-6, atol=1e-6)
    assert plugin.last_debug_metadata[0]["sources"] == ("near", "far")
    assert plugin.last_debug_metadata[0]["strategy"] == "static_weighted"


def test_ensemble_plain_average_smoke_without_preprocess() -> None:
    """Warmup-style calls use identity matrices when pre_process has not run."""
    adapters = [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("a", coordinate_space="frame"),
            _points(0.2),
        ),
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("b", coordinate_space="frame"),
            _points(0.6),
        ),
    ]
    plugin = Ensemble(adapters=adapters, reject_outliers=False, strategy="plain_average")
    plugin.model = plugin.load_model()

    result = plugin.process(np.zeros((2, 256, 256, 3), dtype="float32"))

    assert result.shape == (2, 68, 2)
    np.testing.assert_allclose(result, np.full((2, 68, 2), 0.4, dtype="float32"))
    assert len(plugin.last_debug_metadata) == 2


def test_ensemble_predict_landmarks_68_returns_frame_space_points() -> None:
    """The public ensemble API returns fused canonical frame pixels."""
    adapters = [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("a", coordinate_space="normalized_crop"),
            _points(0.25),
        ),
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("b", coordinate_space="normalized_crop"),
            _points(0.75),
        ),
    ]
    plugin = Ensemble(adapters=adapters, reject_outliers=False, strategy="plain_average")
    plugin.model = plugin.load_model()
    matrix = roi_to_matrix(np.array([10, 20, 110, 120], dtype="int32"))

    result = plugin.predict_landmarks_68(
        np.zeros((256, 256, 3), dtype="float32"),
        matrix=matrix,
    )

    assert result.shape == (68, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.full((68, 2), [60.0, 70.0], dtype="float32"))


class _FakeAligner:
    """Minimal Faceswap aligner plugin double."""

    is_rgb = True
    dtype = np.float32
    scale = (0, 1)

    def __init__(self, points: np.ndarray) -> None:
        self._points = points

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Return one placeholder output per input crop."""
        return np.repeat(self._points[None], batch.shape[0], axis=0)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Expose the fake landmarks in the plugin's public output contract."""
        return raw.astype("float32", copy=False)


@pytest.mark.parametrize(
    "name,schema,points",
    [
        ("hrnet", "2d_68", np.full((68, 2), 0.25, dtype="float32")),
        ("spiga", "2d_98", np.arange(196, dtype="float32").reshape((98, 2)) / 100.0),
        ("orformer", "2d_98", np.arange(196, dtype="float32").reshape((98, 2)) / 200.0),
    ],
)
def test_aligner_adapter_wrappers_return_canonical_68_frame_points(
    name: str,
    schema: str,
    points: np.ndarray,
) -> None:
    """HRNet/SPIGA/ORFormer wrappers normalize fake outputs without model loads."""
    matrix = roi_to_matrix(np.array([10, 20, 110, 120], dtype="int32"))
    adapter = FaceswapAlignerAdapter(
        LandmarkAdapterConfig(name, schema=schema, coordinate_space="normalized_crop"),
        _FakeAligner(points),
    )

    result = adapter.predict_landmarks_68(
        np.zeros((256, 256, 3), dtype="float32"),
        matrix=matrix,
    )

    canonical = points if schema == "2d_68" else points[MAP_2D_68[LandmarkType.LM_2D_98]]
    expected_frame = canonical * 100.0 + np.array([10.0, 20.0], dtype="float32")
    assert result.shape == (68, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected_frame)
