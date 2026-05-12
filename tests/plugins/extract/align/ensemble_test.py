#!/usr/bin/env python3
"""Smoke tests for the landmark ensemble aligner plugin."""

import numpy as np

from lib.landmarks.adapters import LandmarkAdapterConfig, StaticLandmarkAdapter
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
