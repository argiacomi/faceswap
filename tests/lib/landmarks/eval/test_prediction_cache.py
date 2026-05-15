#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.eval.prediction_cache`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction


def test_cache_encodes_windows_unsafe_sample_ids(tmp_path: Path) -> None:
    """Sample IDs with reserved Windows characters are safe cache directories."""
    cache = DiskPredictionCache(tmp_path)
    sample_id = "fixture:clean-000"
    prediction = LandmarkPrediction(
        np.zeros((68, 2), dtype="float32"), model_name="hrnet"
    )

    path = cache.write(sample_id, prediction)

    assert ":" not in path.relative_to(tmp_path).parts[0]
    assert cache.available_models(sample_id) == ("hrnet",)
    assert cache.sample_ids() == (sample_id,)
    assert cache.read(sample_id, "hrnet").metadata["sample_id"] == sample_id
