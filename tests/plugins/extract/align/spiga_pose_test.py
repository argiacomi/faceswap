#!/usr/bin/env python3
"""Tests for SPIGA native pose metadata."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from lib.infer.align import Align
from lib.infer.objects import ExtractBatchAligned
from plugins.extract.align.spiga import SPIGA, _pose_rows_from_raw


def test_pose_rows_convert_spiga_euler_to_faceswap_degrees() -> None:
    """SPIGA raw euler output is converted into yaw/pitch/roll rows."""
    raw_pose = torch.tensor([[80.0, -5.0, -100.0, 1.0, 2.0, 3.0]], dtype=torch.float32)

    rows = _pose_rows_from_raw(raw_pose).detach().cpu().numpy()

    np.testing.assert_allclose(rows[0, :, 0], np.array([10.0, 5.0, 10.0], dtype="float32"))
    np.testing.assert_allclose(rows[0, :, 1], np.zeros(3, dtype="float32"))


def test_spiga_post_process_strips_pose_and_stores_metadata() -> None:
    """SPIGA keeps native pose metadata while returning only landmarks."""
    plugin = SPIGA()
    landmarks = np.zeros((1, plugin._model_config.num_landmarks, 2), dtype="float32")
    pose_rows = np.array([[[12.0, 0.0], [-3.0, 0.0], [4.0, 0.0]]], dtype="float32")
    output = np.concatenate([landmarks, pose_rows], axis=1)

    result = plugin.post_process(output)

    assert result.shape == landmarks.shape
    assert plugin.last_debug_metadata == [
        {
            "pose": {
                "yaw": 12.0,
                "pitch": -3.0,
                "roll": 4.0,
                "source": "spiga",
                "model": "spiga",
                "units": "degrees",
                "coordinate_convention": "faceswap",
            }
        }
    ]


def test_uninitialized_spiga_post_process_passes_plain_landmarks() -> None:
    """Post-process remains compatible with object.__new__ test doubles."""
    plugin = object.__new__(SPIGA)
    arr = np.random.default_rng(0).uniform(0.0, 1.0, (3, 98, 2)).astype(np.float32)

    result = plugin.post_process(arr)

    assert result is arr
    assert plugin.last_debug_metadata == []


def test_align_stores_spiga_pose_metadata_with_validation() -> None:
    """Align metadata stores SPIGA pose under a traceable plugin namespace."""
    handler = object.__new__(Align)
    handler.plugin = SimpleNamespace(
        name="SPIGA",
        last_debug_metadata=[
            {
                "pose": {
                    "yaw": 12.0,
                    "pitch": -3.0,
                    "roll": 4.0,
                    "source": "spiga",
                    "model": "spiga",
                    "units": "degrees",
                }
            }
        ],
    )
    handler._re_feed = SimpleNamespace(total_feeds=1)
    aligned = ExtractBatchAligned()
    aligned._cache_rotation = np.zeros((1, 3, 1), dtype="float32")
    batch = SimpleNamespace(aligned=aligned)

    Align._store_plugin_metadata(handler, batch, face_count=1, feed_count=1)

    pose = batch.aligned.metadata[0]["spiga"]["pose"]
    assert pose["source"] == "spiga"
    assert pose["derived"]["source"] == "faceswap_landmarks"
    assert pose["delta"] == {"yaw": 12.0, "pitch": -3.0, "roll": 4.0}
    assert pose["validation"]["max_abs_delta"] == 12.0
