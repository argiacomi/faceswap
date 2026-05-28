#!/usr/bin/env python3
"""Tests for SPIGA native pose metadata."""

from __future__ import annotations

import typing as T
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lib.infer.align import Align
from lib.infer.objects import ExtractBatchAligned
from plugins.extract.align.spiga import SPIGA, _pose_rows_from_raw


def _raw_pose_from_faceswap(yaw: float, pitch: float, roll: float) -> list[float]:
    """Return SPIGA raw euler values that should map to Faceswap pose values."""
    return [90.0 - yaw, -pitch, -roll - 90.0, 0.0, 0.0, 0.0]


def test_pose_rows_convert_spiga_euler_to_faceswap_degrees() -> None:
    """SPIGA raw euler output is converted into yaw/pitch/roll rows."""
    raw_pose = torch.tensor([[80.0, -5.0, -100.0, 1.0, 2.0, 3.0]], dtype=torch.float32)

    rows = _pose_rows_from_raw(raw_pose).detach().cpu().numpy()

    np.testing.assert_allclose(rows[0, :, 0], np.array([10.0, 5.0, 10.0], dtype="float32"))
    np.testing.assert_allclose(rows[0, :, 1], np.zeros(3, dtype="float32"))


@pytest.mark.parametrize(
    "label,raw_pose,expected",
    [
        ("frontal", _raw_pose_from_faceswap(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("left_yaw", _raw_pose_from_faceswap(-30.0, 0.0, 0.0), (-30.0, 0.0, 0.0)),
        ("right_yaw", _raw_pose_from_faceswap(30.0, 0.0, 0.0), (30.0, 0.0, 0.0)),
        ("up_pitch", _raw_pose_from_faceswap(0.0, 20.0, 0.0), (0.0, 20.0, 0.0)),
        ("down_pitch", _raw_pose_from_faceswap(0.0, -20.0, 0.0), (0.0, -20.0, 0.0)),
        ("clockwise_roll", _raw_pose_from_faceswap(0.0, 0.0, 20.0), (0.0, 0.0, 20.0)),
        (
            "counterclockwise_roll",
            _raw_pose_from_faceswap(0.0, 0.0, -20.0),
            (0.0, 0.0, -20.0),
        ),
    ],
)
def test_pose_rows_pin_faceswap_pose_conventions(
    label: str,
    raw_pose: list[float],
    expected: tuple[float, float, float],
) -> None:
    """Known SPIGA euler fixtures should map to Faceswap yaw/pitch/roll conventions."""
    del label
    rows = _pose_rows_from_raw(torch.tensor([raw_pose], dtype=torch.float32)).detach().numpy()

    np.testing.assert_allclose(rows[0, :, 0], np.array(expected, dtype="float32"))
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
    plugin._model_config = SimpleNamespace(num_landmarks=98)
    arr = np.random.default_rng(0).uniform(0.0, 1.0, (3, 98, 2)).astype(np.float32)

    result = plugin.post_process(arr)

    assert result is arr
    assert plugin.last_debug_metadata == []


def test_align_stores_spiga_pose_metadata_with_validation() -> None:
    """Align metadata stores SPIGA pose under a traceable plugin namespace."""
    handler = T.cast(T.Any, object.__new__(Align))
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


def test_align_averages_spiga_pose_metadata_across_refeeds() -> None:
    """SPIGA pose metadata should match the merged re-feed prediction."""
    handler = T.cast(T.Any, object.__new__(Align))
    handler.plugin = SimpleNamespace(
        name="SPIGA",
        last_debug_metadata=[
            {"pose": {"yaw": 10.0, "pitch": 1.0, "roll": -2.0, "source": "spiga"}},
            {"pose": {"yaw": 20.0, "pitch": 3.0, "roll": 2.0, "source": "spiga"}},
            {"pose": {"yaw": 30.0, "pitch": 5.0, "roll": 4.0, "source": "spiga"}},
            {"pose": {"yaw": -10.0, "pitch": -1.0, "roll": 6.0, "source": "spiga"}},
            {"pose": {"yaw": -20.0, "pitch": -3.0, "roll": 8.0, "source": "spiga"}},
            {"pose": {"yaw": -30.0, "pitch": -5.0, "roll": 10.0, "source": "spiga"}},
        ],
    )
    handler._re_feed = SimpleNamespace(total_feeds=3)
    aligned = ExtractBatchAligned()
    aligned._cache_rotation = np.zeros((2, 3, 1), dtype="float32")
    batch = SimpleNamespace(aligned=aligned)

    Align._store_plugin_metadata(handler, batch, face_count=2, feed_count=6)

    first_pose = batch.aligned.metadata[0]["spiga"]["pose"]
    assert first_pose["yaw"] == pytest.approx(20.0)
    assert first_pose["pitch"] == pytest.approx(3.0)
    assert first_pose["roll"] == pytest.approx(4.0 / 3.0)
    assert first_pose["merged_feeds"] == 3
    assert first_pose["merge_strategy"] == "mean"
    assert first_pose["delta"]["yaw"] == pytest.approx(20.0)

    second_pose = batch.aligned.metadata[1]["spiga"]["pose"]
    assert second_pose["yaw"] == pytest.approx(-20.0)
    assert second_pose["pitch"] == pytest.approx(-3.0)
    assert second_pose["roll"] == pytest.approx(8.0)
