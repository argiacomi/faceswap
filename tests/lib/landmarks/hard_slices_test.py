#!/usr/bin/env python3
"""Tests for pose hard-slice labelling."""

from __future__ import annotations

import math
import typing as T

import pytest

from lib.landmarks.evaluation.hard_slices import (
    HardSliceThresholds,
    hard_slice_label,
    roll_degrees,
    slice_manifest_samples,
    yaw_degrees,
    yaw_slice_label,
)


def _sample(
    *, yaw: float | None, roll: float | None, sample_id: str = "sample"
) -> dict[str, T.Any]:
    metadata: dict[str, T.Any] = {}
    if yaw is not None and roll is not None:
        metadata["Pose_Para"] = [0.0, math.radians(yaw), math.radians(roll), 0.0, 0.0, 0.0, 1.0]
    return {
        "sample_id": sample_id,
        "dataset": "aflw2000-3d",
        "condition": "default",
        "conditions": ["default"],
        "metadata": metadata,
    }


def test_yaw_and_roll_degrees_read_pose_para() -> None:
    sample = _sample(yaw=-42.0, roll=37.0)

    assert yaw_degrees(sample) == pytest.approx(-42.0)
    assert roll_degrees(sample) == pytest.approx(37.0)


def test_yaw_slice_label_preserves_existing_yaw_buckets() -> None:
    assert yaw_slice_label(0.0) == "frontal"
    assert yaw_slice_label(20.0) == "intermediate"
    assert yaw_slice_label(-45.0) == "profile_left"
    assert yaw_slice_label(45.0) == "profile_right"
    assert yaw_slice_label(-70.0) == "large_yaw_left"
    assert yaw_slice_label(70.0) == "large_yaw_right"
    assert yaw_slice_label(None) == "no_pose"


def test_hard_slice_label_routes_frontal_roll_to_roll_buckets() -> None:
    assert hard_slice_label(0.0, roll_deg=29.9) == "frontal"
    assert hard_slice_label(0.0, roll_deg=30.0) == "large_roll"
    assert hard_slice_label(0.0, roll_deg=-44.9) == "large_roll"
    assert hard_slice_label(0.0, roll_deg=45.0) == "extreme_roll"


def test_hard_slice_label_routes_yaw_hard_roll_to_combined_buckets() -> None:
    assert hard_slice_label(-45.0, roll_deg=35.0) == "rolled_profile_left"
    assert hard_slice_label(45.0, roll_deg=35.0) == "rolled_profile_right"
    assert hard_slice_label(-70.0, roll_deg=35.0) == "rolled_large_yaw_left"
    assert hard_slice_label(70.0, roll_deg=35.0) == "rolled_large_yaw_right"


def test_slice_manifest_samples_includes_roll_buckets_and_pose_metadata() -> None:
    samples = [
        _sample(yaw=5.0, roll=0.0, sample_id="frontal"),
        _sample(yaw=5.0, roll=31.0, sample_id="large-roll"),
        _sample(yaw=5.0, roll=46.0, sample_id="extreme-roll"),
        _sample(yaw=-40.0, roll=0.0, sample_id="profile-left"),
        _sample(yaw=-40.0, roll=31.0, sample_id="rolled-profile-left"),
        _sample(yaw=70.0, roll=31.0, sample_id="rolled-large-yaw-right"),
        _sample(yaw=None, roll=None, sample_id="no-pose"),
    ]

    sliced, counts = slice_manifest_samples(samples, hard_only=True, include_unposed=False)

    labels = {sample["sample_id"]: sample["hard_slice"] for sample in sliced}
    assert labels == {
        "large-roll": "large_roll",
        "extreme-roll": "extreme_roll",
        "profile-left": "profile_left",
        "rolled-profile-left": "rolled_profile_left",
        "rolled-large-yaw-right": "rolled_large_yaw_right",
    }
    assert counts["frontal"] == 1
    assert counts["large_roll"] == 1
    assert counts["extreme_roll"] == 1
    assert counts["profile_left"] == 1
    assert counts["rolled_profile_left"] == 1
    assert counts["rolled_large_yaw_right"] == 1
    assert counts["no_pose"] == 1
    assert all("roll_degrees" in sample for sample in sliced)
    assert all("yaw_slice" in sample for sample in sliced)


def test_slice_manifest_samples_keeps_unposed_only_when_requested() -> None:
    samples = [_sample(yaw=None, roll=None, sample_id="no-pose")]

    sliced, counts = slice_manifest_samples(samples, hard_only=True, include_unposed=False)
    assert sliced == []
    assert counts["no_pose"] == 1

    sliced, _ = slice_manifest_samples(samples, hard_only=False, include_unposed=True)
    assert sliced[0]["hard_slice"] == "no_pose"


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        HardSliceThresholds(roll_degrees=0.0)
    with pytest.raises(ValueError):
        HardSliceThresholds(roll_degrees=45.0, extreme_roll_degrees=30.0)
