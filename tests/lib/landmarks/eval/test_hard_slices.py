#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.eval.hard_slices` (#82)."""

from __future__ import annotations

import math

import pytest

from lib.landmarks.eval.hard_slices import (
    HARD_SLICES,
    HardSliceThresholds,
    hard_slice_label,
    is_hard_slice,
    slice_manifest_samples,
    yaw_degrees,
)


def _sample(yaw_radians: float | None = None, **extras) -> dict:
    sample = {
        "sample_id": extras.pop("sample_id", "s"),
        "image": "img.png",
        "landmarks": "truth.npy",
    }
    sample.update(extras)
    if yaw_radians is not None:
        sample["metadata"] = {"Pose_Para": [0.0, yaw_radians, 0.0, 0.0, 0.0, 0.0, 1.0]}
    return sample


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [
        (0.0, "frontal"),
        (10.0, "frontal"),
        (20.0, "intermediate"),
        (35.0, "profile_right"),
        (-35.0, "profile_left"),
        (75.0, "large_yaw_right"),
        (-75.0, "large_yaw_left"),
    ],
)
def test_hard_slice_label_bucketing(degrees: float, expected: str) -> None:
    """Bucket boundaries match the documented yaw thresholds."""
    assert hard_slice_label(degrees) == expected


def test_hard_slice_label_handles_missing_pose() -> None:
    """Samples without pose annotations land in ``no_pose``."""
    assert hard_slice_label(None) == "no_pose"
    assert is_hard_slice("no_pose") is False


def test_hard_slices_are_explicitly_the_four_buckets() -> None:
    """The HARD_SLICES tuple is part of the artifact contract."""
    assert set(HARD_SLICES) == {
        "profile_left",
        "profile_right",
        "large_yaw_left",
        "large_yaw_right",
    }


def test_yaw_degrees_returns_none_when_pose_is_missing() -> None:
    """Manifest samples without ``Pose_Para`` return ``None`` for yaw."""
    assert yaw_degrees(_sample()) is None


def test_yaw_degrees_reads_radians_from_metadata() -> None:
    """Yaw is taken from ``metadata.Pose_Para[1]`` (radians) and converted."""
    sample = _sample(yaw_radians=math.radians(45.0))
    assert yaw_degrees(sample) == pytest.approx(45.0, abs=1e-6)


def test_slice_manifest_samples_keeps_only_hard_buckets_by_default() -> None:
    """Default ``hard_only=True`` drops frontal and intermediate samples."""
    samples = [
        _sample(sample_id="frontal", yaw_radians=0.0),
        _sample(sample_id="profile_l", yaw_radians=math.radians(-35.0)),
        _sample(sample_id="profile_r", yaw_radians=math.radians(35.0)),
        _sample(sample_id="extreme", yaw_radians=math.radians(70.0)),
        _sample(sample_id="unknown"),  # no pose
    ]
    sliced, counts = slice_manifest_samples(samples)
    ids = sorted(item["sample_id"] for item in sliced)
    assert ids == ["extreme", "profile_l", "profile_r"]
    # Each kept sample is tagged with its bucket.
    by_id = {item["sample_id"]: item["hard_slice"] for item in sliced}
    assert by_id["profile_l"] == "profile_left"
    assert by_id["profile_r"] == "profile_right"
    assert by_id["extreme"] == "large_yaw_right"
    # Histogram covers every input bucket including no_pose / intermediate.
    assert counts["frontal"] == 1
    assert counts["no_pose"] == 1


def test_slice_manifest_samples_can_include_all_buckets() -> None:
    """``hard_only=False`` keeps every sample and merely tags them."""
    samples = [
        _sample(sample_id="frontal", yaw_radians=0.0),
        _sample(sample_id="profile_r", yaw_radians=math.radians(40.0)),
    ]
    sliced, _ = slice_manifest_samples(samples, hard_only=False)
    assert len(sliced) == 2
    assert all("hard_slice" in item for item in sliced)
    # The new bucket name replaces any prior condition so harness grouping works.
    assert sliced[0]["condition"] == "frontal"


def test_slice_manifest_samples_drops_unposed_by_default() -> None:
    """``no_pose`` samples require explicit opt-in via ``include_unposed``."""
    samples = [
        _sample(sample_id="profile", yaw_radians=math.radians(35.0)),
        _sample(sample_id="unknown"),
    ]
    sliced, _ = slice_manifest_samples(samples)
    assert [item["sample_id"] for item in sliced] == ["profile"]
    sliced_unposed, _ = slice_manifest_samples(samples, include_unposed=True, hard_only=False)
    assert {item["sample_id"] for item in sliced_unposed} == {"profile", "unknown"}


def test_hard_slice_thresholds_validate_ordering() -> None:
    """Thresholds must satisfy frontal < profile_min < profile_max."""
    with pytest.raises(ValueError):
        HardSliceThresholds(frontal_degrees=40.0, profile_min_degrees=30.0)
    with pytest.raises(ValueError):
        HardSliceThresholds(profile_min_degrees=80.0, profile_max_degrees=60.0)
