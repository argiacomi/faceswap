#!/usr/bin/env python3
"""Tests for pose hard-slice labelling."""

from __future__ import annotations

import math
import typing as T

import pytest

from lib.landmarks.evaluation.hard_slices import (
    HARD_NEGATIVE_BUCKETS,
    HardSliceThresholds,
    hard_negative_bucket,
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


def _mined_sample(bucket: str, *, conditions: list[str] | None = None) -> dict[str, T.Any]:
    return {
        "sample_id": f"mined_{bucket}",
        "dataset": "wflw",
        "conditions": list(conditions or []),
        "metadata": {"hard_negative_bucket": bucket, "hard_negative_weight": 5.0},
    }


def test_hard_negative_bucket_reads_metadata() -> None:
    assert hard_negative_bucket(_mined_sample("profile_occlusion")) == "profile_occlusion"
    assert hard_negative_bucket({"metadata": {"hard_negative_bucket": "BOGUS"}}) is None
    assert hard_negative_bucket({"sample_id": "x"}) is None
    assert frozenset(
        {"profile_occlusion", "profile", "occlusion", "anchor"}
    ) == HARD_NEGATIVE_BUCKETS


def test_mined_samples_bypass_pose_slicing_without_pose_para() -> None:
    samples = [
        _mined_sample("profile_occlusion"),
        _mined_sample("profile"),
        _mined_sample("occlusion"),
    ]
    sliced, counts = slice_manifest_samples(samples, hard_only=True, include_unposed=False)
    by_id = {s["sample_id"]: s for s in sliced}
    assert set(by_id) == {"mined_profile_occlusion", "mined_profile", "mined_occlusion"}
    profile_occ = by_id["mined_profile_occlusion"]
    assert profile_occ["hard_slice"] == "profile_occlusion"
    assert profile_occ["condition"] == "profile_occlusion"
    assert profile_occ["conditions"][0] == "profile_occlusion"
    assert counts["profile_occlusion"] == 1
    assert counts["occlusion"] == 1


def test_mined_anchors_are_kept_even_with_hard_only() -> None:
    sliced, counts = slice_manifest_samples(
        [_mined_sample("anchor")], hard_only=True, include_unposed=False
    )
    assert [s["sample_id"] for s in sliced] == ["mined_anchor"]
    assert sliced[0]["condition"] == "anchor"
    assert counts["anchor"] == 1


def test_mined_path_preserves_existing_conditions() -> None:
    sliced, _ = slice_manifest_samples(
        [_mined_sample("profile", conditions=["pose"])], hard_only=True
    )
    assert sliced[0]["conditions"] == ["profile", "pose"]


def test_non_mined_samples_still_use_pose_slicing() -> None:
    # A non-mined AFLW sample with profile yaw and no hard-negative metadata
    # must still go through pose slicing (old behavior preserved).
    sliced, counts = slice_manifest_samples(
        [_sample(yaw=45.0, roll=0.0)], hard_only=True, include_unposed=False
    )
    assert len(sliced) == 1
    assert sliced[0]["hard_slice"] == "profile_right"
    assert counts["profile_right"] == 1
