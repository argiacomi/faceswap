#!/usr/bin/env python3
"""Tests for the hard-negative classification helpers."""

from __future__ import annotations

import pytest

from lib.landmarks.datasets import hard_negative_mining as hnm


def test_wflw_pose_and_occlusion_is_profile_occlusion() -> None:
    sample = {"conditions": ["pose", "occlusion"]}
    result = hnm.classify_hard_negative(sample)
    assert result is not None
    assert result.bucket == "profile_occlusion"
    assert result.priority == 1
    assert result.weight == 5.0


def test_wflw_pose_only_is_profile() -> None:
    result = hnm.classify_hard_negative({"conditions": ["pose"]})
    assert result is not None
    assert result.bucket == "profile"
    assert result.weight == 3.0


def test_wflw_occlusion_only_is_occlusion() -> None:
    result = hnm.classify_hard_negative({"conditions": ["occlusion"]})
    assert result is not None
    assert result.bucket == "occlusion"
    assert result.weight == 2.0


def test_merl_rav_left_plus_external_occlusion_is_profile_occlusion() -> None:
    sample = {
        "condition": "left",
        "metadata": {"attributes": {"externally_occluded": True}},
    }
    result = hnm.classify_hard_negative(sample)
    assert result is not None
    assert result.bucket == "profile_occlusion"


def test_aflw2000_rolled_profile_right_is_profile() -> None:
    result = hnm.classify_hard_negative({"hard_slice": "rolled_profile_right"})
    assert result is not None
    assert result.bucket == "profile"


def test_anchor_label_is_anchor() -> None:
    result = hnm.classify_hard_negative({"condition": "frontal"})
    assert result is not None
    assert result.bucket == "anchor"
    assert result.weight == 1.0


def test_unlabeled_sample_returns_none() -> None:
    assert hnm.classify_hard_negative({"sample_id": "x"}) is None


def test_visibility_mask_with_hidden_point_marks_occlusion() -> None:
    sample = {"metadata": {"visibility": [True, True, False, True]}}
    result = hnm.classify_hard_negative(sample)
    assert result is not None
    assert result.bucket == "occlusion"


def test_visibility_all_visible_is_not_occlusion() -> None:
    sample = {"metadata": {"visibility": [True, True, True]}}
    assert hnm.classify_hard_negative(sample) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Profile-Left", "profile_left"),
        ("  large  yaw ", "large_yaw"),
        ("__pose__", "pose"),
        (None, ""),
    ],
)
def test_normalize_label(raw: str | None, expected: str) -> None:
    assert hnm.normalize_label(raw) == expected


def test_annotate_sample_preserves_fields_and_adds_metadata() -> None:
    sample = {
        "sample_id": "wflw_0001",
        "image": "img.png",
        "landmarks": "lm.npy",
        "conditions": ["pose", "occlusion"],
        "dataset": "wflw",
    }
    classification = hnm.classify_hard_negative(sample)
    assert classification is not None
    annotated = hnm.annotate_sample(sample, classification)
    # original payload fields preserved
    assert annotated["image"] == "img.png"
    assert annotated["landmarks"] == "lm.npy"
    assert annotated["condition"] == "profile_occlusion"
    assert "profile" in annotated["conditions"]
    assert "occlusion" in annotated["conditions"]
    metadata = annotated["metadata"]
    assert metadata["hard_negative_bucket"] == "profile_occlusion"
    assert metadata["hard_negative_priority"] == 1
    assert metadata["hard_negative_weight"] == 5.0
    assert metadata["hard_negative_source_dataset"] == "wflw"
    assert isinstance(metadata["hard_negative_reason"], list)


def test_source_key_prefers_source_block() -> None:
    sample = {"source": {"dataset": "wflw", "image_id": "42"}, "image": "other.png"}
    assert hnm.source_key(sample) == ("wflw", "42")


def test_source_key_falls_back_to_image() -> None:
    sample = {"dataset": "cofw", "image": "a/b/c.png"}
    assert hnm.source_key(sample) == ("cofw", "a/b/c.png")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, 1.0),
        (3.0, 3.0),
        (5.0, 5.0),
        (9.0, 5.0),
        (0.0, 1.0),
        (-2.0, 1.0),
        (float("nan"), 1.0),
    ],
)
def test_clamp_hard_negative_weight(value: float, expected: float) -> None:
    assert hnm.clamp_hard_negative_weight(value) == expected
