#!/usr/bin/env python3
"""Tests for landmark resolver hard-condition taxonomy."""

from __future__ import annotations

from lib.landmarks.datasets.manifest_io import LandmarkSample
from lib.landmarks.ensemble.hard_condition_taxonomy import derive_hard_condition_taxonomy


def _sample(
    *,
    condition: str = "default",
    conditions: tuple[str, ...] = (),
    visibility: tuple[bool, ...] | None = None,
    metadata: dict[str, object] | None = None,
) -> LandmarkSample:
    return LandmarkSample(
        sample_id="sample",
        image="image.jpg",
        landmarks="landmarks.npy",
        dataset="test",
        condition=condition,
        conditions=conditions,
        visibility=visibility,
        metadata=metadata or {},
    )


def test_wflw_occlusion_and_runtime_pose_derive_profile_intersection() -> None:
    taxonomy = derive_hard_condition_taxonomy(
        _sample(
            condition="occlusion",
            conditions=("occlusion",),
            metadata={"attributes": {"occlusion": 1, "pose": 1}},
        ),
        runtime_bucket="profile_left",
        yaw_estimate=-61.0,
        roll_estimate=3.0,
    )

    assert taxonomy.condition == "profile_occlusion"
    assert taxonomy.runtime_bucket == "profile_left"
    assert "profile_occlusion" in taxonomy.hard_case_tags
    assert "large_yaw_occlusion" in taxonomy.hard_case_tags


def test_roll_profile_and_merl_visibility_derive_rolled_profile_intersection() -> None:
    visibility = ["visible"] * 68
    visibility[36:42] = ["self_occluded"] * 6
    taxonomy = derive_hard_condition_taxonomy(
        _sample(metadata={"visibility": visibility}),
        runtime_bucket="rolled_profile_right",
        yaw_estimate=66.0,
        roll_estimate=-35.0,
    )

    assert taxonomy.condition == "rolled_profile_occlusion"
    assert "single_eye_visible" in taxonomy.hard_case_tags
    assert "rolled_profile_occlusion" in taxonomy.hard_case_tags


def test_visibility_heuristic_derives_mouth_or_jaw_occluded() -> None:
    visibility = [True] * 68
    visibility[0] = False
    visibility[52] = False

    taxonomy = derive_hard_condition_taxonomy(
        _sample(visibility=tuple(visibility)),
        runtime_bucket="frontal",
    )

    assert taxonomy.condition == "mouth_or_jaw_occluded"
    assert taxonomy.hard_case_tags[:2] == ("mouth_or_jaw_occluded", "occlusion")
