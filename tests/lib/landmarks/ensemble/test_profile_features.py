#!/usr/bin/env python3
"""Tests for profile/occlusion routing and visible-side candidate features (#218)."""

from __future__ import annotations

import types
from typing import cast

import numpy as np
import pytest

from lib.landmarks.ensemble import profile_features as pf
from lib.landmarks.ensemble import profile_routing as pr


def _candidate(name: str, landmarks: np.ndarray) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, landmarks=landmarks, is_fusion=False)


def _metric(veto: tuple[str, ...] = ()) -> types.SimpleNamespace:
    return types.SimpleNamespace(geometry_veto_reasons=veto)


def _base_face() -> np.ndarray:
    rng = np.random.default_rng(0)
    return cast(np.ndarray, rng.uniform(0.0, 100.0, size=(68, 2)))


# --- routing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"runtime_bucket": "profile_left"}, True),
        ({"runtime_bucket": "large_yaw_right"}, True),
        ({"condition": "occlusion"}, True),
        ({"runtime_bucket": "rolled_profile_left"}, True),
        ({"runtime_bucket": "frontal"}, False),
        ({"runtime_bucket": "intermediate", "condition": "normal"}, False),
    ],
)
def test_is_profile_or_occlusion_context(tags: dict, expected: bool) -> None:
    context = types.SimpleNamespace(hard_case_tags=(), **tags)
    assert pr.is_profile_or_occlusion_context(context) is expected


def test_scorer_route_for_context() -> None:
    assert pr.scorer_route_for_context("profile_left") == pr.SCORER_POLICY_PROFILE
    assert pr.scorer_route_for_context("occlusion") == pr.SCORER_POLICY_PROFILE
    assert pr.scorer_route_for_context("frontal") == pr.SCORER_POLICY_GENERAL


def test_condition_tags_accepts_tuple_and_mapping() -> None:
    assert pr.condition_tags(("Profile_Left", "Occlusion")) == ("profile_left", "occlusion")
    assert pr.condition_tags({"condition": "Frontal", "runtime_bucket": "no_pose"}) == (
        "frontal",
        "no_pose",
    )


# --- side inference ------------------------------------------------------


def test_profile_side_from_bucket_labels() -> None:
    assert pf.profile_side_from_context(runtime_bucket="profile_left") == "left"
    assert pf.profile_side_from_context(runtime_bucket="large_yaw_right") == "right"


def test_profile_side_from_yaw_estimate() -> None:
    assert pf.profile_side_from_context(yaw_estimate=-30.0) == "left"
    assert pf.profile_side_from_context(yaw_estimate=30.0) == "right"
    assert pf.profile_side_from_context(yaw_estimate=0.0) == ""


def test_visible_and_occluded_indices_are_disjoint() -> None:
    assert set(pf.visible_side_indices("left")) == set(pf.LEFT_SIDE_68)
    assert set(pf.occluded_side_indices("left")) == set(pf.RIGHT_SIDE_68)
    assert not set(pf.visible_side_indices("left")) & set(pf.occluded_side_indices("left"))


# --- features ------------------------------------------------------------


def test_frontal_context_emits_no_profile_features() -> None:
    candidates = [_candidate("hrnet", _base_face())]
    metrics = {"hrnet": _metric()}
    out = pf.profile_candidate_features(
        candidates,
        metrics,
        diag=100.0,
        side="",
        yaw_estimate=0.0,
        roll_estimate=0.0,
        has_occlusion=False,
    )
    assert out == {}


def test_profile_context_emits_features_for_each_candidate() -> None:
    base = _base_face()
    shifted = base.copy()
    shifted[pf.RIGHT_SIDE_68, :] += 8.0  # perturb the (occluded) right side
    candidates = [_candidate("hrnet", base), _candidate("spiga", shifted)]
    metrics = {"hrnet": _metric(), "spiga": _metric()}
    out = pf.profile_candidate_features(
        candidates,
        metrics,
        diag=100.0,
        side="left",
        yaw_estimate=-40.0,
        roll_estimate=5.0,
        has_occlusion=True,
    )
    assert set(out) == {"hrnet", "spiga"}
    for name in ("hrnet", "spiga"):
        for feature in pf.PROFILE_FEATURE_NAMES:
            assert feature in out[name]
            assert np.isfinite(out[name][feature])
    # Face-level flags reflect the route.
    assert out["hrnet"]["profile_is_left"] == 1.0
    assert out["hrnet"]["profile_is_large_yaw"] == 1.0
    assert out["hrnet"]["profile_has_occlusion"] == 1.0
    assert out["hrnet"]["profile_yaw_signed"] == -40.0


def test_occluded_side_perturbation_raises_occluded_spread() -> None:
    base = _base_face()
    shifted = base.copy()
    shifted[pf.RIGHT_SIDE_68, :] += 20.0
    candidates = [
        _candidate("a", base),
        _candidate("b", base.copy()),
        _candidate("c", shifted),
    ]
    metrics = {name: _metric() for name in ("a", "b", "c")}
    out = pf.profile_candidate_features(
        candidates,
        metrics,
        diag=100.0,
        side="left",
        yaw_estimate=-40.0,
        roll_estimate=0.0,
        has_occlusion=True,
    )
    # Candidate c diverges on the occluded (right) side from the consensus.
    assert out["c"]["occluded_side_spread"] > out["a"]["occluded_side_spread"]
