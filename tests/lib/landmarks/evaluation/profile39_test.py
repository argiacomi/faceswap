#!/usr/bin/env python3
"""Tests for partial-schema (39-point profile) scoring."""

from __future__ import annotations

import types

import numpy as np
import pytest

from lib.landmarks.evaluation import profile39


def _candidate(name: str, landmarks: np.ndarray) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, landmarks=landmarks)


def _face68(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    face: np.ndarray = rng.uniform(0.0, 100.0, size=(68, 2)).astype("float32")
    return face


# --- map integrity -------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_map_has_39_valid_indices(side: str) -> None:
    idx = profile39.PROFILE39_TO_CANONICAL68[side]
    assert len(idx) == profile39.PROFILE39_POINT_COUNT == 39
    assert all(0 <= i < 68 for i in idx)


# --- projection ----------------------------------------------------------


def test_projection_shape_and_values() -> None:
    face = _face68()
    pred39 = profile39.project_68_to_profile39(face, side="left")
    assert pred39.shape == (39, 2)
    expected = face[np.asarray(profile39.PROFILE39_TO_CANONICAL68["left"])]
    assert np.allclose(pred39, expected)


def test_projection_rejects_bad_shape_and_side() -> None:
    with pytest.raises(ValueError):
        profile39.project_68_to_profile39(np.zeros((39, 2), dtype="float32"), side="left")
    with pytest.raises(ValueError):
        profile39.project_68_to_profile39(_face68(), side="up")


# --- error / cost --------------------------------------------------------


def test_point_error_zero_when_pred_matches_projection() -> None:
    face = _face68()
    truth39 = profile39.project_68_to_profile39(face, side="right")
    err = profile39.profile39_point_error(face, truth39, side="right", normalizer=100.0)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_transform_cost_zero_when_pred_matches() -> None:
    face = _face68()
    truth39 = profile39.project_68_to_profile39(face, side="left")
    cost = profile39.profile39_transform_cost(face, truth39, side="left")
    assert cost["profile39_transform_cost"] == pytest.approx(0.0, abs=1e-6)
    assert cost["profile39_fit_delta"] == pytest.approx(0.0, abs=1e-6)


def test_transform_cost_increases_with_perturbation() -> None:
    face = _face68()
    truth39 = profile39.project_68_to_profile39(face, side="left")
    perturbed = face.copy()
    perturbed[profile39.PROFILE39_TO_CANONICAL68["left"][0]] += 25.0
    good = profile39.profile39_transform_cost(face, truth39, side="left")
    bad = profile39.profile39_transform_cost(perturbed, truth39, side="left")
    assert bad["profile39_transform_cost"] > good["profile39_transform_cost"]


def test_transform_cost_rejects_bad_truth_shape() -> None:
    with pytest.raises(ValueError):
        profile39.profile39_transform_cost(_face68(), np.zeros((68, 2), "float32"), side="left")


# --- side resolution -----------------------------------------------------


def test_side_from_condition_labels() -> None:
    assert profile39.profile39_side_from_sample({"condition": "profile_left"}) == "left"
    assert profile39.profile39_side_from_sample({"conditions": ["profile_right"]}) == "right"


def test_side_from_metadata_yaw_side() -> None:
    assert profile39.profile39_side_from_sample({"metadata": {"yaw_side": "Left"}}) == "left"


def test_side_unknown_returns_none() -> None:
    assert profile39.profile39_side_from_sample({"condition": "occlusion"}) is None
    # ambiguous (both left and right) -> skip
    assert profile39.profile39_side_from_sample({"conditions": ["left", "right"]}) is None


# --- oracle / regret rows ------------------------------------------------


def test_rows_produce_oracle_and_nonnegative_regret() -> None:
    face = _face68(1)
    truth39 = profile39.project_68_to_profile39(face, side="left")
    good = _candidate("good", face)  # exact match -> cost 0 -> oracle
    worse = face.copy()
    worse[profile39.PROFILE39_TO_CANONICAL68["left"][5]] += 40.0
    bad = _candidate("bad", worse)
    rows = profile39.profile39_rows([good, bad], truth39, side="left", normalizer=100.0)
    by_name = {row["candidate_name"]: row for row in rows}
    assert by_name["good"]["profile39_is_oracle"] is True
    assert by_name["good"]["profile39_oracle_candidate"] == "good"
    assert by_name["good"]["profile39_transform_regret"] == pytest.approx(0.0, abs=1e-6)
    assert by_name["bad"]["profile39_transform_regret"] > 0.0
    assert all(row["profile39_side"] == "left" for row in rows)


def test_candidate_costs_skip_non_68_candidates() -> None:
    face = _face68(2)
    truth39 = profile39.project_68_to_profile39(face, side="right")
    valid = _candidate("v", face)
    bad_shape = _candidate("b", np.zeros((39, 2), dtype="float32"))
    costs = profile39.profile39_candidate_costs([valid, bad_shape], truth39, side="right")
    assert set(costs) == {"v"}
