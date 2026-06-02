#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.evaluation.transform_alignment_cost`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lib.landmarks.evaluation.transform_alignment_cost import (
    TransformCostWeightsV3,
    transform_cost_v3,
    visible_landmark_indices,
    visible_subset_alignment_summary,
)


def _truth_face() -> np.ndarray:
    """Plausible iBUG/300W-style 68-point face for test fixtures."""
    points = np.zeros((68, 2), dtype="float64")
    points[0:17, 0] = np.linspace(40, 160, 17)
    points[0:17, 1] = 120 + 30 * np.sin(np.linspace(0, np.pi, 17))
    points[17:22, 0] = np.linspace(50, 90, 5)
    points[17:22, 1] = 70
    points[22:27, 0] = np.linspace(110, 150, 5)
    points[22:27, 1] = 70
    points[27:36, 0] = 100
    points[27:36, 1] = np.linspace(75, 110, 9)
    points[36:42, 0] = np.linspace(60, 80, 6)
    points[36:42, 1] = 85
    points[42:48, 0] = np.linspace(120, 140, 6)
    points[42:48, 1] = 85
    points[48:60, 0] = np.linspace(70, 130, 12)
    points[48:60, 1] = 130
    points[60:68, 0] = np.linspace(80, 120, 8)
    points[60:68, 1] = 130
    return points


def test_visible_landmark_indices_default_to_all_68() -> None:
    assert visible_landmark_indices(None) == tuple(range(68))


def test_visible_landmark_indices_respect_manifest_visibility() -> None:
    visibility = [False] * 68
    visibility[0] = True
    visibility[17:25] = [True] * 8

    indices = visible_landmark_indices(visibility)
    summary = visible_subset_alignment_summary(_truth_face(), visibility, min_visible_points=8)

    assert indices == (0, *range(17, 25))
    assert summary.visible_indices == indices
    assert summary.fit_indices == tuple(range(17, 25))


def test_visible_subset_alignment_summary_requires_enough_usable_core_points() -> None:
    visibility = [True] * 68
    visibility[17:68] = [False] * 51

    with pytest.raises(ValueError, match="visible-subset transform fit needs"):
        visible_subset_alignment_summary(_truth_face(), visibility)


def test_transform_cost_v3_zero_for_identical_landmarks() -> None:
    truth = _truth_face()

    cost = transform_cost_v3(truth, truth)

    assert cost.hard_invalid is False
    assert cost.center_delta == pytest.approx(0.0, abs=1e-9)
    assert cost.scale_delta == pytest.approx(0.0, abs=1e-9)
    assert cost.roll_delta_degrees == pytest.approx(0.0, abs=1e-9)
    assert cost.fit_delta == pytest.approx(0.0, abs=1e-9)
    assert cost.total_cost == pytest.approx(0.0, abs=1e-9)


def test_transform_cost_v3_center_delta_uses_gt_output_frame() -> None:
    truth = _truth_face()
    predicted = truth + np.asarray([8.0, 0.0], dtype="float64")

    cost_512 = transform_cost_v3(predicted, truth, size=512)
    cost_256 = transform_cost_v3(predicted, truth, size=256)

    assert cost_512.hard_invalid is False
    assert cost_512.center_delta > 0.0
    # The delta is normalized by aligned crop size, so changing output size
    # should not materially change the label.
    assert cost_256.center_delta == pytest.approx(cost_512.center_delta, rel=1e-6)


def test_transform_cost_v3_scale_delta_uses_log_ratio() -> None:
    truth = _truth_face()
    center = truth.mean(axis=0)
    predicted = center + (truth - center) * 1.10

    pred_summary = visible_subset_alignment_summary(predicted)
    truth_summary = visible_subset_alignment_summary(truth)
    expected = abs(math.log(pred_summary.summary.scale / truth_summary.summary.scale))
    cost = transform_cost_v3(predicted, truth)

    assert cost.scale_delta == pytest.approx(expected)


def test_transform_cost_v3_roll_delta_wraps_degrees() -> None:
    truth = _truth_face()
    angle = math.radians(7.5)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype="float64",
    )
    center = truth.mean(axis=0)
    predicted = (truth - center) @ rotation.T + center

    cost = transform_cost_v3(predicted, truth)

    assert cost.roll_delta_degrees == pytest.approx(7.5, abs=1e-3)


def test_transform_cost_v3_fit_delta_uses_output_frame_rms_regret() -> None:
    truth = _truth_face()
    predicted = truth.copy()
    # Distort visible landmarks non-rigidly so the similarity transform cannot
    # absorb all residual error.
    predicted[36:42, 1] += 5.0
    predicted[48:60, 1] -= 3.0

    cost_512 = transform_cost_v3(predicted, truth, size=512)
    cost_256 = transform_cost_v3(predicted, truth, size=256)

    assert cost_512.fit_delta > 0.0
    # fit_delta is RMS regret divided by aligned size, so it stays comparable
    # across output sizes.
    assert cost_256.fit_delta == pytest.approx(cost_512.fit_delta, rel=1e-6)


def test_transform_cost_v3_includes_soft_structural_penalty() -> None:
    truth = _truth_face()
    penalty = 0.25

    cost = transform_cost_v3(truth, truth, soft_structural_penalty=penalty)

    assert cost.total_cost == pytest.approx(penalty)
    assert cost.soft_structural_penalty == pytest.approx(penalty)


def test_transform_cost_v3_applies_weights() -> None:
    truth = _truth_face()
    predicted = truth + np.asarray([8.0, 0.0], dtype="float64")
    weights = TransformCostWeightsV3(center=2.0, scale=3.0, roll=4.0, fit=5.0)

    cost = transform_cost_v3(predicted, truth, weights=weights)

    expected = (
        weights.center * cost.center_delta
        + weights.scale * cost.scale_delta
        + weights.roll * cost.roll_delta_degrees
        + weights.fit * cost.fit_delta
    )
    assert math.isfinite(cost.total_cost)
    assert cost.total_cost == pytest.approx(expected)


def test_visibility_gates_transform_fit_for_candidate_and_gt() -> None:
    truth = _truth_face()
    predicted = truth.copy()
    predicted[17] += np.asarray([500.0, 500.0], dtype="float64")

    visible = [True] * 68
    visible[17] = False
    visible_summary = visible_subset_alignment_summary(predicted, visible)
    truth_visible_summary = visible_subset_alignment_summary(truth, visible)
    full_summary = visible_subset_alignment_summary(predicted, None)
    truth_full_summary = visible_subset_alignment_summary(truth, None)

    visible_cost = transform_cost_v3(predicted, truth, visibility=visible)
    full_cost = transform_cost_v3(predicted, truth)

    assert 17 not in visible_summary.fit_indices
    assert visible_summary.visible_indices == truth_visible_summary.visible_indices
    assert visible_summary.fit_indices == truth_visible_summary.fit_indices
    assert visible_summary.summary.translation == pytest.approx(
        truth_visible_summary.summary.translation,
        abs=1e-6,
    )
    assert full_summary.summary.translation != pytest.approx(
        truth_full_summary.summary.translation,
        abs=1e-6,
    )
    assert visible_cost.total_cost < full_cost.total_cost


def test_transform_cost_v3_marks_invalid_fit_as_hard_invalid() -> None:
    truth = _truth_face()
    visibility = [False] * 68
    visibility[17:19] = [True, True]

    cost = transform_cost_v3(truth, truth, visibility=visibility)

    assert cost.hard_invalid is True
    assert cost.total_cost == float("inf")
    assert cost.hard_invalid_reasons


def test_transform_cost_v3_preserves_explicit_hard_invalid_reasons() -> None:
    truth = _truth_face()

    cost = transform_cost_v3(
        truth,
        truth,
        hard_invalid_reasons=("cloud_collapse", "eye_mouth_flip"),
    )

    assert cost.hard_invalid is True
    assert cost.total_cost == float("inf")
    assert cost.hard_invalid_reasons == ("cloud_collapse", "eye_mouth_flip")
