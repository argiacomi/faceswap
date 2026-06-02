#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.evaluation.transform_alignment_cost`."""

from __future__ import annotations

import math
import typing as T

import numpy as np
import pytest

from lib.landmarks.evaluation.transform_alignment_cost import (
    DEFAULT_SOFT_STRUCTURAL_PENALTY_V3,
    TransformCostWeightsV3,
    structural_validity_v3,
    transform_cost_v3,
    visible_landmark_indices,
    visible_subset_alignment_summary,
)


def _truth_face() -> np.ndarray:
    """Plausible iBUG/300W-style 68-point face for test fixtures."""
    points: np.ndarray = np.zeros((68, 2), dtype="float64")
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


def _alignment_crop_center(points: np.ndarray) -> np.ndarray:
    """Return the fitted source-frame crop center for stable pure transforms."""
    return T.cast(np.ndarray, visible_subset_alignment_summary(points).summary.roi.mean(axis=0))


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


def test_transform_cost_v3_translation_only_changes_center_delta() -> None:
    truth = _truth_face()
    predicted = truth + np.asarray([8.0, 0.0], dtype="float64")

    cost_512 = transform_cost_v3(predicted, truth, size=512)
    cost_256 = transform_cost_v3(predicted, truth, size=256)

    assert cost_512.hard_invalid is False
    assert cost_512.center_delta > 0.0
    assert cost_512.scale_delta == pytest.approx(0.0, abs=1e-9)
    assert cost_512.roll_delta_degrees == pytest.approx(0.0, abs=1e-9)
    assert cost_512.fit_delta == pytest.approx(0.0, abs=1e-9)
    # The delta is normalized by aligned crop size, so changing output size
    # should not materially change the label.
    assert cost_256.center_delta == pytest.approx(cost_512.center_delta, rel=1e-6)


def test_transform_cost_v3_scale_only_changes_scale_delta() -> None:
    truth = _truth_face()
    center = _alignment_crop_center(truth)
    predicted = center + (truth - center) * 1.10

    pred_summary = visible_subset_alignment_summary(predicted)
    truth_summary = visible_subset_alignment_summary(truth)
    expected = abs(math.log(pred_summary.summary.scale / truth_summary.summary.scale))
    cost = transform_cost_v3(predicted, truth)

    assert cost.center_delta == pytest.approx(0.0, abs=1e-6)
    assert cost.scale_delta == pytest.approx(expected)
    assert cost.roll_delta_degrees == pytest.approx(0.0, abs=1e-6)
    assert cost.fit_delta == pytest.approx(0.0, abs=1e-6)


def test_transform_cost_v3_roll_only_changes_roll_delta() -> None:
    truth = _truth_face()
    angle_degrees = 7.5
    angle = math.radians(angle_degrees)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype="float64",
    )
    center = _alignment_crop_center(truth)
    predicted = (truth - center) @ rotation.T + center

    cost = transform_cost_v3(predicted, truth)

    assert cost.center_delta == pytest.approx(0.0, abs=1e-6)
    assert cost.scale_delta == pytest.approx(0.0, abs=1e-6)
    assert cost.roll_delta_degrees == pytest.approx(angle_degrees, abs=1e-3)
    assert cost.fit_delta == pytest.approx(0.0, abs=1e-6)


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


def test_soft_suspect_is_finite_additive_penalty() -> None:
    truth = _truth_face()

    cost = transform_cost_v3(
        truth,
        truth,
        soft_suspect_reasons=("low_plausibility", "mild_roi_warning"),
    )

    assert cost.hard_invalid is False
    assert cost.soft_suspect_reasons == ("low_plausibility", "mild_roi_warning")
    assert cost.soft_structural_penalty == pytest.approx(DEFAULT_SOFT_STRUCTURAL_PENALTY_V3)
    assert cost.total_cost == pytest.approx(DEFAULT_SOFT_STRUCTURAL_PENALTY_V3)


def test_explicit_soft_penalty_without_reason_becomes_soft_suspect() -> None:
    truth = _truth_face()

    cost = transform_cost_v3(truth, truth, soft_structural_penalty=0.25)

    assert cost.hard_invalid is False
    assert cost.soft_suspect_reasons == ("explicit_soft_structural_penalty",)
    assert cost.soft_structural_penalty == pytest.approx(0.25)
    assert cost.total_cost == pytest.approx(0.25)


def test_structural_validity_rejects_negative_soft_penalty() -> None:
    with pytest.raises(ValueError, match="soft_structural_penalty must be non-negative"):
        structural_validity_v3(soft_structural_penalty=-0.01)


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


def test_profile_hidden_landmarks_do_not_pollute_gt_visible_subset_transform() -> None:
    truth = _truth_face()
    polluted_truth = truth.copy()
    profile_hidden_indices = (
        *range(9, 17),
        *range(22, 27),
        *range(42, 48),
    )
    polluted_truth[list(profile_hidden_indices)] += np.asarray([500.0, -300.0], dtype="float64")

    visibility = [True] * 68
    for index in profile_hidden_indices:
        visibility[index] = False

    visible_summary = visible_subset_alignment_summary(polluted_truth, visibility)
    clean_visible_summary = visible_subset_alignment_summary(truth, visibility)
    visible_cost = transform_cost_v3(truth, polluted_truth, visibility=visibility)
    full_cost = transform_cost_v3(truth, polluted_truth)

    assert set(profile_hidden_indices).isdisjoint(visible_summary.fit_indices)
    assert visible_summary.summary.translation == pytest.approx(
        clean_visible_summary.summary.translation,
        abs=1e-6,
    )
    assert visible_summary.summary.scale == pytest.approx(clean_visible_summary.summary.scale)
    assert visible_summary.summary.rotation_degrees == pytest.approx(
        clean_visible_summary.summary.rotation_degrees,
        abs=1e-6,
    )
    assert visible_cost.total_cost == pytest.approx(0.0, abs=1e-9)
    assert full_cost.total_cost > visible_cost.total_cost


def test_transform_fit_failure_marks_hard_invalid_without_inf_cost() -> None:
    truth = _truth_face()
    visibility = [False] * 68
    visibility[17:19] = [True, True]

    cost = transform_cost_v3(truth, truth, visibility=visibility)

    assert cost.hard_invalid is True
    assert cost.total_cost == pytest.approx(0.0)
    assert cost.hard_invalid_reasons
    assert cost.hard_invalid_reasons[0].startswith("unable_to_fit_visible_subset_transform")


def test_explicit_hard_invalid_reasons_mark_exclusion_not_giant_cost() -> None:
    truth = _truth_face()

    cost = transform_cost_v3(
        truth,
        truth,
        hard_invalid_reasons=("cloud_collapse", "eye_mouth_flip"),
    )

    assert cost.hard_invalid is True
    assert cost.total_cost == pytest.approx(0.0)
    assert cost.hard_invalid_reasons == ("cloud_collapse", "eye_mouth_flip")


def test_hard_invalid_can_still_report_soft_suspect_diagnostics() -> None:
    truth = _truth_face()

    cost = transform_cost_v3(
        truth,
        truth,
        hard_invalid_reasons=("self_intersection",),
        soft_suspect_reasons=("borderline_hull_warning",),
    )

    assert cost.hard_invalid is True
    assert cost.total_cost == pytest.approx(0.0)
    assert cost.soft_structural_penalty == pytest.approx(DEFAULT_SOFT_STRUCTURAL_PENALTY_V3)
    assert cost.soft_suspect_reasons == ("borderline_hull_warning",)
