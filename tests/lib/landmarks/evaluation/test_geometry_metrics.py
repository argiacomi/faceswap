#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.evaluation.geometry_metrics` (#76)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.evaluation.geometry_metrics import (
    GEOMETRY_OBJECTIVE,
    REGION_DEFINITIONS,
    aggregate_geometry_samples,
    evaluate_geometry_sample,
    score_alignment_geometry_v1,
)
from lib.landmarks.evaluation.geometry_signals import alignment_summary


def _truth_face() -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")
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


def _bbox() -> tuple[float, float, float, float]:
    return (40.0, 60.0, 160.0, 150.0)


def test_geometry_objective_constant_is_v1() -> None:
    """Pinned for downstream artifact contracts."""
    assert GEOMETRY_OBJECTIVE == "alignment_geometry_v1"


def test_perfect_prediction_scores_zero() -> None:
    """Identical landmarks yield a perfect zero geometry score."""
    truth = _truth_face()
    metrics = evaluate_geometry_sample(
        truth.copy(),
        truth,
        sample_id="perfect",
        bbox=_bbox(),
    )
    assert metrics.overall_score == pytest.approx(0.0, abs=1e-6)
    assert metrics.catastrophic_flags.any is False
    assert metrics.roi_delta.iou == pytest.approx(1.0, abs=1e-6)
    assert metrics.hull_iou == pytest.approx(1.0, abs=1e-3)
    for region in REGION_DEFINITIONS:
        assert metrics.per_region_error[region] == pytest.approx(0.0, abs=1e-3)


def test_shrunken_prediction_has_lower_roi_iou_and_higher_score() -> None:
    """A scaled-down prediction increases the score and drops ROI IoU."""
    truth = _truth_face()
    shrunk = truth.mean(axis=0) + (truth - truth.mean(axis=0)) * 0.5
    metrics = evaluate_geometry_sample(
        shrunk,
        truth,
        sample_id="shrunk",
        bbox=_bbox(),
    )
    assert metrics.overall_score > 0.5
    assert metrics.roi_delta.iou < 0.7


def test_per_region_error_is_zero_for_identical_landmarks() -> None:
    """Per-region geometry error vanishes when predictions match GT exactly."""
    truth = _truth_face()
    metrics = evaluate_geometry_sample(truth, truth, sample_id="identical", bbox=_bbox())
    for value in metrics.per_region_error.values():
        assert value == pytest.approx(0.0, abs=1e-3)


def test_cached_truth_summary_matches_uncached_geometry_metrics() -> None:
    """Precomputing the GT-side summary must not change geometry outputs."""
    truth = _truth_face()
    predicted = truth + np.array([3.0, -2.0], dtype="float32")
    cached_summary = alignment_summary(truth)

    uncached = evaluate_geometry_sample(
        predicted,
        truth,
        sample_id="uncached",
        dataset="ds",
        condition="clean",
        bbox=_bbox(),
    ).to_payload()
    cached = evaluate_geometry_sample(
        predicted,
        truth,
        sample_id="cached",
        dataset="ds",
        condition="clean",
        bbox=_bbox(),
        truth_summary=cached_summary,
    ).to_payload()

    uncached.pop("sample_id")
    cached.pop("sample_id")
    assert cached == uncached


def test_catastrophic_flag_trips_for_outside_bbox_outlier() -> None:
    """A single far-outside-bbox landmark trips the catastrophic flag."""
    truth = _truth_face()
    predicted = truth.copy()
    predicted[0] = np.array([10000.0, 10000.0])
    metrics = evaluate_geometry_sample(predicted, truth, sample_id="outlier", bbox=_bbox())
    assert metrics.catastrophic_flags.any is True
    assert metrics.catastrophic_flags.points_outside_bbox is True
    # Catastrophic adds a full 1.0 penalty; overall score must reflect that.
    assert metrics.overall_score >= 1.0


def test_aggregate_geometry_samples_reports_percentiles_and_per_bucket() -> None:
    """Aggregation surfaces P95 transform/roll, per-region rates, and per-bucket scores."""
    truth = _truth_face()
    good = evaluate_geometry_sample(
        truth.copy(),
        truth,
        sample_id="good",
        dataset="ds",
        condition="clean",
        bbox=_bbox(),
    )
    shifted = evaluate_geometry_sample(
        truth + 5.0,
        truth,
        sample_id="shifted",
        dataset="ds",
        condition="clean",
        bbox=_bbox(),
    )
    profile = evaluate_geometry_sample(
        truth.mean(axis=0) + (truth - truth.mean(axis=0)) * 0.7,
        truth,
        sample_id="profile",
        dataset="ds",
        condition="profile",
        bbox=_bbox(),
    )

    aggregate = aggregate_geometry_samples("hrnet", [good, shifted, profile])

    assert aggregate.sample_count == 3
    assert aggregate.catastrophic_failure_rate == pytest.approx(0.0)
    assert aggregate.mean_translation_normalized >= 0
    assert "ds:clean" in aggregate.per_bucket
    assert "ds:profile" in aggregate.per_bucket
    assert aggregate.p95_translation_normalized >= aggregate.mean_translation_normalized
    for region in REGION_DEFINITIONS:
        assert region in aggregate.per_region_error


def test_score_alignment_geometry_v1_catastrophic_adds_penalty() -> None:
    """The composite score adds a full 1.0 penalty when catastrophic=True."""
    base = score_alignment_geometry_v1(
        translation_normalized=0.0,
        relative_scale_delta=0.0,
        rotation_degrees_delta=0.0,
        roi_iou=1.0,
        hull_iou=1.0,
        catastrophic=False,
    )
    with_cat = score_alignment_geometry_v1(
        translation_normalized=0.0,
        relative_scale_delta=0.0,
        rotation_degrees_delta=0.0,
        roi_iou=1.0,
        hull_iou=1.0,
        catastrophic=True,
    )
    assert base == pytest.approx(0.0)
    assert with_cat == pytest.approx(1.0)


def test_score_combines_components_in_expected_order() -> None:
    """Roll, translation, scale, IoU drops all contribute to the score."""
    score = score_alignment_geometry_v1(
        translation_normalized=0.01,
        relative_scale_delta=0.02,
        rotation_degrees_delta=2.0,
        roi_iou=0.95,
        hull_iou=0.90,
        catastrophic=False,
    )
    # Hand-computed: 5*0.01 + 1*0.02 + 0.02*2 + 0.5*0.05 + 0.5*0.10
    expected = 0.05 + 0.02 + 0.04 + 0.025 + 0.05
    assert score == pytest.approx(expected, rel=1e-9)


def test_geometry_evaluation_uses_visible_mask_for_hull() -> None:
    """Visibility filtering for hull IoU still produces a valid result."""
    truth = _truth_face()
    predicted = truth.copy()
    predicted[0] = np.array([10000.0, 10000.0])  # one absurd outlier
    visibility = [True] * 68
    visibility[0] = False
    metrics = evaluate_geometry_sample(
        predicted,
        truth,
        sample_id="masked",
        bbox=_bbox(),
        visibility=visibility,
    )
    # With visibility masking the hull IoU should stay high for the remaining
    # 67 visible landmarks.
    assert metrics.hull_iou > 0.5
