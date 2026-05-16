#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.eval.profile_metrics` (issue #76)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.eval.profile_metrics import (
    DEFAULT_PCK_THRESHOLDS,
    DEFAULT_REGION_WEIGHTS,
    PROFILE_OBJECTIVE,
    REGION_INDICES,
    aggregate_profile_samples,
    evaluate_profile_sample,
    normalizer_from_bbox,
)


def _truth_points() -> np.ndarray:
    return np.stack(
        (
            np.linspace(0.0, 100.0, 68, dtype="float32"),
            np.linspace(0.0, 200.0, 68, dtype="float32"),
        ),
        axis=1,
    )


def _bbox() -> tuple[float, float, float, float]:
    truth = _truth_points()
    return (
        float(truth[:, 0].min()),
        float(truth[:, 1].min()),
        float(truth[:, 0].max()),
        float(truth[:, 1].max()),
    )


def test_normalizer_from_bbox_diagonal() -> None:
    """Diagonal normalizer uses ``sqrt(w^2 + h^2)``."""
    assert normalizer_from_bbox((0.0, 0.0, 3.0, 4.0)) == pytest.approx(5.0)


def test_normalizer_from_bbox_sqrt_area() -> None:
    """``bbox_sqrt_area`` returns ``sqrt(w * h)``."""
    assert normalizer_from_bbox((0.0, 0.0, 10.0, 10.0), method="bbox_sqrt_area") == pytest.approx(
        10.0
    )


def test_normalizer_rejects_zero_area_bbox() -> None:
    """Degenerate bboxes raise rather than producing a divide-by-zero downstream."""
    with pytest.raises(ValueError):
        normalizer_from_bbox((0.0, 0.0, 0.0, 4.0))


def test_perfect_prediction_scores_zero() -> None:
    """A perfect prediction has zero region error and zero overall score."""
    truth = _truth_points()
    metrics = evaluate_profile_sample(
        truth.copy(),
        truth,
        sample_id="perfect",
        face_bbox=_bbox(),
    )
    assert metrics.overall_score == pytest.approx(0.0)
    assert metrics.p90_visible_error == pytest.approx(0.0)
    for value in metrics.per_region_error.values():
        assert value == pytest.approx(0.0)
    for flag in metrics.region_failures.values():
        assert flag is False


def test_localized_jaw_failure_flags_region_and_drives_p90() -> None:
    """A large jaw shift triggers a jaw region failure and inflates p90."""
    truth = _truth_points()
    predicted = truth.copy()
    predicted[REGION_INDICES["visible_jaw"], :] += 30.0  # ~13.4% of bbox diagonal

    metrics = evaluate_profile_sample(
        predicted,
        truth,
        sample_id="jaw_fail",
        face_bbox=_bbox(),
    )

    assert metrics.region_failures["visible_jaw"] is True
    assert metrics.region_failures["nose"] is False
    assert metrics.p90_visible_error > 0.05
    assert metrics.overall_score > 0.0


def test_visibility_excludes_non_visible_jaw_points() -> None:
    """Non-visible jaw landmarks must not count toward the jaw region error."""
    truth = _truth_points()
    predicted = truth.copy()
    # Inject a catastrophic error only on the first jaw point.
    predicted[0, :] += 100.0
    visibility = [True] * 68
    visibility[0] = False

    metrics = evaluate_profile_sample(
        predicted,
        truth,
        sample_id="masked",
        face_bbox=_bbox(),
        visibility=visibility,
    )

    # Visible jaw points are still perfectly aligned.
    assert metrics.per_region_error["visible_jaw"] == pytest.approx(0.0)
    assert metrics.visible_landmark_count == 67


def test_evaluate_profile_sample_rejects_visibility_length_mismatch() -> None:
    """Visibility arrays of the wrong length raise a clear error."""
    truth = _truth_points()
    with pytest.raises(ValueError, match="visibility length"):
        evaluate_profile_sample(
            truth.copy(),
            truth,
            sample_id="bad",
            face_bbox=_bbox(),
            visibility=[True] * 10,
        )


def test_aggregate_profile_samples_weighted_and_failure_rate() -> None:
    """Aggregation averages region errors and reports priority-region failure rate."""
    truth = _truth_points()
    good = evaluate_profile_sample(
        truth.copy(), truth, sample_id="good", face_bbox=_bbox()
    )
    bad_predicted = truth.copy()
    bad_predicted[REGION_INDICES["visible_jaw"], :] += 50.0
    bad_predicted[REGION_INDICES["nose"], :] += 60.0
    bad = evaluate_profile_sample(
        bad_predicted, truth, sample_id="bad", face_bbox=_bbox()
    )

    aggregate = aggregate_profile_samples("hrnet", [good, bad])

    assert aggregate.sample_count == 2
    # Priority failure rate: jaw + nose failed on 1 of 2 samples → 0.25 (half of two regions).
    assert aggregate.region_failure_rate == pytest.approx(0.25)
    assert aggregate.overall_score > 0.0
    for key in (f"{t:.2f}" for t in DEFAULT_PCK_THRESHOLDS):
        assert 0.0 <= aggregate.pck_at[key] <= 1.0


def test_profile_objective_constant_is_v1() -> None:
    """The objective name is part of the artifact contract; pin it."""
    assert PROFILE_OBJECTIVE == "profile_alignment_v1"


def test_region_weights_sum_to_one() -> None:
    """Default region weights form a probability distribution."""
    total = sum(DEFAULT_REGION_WEIGHTS.values())
    assert total == pytest.approx(1.0)


def test_visible_eye_region_drops_non_visible_eye_landmarks() -> None:
    """``visible_eye`` shrinks when half the eye landmarks are flagged non-visible."""
    truth = _truth_points()
    predicted = truth.copy()
    eye_indices = REGION_INDICES["visible_eye"]
    visibility = [True] * 68
    half = len(eye_indices) // 2
    for idx in eye_indices[:half]:
        visibility[idx] = False
        predicted[idx, :] += 60.0  # would inflate visible_eye error if counted

    metrics = evaluate_profile_sample(
        predicted,
        truth,
        sample_id="half_eye_visible",
        face_bbox=_bbox(),
        visibility=visibility,
    )

    # Only the visible half of the eye contributes; those points are perfect.
    assert metrics.per_region_error["visible_eye"] == pytest.approx(0.0)
