#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.evaluation.roi_diagnostics` (#81)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.evaluation.geometry_signals import alignment_summary
from lib.landmarks.evaluation.roi_diagnostics import (
    evaluate_roi_diagnostics,
    landmarks_inside_polygon,
    landmarks_outside_bbox,
)


def _truth_face() -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
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


def test_bbox_dimensions_and_aspect_match_expected() -> None:
    """bbox_aspect_ratio = width/height, bbox_diagonal matches Pythagoras."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    diag = evaluate_roi_diagnostics(predicted_summary=summary, truth_landmarks=truth, bbox=_bbox())
    assert diag.bbox_width == pytest.approx(120.0)
    assert diag.bbox_height == pytest.approx(90.0)
    assert diag.bbox_aspect_ratio == pytest.approx(120.0 / 90.0)
    assert diag.bbox_diagonal == pytest.approx(150.0)


def test_perfect_prediction_covers_visible_face() -> None:
    """A perfect prediction covers all landmarks and does not miss the visible face."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    diag = evaluate_roi_diagnostics(predicted_summary=summary, truth_landmarks=truth, bbox=_bbox())
    assert diag.landmarks_inside_aligned_crop_fraction == pytest.approx(1.0)
    assert diag.aligned_crop_misses_visible_face is False


def test_outside_bbox_landmark_is_counted() -> None:
    """Landmarks placed outside the detector bbox are reported as out-of-frame."""
    truth = _truth_face().copy()
    truth[0] = np.array([5.0, 5.0], dtype="float32")  # well outside bbox
    summary = alignment_summary(truth)
    diag = evaluate_roi_diagnostics(predicted_summary=summary, truth_landmarks=truth, bbox=_bbox())
    assert diag.landmarks_outside_detector_bbox_fraction > 0.0


def test_bad_alignment_flags_aligned_crop_misses_visible_face() -> None:
    """A misaligned prediction's ROI no longer covers most of the GT landmarks."""
    truth = _truth_face()
    # Shift the predicted landmarks dramatically so the AlignedFace ROI moves
    # off the GT cluster.
    predicted = truth + np.array([200.0, 200.0], dtype="float32")
    summary = alignment_summary(predicted)
    diag = evaluate_roi_diagnostics(
        predicted_summary=summary,
        truth_landmarks=truth,
        bbox=_bbox(),
        coverage_floor=0.85,
    )
    assert diag.landmarks_inside_aligned_crop_fraction < 0.85
    assert diag.aligned_crop_misses_visible_face is True


def test_landmarks_inside_polygon_handles_a_simple_square() -> None:
    """Convex polygon counts work for a canonical CCW square."""
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    landmarks = np.array([[0.5, 0.5], [-1.0, 0.5], [0.5, 1.5], [0.9, 0.9]], dtype="float32")
    assert landmarks_inside_polygon(landmarks, polygon) == 2


def test_landmarks_inside_polygon_handles_clockwise_square_and_edges() -> None:
    """Clockwise convex polygons and edge points are treated as inside."""
    polygon = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
    landmarks = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [2.1, 1.0],
        ],
        dtype="float32",
    )
    assert landmarks_inside_polygon(landmarks, polygon) == 3


def test_landmarks_inside_polygon_handles_rotated_convex_quad() -> None:
    """The vectorized convex sign test works for rotated ROI-like quads."""
    polygon = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    landmarks = np.array(
        [
            [1.0, 1.0],
            [0.5, 1.0],
            [1.5, 1.0],
            [1.0, 2.2],
        ],
        dtype="float32",
    )
    assert landmarks_inside_polygon(landmarks, polygon) == 3


def test_landmarks_outside_bbox_counts_violations() -> None:
    """Out-of-bbox landmarks are counted by both axes."""
    # bbox is (40, 60, 160, 150); first and third points are clearly outside,
    # second is inside (50, 80) but well within the bbox.
    landmarks = np.array([[5.0, 5.0], [50.0, 80.0], [200.0, 50.0]], dtype="float32")
    assert landmarks_outside_bbox(landmarks, _bbox()) == 2


def test_evaluate_roi_diagnostics_rejects_degenerate_bbox() -> None:
    """A zero-width/height bbox raises before any further computation."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    with pytest.raises(ValueError):
        evaluate_roi_diagnostics(
            predicted_summary=summary,
            truth_landmarks=truth,
            bbox=(40.0, 60.0, 40.0, 150.0),
        )


def test_roi_diagnostics_thread_through_evaluate_geometry_sample() -> None:
    """``evaluate_geometry_sample`` populates ``roi_diagnostics`` when bbox is given."""
    from lib.landmarks.evaluation.geometry_metrics import evaluate_geometry_sample

    truth = _truth_face()
    metrics = evaluate_geometry_sample(truth.copy(), truth, sample_id="perfect", bbox=_bbox())
    assert metrics.roi_diagnostics is not None
    assert metrics.roi_diagnostics.landmarks_inside_aligned_crop_fraction == pytest.approx(1.0)


def test_aggregate_geometry_samples_reports_roi_fields() -> None:
    """The aggregate exposes mean hull IoU, coverage, and miss rate when ROI is present."""
    from lib.landmarks.evaluation.geometry_metrics import (
        aggregate_geometry_samples,
        evaluate_geometry_sample,
    )

    truth = _truth_face()
    good = evaluate_geometry_sample(truth, truth, sample_id="good", bbox=_bbox())
    bad = evaluate_geometry_sample(truth + 80.0, truth, sample_id="bad", bbox=_bbox())
    aggregate = aggregate_geometry_samples("hrnet", [good, bad])
    assert aggregate.mean_landmarks_inside_aligned_crop_fraction < 1.0
    assert aggregate.aligned_crop_miss_rate > 0.0
    assert aggregate.mean_bbox_aspect_ratio == pytest.approx(120.0 / 90.0)
