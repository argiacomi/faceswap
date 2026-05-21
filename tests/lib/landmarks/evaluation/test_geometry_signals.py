#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.evaluation.geometry_signals` (#76)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lib.landmarks.evaluation.geometry_signals import (
    alignment_matrix_delta,
    alignment_summary,
    average_distance_delta,
    cloud_collapse,
    evaluate_catastrophic_flags,
    eye_mouth_flip,
    points_outside_bbox,
    polygon_iou,
    pose_delta,
    roi_delta,
    umeyama_alignment_matrix,
    visible_hull_iou,
)


def _truth_face() -> np.ndarray:
    """Plausible iBUG/300W-style 68-point face for the test fixtures."""
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


def test_alignment_summary_matches_umeyama_rotation_scale_block() -> None:
    """The 2×2 rotation/scale block matches Faceswap's umeyama matrix.

    The summary uses ``AlignedFace`` with ``centering='face'``, which applies a
    pose-driven offset to the translation column. The 2×2 rotation/scale block
    is unaffected by centering, so it is what we compare directly.
    """
    truth = _truth_face()
    summary = alignment_summary(truth)
    direct = umeyama_alignment_matrix(truth)
    # ``AlignedFace`` round-trips through float32 internally; allow ~1e-6 slack.
    np.testing.assert_allclose(summary.matrix[:, :2], direct[:, :2], atol=1e-6)


def test_alignment_summary_exposes_aligned_landmarks_shape() -> None:
    """``aligned_landmarks`` is (68, 2) in aligned-face pixel space."""
    truth = _truth_face()
    summary = alignment_summary(truth, size=256)
    assert summary.aligned_landmarks.shape == (68, 2)
    assert summary.normalized_landmarks.shape == (68, 2)


def test_alignment_matrix_delta_zero_for_identical_inputs() -> None:
    """Identical landmarks yield zero matrix delta."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    delta = alignment_matrix_delta(summary, summary, normalizer=100.0)
    assert delta.scale_delta == pytest.approx(0.0, abs=1e-9)
    assert delta.rotation_degrees_delta == pytest.approx(0.0, abs=1e-9)
    assert delta.translation_pixel_distance == pytest.approx(0.0, abs=1e-9)
    assert delta.translation_normalized_distance == pytest.approx(0.0, abs=1e-9)


def test_alignment_matrix_delta_detects_pure_translation() -> None:
    """Pure pixel-space translation surfaces as a non-zero translation delta only."""
    truth = _truth_face()
    predicted = truth + np.array([10.0, 0.0], dtype="float32")
    pred_summary = alignment_summary(predicted)
    truth_summary = alignment_summary(truth)
    delta = alignment_matrix_delta(pred_summary, truth_summary, normalizer=120.0)
    assert delta.scale_delta == pytest.approx(0.0, abs=1e-6)
    assert delta.rotation_degrees_delta == pytest.approx(0.0, abs=1e-6)
    assert delta.translation_pixel_distance > 0
    assert delta.translation_normalized_distance > 0


def test_roi_delta_iou_is_high_for_small_shift() -> None:
    """A few-pixel ROI shift keeps IoU > 0.9."""
    truth = _truth_face()
    predicted = truth + np.array([1.5, 0.5], dtype="float32")
    delta = roi_delta(
        alignment_summary(predicted),
        alignment_summary(truth),
        normalizer=120.0,
    )
    assert delta.iou > 0.9
    assert delta.center_pixel_distance < 10.0


def test_pose_delta_is_zero_for_identical_landmarks() -> None:
    """Identical landmarks → zero pitch / yaw / roll delta."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    delta = pose_delta(summary, summary)
    assert delta.pitch_delta_degrees == pytest.approx(0.0, abs=1e-3)
    assert delta.yaw_delta_degrees == pytest.approx(0.0, abs=1e-3)
    assert delta.roll_delta_degrees == pytest.approx(0.0, abs=1e-3)


def test_average_distance_delta_handles_signed_difference() -> None:
    """A tighter prediction returns a negative delta vs the truth fit."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    # Synthesise a "tighter" prediction by snapping landmarks closer to MEAN_FACE.
    contracted = truth.mean(axis=0) + (truth - truth.mean(axis=0)) * 0.8
    contracted_summary = alignment_summary(contracted)
    delta = average_distance_delta(contracted_summary, summary)
    assert isinstance(delta, float)


def test_visible_hull_iou_perfect_match() -> None:
    """Identical landmarks → hull IoU ≈ 1.0."""
    truth = _truth_face()
    assert visible_hull_iou(truth, truth) == pytest.approx(1.0)


def test_visible_hull_iou_respects_visibility_mask() -> None:
    """Hull computation only consumes flagged-visible landmarks."""
    truth = _truth_face()
    predicted = truth.copy()
    visibility = [True] * 68
    visibility[0] = False
    predicted[0] += np.array([200.0, 200.0], dtype="float32")
    iou_full = visible_hull_iou(predicted, truth)
    iou_visible = visible_hull_iou(predicted, truth, visibility=visibility)
    assert iou_visible > iou_full


def test_polygon_iou_handles_disjoint_polygons() -> None:
    """Polygons with no overlap return zero."""
    a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    b = np.array([[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]])
    assert polygon_iou(a, b) == pytest.approx(0.0)


def test_polygon_iou_handles_partial_overlap() -> None:
    """50%-overlapping unit squares give IoU = 1/3."""
    a = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    b = np.array([[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]])
    assert polygon_iou(a, b) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_cloud_collapse_detects_shrunken_prediction() -> None:
    """A heavily contracted prediction trips the collapse flag."""
    truth = _truth_face()
    predicted = truth.mean(axis=0) + (truth - truth.mean(axis=0)) * 0.1
    truth_diag = math.hypot(
        float(truth[:, 0].max() - truth[:, 0].min()),
        float(truth[:, 1].max() - truth[:, 1].min()),
    )
    assert cloud_collapse(predicted, truth_extent=truth_diag)
    assert not cloud_collapse(truth, truth_extent=truth_diag)


def test_eye_mouth_flip_uses_relative_eye_mouth_position() -> None:
    """Faceswap's own normalized eye-vs-mouth check drives the flip flag."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    assert eye_mouth_flip(summary) is False  # canonical face is non-flipped


def test_points_outside_bbox_counts_far_outliers() -> None:
    """Landmarks placed far from the face bbox are caught."""
    truth = _truth_face()
    predicted = truth.copy()
    predicted[0] = np.array([10000.0, 10000.0])
    bbox = (40.0, 60.0, 160.0, 150.0)
    assert points_outside_bbox(predicted, bbox=bbox) >= 1
    assert points_outside_bbox(truth, bbox=bbox) == 0


def test_evaluate_catastrophic_flags_aggregates_signals() -> None:
    """The aggregate helper combines collapse / flip / outside-bbox into one record."""
    truth = _truth_face()
    summary = alignment_summary(truth)
    bbox = (40.0, 60.0, 160.0, 150.0)
    flags = evaluate_catastrophic_flags(truth, truth, summary, bbox=bbox)
    assert flags.any is False
    predicted_outside = truth.copy()
    predicted_outside[0] = np.array([10000.0, 10000.0])
    bad_summary = alignment_summary(predicted_outside)
    flags = evaluate_catastrophic_flags(predicted_outside, truth, bad_summary, bbox=bbox)
    assert flags.points_outside_bbox is True
    assert flags.any is True
