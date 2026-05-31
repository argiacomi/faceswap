#!/usr/bin/env python3
"""Tests for bbox handling in GT-derived geometry metrics."""

from __future__ import annotations

import numpy as np

from lib.landmarks.evaluation.geometry_metrics import evaluate_geometry_sample


def _face_points() -> np.ndarray:
    """Return a simple non-degenerate 68-point face-like landmark set."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[0:17, 0] = np.linspace(180, 250, 17)
    points[0:17, 1] = 160 + 18 * np.sin(np.linspace(0, np.pi, 17))
    points[17:22, 0] = np.linspace(188, 208, 5)
    points[17:22, 1] = 118
    points[22:27, 0] = np.linspace(222, 242, 5)
    points[22:27, 1] = 118
    points[27:36, 0] = 215
    points[27:36, 1] = np.linspace(122, 145, 9)
    points[36:42, 0] = np.linspace(192, 206, 6)
    points[36:42, 1] = 128
    points[42:48, 0] = np.linspace(224, 238, 6)
    points[42:48, 1] = 128
    points[48:60, 0] = np.linspace(198, 232, 12)
    points[48:60, 1] = 158
    points[60:68, 0] = np.linspace(205, 225, 8)
    points[60:68, 1] = 158
    return points


def test_geometry_metrics_normalizes_xywh_bbox() -> None:
    """COFW-style x/y/width/height boxes should not crash ROI diagnostics."""
    truth = _face_points()
    prediction = truth.copy()

    metrics = evaluate_geometry_sample(
        prediction,
        truth,
        sample_id="cofw68/0001",
        dataset="cofw",
        condition="occlusion",
        bbox=(171.0, 86.0, 108.0, 110.0),
    )

    assert metrics.roi_diagnostics is not None
    assert metrics.roi_diagnostics.bbox_width == 108.0
    assert metrics.roi_diagnostics.bbox_height == 110.0
    assert metrics.points_outside_bbox == 0
