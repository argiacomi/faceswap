#!/usr/bin/env python3
"""Tests for 68-point shape plausibility diagnostics."""

from __future__ import annotations

import numpy as np

from lib.landmarks.evaluation.shape_plausibility import evaluate_shape_plausibility


def _face() -> np.ndarray:
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


def _set_nondegenerate_mouth(points: np.ndarray) -> np.ndarray:
    points = points.copy()
    points[48:60] = np.asarray(
        [
            (70, 130),
            (78, 124),
            (88, 121),
            (100, 120),
            (112, 121),
            (122, 124),
            (130, 130),
            (122, 136),
            (112, 140),
            (100, 141),
            (88, 140),
            (78, 136),
        ],
        dtype="float32",
    )
    points[60:68] = np.asarray(
        [
            (82, 130),
            (90, 127),
            (100, 126),
            (110, 127),
            (118, 130),
            (110, 134),
            (100, 135),
            (90, 134),
        ],
        dtype="float32",
    )
    return points


def test_shape_plausibility_accepts_ordered_68_point_shape() -> None:
    plausibility = evaluate_shape_plausibility(_face())

    assert plausibility.severe is False
    assert plausibility.reasons == ()
    assert plausibility.metrics["topology_violation_count"] == 0.0


def test_shape_plausibility_accepts_nondegenerate_mouth_polygon() -> None:
    plausibility = evaluate_shape_plausibility(_set_nondegenerate_mouth(_face()))

    assert plausibility.severe is False
    assert "inner_mouth_outside_outer_mouth" not in plausibility.reasons
    assert plausibility.metrics["inner_mouth_outside_fraction"] == 0.0


def test_shape_plausibility_vetoes_scrambled_local_topology() -> None:
    landmarks = _face()
    landmarks[37] = landmarks[8]

    plausibility = evaluate_shape_plausibility(landmarks)

    assert plausibility.severe is True
    assert "edge_length_extreme" in plausibility.reasons
    assert plausibility.metrics["max_edge_length_ratio"] > 1.0
