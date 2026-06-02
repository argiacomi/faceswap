#!/usr/bin/env python3
"""Regression tests for non-negative v3 fit-regret accounting."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.evaluation.transform_alignment_cost import transform_cost_v3


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


def test_fit_delta_does_not_reward_better_than_gt_mean_face_fit() -> None:
    """A candidate fitting the mean face better than GT gets no negative credit.

    ``fit_delta`` is a non-negative regret term, not a bonus.  Center, scale,
    and roll deltas still capture transform mismatch when a candidate overfits
    the mean face or when GT carries non-rigid annotation noise.
    """
    clean_candidate = _truth_face()
    noisy_gt = clean_candidate.copy()
    # Add non-rigid GT-only annotation noise that makes GT fit the mean face
    # worse than the clean candidate.  The negative raw fit residual must clamp
    # to zero instead of rewarding the candidate with a negative cost.
    noisy_gt[36:42, 1] += 8.0
    noisy_gt[42:48, 1] -= 6.0
    noisy_gt[48:60, 0] += np.linspace(-5.0, 5.0, 12)

    cost = transform_cost_v3(clean_candidate, noisy_gt)

    assert cost.hard_invalid is False
    assert cost.fit_delta == pytest.approx(0.0)
    assert cost.total_cost >= 0.0
