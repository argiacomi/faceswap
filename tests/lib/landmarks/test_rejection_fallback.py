#!/usr/bin/env python3
"""Regression tests for per-landmark outlier rejection fallback weights."""

from __future__ import annotations

import numpy as np

from lib.landmarks.rejection import reject_outliers


def test_hard_drop_fallback_restores_positive_weight_when_best_base_weight_is_zero() -> None:
    """An empty landmark column must recover with a positive fallback weight.

    This covers candidates whose static weights assign zero weight to the model
    closest to the per-landmark median. Hard-drop outlier rejection can zero all
    positive-weight models for a landmark; the fallback must choose the closest
    positive-base model, or assign unit weight when none exists.
    """
    stack = np.zeros((3, 68, 2), dtype="float32")
    stack[0, :, :] = 0.0
    stack[1, :, :] = 100.0
    stack[2, :, :] = 200.0

    # For landmark 0, model 0 is closest to the median of all models after hard
    # drops, but its incoming base weight is zero. Models 1 and 2 are positive
    # and get dropped, emptying the column. The fallback must not restore the
    # zero-weight model and leave the column empty again.
    weights = np.ones((3, 68), dtype="float32")
    weights[0, 0] = 0.0
    weights[1, 0] = 1.0
    weights[2, 0] = 1.0

    result = reject_outliers(
        stack,
        weights,
        threshold=0.1,
        method="hard_drop",
    )

    assert np.all(result.weights.sum(axis=0) > 0)
    assert result.weights[:, 0].sum() == 1.0
    assert result.weights[0, 0] == 0.0
    assert result.weights[1, 0] > 0.0 or result.weights[2, 0] > 0.0
