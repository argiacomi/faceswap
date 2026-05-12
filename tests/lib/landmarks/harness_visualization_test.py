#!/usr/bin/env python3
"""Tests for harness and visualization helpers."""

import numpy as np

from lib.landmarks.adapters import LandmarkAdapterConfig, StaticLandmarkAdapter
from lib.landmarks.harness import (
    EvaluationSample,
    collect_adapter_predictions,
    evaluate_predictions,
)
from lib.landmarks.visualization import draw_landmarks, make_debug_overlay


def _points(x_val: float = 2.0, y_val: float = 3.0) -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")
    points[:, 0] = x_val
    points[:, 1] = y_val
    points[36] = (0, 0)
    points[45] = (10, 0)
    return points


def test_collect_adapter_predictions_skips_disabled() -> None:
    """Harness collection only runs enabled adapters."""
    image = np.zeros((8, 8, 3), dtype="uint8")
    enabled = StaticLandmarkAdapter(LandmarkAdapterConfig("enabled"), _points())
    disabled = StaticLandmarkAdapter(
        LandmarkAdapterConfig("disabled", enabled=False),
        _points(),
    )
    predictions = collect_adapter_predictions([enabled, disabled], image)
    assert sorted(predictions) == ["enabled"]


def test_evaluate_predictions_aggregates_by_source() -> None:
    """Evaluation aggregation reports count and averaged metrics per source."""
    truth = _points()
    samples = [
        EvaluationSample(
            sample_id="sample",
            ground_truth=truth,
            predictions={"adapter": truth.copy()},
            normalizer=10.0,
        )
    ]
    result = evaluate_predictions(samples)
    assert result["adapter"]["count"] == 1.0
    assert result["adapter"]["mean_point_error"] == 0.0
    assert result["adapter"]["normalized_mean_error"] == 0.0


def test_draw_landmarks_returns_modified_copy() -> None:
    """Debug drawing produces a modified copy without mutating the source."""
    image = np.zeros((12, 12, 3), dtype="uint8")
    output = draw_landmarks(image, _points(), color=(255, 0, 0), radius=1)
    assert image.sum() == 0
    assert output.sum() > 0


def test_make_debug_overlay_handles_multiple_predictions() -> None:
    """Overlay helper draws multiple predictions on one image."""
    image = np.zeros((12, 12, 3), dtype="uint8")
    output = make_debug_overlay(
        image, {"first": _points(2, 3), "second": _points(5, 6)}
    )
    assert output.sum() > 0
