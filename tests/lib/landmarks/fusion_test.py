#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.core.fusion`."""

import numpy as np

from lib.landmarks.core.fusion import (
    normalize_weight_matrix,
    plain_average,
    weighted_average,
    weights_from_sources,
)
from lib.landmarks.core.schema import LandmarkPrediction


def _points(value: float) -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[:, 0] = value
    points[:, 1] = value
    return points


def test_plain_average_equal_weights() -> None:
    """Plain averaging gives each prediction equal influence."""
    result = plain_average([_points(0), _points(2)])
    np.testing.assert_allclose(result.points, _points(1))
    assert result.strategy == "plain_average"
    np.testing.assert_allclose(
        result.weights,
        np.full((2, 68), 0.5, dtype="float32"),
    )


def test_weighted_average_static_weights() -> None:
    """Static weights are normalized before fusion."""
    result = weighted_average([_points(0), _points(10)], weights=[3, 1])
    np.testing.assert_allclose(result.points, _points(2.5))
    assert result.strategy == "static_weighted"
    np.testing.assert_allclose(
        result.weights,
        np.vstack(
            (
                np.full(68, 0.75, dtype="float32"),
                np.full(68, 0.25, dtype="float32"),
            )
        ),
    )


def test_weighted_average_per_landmark_weights() -> None:
    """Static weights can vary independently for each landmark."""
    weights = np.ones((2, 68), dtype="float32")  # type: ignore[var-annotated]
    weights[0, 0] = 3.0
    weights[1, 0] = 1.0

    result = weighted_average([_points(0), _points(10)], weights=weights)  # type: ignore[arg-type]

    assert result.points[0, 0] == np.float32(2.5)
    np.testing.assert_allclose(result.points[1:], _points(5)[1:])
    np.testing.assert_allclose(result.weights.sum(axis=0), np.ones(68))


def test_weights_from_sources() -> None:
    """Source-specific weights are applied by prediction source name."""
    predictions = [
        LandmarkPrediction(_points(0), source="fast"),
        LandmarkPrediction(_points(1), source="precise"),
    ]
    weights = weights_from_sources(predictions, {"precise": 3.0})
    np.testing.assert_allclose(weights, np.array([0.25, 0.75], dtype="float32"))


def test_normalize_weight_matrix_requires_weight_per_landmark() -> None:
    """A 2D static weight matrix must include every canonical landmark."""
    weights = normalize_weight_matrix(np.ones((2, 68)), model_count=2)

    assert weights.shape == (2, 68)
    np.testing.assert_allclose(weights.sum(axis=0), np.ones(68))


def test_weighted_average_rejects_outlier_prediction() -> None:
    """Optional outlier rejection removes a distant ensemble member."""
    result = plain_average(
        [_points(1.0), _points(1.1), _points(100.0)],
        reject_outliers=True,
        outlier_threshold=3.5,
    )
    assert result.kept_indices == (0, 1)
    assert result.rejected_indices == (2,)
    np.testing.assert_allclose(result.points, _points(1.05), rtol=1e-6)


def test_hard_drop_no_outlier_keeps_all_models() -> None:
    """Hard-drop rejection leaves close predictions unchanged."""
    result = plain_average(
        [_points(1.0), _points(1.1), _points(1.2)],
        outlier_method="hard_drop",
        outlier_threshold=3.5,
    )

    assert result.kept_indices == (0, 1, 2)
    assert result.rejected_indices == ()
    np.testing.assert_allclose(result.weights.sum(axis=0), np.ones(68))
    np.testing.assert_allclose(result.points, _points(1.1), rtol=1e-6)


def test_hard_drop_one_model_outlier_uses_remaining_models() -> None:
    """A single distant model is removed consistently by the canonical path."""
    result = plain_average(
        [_points(1.0), _points(1.1), _points(100.0)],
        outlier_method="hard_drop",
        outlier_threshold=3.5,
    )

    assert result.kept_indices == (0, 1)
    assert result.rejected_indices == (2,)
    np.testing.assert_allclose(result.weights[2], np.zeros(68))
    np.testing.assert_allclose(result.points, _points(1.05), rtol=1e-6)


def test_hard_drop_all_model_disagreement_keeps_median_model() -> None:
    """When every model disagrees, hard drop preserves the median prediction."""
    result = plain_average(
        [_points(0.0), _points(100.0), _points(200.0)],
        outlier_method="hard_drop",
        outlier_threshold=3.5,
    )

    assert result.kept_indices == (1,)
    assert result.rejected_indices == (0, 2)
    np.testing.assert_allclose(result.points, _points(100.0), rtol=1e-6)


def test_all_model_rejection_falls_back_to_one_model_per_landmark() -> None:
    """If a threshold rejects every model for a landmark, the closest model survives."""
    result = plain_average(
        [_points(0.0), _points(10.0)],
        outlier_method="hard_drop",
        outlier_threshold=0.1,
    )

    assert result.kept_indices == (0,)
    assert result.rejected_indices == (1,)
    np.testing.assert_allclose(result.weights.sum(axis=0), np.ones(68))
    np.testing.assert_allclose(result.points, _points(0.0), rtol=1e-6)


def test_downweight_reduces_outlier_influence_without_rejection() -> None:
    """Downweight keeps all models while reducing outlier contribution."""
    result = plain_average(
        [_points(1.0), _points(1.1), _points(100.0)],
        outlier_method="downweight",
        outlier_threshold=3.5,
    )

    assert result.kept_indices == (0, 1, 2)
    assert result.rejected_indices == ()
    assert np.all(result.weights[2] < result.weights[0])
    np.testing.assert_allclose(result.points, _points(12.044444), rtol=1e-6)


def test_weighted_median_method_uses_static_weights() -> None:
    """Weighted median is available through the same fusion option."""
    result = weighted_average(
        [_points(0.0), _points(10.0), _points(100.0)],
        weights=[0.2, 0.6, 0.2],
        outlier_method="weighted_median",
    )

    assert result.kept_indices == (0, 1, 2)
    assert result.rejected_indices == ()
    np.testing.assert_allclose(result.points, _points(10.0), rtol=1e-6)
