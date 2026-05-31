#!/usr/bin/env python3
"""Correctness tests for the vectorized ``weighted_median``.

These tests pin the vectorized implementation against (a) a simple per-landmark
loop reference and (b) hand-computed expected values for canonical inputs. They
exist because ``weighted_median`` is one of the five canonical fusion
strategies surfaced to the runtime extract plugin (#70); regressing its
output would silently shift fused landmark coordinates for every face.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.core.rejection import _normalize_weights, weighted_median


def _weighted_median_loop_reference(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-landmark loop reference (the pre-vectorization implementation)."""
    weight_matrix = _normalize_weights(
        weights, model_count=points.shape[0], landmark_count=points.shape[1]
    )
    output = np.empty(points.shape[1:], dtype="float32")
    for landmark_idx in range(points.shape[1]):
        landmark_weights = weight_matrix[:, landmark_idx]
        for coord_idx in range(2):
            values = points[:, landmark_idx, coord_idx]
            order = np.argsort(values)
            cumulative = np.cumsum(landmark_weights[order])
            output[landmark_idx, coord_idx] = values[order[np.searchsorted(cumulative, 0.5)]]
    return output  # type: ignore[no-any-return]


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1337])
def test_vectorized_weighted_median_matches_loop_reference(seed: int) -> None:
    """Vectorized output must match the per-landmark loop for random inputs."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((3, 68, 2)).astype("float32")
    weights = rng.uniform(0.1, 5.0, size=(3,)).astype("float32")

    expected = _weighted_median_loop_reference(points, weights)
    actual = weighted_median(points, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_vectorized_weighted_median_matches_loop_for_per_landmark_weights() -> None:
    """Per-landmark weight matrices (not just per-model scalars) must agree."""
    rng = np.random.default_rng(2024)
    points = rng.standard_normal((4, 68, 2)).astype("float32")
    weights = rng.uniform(0.1, 5.0, size=(4, 68)).astype("float32")

    expected = _weighted_median_loop_reference(points, weights)
    actual = weighted_median(points, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_weighted_median_equal_weights_returns_middle_value() -> None:
    """With equal weights and three models, the median is the middle value."""
    points = np.stack(
        [
            np.full((68, 2), 1.0, dtype="float32"),
            np.full((68, 2), 5.0, dtype="float32"),
            np.full((68, 2), 100.0, dtype="float32"),
        ],
        axis=0,
    )
    weights = np.array([1.0, 1.0, 1.0], dtype="float32")

    result = weighted_median(points, weights)

    np.testing.assert_allclose(result, np.full((68, 2), 5.0, dtype="float32"))


def test_weighted_median_dominant_weight_drives_output() -> None:
    """When one model holds >50% of the weight, its value is selected."""
    points = np.stack(
        [
            np.full((68, 2), 1.0, dtype="float32"),
            np.full((68, 2), 5.0, dtype="float32"),
            np.full((68, 2), 100.0, dtype="float32"),
        ],
        axis=0,
    )
    weights = np.array([0.1, 0.8, 0.1], dtype="float32")

    result = weighted_median(points, weights)

    np.testing.assert_allclose(result, np.full((68, 2), 5.0, dtype="float32"))


def test_weighted_median_independent_landmarks_resolve_independently() -> None:
    """Each landmark's weighted median resolves separately from the others."""
    points = np.zeros((3, 68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[0] = 1.0
    points[1] = 5.0
    points[2] = 100.0
    weights = np.ones((3, 68), dtype="float32")  # type: ignore[var-annotated]
    weights[:, 0] = (1.0, 0.0, 0.0)  # landmark 0: only model 0 has weight
    weights[:, 1] = (0.0, 0.0, 1.0)  # landmark 1: only model 2 has weight

    result = weighted_median(points, weights)

    assert result[0, 0] == pytest.approx(1.0)
    assert result[1, 0] == pytest.approx(100.0)
    # Remaining landmarks fall back to the equal-weights median (5.0).
    np.testing.assert_allclose(result[2:], np.full((66, 2), 5.0, dtype="float32"))


def test_weighted_median_returns_float32() -> None:
    """Output dtype is float32 regardless of input dtype."""
    points = np.full((3, 68, 2), 1.0, dtype="float64")  # type: ignore[var-annotated]
    weights = np.array([1.0, 1.0, 1.0], dtype="float64")

    result = weighted_median(points, weights)

    assert result.dtype == np.float32
