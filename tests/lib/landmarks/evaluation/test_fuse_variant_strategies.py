#!/usr/bin/env python3
"""Harness ``_fuse_variant`` strategy coverage tests (issue #70)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.core.fusion import (
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.outliers import weighted_median
from lib.landmarks.ensemble.strategies import (
    CANONICAL_STRATEGIES,
    canonical_strategy,
    strategy_outlier_method,
)
from lib.landmarks.evaluation.harness import _fuse_variant


def _prediction(source: str, value: float) -> LandmarkPrediction:
    points = np.full((68, 2), value, dtype="float32")
    return LandmarkPrediction(points, source=source)


def _expected_points(strategy: str, values: tuple[float, float, float]) -> np.ndarray:
    """Return expected fused points using harness-equivalent fusion helpers."""
    canonical = canonical_strategy(strategy)
    method = strategy_outlier_method(canonical)
    predictions = [_prediction(f"m{idx}", value) for idx, value in enumerate(values)]
    weights = np.array([[3.0] * 68, [2.0] * 68, [1.0] * 68], dtype="float32")
    if canonical == "plain_average":
        return plain_average(predictions, outlier_method=method).points
    if canonical == "weighted_median":
        stack = np.stack([prediction.points for prediction in predictions], axis=0)
        normalized = normalize_weight_matrix(weights, model_count=3, landmark_count=stack.shape[1])
        return weighted_median(stack, normalized)
    return static_weighted(predictions, weights, outlier_method=method).points


@pytest.mark.parametrize(
    "variant",
    [*CANONICAL_STRATEGIES, "static_weighted_outliers"],
)
def test_fuse_variant_dispatches_every_strategy_and_legacy_alias(variant: str) -> None:
    """``_fuse_variant`` must route every canonical strategy and the alias."""
    values = (0.1, 0.2, 0.9)
    predictions = [
        _prediction("hrnet", values[0]),
        _prediction("spiga", values[1]),
        _prediction("orformer", values[2]),
    ]
    weights = {
        "hrnet": [3.0] * 68,
        "spiga": [2.0] * 68,
        "orformer": [1.0] * 68,
    }

    result, rejected = _fuse_variant(
        variant,
        predictions,
        ("hrnet", "spiga", "orformer"),
        weights,
        outlier_threshold=3.5,
    )

    expected = _expected_points(variant, values)
    np.testing.assert_allclose(result.points, expected, rtol=1e-6, atol=1e-6)
    assert result.strategy == variant
    assert isinstance(rejected, int)


def test_fuse_variant_rejects_unknown_strategy_with_supported_list() -> None:
    """Unknown variants fail with the supported strategy list."""
    with pytest.raises(ValueError) as exc:
        _fuse_variant(
            "made_up_strategy",
            [],
            (),
            {},
            outlier_threshold=3.5,
        )
    message = str(exc.value)
    assert "made_up_strategy" in message
    for name in CANONICAL_STRATEGIES:
        assert name in message
