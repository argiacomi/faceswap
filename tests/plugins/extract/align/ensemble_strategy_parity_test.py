#!/usr/bin/env python3
"""Strategy parity tests for the landmark ensemble aligner plugin (issue #70).

These tests pair every canonical strategy from
``lib.landmarks.ensemble.strategies`` against an equivalent harness-style fusion
call, so any divergence in fusion behavior between the offline evaluation
harness and the runtime extract aligner shows up as a test failure rather than
as silently different production output.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from lib.landmarks.adapters import (
    LandmarkAdapterConfig,
    StaticLandmarkAdapter,
)
from lib.landmarks.ensemble.outliers import weighted_median
from lib.landmarks.ensemble.strategies import (
    CANONICAL_STRATEGIES,
    canonical_strategy,
    strategy_outlier_method,
)
from lib.landmarks.fusion import (
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from plugins.extract.align.ensemble import Ensemble


def _points(value: float) -> np.ndarray:
    return np.full((68, 2), value, dtype="float32")


def _three_static_adapters(values: tuple[float, float, float]) -> list[StaticLandmarkAdapter]:
    """Return three adapters with distinct names, weights, and constant outputs."""
    return [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig(name, coordinate_space="frame", weight=weight),
            _points(value),
        )
        for name, weight, value in (
            ("near", 3.0, values[0]),
            ("mid", 2.0, values[1]),
            ("far", 1.0, values[2]),
        )
    ]


def _expected_for_strategy(strategy: str, values: tuple[float, float, float]) -> np.ndarray:
    """Compute the expected fused output for a strategy using harness-side helpers."""
    canonical = canonical_strategy(strategy)
    method = strategy_outlier_method(canonical)
    predictions = [_points(value) for value in values]
    if canonical == "plain_average":
        return plain_average(
            predictions,
            outlier_method=method,
            outlier_threshold=3.5,
        ).points
    weights = np.array([3.0, 2.0, 1.0], dtype="float32")
    if canonical == "weighted_median":
        stack = np.stack(predictions, axis=0)
        normalized = normalize_weight_matrix(
            weights, model_count=stack.shape[0], landmark_count=stack.shape[1]
        )
        return weighted_median(stack, normalized)
    return static_weighted(
        predictions,
        weights,
        outlier_method=method,
        outlier_threshold=3.5,
    ).points


@pytest.mark.parametrize("strategy", CANONICAL_STRATEGIES)
def test_plugin_matches_harness_fusion_for_every_canonical_strategy(strategy: str) -> None:
    """Plugin output must match harness-equivalent fusion for every canonical name."""
    values = (0.1, 0.2, 0.9)
    plugin = Ensemble(
        adapters=_three_static_adapters(values),
        crop_scale=1.0,
        reject_outliers=False,
        strategy=strategy,
        outlier_threshold=3.5,
    )
    plugin.model = plugin.load_model()

    result = plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    expected = _expected_for_strategy(strategy, values)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert plugin.last_debug_metadata[0]["strategy"] == strategy
    assert plugin.last_debug_metadata[0]["outlier_method"] == strategy_outlier_method(strategy)


def test_legacy_alias_static_weighted_outliers_resolves_to_canonical() -> None:
    """``static_weighted_outliers`` is accepted and behaves like ``static_weighted_hard_drop``."""
    values = (0.1, 0.2, 0.9)
    plugin = Ensemble(
        adapters=_three_static_adapters(values),
        crop_scale=1.0,
        reject_outliers=False,
        strategy="static_weighted_outliers",
        outlier_threshold=3.5,
    )
    plugin.model = plugin.load_model()

    result = plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    expected = _expected_for_strategy("static_weighted_hard_drop", values)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert plugin.last_debug_metadata[0]["strategy"] == "static_weighted_hard_drop"


def test_reject_outliers_compat_flag_promotes_static_weighted_to_hard_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``reject_outliers=True`` + ``static_weighted`` becomes ``static_weighted_hard_drop``."""
    values = (0.1, 0.2, 0.9)
    caplog.set_level(logging.INFO, logger="plugins.extract.align.ensemble")
    plugin = Ensemble(
        adapters=_three_static_adapters(values),
        crop_scale=1.0,
        reject_outliers=True,
        strategy="static_weighted",
        outlier_threshold=3.5,
    )
    plugin.model = plugin.load_model()

    result = plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    expected = _expected_for_strategy("static_weighted_hard_drop", values)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert plugin.last_debug_metadata[0]["strategy"] == "static_weighted_hard_drop"
    assert any("deprecated" in record.message.lower() for record in caplog.records)


def test_reject_outliers_compat_flag_is_ignored_for_other_strategies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``reject_outliers`` does not override strategies that already govern outliers."""
    values = (0.1, 0.2, 0.9)
    caplog.set_level(logging.INFO, logger="plugins.extract.align.ensemble")
    plugin = Ensemble(
        adapters=_three_static_adapters(values),
        crop_scale=1.0,
        reject_outliers=True,
        strategy="static_weighted_downweight",
        outlier_threshold=3.5,
    )
    plugin.model = plugin.load_model()

    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    assert plugin.last_debug_metadata[0]["strategy"] == "static_weighted_downweight"
    assert any("ignored" in record.message.lower() for record in caplog.records)


def test_unknown_strategy_raises_with_supported_list() -> None:
    """An unsupported strategy fails at construction time with the supported list."""
    with pytest.raises(ValueError) as exc:
        Ensemble(adapters=[], strategy="nonexistent_strategy")
    message = str(exc.value)
    assert "nonexistent_strategy" in message
    for name in CANONICAL_STRATEGIES:
        assert name in message
