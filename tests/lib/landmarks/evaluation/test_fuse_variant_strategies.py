#!/usr/bin/env python3
"""Wrapper-level smoke + error contract for ``harness._fuse_variant``.

The exhaustive registry behaviour - canonical names, alias resolution, outlier
methods, threshold requirements - lives in ``ensemble_strategies_test.py``.
The underlying fusion math is covered by the ``plain_average``,
``static_weighted``, and ``weighted_median`` unit tests next to each function.
This file only proves that ``_fuse_variant`` routes through every canonical
strategy and the legacy alias and surfaces a useful error for unknown
strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.strategies import CANONICAL_STRATEGIES
from lib.landmarks.evaluation.harness import _fuse_variant


def _prediction(source: str, value: float) -> LandmarkPrediction:
    return LandmarkPrediction(np.full((68, 2), value, dtype="float32"), source=source)


@pytest.mark.parametrize("variant", [*CANONICAL_STRATEGIES, "static_weighted_outliers"])
def test_fuse_variant_dispatches_every_strategy_and_legacy_alias(variant: str) -> None:
    """``_fuse_variant`` should run for every canonical strategy + legacy alias."""
    predictions = [
        _prediction("hrnet", 0.1),
        _prediction("spiga", 0.2),
        _prediction("orformer", 0.9),
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

    assert result.strategy == variant
    assert result.points.shape == (68, 2)
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
