#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.ensemble.strategies` registry."""

import pytest

from lib.landmarks.ensemble.strategies import (
    CANONICAL_STRATEGIES,
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
    validate_threshold,
)


def test_canonical_strategies_are_stable() -> None:
    """The canonical strategy list is the public contract for #65 artifacts."""
    assert CANONICAL_STRATEGIES == (
        "plain_average",
        "static_weighted",
        "static_weighted_hard_drop",
        "static_weighted_downweight",
        "weighted_median",
    )


@pytest.mark.parametrize("name", CANONICAL_STRATEGIES)
def test_canonical_strategy_is_identity_for_canonical_names(name: str) -> None:
    """Canonical names resolve to themselves without alias translation."""
    assert canonical_strategy(name) == name


def test_canonical_strategy_translates_legacy_alias() -> None:
    """``static_weighted_outliers`` resolves to ``static_weighted_hard_drop``."""
    assert canonical_strategy("static_weighted_outliers") == "static_weighted_hard_drop"


def test_canonical_strategy_translates_static_weighted_none_alias() -> None:
    """``static_weighted_none`` resolves to ``static_weighted``."""
    assert canonical_strategy("static_weighted_none") == "static_weighted"


def test_canonical_strategy_rejects_unknown_with_supported_list() -> None:
    """Unknown strategies raise a ValueError that lists every supported name."""
    with pytest.raises(ValueError) as exc:
        canonical_strategy("unknown_strategy")
    message = str(exc.value)
    assert "unknown_strategy" in message
    for name in CANONICAL_STRATEGIES:
        assert name in message


@pytest.mark.parametrize("bad", ["", None, 0, []])
def test_canonical_strategy_rejects_non_string_inputs(bad: object) -> None:
    """Empty/non-string strategy values fail fast."""
    with pytest.raises(ValueError):
        canonical_strategy(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "strategy,method",
    [
        ("plain_average", "none"),
        ("static_weighted", "none"),
        ("static_weighted_hard_drop", "hard_drop"),
        ("static_weighted_downweight", "downweight"),
        ("weighted_median", "weighted_median"),
        ("static_weighted_outliers", "hard_drop"),
    ],
)
def test_strategy_outlier_method_maps_canonical_and_aliases(strategy: str, method: str) -> None:
    """Every strategy maps to exactly one fusion outlier method."""
    assert strategy_outlier_method(strategy) == method


@pytest.mark.parametrize(
    "strategy,uses",
    [
        ("plain_average", False),
        ("static_weighted", False),
        ("static_weighted_hard_drop", True),
        ("static_weighted_downweight", True),
        ("weighted_median", False),
    ],
)
def test_strategy_uses_threshold_matches_outlier_aware_strategies(
    strategy: str, uses: bool
) -> None:
    """Only threshold-aware strategies report ``strategy_uses_threshold=True``."""
    assert strategy_uses_threshold(strategy) is uses


@pytest.mark.parametrize(
    "strategy,requires",
    [
        ("plain_average", False),
        ("static_weighted", True),
        ("static_weighted_hard_drop", True),
        ("static_weighted_downweight", True),
        ("weighted_median", True),
    ],
)
def test_strategy_requires_weights_is_true_except_plain_average(
    strategy: str, requires: bool
) -> None:
    """Plain averaging is the only canonical strategy that does not use weights."""
    assert strategy_requires_weights(strategy) is requires


@pytest.mark.parametrize("strategy", ["static_weighted_hard_drop", "static_weighted_downweight"])
def test_validate_threshold_requires_positive_value_for_threshold_strategies(
    strategy: str,
) -> None:
    """Threshold strategies must receive a strictly positive numeric value."""
    validate_threshold(strategy, 3.5)
    with pytest.raises(ValueError):
        validate_threshold(strategy, None)
    with pytest.raises(ValueError):
        validate_threshold(strategy, 0)
    with pytest.raises(ValueError):
        validate_threshold(strategy, -1)


@pytest.mark.parametrize("strategy", ["plain_average", "static_weighted", "weighted_median"])
def test_validate_threshold_rejects_threshold_for_non_threshold_strategies(
    strategy: str,
) -> None:
    """Non-threshold strategies must leave outlier_threshold unset."""
    validate_threshold(strategy, None)
    with pytest.raises(ValueError):
        validate_threshold(strategy, 3.5)
