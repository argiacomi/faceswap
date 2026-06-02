#!/usr/bin/env python3
"""Shared landmark ensemble fusion strategy registry.

This is the single source of truth for the strategy names that flow between the
offline evaluation harness, the candidate search/promotion tooling, and the
runtime extract ensemble aligner. Promoted artifacts must serialize one of the
canonical strategies named here; aliases below exist only to accept legacy
inputs and translate them to the canonical name.
"""

from __future__ import annotations

CANONICAL_STRATEGIES: tuple[str, ...] = (
    "plain_average",
    "static_weighted",
    "static_weighted_hard_drop",
    "static_weighted_downweight",
    "weighted_median",
    "region_weighted",
)

_OUTLIER_METHOD_BY_STRATEGY: dict[str, str] = {
    "plain_average": "none",
    "static_weighted": "none",
    "static_weighted_hard_drop": "hard_drop",
    "static_weighted_downweight": "downweight",
    "weighted_median": "weighted_median",
    # Region-weighted fusion (Phase 5 #9) assembles a static-weighted average
    # from a region-broadcast weight vector; it runs no outlier rejection of
    # its own (the per-region weights already encode the model trust profile).
    "region_weighted": "none",
}

_THRESHOLD_STRATEGIES: frozenset[str] = frozenset(
    {
        "static_weighted_hard_drop",
        "static_weighted_downweight",
    }
)

# Legacy/alternate spellings translated to canonical names. Kept narrow so the
# canonical list remains the single source of truth.
_STRATEGY_ALIASES: dict[str, str] = {
    "static_weighted_outliers": "static_weighted_hard_drop",
    "static_weighted_none": "static_weighted",
}


def canonical_strategy(name: str) -> str:
    """Resolve a strategy name, including legacy aliases, to its canonical form.

    Raises ``ValueError`` with a message listing every supported strategy when
    the input is empty, unrecognized, or not a string.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            "strategy must be a non-empty string; supported strategies: "
            + ", ".join(CANONICAL_STRATEGIES)
        )
    if name in CANONICAL_STRATEGIES:
        return name
    aliased = _STRATEGY_ALIASES.get(name)
    if aliased is not None:
        return aliased
    raise ValueError(
        f"unknown ensemble strategy {name!r}; supported strategies: "
        + ", ".join(CANONICAL_STRATEGIES)
    )


def strategy_outlier_method(strategy: str) -> str:
    """Return the canonical fusion ``outlier_method`` for ``strategy``."""
    return _OUTLIER_METHOD_BY_STRATEGY[canonical_strategy(strategy)]


def strategy_uses_threshold(strategy: str) -> bool:
    """Return True if a strategy consumes ``outlier_threshold``."""
    return canonical_strategy(strategy) in _THRESHOLD_STRATEGIES


def strategy_requires_weights(strategy: str) -> bool:
    """Return True if a strategy needs static per-model weights."""
    return canonical_strategy(strategy) != "plain_average"


def validate_threshold(strategy: str, threshold: float | None) -> None:
    """Validate an ``outlier_threshold`` against the selected strategy.

    Threshold-using strategies (``static_weighted_hard_drop``,
    ``static_weighted_downweight``) require a positive numeric value. Every
    other strategy must leave the threshold unset (``None``). Mismatches raise
    ``ValueError`` so downstream callers fail fast on inconsistent setups.
    """
    canonical = canonical_strategy(strategy)
    uses_threshold = canonical in _THRESHOLD_STRATEGIES
    if uses_threshold:
        if threshold is None:
            raise ValueError(f"strategy {canonical!r} requires a positive outlier_threshold")
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(
                f"strategy {canonical!r} requires a positive outlier_threshold, got {threshold!r}"
            )
    elif threshold is not None:
        raise ValueError(
            f"strategy {canonical!r} does not accept outlier_threshold; "
            "value must be omitted or null"
        )


__all__ = [
    "CANONICAL_STRATEGIES",
    "canonical_strategy",
    "strategy_outlier_method",
    "strategy_requires_weights",
    "strategy_uses_threshold",
    "validate_threshold",
]
