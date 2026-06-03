#!/usr/bin/env python3
"""Shared target and model constants for runtime resolver scorer training."""

from __future__ import annotations

TARGET_TRANSFORM_REGRET_V3 = "transform_alignment_regret_v3"
TARGET_TRANSFORM_COST_V3 = "transform_alignment_cost_v3"
# Partial-schema 39-point profile ranking target used by the profile specialist
# (learned_quality_v3_profile). Kept separate from the canonical-68 targets.
TARGET_PROFILE39_TRANSFORM_REGRET = "profile39_transform_regret"
# Mixed canonical-68/profile39 ranking target used when the profile specialist is
# trained from both row sources in one LambdaRank model. Profile-policy only.
TARGET_PROFILE_MIXED_TRANSFORM_REGRET = "profile_mixed_transform_regret_v1"

REGRESSION_TARGETS: tuple[str, ...] = (TARGET_TRANSFORM_REGRET_V3,)
SCORER_TARGETS: tuple[str, ...] = REGRESSION_TARGETS

MODEL_TYPE_LIGHTGBM_LAMBDARANK = "lightgbm_lambdarank"
SCORE_SEMANTICS_PREDICTED_COST = "predicted_cost"

DEFAULT_NME_FAILURE_THRESHOLD: float = 0.08
DEFAULT_LARGE_REGRET_THRESHOLD: float = 0.01
DEFAULT_REGRET_NORMALIZER: float = 0.03
DEFAULT_FAILURE_COST_PENALTY: float = 2.0
DEFAULT_COLLAPSE_COST_PENALTY: float = 0.5
DEFAULT_LARGE_COST_THRESHOLD: float = 1.0


__all__ = [
    "DEFAULT_COLLAPSE_COST_PENALTY",
    "DEFAULT_FAILURE_COST_PENALTY",
    "DEFAULT_LARGE_COST_THRESHOLD",
    "DEFAULT_LARGE_REGRET_THRESHOLD",
    "DEFAULT_NME_FAILURE_THRESHOLD",
    "DEFAULT_REGRET_NORMALIZER",
    "MODEL_TYPE_LIGHTGBM_LAMBDARANK",
    "REGRESSION_TARGETS",
    "SCORE_SEMANTICS_PREDICTED_COST",
    "SCORER_TARGETS",
    "TARGET_PROFILE39_TRANSFORM_REGRET",
    "TARGET_PROFILE_MIXED_TRANSFORM_REGRET",
    "TARGET_TRANSFORM_COST_V3",
    "TARGET_TRANSFORM_REGRET_V3",
]
