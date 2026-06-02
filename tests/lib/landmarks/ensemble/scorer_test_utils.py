#!/usr/bin/env python3
"""Test helpers for learned-quality runtime scorer behavior."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

from lib.landmarks.ensemble.scorer_target_config import (
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    SCORE_SEMANTICS_PREDICTED_COST,
    TARGET_TRANSFORM_REGRET_V3,
)


@dataclass(frozen=True)
class LinearTestScorer:
    """Deterministic scorer implementing the learned v3 runtime scorer protocol."""

    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float = 0.0
    model_type: str = MODEL_TYPE_LIGHTGBM_LAMBDARANK
    target: str = TARGET_TRANSFORM_REGRET_V3
    score_semantics: str = SCORE_SEMANTICS_PREDICTED_COST
    higher_is_better: bool = False
    source_path: str = ""
    version: str = "learned_quality_v3"
    runtime_policy: str = "learned_quality_v3"
    feature_importances: dict[str, float] | None = None

    def score_feature_map(self, features: T.Mapping[str, float]) -> float:
        score = self.intercept
        for name, coefficient in zip(self.features, self.coefficients, strict=True):
            score += coefficient * float(features.get(name, 0.0) or 0.0)
        return float(score)

    def score_feature_maps(self, feature_maps: T.Sequence[T.Mapping[str, float]]) -> list[float]:
        return [self.score_feature_map(features) for features in feature_maps]


__all__ = ["LinearTestScorer"]
