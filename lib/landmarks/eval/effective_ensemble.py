#!/usr/bin/env python3
"""Effective-ensemble diagnostics for candidate search (#79).

A "weighted ensemble" with per-landmark weights heavily skewed to one model is
functionally a disguised single-model setup at runtime cost. This module
quantifies that for candidate search so promotion stops rewarding collapsed
ensembles.

Two collapse signals matter for extract alignment:

* **Mean effective model count** — ``1 / sum_i(w_i^2)`` averaged across
  landmark columns. Equals 1 when one model carries all the weight at every
  column and ``M`` when all models share equally. The principled signal for
  weighted-average strategies: if the mean effective count is below
  ``effective_models_floor``, the ensemble behaves like a single model and
  the extra adapter calls are wasted compute.
* **Weighted-median dominance** — for ``strategy='weighted_median'``, any
  model whose weight exceeds 0.5 at every landmark wins every per-landmark
  median. That candidate is functionally a single-model swap at runtime
  cost, even though its mean effective count may still look healthy.

``majority_model_by_landmark`` and ``landmark_share_by_model`` are reported
as informational metadata. Columns where weights are tied (within ``1e-6``)
do not count toward any model's share, so equal-weight ensembles do not
falsely flag as collapsed.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import numpy as np

DEFAULT_EFFECTIVE_MODELS_FLOOR: float = 1.5
WEIGHTED_MEDIAN_DOMINANCE: float = 0.5
_STRICT_WINNER_EPSILON: float = 1e-6


@dataclass(frozen=True)
class EffectiveEnsembleDiagnostics:
    """Effective-ensemble fingerprint of one weight matrix."""

    models: tuple[str, ...]
    strategy: str
    majority_model_by_landmark: tuple[str, ...]
    landmark_share_by_model: dict[str, float]
    mean_effective_models: float
    weighted_median_collapsed: bool
    weighted_median_dominant_model: str
    collapsed: bool
    collapsed_dominant_model: str
    effective_models_floor: float

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "models": list(self.models),
            "strategy": self.strategy,
            "majority_model_by_landmark": list(self.majority_model_by_landmark),
            "landmark_share_by_model": dict(self.landmark_share_by_model),
            "mean_effective_models": float(self.mean_effective_models),
            "weighted_median_collapsed": bool(self.weighted_median_collapsed),
            "weighted_median_dominant_model": self.weighted_median_dominant_model,
            "collapsed": bool(self.collapsed),
            "collapsed_dominant_model": self.collapsed_dominant_model,
            "effective_models_floor": float(self.effective_models_floor),
        }


def diagnose(
    weights: T.Mapping[str, T.Sequence[float]],
    *,
    strategy: str,
    models: T.Sequence[str] | None = None,
    effective_models_floor: float = DEFAULT_EFFECTIVE_MODELS_FLOOR,
) -> EffectiveEnsembleDiagnostics:
    """Return the effective-ensemble fingerprint for a weight column matrix.

    ``effective_models_floor`` is the threshold below which a weighted-average
    ensemble is considered collapsed; pick higher values (closer to ``M``) for
    a stricter ensemble definition. Single-model candidates trivially have
    ``mean_effective_models == 1.0`` so they always count as collapsed.
    """
    if not weights:
        raise ValueError("weights cannot be empty")
    if effective_models_floor <= 1.0:
        raise ValueError(
            f"effective_models_floor must be > 1.0, got {effective_models_floor!r}"
        )
    chosen = tuple(models) if models is not None else tuple(weights)
    missing = [model for model in chosen if model not in weights]
    if missing:
        raise ValueError(f"weights missing columns for models: {missing}")
    matrix = np.array([weights[model] for model in chosen], dtype="float32")
    if matrix.ndim != 2:
        raise ValueError(f"weights must be a (model, landmark) matrix, got shape {matrix.shape}")
    if np.any(matrix < 0):
        raise ValueError("weights must be non-negative")

    column_sums = matrix.sum(axis=0)
    if np.any(column_sums <= 0):
        raise ValueError("each landmark column must have at least one non-zero weight")
    normalized = matrix / column_sums[None, :]

    # informational: argmax-based majority per landmark
    majority_indices = np.argmax(normalized, axis=0)
    majority_model_by_landmark = tuple(chosen[index] for index in majority_indices.tolist())
    # strict-winner share: ties (within epsilon) do not count toward any share
    sorted_normalized = np.sort(normalized, axis=0)
    top = sorted_normalized[-1]
    second = sorted_normalized[-2] if normalized.shape[0] > 1 else np.zeros_like(top)
    has_strict_winner = (top - second) > _STRICT_WINNER_EPSILON
    share: dict[str, float] = {}
    for index, model in enumerate(chosen):
        wins = np.logical_and(has_strict_winner, majority_indices == index)
        share[model] = float(np.mean(wins))

    effective_per_landmark = 1.0 / np.maximum(np.sum(normalized**2, axis=0), 1e-12)
    mean_effective_models = float(np.mean(effective_per_landmark))

    weighted_median_collapsed = False
    weighted_median_dominant_model = ""
    if strategy == "weighted_median":
        dominant_mask = normalized > WEIGHTED_MEDIAN_DOMINANCE
        all_dominant = dominant_mask.all(axis=1)
        if np.any(all_dominant):
            weighted_median_collapsed = True
            weighted_median_dominant_model = chosen[int(np.argmax(all_dominant))]

    effective_collapse = mean_effective_models < effective_models_floor
    collapsed = bool(weighted_median_collapsed or effective_collapse)
    collapsed_dominant_model = ""
    if weighted_median_collapsed:
        collapsed_dominant_model = weighted_median_dominant_model
    elif effective_collapse:
        collapsed_dominant_model = max(share, key=lambda key: share[key]) if share else ""

    return EffectiveEnsembleDiagnostics(
        models=chosen,
        strategy=strategy,
        majority_model_by_landmark=majority_model_by_landmark,
        landmark_share_by_model=share,
        mean_effective_models=mean_effective_models,
        weighted_median_collapsed=weighted_median_collapsed,
        weighted_median_dominant_model=weighted_median_dominant_model,
        collapsed=collapsed,
        collapsed_dominant_model=collapsed_dominant_model,
        effective_models_floor=effective_models_floor,
    )


__all__ = [
    "DEFAULT_EFFECTIVE_MODELS_FLOOR",
    "EffectiveEnsembleDiagnostics",
    "WEIGHTED_MEDIAN_DOMINANCE",
    "diagnose",
]
