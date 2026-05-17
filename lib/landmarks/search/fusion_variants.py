#!/usr/bin/env python3
"""Canonical fusion-variant helpers (Ticket 3).

Three CLI tools used to ship byte-identical clones of ``_fuse_variant`` and
the candidate-eval helper added a fourth ``fuse_candidate``. Both paths are
just thin shells around the canonical strategy registry + Faceswap fusion
primitives; consolidating them here cuts the drift surface and gives every
downstream consumer the same code path.

Two entry points are exposed:

* :func:`fuse_variant` — strategy-name based; matches the CLI surface where
  the operator names variants on the command line.
* :func:`fuse_candidate` — :class:`Candidate` based; matches the search
  subsystem where the strategy + threshold live on a dataclass.

Both return ``(68, 2)`` numpy arrays in the canonical landmark schema.
``weighted_median`` requires static weights; the helpers raise
``ValueError`` if a weighted variant is requested without them.
"""

from __future__ import annotations

import typing as T

import numpy as np

from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.ensemble.weights import weights_matrix_for_models
from lib.landmarks.eval.candidate_search import Candidate
from lib.landmarks.fusion import normalize_weight_matrix, plain_average, static_weighted
from lib.landmarks.rejection import weighted_median
from lib.landmarks.schema import LandmarkPrediction

#: Default outlier threshold used when a strategy needs one but the caller
#: hasn't supplied a value. Matches the legacy Faceswap default.
DEFAULT_OUTLIER_THRESHOLD: float = 3.5


def fuse_variant(
    variant: str,
    predictions: T.Sequence[LandmarkPrediction],
    *,
    models: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]] | None = None,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> np.ndarray:
    """Fuse a sequence of :class:`LandmarkPrediction` items via a named variant.

    ``variant`` is resolved to its canonical name via the strategy registry,
    so legacy aliases (e.g. ``static_weighted_outliers``) keep working.
    Threshold-aware strategies (``static_weighted_hard_drop``,
    ``static_weighted_downweight``) consume ``outlier_threshold``; the rest
    silently ignore it.
    """
    canonical = canonical_strategy(variant)
    method = strategy_outlier_method(canonical)
    threshold = (
        outlier_threshold if strategy_uses_threshold(canonical) else DEFAULT_OUTLIER_THRESHOLD
    )

    if not strategy_requires_weights(canonical):
        return plain_average(
            predictions, outlier_method=method, outlier_threshold=threshold
        ).points

    if weights is None:
        raise ValueError(f"variant {variant!r} requires a static weights file")
    matrix = weights_matrix_for_models(weights, tuple(models))
    if canonical == "weighted_median":
        stack = np.stack([prediction.canonical_68().points for prediction in predictions], axis=0)
        normalized = normalize_weight_matrix(
            matrix, model_count=stack.shape[0], landmark_count=stack.shape[1]
        )
        return weighted_median(stack, normalized)
    return static_weighted(
        predictions,
        matrix,
        outlier_method=method,
        outlier_threshold=threshold,
    ).points


def fuse_candidate(
    candidate: Candidate,
    cached_points: T.Sequence[np.ndarray],
    *,
    weights: dict[str, list[float]],
) -> np.ndarray:
    """Fuse one candidate's cached per-model predictions to a ``(68, 2)`` array.

    Convenience wrapper around :func:`fuse_variant`: the strategy and
    threshold are taken from the :class:`Candidate` dataclass, and the
    cached numpy predictions are wrapped in :class:`LandmarkPrediction`
    instances tagged with their model source.
    """
    predictions = [
        LandmarkPrediction(np.asarray(points, dtype="float32"), source=model)
        for points, model in zip(cached_points, candidate.models, strict=True)
    ]
    threshold = (
        candidate.outlier_threshold
        if candidate.outlier_threshold is not None
        else DEFAULT_OUTLIER_THRESHOLD
    )
    return fuse_variant(
        candidate.strategy,
        predictions,
        models=candidate.models,
        weights=weights,
        outlier_threshold=threshold,
    )


__all__ = [
    "DEFAULT_OUTLIER_THRESHOLD",
    "fuse_candidate",
    "fuse_variant",
]
