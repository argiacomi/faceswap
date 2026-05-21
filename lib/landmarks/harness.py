#!/usr/bin/env python3
"""Minimal evaluation harness utilities for landmark experiments."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.adapters import LandmarkAdapter
from lib.landmarks.core.metrics import mean_point_error, normalized_mean_error
from lib.landmarks.core.schema import LandmarkPrediction, to_canonical_68


@dataclass(frozen=True)
class EvaluationSample:
    """Ground truth and predictions for one evaluated face."""

    sample_id: str
    ground_truth: np.ndarray
    predictions: T.Mapping[str, LandmarkPrediction | np.ndarray] = field(default_factory=dict)
    normalizer: float | None = None


def collect_adapter_predictions(
    adapters: T.Iterable[LandmarkAdapter],
    image: np.ndarray,
    *,
    face: object | None = None,
) -> dict[str, LandmarkPrediction]:
    """Run enabled adapters and collect predictions by adapter name."""
    output: dict[str, LandmarkPrediction] = {}
    for adapter in adapters:
        if not adapter.config.enabled:
            continue
        output[adapter.config.name] = adapter.predict(image, face=face)
    return output


def evaluate_predictions(
    samples: T.Iterable[EvaluationSample],
) -> dict[str, dict[str, float]]:
    """Aggregate mean point error and NME by prediction source."""
    totals: dict[str, dict[str, float]] = {}
    for sample in samples:
        truth = to_canonical_68(sample.ground_truth)
        for source, prediction in sample.predictions.items():
            points = (
                prediction.canonical_68().points
                if isinstance(prediction, LandmarkPrediction)
                else prediction
            )
            source_totals = totals.setdefault(
                source,
                {"count": 0.0, "mean_point_error": 0.0, "normalized_mean_error": 0.0},
            )
            source_totals["count"] += 1.0
            source_totals["mean_point_error"] += mean_point_error(points, truth)
            source_totals["normalized_mean_error"] += normalized_mean_error(
                points,
                truth,
                normalizer=sample.normalizer,
            )
    for values in totals.values():
        count = values["count"]
        if count:
            values["mean_point_error"] /= count
            values["normalized_mean_error"] /= count
    return totals
