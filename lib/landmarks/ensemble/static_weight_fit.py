#!/usr/bin/env python3
"""Library helper for fitting static landmark ensemble weights."""

from __future__ import annotations

import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.metrics import per_landmark_error
from lib.landmarks.datasets.manifest_io import filter_canonical_68_samples
from lib.landmarks.ensemble.weights import MODEL_NAMES, weights_from_errors
from lib.landmarks.evaluation.harness import load_manifest


def compute_static_weights(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str] = MODEL_NAMES,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute static weights and mean validation errors from cached predictions."""
    model_names = tuple(model.strip() for model in models if model.strip())
    if not model_names:
        raise ValueError("at least one model is required")
    cache = DiskPredictionCache(cache_dir)
    samples = filter_canonical_68_samples(
        load_manifest(manifest_path), context="static weight fit"
    )
    if not samples:
        raise ValueError("manifest contains no validation samples")

    errors: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    for sample in samples:
        truth = np.load(sample.landmarks).astype("float32")
        for model in model_names:
            prediction = cache.read(sample.sample_id, model)
            errors[model].append(per_landmark_error(prediction.landmarks, truth))
    mean_errors = {
        model: np.stack(model_errors, axis=0).mean(axis=0).astype("float32").tolist()
        for model, model_errors in errors.items()
    }
    return weights_from_errors(mean_errors), mean_errors


__all__ = ["compute_static_weights"]
