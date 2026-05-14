#!/usr/bin/env python3
"""Compute static per-landmark ensemble weights from cached predictions."""

from __future__ import annotations

import argparse
import json
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.weights import (
    MODEL_NAMES,
    save_weights,
    weights_from_errors,
)
from lib.landmarks.eval.harness import load_manifest
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.metrics import per_landmark_error

DEFAULT_OUTPUT = "configs/ensemble/static_landmark_weights.json"


def compute_static_weights(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str] = MODEL_NAMES,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute static weights and mean validation errors from cached predictions."""
    model_names = tuple(model.strip() for model in models if model.strip())
    if not model_names:
        raise ValueError("at least one model is required")

    errors: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    cache = DiskPredictionCache(cache_dir)
    samples = load_manifest(manifest_path)
    if not samples:
        raise ValueError("manifest contains no validation samples")

    for sample in samples:
        truth = np.load(sample.landmarks).astype("float32")
        for model in model_names:
            prediction = cache.read(sample.sample_id, model)
            errors[model].append(per_landmark_error(prediction.landmarks, truth))

    mean_errors = {
        model: np.stack(values).mean(axis=0).tolist() for model, values in errors.items()
    }
    return weights_from_errors(mean_errors), mean_errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    weights, mean_errors = compute_static_weights(args.manifest, args.cache_dir, models)
    save_weights(args.output, weights)
    if args.report:
        dominant = np.asarray([weights[model] for model in weights]).argmax(axis=0)
        payload = {
            "models": list(weights),
            "mean_errors": mean_errors,
            "dominant_model_by_landmark": dominant.astype(int).tolist(),
        }
        with open(args.report, "w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, indent=2, sort_keys=True)
            outfile.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
