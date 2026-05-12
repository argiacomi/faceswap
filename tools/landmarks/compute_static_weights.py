#!/usr/bin/env python3
"""Compute static per-landmark ensemble weights from cached predictions."""

from __future__ import annotations

import argparse
import json

import numpy as np

from lib.landmarks.ensemble.weights import save_weights, weights_from_errors
from lib.landmarks.eval.harness import load_manifest
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.metrics import per_landmark_error


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", default="configs/landmarks/static_landmark_weights.json")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    errors = {model: [] for model in models}
    cache = DiskPredictionCache(args.cache_dir)
    for sample in load_manifest(args.manifest):
        truth = np.load(sample.landmarks).astype("float32")
        for model in models:
            prediction = cache.read(sample.sample_id, model)
            errors[model].append(per_landmark_error(prediction.landmarks, truth))
    mean_errors = {
        model: np.stack(values).mean(axis=0).tolist() for model, values in errors.items() if values
    }
    weights = weights_from_errors(mean_errors)
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
