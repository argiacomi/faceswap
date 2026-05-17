#!/usr/bin/env python3
"""Write landmark debug overlays from cached predictions."""

from __future__ import annotations

import argparse

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.evaluation.harness import load_manifest
from lib.landmarks.evaluation.visualize import write_debug_records, write_overlay


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--output-dir", default="outputs/landmark_debug")
    parser.add_argument("--models", required=True)
    args = parser.parse_args(argv)
    cache = DiskPredictionCache(args.cache_dir)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    records = []
    for sample in load_manifest(args.manifest):
        predictions = {
            model: cache.read(sample.sample_id, model).landmarks
            for model in models
            if model in cache.available_models(sample.sample_id)
        }
        overlay = write_overlay(
            sample.image, predictions, f"{args.output_dir}/{sample.sample_id}.png"
        )
        records.append(
            {
                "sample_id": sample.sample_id,
                "overlay": str(overlay),
                "models": ",".join(predictions),
            }
        )
    write_debug_records(records, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
