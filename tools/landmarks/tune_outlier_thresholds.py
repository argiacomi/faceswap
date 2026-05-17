#!/usr/bin/env python3
"""Sweep landmark outlier thresholds from cached predictions."""

from __future__ import annotations

import argparse
import json

from lib.landmarks.evaluation.harness import run_quality_harness


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default="outputs/landmark_quality/outlier_tuning.json")
    parser.add_argument("--thresholds", default="2.0,2.5,3.0,3.5,4.0")
    args = parser.parse_args(argv)
    candidates = []
    for raw in args.thresholds.split(","):
        threshold = float(raw.strip())
        result = run_quality_harness(
            args.manifest,
            args.cache_dir,
            variants=("static_weighted_outliers", "static_weighted_downweight", "weighted_median"),
            weights_path=args.weights,
            outlier_threshold=threshold,
            output_dir=f"outputs/landmark_quality/tune_{threshold:g}",
        )
        candidates.append({"threshold": threshold, "overall": result["overall"]})
    with open(args.output, "w", encoding="utf-8") as outfile:
        json.dump({"candidates": candidates}, outfile, indent=2, sort_keys=True)
        outfile.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
