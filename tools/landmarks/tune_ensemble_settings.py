#!/usr/bin/env python3
"""Compare landmark ensemble settings from cached predictions."""

from __future__ import annotations

import argparse
import json

from lib.landmarks.eval.harness import run_quality_harness


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default="outputs/landmark_quality/ensemble_tuning.json")
    args = parser.parse_args(argv)
    variants = (
        "plain_average",
        "static_weighted",
        "static_weighted_outliers",
        "static_weighted_downweight",
        "weighted_median",
    )
    result = run_quality_harness(
        args.manifest,
        args.cache_dir,
        variants=variants,
        weights_path=args.weights,
        output_dir="outputs/landmark_quality/tune_ensemble",
    )
    with open(args.output, "w", encoding="utf-8") as outfile:
        json.dump(
            {"variants": variants, "overall": result["overall"]},
            outfile,
            indent=2,
            sort_keys=True,
        )
        outfile.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
