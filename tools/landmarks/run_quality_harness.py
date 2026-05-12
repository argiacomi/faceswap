#!/usr/bin/env python3
"""Run cache-driven landmark quality evaluation."""

from __future__ import annotations

import argparse

from lib.landmarks.eval.harness import run_quality_harness


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--output-dir", default="outputs/landmark_quality")
    parser.add_argument("--models", default="")
    parser.add_argument("--variants", default="plain_average,static_weighted")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--failure-threshold", type=float, default=0.08)
    args = parser.parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip()) or None
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    run_quality_harness(
        args.manifest,
        args.cache_dir,
        models=models,
        variants=variants,
        weights_path=args.weights,
        output_dir=args.output_dir,
        failure_threshold=args.failure_threshold,
    )
    print(f"Wrote landmark metrics to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
