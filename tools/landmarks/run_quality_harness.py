#!/usr/bin/env python3
"""Run cache-driven landmark quality evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument("--max-nme", type=float, default=None)
    parser.add_argument("--max-failure-rate", type=float, default=None)
    args = parser.parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip()) or None
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    result = run_quality_harness(
        args.manifest,
        args.cache_dir,
        models=models,
        variants=variants,
        weights_path=args.weights,
        output_dir=args.output_dir,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
    )
    print(f"Wrote landmark metrics to: {args.output_dir}")
    failures = []
    if result.get("threshold_failed"):
        failures.append(f"sample failure threshold exceeded ({args.failure_threshold:.6f})")
    for label, metrics in sorted(result.get("overall", {}).items()):
        if args.max_nme is not None and metrics.get("nme", 0.0) > args.max_nme:
            failures.append(f"{label} nme={metrics['nme']:.6f} > {args.max_nme:.6f}")
        if (
            args.max_failure_rate is not None
            and metrics.get("failure_rate", 0.0) > args.max_failure_rate
        ):
            failures.append(
                f"{label} failure_rate={metrics['failure_rate']:.6f} > {args.max_failure_rate:.6f}"
            )
    if failures:
        print("Landmark quality thresholds failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
