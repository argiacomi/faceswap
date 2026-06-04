#!/usr/bin/env python3
"""Train the stacked residual landmark regressor candidate (#223).

Builds runtime-visible training rows from the same cached candidate/sample data
used by resolver scorer training, fits a constrained residual model, and writes a
loadable ``stacked_regressor.json`` artifact plus a training metrics summary.

The trained candidate is disabled by default at runtime; install it into the
production bundle and enable ``use_stacked_landmark_regressor`` to use it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import typing as T
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    parse_candidates,
)
from lib.landmarks.ensemble.stacked_regressor import (
    DEFAULT_STACKED_CANDIDATE_NAME,
    OUTPUT_MODE_GLOBAL_TRANSFORM,
    OUTPUT_MODE_LANDMARK_RESIDUAL_68,
    OUTPUT_MODE_REGION_RESIDUAL,
    write_stacked_regressor,
)
from lib.landmarks.ensemble.stacked_regressor_training import (
    DEFAULT_STACKED_L2,
    STACKED_REGRESSOR_ARTIFACT,
    train_stacked_regressor,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("train_stacked_landmark_regressor")

TRAINING_METRICS_JSON = "stacked_regressor_training.json"


def _show_progress() -> bool:
    return logger.isEnabledFor(logging.INFO) and sys.stderr.isatty()


def _context_progress(values: T.Sequence[T.Any], desc: str) -> T.Iterable[T.Any]:
    return T.cast(
        T.Iterable[T.Any],
        tqdm(
            values,
            total=len(values),
            desc=desc,
            unit="sample",
            disable=not _show_progress(),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-manifest", type=Path)
    parser.add_argument("--gt-cache-dir", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-cache-dir", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument(
        "--output-mode",
        choices=(
            OUTPUT_MODE_GLOBAL_TRANSFORM,
            OUTPUT_MODE_REGION_RESIDUAL,
            OUTPUT_MODE_LANDMARK_RESIDUAL_68,
        ),
        default=OUTPUT_MODE_GLOBAL_TRANSFORM,
    )
    parser.add_argument("--base-candidate-policy", default="static_weighted")
    parser.add_argument("--candidate-name", default=DEFAULT_STACKED_CANDIDATE_NAME)
    parser.add_argument("--residual-clip-fraction", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=DEFAULT_STACKED_L2)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", default="stacked_residual_v1")
    parser.add_argument(
        "--allow-landmark-residual-68",
        action="store_true",
        help="Permit the full 68-point residual mode (gated; constrained modes preferred).",
    )
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument(
        "--allow-image-backfill",
        action="store_true",
        help="Compute image-aware runtime metadata for rows without stored metadata.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_all_contexts(args: argparse.Namespace, candidates: T.Sequence[str]) -> list[T.Any]:
    from lib.landmarks.ensemble.runtime_resolver_scorer_data import load_contexts

    contexts: list[T.Any] = []
    pairs = [
        ("gt", args.gt_manifest, args.gt_cache_dir),
        ("production", args.production_manifest, args.production_cache_dir),
    ]
    for source, manifest, cache_dir in pairs:
        if manifest is None or cache_dir is None:
            continue
        contexts.extend(
            load_contexts(
                manifest_path=manifest,
                cache_dir=cache_dir,
                weights_path=args.weights,
                candidates=candidates,
                source=source,
                failure_threshold=args.failure_threshold,
                outlier_threshold=args.outlier_threshold,
                allow_image_backfill=args.allow_image_backfill,
                progress=_context_progress,
            )
        )
    return contexts


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    if args.gt_manifest is None and args.production_manifest is None:
        parser.error("at least one of --gt-manifest / --production-manifest is required")

    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    contexts = _load_all_contexts(args, candidates)
    if not contexts:
        logger.error("no sample contexts loaded; nothing to train")
        return 1

    regressor, metrics = train_stacked_regressor(
        contexts,
        output_mode=args.output_mode,
        base_candidate_policy=args.base_candidate_policy,
        candidate_name=args.candidate_name,
        residual_clip_fraction=args.residual_clip_fraction,
        l2=args.l2,
        eval_fraction=args.eval_fraction,
        split_seed=args.split_seed,
        allow_landmark_residual_68=args.allow_landmark_residual_68,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = write_stacked_regressor(regressor, output_dir / STACKED_REGRESSOR_ARTIFACT)
    metrics_path = output_dir / TRAINING_METRICS_JSON
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote stacked regressor artifact to %s", artifact_path)
    logger.info("Wrote stacked regressor training metrics to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TRAINING_METRICS_JSON", "main"]
