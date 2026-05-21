#!/usr/bin/env python3
"""Train a candidate-quality scorer for the landmark runtime resolver."""

from __future__ import annotations

import argparse
import logging
import sys
import typing as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_HIGH_GAP_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SCORER_CANDIDATE_CSV,
    parse_candidates,
)
from lib.landmarks.ensemble.scorer_target_config import (
    SCORER_TARGETS,
    TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
)
from lib.landmarks.ensemble.scorer_training import (
    EVAL_ROWS_CSV,
    SCORER_ARTIFACT,
    SCORER_V2_ARTIFACT,
    TRAINING_CANDIDATE_TABLE_CSV,
    TRAINING_METRICS_JSON,
    TRAINING_ROWS_CSV,
    train_runtime_resolver_scorer,
    train_runtime_resolver_scorer_v2,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("train_runtime_resolver_scorer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-manifest", type=Path)
    parser.add_argument("--gt-cache-dir", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-cache-dir", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        default="",
        help=f"Comma-separated candidate list. Defaults to {DEFAULT_SCORER_CANDIDATE_CSV}.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--high-gap-threshold", type=float, default=DEFAULT_HIGH_GAP_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument(
        "--target",
        choices=SCORER_TARGETS,
        default=TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
        help=(
            "Training target. The default preserves the binary v1 classifier; "
            "normalized_regret and selection_cost train a v1.1 linear regressor."
        ),
    )
    parser.add_argument(
        "--training-mode",
        choices=("learned_quality_v1", "continuous_regret_v1_1", "learned_quality_v2"),
        default="",
        help=(
            "Explicit scorer training mode. learned_quality_v2 trains a LightGBM "
            "LambdaRank artifact and writes runtime_resolver_scorer_v2.json."
        ),
    )
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--allow-image-backfill",
        action="store_true",
        help="Compute image-aware runtime metadata for non-GT-hard rows without stored metadata.",
    )
    parser.add_argument(
        "--gt-hard-resolver-metadata",
        type=Path,
        default=None,
        help="Frozen GT-hard resolver_metadata.jsonl sidecar keyed by sample_id and face_index.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    if args.training_mode == "learned_quality_v2":
        metrics = train_runtime_resolver_scorer_v2(
            gt_manifest=args.gt_manifest,
            gt_cache_dir=args.gt_cache_dir,
            production_manifest=args.production_manifest,
            production_cache_dir=args.production_cache_dir,
            weights_path=args.weights,
            candidates=candidates,
            output_dir=args.output_dir,
            gt_hard_resolver_metadata=args.gt_hard_resolver_metadata,
            failure_threshold=args.failure_threshold,
            high_gap_threshold=args.high_gap_threshold,
            outlier_threshold=args.outlier_threshold,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            eval_fraction=args.eval_fraction,
            split_seed=args.split_seed,
            allow_image_backfill=args.allow_image_backfill,
        )
    else:
        metrics = train_runtime_resolver_scorer(
            gt_manifest=args.gt_manifest,
            gt_cache_dir=args.gt_cache_dir,
            production_manifest=args.production_manifest,
            production_cache_dir=args.production_cache_dir,
            weights_path=args.weights,
            candidates=candidates,
            output_dir=args.output_dir,
            gt_hard_resolver_metadata=args.gt_hard_resolver_metadata,
            failure_threshold=args.failure_threshold,
            high_gap_threshold=args.high_gap_threshold,
            outlier_threshold=args.outlier_threshold,
            l2=args.l2,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            eval_fraction=args.eval_fraction,
            split_seed=args.split_seed,
            allow_image_backfill=args.allow_image_backfill,
            target=args.target,
        )
    logger.info("Wrote runtime resolver scorer to %s", metrics["artifact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVAL_ROWS_CSV",
    "SCORER_ARTIFACT",
    "SCORER_V2_ARTIFACT",
    "TRAINING_CANDIDATE_TABLE_CSV",
    "TRAINING_METRICS_JSON",
    "TRAINING_ROWS_CSV",
    "main",
    "train_runtime_resolver_scorer",
    "train_runtime_resolver_scorer_v2",
]
