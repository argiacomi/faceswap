#!/usr/bin/env python3
"""Train a candidate-quality scorer for the landmark runtime resolver."""

from __future__ import annotations

import argparse
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
    DEFAULT_HIGH_GAP_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SCORER_CANDIDATE_CSV,
    parse_candidates,
)
from lib.landmarks.ensemble.scorer_target_config import (
    TARGET_TRANSFORM_REGRET_V3,
)
from lib.landmarks.ensemble.scorer_training import (
    EVAL_ROWS_CSV,
    SCORER_ARTIFACT,
    SCORER_V3_ARTIFACT,
    TRAINING_CANDIDATE_TABLE_CSV,
    TRAINING_METRICS_JSON,
    TRAINING_ROWS_CSV,
    train_runtime_resolver_scorer,
    train_runtime_resolver_scorer_suite,
    train_runtime_resolver_scorer_v3,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("train_runtime_resolver_scorer")


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
        choices=(TARGET_TRANSFORM_REGRET_V3,),
        default=TARGET_TRANSFORM_REGRET_V3,
        help=("Active scorer_suite target. v3 training uses only transform_alignment_regret_v3."),
    )
    parser.add_argument(
        "--training-mode",
        choices=("scorer_suite",),
        default="scorer_suite",
        help="Explicit scorer training mode. Writes the canonical learned_quality_v3 artifact.",
    )
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--num-leaves", type=int, default=31)
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
    parser.add_argument(
        "--profile39-manifest",
        type=Path,
        default=None,
        help=(
            "Manifest with 39-point profile GT. When given with --profile39-cache-dir, the "
            "profile specialist (learned_quality_v3_profile) is trained from the partial-schema "
            "39-point rows. 39-point GT never enters the canonical-68 scorer target."
        ),
    )
    parser.add_argument(
        "--profile39-cache-dir",
        type=Path,
        default=None,
        help="Prediction cache directory for the --profile39-manifest samples.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    metrics = train_runtime_resolver_scorer_suite(
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
        num_leaves=args.num_leaves,
        eval_fraction=args.eval_fraction,
        split_seed=args.split_seed,
        allow_image_backfill=args.allow_image_backfill,
        profile39_manifest=args.profile39_manifest,
        profile39_cache_dir=args.profile39_cache_dir,
        progress=_context_progress,
    )
    logger.info("Wrote runtime resolver scorer artifacts to %s", metrics["artifact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVAL_ROWS_CSV",
    "SCORER_ARTIFACT",
    "SCORER_V3_ARTIFACT",
    "TRAINING_CANDIDATE_TABLE_CSV",
    "TRAINING_METRICS_JSON",
    "TRAINING_ROWS_CSV",
    "main",
    "train_runtime_resolver_scorer",
    "train_runtime_resolver_scorer_suite",
    "train_runtime_resolver_scorer_v3",
]
