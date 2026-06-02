#!/usr/bin/env python3
"""Evaluate learned runtime resolver scorer policy against resolver baselines."""

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
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SCORER_CANDIDATE_CSV,
    parse_candidates,
)
from lib.landmarks.ensemble.scorer_eval import (
    DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK,
    DEFAULT_SAFE_FALLBACK_MIN_DELTA,
    PROMOTION_SCOPES,
    evaluate_runtime_resolver_scorer,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("evaluate_runtime_resolver_scorer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-manifest", type=Path)
    parser.add_argument("--gt-cache-dir", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-cache-dir", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument(
        "--v2-scorer",
        type=Path,
        default=None,
        help="Optional learned_quality_v2 LightGBM scorer artifact to compare.",
    )
    parser.add_argument(
        "--eval-split",
        type=Path,
        help="Optional scorer eval-row CSV; when supplied, policy metrics use only held-out rows.",
    )
    parser.add_argument(
        "--scorer-rows",
        type=Path,
        default=None,
        help=(
            "Canonical scorer rows CSV from scorer_training/scorer_dataset/rows.csv. "
            "When supplied without --eval-split, evaluation filters to rows marked split=eval."
        ),
    )
    parser.add_argument(
        "--scorer-dataset",
        type=Path,
        default=None,
        help=(
            "Canonical scorer_dataset directory or manifest. Used to resolve rows.csv "
            "when --scorer-rows is omitted."
        ),
    )
    parser.add_argument(
        "--candidates",
        default="",
        help=f"Comma-separated candidate list. Defaults to {DEFAULT_SCORER_CANDIDATE_CSV}.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument("--epsilon-mean-nme", type=float, default=0.001)
    parser.add_argument("--epsilon-failure-rate", type=float, default=0.0)
    parser.add_argument("--worst-sample-count", type=int, default=25)
    parser.add_argument(
        "--risk-floor-for-safe-fallback",
        type=float,
        default=DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK,
    )
    parser.add_argument(
        "--safe-fallback-min-delta",
        type=float,
        default=DEFAULT_SAFE_FALLBACK_MIN_DELTA,
        help="Require a safe-fallback replacement risk score to beat selected risk by this margin.",
    )
    parser.add_argument(
        "--promotion-scope",
        choices=PROMOTION_SCOPES,
        default="universal",
        help="Gate production only, or require both production and GT-hard diagnostics to pass.",
    )
    parser.add_argument(
        "--promotion-policy",
        default="",
        help=(
            "Explicit policy to gate for promotion. Internal scorer/static gates remain "
            "diagnostic only."
        ),
    )
    parser.add_argument(
        "--installed-scorer-dir",
        type=Path,
        default=Path(".fs_cache/landmark_ensemble/current/scorers"),
        help="Directory containing currently installed production scorer artifacts.",
    )
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
        "--allow-derived-no-image-gt-hard",
        action="store_true",
        help="Allow GT hard diagnostics to use landmark-only derived runtime buckets.",
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
    report = evaluate_runtime_resolver_scorer(
        gt_manifest=args.gt_manifest,
        gt_cache_dir=args.gt_cache_dir,
        production_manifest=args.production_manifest,
        production_cache_dir=args.production_cache_dir,
        weights_path=args.weights,
        scorer_path=args.scorer,
        v2_scorer_path=args.v2_scorer,
        candidates=candidates,
        output_dir=args.output_dir,
        eval_split=args.eval_split,
        scorer_rows=args.scorer_rows,
        scorer_dataset=args.scorer_dataset,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
        epsilon_mean_nme=args.epsilon_mean_nme,
        epsilon_failure_rate=args.epsilon_failure_rate,
        worst_sample_count=args.worst_sample_count,
        risk_floor_for_safe_fallback=args.risk_floor_for_safe_fallback,
        safe_fallback_min_delta=args.safe_fallback_min_delta,
        promotion_scope=args.promotion_scope,
        promotion_policy=args.promotion_policy,
        installed_scorer_dir=args.installed_scorer_dir,
        allow_image_backfill=args.allow_image_backfill,
        allow_derived_no_image_gt_hard=args.allow_derived_no_image_gt_hard,
        gt_hard_resolver_metadata=args.gt_hard_resolver_metadata,
    )
    logger.info("Scorer policy status: %s", report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_runtime_resolver_scorer", "main"]
