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
    EXPERT_POLICY_RUNTIME_BUCKET,
    EXPERT_POLICY_SINGLE,
    OUTPUT_MODE_GLOBAL_TRANSFORM,
    OUTPUT_MODE_LANDMARK_RESIDUAL_68,
    OUTPUT_MODE_REGION_RESIDUAL,
    write_stacked_regressor,
)
from lib.landmarks.ensemble.stacked_regressor_context_cache import (
    load_contexts_maybe_parallel,
    load_or_build_stacked_contexts,
)
from lib.landmarks.ensemble.stacked_regressor_training import (
    CONTEXT_SCOPE_ALL,
    CONTEXT_SCOPES,
    DEFAULT_STACKED_EXPERT_MIN_EXAMPLES,
    DEFAULT_STACKED_L2,
    SAMPLE_WEIGHT_POLICIES,
    SAMPLE_WEIGHT_POLICY_UNIFORM,
    STACKED_REGRESSOR_ARTIFACT,
    TARGET_POLICIES,
    TARGET_POLICY_FULL_RESIDUAL,
    train_stacked_regressor,
    train_stacked_regressor_experts,
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
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--context-cache",
        type=Path,
        help=(
            "Optional pickle cache for prebuilt SampleCandidateContext objects. "
            "Use this to avoid rebuilding contexts for every stacked-regressor sweep."
        ),
    )
    parser.add_argument(
        "--context-scope",
        choices=tuple(sorted(CONTEXT_SCOPES)),
        default=CONTEXT_SCOPE_ALL,
        help=(
            "Filter contexts for experimental training. non_profile excludes "
            "profile buckets; profile_only trains only profile buckets. This is "
            "not a runtime gate."
        ),
    )
    parser.add_argument(
        "--runtime-context-scope",
        choices=tuple(sorted(CONTEXT_SCOPES)),
        default=CONTEXT_SCOPE_ALL,
        help=(
            "Runtime bucket scope supported by the emitted stacked regressor artifact. "
            "Buckets outside this scope skip stacked_residual candidate generation."
        ),
    )
    parser.add_argument(
        "--rebuild-context-cache",
        action="store_true",
        help="Ignore an existing --context-cache and rebuild it from manifests/caches.",
    )
    parser.add_argument(
        "--context-workers",
        type=int,
        default=0,
        help="Parallel workers for building contexts before caching. 0/1 means serial.",
    )
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
    parser.add_argument(
        "--target-policy",
        choices=tuple(sorted(TARGET_POLICIES)),
        default=TARGET_POLICY_FULL_RESIDUAL,
        help=(
            "Training target policy. clipped_residual fits the correction after "
            "runtime residual clipping, aligning training with what inference can apply."
        ),
    )
    parser.add_argument(
        "--expert-policy",
        choices=(EXPERT_POLICY_SINGLE, EXPERT_POLICY_RUNTIME_BUCKET),
        default=EXPERT_POLICY_SINGLE,
        help=(
            "Train one global stacked regressor or one deterministic runtime-bucket "
            "expert artifact. runtime_bucket still emits one stacked_residual "
            "candidate; it only changes which fitted expert produces the residual."
        ),
    )
    parser.add_argument(
        "--expert-min-examples",
        type=int,
        default=DEFAULT_STACKED_EXPERT_MIN_EXAMPLES,
        help="Minimum context count required to train a non-default bucket expert.",
    )
    parser.add_argument(
        "--sample-weight-policy",
        choices=tuple(sorted(SAMPLE_WEIGHT_POLICIES)),
        default=SAMPLE_WEIGHT_POLICY_UNIFORM,
        help=(
            "Training objective weighting policy. Use hard_slice or "
            "hard_slice_plus_base_error to emphasize profile/large-yaw examples."
        ),
    )
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
    contexts: list[T.Any] = []
    pairs = [
        ("gt", args.gt_manifest, args.gt_cache_dir),
        ("production", args.production_manifest, args.production_cache_dir),
    ]
    if args.weights is None:
        raise ValueError("--weights is required when building stacked regressor contexts")
    for source, manifest, cache_dir in pairs:
        if manifest is None or cache_dir is None:
            continue
        contexts.extend(
            load_contexts_maybe_parallel(
                manifest_path=manifest,
                cache_dir=cache_dir,
                weights_path=args.weights,
                candidates=candidates,
                source=source,
                failure_threshold=args.failure_threshold,
                outlier_threshold=args.outlier_threshold,
                allow_image_backfill=args.allow_image_backfill,
                workers=max(int(args.context_workers), 0),
                progress=_context_progress,
                logger=logger,
            )
        )
    return contexts


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    if (
        args.context_cache is not None
        and not args.rebuild_context_cache
        and not args.context_cache.is_file()
    ):
        parser.error(
            f"--context-cache does not exist: {args.context_cache}. "
            "Set CONTEXT_CACHE to an existing cache, or pass --rebuild-context-cache "
            "with --weights to build it."
        )

    needs_context_build = (
        args.context_cache is None
        or args.rebuild_context_cache
        or not args.context_cache.is_file()
    )
    if needs_context_build:
        if args.weights is None:
            parser.error("--weights is required when building stacked regressor contexts")
        if args.gt_manifest is None and args.production_manifest is None:
            parser.error("at least one of --gt-manifest / --production-manifest is required")

    if args.weights is not None:
        weights = load_weights(args.weights)
        candidates = parse_candidates(args.candidates, weights)
    else:
        if args.candidates:
            parser.error("--weights is required when --candidates is provided")
        candidates = ()

    contexts = load_or_build_stacked_contexts(
        context_cache=args.context_cache,
        rebuild_context_cache=args.rebuild_context_cache,
        build=lambda: _load_all_contexts(args, candidates),
        logger=logger,
        metadata={
            "tool": "train_stacked_landmark_regressor",
            "gt_manifest": "" if args.gt_manifest is None else str(args.gt_manifest),
            "gt_cache_dir": "" if args.gt_cache_dir is None else str(args.gt_cache_dir),
            "production_manifest": (
                "" if args.production_manifest is None else str(args.production_manifest)
            ),
            "production_cache_dir": (
                "" if args.production_cache_dir is None else str(args.production_cache_dir)
            ),
            "weights": "" if args.weights is None else str(args.weights),
            "candidates": list(candidates),
            "failure_threshold": args.failure_threshold,
            "outlier_threshold": args.outlier_threshold,
            "allow_image_backfill": args.allow_image_backfill,
        },
    )
    if not contexts:
        logger.error("no sample contexts loaded; nothing to train")
        return 1

    common_kwargs: dict[str, T.Any] = {
        "context_scope": args.context_scope,
        "runtime_context_scope": args.runtime_context_scope,
        "output_mode": args.output_mode,
        "base_candidate_policy": args.base_candidate_policy,
        "candidate_name": args.candidate_name,
        "residual_clip_fraction": args.residual_clip_fraction,
        "l2": args.l2,
        "eval_fraction": args.eval_fraction,
        "split_seed": args.split_seed,
        "allow_landmark_residual_68": args.allow_landmark_residual_68,
    }
    if hasattr(args, "sample_weight_policy"):
        common_kwargs["sample_weight_policy"] = args.sample_weight_policy
    common_kwargs["target_policy"] = args.target_policy

    if args.expert_policy == EXPERT_POLICY_RUNTIME_BUCKET:
        regressor, metrics = train_stacked_regressor_experts(
            contexts,
            **common_kwargs,
            expert_min_examples=args.expert_min_examples,
        )
    else:
        regressor, metrics = train_stacked_regressor(
            contexts,
            **common_kwargs,
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
