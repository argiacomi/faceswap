#!/usr/bin/env python3
"""Prebuild stacked-regressor sample contexts for fast train/eval sweeps."""

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
    DEFAULT_OUTLIER_THRESHOLD,
    parse_candidates,
)
from lib.landmarks.ensemble.stacked_regressor_context_cache import (
    load_contexts_maybe_parallel,
    write_stacked_context_cache,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("build_stacked_regressor_contexts")


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--context-workers", type=int, default=0)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument("--allow-image-backfill", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_all_contexts(args: argparse.Namespace, candidates: T.Sequence[str]) -> list[T.Any]:
    contexts: list[T.Any] = []
    for source, manifest, cache_dir in (
        ("gt", args.gt_manifest, args.gt_cache_dir),
        ("production", args.production_manifest, args.production_cache_dir),
    ):
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
    if args.gt_manifest is None and args.production_manifest is None:
        parser.error("at least one of --gt-manifest / --production-manifest is required")

    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    contexts = _load_all_contexts(args, candidates)
    if not contexts:
        logger.error("no sample contexts loaded; nothing to cache")
        return 1

    path = write_stacked_context_cache(
        args.output,
        contexts,
        metadata={
            "tool": "build_stacked_regressor_contexts",
            "gt_manifest": "" if args.gt_manifest is None else str(args.gt_manifest),
            "gt_cache_dir": "" if args.gt_cache_dir is None else str(args.gt_cache_dir),
            "production_manifest": (
                "" if args.production_manifest is None else str(args.production_manifest)
            ),
            "production_cache_dir": (
                "" if args.production_cache_dir is None else str(args.production_cache_dir)
            ),
            "weights": str(args.weights),
            "candidates": list(candidates),
            "context_workers": int(args.context_workers),
            "failure_threshold": float(args.failure_threshold),
            "outlier_threshold": float(args.outlier_threshold),
            "allow_image_backfill": bool(args.allow_image_backfill),
        },
    )
    logger.info("Wrote %d stacked regressor context(s) to %s", len(contexts), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
