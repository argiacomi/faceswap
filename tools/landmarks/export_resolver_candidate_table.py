#!/usr/bin/env python3
"""Export canonical resolver candidate diagnostics and target metrics."""

from __future__ import annotations

import argparse
import logging
import sys
import typing as T
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SCORER_CANDIDATE_CSV,
    export_candidate_table,
    parse_candidates,
    write_candidate_table_csv,
    write_candidate_table_parquet,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("export_resolver_candidate_table")


def export_resolver_candidate_table(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str],
    output: Path | None = None,
    output_csv: Path | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    allow_image_backfill: bool = False,
) -> dict[str, T.Any]:
    """Build and write the canonical candidate diagnostic table."""
    rows = export_candidate_table(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=candidates,
        failure_threshold=failure_threshold,
        outlier_threshold=outlier_threshold,
        allow_image_backfill=allow_image_backfill,
    )
    artifacts: dict[str, str] = {}
    if output_csv is not None:
        artifacts["csv"] = str(write_candidate_table_csv(rows, output_csv))
    if output is not None:
        artifacts["parquet"] = str(write_candidate_table_parquet(rows, output))
    return {
        "row_count": len(rows),
        "candidates": list(candidates),
        "artifacts": artifacts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        default=DEFAULT_SCORER_CANDIDATE_CSV,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument(
        "--allow-image-backfill",
        action="store_true",
        help="Compute image-aware runtime metadata for rows without stored metadata.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    if args.output is None and args.output_csv is None:
        parser.error("at least one of --output or --output-csv is required")
    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    report = export_resolver_candidate_table(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        weights_path=args.weights,
        candidates=candidates,
        output=args.output,
        output_csv=args.output_csv,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
        allow_image_backfill=args.allow_image_backfill,
    )
    logger.info("Exported %d candidate rows", report["row_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
