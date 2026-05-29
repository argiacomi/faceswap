#!/usr/bin/env python3
"""Build image-aware production resolver metadata for promotion gates."""

from __future__ import annotations

import argparse
import sys
import typing as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    DEFAULT_OUTLIER_THRESHOLD,
)
from lib.landmarks.pipeline_conventions import SOURCE_PRODUCTION_VALIDATED
from tools.landmarks.build_gt_hard_resolver_metadata import (
    _parse_candidates_arg,
    build_gt_hard_resolver_metadata,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument(
        "--image-backfill-crop-scale",
        type=float,
        default=DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    )
    parser.add_argument(
        "--image-backfill-crop-size",
        type=int,
        default=DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    build_gt_hard_resolver_metadata(
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        weights=args.weights,
        candidates=_parse_candidates_arg(args.candidates),
        output=args.output,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
        allow_image_backfill=True,
        image_backfill_crop_scale=args.image_backfill_crop_scale,
        image_backfill_crop_size=args.image_backfill_crop_size,
        source=SOURCE_PRODUCTION_VALIDATED,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
