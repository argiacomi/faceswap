#!/usr/bin/env python3
"""Landmark ensemble tool helpers.

Deprecated public CLI files that were merged into the pipeline are not present
on disk anymore. A few legacy orchestrators still import their historical
module names internally, so this package registers compatibility aliases that
point at library implementations without restoring the deleted entrypoint files.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

from lib.landmarks.ensemble import static_weight_fit as compute_static_weights
from lib.landmarks.evaluation.failure_report import write_failure_report_from_metrics


def _failure_viewer_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for the deleted failure_viewer CLI."
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--debug-dir", default="outputs/landmark_debug")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    args = parser.parse_args(argv)
    write_failure_report_from_metrics(
        metrics_path=Path(args.metrics),
        output_dir=Path(args.output_dir or args.debug_dir),
        limit=args.limit,
    )
    return 0


failure_viewer = types.ModuleType("tools.landmarks.failure_viewer")
failure_viewer.__doc__ = "Compatibility module for deleted failure_viewer CLI."
failure_viewer.main = _failure_viewer_main  # type: ignore[attr-defined]

sys.modules.setdefault("tools.landmarks.compute_static_weights", compute_static_weights)
sys.modules.setdefault("tools.landmarks.failure_viewer", failure_viewer)

__all__ = ["compute_static_weights", "failure_viewer"]
