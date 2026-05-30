#!/usr/bin/env python3
"""Build full-GT production / runtime bucket aggregate metrics for the
landmark candidate-search corpus.

Issue #205. Consumes an existing ``candidate_table.csv`` (produced by
the runtime resolver scorer / training pipeline against the full GT
manifest) and emits:

* ``gt_runtime_bucket_metrics.json`` — per-bucket aggregate payload.
* ``gt_runtime_bucket_metrics.csv`` — one row per (bucket, candidate).

Intended invocation from inside a ``resolver_pipeline`` directory:

.. code-block:: shell

    rtk python tools/landmarks/build_gt_runtime_bucket_metrics.py \\
        --candidate-table outputs/.../candidate_table.csv \\
        --output-dir outputs/.../candidate_search \\
        --selected-candidate weighted_median
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.search.gt_runtime_bucket_metrics import (
    aggregate_runtime_bucket_metrics,
    load_candidate_table_csv,
    write_runtime_bucket_csv,
    write_runtime_bucket_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the full-GT candidate diagnostic table into per-runtime "
            "bucket production metrics (issue #205)."
        )
    )
    parser.add_argument(
        "--candidate-table",
        type=Path,
        required=True,
        help="Path to candidate_table.csv produced by the resolver scorer pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write gt_runtime_bucket_metrics.json + "
            "gt_runtime_bucket_metrics.csv into. Created if missing."
        ),
    )
    parser.add_argument(
        "--selected-candidate",
        type=str,
        default=None,
        help=(
            "Optional name of the currently-selected setup candidate "
            "(e.g. 'weighted_median'); included in the per-bucket "
            "'selected_candidate_*' fields when supplied."
        ),
    )
    parser.add_argument(
        "--single-model-candidates",
        type=str,
        default="fan,hrnet,spiga,orformer",
        help=(
            "Comma-separated list of single-model candidate names used for "
            "the 'best_single_candidate' tie-break. Default mirrors the "
            "current DEFAULT_RESOLVER_CANDIDATES single-model entries."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    table_path: Path = args.candidate_table.expanduser().resolve()
    if not table_path.is_file():
        raise SystemExit(f"candidate-table not found: {table_path}")

    output_dir: Path = args.output_dir.expanduser().resolve()

    rows = load_candidate_table_csv(table_path)
    if not rows:
        print(f"No rows found in {table_path}; nothing to aggregate.")
        return 0

    single_models = [s.strip() for s in args.single_model_candidates.split(",") if s.strip()]
    metrics = aggregate_runtime_bucket_metrics(
        rows,
        selected_candidate=args.selected_candidate,
        single_model_candidates=single_models,
    )

    json_path = write_runtime_bucket_json(metrics, output_dir / "gt_runtime_bucket_metrics.json")
    csv_path = write_runtime_bucket_csv(metrics, output_dir / "gt_runtime_bucket_metrics.csv")

    print(f"Aggregated {len(rows)} candidate-table rows across {len(metrics)} runtime bucket(s).")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
