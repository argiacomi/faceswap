#!/usr/bin/env python3
"""Shared report writers for runtime resolver scorer tools."""

from __future__ import annotations

import csv
import typing as T
from pathlib import Path

from lib.landmarks.pipeline_conventions import (
    SCORER_FEATURE_IMPORTANCE_CSV,
    SCORER_HELDOUT_POLICY_REPORT_JSON,
    SCORER_METRICS_JSON,
    SCORER_POLICY_REPORT_CSV,
    SCORER_POLICY_REPORT_JSON,
    SCORER_WORST_SAMPLES_JSON,
    write_json,
)


def write_scorer_policy_outputs(
    *,
    report: dict[str, T.Any],
    rows: T.Sequence[dict[str, T.Any]],
    scorer: T.Any,
    output_dir: Path,
    worst_sample_count: int,
) -> None:
    """Write the standard scorer policy report bundle."""
    primary_policy = str(report.get("primary_scorer_policy") or "learned_quality_v1_1")
    write_json(output_dir / SCORER_METRICS_JSON, report[primary_policy])
    write_json(output_dir / SCORER_POLICY_REPORT_JSON, report)
    if report.get("heldout_eval"):
        write_json(output_dir / SCORER_HELDOUT_POLICY_REPORT_JSON, report)
    if rows:
        with (output_dir / SCORER_POLICY_REPORT_CSV).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    worst = sorted(rows, key=lambda row: float(row["gap_vs_oracle"]), reverse=True)[
        :worst_sample_count
    ]
    write_json(output_dir / SCORER_WORST_SAMPLES_JSON, {"samples": worst})
    with (output_dir / SCORER_FEATURE_IMPORTANCE_CSV).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "coefficient", "abs_coefficient"])
        writer.writeheader()
        for feature, coefficient in sorted(
            zip(scorer.features, scorer.coefficients, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        ):
            writer.writerow(
                {
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )


__all__ = ["write_scorer_policy_outputs"]
