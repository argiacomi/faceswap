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
    primary_policy = str(report.get("primary_scorer_policy") or "learned_quality_v3")
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
    feature_importances = getattr(scorer, "feature_importances", None)
    if isinstance(feature_importances, dict) and feature_importances:
        with (output_dir / SCORER_FEATURE_IMPORTANCE_CSV).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["feature", "importance", "coefficient", "abs_coefficient", "kind"],
            )
            writer.writeheader()
            for feature, importance in sorted(
                feature_importances.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            ):
                writer.writerow(
                    {
                        "feature": feature,
                        "importance": importance,
                        "coefficient": "",
                        "abs_coefficient": "",
                        "kind": "feature_importance",
                    }
                )
        return

    coefficients = getattr(scorer, "coefficients", None)
    if coefficients is None:
        return
    with (output_dir / SCORER_FEATURE_IMPORTANCE_CSV).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature", "importance", "coefficient", "abs_coefficient", "kind"],
        )
        writer.writeheader()
        for feature, coefficient in sorted(
            zip(scorer.features, coefficients, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        ):
            writer.writerow(
                {
                    "feature": feature,
                    "importance": "",
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                    "kind": "coefficient",
                }
            )


PROFILE_SCORER_METRICS_JSON = "profile_scorer_metrics.json"
PROFILE_REGION_REPORT_CSV = "profile_region_report.csv"


def write_profile_scorer_report(
    *,
    report: T.Mapping[str, T.Any],
    output_dir: Path,
) -> Path:
    """Write the profile-specific scorer metrics and per-bucket region report.

    ``report`` is the payload from
    :func:`lib.landmarks.ensemble.scorer_eval.profile_scorer_report`. Writes
    ``profile_scorer_metrics.json`` (full payload) plus a flat
    ``profile_region_report.csv`` with one row per profile bucket.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / PROFILE_SCORER_METRICS_JSON
    write_json(metrics_path, dict(report))

    buckets = report.get("buckets", {})
    fieldnames = [
        "bucket",
        "transform_group_count_v3",
        "mean_transform_regret_v3",
        "p95_transform_regret_v3",
        "avoidable_invalid_selection_rate_v3",
        "all_invalid_selected_rate_v3",
        "validity_stage_valid_rankable_count_v3",
        "validity_stage_profile_soft_valid_count_v3",
        "validity_stage_all_invalid_count_v3",
        "profile_all_invalid_degraded_fallback_count",
    ]
    with (output_dir / PROFILE_REGION_REPORT_CSV).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bucket, summary in buckets.items():
            row = {"bucket": bucket}
            row.update({key: summary.get(key, 0) for key in fieldnames[1:]})
            writer.writerow(row)
    return metrics_path


__all__ = ["write_profile_scorer_report", "write_scorer_policy_outputs"]
