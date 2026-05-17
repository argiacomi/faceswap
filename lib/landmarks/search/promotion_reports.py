#!/usr/bin/env python3
"""Markdown promotion-report writer.

Lives in the search package because the report is the human-readable face
of a candidate-search run: it summarizes the promoted winner, held-out
report metrics, effective-ensemble diagnostics, profile aggregates, and
the gate outcomes that drove the decision. The writer was previously
inlined in the search CLI; extracting it here lets other surfaces
(experiment runners, dry-run validators) reuse the same format.
"""

from __future__ import annotations

import typing as T
from pathlib import Path

from lib.landmarks.ensemble.promoted_setup import PROMOTION_REPORT_FILENAME
from lib.landmarks.evaluation.profile_metrics import ProfileAggregate
from lib.landmarks.search.candidate_search import CandidateResult
from lib.landmarks.search.promotion_gates import GateApplication


def write_promotion_report(
    output_dir: Path,
    *,
    winner: CandidateResult,
    results: T.Sequence[CandidateResult],
    objective: str,
    regression_epsilon_nme: float,
    report_metrics: T.Mapping[str, T.Any],
    gate_application: GateApplication | None = None,
    profile_aggregate: ProfileAggregate | None = None,
) -> Path:
    """Render ``promotion_report.md`` for one candidate-search run.

    Returns the file path written. ``profile_aggregate`` and
    ``gate_application`` are optional; the corresponding sections are
    omitted from the report when ``None``.
    """
    path = output_dir / PROMOTION_REPORT_FILENAME
    lines = [
        "# Promotion Report",
        "",
        f"- objective: `{objective}`",
        f"- regression_epsilon_nme: `{regression_epsilon_nme}`",
        f"- evaluated_candidates: `{len(results)}`",
        "",
        "## Winner",
        "",
        f"- candidate_id: `{winner.candidate_id}`",
        f"- models: `{', '.join(winner.candidate.models)}`",
        f"- weight_generator: `{winner.candidate.weight_generator}`",
        f"- strategy: `{winner.candidate.strategy}`",
        f"- outlier_threshold: `{winner.candidate.outlier_threshold}`",
        f"- selection_score: `{winner.score:.6f}`",
        f"- selection_nme: `{winner.metrics.overall_nme:.6f}`",
        f"- selection_failure_rate: `{winner.metrics.failure_rate:.6f}`",
        f"- selection_regression_rate: `{winner.metrics.regression_rate_vs_best_single:.6f}`",
        f"- bucket_regression_rate: `{winner.metrics.bucket_regression_rate_vs_best_single:.6f}`",
        "",
        "## Held-out report metrics",
        "",
        f"- report_nme: `{report_metrics.get('overall_nme', 0.0):.6f}`",
        f"- report_failure_rate: `{report_metrics.get('failure_rate', 0.0):.6f}`",
        f"- report_regression_rate: `{report_metrics.get('regression_rate_vs_best_single', 0.0):.6f}`",
        f"- report_bucket_regression_rate: `{report_metrics.get('bucket_regression_rate_vs_best_single', 0.0):.6f}`",
    ]
    if winner.effective_ensemble is not None:
        diag = winner.effective_ensemble
        lines.extend(
            [
                "",
                "## Effective ensemble diagnostics",
                "",
                f"- mean_effective_models: `{diag.mean_effective_models:.3f}` (floor `{diag.effective_models_floor:.3f}`)",
                f"- collapsed: `{diag.collapsed}`",
                f"- weighted_median_collapsed: `{diag.weighted_median_collapsed}`",
                "- landmark share by model: "
                + ", ".join(
                    f"{model}={share:.2f}"
                    for model, share in sorted(
                        diag.landmark_share_by_model.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ),
            ]
        )
        if winner.is_single_model_baseline:
            lines.append(
                "- note: this winner is a single-model baseline; promotion of single-model setups was explicitly allowed."
            )
    if gate_application is not None:
        lines.extend(
            [
                "",
                "## Promotion gates",
                "",
                f"- gates_passed: `{gate_application.passed_count}`",
                f"- gates_failed: `{gate_application.failed_count}`",
            ]
        )
        if gate_application.promoted_outcome is not None:
            lines.append("- selected candidate cleared every active gate.")
        if profile_aggregate is not None:
            lines.extend(
                [
                    "",
                    "## Profile metrics (held-out report split)",
                    "",
                    f"- profile_overall_score: `{profile_aggregate.overall_score:.6f}`",
                    f"- profile_region_failure_rate: `{profile_aggregate.region_failure_rate:.6f}`",
                    f"- profile_p90_visible_error: `{profile_aggregate.p90_visible_error:.6f}`",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["write_promotion_report"]
