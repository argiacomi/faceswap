#!/usr/bin/env python3
"""Search the landmark ensemble setup space and emit promoted artifacts (#69).

Reads a manifest, a populated prediction cache, and a fit/select/report split
assignment (#67), enumerates ensemble candidates over the requested model
subsets / weight generators / strategies / outlier thresholds, scores each on
the select split, and writes ``best_setup.json``, ``best_weights.json``,
``candidate_results.{csv,json}``, and ``promotion_report.md``.

The search is cache-only: it never invokes landmark adapters. Re-prediction
belongs in the cache-building stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import typing as T
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.landmarks.ensemble.promoted_setup import (
    PROMOTION_REPORT_FILENAME,
    SETUP_FILENAME,
    WEIGHTS_FILENAME,
    write_best_setup,
    write_best_weights,
)
from lib.landmarks.eval.candidate_search import (
    DEFAULT_OBJECTIVE,
    DEFAULT_REGRESSION_EPSILON_NME,
    CandidateResult,
    enumerate_candidates,
    evaluate_candidate,
    load_split_samples,
    run_candidate_search,
    select_best_candidate,
)
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.eval.splits import load_split_file, split_assignment_hash


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _write_candidate_results(
    output_dir: Path,
    results: T.Sequence[CandidateResult],
    *,
    objective: str,
    regression_epsilon_nme: float,
) -> tuple[Path, Path]:
    """Persist the full candidate evaluation log as CSV + JSON."""
    csv_path = output_dir / "candidate_results.csv"
    json_path = output_dir / "candidate_results.json"
    fieldnames = [
        "rank",
        "candidate_id",
        "score",
        "objective",
        "models",
        "weight_generator",
        "strategy",
        "outlier_threshold",
        "overall_nme",
        "failure_rate",
        "auc",
        "regression_rate_vs_best_single",
        "bucket_regression_rate_vs_best_single",
        "best_single_model",
        "weights_hash",
    ]
    rows: list[dict[str, T.Any]] = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "candidate_id": result.candidate_id,
                "score": result.score,
                "objective": result.objective,
                "models": "|".join(result.candidate.models),
                "weight_generator": result.candidate.weight_generator,
                "strategy": result.candidate.strategy,
                "outlier_threshold": (
                    ""
                    if result.candidate.outlier_threshold is None
                    else result.candidate.outlier_threshold
                ),
                "overall_nme": result.metrics.overall_nme,
                "failure_rate": result.metrics.failure_rate,
                "auc": result.metrics.auc,
                "regression_rate_vs_best_single": result.metrics.regression_rate_vs_best_single,
                "bucket_regression_rate_vs_best_single": result.metrics.bucket_regression_rate_vs_best_single,
                "best_single_model": result.metrics.best_single_model,
                "weights_hash": result.weights_hash,
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_payload = {
        "objective": objective,
        "regression_epsilon_nme": regression_epsilon_nme,
        "candidates": [result.to_payload() for result in results],
    }
    json_path.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, json_path


def _write_promotion_report(
    output_dir: Path,
    *,
    winner: CandidateResult,
    results: T.Sequence[CandidateResult],
    objective: str,
    regression_epsilon_nme: float,
    report_metrics: T.Mapping[str, T.Any],
) -> Path:
    """Write a short human-readable Markdown summary of the promotion decision."""
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
        f"- report_bucket_regression_rate: "
        f"`{report_metrics.get('bucket_regression_rate_vs_best_single', 0.0):.6f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--splits", required=True, help="Path to splits.json (#67).")
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument(
        "--model-subsets",
        default="all",
        help="Comma-separated subset presets: 'all', 'pairs', and/or 'triples'.",
    )
    parser.add_argument(
        "--weight-generators",
        default="equal,inverse_mean_error,regularized_inverse_error",
    )
    parser.add_argument(
        "--strategies",
        default="static_weighted,static_weighted_downweight,weighted_median",
    )
    parser.add_argument("--outlier-thresholds", default="2.5,3.5,4.5")
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument(
        "--regression-epsilon-nme",
        type=float,
        default=DEFAULT_REGRESSION_EPSILON_NME,
    )
    parser.add_argument("--bbox-source", default="manifest")
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--failure-threshold", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignment = load_split_file(args.splits)
    sah = split_assignment_hash(assignment)
    cache = DiskPredictionCache(args.cache_dir)

    fit_samples = load_split_samples(args.manifest, assignment, "fit")
    select_samples = load_split_samples(args.manifest, assignment, "select")
    report_samples = load_split_samples(args.manifest, assignment, "report")

    candidates = enumerate_candidates(
        models=_parse_csv(args.models),
        model_subset_presets=_parse_csv(args.model_subsets),
        weight_generators=_parse_csv(args.weight_generators),
        strategies=_parse_csv(args.strategies),
        outlier_thresholds=_parse_csv_floats(args.outlier_thresholds),
        bbox_source=args.bbox_source,
        crop_scale=args.crop_scale,
    )
    if not candidates:
        raise SystemExit("no candidates were enumerated from the requested dimensions")

    results = run_candidate_search(
        candidates,
        fit_samples=fit_samples,
        select_samples=select_samples,
        cache=cache,
        split_assignment_hash=sah,
        objective=args.objective,
        regression_epsilon_nme=args.regression_epsilon_nme,
        failure_threshold=args.failure_threshold,
    )

    csv_path, json_path = _write_candidate_results(
        output_dir,
        results,
        objective=args.objective,
        regression_epsilon_nme=args.regression_epsilon_nme,
    )

    winner = select_best_candidate(results)
    _fit_result, report_metrics = evaluate_candidate(
        winner.candidate,
        fit_samples=fit_samples,
        select_samples=report_samples,
        cache=cache,
        failure_threshold=args.failure_threshold,
        regression_epsilon_nme=args.regression_epsilon_nme,
    )

    weights_path = output_dir / WEIGHTS_FILENAME
    setup_path = output_dir / SETUP_FILENAME
    write_best_weights(weights_path, winner.weights, models=winner.candidate.models)
    write_best_setup(
        setup_path,
        candidate_id=winner.candidate_id,
        models=winner.candidate.models,
        strategy=winner.candidate.strategy,
        outlier_threshold=winner.candidate.outlier_threshold,
        weight_generator_name=winner.candidate.weight_generator,
        weight_generator_params=winner.candidate.generator_params_dict(),
        crop_scale=winner.candidate.crop_scale,
        bbox_source=winner.candidate.bbox_source,
        regression_epsilon_nme=args.regression_epsilon_nme,
        reproducibility={
            "split_assignment_hash": sah,
            "candidate_search_seed": int(args.seed),
            "objective": args.objective,
        },
        fit={
            "sample_count": len(fit_samples),
            "datasets": sorted({sample.dataset for sample in fit_samples if sample.dataset}),
            "scenario_buckets": sorted(
                {
                    f"{sample.dataset or 'unspecified'}:{sample.condition or 'unspecified'}"
                    for sample in fit_samples
                }
            ),
        },
        selection_metrics=winner.metrics.to_payload(),
        report_metrics=report_metrics.to_payload(),
        evaluation_log_path=str(json_path.name),
        weights_path=WEIGHTS_FILENAME,
    )

    _write_promotion_report(
        output_dir,
        winner=winner,
        results=results,
        objective=args.objective,
        regression_epsilon_nme=args.regression_epsilon_nme,
        report_metrics=report_metrics.to_payload(),
    )

    print(
        f"Promoted candidate {winner.candidate_id} "
        f"(strategy={winner.candidate.strategy}, score={winner.score:.6f})"
    )
    print(f"  setup:   {setup_path}")
    print(f"  weights: {weights_path}")
    print(f"  csv:     {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
