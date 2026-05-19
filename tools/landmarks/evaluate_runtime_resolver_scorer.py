#!/usr/bin/env python3
"""Evaluate learned runtime resolver scorer policy against resolver baselines."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import typing as T
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.ensemble.runtime_resolver_scorer import load_runtime_resolver_scorer
from lib.landmarks.ensemble.strategies import canonical_strategy
from lib.landmarks.ensemble.weights import load_weights
from tools.landmarks.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    SampleCandidateContext,
    load_contexts,
    parse_candidates,
    rows_for_context,
)

logger = logging.getLogger("evaluate_runtime_resolver_scorer")

SCORER_METRICS_JSON = "scorer_metrics.json"
SCORER_POLICY_REPORT_JSON = "scorer_policy_report.json"
SCORER_POLICY_REPORT_CSV = "scorer_policy_report.csv"
SCORER_WORST_SAMPLES_JSON = "scorer_worst_samples.json"
SCORER_FEATURE_IMPORTANCE_CSV = "scorer_feature_importance.csv"


def _collect_contexts(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    candidates: T.Sequence[str],
    failure_threshold: float,
    outlier_threshold: float,
) -> list[SampleCandidateContext]:
    contexts: list[SampleCandidateContext] = []
    for label, manifest_path, cache_dir in (
        ("gt", gt_manifest, gt_cache_dir),
        ("production", production_manifest, production_cache_dir),
    ):
        if manifest_path is None and cache_dir is None:
            continue
        if manifest_path is None or cache_dir is None:
            raise ValueError(f"{label} manifest/cache inputs must be supplied together")
        logger.info("Loading %s scorer evaluation contexts from %s", label, manifest_path)
        contexts.extend(
            load_contexts(
                manifest_path=manifest_path,
                cache_dir=cache_dir,
                weights_path=weights_path,
                candidates=candidates,
                failure_threshold=failure_threshold,
                outlier_threshold=outlier_threshold,
            )
        )
    if not contexts:
        raise ValueError("no scorer evaluation contexts were loaded")
    return contexts


def _summary(values: T.Sequence[float], failures: T.Sequence[bool]) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return {"mean_nme": 0.0, "p90_nme": 0.0, "failure_rate": 0.0}
    return {
        "mean_nme": float(arr.mean()),
        "p90_nme": float(np.percentile(arr, 90)),
        "failure_rate": float(sum(failures) / len(failures)) if failures else 0.0,
    }


def _candidate_summary(
    contexts: T.Sequence[SampleCandidateContext],
    candidate: str,
) -> dict[str, float]:
    return _summary(
        [context.nme_by_candidate[candidate] for context in contexts],
        [context.failure_by_candidate[candidate] for context in contexts],
    )


def _is_fusion_candidate(name: str) -> bool:
    try:
        canonical_strategy(name)
    except (KeyError, ValueError):
        return False
    return True


def _best_single(
    contexts: T.Sequence[SampleCandidateContext],
    candidates: T.Sequence[str],
) -> tuple[str, dict[str, float]]:
    single_names = [
        name
        for name in candidates
        if name in contexts[0].nme_by_candidate
        and not _is_fusion_candidate(name)
    ]
    if not single_names:
        raise ValueError("best-single baseline requires at least one non-fusion model candidate")
    summaries = {name: _candidate_summary(contexts, name) for name in single_names}
    best = min(
        summaries,
        key=lambda name: (
            summaries[name]["mean_nme"],
            summaries[name]["p90_nme"],
            summaries[name]["failure_rate"],
            name,
        ),
    )
    return best, summaries[best]


def _choose_scorer(
    context: SampleCandidateContext,
    scores: T.Mapping[str, float],
) -> tuple[str, bool, str]:
    available = set(context.nme_by_candidate)
    survivors = {
        name
        for name, metric in context.metrics.items()
        if name in available and not metric.geometry_veto_reasons
    }
    fallback_used = not survivors
    fallback_reason = "all_candidates_vetoed" if fallback_used else ""
    selectable = survivors if survivors else available
    chosen = min(selectable, key=lambda name: (scores.get(name, float("inf")), name))
    return chosen, fallback_used, fallback_reason


def _policy_summary(
    contexts: T.Sequence[SampleCandidateContext],
    choices: T.Mapping[str, str],
) -> dict[str, T.Any]:
    values: list[float] = []
    failures: list[bool] = []
    oracle_matches = 0
    gaps: list[float] = []
    for context in contexts:
        chosen = choices[context.sample_id]
        values.append(context.nme_by_candidate[chosen])
        failures.append(context.failure_by_candidate[chosen])
        oracle_matches += int(chosen == context.oracle)
        gaps.append(context.nme_by_candidate[chosen] - context.nme_by_candidate[context.oracle])
    summary = _summary(values, failures)
    summary.update(
        {
            "pick_counts": dict(Counter(choices.values())),
            "oracle_match_rate": oracle_matches / len(contexts) if contexts else 0.0,
            "mean_gap_vs_oracle": float(np.mean(gaps)) if gaps else 0.0,
        }
    )
    return summary


def _per_bucket(
    contexts: T.Sequence[SampleCandidateContext],
    choices: T.Mapping[str, str],
) -> dict[str, dict[str, T.Any]]:
    grouped: dict[str, list[SampleCandidateContext]] = defaultdict(list)
    for context in contexts:
        grouped[context.runtime_bucket or context.condition or "unknown"].append(context)
    payload: dict[str, dict[str, T.Any]] = {}
    for bucket, rows in sorted(grouped.items()):
        row_choices = {context.sample_id: choices[context.sample_id] for context in rows}
        summary = _policy_summary(rows, row_choices)
        payload[bucket] = {
            "sample_count": len(rows),
            "mean_nme": summary["mean_nme"],
            "p90_nme": summary["p90_nme"],
            "failure_rate": summary["failure_rate"],
            "pick_counts": summary["pick_counts"],
        }
    return payload


def evaluate_runtime_resolver_scorer(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    scorer_path: Path,
    candidates: T.Sequence[str],
    output_dir: Path,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    epsilon_mean_nme: float = 0.001,
    epsilon_failure_rate: float = 0.0,
    worst_sample_count: int = 25,
) -> dict[str, T.Any]:
    """Evaluate learned scorer policy and write reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_runtime_resolver_scorer(scorer_path)
    contexts = _collect_contexts(
        gt_manifest=gt_manifest,
        gt_cache_dir=gt_cache_dir,
        production_manifest=production_manifest,
        production_cache_dir=production_cache_dir,
        weights_path=weights_path,
        candidates=candidates,
        failure_threshold=failure_threshold,
        outlier_threshold=outlier_threshold,
    )
    missing_current = [
        context.sample_id for context in contexts if context.selected_candidate_missing_from_eval
    ]
    if missing_current:
        raise ValueError(
            "current runtime policy selected candidates missing from evaluation set for "
            f"{len(missing_current)} sample(s): {missing_current[:10]}"
        )

    rows: list[dict[str, T.Any]] = []
    scorer_choices: dict[str, str] = {}
    current_choices: dict[str, str] = {}
    oracle_choices: dict[str, str] = {}
    fallback_count = 0
    for context in contexts:
        score_by_candidate = {
            row.candidate_name: scorer.score_feature_map(row.feature_values)
            for row in rows_for_context(context)
        }
        chosen, fallback_used, fallback_reason = _choose_scorer(context, score_by_candidate)
        fallback_count += int(fallback_used)
        scorer_choices[context.sample_id] = chosen
        current_choices[context.sample_id] = (
            context.current_policy_choice
        )
        oracle_choices[context.sample_id] = context.oracle
        rows.append(
            {
                "sample_id": context.sample_id,
                "dataset": context.dataset,
                "condition": context.condition,
                "runtime_bucket": context.runtime_bucket,
                "runtime_bucket_source": context.runtime_bucket_source,
                "chosen": chosen,
                "chosen_nme": context.nme_by_candidate[chosen],
                "chosen_failure": int(context.failure_by_candidate[chosen]),
                "current_bucket_policy": current_choices[context.sample_id],
                "current_bucket_policy_nme": context.nme_by_candidate[
                    current_choices[context.sample_id]
                ],
                "oracle": context.oracle,
                "oracle_nme": context.nme_by_candidate[context.oracle],
                "gap_vs_oracle": (
                    context.nme_by_candidate[chosen] - context.nme_by_candidate[context.oracle]
                ),
                "candidate_scores": json.dumps(score_by_candidate, sort_keys=True),
                "fallback_used": int(fallback_used),
                "fallback_reason": fallback_reason,
            }
        )

    best_single_name, best_single = _best_single(contexts, candidates)
    static_name = (
        "static_weighted_downweight" if "static_weighted_downweight" in candidates else ""
    )
    static = _candidate_summary(contexts, static_name) if static_name else best_single
    scorer_summary = _policy_summary(contexts, scorer_choices)
    current_summary = _policy_summary(contexts, current_choices)
    oracle_summary = _policy_summary(contexts, oracle_choices)
    failed_gates: list[str] = []
    if scorer_summary["mean_nme"] > best_single["mean_nme"] + epsilon_mean_nme:
        failed_gates.append("scorer_mean_nme_regresses_vs_best_single")
    if scorer_summary["failure_rate"] > best_single["failure_rate"] + epsilon_failure_rate:
        failed_gates.append("scorer_failure_rate_regresses_vs_best_single")
    if static_name and scorer_summary["mean_nme"] > static["mean_nme"] + epsilon_mean_nme:
        failed_gates.append("scorer_mean_nme_regresses_vs_static_downweight")
    if (
        static_name
        and scorer_summary["failure_rate"] > static["failure_rate"] + epsilon_failure_rate
    ):
        failed_gates.append("scorer_failure_rate_regresses_vs_static_downweight")

    report: dict[str, T.Any] = {
        "status": "pass" if not failed_gates else "fail",
        "failed_gates": failed_gates,
        "sample_count": len(contexts),
        "candidate_count": len(candidates),
        "candidates": list(candidates),
        "scorer_path": str(scorer_path),
        "scorer_version": scorer.version,
        "best_single": {"candidate": best_single_name, **best_single},
        "static_weighted_downweight": {"candidate": static_name, **static},
        "learned_quality_v1": scorer_summary,
        "current_bucket_aware_veto": current_summary,
        "oracle": oracle_summary,
        "fallback_count": fallback_count,
        "per_bucket": _per_bucket(contexts, scorer_choices),
    }

    _write_outputs(
        report,
        rows,
        scorer,
        output_dir,
        worst_sample_count=worst_sample_count,
    )
    return report


def _write_outputs(
    report: dict[str, T.Any],
    rows: T.Sequence[dict[str, T.Any]],
    scorer: T.Any,
    output_dir: Path,
    *,
    worst_sample_count: int,
) -> None:
    (output_dir / SCORER_METRICS_JSON).write_text(
        json.dumps(report["learned_quality_v1"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / SCORER_POLICY_REPORT_JSON).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    (output_dir / SCORER_WORST_SAMPLES_JSON).write_text(
        json.dumps({"samples": worst}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-manifest", type=Path)
    parser.add_argument("--gt-cache-dir", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-cache-dir", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        default="",
        help="Comma-separated candidate list. Defaults to models from weights plus static fusions.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument("--epsilon-mean-nme", type=float, default=0.001)
    parser.add_argument("--epsilon-failure-rate", type=float, default=0.0)
    parser.add_argument("--worst-sample-count", type=int, default=25)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    report = evaluate_runtime_resolver_scorer(
        gt_manifest=args.gt_manifest,
        gt_cache_dir=args.gt_cache_dir,
        production_manifest=args.production_manifest,
        production_cache_dir=args.production_cache_dir,
        weights_path=args.weights,
        scorer_path=args.scorer,
        candidates=candidates,
        output_dir=args.output_dir,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
        epsilon_mean_nme=args.epsilon_mean_nme,
        epsilon_failure_rate=args.epsilon_failure_rate,
        worst_sample_count=args.worst_sample_count,
    )
    logger.info("Scorer policy status: %s", report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
