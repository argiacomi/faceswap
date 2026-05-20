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

from lib.landmarks.ensemble.runtime_resolver import (
    _hard_slice_safe_single_candidate,
    _high_risk_safe_fallback_candidate,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import load_runtime_resolver_scorer
from lib.landmarks.ensemble.strategies import canonical_strategy
from lib.landmarks.ensemble.weights import load_weights
from tools.landmarks.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SCORER_CANDIDATE_CSV,
    SampleCandidateContext,
    load_contexts,
    parse_candidates,
    rows_for_context,
)

logger = logging.getLogger("evaluate_runtime_resolver_scorer")

SCORER_METRICS_JSON = "scorer_metrics.json"
SCORER_POLICY_REPORT_JSON = "scorer_policy_report.json"
SCORER_HELDOUT_POLICY_REPORT_JSON = "scorer_policy_eval_report.json"
SCORER_POLICY_REPORT_CSV = "scorer_policy_report.csv"
SCORER_WORST_SAMPLES_JSON = "scorer_worst_samples.json"
SCORER_FEATURE_IMPORTANCE_CSV = "scorer_feature_importance.csv"
DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK = 0.50
DEFAULT_SAFE_FALLBACK_MIN_DELTA = 0.05
DEFAULT_FALLBACK_CATASTROPHIC_WORSE_NME = 0.02
PROMOTION_SCOPES = ("universal", "production")
SOURCE_GT_HARD = "gt_hard"
SOURCE_PRODUCTION_VALIDATED = "production_validated"
HARD_SLICE_POLICY_BUCKETS = {
    "extreme_roll",
    "rolled_large_yaw_left",
    "rolled_large_yaw_right",
    "rolled_profile_left",
    "rolled_profile_right",
}


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
    allow_image_backfill: bool,
) -> list[SampleCandidateContext]:
    contexts: list[SampleCandidateContext] = []
    for label, manifest_path, cache_dir in (
        (SOURCE_GT_HARD, gt_manifest, gt_cache_dir),
        (SOURCE_PRODUCTION_VALIDATED, production_manifest, production_cache_dir),
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
                allow_image_backfill=allow_image_backfill,
            )
        )
    if not contexts:
        raise ValueError("no scorer evaluation contexts were loaded")
    return contexts


def _eval_split_sources(path: Path) -> tuple[set[tuple[str, str, str]], dict[str, str]]:
    """Return held-out ``(source, dataset, sample_id)`` keys and per-sample source."""
    keys: set[tuple[str, str, str]] = set()
    source_by_sample_id: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or ()):
            raise ValueError(f"eval split {path} must contain a sample_id column")
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            source = str(row.get("source", "")).strip()
            dataset = str(row.get("dataset", "")).strip()
            keys.add((source, dataset, sample_id))
            if source:
                existing = source_by_sample_id.get(sample_id)
                if existing is not None and existing != source:
                    raise ValueError(
                        f"eval split {path} maps sample {sample_id!r} to multiple sources: "
                        f"{existing!r}, {source!r}"
                    )
                source_by_sample_id[sample_id] = source
    if not keys:
        raise ValueError(f"eval split {path} did not contain any sample ids")
    return keys, source_by_sample_id


def _filter_contexts_by_eval_split(
    contexts: T.Sequence[SampleCandidateContext],
    split_path: Path,
) -> tuple[list[SampleCandidateContext], dict[str, str]]:
    """Restrict contexts to held-out sample ids from a scorer eval-row CSV."""
    keys, source_by_sample_id = _eval_split_sources(split_path)
    dataset_sample_keys = {(dataset, sample_id) for _source, dataset, sample_id in keys}
    split_sample_ids = {sample_id for _source, _dataset, sample_id in keys}
    filtered = [
        context
        for context in contexts
        if (context.dataset, context.sample_id) in dataset_sample_keys
        or context.sample_id in split_sample_ids
    ]
    if not filtered:
        raise ValueError(f"eval split {split_path} did not match any evaluation contexts")
    unique_context_ids = {context.sample_id for context in filtered}
    if len(unique_context_ids) != len(split_sample_ids):
        missing = sorted(split_sample_ids - unique_context_ids)
        raise ValueError(
            f"eval split {split_path} matched {len(unique_context_ids)} unique contexts for "
            f"{len(split_sample_ids)} split samples; missing examples: {missing[:10]}"
        )
    return filtered, source_by_sample_id


def _context_source(
    context: SampleCandidateContext,
    source_by_sample_id: T.Mapping[str, str],
) -> str:
    """Return the scorer source for a context, preferring explicit eval-split source."""
    source = source_by_sample_id.get(context.sample_id, "")
    if source:
        return source
    if (
        context.dataset == SOURCE_PRODUCTION_VALIDATED
        or context.runtime_bucket_source == "stored_manifest_landmark_ensemble"
    ):
        return SOURCE_PRODUCTION_VALIDATED
    return SOURCE_GT_HARD


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
        if name in contexts[0].nme_by_candidate and not _is_fusion_candidate(name)
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


def _score_delta_passes(
    *,
    replacement: str,
    selected: str,
    scores: T.Mapping[str, float],
    min_delta: float,
) -> bool:
    """Return whether replacement is materially lower risk than selected."""
    replacement_score = scores.get(replacement)
    selected_score = scores.get(selected)
    if replacement_score is None or selected_score is None:
        return False
    if not math_is_finite(replacement_score) or not math_is_finite(selected_score):
        return False
    return float(replacement_score) < float(selected_score) - min_delta


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _choose_scorer(
    context: SampleCandidateContext,
    scores: T.Mapping[str, float],
    *,
    risk_floor_for_safe_fallback: float,
    safe_fallback_min_delta: float,
) -> tuple[str, bool, str, str, str]:
    available = set(context.nme_by_candidate)
    survivors = {
        name
        for name, metric in context.metrics.items()
        if name in available and not metric.geometry_veto_reasons
    }
    fallback_used = not survivors
    fallback_reason = "all_candidates_vetoed" if fallback_used else ""
    rejected_candidate = ""
    replacement_candidate = ""
    selectable = survivors if survivors else available
    chosen = min(selectable, key=lambda name: (scores.get(name, float("inf")), name))
    candidates_by_name = {candidate.name: candidate for candidate in context.candidates}
    hard_slice_fallback = _hard_slice_safe_single_candidate(
        selected=chosen,
        candidates=candidates_by_name,
        metrics=context.metrics,
        candidate_extra_features=context.candidate_extra_features,
        condition=context.condition,
        runtime_bucket=context.runtime_bucket,
        runtime_bucket_source=context.runtime_bucket_source,
        scores=scores,
        selectable=selectable,
    )
    if hard_slice_fallback is not None and hard_slice_fallback != chosen:
        rejected_candidate = chosen
        chosen = hard_slice_fallback
        replacement_candidate = chosen
        fallback_used = True
        fallback_reason = "consensus_collapse_fusion_rejected"
    safe_fallback = _high_risk_safe_fallback_candidate(
        scores=scores,
        selectable=selectable,
        candidates=candidates_by_name,
        metrics=context.metrics,
        risk_floor=risk_floor_for_safe_fallback,
    )
    if (
        safe_fallback is not None
        and safe_fallback != chosen
        and _score_delta_passes(
            replacement=safe_fallback,
            selected=chosen,
            scores=scores,
            min_delta=safe_fallback_min_delta,
        )
    ):
        rejected_candidate = chosen
        chosen = safe_fallback
        replacement_candidate = chosen
        fallback_used = True
        fallback_reason = "scorer_high_risk_safe_fallback"
    return chosen, fallback_used, fallback_reason, rejected_candidate, replacement_candidate


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


def _choice_subset(
    contexts: T.Sequence[SampleCandidateContext],
    choices: T.Mapping[str, str],
) -> dict[str, str]:
    return {context.sample_id: choices[context.sample_id] for context in contexts}


def _policy_metric_bundle(
    contexts: T.Sequence[SampleCandidateContext],
    *,
    candidates: T.Sequence[str],
    scorer_choices: T.Mapping[str, str],
    current_choices: T.Mapping[str, str],
    oracle_choices: T.Mapping[str, str],
) -> dict[str, T.Any]:
    """Return actual selected-policy NME/failure summaries for one source slice."""
    if not contexts:
        return {
            "sample_count": 0,
            "learned_quality_v1": _policy_summary((), {}),
            "current_bucket_aware_veto": _policy_summary((), {}),
            "oracle": _policy_summary((), {}),
        }
    payload: dict[str, T.Any] = {
        "sample_count": len(contexts),
        "learned_quality_v1": _policy_summary(contexts, _choice_subset(contexts, scorer_choices)),
        "current_bucket_aware_veto": _policy_summary(
            contexts, _choice_subset(contexts, current_choices)
        ),
        "oracle": _policy_summary(contexts, _choice_subset(contexts, oracle_choices)),
    }
    best_single_name, best_single = _best_single(contexts, candidates)
    payload["best_single"] = {"candidate": best_single_name, **best_single}
    if "static_weighted_downweight" in candidates:
        payload["static_weighted_downweight"] = {
            "candidate": "static_weighted_downweight",
            **_candidate_summary(contexts, "static_weighted_downweight"),
        }
    if "hrnet" in candidates:
        payload["hrnet"] = {"candidate": "hrnet", **_candidate_summary(contexts, "hrnet")}
    return payload


def _fallback_impact_summary(impacts: T.Sequence[dict[str, T.Any]]) -> dict[str, T.Any]:
    if not impacts:
        return {
            "count_with_rejected_candidate": 0,
            "mean_nme_delta_vs_rejected": 0.0,
            "mean_failure_delta_vs_rejected": 0.0,
            "worse_count": 0,
            "catastrophic_worse_count": 0,
        }
    nme_deltas = [float(item["nme_delta_vs_rejected"]) for item in impacts]
    failure_deltas = [float(item["failure_delta_vs_rejected"]) for item in impacts]
    return {
        "count_with_rejected_candidate": len(impacts),
        "mean_nme_delta_vs_rejected": float(np.mean(nme_deltas)),
        "mean_failure_delta_vs_rejected": float(np.mean(failure_deltas)),
        "worse_count": sum(delta > 0.0 for delta in nme_deltas),
        "catastrophic_worse_count": sum(
            delta >= DEFAULT_FALLBACK_CATASTROPHIC_WORSE_NME or failure_delta > 0.0
            for delta, failure_delta in zip(nme_deltas, failure_deltas, strict=True)
        ),
    }


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
    eval_split: Path | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    epsilon_mean_nme: float = 0.001,
    epsilon_failure_rate: float = 0.0,
    worst_sample_count: int = 25,
    risk_floor_for_safe_fallback: float = DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK,
    safe_fallback_min_delta: float = DEFAULT_SAFE_FALLBACK_MIN_DELTA,
    promotion_scope: str = "universal",
    allow_image_backfill: bool = False,
    allow_derived_no_image_gt_hard: bool = False,
) -> dict[str, T.Any]:
    """Evaluate learned scorer policy and write reports."""
    if promotion_scope not in PROMOTION_SCOPES:
        raise ValueError(f"promotion_scope must be one of {PROMOTION_SCOPES}, got {promotion_scope!r}")
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
        allow_image_backfill=allow_image_backfill,
    )
    source_by_sample_id: dict[str, str] = {}
    if eval_split is not None:
        contexts, source_by_sample_id = _filter_contexts_by_eval_split(contexts, eval_split)
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
    fallback_impacts: list[dict[str, T.Any]] = []
    fallback_count = 0
    safe_fallback_count = 0
    hard_slice_fallback_count = 0
    for context in contexts:
        score_by_candidate = {
            row.candidate_name: scorer.score_feature_map(row.feature_values)
            for row in rows_for_context(context)
        }
        (
            chosen,
            fallback_used,
            fallback_reason,
            rejected_candidate,
            replacement_candidate,
        ) = _choose_scorer(
            context,
            score_by_candidate,
            risk_floor_for_safe_fallback=risk_floor_for_safe_fallback,
            safe_fallback_min_delta=safe_fallback_min_delta,
        )
        fallback_count += int(fallback_used)
        safe_fallback_count += int(fallback_reason == "scorer_high_risk_safe_fallback")
        hard_slice_fallback_count += int(fallback_reason == "consensus_collapse_fusion_rejected")
        rejected_nme = None
        replacement_nme = None
        rejected_failure = None
        replacement_failure = None
        if rejected_candidate and rejected_candidate in context.nme_by_candidate:
            rejected_nme = context.nme_by_candidate[rejected_candidate]
            replacement_nme = context.nme_by_candidate[chosen]
            rejected_failure = context.failure_by_candidate[rejected_candidate]
            replacement_failure = context.failure_by_candidate[chosen]
            fallback_impacts.append(
                {
                    "sample_id": context.sample_id,
                    "fallback_reason": fallback_reason,
                    "rejected_candidate": rejected_candidate,
                    "replacement_candidate": chosen,
                    "nme_delta_vs_rejected": replacement_nme - rejected_nme,
                    "failure_delta_vs_rejected": int(replacement_failure) - int(rejected_failure),
                }
            )
        scorer_choices[context.sample_id] = chosen
        current_choices[context.sample_id] = context.current_policy_choice
        oracle_choices[context.sample_id] = context.oracle
        rows.append(
            {
                "source": _context_source(context, source_by_sample_id),
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
                "rejected_candidate": rejected_candidate,
                "replacement_candidate": replacement_candidate,
                "rejected_candidate_nme": rejected_nme,
                "replacement_candidate_nme": replacement_nme,
                "fallback_nme_delta_vs_rejected": (
                    None if rejected_nme is None else replacement_nme - rejected_nme
                ),
                "fallback_failure_delta_vs_rejected": (
                    None if rejected_failure is None else int(replacement_failure) - int(rejected_failure)
                ),
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
    production_contexts = [
        context
        for context in contexts
        if _context_source(context, source_by_sample_id) == SOURCE_PRODUCTION_VALIDATED
    ]
    gt_hard_all_contexts = [
        context
        for context in contexts
        if _context_source(context, source_by_sample_id) == SOURCE_GT_HARD
    ]
    gt_roll_hard_contexts = [
        context
        for context in gt_hard_all_contexts
        if context.condition in HARD_SLICE_POLICY_BUCKETS
        or context.runtime_bucket in HARD_SLICE_POLICY_BUCKETS
    ]
    derived_no_image_contexts = [
        context
        for context in contexts
        if context.runtime_bucket_source == "derived_no_image_evidence"
    ]
    derived_no_image_gt_hard_contexts = [
        context
        for context in gt_hard_all_contexts
        if context.runtime_bucket_source == "derived_no_image_evidence"
    ]
    production_only_policy_metrics = _policy_metric_bundle(
        production_contexts,
        candidates=candidates,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
    )
    gt_hard_all_policy_metrics = _policy_metric_bundle(
        gt_hard_all_contexts,
        candidates=candidates,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
    )
    gt_roll_hard_policy_metrics = _policy_metric_bundle(
        gt_roll_hard_contexts,
        candidates=candidates,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
    )
    combined_failed_gates: list[str] = []
    production_failed_gates: list[str] = []
    gt_hard_failed_gates: list[str] = []
    if static_name and scorer_summary["mean_nme"] >= static["mean_nme"]:
        combined_failed_gates.append("scorer_mean_nme_not_better_than_static_downweight")
    if static_name and scorer_summary["p90_nme"] > static["p90_nme"]:
        combined_failed_gates.append("scorer_p90_nme_regresses_vs_static_downweight")
    if (
        static_name
        and scorer_summary["failure_rate"] > static["failure_rate"] + epsilon_failure_rate
    ):
        combined_failed_gates.append("scorer_failure_rate_regresses_vs_static_downweight")
    if static_name and production_contexts:
        production_scorer = production_only_policy_metrics["learned_quality_v1"]
        production_static = production_only_policy_metrics["static_weighted_downweight"]
        production_hrnet = production_only_policy_metrics.get("hrnet")
        if production_scorer["mean_nme"] >= production_static["mean_nme"]:
            production_failed_gates.append("production_scorer_mean_nme_not_better_than_static_downweight")
        if production_scorer["p90_nme"] > production_static["p90_nme"]:
            production_failed_gates.append("production_scorer_p90_nme_regresses_vs_static_downweight")
        if (
            production_scorer["failure_rate"]
            > production_static["failure_rate"] + epsilon_failure_rate
        ):
            production_failed_gates.append("production_scorer_failure_rate_regresses_vs_static_downweight")
        if (
            production_hrnet is not None
            and production_scorer["failure_rate"]
            > production_hrnet["failure_rate"] + epsilon_failure_rate
        ):
            production_failed_gates.append("production_scorer_failure_rate_regresses_vs_hrnet")
    if static_name and gt_hard_all_contexts:
        gt_hard_scorer = gt_hard_all_policy_metrics["learned_quality_v1"]
        gt_hard_static = gt_hard_all_policy_metrics["static_weighted_downweight"]
        if gt_hard_scorer["failure_rate"] > gt_hard_static["failure_rate"] + epsilon_failure_rate:
            gt_hard_failed_gates.append("gt_hard_scorer_failure_rate_regresses_vs_static_downweight")
    if derived_no_image_gt_hard_contexts and not allow_derived_no_image_gt_hard:
        gt_hard_failed_gates.append("gt_hard_derived_no_image_evidence_requires_explicit_allow")
    universal_failed_gates = [
        *combined_failed_gates,
        *production_failed_gates,
        *gt_hard_failed_gates,
    ]
    failed_gates = production_failed_gates if promotion_scope == "production" else universal_failed_gates

    report: dict[str, T.Any] = {
        "status": "pass" if not failed_gates else "fail",
        "promotion_status": "pass" if not failed_gates else "fail",
        "promotion_scope": promotion_scope,
        "failed_gates": failed_gates,
        "universal_failed_gates": universal_failed_gates,
        "combined_failed_gates": combined_failed_gates,
        "production_failed_gates": production_failed_gates,
        "gt_hard_failed_gates": gt_hard_failed_gates,
        "production_gate_status": "pass" if not production_failed_gates else "fail",
        "gt_hard_gate_status": "pass" if not gt_hard_failed_gates else "diagnostic_fail",
        "sample_count": len(contexts),
        "heldout_eval": eval_split is not None,
        "eval_split": "" if eval_split is None else str(eval_split),
        "allow_image_backfill": allow_image_backfill,
        "allow_derived_no_image_gt_hard": allow_derived_no_image_gt_hard,
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
        "safe_fallback_count": safe_fallback_count,
        "hard_slice_fallback_count": hard_slice_fallback_count,
        "consensus_collapse_rejection_count": hard_slice_fallback_count,
        "consensus_collapse_fallback_count": hard_slice_fallback_count,
        "fallback_impact": _fallback_impact_summary(fallback_impacts),
        "derived_no_image_sample_count": len(derived_no_image_contexts),
        "derived_no_image_gt_hard_sample_count": len(derived_no_image_gt_hard_contexts),
        "risk_floor_for_safe_fallback": risk_floor_for_safe_fallback,
        "safe_fallback_min_delta": safe_fallback_min_delta,
        "safe_fallback_tie_breaker": ["hrnet", "spiga", "orformer"],
        "per_bucket": _per_bucket(contexts, scorer_choices),
        "production_only_policy_metrics": production_only_policy_metrics,
        "gt_hard_all_policy_metrics": gt_hard_all_policy_metrics,
        "gt_hard_only_policy_metrics": gt_hard_all_policy_metrics,
        "gt_roll_hard_policy_metrics": gt_roll_hard_policy_metrics,
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
    if report.get("heldout_eval"):
        (output_dir / SCORER_HELDOUT_POLICY_REPORT_JSON).write_text(
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
        "--eval-split",
        type=Path,
        help="Optional scorer eval-row CSV; when supplied, policy metrics use only held-out rows.",
    )
    parser.add_argument(
        "--candidates",
        default="",
        help=f"Comma-separated candidate list. Defaults to {DEFAULT_SCORER_CANDIDATE_CSV}.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument("--epsilon-mean-nme", type=float, default=0.001)
    parser.add_argument("--epsilon-failure-rate", type=float, default=0.0)
    parser.add_argument("--worst-sample-count", type=int, default=25)
    parser.add_argument(
        "--risk-floor-for-safe-fallback",
        type=float,
        default=DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK,
    )
    parser.add_argument(
        "--safe-fallback-min-delta",
        type=float,
        default=DEFAULT_SAFE_FALLBACK_MIN_DELTA,
        help="Require a safe-fallback replacement risk score to beat selected risk by this margin.",
    )
    parser.add_argument(
        "--promotion-scope",
        choices=PROMOTION_SCOPES,
        default="universal",
        help="Gate production only, or require both production and GT-hard diagnostics to pass.",
    )
    parser.add_argument(
        "--allow-image-backfill",
        action="store_true",
        help="Compute image-aware runtime metadata for rows without stored metadata.",
    )
    parser.add_argument(
        "--allow-derived-no-image-gt-hard",
        action="store_true",
        help="Allow GT hard diagnostics to use landmark-only derived runtime buckets.",
    )
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
        eval_split=args.eval_split,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
        epsilon_mean_nme=args.epsilon_mean_nme,
        epsilon_failure_rate=args.epsilon_failure_rate,
        worst_sample_count=args.worst_sample_count,
        risk_floor_for_safe_fallback=args.risk_floor_for_safe_fallback,
        safe_fallback_min_delta=args.safe_fallback_min_delta,
        promotion_scope=args.promotion_scope,
        allow_image_backfill=args.allow_image_backfill,
        allow_derived_no_image_gt_hard=args.allow_derived_no_image_gt_hard,
    )
    logger.info("Scorer policy status: %s", report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
