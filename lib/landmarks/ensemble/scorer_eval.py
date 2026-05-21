#!/usr/bin/env python3
"""Reusable runtime resolver scorer policy evaluation implementation."""

from __future__ import annotations

import csv
import json
import typing as T
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.runtime_resolver import (
    _hard_slice_safe_single_candidate,
    _high_risk_safe_fallback_candidate,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverScorer,
    load_runtime_resolver_scorer,
)
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    SampleCandidateContext,
    rows_for_context,
)
from lib.landmarks.ensemble.scorer_contexts import load_scorer_contexts
from lib.landmarks.ensemble.scorer_reports import write_scorer_policy_outputs
from lib.landmarks.ensemble.scorer_target_config import (
    MODEL_TYPE_LINEAR_REGRESSION,
    TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
)
from lib.landmarks.ensemble.strategies import canonical_strategy
from lib.landmarks.pipeline_conventions import (
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
)

DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK = 0.50
DEFAULT_SAFE_FALLBACK_MIN_DELTA = 0.05
DEFAULT_FALLBACK_CATASTROPHIC_WORSE_NME = 0.02
PROMOTION_SCOPES = ("universal", "production")
SCORER_VERSION_REPORT_LABEL = "scorer_version"
RUNTIME_POLICY_REPORT_LABEL = "runtime_policy_learned_quality"
HARD_SLICE_POLICY_BUCKETS = {
    "extreme_roll",
    "rolled_large_yaw_left",
    "rolled_large_yaw_right",
    "rolled_profile_left",
    "rolled_profile_right",
}


def eval_split_sources(path: Path) -> tuple[set[tuple[str, str, str]], dict[str, str]]:
    """Return held-out `(source, dataset, sample_id)` keys and per-sample source."""
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


def filter_contexts_by_eval_split(
    contexts: T.Sequence[SampleCandidateContext],
    split_path: Path,
) -> tuple[list[SampleCandidateContext], dict[str, str]]:
    """Restrict contexts to held-out sample ids from a scorer eval-row CSV."""
    keys, source_by_sample_id = eval_split_sources(split_path)
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


def context_source(
    context: SampleCandidateContext,
    source_by_sample_id: T.Mapping[str, str],
) -> str:
    """Return scorer source for a context, preferring explicit eval-split source."""
    source = source_by_sample_id.get(context.sample_id, "")
    if source:
        return source
    if context.source:
        return context.source
    if (
        context.dataset == SOURCE_PRODUCTION_VALIDATED
        or context.runtime_bucket_source == "stored_manifest_landmark_ensemble"
    ):
        return SOURCE_PRODUCTION_VALIDATED
    return SOURCE_GT_HARD


def summary(values: T.Sequence[float], failures: T.Sequence[bool]) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return {"mean_nme": 0.0, "p90_nme": 0.0, "failure_rate": 0.0}
    return {
        "mean_nme": float(arr.mean()),
        "p90_nme": float(np.percentile(arr, 90)),
        "failure_rate": float(sum(failures) / len(failures)) if failures else 0.0,
    }


def candidate_summary(
    contexts: T.Sequence[SampleCandidateContext],
    candidate: str,
) -> dict[str, float]:
    return summary(
        [context.nme_by_candidate[candidate] for context in contexts],
        [context.failure_by_candidate[candidate] for context in contexts],
    )


def is_fusion_candidate(name: str) -> bool:
    try:
        canonical_strategy(name)
    except (KeyError, ValueError):
        return False
    return True


def best_single(
    contexts: T.Sequence[SampleCandidateContext],
    candidates: T.Sequence[str],
) -> tuple[str, dict[str, float]]:
    single_names = [
        name
        for name in candidates
        if name in contexts[0].nme_by_candidate and not is_fusion_candidate(name)
    ]
    if not single_names:
        raise ValueError("best-single baseline requires at least one non-fusion model candidate")
    summaries = {name: candidate_summary(contexts, name) for name in single_names}
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


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def score_delta_passes(
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


def choose_scorer(
    context: SampleCandidateContext,
    scores: T.Mapping[str, float],
    *,
    risk_floor_for_safe_fallback: float,
    safe_fallback_min_delta: float,
) -> tuple[str, bool, str, str, str]:
    """Choose the scorer candidate plus fallback metadata."""
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
        and score_delta_passes(
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


def policy_summary(
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
    base = summary(values, failures)
    base.update(
        {
            "pick_counts": dict(Counter(choices.values())),
            "oracle_match_rate": oracle_matches / len(contexts) if contexts else 0.0,
            "mean_gap_vs_oracle": float(np.mean(gaps)) if gaps else 0.0,
        }
    )
    return base


def per_bucket(
    contexts: T.Sequence[SampleCandidateContext],
    choices: T.Mapping[str, str],
) -> dict[str, dict[str, T.Any]]:
    grouped: dict[str, list[SampleCandidateContext]] = defaultdict(list)
    for context in contexts:
        grouped[context.runtime_bucket or context.condition or "unknown"].append(context)
    payload: dict[str, dict[str, T.Any]] = {}
    for bucket, rows in sorted(grouped.items()):
        row_choices = {context.sample_id: choices[context.sample_id] for context in rows}
        bucket_summary = policy_summary(rows, row_choices)
        payload[bucket] = {
            "sample_count": len(rows),
            "mean_nme": bucket_summary["mean_nme"],
            "p90_nme": bucket_summary["p90_nme"],
            "failure_rate": bucket_summary["failure_rate"],
            "pick_counts": bucket_summary["pick_counts"],
        }
    return payload


def choice_subset(
    contexts: T.Sequence[SampleCandidateContext],
    choices: T.Mapping[str, str],
) -> dict[str, str]:
    return {context.sample_id: choices[context.sample_id] for context in contexts}


def policy_metric_bundle(
    contexts: T.Sequence[SampleCandidateContext],
    *,
    candidates: T.Sequence[str],
    scorer_policy_name: str,
    scorer_choices: T.Mapping[str, str],
    current_choices: T.Mapping[str, str],
    oracle_choices: T.Mapping[str, str],
    extra_scorer_choices: T.Mapping[str, T.Mapping[str, str]] | None = None,
) -> dict[str, T.Any]:
    """Return selected-policy NME/failure summaries for one source slice."""
    if not contexts:
        payload = {
            "sample_count": 0,
            scorer_policy_name: policy_summary((), {}),
            RUNTIME_POLICY_REPORT_LABEL: policy_summary((), {}),
            "oracle": policy_summary((), {}),
        }
        for policy_name in extra_scorer_choices or {}:
            if policy_name == scorer_policy_name:
                continue
            payload[policy_name] = policy_summary((), {})
        return payload
    payload: dict[str, T.Any] = {
        "sample_count": len(contexts),
        scorer_policy_name: policy_summary(contexts, choice_subset(contexts, scorer_choices)),
        RUNTIME_POLICY_REPORT_LABEL: policy_summary(
            contexts, choice_subset(contexts, current_choices)
        ),
        "oracle": policy_summary(contexts, choice_subset(contexts, oracle_choices)),
    }
    for policy_name, policy_choices in (extra_scorer_choices or {}).items():
        if policy_name == scorer_policy_name:
            continue
        payload[policy_name] = policy_summary(
            contexts,
            choice_subset(contexts, policy_choices),
        )
    best_single_name, best_single_summary = best_single(contexts, candidates)
    payload["best_single"] = {"candidate": best_single_name, **best_single_summary}
    if "static_weighted_downweight" in candidates:
        payload["static_weighted_downweight"] = {
            "candidate": "static_weighted_downweight",
            **candidate_summary(contexts, "static_weighted_downweight"),
        }
    if "hrnet" in candidates:
        payload["hrnet"] = {"candidate": "hrnet", **candidate_summary(contexts, "hrnet")}
    return payload


def fallback_impact_summary(impacts: T.Sequence[dict[str, T.Any]]) -> dict[str, T.Any]:
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


def scorer_policy_key(scorer: RuntimeResolverScorer) -> str:
    """Return the stable report key for a scorer artifact."""
    if (
        scorer.model_type == MODEL_TYPE_LINEAR_REGRESSION
        or scorer.target != TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP
    ):
        return SCORER_VERSION_REPORT_LABEL
    return "current_binary_logistic_scorer"


def assert_lower_score_is_better(scorer: RuntimeResolverScorer) -> None:
    """Fail fast if a scorer artifact cannot be ranked by ascending score."""
    if scorer.higher_is_better:
        raise ValueError(
            f"runtime resolver scorer {scorer.source_path or '<memory>'} is not lower-is-better"
        )


def evaluate_runtime_resolver_scorer(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    scorer_path: Path,
    binary_scorer_path: Path | None = None,
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
    gt_hard_resolver_metadata: Path | None = None,
) -> dict[str, T.Any]:
    """Evaluate learned scorer policy and write reports."""
    if promotion_scope not in PROMOTION_SCOPES:
        raise ValueError(
            f"promotion_scope must be one of {PROMOTION_SCOPES}, got {promotion_scope!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_runtime_resolver_scorer(scorer_path)
    assert_lower_score_is_better(scorer)
    binary_scorer = (
        None if binary_scorer_path is None else load_runtime_resolver_scorer(binary_scorer_path)
    )
    if binary_scorer is not None:
        assert_lower_score_is_better(binary_scorer)
    contexts = load_scorer_contexts(
        gt_manifest=gt_manifest,
        gt_cache_dir=gt_cache_dir,
        production_manifest=production_manifest,
        production_cache_dir=production_cache_dir,
        weights_path=weights_path,
        candidates=candidates,
        failure_threshold=failure_threshold,
        outlier_threshold=outlier_threshold,
        allow_image_backfill=allow_image_backfill,
        gt_hard_resolver_metadata=gt_hard_resolver_metadata,
        require_gt_hard_metadata=not allow_derived_no_image_gt_hard,
    )
    source_by_sample_id: dict[str, str] = {}
    if eval_split is not None:
        contexts, source_by_sample_id = filter_contexts_by_eval_split(contexts, eval_split)
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
    binary_scorer_choices: dict[str, str] = {}
    current_choices: dict[str, str] = {}
    oracle_choices: dict[str, str] = {}
    fallback_impacts: list[dict[str, T.Any]] = []
    fallback_count = 0
    safe_fallback_count = 0
    hard_slice_fallback_count = 0
    for context in contexts:
        context_rows = rows_for_context(context)
        score_by_candidate = {
            row.candidate_name: scorer.score_feature_map(row.feature_values)
            for row in context_rows
        }
        binary_score_by_candidate: dict[str, float] = {}
        binary_chosen = ""
        if binary_scorer is not None:
            binary_score_by_candidate = {
                row.candidate_name: binary_scorer.score_feature_map(row.feature_values)
                for row in context_rows
            }
            binary_chosen = choose_scorer(
                context,
                binary_score_by_candidate,
                risk_floor_for_safe_fallback=risk_floor_for_safe_fallback,
                safe_fallback_min_delta=safe_fallback_min_delta,
            )[0]
            binary_scorer_choices[context.sample_id] = binary_chosen
        (
            chosen,
            fallback_used,
            fallback_reason,
            rejected_candidate,
            replacement_candidate,
        ) = choose_scorer(
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
                "source": context_source(context, source_by_sample_id),
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
                "binary_logistic_scorer_chosen": binary_chosen,
                "binary_logistic_candidate_scores": json.dumps(
                    binary_score_by_candidate,
                    sort_keys=True,
                ),
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
                    None
                    if rejected_failure is None
                    else int(replacement_failure) - int(rejected_failure)
                ),
            }
        )

    best_single_name, best_single_summary = best_single(contexts, candidates)
    static_name = (
        "static_weighted_downweight" if "static_weighted_downweight" in candidates else ""
    )
    static = candidate_summary(contexts, static_name) if static_name else best_single_summary
    scorer_summary = policy_summary(contexts, scorer_choices)
    primary_scorer_policy = scorer_policy_key(scorer)
    extra_scorer_choices: dict[str, T.Mapping[str, str]] = {
        primary_scorer_policy: scorer_choices,
    }
    if binary_scorer is not None:
        extra_scorer_choices["current_binary_logistic_scorer"] = binary_scorer_choices
    elif primary_scorer_policy == "current_binary_logistic_scorer":
        binary_scorer_choices = dict(scorer_choices)
    current_summary = policy_summary(contexts, current_choices)
    oracle_summary = policy_summary(contexts, oracle_choices)
    production_contexts = [
        context
        for context in contexts
        if context_source(context, source_by_sample_id) == SOURCE_PRODUCTION_VALIDATED
    ]
    gt_hard_all_contexts = [
        context
        for context in contexts
        if context_source(context, source_by_sample_id) == SOURCE_GT_HARD
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
    production_only_policy_metrics = policy_metric_bundle(
        production_contexts,
        candidates=candidates,
        scorer_policy_name=primary_scorer_policy,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
        extra_scorer_choices=extra_scorer_choices,
    )
    gt_hard_all_policy_metrics = policy_metric_bundle(
        gt_hard_all_contexts,
        candidates=candidates,
        scorer_policy_name=primary_scorer_policy,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
        extra_scorer_choices=extra_scorer_choices,
    )
    gt_roll_hard_policy_metrics = policy_metric_bundle(
        gt_roll_hard_contexts,
        candidates=candidates,
        scorer_policy_name=primary_scorer_policy,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
        extra_scorer_choices=extra_scorer_choices,
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
        production_scorer = production_only_policy_metrics[primary_scorer_policy]
        production_static = production_only_policy_metrics["static_weighted_downweight"]
        production_hrnet = production_only_policy_metrics.get("hrnet")
        if production_scorer["mean_nme"] >= production_static["mean_nme"]:
            production_failed_gates.append(
                "production_scorer_mean_nme_not_better_than_static_downweight"
            )
        if production_scorer["p90_nme"] > production_static["p90_nme"]:
            production_failed_gates.append(
                "production_scorer_p90_nme_regresses_vs_static_downweight"
            )
        if (
            production_scorer["failure_rate"]
            > production_static["failure_rate"] + epsilon_failure_rate
        ):
            production_failed_gates.append(
                "production_scorer_failure_rate_regresses_vs_static_downweight"
            )
        if (
            production_hrnet is not None
            and production_scorer["failure_rate"]
            > production_hrnet["failure_rate"] + epsilon_failure_rate
        ):
            production_failed_gates.append("production_scorer_failure_rate_regresses_vs_hrnet")
    if static_name and gt_hard_all_contexts:
        gt_hard_scorer = gt_hard_all_policy_metrics[primary_scorer_policy]
        gt_hard_static = gt_hard_all_policy_metrics["static_weighted_downweight"]
        if gt_hard_scorer["failure_rate"] > gt_hard_static["failure_rate"] + epsilon_failure_rate:
            gt_hard_failed_gates.append(
                "gt_hard_scorer_failure_rate_regresses_vs_static_downweight"
            )
    if derived_no_image_gt_hard_contexts and not allow_derived_no_image_gt_hard:
        gt_hard_failed_gates.append("gt_hard_derived_no_image_evidence_requires_explicit_allow")
    universal_failed_gates = [
        *combined_failed_gates,
        *production_failed_gates,
        *gt_hard_failed_gates,
    ]
    failed_gates = (
        production_failed_gates if promotion_scope == "production" else universal_failed_gates
    )

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
        "gt_hard_resolver_metadata": (
            "" if gt_hard_resolver_metadata is None else str(gt_hard_resolver_metadata)
        ),
        "candidate_count": len(candidates),
        "candidates": list(candidates),
        "scorer_path": str(scorer_path),
        "binary_scorer_path": "" if binary_scorer_path is None else str(binary_scorer_path),
        "scorer_comparison": {
            "context_count": len(contexts),
            "candidate_count": len(candidates),
            "uses_same_contexts": True,
            "uses_same_candidates": True,
            "binary_scorer_present": binary_scorer is not None,
        },
        "primary_scorer_policy": primary_scorer_policy,
        "scorer_model_type": scorer.model_type,
        "scorer_target": scorer.target,
        "promoted_scorer_version": scorer.version,
        "promoted_scorer_target": scorer.target,
        "promoted_scorer_label": primary_scorer_policy,
        "runtime_policy": "learned_quality_v1",
        "best_single": {"candidate": best_single_name, **best_single_summary},
        "static_weighted_downweight": {"candidate": static_name, **static},
        primary_scorer_policy: scorer_summary,
        RUNTIME_POLICY_REPORT_LABEL: current_summary,
        "oracle": oracle_summary,
        "fallback_count": fallback_count,
        "safe_fallback_count": safe_fallback_count,
        "hard_slice_fallback_count": hard_slice_fallback_count,
        "consensus_collapse_rejection_count": hard_slice_fallback_count,
        "consensus_collapse_fallback_count": hard_slice_fallback_count,
        "fallback_impact": fallback_impact_summary(fallback_impacts),
        "derived_no_image_sample_count": len(derived_no_image_contexts),
        "derived_no_image_gt_hard_sample_count": len(derived_no_image_gt_hard_contexts),
        "risk_floor_for_safe_fallback": risk_floor_for_safe_fallback,
        "safe_fallback_min_delta": safe_fallback_min_delta,
        "safe_fallback_tie_breaker": ["hrnet", "spiga", "orformer"],
        "per_bucket": per_bucket(contexts, scorer_choices),
        "production_only_policy_metrics": production_only_policy_metrics,
        "gt_hard_all_policy_metrics": gt_hard_all_policy_metrics,
        "gt_hard_only_policy_metrics": gt_hard_all_policy_metrics,
        "gt_roll_hard_policy_metrics": gt_roll_hard_policy_metrics,
    }
    if binary_scorer is not None:
        report["current_binary_logistic_scorer"] = policy_summary(
            contexts,
            binary_scorer_choices,
        )

    write_scorer_policy_outputs(
        report=report,
        rows=rows,
        scorer=scorer,
        output_dir=output_dir,
        worst_sample_count=worst_sample_count,
    )
    return report


__all__ = [
    "DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK",
    "DEFAULT_SAFE_FALLBACK_MIN_DELTA",
    "PROMOTION_SCOPES",
    "assert_lower_score_is_better",
    "evaluate_runtime_resolver_scorer",
]
