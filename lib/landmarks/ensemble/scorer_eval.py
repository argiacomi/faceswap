#!/usr/bin/env python3
"""Reusable runtime resolver scorer policy evaluation implementation."""

from __future__ import annotations

import csv
import json
import typing as T
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lib.landmarks.ensemble.production_artifacts import (
    LEARNED_POLICIES,
    ProductionBundleError,
    load_production_bundle,
)
from lib.landmarks.ensemble.runtime_resolver import (
    _hard_slice_safe_single_candidate,
    _high_risk_safe_fallback_candidate,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverLearnedScorer,
    load_runtime_resolver_scorer,
)
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    SampleCandidateContext,
    rows_for_context,
)
from lib.landmarks.ensemble.scorer_contexts import load_scorer_contexts
from lib.landmarks.ensemble.scorer_dataset import (
    read_scorer_dataset,
    resolve_scorer_dataset_path,
)
from lib.landmarks.ensemble.scorer_reports import write_scorer_policy_outputs
from lib.landmarks.ensemble.scorer_target_config import TARGET_TRANSFORM_REGRET_V3
from lib.landmarks.ensemble.strategies import canonical_strategy
from lib.landmarks.evaluation.geometry_signals import alignment_summary
from lib.landmarks.pipeline_conventions import (
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
)

RuntimeResolverLearnedScorerLike = RuntimeResolverLearnedScorer

DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK = 0.50
DEFAULT_SAFE_FALLBACK_MIN_DELTA = 0.05
DEFAULT_FALLBACK_CATASTROPHIC_WORSE_NME = 0.02
PROMOTION_SCOPES = ("universal", "production")
SCORER_VERSION_LEARNED_QUALITY_V3 = "learned_quality_v3"
RUNTIME_POLICY_REPORT_LABEL = "runtime_policy_learned_quality"
MIN_V3_ORACLE_GAP = 1e-4
HARD_SLICE_POLICY_BUCKETS = {
    "extreme_roll",
    "rolled_large_yaw_left",
    "rolled_large_yaw_right",
    "rolled_profile_left",
    "rolled_profile_right",
}
SCORER_REQUIRED_SPLITS: tuple[str, ...] = (
    "normal",
    "profile",
    "occlusion",
    "profile_occlusion",
    "production_failures",
)
SCORER_REGION_NAMES: tuple[str, ...] = (
    "jaw",
    "brows",
    "eyes",
    "nose",
    "mouth",
    "occluded_side",
    "visible_side",
)
SCORER_REGION_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "jaw": ((0, 17),),
    "brows": ((17, 22), (22, 27)),
    "eyes": ((36, 42), (42, 48)),
    "nose": ((27, 36),),
    "mouth": ((48, 60), (60, 68)),
    "right_side": ((0, 8), (17, 22), (36, 42)),
    "left_side": ((9, 17), (22, 27), (42, 48)),
}
PER_REGION_GEOMETRY_CSV = "per_region_geometry.csv"
PER_REGION_NME_CSV = "per_region_nme.csv"
PER_REGION_WORST_SAMPLES_JSON = "per_region_worst_samples.json"

SIDE_YAW_EVIDENCE_DEGREES = 5.0
HARD_BUCKET_PROMOTION_SPLITS: tuple[str, ...] = (
    "profile",
    "occlusion",
    "profile_occlusion",
    "production_failures",
)
HARD_BUCKET_MAX_FAILURE_RATE = 0.0
HARD_BUCKET_MAX_CATASTROPHIC_FAILURES = 0
V3_LEARNABILITY_MEAN_MARGIN = 1e-4
V3_LEARNABILITY_P95_TOLERANCE = 1e-4
V3_INVALID_SELECTION_RATE_THRESHOLD = 0.0
V3_NORMAL_BUCKET_REGRESSION_TOLERANCE = 1e-4
V3_HARD_BUCKET_LEARNABILITY_SPLITS: tuple[str, ...] = (
    "profile",
    "occlusion",
    "profile_occlusion",
)
V3_GEOMETRY_GATE_VOCABULARY: tuple[str, ...] = (
    "transform_error",
    "crop_center_error",
    "roll_error",
    "hull_iou",
    "catastrophic_failure",
)
REWEIGHTED_POLICY_METRICS: tuple[str, ...] = (
    "mean_nme",
    "p90_nme",
    "failure_rate",
    "oracle_match_rate",
    "mean_gap_vs_oracle",
    "mean_transform_regret_v3",
    "p95_transform_regret_v3",
    "oracle_match_rate_v3",
    "invalid_selection_rate_v3",
    "near_tie_excluded_count_v3",
    "zero_valid_group_count_v3",
    "too_few_valid_group_count_v3",
    "single_valid_group_count_v3",
    "transform_group_count_v3",
    "transform_eval_count_v3",
    "learnability_failed_gate_count_v3",
)
V3_RESERVED_ROW_FIELDS: tuple[str, ...] = (
    "transform_cost_v3",
    "transform_oracle_cost_v3",
    "transform_regret_v3",
    "transform_oracle_candidate_v3",
    "transform_oracle_gap_v3",
    "rankable_v3",
    "hard_invalid_v3",
    "hard_invalid_reasons_v3",
    "soft_structural_penalty_v3",
)


def eval_split_sources(path: Path) -> tuple[set[tuple[str, str, str]], dict[str, str]]:
    """Return held-out `(source, dataset, sample_id)` keys and per-sample source."""
    keys: set[tuple[str, str, str]] = set()
    source_by_sample_id: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or ()):
            raise ValueError(f"eval split {path} must contain a sample_id column")
        has_split = "split" in (reader.fieldnames or ())
        for row in reader:
            if has_split and str(row.get("split", "")).strip() not in {"", "eval"}:
                continue
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
        return str(context.source)
    if (
        context.dataset == SOURCE_PRODUCTION_VALIDATED
        or context.runtime_bucket_source == "stored_manifest_landmark_ensemble"
    ):
        return str(SOURCE_PRODUCTION_VALIDATED)
    return str(SOURCE_GT_HARD)


def summary(values: T.Sequence[float], failures: T.Sequence[bool]) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return {"mean_nme": 0.0, "p90_nme": 0.0, "failure_rate": 0.0}
    return {
        "mean_nme": float(arr.mean()),
        "p90_nme": float(np.percentile(arr, 90)),
        "failure_rate": float(sum(failures) / len(failures)) if failures else 0.0,
    }


def _percentile(values: T.Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype="float64"), percent))


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


def _v3_row_value(row: T.Any, name: str, default: T.Any = None) -> T.Any:
    if hasattr(row, name):
        return getattr(row, name)
    if isinstance(row, T.Mapping):
        return row.get(name, default)
    return default


def _v3_rows_by_candidate(context: T.Any) -> dict[str, T.Any]:
    rows: dict[str, T.Any] = {}
    for row in _context_rows_for_eval(context):
        candidate = str(_v3_row_value(row, "candidate_name", "") or "")
        if candidate and _v3_row_value(row, "transform_regret_v3") is not None:
            rows[candidate] = row
    return rows


def _v3_group_diagnostics(context: T.Any) -> tuple[dict[str, T.Any], bool, bool, bool]:
    rows = _v3_rows_by_candidate(context)
    valid = {
        name: row
        for name, row in rows.items()
        if _row_bool(_v3_row_value(row, "rankable_v3", False))
        and not _row_bool(_v3_row_value(row, "hard_invalid_v3", False))
    }
    zero_valid = not valid
    single_valid = len(valid) == 1
    if zero_valid or single_valid:
        return rows, zero_valid, single_valid, False
    oracle_gap = min(
        _row_float(_v3_row_value(row, "transform_oracle_gap_v3")) for row in valid.values()
    )
    return rows, zero_valid, False, oracle_gap < MIN_V3_ORACLE_GAP


def _transform_summary_empty() -> dict[str, T.Any]:
    return {
        "mean_transform_regret_v3": 0.0,
        "p95_transform_regret_v3": 0.0,
        "oracle_match_rate_v3": 0.0,
        "invalid_selection_count_v3": 0,
        "invalid_selection_rate_v3": 0.0,
        "near_tie_excluded_count_v3": 0,
        "zero_valid_group_count_v3": 0,
        "single_valid_group_count_v3": 0,
        "transform_group_count_v3": 0,
        "transform_eval_count_v3": 0,
    }


def transform_policy_summary_v3(
    contexts: T.Sequence[T.Any],
    choices: T.Mapping[str, str],
    *,
    source_by_sample_id: T.Mapping[str, str],
) -> dict[str, T.Any]:
    """Return v3 transform-regret metrics for labeled, non-production contexts."""
    regrets: list[float] = []
    oracle_matches = 0
    invalid_selection_count = 0
    near_tie_count = 0
    zero_valid_count = 0
    single_valid_count = 0
    eval_count = 0
    valid_eval_count = 0
    for context in contexts:
        if context_source(context, source_by_sample_id) == SOURCE_PRODUCTION_VALIDATED:
            continue
        rows, zero_valid, single_valid, near_tie = _v3_group_diagnostics(context)
        if not rows:
            continue
        eval_count += 1
        selected = choices.get(context.sample_id, "")
        selected_row = rows.get(selected)
        if zero_valid:
            zero_valid_count += 1
        if single_valid:
            single_valid_count += 1
        if near_tie:
            near_tie_count += 1
        selected_invalid = (
            selected_row is None
            or not _row_bool(_v3_row_value(selected_row, "rankable_v3", False))
            or _row_bool(_v3_row_value(selected_row, "hard_invalid_v3", False))
        )
        invalid_selection_count += int(selected_invalid)
        if selected_invalid or zero_valid or single_valid or near_tie:
            continue
        valid_eval_count += 1
        oracle = str(_v3_row_value(selected_row, "transform_oracle_candidate_v3", "") or "")
        oracle_matches += int(bool(oracle) and selected == oracle)
        regrets.append(max(_row_float(_v3_row_value(selected_row, "transform_regret_v3")), 0.0))

    if eval_count == 0:
        return _transform_summary_empty()
    return {
        "mean_transform_regret_v3": float(np.mean(regrets)) if regrets else 0.0,
        "p95_transform_regret_v3": _percentile(regrets, 95),
        "oracle_match_rate_v3": oracle_matches / valid_eval_count if valid_eval_count else 0.0,
        "invalid_selection_count_v3": invalid_selection_count,
        "invalid_selection_rate_v3": invalid_selection_count / eval_count,
        "near_tie_excluded_count_v3": near_tie_count,
        "zero_valid_group_count_v3": zero_valid_count,
        "single_valid_group_count_v3": single_valid_count,
        "transform_group_count_v3": eval_count,
        "transform_eval_count_v3": valid_eval_count,
    }


def policy_summary(
    contexts: T.Sequence[SampleCandidateContext],
    choices: T.Mapping[str, str],
    *,
    source_by_sample_id: T.Mapping[str, str] | None = None,
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
            "pick_counts": dict(Counter(choices.values())),  # type: ignore[dict-item]
            "oracle_match_rate": oracle_matches / len(contexts) if contexts else 0.0,
            "mean_gap_vs_oracle": float(np.mean(gaps)) if gaps else 0.0,
        }
    )
    base.update(
        transform_policy_summary_v3(
            contexts,
            choices,
            source_by_sample_id=source_by_sample_id or {},
        )
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


def _condition_tags(context: T.Any) -> set[str]:
    tags = {str(context.condition or "").strip().lower()}
    tags.add(str(context.runtime_bucket or "").strip().lower())
    tags.update(str(tag).strip().lower() for tag in getattr(context, "hard_case_tags", ()) or ())
    return {tag for tag in tags if tag}


def split_labels_for_context(
    context: T.Any,
    *,
    choices: T.Mapping[str, str],
    source_by_sample_id: T.Mapping[str, str],
) -> tuple[str, ...]:
    """Return report split labels for one evaluated context."""
    tags = _condition_tags(context)
    selected = choices.get(context.sample_id, "")
    labels: list[str] = []
    is_profile_like = any("profile" in tag or "large_yaw" in tag or "yaw_" in tag for tag in tags)
    is_occlusion_like = any("occlusion" in tag or "occluded" in tag for tag in tags)
    if not is_profile_like and not is_occlusion_like:
        labels.append("normal")
    if any("profile" in tag for tag in tags):
        labels.append("profile")
    if is_occlusion_like:
        labels.append("occlusion")
    if "profile_occlusion" in tags or "rolled_profile_occlusion" in tags:
        labels.append("profile_occlusion")
    if (
        context_source(context, source_by_sample_id) == SOURCE_PRODUCTION_VALIDATED
        and selected
        and bool(context.failure_by_candidate.get(selected, False))
    ):
        labels.append("production_failures")
    return tuple(dict.fromkeys(labels))


def split_metric_summary(
    contexts: T.Sequence[T.Any],
    *,
    choices: T.Mapping[str, str],
    source_by_sample_id: T.Mapping[str, str],
) -> dict[str, dict[str, T.Any]]:
    """Return required split-level scorer metrics."""
    grouped: dict[str, list[T.Any]] = {split: [] for split in SCORER_REQUIRED_SPLITS}
    for context in contexts:
        for label in split_labels_for_context(
            context,
            choices=choices,
            source_by_sample_id=source_by_sample_id,
        ):
            if label in grouped:
                grouped[label].append(context)

    payload: dict[str, dict[str, T.Any]] = {}
    for split in SCORER_REQUIRED_SPLITS:
        rows = grouped[split]
        selected_nme: list[float] = []
        failures: list[bool] = []
        gaps: list[float] = []
        catastrophic = 0
        for context in rows:
            selected = choices.get(context.sample_id, "")
            oracle = str(context.oracle)
            if selected not in context.nme_by_candidate or oracle not in context.nme_by_candidate:
                continue
            selected_value = float(context.nme_by_candidate[selected])
            oracle_value = float(context.nme_by_candidate[oracle])
            gap = selected_value - oracle_value
            selected_nme.append(selected_value)
            failures.append(bool(context.failure_by_candidate.get(selected, False)))
            gaps.append(gap)
            catastrophic += int(
                bool(context.failure_by_candidate.get(selected, False))
                or gap >= DEFAULT_FALLBACK_CATASTROPHIC_WORSE_NME
            )
        base = summary(selected_nme, failures)
        payload[split] = {
            "sample_count": len(selected_nme),
            "full_face_mean_nme": base["mean_nme"],
            "full_face_p90_nme": base["p90_nme"],
            "failure_rate": base["failure_rate"],
            "oracle_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
            "selected_vs_oracle_regret_mean": float(np.mean([max(gap, 0.0) for gap in gaps]))
            if gaps
            else 0.0,
            "selected_vs_oracle_regret_p90": _percentile([max(gap, 0.0) for gap in gaps], 90),
            "catastrophic_failure_count": catastrophic,
        }
    return payload


def _indices_for_region(context: T.Any, region: str) -> tuple[int, ...]:
    if region in {"jaw", "brows", "eyes", "nose", "mouth"}:
        return _indices_for_spans(SCORER_REGION_RANGES[region])

    visibility = getattr(context, "visibility", None)
    right_hidden = 0
    left_hidden = 0
    has_side_evidence = False

    if visibility is not None and len(visibility) >= 48:
        right_indices = _indices_for_spans(SCORER_REGION_RANGES["right_side"])
        left_indices = _indices_for_spans(SCORER_REGION_RANGES["left_side"])
        right_hidden = sum(1 for index in right_indices if not bool(visibility[index]))
        left_hidden = sum(1 for index in left_indices if not bool(visibility[index]))
        has_side_evidence = right_hidden != left_hidden

    yaw = getattr(context, "yaw_estimate", None)
    if right_hidden == left_hidden and yaw is not None:
        try:
            yaw_value = float(yaw)
        except (TypeError, ValueError):
            yaw_value = 0.0
        if yaw_value <= -SIDE_YAW_EVIDENCE_DEGREES:
            right_hidden += 1
            has_side_evidence = True
        elif yaw_value >= SIDE_YAW_EVIDENCE_DEGREES:
            left_hidden += 1
            has_side_evidence = True

    if not has_side_evidence or right_hidden == left_hidden:
        return ()

    if right_hidden > left_hidden:
        occluded = "right_side"
        visible = "left_side"
    else:
        occluded = "left_side"
        visible = "right_side"

    return _indices_for_spans(
        SCORER_REGION_RANGES[occluded if region == "occluded_side" else visible]
    )


def _indices_for_spans(spans: T.Sequence[tuple[int, int]]) -> tuple[int, ...]:
    indices: list[int] = []
    for start, end in spans:
        indices.extend(range(start, end))
    return tuple(indices)


def _region_nme(
    predicted: np.ndarray,
    truth: np.ndarray,
    indices: T.Sequence[int],
    *,
    normalizer: float,
) -> float:
    pred = np.asarray(predicted, dtype="float64")[list(indices), :2]
    gt = np.asarray(truth, dtype="float64")[list(indices), :2]
    return float(np.linalg.norm(pred - gt, axis=1).mean() / max(float(normalizer), 1e-9))


def _region_geometry_error(
    predicted: np.ndarray,
    truth: np.ndarray,
    indices: T.Sequence[int],
) -> float:
    pred_summary = alignment_summary(np.asarray(predicted, dtype="float32"))
    truth_summary = alignment_summary(np.asarray(truth, dtype="float32"))
    pred = pred_summary.aligned_landmarks[list(indices)]
    gt = truth_summary.aligned_landmarks[list(indices)]
    face_size = max(
        float(
            max(
                np.ptp(truth_summary.aligned_landmarks[:, 0]),
                np.ptp(truth_summary.aligned_landmarks[:, 1]),
            )
        ),
        1.0,
    )
    return float(np.linalg.norm(pred - gt, axis=1).mean() / face_size)


def choice_subset(
    contexts: T.Sequence[SampleCandidateContext],
    choices: T.Mapping[str, str],
) -> dict[str, str]:
    return {context.sample_id: choices[context.sample_id] for context in contexts}


def fixed_candidate_choices(
    contexts: T.Sequence[SampleCandidateContext],
    candidate: str,
) -> dict[str, str]:
    """Return per-sample choices for a fixed candidate baseline."""
    return {context.sample_id: candidate for context in contexts}


def bucket_key(context: T.Any) -> str:
    """Return the production/report bucket key for one context."""
    return str(context.runtime_bucket or context.condition or "unknown")


def production_bucket_mix(contexts: T.Sequence[SampleCandidateContext]) -> dict[str, T.Any]:
    """Return production bucket counts and rates without deriving GT labels."""
    counts = Counter(bucket_key(context) for context in contexts)
    total = sum(counts.values())
    return {
        "sample_count": total,
        "counts": dict(sorted(counts.items())),
        "rates": {bucket: count / total for bucket, count in sorted(counts.items())}
        if total
        else {},
    }


def context_has_feature_coverage(context: T.Any) -> bool:
    """Return whether a production context carries scorer-visible features."""
    rows = _context_rows_for_eval(context)
    if any(getattr(row, "feature_values", None) for row in rows):
        return True
    candidate_features = getattr(context, "candidate_extra_features", {}) or {}
    if any(candidate_features.values()):
        return True
    model_available = getattr(context, "model_predictions_available", None)
    return bool(model_available)


def context_has_invalid_detector(context: T.Any) -> bool:
    """Return whether no detector/model prediction is available for a production context."""
    model_available = getattr(context, "model_predictions_available", None)
    if isinstance(model_available, T.Mapping):
        return not any(bool(value) for value in model_available.values())
    return False


def production_only_diagnostics(
    contexts: T.Sequence[SampleCandidateContext],
    *,
    fallback_used_by_sample_id: T.Mapping[str, bool],
) -> dict[str, T.Any]:
    """Return production-only checks that do not require GT transform labels."""
    sample_count = len(contexts)
    fallback_count = sum(
        bool(fallback_used_by_sample_id.get(context.sample_id)) for context in contexts
    )
    feature_coverage_count = sum(context_has_feature_coverage(context) for context in contexts)
    invalid_detector_count = sum(context_has_invalid_detector(context) for context in contexts)
    return {
        "sample_count": sample_count,
        "bucket_mix": production_bucket_mix(contexts),
        "feature_coverage_count": feature_coverage_count,
        "feature_coverage_rate": feature_coverage_count / sample_count if sample_count else 0.0,
        "invalid_detector_count": invalid_detector_count,
        "invalid_detector_rate": invalid_detector_count / sample_count if sample_count else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / sample_count if sample_count else 0.0,
    }


def write_region_reports(
    *,
    contexts: T.Sequence[T.Any],
    choices: T.Mapping[str, str],
    source_by_sample_id: T.Mapping[str, str],
    output_dir: Path,
    failure_threshold: float,
    worst_sample_count: int,
) -> dict[str, T.Any]:
    """Write region-level NME/geometry reports and return a summary payload."""
    detail_rows: list[dict[str, T.Any]] = []
    for context in contexts:
        truth = getattr(context, "truth_landmarks", None)
        if truth is None:
            continue
        selected = choices.get(context.sample_id, "")
        candidate_by_name = {candidate.name: candidate for candidate in context.candidates}
        selected_candidate = candidate_by_name.get(selected)
        oracle_candidate = candidate_by_name.get(str(context.oracle))
        if selected_candidate is None or oracle_candidate is None:
            continue
        normalizer = float(getattr(context, "normalizer", None) or 0.0)
        if normalizer <= 0.0:
            normalizer = float(
                max(np.ptp(np.asarray(truth)[:, 0]), np.ptp(np.asarray(truth)[:, 1]), 1.0)
            )
        splits = split_labels_for_context(
            context,
            choices=choices,
            source_by_sample_id=source_by_sample_id,
        )
        if not splits:
            splits = ("normal",)
        for region in SCORER_REGION_NAMES:
            indices = _indices_for_region(context, region)
            if not indices:
                continue
            selected_nme = _region_nme(
                selected_candidate.landmarks,
                truth,
                indices,
                normalizer=normalizer,
            )
            oracle_nme = _region_nme(
                oracle_candidate.landmarks,
                truth,
                indices,
                normalizer=normalizer,
            )
            geometry_error = _region_geometry_error(selected_candidate.landmarks, truth, indices)
            row = {
                "source": context_source(context, source_by_sample_id),
                "sample_id": context.sample_id,
                "dataset": context.dataset,
                "condition": context.condition,
                "runtime_bucket": context.runtime_bucket,
                "hard_case_tags": "|".join(getattr(context, "hard_case_tags", ()) or ()),
                "splits": "|".join(splits),
                "region": region,
                "selected_candidate": selected,
                "oracle": context.oracle,
                "selected_region_nme": selected_nme,
                "oracle_region_nme": oracle_nme,
                "region_regret": selected_nme - oracle_nme,
                "region_failure": int(selected_nme > failure_threshold),
                "region_geometry_error": geometry_error,
                "region_geometry_failure": int(geometry_error > 0.05),
            }
            detail_rows.append(row)

    def aggregate(metric: str, failure_key: str) -> list[dict[str, T.Any]]:
        grouped: dict[tuple[str, str], list[dict[str, T.Any]]] = defaultdict(list)
        for row in detail_rows:
            for split in str(row["splits"]).split("|"):
                if split in SCORER_REQUIRED_SPLITS:
                    grouped[(split, str(row["region"]))].append(row)
        rows: list[dict[str, T.Any]] = []
        for split in SCORER_REQUIRED_SPLITS:
            for region in SCORER_REGION_NAMES:
                values = [float(row[metric]) for row in grouped.get((split, region), [])]
                failures = [bool(row[failure_key]) for row in grouped.get((split, region), [])]
                regrets = [
                    float(row.get("region_regret", 0.0))
                    for row in grouped.get((split, region), [])
                ]
                rows.append(
                    {
                        "split": split,
                        "region": region,
                        "sample_count": len(values),
                        "mean": float(np.mean(values)) if values else 0.0,
                        "p90": _percentile(values, 90),
                        "failure_rate": float(sum(failures) / len(failures)) if failures else 0.0,
                        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
                        "p90_regret": _percentile([max(value, 0.0) for value in regrets], 90),
                        "worst_sample_id": max(
                            grouped.get((split, region), []),
                            key=lambda row: float(row[metric]),
                            default={},
                        ).get("sample_id", ""),
                    }
                )
        return rows

    nme_rows = aggregate("selected_region_nme", "region_failure")
    geometry_rows = aggregate("region_geometry_error", "region_geometry_failure")

    if nme_rows:
        with (output_dir / PER_REGION_NME_CSV).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(nme_rows[0]))
            writer.writeheader()
            writer.writerows(nme_rows)
    if geometry_rows:
        with (output_dir / PER_REGION_GEOMETRY_CSV).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0]))
            writer.writeheader()
            writer.writerows(geometry_rows)

    worst: dict[str, list[dict[str, T.Any]]] = {}
    for region in SCORER_REGION_NAMES:
        region_rows = [row for row in detail_rows if row["region"] == region]
        worst[region] = sorted(
            region_rows,
            key=lambda row: (float(row["selected_region_nme"]), float(row["region_regret"])),
            reverse=True,
        )[:worst_sample_count]
    with (output_dir / PER_REGION_WORST_SAMPLES_JSON).open("w", encoding="utf-8") as handle:
        json.dump({"samples_by_region": worst}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    unavailable_reason = ""
    if not detail_rows:
        if not contexts:
            unavailable_reason = "no_contexts"
        elif all(getattr(context, "truth_landmarks", None) is None for context in contexts):
            unavailable_reason = "missing_truth_landmarks"
        else:
            unavailable_reason = "no_region_rows"

    return {
        "region_sample_rows": len(detail_rows),
        "per_region_nme_csv": str(output_dir / PER_REGION_NME_CSV),
        "per_region_geometry_csv": str(output_dir / PER_REGION_GEOMETRY_CSV),
        "per_region_worst_samples_json": str(output_dir / PER_REGION_WORST_SAMPLES_JSON),
        "region_metrics_available": bool(detail_rows),
        "region_metrics_unavailable_reason": unavailable_reason,
    }


def hard_bucket_promotion_gates(
    metrics_by_split: T.Mapping[str, T.Mapping[str, T.Any]],
) -> list[str]:
    """Return hard-bucket promotion failures for the selected scorer policy."""
    failed: list[str] = []
    for split in HARD_BUCKET_PROMOTION_SPLITS:
        metrics = metrics_by_split.get(split, {})
        sample_count = int(metrics.get("sample_count", 0) or 0)
        if sample_count <= 0:
            continue
        failure_rate = float(metrics.get("failure_rate", 0.0) or 0.0)
        catastrophic = int(metrics.get("catastrophic_failure_count", 0) or 0)
        if failure_rate > HARD_BUCKET_MAX_FAILURE_RATE:
            failed.append(f"{split}_failure_rate_above_hard_bucket_gate")
        if catastrophic > HARD_BUCKET_MAX_CATASTROPHIC_FAILURES:
            failed.append(f"{split}_catastrophic_failures_above_hard_bucket_gate")
    return failed


def _policy_metric_float(
    metrics: T.Mapping[str, T.Any],
    key: str,
    default: float = 0.0,
) -> float:
    try:
        return float(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _policy_metric_count(metrics: T.Mapping[str, T.Any], key: str) -> int:
    try:
        return int(metrics.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _v3_learnability_failure(
    *,
    gate: str,
    attribution: str,
    geometry_gate: str,
    observed: float | int,
    threshold: float | int,
    baseline_policy: str = "",
    detail: T.Mapping[str, T.Any] | None = None,
) -> dict[str, T.Any]:
    """Return one attributed Step 11 promotion-gate failure."""
    return {
        "gate": gate,
        "attribution": attribution,
        "geometry_gate": geometry_gate,
        "geometry_gate_vocabulary": list(V3_GEOMETRY_GATE_VOCABULARY),
        "observed": observed,
        "threshold": threshold,
        "baseline_policy": baseline_policy,
        "detail": dict(detail or {}),
    }


def _v3_baseline_candidates(
    comparison_metrics: T.Mapping[str, T.Mapping[str, T.Any]],
) -> list[tuple[str, T.Mapping[str, T.Any]]]:
    """Return simple v3 baselines that have transform-eval support."""
    baselines: list[tuple[str, T.Mapping[str, T.Any]]] = []
    for name in ("best_single", "static_weighted_downweight"):
        metrics = comparison_metrics.get(name, {})
        if _policy_metric_count(metrics, "transform_eval_count_v3") > 0:
            baselines.append((name, metrics))
    return baselines


def v3_learnability_promotion_gates(
    *,
    scorer_policy_name: str,
    comparison_metrics: T.Mapping[str, T.Mapping[str, T.Any]],
    normal_policy_metrics: T.Mapping[str, T.Mapping[str, T.Any]] | None,
    hard_bucket_failed_gates: T.Sequence[str],
    mean_margin: float = V3_LEARNABILITY_MEAN_MARGIN,
    p95_tolerance: float = V3_LEARNABILITY_P95_TOLERANCE,
    invalid_selection_rate_threshold: float = V3_INVALID_SELECTION_RATE_THRESHOLD,
    normal_bucket_tolerance: float = V3_NORMAL_BUCKET_REGRESSION_TOLERANCE,
) -> dict[str, T.Any]:
    """Return Step 11 v3 learnability promotion-gate results.

    These gates answer: can runtime-visible features predict the direct
    transform-regret oracle well enough to beat simple baselines?
    """
    scorer = comparison_metrics.get(scorer_policy_name, {})
    scorer_count = _policy_metric_count(scorer, "transform_eval_count_v3")
    failures: list[dict[str, T.Any]] = []
    skipped_gates: list[str] = []

    def fail(
        gate: str,
        *,
        attribution: str,
        geometry_gate: str,
        observed: float | int,
        threshold: float | int,
        baseline_policy: str = "",
        detail: T.Mapping[str, T.Any] | None = None,
    ) -> None:
        failures.append(
            _v3_learnability_failure(
                gate=gate,
                attribution=attribution,
                geometry_gate=geometry_gate,
                observed=observed,
                threshold=threshold,
                baseline_policy=baseline_policy,
                detail=detail,
            )
        )

    if scorer_count <= 0:
        fail(
            "transform_error_missing_rankable_pair_eval",
            attribution="ranker_or_runtime_feature_predictability_problem",
            geometry_gate="transform_error",
            observed=scorer_count,
            threshold=1,
            detail={"reason": "no rankable v3 pair groups reached promotion evaluation"},
        )

    scorer_mean = _policy_metric_float(scorer, "mean_transform_regret_v3")
    scorer_p95 = _policy_metric_float(scorer, "p95_transform_regret_v3")
    scorer_invalid = _policy_metric_float(scorer, "invalid_selection_rate_v3")

    best_single = comparison_metrics.get("best_single", {})
    best_single_count = _policy_metric_count(best_single, "transform_eval_count_v3")
    if scorer_count > 0 and best_single_count <= 0:
        fail(
            "transform_error_missing_best_single_baseline_eval",
            attribution="evaluation_coverage_problem",
            geometry_gate="transform_error",
            observed=best_single_count,
            threshold=1,
            baseline_policy="best_single",
            detail={"reason": "best_single baseline missing transform_eval_count_v3"},
        )

    for baseline_name in ("best_single", "static_weighted_downweight"):
        baseline = comparison_metrics.get(baseline_name, {})
        baseline_count = _policy_metric_count(baseline, "transform_eval_count_v3")
        if scorer_count <= 0 or baseline_count <= 0:
            skipped_gates.append(f"transform_error_mean_vs_{baseline_name}_skipped_missing_eval")
            continue
        baseline_mean = _policy_metric_float(baseline, "mean_transform_regret_v3")
        threshold = baseline_mean - mean_margin
        if scorer_mean > threshold:
            fail(
                f"transform_error_mean_not_learnable_vs_{baseline_name}",
                attribution="ranker_or_runtime_feature_predictability_problem",
                geometry_gate="transform_error",
                observed=scorer_mean,
                threshold=threshold,
                baseline_policy=baseline_name,
                detail={
                    "baseline_mean_transform_regret_v3": baseline_mean,
                    "margin": mean_margin,
                    "scorer_transform_eval_count_v3": scorer_count,
                    "baseline_transform_eval_count_v3": baseline_count,
                },
            )

    baselines = _v3_baseline_candidates(comparison_metrics)
    if scorer_count > 0 and baselines:
        baseline_name, baseline = min(
            baselines,
            key=lambda item: _policy_metric_float(item[1], "p95_transform_regret_v3"),
        )
        baseline_p95 = _policy_metric_float(baseline, "p95_transform_regret_v3")
        threshold = baseline_p95 + p95_tolerance
        if scorer_p95 > threshold:
            fail(
                "transform_error_p95_regresses_vs_best_available_baseline",
                attribution="ranker_or_runtime_feature_predictability_problem",
                geometry_gate="transform_error",
                observed=scorer_p95,
                threshold=threshold,
                baseline_policy=baseline_name,
                detail={
                    "baseline_p95_transform_regret_v3": baseline_p95,
                    "tolerance": p95_tolerance,
                },
            )
    elif scorer_count > 0:
        fail(
            "transform_error_missing_simple_baseline_eval",
            attribution="evaluation_coverage_problem",
            geometry_gate="transform_error",
            observed=0,
            threshold=1,
            detail={"reason": "no simple baseline had transform_eval_count_v3 > 0"},
        )

    if scorer_count > 0 and scorer_invalid > invalid_selection_rate_threshold:
        fail(
            "invalid_selection_rate_above_validity_detector_gate",
            attribution="validity_detector_or_runtime_feature_coverage_problem",
            geometry_gate="transform_error",
            observed=scorer_invalid,
            threshold=invalid_selection_rate_threshold,
            detail={
                "invalid_selection_count_v3": _policy_metric_count(
                    scorer, "invalid_selection_count_v3"
                ),
                "transform_eval_count_v3": scorer_count,
            },
        )

    normal_metrics = normal_policy_metrics or {}
    normal_scorer = normal_metrics.get(scorer_policy_name, {})
    normal_baselines = _v3_baseline_candidates(normal_metrics)
    normal_count = _policy_metric_count(normal_scorer, "transform_eval_count_v3")
    if normal_count > 0 and normal_baselines:
        normal_baseline_name, normal_baseline = min(
            normal_baselines,
            key=lambda item: _policy_metric_float(item[1], "mean_transform_regret_v3"),
        )
        normal_scorer_mean = _policy_metric_float(normal_scorer, "mean_transform_regret_v3")
        normal_baseline_mean = _policy_metric_float(normal_baseline, "mean_transform_regret_v3")
        normal_scorer_p95 = _policy_metric_float(normal_scorer, "p95_transform_regret_v3")
        normal_baseline_p95 = _policy_metric_float(normal_baseline, "p95_transform_regret_v3")
        normal_scorer_invalid = _policy_metric_float(normal_scorer, "invalid_selection_rate_v3")
        normal_baseline_invalid = _policy_metric_float(
            normal_baseline, "invalid_selection_rate_v3"
        )
        normal_reasons: list[str] = []
        if normal_scorer_mean > normal_baseline_mean + normal_bucket_tolerance:
            normal_reasons.append("mean_transform_regret_v3")
        if normal_scorer_p95 > normal_baseline_p95 + normal_bucket_tolerance:
            normal_reasons.append("p95_transform_regret_v3")
        if normal_scorer_invalid > max(
            invalid_selection_rate_threshold,
            normal_baseline_invalid + invalid_selection_rate_threshold,
        ):
            normal_reasons.append("invalid_selection_rate_v3")
        if normal_reasons:
            fail(
                "normal_bucket_no_regression",
                attribution="hard_case_query_weighting_or_contested_subset_overfit",
                geometry_gate="transform_error",
                observed=normal_scorer_mean,
                threshold=normal_baseline_mean + normal_bucket_tolerance,
                baseline_policy=normal_baseline_name,
                detail={
                    "regressed_metrics": normal_reasons,
                    "normal_scorer_metrics": dict(normal_scorer),
                    "normal_baseline_metrics": dict(normal_baseline),
                    "tolerance": normal_bucket_tolerance,
                },
            )
    else:
        skipped_gates.append("normal_bucket_no_regression_skipped_missing_eval")

    relevant_hard_bucket_failures = [
        gate
        for gate in hard_bucket_failed_gates
        if any(gate.startswith(f"{split}_") for split in V3_HARD_BUCKET_LEARNABILITY_SPLITS)
    ]
    if relevant_hard_bucket_failures:
        fail(
            "hard_bucket_catastrophic_failure_gate_failed",
            attribution="ranker_or_runtime_feature_predictability_problem",
            geometry_gate="catastrophic_failure",
            observed=len(relevant_hard_bucket_failures),
            threshold=0,
            detail={
                "failed_hard_bucket_gates": relevant_hard_bucket_failures,
                "required_splits": list(V3_HARD_BUCKET_LEARNABILITY_SPLITS),
            },
        )

    failed_gates = [str(item["gate"]) for item in failures]
    return {
        "status": "pass" if not failed_gates else "fail",
        "failed_gates": failed_gates,
        "skipped_gates": skipped_gates,
        "failures": failures,
        "thresholds": {
            "mean_margin": mean_margin,
            "p95_tolerance": p95_tolerance,
            "invalid_selection_rate_threshold": invalid_selection_rate_threshold,
            "normal_bucket_tolerance": normal_bucket_tolerance,
        },
        "scorer_policy": scorer_policy_name,
        "scorer_metrics": dict(scorer),
        "baseline_metrics": {
            name: dict(metrics)
            for name, metrics in comparison_metrics.items()
            if name in {"best_single", "static_weighted_downweight"}
        },
        "normal_bucket_policy_metrics": {
            name: dict(metrics)
            for name, metrics in normal_metrics.items()
            if isinstance(metrics, dict)
        },
        "hard_bucket_failed_gates": list(hard_bucket_failed_gates),
        "geometry_gate_vocabulary": list(V3_GEOMETRY_GATE_VOCABULARY),
    }


def policy_metric_bundle(
    contexts: T.Sequence[SampleCandidateContext],
    *,
    candidates: T.Sequence[str],
    scorer_policy_name: str,
    scorer_choices: T.Mapping[str, str],
    current_choices: T.Mapping[str, str],
    oracle_choices: T.Mapping[str, str],
    extra_scorer_choices: T.Mapping[str, T.Mapping[str, str]] | None = None,
    source_by_sample_id: T.Mapping[str, str] | None = None,
) -> dict[str, T.Any]:
    """Return selected-policy NME/failure summaries for one source slice."""
    source_lookup = source_by_sample_id or {}
    if not contexts:
        payload = {
            "sample_count": 0,
            scorer_policy_name: policy_summary((), {}, source_by_sample_id=source_lookup),
            RUNTIME_POLICY_REPORT_LABEL: policy_summary((), {}, source_by_sample_id=source_lookup),
            "oracle": policy_summary((), {}, source_by_sample_id=source_lookup),
        }
        for policy_name in extra_scorer_choices or {}:
            if policy_name == scorer_policy_name:
                continue
            payload[policy_name] = policy_summary((), {}, source_by_sample_id=source_lookup)
        return payload
    payload: dict[str, T.Any] = {  # type: ignore[no-redef]
        "sample_count": len(contexts),
        scorer_policy_name: policy_summary(
            contexts,
            choice_subset(contexts, scorer_choices),
            source_by_sample_id=source_lookup,
        ),
        RUNTIME_POLICY_REPORT_LABEL: policy_summary(
            contexts,
            choice_subset(contexts, current_choices),
            source_by_sample_id=source_lookup,
        ),
        "oracle": policy_summary(
            contexts,
            choice_subset(contexts, oracle_choices),
            source_by_sample_id=source_lookup,
        ),
    }
    for policy_name, policy_choices in (extra_scorer_choices or {}).items():
        if policy_name == scorer_policy_name:
            continue
        payload[policy_name] = policy_summary(
            contexts,
            choice_subset(contexts, policy_choices),
            source_by_sample_id=source_lookup,
        )
    best_single_name, _best_single_summary = best_single(contexts, candidates)
    payload["best_single"] = {
        "candidate": best_single_name,
        **policy_summary(
            contexts,
            fixed_candidate_choices(contexts, best_single_name),
            source_by_sample_id=source_lookup,
        ),
    }
    if "static_weighted_downweight" in candidates:
        payload["static_weighted_downweight"] = {
            "candidate": "static_weighted_downweight",
            **policy_summary(
                contexts,
                fixed_candidate_choices(contexts, "static_weighted_downweight"),
                source_by_sample_id=source_lookup,
            ),
        }
    if "hrnet" in candidates:
        payload["hrnet"] = {
            "candidate": "hrnet",
            **policy_summary(
                contexts,
                fixed_candidate_choices(contexts, "hrnet"),
                source_by_sample_id=source_lookup,
            ),
        }
    return payload


def _weighted_policy_payload(
    bucket_payloads: T.Mapping[str, T.Mapping[str, T.Any]],
    bucket_rates: T.Mapping[str, float],
) -> dict[str, T.Any]:
    policies = {
        policy_name
        for payload in bucket_payloads.values()
        for policy_name, value in payload.items()
        if isinstance(value, dict)
    }
    reweighted: dict[str, T.Any] = {}
    for policy_name in sorted(policies):
        weighted_metrics: dict[str, float] = {}
        covered_weight = 0.0
        for bucket, payload in bucket_payloads.items():
            rate = float(bucket_rates.get(bucket, 0.0))
            metrics = payload.get(policy_name)
            if not isinstance(metrics, dict) or rate <= 0.0:
                continue
            covered_weight += rate
            for metric in REWEIGHTED_POLICY_METRICS:
                if metric in metrics:
                    weighted_metrics[metric] = weighted_metrics.get(metric, 0.0) + (
                        rate * float(metrics[metric] or 0.0)
                    )
        if covered_weight <= 0.0:
            continue
        reweighted[policy_name] = {
            **weighted_metrics,
            "production_bucket_coverage_rate": covered_weight,
        }
    return reweighted


def production_bucket_reweighted_gt_policy_metrics(
    *,
    gt_contexts: T.Sequence[SampleCandidateContext],
    production_contexts: T.Sequence[SampleCandidateContext],
    candidates: T.Sequence[str],
    scorer_policy_name: str,
    scorer_choices: T.Mapping[str, str],
    current_choices: T.Mapping[str, str],
    oracle_choices: T.Mapping[str, str],
    extra_scorer_choices: T.Mapping[str, T.Mapping[str, str]] | None,
    source_by_sample_id: T.Mapping[str, str],
) -> dict[str, T.Any]:
    """Reweight labeled GT policy metrics using production bucket distribution."""
    production_mix = production_bucket_mix(production_contexts)
    bucket_rates = T.cast("dict[str, float]", production_mix["rates"])
    gt_by_bucket: dict[str, list[SampleCandidateContext]] = defaultdict(list)
    for context in gt_contexts:
        gt_by_bucket[bucket_key(context)].append(context)

    bucket_payloads: dict[str, dict[str, T.Any]] = {}
    missing_gt_buckets: list[str] = []
    for bucket in sorted(bucket_rates):
        bucket_contexts = gt_by_bucket.get(bucket, [])
        if not bucket_contexts:
            missing_gt_buckets.append(bucket)
            continue
        bucket_payloads[bucket] = policy_metric_bundle(
            bucket_contexts,
            candidates=candidates,
            scorer_policy_name=scorer_policy_name,
            scorer_choices=scorer_choices,
            current_choices=current_choices,
            oracle_choices=oracle_choices,
            extra_scorer_choices=extra_scorer_choices,
            source_by_sample_id=source_by_sample_id,
        )

    return {
        "sample_count": len(gt_contexts),
        "production_bucket_mix": production_mix,
        "missing_gt_buckets": missing_gt_buckets,
        "bucket_policy_metrics": bucket_payloads,
        "policies": _weighted_policy_payload(bucket_payloads, bucket_rates),
    }


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


def scorer_policy_key(scorer: RuntimeResolverLearnedScorerLike) -> str:
    """Return the stable report key for a learned scorer policy."""
    runtime_policy = str(getattr(scorer, "runtime_policy", "") or "")
    if runtime_policy in LEARNED_POLICIES:
        return runtime_policy
    version = str(getattr(scorer, "version", "") or getattr(scorer, "scorer_version", "") or "")
    if version in LEARNED_POLICIES:
        return version
    target = str(getattr(scorer, "target", "") or "")
    if target == TARGET_TRANSFORM_REGRET_V3:
        return SCORER_VERSION_LEARNED_QUALITY_V3
    return SCORER_VERSION_LEARNED_QUALITY_V3


def scorer_policy_key_for_path(path: Path, scorer: RuntimeResolverLearnedScorerLike) -> str:
    """Return a stable policy key for a scorer path/artifact."""
    name = path.stem
    if name in LEARNED_POLICIES:
        return name
    runtime_policy = str(getattr(scorer, "runtime_policy", "") or "")
    if runtime_policy in LEARNED_POLICIES:
        return runtime_policy
    version = str(getattr(scorer, "version", "") or getattr(scorer, "scorer_version", "") or "")
    if version in LEARNED_POLICIES:
        return version
    return scorer_policy_key(scorer)


def score_policy_choices(
    contexts: T.Sequence[SampleCandidateContext],
    scorer: RuntimeResolverLearnedScorerLike,
    *,
    risk_floor_for_safe_fallback: float,
    safe_fallback_min_delta: float,
    progress: T.Callable[[T.Sequence[T.Any], str], T.Iterable[T.Any]] | None = None,
) -> dict[str, str]:
    """Choose one candidate per sample for a scorer artifact."""
    choices: dict[str, str] = {}
    iterator = progress(contexts, "Score scorer policy") if progress is not None else contexts
    for context in iterator:
        context_rows = _context_rows_for_eval(context)
        scores = _score_context_rows(scorer, context_rows)
        choices[context.sample_id] = choose_scorer(
            context,
            scores,
            risk_floor_for_safe_fallback=risk_floor_for_safe_fallback,
            safe_fallback_min_delta=safe_fallback_min_delta,
        )[0]
    return choices


def load_installed_policy_scorers(
    scorer_dir: Path | None,
) -> tuple[str, dict[str, RuntimeResolverLearnedScorerLike], str]:
    """Load installed current production scorers keyed by runtime policy."""
    if scorer_dir is None:
        return "", {}, "not_configured"
    scorer_dir = scorer_dir.expanduser()
    if not scorer_dir.is_absolute():
        scorer_dir = scorer_dir.resolve()
    if not scorer_dir.is_dir():
        return "", {}, f"missing_scorer_dir:{scorer_dir}"

    active_policy = ""
    manifest_path = scorer_dir.parent / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if isinstance(manifest, dict):
            active_policy = str(manifest.get("active_policy") or "")

    if not active_policy:
        try:
            bundle = load_production_bundle()
        except ProductionBundleError:
            bundle = None
        if bundle is not None and (bundle.bundle_dir / "scorers").resolve() == scorer_dir:
            active_policy = str(bundle.active_policy or "")

    scorers: dict[str, RuntimeResolverLearnedScorerLike] = {}
    for path in sorted(scorer_dir.glob("*.json")):
        try:
            scorer = load_runtime_resolver_scorer(path)
            assert_lower_score_is_better(scorer)
        except (OSError, ValueError, RuntimeError):
            continue
        policy = scorer_policy_key_for_path(path, scorer)
        if policy in LEARNED_POLICIES:
            scorers[policy] = scorer

    if not active_policy and len(scorers) == 1:
        active_policy = next(iter(scorers))

    return active_policy, scorers, "loaded" if scorers else "no_scorers"


def installed_baseline_promotion_gates(
    *,
    selected_policy_metrics: T.Mapping[str, T.Mapping[str, T.Any]],
    installed_policy: str,
    installed_metrics: T.Mapping[str, T.Any] | None,
    promotion_policy: str,
) -> dict[str, T.Any]:
    """Return installed-baseline promotion gates for all selected policies."""
    gates: dict[str, T.Any] = {}
    if not installed_policy or installed_metrics is None:
        for policy_name in selected_policy_metrics:
            gates[policy_name] = {
                "status": "skipped_no_installed_baseline",
                "failed_gates": [],
                "installed_policy": installed_policy,
                "selected_policy": policy_name,
                "selected_metrics": selected_policy_metrics[policy_name],
                "installed_metrics": {},
            }
        return gates

    for policy_name, selected_metrics in selected_policy_metrics.items():
        failed: list[str] = []
        selected_transform_count = int(selected_metrics.get("transform_eval_count_v3", 0) or 0)
        installed_transform_count = int(installed_metrics.get("transform_eval_count_v3", 0) or 0)
        if selected_transform_count > 0 and installed_transform_count > 0:
            selected_regret = float(selected_metrics.get("mean_transform_regret_v3", 0.0) or 0.0)
            installed_regret = float(installed_metrics.get("mean_transform_regret_v3", 0.0) or 0.0)
            if selected_regret >= installed_regret:
                failed.append("transform_regret_v3_not_better_than_installed_current")
        else:
            selected_failure = float(selected_metrics.get("failure_rate", 0.0) or 0.0)
            installed_failure = float(installed_metrics.get("failure_rate", 0.0) or 0.0)
            selected_mean = float(selected_metrics.get("mean_nme", 0.0) or 0.0)
            installed_mean = float(installed_metrics.get("mean_nme", 0.0) or 0.0)
            if selected_failure > installed_failure:
                failed.append("failure_rate_regresses_vs_installed_current")
            if selected_mean >= installed_mean:
                failed.append("mean_nme_not_better_than_installed_current")
        gates[policy_name] = {
            "status": "pass" if not failed else "fail",
            "failed_gates": failed,
            "installed_policy": installed_policy,
            "selected_policy": policy_name,
            "selected_metrics": selected_metrics,
            "installed_metrics": installed_metrics,
            "is_requested_promotion_policy": policy_name == promotion_policy,
        }
    return gates


def assert_lower_score_is_better(scorer: RuntimeResolverLearnedScorerLike) -> None:
    """Fail fast if a scorer artifact cannot be ranked by ascending score."""
    if scorer.higher_is_better:
        raise ValueError(
            f"runtime resolver scorer {scorer.source_path or '<memory>'} is not lower-is-better"
        )


def _row_bool(value: T.Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row_float(value: T.Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_optional_float(value: T.Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_int(value: T.Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_geometry_reasons(value: T.Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value if str(item))
    raw = str(value).strip()
    if not raw:
        return ()
    return tuple(item for item in raw.split("|") if item)


def _row_feature_values(row: T.Mapping[str, T.Any]) -> dict[str, float]:
    features = row.get("features")
    if isinstance(features, dict):
        return {str(key): _row_float(value) for key, value in features.items()}

    features_json = row.get("features_json")
    if isinstance(features_json, str) and features_json.strip():
        try:
            decoded = json.loads(features_json)
        except json.JSONDecodeError:
            decoded = {}
        if isinstance(decoded, dict):
            return {str(key): _row_float(value) for key, value in decoded.items()}

    reserved = {
        "split",
        "source",
        "sample_id",
        "face_index",
        "dataset",
        "condition",
        "candidate_name",
        "candidate_nme",
        "oracle_nme",
        "regret_vs_oracle",
        "normalized_regret",
        "failure_label",
        "large_regret_label",
        "candidate_failure_or_high_gap",
        "selection_cost",
        "is_oracle",
        "was_selected_by_current_policy",
        "gap_vs_oracle",
        "runtime_bucket",
        "runtime_bucket_source",
        "hard_case_tags",
        "risk_route",
        "geometry_veto_reasons",
        "selected_by_current_policy",
        "selected_candidate_missing_from_eval",
        "oracle",
        "features_json",
        *V3_RESERVED_ROW_FIELDS,
    }
    payload: dict[str, float] = {}
    for key, value in row.items():
        if key in reserved:
            continue
        try:
            payload[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


def _context_rows_for_eval(context: T.Any) -> list[T.Any]:
    row_payload = getattr(context, "scorer_rows", None)
    if row_payload is not None:
        return list(row_payload)
    return rows_for_context(context)


def _score_context_rows(
    scorer: T.Any,
    context_rows: T.Sequence[T.Any],
) -> dict[str, float]:
    """Batch-score one sample's candidate rows for a scorer artifact."""
    rows = list(context_rows)
    feature_maps = [row.feature_values for row in rows]
    if hasattr(scorer, "score_feature_maps"):
        scores = scorer.score_feature_maps(feature_maps)
    else:  # defensive compatibility for ad-hoc scorer-like test doubles
        scores = [scorer.score_feature_map(feature_map) for feature_map in feature_maps]
    return {row.candidate_name: float(score) for row, score in zip(rows, scores, strict=True)}


def row_contexts_from_scorer_rows(
    scorer_rows: Path,
) -> tuple[list[T.Any], dict[str, str]]:
    """Build lightweight evaluation contexts directly from canonical scorer rows."""

    dataset = read_scorer_dataset(scorer_rows)
    source_rows = dataset.eval_rows or dataset.rows
    if not source_rows:
        raise ValueError(f"scorer rows {scorer_rows} did not contain any evaluation rows")

    groups: dict[tuple[str, str, str, int], list[dict[str, T.Any]]] = defaultdict(list)
    for row in source_rows:
        split = str(row.get("split", "")).strip()
        if split and split != "eval":
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        source = str(row.get("source", "")).strip()
        row_dataset = str(row.get("dataset", "")).strip()
        face_index = _row_int(row.get("face_index", 0))
        groups[(source, row_dataset, sample_id, face_index)].append(dict(row))

    if not groups:
        raise ValueError(f"scorer rows {scorer_rows} did not contain any eval rows")

    contexts: list[T.Any] = []
    source_by_sample_id: dict[str, str] = {}

    for (source, row_dataset, sample_id, face_index), rows in sorted(groups.items()):
        first = rows[0]
        nme_by_candidate: dict[str, float] = {}
        failure_by_candidate: dict[str, bool] = {}
        metrics: dict[str, T.Any] = {}
        candidates_payload: list[T.Any] = []
        scorer_row_payload: list[T.Any] = []

        oracle = str(first.get("oracle") or "")
        current_policy_choice = str(first.get("selected_by_current_policy") or "")
        runtime_bucket = str(first.get("runtime_bucket") or "")
        runtime_bucket_source = str(first.get("runtime_bucket_source") or "")
        hard_case_tags = tuple(
            tag for tag in str(first.get("hard_case_tags") or "").split("|") if tag
        )
        condition = str(first.get("condition") or "")
        selected_missing = _row_bool(first.get("selected_candidate_missing_from_eval", 0))

        for row in rows:
            candidate_name = str(row.get("candidate_name") or "").strip()
            if not candidate_name:
                continue
            candidate_nme = _row_float(row.get("candidate_nme"))
            candidate_failure = _row_bool(
                row.get("failure_label", row.get("candidate_failure_or_high_gap", 0))
            )
            feature_values = _row_feature_values(row)
            geometry_reasons = _row_geometry_reasons(row.get("geometry_veto_reasons"))

            nme_by_candidate[candidate_name] = candidate_nme
            failure_by_candidate[candidate_name] = candidate_failure
            metrics[candidate_name] = SimpleNamespace(geometry_veto_reasons=geometry_reasons)
            candidates_payload.append(
                SimpleNamespace(
                    name=candidate_name,
                    is_fusion=is_fusion_candidate(candidate_name),
                )
            )
            scorer_row_payload.append(
                SimpleNamespace(
                    candidate_name=candidate_name,
                    feature_values=feature_values,
                    transform_cost_v3=_row_optional_float(row.get("transform_cost_v3")),
                    transform_oracle_cost_v3=_row_optional_float(
                        row.get("transform_oracle_cost_v3")
                    ),
                    transform_regret_v3=_row_optional_float(row.get("transform_regret_v3")),
                    transform_oracle_candidate_v3=str(
                        row.get("transform_oracle_candidate_v3") or ""
                    ),
                    transform_oracle_gap_v3=_row_optional_float(
                        row.get("transform_oracle_gap_v3")
                    ),
                    rankable_v3=_row_bool(row.get("rankable_v3", False)),
                    hard_invalid_v3=_row_bool(row.get("hard_invalid_v3", False)),
                    hard_invalid_reasons_v3=_row_geometry_reasons(
                        row.get("hard_invalid_reasons_v3")
                    ),
                    soft_structural_penalty_v3=_row_float(row.get("soft_structural_penalty_v3")),
                )
            )

            if not oracle and _row_bool(row.get("is_oracle", 0)):
                oracle = candidate_name
            if not current_policy_choice and _row_bool(
                row.get("was_selected_by_current_policy", 0)
            ):
                current_policy_choice = candidate_name

        if not nme_by_candidate:
            continue

        if not oracle:
            oracle = min(nme_by_candidate, key=lambda name: (nme_by_candidate[name], name))
        if not current_policy_choice or current_policy_choice not in nme_by_candidate:
            current_policy_choice = oracle

        contexts.append(
            SimpleNamespace(
                sample_id=sample_id,
                face_index=face_index,
                dataset=row_dataset,
                source=source,
                condition=condition,
                candidates=tuple(candidates_payload),
                metrics=metrics,
                nme_by_candidate=nme_by_candidate,
                failure_by_candidate=failure_by_candidate,
                current_policy_choice=current_policy_choice,
                selected_candidate_missing_from_eval=selected_missing,
                oracle=oracle,
                runtime_bucket=runtime_bucket,
                runtime_bucket_source=runtime_bucket_source,
                hard_case_tags=hard_case_tags,
                risk_route=str(first.get("risk_route") or ""),
                candidate_extra_features={},
                scorer_rows=tuple(scorer_row_payload),
            )
        )
        if source:
            source_by_sample_id[sample_id] = source

    if not contexts:
        raise ValueError(f"scorer rows {scorer_rows} did not produce any evaluation contexts")

    return contexts, source_by_sample_id


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
    scorer_rows: Path | None = None,
    scorer_dataset: Path | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    epsilon_mean_nme: float = 0.001,
    epsilon_failure_rate: float = 0.0,
    worst_sample_count: int = 25,
    risk_floor_for_safe_fallback: float = DEFAULT_RISK_FLOOR_FOR_SAFE_FALLBACK,
    safe_fallback_min_delta: float = DEFAULT_SAFE_FALLBACK_MIN_DELTA,
    promotion_scope: str = "universal",
    promotion_policy: str = "",
    installed_scorer_dir: Path | None = None,
    allow_image_backfill: bool = False,
    allow_derived_no_image_gt_hard: bool = False,
    gt_hard_resolver_metadata: Path | None = None,
    progress: T.Callable[[T.Sequence[T.Any], str], T.Iterable[T.Any]] | None = None,
) -> dict[str, T.Any]:
    """Evaluate learned scorer policy and write reports."""
    if promotion_scope not in PROMOTION_SCOPES:
        raise ValueError(
            f"promotion_scope must be one of {PROMOTION_SCOPES}, got {promotion_scope!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if scorer_dataset is not None and scorer_rows is None:
        _dataset_dir, scorer_rows, _manifest = resolve_scorer_dataset_path(scorer_dataset)

    scorer = load_runtime_resolver_scorer(scorer_path)
    assert_lower_score_is_better(scorer)
    source_by_sample_id: dict[str, str] = {}
    if scorer_rows is not None:
        contexts, source_by_sample_id = row_contexts_from_scorer_rows(scorer_rows)
    else:
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
            progress=progress,
        )
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
    current_choices: dict[str, str] = {}
    oracle_choices: dict[str, str] = {}
    fallback_impacts: list[dict[str, T.Any]] = []
    fallback_count = 0
    safe_fallback_count = 0
    hard_slice_fallback_count = 0
    fallback_used_by_sample_id: dict[str, bool] = {}
    score_iter = progress(contexts, "Score resolver scorer") if progress is not None else contexts
    for context in score_iter:
        context_rows = _context_rows_for_eval(context)
        score_by_candidate = _score_context_rows(scorer, context_rows)
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
        fallback_used_by_sample_id[context.sample_id] = fallback_used
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
                "hard_case_tags": "|".join(getattr(context, "hard_case_tags", ())),
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
                    None
                    if rejected_failure is None
                    else int(replacement_failure) - int(rejected_failure)  # type: ignore[arg-type]
                ),
            }
        )

    metrics_by_split = split_metric_summary(
        contexts,
        choices=scorer_choices,
        source_by_sample_id=source_by_sample_id,
    )
    region_report_summary = write_region_reports(
        contexts=contexts,
        choices=scorer_choices,
        source_by_sample_id=source_by_sample_id,
        output_dir=output_dir,
        failure_threshold=failure_threshold,
        worst_sample_count=worst_sample_count,
    )
    hard_bucket_failed_gates = hard_bucket_promotion_gates(metrics_by_split)

    best_single_name, best_single_summary = best_single(contexts, candidates)
    best_single_report_summary = policy_summary(
        contexts,
        fixed_candidate_choices(contexts, best_single_name),
        source_by_sample_id=source_by_sample_id,
    )
    static_name = (
        "static_weighted_downweight" if "static_weighted_downweight" in candidates else ""
    )
    static = candidate_summary(contexts, static_name) if static_name else best_single_summary
    static_report_summary = (
        policy_summary(
            contexts,
            fixed_candidate_choices(contexts, static_name),
            source_by_sample_id=source_by_sample_id,
        )
        if static_name
        else best_single_report_summary
    )
    primary_scorer_policy = scorer_policy_key(scorer)
    scorer_summary = policy_summary(
        contexts,
        scorer_choices,
        source_by_sample_id=source_by_sample_id,
    )
    use_v3_transform_metrics = (
        primary_scorer_policy == SCORER_VERSION_LEARNED_QUALITY_V3
        or scorer.target == TARGET_TRANSFORM_REGRET_V3
    )
    extra_scorer_choices: dict[str, T.Mapping[str, str]] = {
        primary_scorer_policy: scorer_choices,
    }

    report_extra_scorer_choices = dict(extra_scorer_choices)

    installed_policy, installed_scorers, installed_status = load_installed_policy_scorers(
        installed_scorer_dir
    )
    installed_scorer_choices: dict[str, dict[str, str]] = {}
    for policy_name, installed_scorer in installed_scorers.items():
        installed_progress: T.Callable[[T.Sequence[T.Any], str], T.Iterable[T.Any]] | None = None
        if progress is not None:

            def installed_progress(
                values: T.Sequence[T.Any],
                desc: str,
                *,
                _policy_name: str = policy_name,
            ) -> T.Iterable[T.Any]:
                return progress(values, f"Score installed scorer [{_policy_name}]")

        installed_scorer_choices[policy_name] = score_policy_choices(
            contexts,
            installed_scorer,
            risk_floor_for_safe_fallback=risk_floor_for_safe_fallback,
            safe_fallback_min_delta=safe_fallback_min_delta,
            progress=installed_progress,
        )
        extra_scorer_choices[f"installed_current_{policy_name}"] = installed_scorer_choices[
            policy_name
        ]

    current_summary = policy_summary(
        contexts,
        current_choices,
        source_by_sample_id=source_by_sample_id,
    )
    oracle_summary = policy_summary(
        contexts,
        oracle_choices,
        source_by_sample_id=source_by_sample_id,
    )
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
        extra_scorer_choices=report_extra_scorer_choices,
        source_by_sample_id=source_by_sample_id,
    )
    gt_hard_all_policy_metrics = policy_metric_bundle(
        gt_hard_all_contexts,
        candidates=candidates,
        scorer_policy_name=primary_scorer_policy,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
        extra_scorer_choices=report_extra_scorer_choices,
        source_by_sample_id=source_by_sample_id,
    )
    gt_roll_hard_policy_metrics = policy_metric_bundle(
        gt_roll_hard_contexts,
        candidates=candidates,
        scorer_policy_name=primary_scorer_policy,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
        extra_scorer_choices=report_extra_scorer_choices,
        source_by_sample_id=source_by_sample_id,
    )
    production_diagnostics = production_only_diagnostics(
        production_contexts,
        fallback_used_by_sample_id=fallback_used_by_sample_id,
    )
    production_reweighted_gt_policy_metrics = production_bucket_reweighted_gt_policy_metrics(
        gt_contexts=gt_hard_all_contexts,
        production_contexts=production_contexts,
        candidates=candidates,
        scorer_policy_name=primary_scorer_policy,
        scorer_choices=scorer_choices,
        current_choices=current_choices,
        oracle_choices=oracle_choices,
        extra_scorer_choices=report_extra_scorer_choices,
        source_by_sample_id=source_by_sample_id,
    )
    v3_normal_policy_metrics: dict[str, T.Any] = {}
    if use_v3_transform_metrics:
        v3_normal_contexts = [
            context
            for context in gt_hard_all_contexts
            if "normal"
            in split_labels_for_context(
                context,
                choices=scorer_choices,
                source_by_sample_id=source_by_sample_id,
            )
        ]
        v3_normal_policy_metrics = policy_metric_bundle(
            v3_normal_contexts,
            candidates=candidates,
            scorer_policy_name=primary_scorer_policy,
            scorer_choices=scorer_choices,
            current_choices=current_choices,
            oracle_choices=oracle_choices,
            extra_scorer_choices=report_extra_scorer_choices,
            source_by_sample_id=source_by_sample_id,
        )

    v3_learnability_result: dict[str, T.Any] = {
        "status": "skipped_not_v3_transform_policy",
        "failed_gates": [],
        "skipped_gates": [],
        "failures": [],
    }

    combined_failed_gates: list[str] = []
    production_failed_gates: list[str] = []
    gt_hard_failed_gates: list[str] = []
    if use_v3_transform_metrics:
        comparison_metrics = (
            production_reweighted_gt_policy_metrics["policies"]
            if production_contexts
            else gt_hard_all_policy_metrics
        )
        v3_learnability_result = v3_learnability_promotion_gates(
            scorer_policy_name=primary_scorer_policy,
            comparison_metrics=comparison_metrics,
            normal_policy_metrics=v3_normal_policy_metrics,
            hard_bucket_failed_gates=hard_bucket_failed_gates,
        )
        combined_failed_gates.extend(
            str(gate) for gate in v3_learnability_result.get("failed_gates", ())
        )
    else:
        if static_name and scorer_summary["mean_nme"] >= static["mean_nme"]:
            combined_failed_gates.append("scorer_mean_nme_not_better_than_static_downweight")
        if static_name and scorer_summary["p90_nme"] > static["p90_nme"]:
            combined_failed_gates.append("scorer_p90_nme_regresses_vs_static_downweight")
        if (
            static_name
            and scorer_summary["failure_rate"] > static["failure_rate"] + epsilon_failure_rate
        ):
            combined_failed_gates.append("scorer_failure_rate_regresses_vs_static_downweight")
    if static_name and production_contexts and not use_v3_transform_metrics:
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
    if static_name and gt_hard_all_contexts and not use_v3_transform_metrics:
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
        *hard_bucket_failed_gates,
    ]
    failed_gates = universal_failed_gates
    if promotion_scope == "production":
        failed_gates = [
            *(combined_failed_gates if use_v3_transform_metrics else ()),
            *production_failed_gates,
            *hard_bucket_failed_gates,
        ]
    if not promotion_policy:
        promotion_policy = primary_scorer_policy

    selected_policy_metrics: dict[str, T.Mapping[str, T.Any]] = {}
    if production_contexts and use_v3_transform_metrics:
        selected_policy_metrics.update(
            {
                key: value
                for key, value in production_reweighted_gt_policy_metrics["policies"].items()
                if key in LEARNED_POLICIES and isinstance(value, dict)
            }
        )
    elif production_contexts:
        selected_policy_metrics.update(
            {
                key: value
                for key, value in production_only_policy_metrics.items()
                if key in LEARNED_POLICIES and isinstance(value, dict)
            }
        )
    else:
        selected_policy_metrics.update(
            {
                key: value
                for key, value in {primary_scorer_policy: scorer_summary}.items()
                if key in LEARNED_POLICIES and isinstance(value, dict) and value
            }
        )
    selected_policy_metrics.setdefault(
        primary_scorer_policy,
        gt_hard_all_policy_metrics.get(primary_scorer_policy, scorer_summary)
        if use_v3_transform_metrics
        else scorer_summary,
    )
    installed_metrics = None
    installed_choices = installed_scorer_choices.get(installed_policy)
    if installed_choices:
        if production_contexts and use_v3_transform_metrics:
            installed_metrics = policy_summary(
                gt_hard_all_contexts,
                choice_subset(gt_hard_all_contexts, installed_choices),
                source_by_sample_id=source_by_sample_id,
            )
        elif production_contexts:
            installed_metrics = policy_summary(
                production_contexts,
                choice_subset(production_contexts, installed_choices),
                source_by_sample_id=source_by_sample_id,
            )
        else:
            installed_metrics = policy_summary(
                contexts,
                installed_choices,
                source_by_sample_id=source_by_sample_id,
            )

    installed_baseline_gates = installed_baseline_promotion_gates(
        selected_policy_metrics=selected_policy_metrics,
        installed_policy=installed_policy,
        installed_metrics=installed_metrics,
        promotion_policy=promotion_policy,
    )
    if promotion_policy not in installed_baseline_gates:
        missing_status = (
            "fail"
            if installed_metrics is not None and installed_policy
            else "skipped_no_installed_baseline"
        )
        installed_baseline_gates[promotion_policy] = {
            "status": missing_status,
            "failed_gates": ["missing_selected_policy_metrics"]
            if missing_status == "fail"
            else [],
            "installed_policy": installed_policy,
            "selected_policy": promotion_policy,
            "selected_metrics": {},
            "installed_metrics": installed_metrics or {},
            "is_requested_promotion_policy": True,
        }
    requested_gate = installed_baseline_gates[promotion_policy]
    installed_failed_gates = list(requested_gate.get("failed_gates", []))
    installed_promotion_status = str(requested_gate.get("status") or "fail")
    using_installed_promotion_gate = installed_promotion_status not in {
        "",
        "skipped_no_installed_baseline",
        "skipped_missing_report_payload",
    }
    report_failed_gates = (
        installed_failed_gates if using_installed_promotion_gate else failed_gates
    )
    report_status = "pass" if not report_failed_gates else "fail"

    report: dict[str, T.Any] = {
        "status": report_status,
        "promotion_status": report_status,
        "promotion_scope": promotion_scope,
        "promotion_policy": promotion_policy,
        "promotion_gate_source": "installed_baseline"
        if using_installed_promotion_gate
        else "diagnostic",
        "installed_scorer_dir": "" if installed_scorer_dir is None else str(installed_scorer_dir),
        "installed_scorer_status": installed_status,
        "installed_current_policy": installed_policy,
        "installed_baseline_promotion_status": installed_promotion_status,
        "installed_baseline_failed_gates": installed_failed_gates,
        "installed_baseline_promotion": installed_baseline_gates,
        "failed_gates": report_failed_gates,
        "diagnostic_failed_gates": failed_gates,
        "diagnostic_universal_failed_gates": universal_failed_gates,
        "diagnostic_combined_failed_gates": combined_failed_gates,
        "diagnostic_production_failed_gates": production_failed_gates,
        "diagnostic_gt_hard_failed_gates": gt_hard_failed_gates,
        "diagnostic_hard_bucket_failed_gates": hard_bucket_failed_gates,
        "v3_learnability_promotion": v3_learnability_result,
        "v3_normal_bucket_policy_metrics": v3_normal_policy_metrics,
        "universal_failed_gates": universal_failed_gates,
        "combined_failed_gates": combined_failed_gates,
        "production_failed_gates": production_failed_gates,
        "gt_hard_failed_gates": gt_hard_failed_gates,
        "hard_bucket_failed_gates": hard_bucket_failed_gates,
        "production_gate_status": "pass",
        "gt_hard_gate_status": "pass" if not gt_hard_failed_gates else "diagnostic_fail",
        "sample_count": len(contexts),
        "row_backed_eval": scorer_rows is not None,
        "scorer_rows": "" if scorer_rows is None else str(scorer_rows),
        "heldout_eval": eval_split is not None or scorer_rows is not None,
        "eval_split": "" if eval_split is None else str(eval_split),
        "allow_image_backfill": allow_image_backfill,
        "allow_derived_no_image_gt_hard": allow_derived_no_image_gt_hard,
        "gt_hard_resolver_metadata": (
            "" if gt_hard_resolver_metadata is None else str(gt_hard_resolver_metadata)
        ),
        "candidate_count": len(candidates),
        "candidates": list(candidates),
        "scorer_path": str(scorer_path),
        "scorer_comparison": {
            "context_count": len(contexts),
            "candidate_count": len(candidates),
            "uses_same_contexts": True,
            "uses_same_candidates": True,
            "batched_policy_scoring": True,
        },
        "primary_scorer_policy": primary_scorer_policy,
        "scorer_model_type": scorer.model_type,
        "scorer_target": scorer.target,
        "promoted_scorer_version": scorer.version,
        "promoted_scorer_target": scorer.target,
        "promoted_scorer_label": primary_scorer_policy,
        "runtime_policy": promotion_policy,
        "best_single": {"candidate": best_single_name, **best_single_report_summary},
        "static_weighted_downweight": {"candidate": static_name, **static_report_summary},
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
        "metrics_by_split": metrics_by_split,
        "region_reports": region_report_summary,
        "production_only_policy_metrics": production_only_policy_metrics,
        "production_only_diagnostics": production_diagnostics,
        "production_bucket_mix": production_diagnostics["bucket_mix"],
        "production_bucket_reweighted_gt_policy_metrics": (
            production_reweighted_gt_policy_metrics
        ),
        "gt_hard_all_policy_metrics": gt_hard_all_policy_metrics,
        "gt_hard_only_policy_metrics": gt_hard_all_policy_metrics,
        "gt_roll_hard_policy_metrics": gt_roll_hard_policy_metrics,
    }

    # New normalized sections for #206. Keep legacy top-level fields above for
    # compatibility while giving new consumers one stable nested schema.
    report["promotion"] = {
        "status": report_status,
        "scope": promotion_scope,
        "policy": promotion_policy,
        "gate_source": report["promotion_gate_source"],
        "failed_gates": list(report_failed_gates),
        "installed_baseline": installed_baseline_gates,
        "learnability": v3_learnability_result,
    }
    report["diagnostics"] = {
        "failed_gates": list(failed_gates),
        "universal_failed_gates": list(universal_failed_gates),
        "combined_failed_gates": list(combined_failed_gates),
        "production_failed_gates": list(production_failed_gates),
        "gt_hard_failed_gates": list(gt_hard_failed_gates),
        "hard_bucket_failed_gates": list(hard_bucket_failed_gates),
        "row_backed_eval": scorer_rows is not None,
        "scorer_rows": "" if scorer_rows is None else str(scorer_rows),
        "derived_no_image_sample_count": len(derived_no_image_contexts),
        "derived_no_image_gt_hard_sample_count": len(derived_no_image_gt_hard_contexts),
    }
    report["metrics_by_source"] = {
        "production_validated": production_only_policy_metrics,
        "gt_hard": gt_hard_all_policy_metrics,
        "gt_roll_hard": gt_roll_hard_policy_metrics,
        "production_bucket_reweighted_gt": production_reweighted_gt_policy_metrics,
        "v3_normal_bucket": v3_normal_policy_metrics,
    }
    report["production_checks"] = production_diagnostics
    report["metrics_by_condition_split"] = metrics_by_split
    report["artifacts_by_region"] = region_report_summary
    report["fallbacks"] = {
        "fallback_count": fallback_count,
        "safe_fallback_count": safe_fallback_count,
        "hard_slice_fallback_count": hard_slice_fallback_count,
        "consensus_collapse_rejection_count": hard_slice_fallback_count,
        "impact": fallback_impact_summary(fallback_impacts),
    }
    report["artifacts"] = {
        "scorer_path": str(scorer_path),
    }
    report["primary_scorer"] = {
        "label": primary_scorer_policy,
        "version": scorer.version,
        "target": scorer.target,
        "model_type": scorer.model_type,
        "metrics": scorer_summary,
    }

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
    "v3_learnability_promotion_gates",
]
