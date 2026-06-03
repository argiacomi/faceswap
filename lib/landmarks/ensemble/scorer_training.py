#!/usr/bin/env python3
"""Reusable runtime resolver scorer training implementation."""

from __future__ import annotations

import csv
import hashlib
import math
import shutil
import typing as T
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lib.landmarks.datasets.hard_negative_mining import (
    DEFAULT_HARD_NEGATIVE_WEIGHT,
    MAX_HARD_NEGATIVE_WEIGHT,
)
from lib.landmarks.ensemble.profile_routing import is_profile_or_occlusion_context
from lib.landmarks.ensemble.runtime_features import (
    RUNTIME_FEATURE_CONTRACT_VERSION,
    runtime_feature_order,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import feature_matrix
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_HIGH_GAP_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    CandidateQualityRow,
    write_candidate_table_csv,
)
from lib.landmarks.ensemble.scorer_contexts import load_scorer_contexts
from lib.landmarks.ensemble.scorer_dataset import (
    SCORER_DATASET_DIR,
    write_scorer_dataset,
)
from lib.landmarks.ensemble.scorer_target_config import (
    DEFAULT_LARGE_COST_THRESHOLD,
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    SCORE_SEMANTICS_PREDICTED_COST,
    TARGET_TRANSFORM_REGRET_V3,
)
from lib.landmarks.ensemble.scorer_targets import (
    TaggedRow,
    scorer_candidate_table_rows,
    tagged_quality_rows,
    untag_quality_rows,
)
from lib.landmarks.pipeline_conventions import (
    SOURCE_PRODUCTION_VALIDATED,
    write_json,
)

SCORER_ARTIFACT = "runtime_resolver_scorer.json"
SCORER_V3_ARTIFACT = "runtime_resolver_scorer_v3.json"
SCORER_VERSION_LEARNED_QUALITY_V3 = "learned_quality_v3"
SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE = "learned_quality_v3_profile"
SCORER_V3_PROFILE_ARTIFACT = "runtime_resolver_scorer_v3_profile.json"
ACTIVE_SCORER_VERSION = SCORER_VERSION_LEARNED_QUALITY_V3
ACTIVE_SCORER_TARGET = TARGET_TRANSFORM_REGRET_V3


def is_profile_specialist_row(row: CandidateQualityRow) -> bool:
    """Return ``True`` when a row belongs to the profile/occlusion specialist scope."""
    return bool(is_profile_or_occlusion_context(row))


def filter_profile_specialist_rows(rows: T.Sequence[TaggedRow]) -> list[TaggedRow]:
    """Return only the rows that fall on the profile/occlusion route.

    Used to train the ``learned_quality_v3_profile`` specialist on profile,
    large-yaw, rolled, and occlusion contexts without disturbing the general
    scorer's normal/frontal/intermediate scope.
    """
    return [tagged for tagged in rows if is_profile_specialist_row(tagged[0])]


SCORERS_DIR = "scorers"
SCORER_SUITE_METRICS_JSON = "metrics.json"
SCORER_SUITE_SENTINEL_JSON = ".scorer_training_complete.json"
TRAINING_ROWS_CSV = "runtime_resolver_scorer_training_rows.csv"
EVAL_ROWS_CSV = "runtime_resolver_scorer_eval_rows.csv"
TRAINING_CANDIDATE_TABLE_CSV = "candidate_table.csv"
TRAINING_METRICS_JSON = "runtime_resolver_scorer_training_metrics.json"
TRAINING_V3_METRICS_JSON = "runtime_resolver_scorer_v3_training_metrics.json"
SCORER_CONDITION_REPORT_CSV = "scorer_report_by_condition.csv"
SCORER_REGION_REPORT_CSV = "scorer_report_by_region.csv"
MIN_V3_ORACLE_GAP = 1e-4
"""Minimum cost gap required to use a v3 group for ranking supervision."""
V3_REGRET_LABEL_CLAMP = 1.0
"""Maximum v3 transform regret mapped into LambdaRank relevance labels."""

HARD_CASE_SAMPLE_WEIGHTS: dict[str, float] = {
    "normal": 1.0,
    "profile": 2.0,
    "occlusion": 2.0,
    "profile_occlusion": 4.0,
    "production_failure": 5.0,
}
CROP_BREAKING_REGION_HINTS: tuple[str, ...] = (
    "jaw",
    "mouth",
    "eye",
    "eyes",
    "crop",
    "mask",
    "cloud_area",
    "bbox",
    "outside",
)


def split_tagged_rows(
    rows: T.Sequence[TaggedRow],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[TaggedRow], list[TaggedRow]]:
    """Split rows by sample while stratifying within source/dataset/condition groups."""
    if eval_fraction < 0.0 or eval_fraction >= 1.0:
        raise ValueError("--eval-fraction must be >= 0.0 and < 1.0")
    groups: dict[tuple[str, str, str], set[str]] = {}
    for row, source in rows:
        groups.setdefault((source, row.dataset, row.condition), set()).add(row.sample_id)

    eval_samples: set[tuple[str, str]] = set()
    for (source, dataset, condition), sample_ids in groups.items():
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: hashlib.sha256(
                "|".join((str(seed), source, dataset, condition, sample_id)).encode("utf-8")
            ).hexdigest(),
        )
        if eval_fraction <= 0.0 or len(ordered) <= 1:
            continue
        eval_count = max(1, int(round(len(ordered) * eval_fraction)))
        eval_count = min(eval_count, len(ordered) - 1)
        eval_samples.update((source, sample_id) for sample_id in ordered[:eval_count])

    train_rows: list[TaggedRow] = []
    eval_rows: list[TaggedRow] = []
    for tagged in rows:
        row, source = tagged
        if (source, row.sample_id) in eval_samples:
            eval_rows.append(tagged)
        else:
            train_rows.append(tagged)
    if not train_rows:
        raise ValueError("scorer split produced no training rows")
    return train_rows, eval_rows


def write_tagged_rows_csv(rows: T.Sequence[TaggedRow], path: Path) -> Path:
    """Write scorer rows with an explicit source column for held-out split reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = list(runtime_feature_order(row.feature_values for row, _source in rows))
    base_fieldnames = [
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
        "transform_cost_v3",
        "corner_delta_v3",
        "center_delta_v3",
        "scale_delta_v3",
        "roll_delta_degrees_v3",
        "fit_delta_v3",
        "transform_oracle_cost_v3",
        "transform_regret_v3",
        "transform_oracle_candidate_v3",
        "transform_oracle_gap_v3",
        "rankable_v3",
        "hard_invalid_v3",
        "hard_invalid_reasons_v3",
        "soft_structural_penalty_v3",
        "hard_negative_weight",
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
    ]
    fieldnames = ["source", *base_fieldnames, *feature_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, source in rows:
            writer.writerow(
                {
                    "source": source,
                    **row.to_csv_row(),
                    **{name: row.feature_values.get(name, 0.0) for name in feature_names},
                }
            )
    return path


def feature_order(rows: T.Sequence[CandidateQualityRow]) -> tuple[str, ...]:
    """Return stable runtime feature order for scorer training."""
    ordered: tuple[str, ...] = runtime_feature_order(row.feature_values for row in rows)
    return ordered


def scorer_target_value(row: CandidateQualityRow, target: str) -> float:
    """Return the configured scorer target value for one training row."""
    if target == TARGET_TRANSFORM_REGRET_V3:
        return float(row.transform_regret_v3)
    raise ValueError(f"unsupported scorer target {target!r}")


def _hard_case_split_label(row: CandidateQualityRow, source: str = "") -> str:
    """Return the highest-priority hard-case split label for weighting/reporting."""
    tags = {
        str(row.condition or "").strip().lower(),
        str(row.runtime_bucket or "").strip().lower(),
    }
    tags.update(str(tag).strip().lower() for tag in row.hard_case_tags or ())
    tags = {tag for tag in tags if tag}
    is_profile = any("profile" in tag or "large_yaw" in tag or "yaw_" in tag for tag in tags)
    is_occlusion = any("occlusion" in tag or "occluded" in tag for tag in tags)
    if source == SOURCE_PRODUCTION_VALIDATED and bool(row.failure_label):
        return "production_failure"
    if is_profile and is_occlusion:
        return "profile_occlusion"
    if is_profile:
        return "profile"
    if is_occlusion:
        return "occlusion"
    return "normal"


def _crop_breaking_weight_bonus(row: CandidateQualityRow) -> float:
    """Boost rows that show crop/mask/identity-critical geometry risks."""
    reasons = "|".join(row.geometry_veto_reasons or ()).lower()
    if any(hint in reasons for hint in CROP_BREAKING_REGION_HINTS):
        return 1.0
    if row.feature_values.get("has_geometry_veto", 0.0) > 0.0:
        return 0.5
    return 0.0


def _hard_negative_multiplier(row: CandidateQualityRow) -> float:
    """Return the mined hard-negative weight multiplier for one row.

    The hard-negative manifest builder writes ``metadata.hard_negative_weight``
    which flows onto :attr:`CandidateQualityRow.hard_negative_weight`. Samples
    from naturally-sampled manifests carry the neutral default and are
    unaffected.
    """
    try:
        weight = float(row.hard_negative_weight)
    except (TypeError, ValueError):
        return float(DEFAULT_HARD_NEGATIVE_WEIGHT)
    if not math.isfinite(weight) or weight <= 0.0:
        return float(DEFAULT_HARD_NEGATIVE_WEIGHT)
    return weight


def scorer_sample_weight(row: CandidateQualityRow, source: str = "") -> float:
    """Return sample weight for one candidate row.

    Normal: 1x
    Profile: 2x
    Occlusion: 2x
    Profile + occlusion: 4x
    Production failure: 5x minimum

    Mined hard-negative manifest weights (``metadata.hard_negative_weight``)
    multiply the condition weight and the combined value is capped at
    :data:`MAX_HARD_NEGATIVE_WEIGHT` to avoid training instability.
    """
    label = _hard_case_split_label(row, source)
    weight = HARD_CASE_SAMPLE_WEIGHTS[label]
    if source == SOURCE_PRODUCTION_VALIDATED and bool(row.failure_label):
        weight = max(weight, HARD_CASE_SAMPLE_WEIGHTS["production_failure"])
    weight += _crop_breaking_weight_bonus(row)
    if row.candidate_failure_or_high_gap:
        weight += 1.0
    multiplier = _hard_negative_multiplier(row)
    if multiplier > DEFAULT_HARD_NEGATIVE_WEIGHT:
        weight = min(weight * multiplier, max(weight, MAX_HARD_NEGATIVE_WEIGHT))
    return float(weight)


def scorer_sample_weighting_stats(tagged_rows: T.Sequence[TaggedRow]) -> dict[str, T.Any]:
    weights_by_row: list[float] = []
    multipliers_by_row: list[float] = []
    capped_by_row: list[bool] = []
    by_split: dict[str, dict[str, T.Any]] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    multiplier_counts: Counter[str] = Counter()

    for row, source in tagged_rows:
        weight = scorer_sample_weight(row, source)
        multiplier = _hard_negative_multiplier(row)
        weights_by_row.append(weight)
        multipliers_by_row.append(multiplier)
        multiplier_counts[f"{multiplier:.3g}"] += 1
        grouped[_hard_case_split_label(row, source)].append(weight)
        capped_by_row.append(
            multiplier > DEFAULT_HARD_NEGATIVE_WEIGHT
            and weight
            >= max(MAX_HARD_NEGATIVE_WEIGHT, HARD_CASE_SAMPLE_WEIGHTS["production_failure"])
        )

    weights = np.asarray(weights_by_row, dtype="float64")
    multipliers = np.asarray(multipliers_by_row, dtype="float64")
    for split in ("normal", "profile", "occlusion", "profile_occlusion", "production_failure"):
        values = grouped.get(split, [])
        by_split[split] = {
            "row_count": len(values),
            "mean_weight": float(np.mean(values)) if values else 0.0,
            "max_weight": float(np.max(values)) if values else 0.0,
            "saturated_at_hard_negative_cap_count": sum(
                value >= MAX_HARD_NEGATIVE_WEIGHT for value in values
            ),
        }

    saturated_count = int(sum(capped_by_row))
    multiplier_gt_default_count = (
        int(np.sum(multipliers > DEFAULT_HARD_NEGATIVE_WEIGHT)) if multipliers.size else 0
    )
    return {
        "strategy": "hard_case_weighting_single_scorer",
        "weights": HARD_CASE_SAMPLE_WEIGHTS,
        "row_count": int(weights.size),
        "mean_weight": float(np.mean(weights)) if weights.size else 0.0,
        "max_weight": float(np.max(weights)) if weights.size else 0.0,
        "hard_negative_multiplier_distribution": dict(sorted(multiplier_counts.items())),
        "hard_negative_multiplier_gt_default_count": multiplier_gt_default_count,
        "hard_negative_multiplier_gt_default_rate": (
            multiplier_gt_default_count / int(weights.size) if weights.size else 0.0
        ),
        "saturated_at_hard_negative_cap_count": saturated_count,
        "saturated_at_hard_negative_cap_rate": (
            saturated_count / int(weights.size) if weights.size else 0.0
        ),
        "by_split": by_split,
    }


def _region_labels_for_row(row: CandidateQualityRow) -> tuple[str, ...]:
    """Infer region labels from stable feature/diagnostic names."""
    labels: list[str] = []
    feature_blob = "|".join(
        [
            *(row.geometry_veto_reasons or ()),
            *(name for name, value in row.feature_values.items() if value),
        ]
    ).lower()
    for region in ("jaw", "brows", "eyes", "nose", "mouth", "occluded_side", "visible_side"):
        if region in feature_blob:
            labels.append(region)
    if not labels and (
        row.feature_values.get("has_geometry_veto", 0.0) > 0.0 or row.candidate_failure_or_high_gap
    ):
        labels.append("geometry_risk")
    return tuple(dict.fromkeys(labels or ("all",)))


def _training_report_rows(
    tagged_rows: T.Sequence[TaggedRow],
    *,
    group_by: str,
) -> list[dict[str, T.Any]]:
    grouped: dict[str, list[tuple[CandidateQualityRow, str]]] = defaultdict(list)
    for row, source in tagged_rows:
        if group_by == "condition":
            grouped[_hard_case_split_label(row, source)].append((row, source))
        elif group_by == "region":
            for region in _region_labels_for_row(row):
                grouped[region].append((row, source))
        else:
            raise ValueError(f"unknown report group {group_by!r}")

    output: list[dict[str, T.Any]] = []
    for label in sorted(grouped):
        rows = grouped[label]
        weights = np.asarray([scorer_sample_weight(row, source) for row, source in rows])
        regret = np.asarray([max(row.candidate_nme - row.oracle_nme, 0.0) for row, _ in rows])
        selection_cost = np.asarray([row.selection_cost for row, _ in rows])
        failures = np.asarray([float(row.failure_label) for row, _ in rows])
        high_gap = np.asarray([float(row.candidate_failure_or_high_gap) for row, _ in rows])
        output.append(
            {
                group_by: label,
                "row_count": len(rows),
                "mean_weight": float(np.mean(weights)) if len(weights) else 0.0,
                "mean_oracle_regret": float(np.mean(regret)) if len(regret) else 0.0,
                "p90_oracle_regret": float(np.percentile(regret, 90)) if len(regret) else 0.0,
                "mean_selection_cost": float(np.mean(selection_cost))
                if len(selection_cost)
                else 0.0,
                "failure_rate": float(np.mean(failures)) if len(failures) else 0.0,
                "failure_or_high_gap_rate": float(np.mean(high_gap)) if len(high_gap) else 0.0,
            }
        )
    return output


def write_training_report_csv(rows: T.Sequence[dict[str, T.Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0])
        if rows
        else [
            "group",
            "row_count",
            "mean_weight",
            "mean_oracle_regret",
            "p90_oracle_regret",
            "mean_selection_cost",
            "failure_rate",
            "failure_or_high_gap_rate",
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def target_distribution_stats(
    rows: T.Sequence[CandidateQualityRow],
    *,
    target: str,
    large_cost_threshold: float = DEFAULT_LARGE_COST_THRESHOLD,
) -> dict[str, T.Any]:
    """Return training-target distribution diagnostics."""
    values = np.asarray([scorer_target_value(row, target) for row in rows], dtype="float64")
    if values.size == 0:
        return {
            "target": target,
            "target_mean": 0.0,
            "target_p50": 0.0,
            "target_p90": 0.0,
            "target_p99": 0.0,
            "zero_cost_rate": 0.0,
            "large_cost_rate": 0.0,
            "large_cost_threshold": large_cost_threshold,
        }
    return {
        "target": target,
        "target_mean": float(np.mean(values)),
        "target_p50": float(np.percentile(values, 50)),
        "target_p90": float(np.percentile(values, 90)),
        "target_p99": float(np.percentile(values, 99)),
        "zero_cost_rate": float(np.mean(values <= 0.0)),
        "large_cost_rate": float(np.mean(values >= large_cost_threshold)),
        "large_cost_threshold": large_cost_threshold,
    }


def _v3_query_condition_label(row: CandidateQualityRow, source: str = "") -> str:
    """Return the face-level condition label for v3 query weighting."""
    del source
    tags = {
        str(row.condition or "").strip().lower(),
        str(row.runtime_bucket or "").strip().lower(),
    }
    tags.update(str(tag).strip().lower() for tag in row.hard_case_tags or ())
    tags = {tag for tag in tags if tag}
    is_profile = any("profile" in tag or "large_yaw" in tag or "yaw_" in tag for tag in tags)
    is_occlusion = any("occlusion" in tag or "occluded" in tag for tag in tags)
    if is_profile and is_occlusion:
        return "profile_occlusion"
    if is_profile:
        return "profile"
    if is_occlusion:
        return "occlusion"
    return "normal"


def v3_lambdarank_query_weight(row: CandidateQualityRow, source: str = "") -> float:
    """Return face-level v3 query weight without candidate-specific signals.

    Mined hard-negative manifest weights multiply the condition weight; the
    combined value never reduces the base condition weight and is capped at
    :data:`MAX_HARD_NEGATIVE_WEIGHT`.
    """
    label = _v3_query_condition_label(row, source)
    weight = float(HARD_CASE_SAMPLE_WEIGHTS[label])
    multiplier = _hard_negative_multiplier(row)
    if multiplier > DEFAULT_HARD_NEGATIVE_WEIGHT:
        weight = min(weight * multiplier, max(weight, MAX_HARD_NEGATIVE_WEIGHT))
    return weight


def _v3_sample_group_key(row: CandidateQualityRow, source: str) -> tuple[str, str, str, str]:
    return source, row.dataset, row.condition, row.sample_id


def grouped_rankable_rows_v3(
    rows: T.Sequence[TaggedRow],
) -> tuple[list[list[TaggedRow]], list[float], dict[str, T.Any]]:
    """Return valid v3 LambdaRank groups and one query weight per group."""
    groups: dict[tuple[str, str, str, str], list[TaggedRow]] = {}
    for tagged in rows:
        row, source = tagged
        groups.setdefault(_v3_sample_group_key(row, source), []).append(tagged)

    valid_groups: list[list[TaggedRow]] = []
    query_weights: list[float] = []
    fallback_abstain_groups = 0
    fallback_abstain_rows = 0
    single_valid_groups = 0
    single_valid_rows = 0
    hard_invalid_rows = 0
    near_tie_groups = 0
    near_tie_rows = 0

    for key in sorted(groups):
        group = groups[key]
        valid = [
            tagged
            for tagged in group
            if bool(tagged[0].rankable_v3) and not bool(tagged[0].hard_invalid_v3)
        ]

        hard_invalid_rows += sum(1 for row, _source in group if bool(row.hard_invalid_v3))

        if not valid:
            fallback_abstain_groups += 1
            fallback_abstain_rows += len(group)
            continue

        if len(valid) < 2:
            single_valid_groups += 1
            single_valid_rows += len(group)
            continue

        oracle_gap = min(float(row.transform_oracle_gap_v3) for row, _source in valid)
        if oracle_gap < MIN_V3_ORACLE_GAP:
            near_tie_groups += 1
            near_tie_rows += len(group)
            continue

        ordered = sorted(valid, key=lambda tagged: tagged[0].candidate_name)
        row, source = ordered[0]
        valid_groups.append(ordered)
        query_weights.append(v3_lambdarank_query_weight(row, source))

    stats = {
        "total_group_count": len(groups),
        "rankable_pair_group_count": (
            len(groups) - fallback_abstain_groups - single_valid_groups - near_tie_groups
        ),
        "fallback_abstain_group_count": fallback_abstain_groups,
        "fallback_abstain_row_count": fallback_abstain_rows,
        "single_valid_group_count": single_valid_groups,
        "single_valid_row_count": single_valid_rows,
        "near_tie_group_count": near_tie_groups,
        "near_tie_row_count": near_tie_rows,
        "min_v3_oracle_gap": MIN_V3_ORACLE_GAP,
        "rankable_row_count": sum(len(group) for group in valid_groups),
        "hard_invalid_row_count": hard_invalid_rows,
    }

    return valid_groups, query_weights, stats


def _flatten_grouped_rows(groups: T.Sequence[T.Sequence[TaggedRow]]) -> list[TaggedRow]:
    """Return grouped rows flattened in group order."""
    return [tagged for group in groups for tagged in group]


def _v3_lambdarank_item_weights(
    query_weights: T.Sequence[float],
    group_sizes: T.Sequence[int],
) -> np.ndarray:
    """Return per-row LightGBM weights from one query weight per group."""
    weights: list[float] = []
    for weight, size in zip(query_weights, group_sizes, strict=True):
        weights.extend([float(weight)] * size)
    return T.cast(np.ndarray, np.asarray(weights, dtype="float64"))


def _lambdarank_label_v3(row: CandidateQualityRow) -> int:
    """Return v3 relevance where higher means lower transform regret.

    This intentionally ignores candidate_nme, oracle_nme, and selection_cost.
    """
    if V3_REGRET_LABEL_CLAMP <= 0.0:
        return 0
    clipped = min(max(float(row.transform_regret_v3), 0.0), V3_REGRET_LABEL_CLAMP)
    return round((1.0 - clipped / V3_REGRET_LABEL_CLAMP) * 30.0)


def _train_v3_lambdarank_from_tagged_rows(
    *,
    train_tagged_rows: T.Sequence[TaggedRow],
    eval_tagged_rows: T.Sequence[TaggedRow],
    candidates: T.Sequence[str],
    output_dir: Path,
    failure_threshold: float,
    eval_fraction: float,
    split_seed: int,
    learning_rate: float,
    iterations: int,
    num_leaves: int,
    policy: str = SCORER_VERSION_LEARNED_QUALITY_V3,
    artifact_filename: str = SCORER_V3_ARTIFACT,
) -> dict[str, T.Any]:
    """Train a v3 LambdaRank scorer (``policy``) from rankable rows only.

    ``policy`` is written into the artifact's version/runtime_policy so the
    general scorer and the ``learned_quality_v3_profile`` specialist share the
    same training path and target, differing only by routed scope.
    """

    try:
        import lightgbm as lgb
    except ModuleNotFoundError as err:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "learned_quality_v3 training requires lightgbm; install project requirements first"
        ) from err

    output_dir.mkdir(parents=True, exist_ok=True)
    train_groups, train_query_weights, train_filter_stats = grouped_rankable_rows_v3(
        train_tagged_rows
    )
    eval_groups, _eval_query_weights, eval_filter_stats = grouped_rankable_rows_v3(
        eval_tagged_rows
    )

    if not train_groups:
        raise ValueError(
            "learned_quality_v3 has no rankable training rows after hard-invalid filtering"
        )

    train_group_sizes = [len(group) for group in train_groups]
    grouped_train = _flatten_grouped_rows(train_groups)
    rankable_eval = _flatten_grouped_rows(eval_groups)
    train_rows = untag_quality_rows(grouped_train)
    features = feature_order(train_rows)
    x = feature_matrix([row.feature_values for row in train_rows], features)
    y = np.asarray([_lambdarank_label_v3(row) for row in train_rows], dtype="int32")
    item_weights = _v3_lambdarank_item_weights(train_query_weights, train_group_sizes)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=iterations,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=split_seed,
        deterministic=True,
        verbosity=-1,
    )
    ranker.fit(x, y, group=train_group_sizes, sample_weight=item_weights)
    booster = ranker.booster_
    importances = _feature_importance_map(
        features,
        booster.feature_importance(importance_type="gain"),
    )
    artifact = {
        "artifact_schema_version": 3,
        "version": policy,
        "scorer_version": policy,
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "target": TARGET_TRANSFORM_REGRET_V3,
        "objective": "lambdarank_visible_transform_regret",
        "training_mode": "grouped_lambdarank_rankable_v3_only",
        "selection_target": "inverse_transform_regret_v3_rank",
        "runtime_policy": policy,
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "failure_threshold": failure_threshold,
        "features": list(features),
        "runtime_feature_contract_version": RUNTIME_FEATURE_CONTRACT_VERSION,
        "model_data": booster.model_to_string(),
        "training_data_counts": {
            "row_count": len(train_rows),
            "sample_group_count": len(train_group_sizes),
            "eval_row_count": len(rankable_eval),
            "candidate_count": len(candidates),
        },
        "split_ids": {
            "seed": split_seed,
            "eval_fraction": eval_fraction,
            "train_group_count": len(train_group_sizes),
        },
        "v3_filtering": {
            "train": train_filter_stats,
            "eval": eval_filter_stats,
        },
        "feature_importances": importances,
        "calibration": {"type": "none", "params": {}},
        "sample_weighting": {
            "strategy": "v3_condition_query_weighting",
            "row_count": len(train_rows),
            "query_count": len(train_group_sizes),
            "mean_weight": float(np.mean(item_weights)) if item_weights.size else 0.0,
            "max_weight": float(np.max(item_weights)) if item_weights.size else 0.0,
        },
        "lightgbm_params": {
            "objective": "lambdarank",
            "n_estimators": iterations,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": split_seed,
            "deterministic": True,
        },
    }
    artifact_path = write_json(output_dir / artifact_filename, artifact)
    rows_path = write_tagged_rows_csv(grouped_train, output_dir / TRAINING_ROWS_CSV)
    eval_rows_path = write_tagged_rows_csv(rankable_eval, output_dir / EVAL_ROWS_CSV)
    importances_path = _write_feature_importance_csv(
        output_dir / "runtime_resolver_scorer_v3_feature_importances.csv",
        importances,
    )
    metrics: dict[str, T.Any] = {
        "artifact": str(artifact_path),
        "training_rows": str(rows_path),
        "eval_rows": str(eval_rows_path),
        "feature_importances": str(importances_path),
        "candidate_count": len(candidates),
        "candidates": list(candidates),
        "feature_count": len(features),
        "runtime_feature_contract_version": RUNTIME_FEATURE_CONTRACT_VERSION,
        "target": TARGET_TRANSFORM_REGRET_V3,
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "split_seed": split_seed,
        "eval_fraction": eval_fraction,
        "training_data_counts": artifact["training_data_counts"],
        "split_ids": artifact["split_ids"],
        "v3_filtering": artifact["v3_filtering"],
        "sample_weighting": artifact["sample_weighting"],
        "lightgbm_params": artifact["lightgbm_params"],
    }
    metrics_path = write_json(output_dir / TRAINING_V3_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def _feature_importance_map(
    features: T.Sequence[str],
    importances: T.Sequence[float],
) -> dict[str, float]:
    return {
        feature: float(value)
        for feature, value in sorted(
            zip(features, importances, strict=True),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
    }


def _write_feature_importance_csv(path: Path, importances: T.Mapping[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
        writer.writeheader()
        for feature, importance in importances.items():
            writer.writerow({"feature": feature, "importance": importance})
    return path


def train_runtime_resolver_scorer_suite(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    candidates: T.Sequence[str],
    output_dir: Path,
    gt_hard_resolver_metadata: Path | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    high_gap_threshold: float = DEFAULT_HIGH_GAP_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    learning_rate: float = 0.05,
    iterations: int = 150,
    num_leaves: int = 31,
    eval_fraction: float = 0.20,
    split_seed: int = 42,
    allow_image_backfill: bool = False,
    progress: T.Callable[[T.Sequence[T.Any], str], T.Iterable[T.Any]] | None = None,
) -> dict[str, T.Any]:
    """Train the active v3 scorer from one canonical row split."""

    output_dir.mkdir(parents=True, exist_ok=True)
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
        require_gt_hard_metadata=True,
        progress=progress,
    )
    tagged_rows = tagged_quality_rows(contexts, high_gap_threshold=high_gap_threshold)
    if not tagged_rows:
        raise ValueError("no scorer training rows were loaded")
    train_tagged_rows, eval_tagged_rows = split_tagged_rows(
        tagged_rows,
        eval_fraction=eval_fraction,
        seed=split_seed,
    )

    dataset_manifest = write_scorer_dataset(
        train_rows=train_tagged_rows,
        eval_rows=eval_tagged_rows,
        output_dir=output_dir / SCORER_DATASET_DIR,
        inputs={
            "gt_manifest": "" if gt_manifest is None else str(gt_manifest),
            "production_manifest": "" if production_manifest is None else str(production_manifest),
            "weights": str(weights_path),
            "gt_hard_resolver_metadata": (
                "" if gt_hard_resolver_metadata is None else str(gt_hard_resolver_metadata)
            ),
        },
        config={
            "active_scorer_version": ACTIVE_SCORER_VERSION,
            "active_target": ACTIVE_SCORER_TARGET,
            "candidates": list(candidates),
            "failure_threshold": failure_threshold,
            "high_gap_threshold": high_gap_threshold,
            "outlier_threshold": outlier_threshold,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "num_leaves": num_leaves,
            "eval_fraction": eval_fraction,
            "split_seed": split_seed,
            "allow_image_backfill": allow_image_backfill,
        },
    )

    candidate_table_path = write_candidate_table_csv(
        scorer_candidate_table_rows(contexts),
        output_dir / TRAINING_CANDIDATE_TABLE_CSV,
    )
    v3_dir = output_dir / "v3_lambdarank"
    scorers_dir = output_dir / SCORERS_DIR
    scorers_dir.mkdir(parents=True, exist_ok=True)

    v3_metrics = _train_v3_lambdarank_from_tagged_rows(
        train_tagged_rows=train_tagged_rows,
        eval_tagged_rows=eval_tagged_rows,
        candidates=candidates,
        output_dir=v3_dir,
        failure_threshold=failure_threshold,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        learning_rate=learning_rate,
        iterations=iterations,
        num_leaves=num_leaves,
    )

    canonical_v3 = scorers_dir / "learned_quality_v3.json"
    shutil.copy2(v3_dir / SCORER_V3_ARTIFACT, canonical_v3)

    profile_specialist_status = "skipped_no_profile_rows"
    profile_specialist_entry: dict[str, T.Any] | None = None
    profile_train_rows = filter_profile_specialist_rows(train_tagged_rows)
    profile_eval_rows = filter_profile_specialist_rows(eval_tagged_rows)
    if profile_train_rows:
        profile_dir = output_dir / "v3_lambdarank_profile"
        try:
            profile_v3_metrics = _train_v3_lambdarank_from_tagged_rows(
                train_tagged_rows=profile_train_rows,
                eval_tagged_rows=profile_eval_rows,
                candidates=candidates,
                output_dir=profile_dir,
                failure_threshold=failure_threshold,
                eval_fraction=eval_fraction,
                split_seed=split_seed,
                learning_rate=learning_rate,
                iterations=iterations,
                num_leaves=num_leaves,
                policy=SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
                artifact_filename=SCORER_V3_PROFILE_ARTIFACT,
            )
            canonical_v3_profile = scorers_dir / "learned_quality_v3_profile.json"
            shutil.copy2(profile_dir / SCORER_V3_PROFILE_ARTIFACT, canonical_v3_profile)
            profile_specialist_entry = {
                **profile_v3_metrics,
                "canonical_artifact": str(canonical_v3_profile),
            }
            profile_specialist_status = "trained"
        except ValueError as err:
            profile_specialist_status = f"skipped_insufficient_rankable_profile_rows: {err}"

    metrics: dict[str, T.Any] = {
        "artifact_schema_version": 1,
        "scorer_dataset": dataset_manifest,
        "active_scorer_version": ACTIVE_SCORER_VERSION,
        "active_target": ACTIVE_SCORER_TARGET,
        "runtime_feature_contract_version": RUNTIME_FEATURE_CONTRACT_VERSION,
        "scorers": {
            "learned_quality_v3": {
                **v3_metrics,
                "canonical_artifact": str(canonical_v3),
            },
        },
        "candidate_table": str(candidate_table_path),
        "compatibility_artifacts": {
            "training_rows": [str(v3_dir / TRAINING_ROWS_CSV)],
            "eval_rows": [str(v3_dir / EVAL_ROWS_CSV)],
            "candidate_table": str(candidate_table_path),
            "note": (
                "The active scorer_suite target is transform_alignment_regret_v3 only. "
                "Consumers should use scorer_dataset/rows.csv, "
                "scorer_dataset/manifest.json, and the canonical learned_quality_v3 artifact."
            ),
        },
        "candidate_table_status": "compatibility_derived_from_scorer_contexts",
        "split_seed": split_seed,
        "eval_fraction": eval_fraction,
        "candidates": list(candidates),
    }
    if profile_specialist_entry is not None:
        metrics["scorers"]["learned_quality_v3_profile"] = profile_specialist_entry
    metrics["profile_specialist_status"] = profile_specialist_status
    metrics["profile_specialist_row_counts"] = {
        "train_rows": len(profile_train_rows),
        "eval_rows": len(profile_eval_rows),
    }
    metrics_path = write_json(scorers_dir / SCORER_SUITE_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    metrics["artifact"] = str(canonical_v3)
    return metrics


def train_runtime_resolver_scorer_v3(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    candidates: T.Sequence[str],
    output_dir: Path,
    gt_hard_resolver_metadata: Path | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    high_gap_threshold: float = DEFAULT_HIGH_GAP_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    eval_fraction: float = 0.20,
    split_seed: int = 42,
    allow_image_backfill: bool = False,
    progress: T.Callable[[T.Sequence[T.Any], str], T.Iterable[T.Any]] | None = None,
    learning_rate: float = 0.05,
    iterations: int = 150,
    num_leaves: int = 31,
) -> dict[str, T.Any]:
    """Train a direct learned_quality_v3 scorer artifact."""
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
        require_gt_hard_metadata=True,
        progress=progress,
    )
    tagged_rows = tagged_quality_rows(contexts, high_gap_threshold=high_gap_threshold)
    if not tagged_rows:
        raise ValueError("no scorer training rows were loaded")
    train_tagged_rows, eval_tagged_rows = split_tagged_rows(
        tagged_rows,
        eval_fraction=eval_fraction,
        seed=split_seed,
    )
    return _train_v3_lambdarank_from_tagged_rows(
        train_tagged_rows=train_tagged_rows,
        eval_tagged_rows=eval_tagged_rows,
        candidates=candidates,
        output_dir=output_dir,
        failure_threshold=failure_threshold,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        learning_rate=learning_rate,
        iterations=iterations,
        num_leaves=num_leaves,
    )


def train_runtime_resolver_scorer(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    candidates: T.Sequence[str],
    output_dir: Path,
    gt_hard_resolver_metadata: Path | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    high_gap_threshold: float = DEFAULT_HIGH_GAP_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    l2: float = 0.001,
    learning_rate: float = 0.1,
    iterations: int = 1500,
    num_leaves: int = 31,
    eval_fraction: float = 0.20,
    split_seed: int = 42,
    allow_image_backfill: bool = False,
    target: str = TARGET_TRANSFORM_REGRET_V3,
) -> dict[str, T.Any]:
    """Compatibility wrapper for the active v3 scorer-suite trainer."""
    del l2
    if target != TARGET_TRANSFORM_REGRET_V3:
        raise ValueError(
            "train_runtime_resolver_scorer trains only on "
            f"{TARGET_TRANSFORM_REGRET_V3!r}; unsupported target {target!r}."
        )
    return train_runtime_resolver_scorer_v3(
        gt_manifest=gt_manifest,
        gt_cache_dir=gt_cache_dir,
        production_manifest=production_manifest,
        production_cache_dir=production_cache_dir,
        weights_path=weights_path,
        candidates=candidates,
        output_dir=output_dir,
        gt_hard_resolver_metadata=gt_hard_resolver_metadata,
        failure_threshold=failure_threshold,
        high_gap_threshold=high_gap_threshold,
        outlier_threshold=outlier_threshold,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        allow_image_backfill=allow_image_backfill,
        learning_rate=learning_rate,
        iterations=iterations,
        num_leaves=num_leaves,
    )


__all__ = [
    "SCORER_ARTIFACT",
    "SCORER_V3_ARTIFACT",
    "SCORER_VERSION_LEARNED_QUALITY_V3",
    "ACTIVE_SCORER_VERSION",
    "ACTIVE_SCORER_TARGET",
    "SCORERS_DIR",
    "SCORER_SUITE_METRICS_JSON",
    "SCORER_SUITE_SENTINEL_JSON",
    "TRAINING_CANDIDATE_TABLE_CSV",
    "TRAINING_METRICS_JSON",
    "TRAINING_ROWS_CSV",
    "EVAL_ROWS_CSV",
    "feature_order",
    "grouped_rankable_rows_v3",
    "scorer_target_value",
    "split_tagged_rows",
    "target_distribution_stats",
    "train_runtime_resolver_scorer",
    "train_runtime_resolver_scorer_suite",
    "train_runtime_resolver_scorer_v3",
    "write_tagged_rows_csv",
]
