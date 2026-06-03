#!/usr/bin/env python3
"""Mixed canonical-68/profile39 LambdaRank training for the profile specialist.

This module intentionally does not wire itself into the scorer suite. It provides
source-neutral adapters and a trainer so ``learned_quality_v3_profile`` can be
trained from canonical 68-point profile rows and partial-schema profile39 rows in
one aggregate LambdaRank model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import typing as T
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.profile39_dataset import Profile39Row, profile39_mix_report
from lib.landmarks.ensemble.profile_routing import is_profile_or_occlusion_context
from lib.landmarks.ensemble.runtime_features import (
    RUNTIME_FEATURE_CONTRACT_VERSION,
    runtime_feature_order,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import feature_matrix
from lib.landmarks.ensemble.runtime_resolver_scorer_data import CandidateQualityRow
from lib.landmarks.ensemble.scorer_target_config import (
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    SCORE_SEMANTICS_PREDICTED_COST,
    TARGET_PROFILE_MIXED_TRANSFORM_REGRET,
)
from lib.landmarks.pipeline_conventions import write_json

MIXED_PROFILE_SCORER_ARTIFACT = "runtime_resolver_scorer_v3_profile_mixed.json"
MIXED_PROFILE_METRICS_JSON = "runtime_resolver_scorer_v3_profile_mixed_metrics.json"
MIXED_PROFILE_TRAINING_ROWS_CSV = "runtime_resolver_scorer_v3_profile_mixed_training_rows.csv"
MIXED_PROFILE_EVAL_ROWS_CSV = "runtime_resolver_scorer_v3_profile_mixed_eval_rows.csv"
MIXED_PROFILE_FEATURE_IMPORTANCES_CSV = (
    "runtime_resolver_scorer_v3_profile_mixed_feature_importances.csv"
)

SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE = "learned_quality_v3_profile"
# Re-exported from scorer_target_config so runtime validation and this trainer
# share a single source of truth without a circular import.
MIXED_PROFILE_OBJECTIVE = "lambdarank_mixed_profile_transform_regret"
MIXED_PROFILE_TRAINING_MODE = "grouped_lambdarank_mixed_profile"
CANONICAL68_PROFILE_SOURCE = "canonical68_profile"
PROFILE39_SOURCE = "profile39"
MIXED_PROFILE_REGRET_LABEL_CLAMP = 1.0
MIN_MIXED_PROFILE_REGRET_GAP = 1e-4


@dataclass(frozen=True)
class MixedProfileRankRow:
    """One candidate row in the mixed profile ranker schema."""

    sample_id: str
    dataset: str
    condition: str
    source: str
    candidate_name: str
    feature_values: dict[str, float]
    regret: float
    rankable: bool
    hard_invalid: bool = False
    weight: float = 1.0


@dataclass(frozen=True)
class MixedProfileRankGroup:
    """One LambdaRank query group for a single profile sample."""

    sample_id: str
    dataset: str
    condition: str
    source: str
    rows: tuple[MixedProfileRankRow, ...]
    query_weight: float


CanonicalInputRow = CandidateQualityRow | tuple[CandidateQualityRow, str]


def _canonical_row_and_source(
    item: CanonicalInputRow,
    *,
    fallback_source: str = CANONICAL68_PROFILE_SOURCE,
) -> tuple[CandidateQualityRow, str]:
    """Return a canonical row plus source label from plain or tagged input."""

    if isinstance(item, tuple) and len(item) == 2:
        row, source = item
        return row, str(source or fallback_source)
    return item, fallback_source


def _group_has_regret_gap(rows: T.Sequence[MixedProfileRankRow]) -> bool:
    regrets = [float(row.regret) for row in rows if math.isfinite(float(row.regret))]
    if len(regrets) < 2:
        return False
    return max(regrets) - min(regrets) > MIN_MIXED_PROFILE_REGRET_GAP


def _sorted_valid_group(rows: T.Sequence[MixedProfileRankRow]) -> tuple[MixedProfileRankRow, ...] | None:
    valid = [
        row
        for row in rows
        if bool(row.rankable)
        and not bool(row.hard_invalid)
        and math.isfinite(float(row.regret))
    ]
    if len(valid) < 2 or not _group_has_regret_gap(valid):
        return None
    return tuple(sorted(valid, key=lambda row: row.candidate_name))


def canonical68_profile_groups_from_rows(
    rows: T.Sequence[CanonicalInputRow],
    *,
    source_label: str = CANONICAL68_PROFILE_SOURCE,
    query_weight: float = 1.0,
) -> list[MixedProfileRankGroup]:
    """Convert canonical 68-point profile/occlusion rows into mixed rank groups.

    The input may be plain :class:`CandidateQualityRow` values or tagged
    ``(CandidateQualityRow, source)`` tuples. Source is retained only for grouping,
    accounting, and reporting. It is never added to the model feature map.
    """

    groups: dict[tuple[str, str, str, str], list[MixedProfileRankRow]] = {}
    for item in rows:
        row, source = _canonical_row_and_source(item, fallback_source=source_label)
        if not is_profile_or_occlusion_context(row):
            continue
        del source
        mixed = MixedProfileRankRow(
            sample_id=str(row.sample_id),
            dataset=str(row.dataset),
            condition=str(row.condition),
            source=str(source_label),
            candidate_name=str(row.candidate_name),
            feature_values=dict(row.feature_values),
            regret=float(row.transform_regret_v3),
            rankable=bool(row.rankable_v3),
            hard_invalid=bool(row.hard_invalid_v3),
            weight=1.0,
        )
        key = (mixed.source, mixed.dataset, mixed.condition, mixed.sample_id)
        groups.setdefault(key, []).append(mixed)

    output: list[MixedProfileRankGroup] = []
    for source, dataset, condition, sample_id in sorted(groups):
        grouped = _sorted_valid_group(groups[(source, dataset, condition, sample_id)])
        if grouped is None:
            continue
        output.append(
            MixedProfileRankGroup(
                sample_id=sample_id,
                dataset=dataset,
                condition=condition,
                source=source,
                rows=grouped,
                query_weight=float(query_weight),
            )
        )
    return output


def profile39_groups_from_rows(
    rows: T.Sequence[Profile39Row],
    *,
    query_weight: float = 1.0,
) -> list[MixedProfileRankGroup]:
    """Convert partial-schema profile39 rows into mixed rank groups."""

    groups: dict[tuple[str, str, str, str], list[MixedProfileRankRow]] = {}
    for row in rows:
        condition = f"profile_{row.side}" if row.side else "profile"
        mixed = MixedProfileRankRow(
            sample_id=str(row.sample_id),
            dataset=str(row.dataset),
            condition=condition,
            source=PROFILE39_SOURCE,
            candidate_name=str(row.candidate_name),
            feature_values=dict(row.feature_values),
            regret=float(row.profile39_transform_regret),
            rankable=bool(row.profile39_rankable),
            hard_invalid=False,
            weight=1.0,
        )
        key = (mixed.source, mixed.dataset, mixed.condition, mixed.sample_id)
        groups.setdefault(key, []).append(mixed)

    output: list[MixedProfileRankGroup] = []
    for source, dataset, condition, sample_id in sorted(groups):
        grouped = _sorted_valid_group(groups[(source, dataset, condition, sample_id)])
        if grouped is None:
            continue
        output.append(
            MixedProfileRankGroup(
                sample_id=sample_id,
                dataset=dataset,
                condition=condition,
                source=source,
                rows=grouped,
                query_weight=float(query_weight),
            )
        )
    return output


def mixed_profile_lambdarank_label(
    regret: float,
    *,
    clamp: float = MIXED_PROFILE_REGRET_LABEL_CLAMP,
) -> int:
    """Return LambdaRank relevance where lower mixed-profile regret ranks higher."""

    if clamp <= 0.0:
        return 0
    clipped = min(max(float(regret), 0.0), float(clamp))
    return round((1.0 - clipped / float(clamp)) * 30.0)


def split_profile39_groups(
    groups: T.Sequence[MixedProfileRankGroup],
    *,
    eval_fraction: float,
    seed: str,
) -> tuple[list[MixedProfileRankGroup], list[MixedProfileRankGroup]]:
    """Deterministically split profile39 groups by sample within source/dataset/condition."""

    if eval_fraction < 0.0 or eval_fraction >= 1.0:
        raise ValueError("profile39 eval_fraction must be >= 0.0 and < 1.0")

    by_stratum: dict[tuple[str, str, str], list[MixedProfileRankGroup]] = defaultdict(list)
    for group in groups:
        by_stratum[(group.source, group.dataset, group.condition)].append(group)

    eval_keys: set[tuple[str, str, str, str]] = set()
    for (source, dataset, condition), stratum_groups in by_stratum.items():
        ordered = sorted(
            stratum_groups,
            key=lambda group: hashlib.sha256(
                "|".join((str(seed), source, dataset, condition, group.sample_id)).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        if eval_fraction <= 0.0 or len(ordered) <= 1:
            continue
        eval_count = max(1, int(round(len(ordered) * eval_fraction)))
        eval_count = min(eval_count, len(ordered) - 1)
        for group in ordered[:eval_count]:
            eval_keys.add((group.source, group.dataset, group.condition, group.sample_id))

    train_groups: list[MixedProfileRankGroup] = []
    eval_groups: list[MixedProfileRankGroup] = []
    for group in groups:
        key = (group.source, group.dataset, group.condition, group.sample_id)
        if key in eval_keys:
            eval_groups.append(group)
        else:
            train_groups.append(group)
    return train_groups, eval_groups


def _flatten_groups(groups: T.Sequence[MixedProfileRankGroup]) -> list[MixedProfileRankRow]:
    return [row for group in groups for row in group.rows]


def _item_weights(groups: T.Sequence[MixedProfileRankGroup]) -> np.ndarray:
    weights: list[float] = []
    for group in groups:
        weights.extend([float(group.query_weight)] * len(group.rows))
    return T.cast(np.ndarray, np.asarray(weights, dtype="float64"))


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


def _write_mixed_rows_csv(
    groups: T.Sequence[MixedProfileRankGroup],
    path: Path,
    *,
    split: str,
    features: T.Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "source",
        "dataset",
        "condition",
        "sample_id",
        "candidate_name",
        "regret",
        "label",
        "rankable",
        "hard_invalid",
        "row_weight",
        "query_weight",
        "features_json",
        *features,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            for row in group.rows:
                writer.writerow(
                    {
                        "split": split,
                        "source": row.source,
                        "dataset": row.dataset,
                        "condition": row.condition,
                        "sample_id": row.sample_id,
                        "candidate_name": row.candidate_name,
                        "regret": row.regret,
                        "label": mixed_profile_lambdarank_label(row.regret),
                        "rankable": row.rankable,
                        "hard_invalid": row.hard_invalid,
                        "row_weight": row.weight,
                        "query_weight": group.query_weight,
                        "features_json": json.dumps(dict(sorted(row.feature_values.items())), sort_keys=True),
                        **{feature: row.feature_values.get(feature, 0.0) for feature in features},
                    }
                )
    return path


def _group_filter_stats(
    *,
    raw_group_count: int,
    groups: T.Sequence[MixedProfileRankGroup],
) -> dict[str, T.Any]:
    return {
        "total_group_count": raw_group_count,
        "rankable_group_count": len(groups),
        "rankable_row_count": sum(len(group.rows) for group in groups),
        "min_regret_gap": MIN_MIXED_PROFILE_REGRET_GAP,
    }


def _raw_profile39_group_count(rows: T.Sequence[Profile39Row]) -> int:
    return len({(row.dataset, row.side, row.sample_id) for row in rows})


def _raw_canonical_group_count(rows: T.Sequence[CanonicalInputRow]) -> int:
    keys: set[tuple[str, str, str, str]] = set()
    for item in rows:
        row, _source = _canonical_row_and_source(item)
        if is_profile_or_occlusion_context(row):
            keys.add((CANONICAL68_PROFILE_SOURCE, row.dataset, row.condition, row.sample_id))
    return len(keys)


def _source_mix_counts(groups: T.Sequence[MixedProfileRankGroup]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for group in groups:
        bucket = counts.setdefault(group.source, {"groups": 0, "rows": 0})
        bucket["groups"] += 1
        bucket["rows"] += len(group.rows)
    return dict(sorted(counts.items()))


def _condition_counts(groups: T.Sequence[MixedProfileRankGroup]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for group in groups:
        bucket = counts.setdefault(group.condition, {"groups": 0, "rows": 0})
        bucket["groups"] += 1
        bucket["rows"] += len(group.rows)
    return dict(sorted(counts.items()))


def _selection_metrics(
    groups: T.Sequence[MixedProfileRankGroup],
    *,
    features: T.Sequence[str],
    booster: T.Any | None = None,
) -> dict[str, T.Any]:
    """Return simple source-specific top-1 metrics for train/eval groups."""

    if not groups:
        return {
            "group_count": 0,
            "row_count": 0,
            "source_mix": {},
            "by_source": {},
            "mean_selected_regret": 0.0,
            "mean_oracle_regret": 0.0,
            "mean_regret_delta_vs_oracle": 0.0,
            "top1_oracle_hit_rate": 0.0,
            "candidate_selection_distribution": {},
        }

    by_source: dict[str, list[dict[str, T.Any]]] = defaultdict(list)
    all_records: list[dict[str, T.Any]] = []
    selection_counts: Counter[str] = Counter()
    for group in groups:
        rows = list(group.rows)
        oracle = min(rows, key=lambda row: (row.regret, row.candidate_name))
        if booster is None:
            selected = oracle
        else:
            x = feature_matrix([row.feature_values for row in rows], features)
            relevance = booster.predict(x)
            selected_idx = int(np.argmax(relevance))
            selected = rows[selected_idx]
        record = {
            "source": group.source,
            "selected_candidate": selected.candidate_name,
            "oracle_candidate": oracle.candidate_name,
            "selected_regret": float(selected.regret),
            "oracle_regret": float(oracle.regret),
            "delta_vs_oracle": float(selected.regret - oracle.regret),
            "oracle_hit": selected.candidate_name == oracle.candidate_name,
        }
        by_source[group.source].append(record)
        all_records.append(record)
        selection_counts[selected.candidate_name] += 1

    def summarize(records: T.Sequence[dict[str, T.Any]]) -> dict[str, T.Any]:
        selected_regrets = np.asarray([row["selected_regret"] for row in records], dtype="float64")
        oracle_regrets = np.asarray([row["oracle_regret"] for row in records], dtype="float64")
        deltas = np.asarray([row["delta_vs_oracle"] for row in records], dtype="float64")
        hits = np.asarray([float(row["oracle_hit"]) for row in records], dtype="float64")
        return {
            "group_count": len(records),
            "mean_selected_regret": float(np.mean(selected_regrets)) if records else 0.0,
            "mean_oracle_regret": float(np.mean(oracle_regrets)) if records else 0.0,
            "mean_regret_delta_vs_oracle": float(np.mean(deltas)) if records else 0.0,
            "top1_oracle_hit_rate": float(np.mean(hits)) if records else 0.0,
        }

    return {
        **summarize(all_records),
        "row_count": sum(len(group.rows) for group in groups),
        "source_mix": _source_mix_counts(groups),
        "by_source": {source: summarize(records) for source, records in sorted(by_source.items())},
        "candidate_selection_distribution": dict(sorted(selection_counts.items())),
    }


def train_mixed_profile_specialist(
    *,
    canonical_train_rows: T.Sequence[CanonicalInputRow],
    canonical_eval_rows: T.Sequence[CanonicalInputRow],
    profile39_rows: T.Sequence[Profile39Row],
    output_dir: Path,
    candidates: T.Sequence[str],
    feature_order: T.Sequence[str] | None = None,
    eval_fraction: float = 0.2,
    split_seed: str = "mixed_profile_v1",
    profile39_query_weight: float = 1.0,
    canonical_query_weight_multiplier: float = 1.0,
    learning_rate: float = 0.03,
    iterations: int = 600,
    num_leaves: int = 31,
) -> dict[str, T.Any]:
    """Train one profile specialist from canonical 68-point and profile39 groups.

    The resulting artifact is intended for ``learned_quality_v3_profile`` only.
    It keeps canonical and profile39 supervision source-local at the query-group
    level, while sharing the runtime-visible feature space.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_train_groups = canonical68_profile_groups_from_rows(
        canonical_train_rows,
        query_weight=canonical_query_weight_multiplier,
    )
    canonical_eval_groups = canonical68_profile_groups_from_rows(
        canonical_eval_rows,
        query_weight=canonical_query_weight_multiplier,
    )
    profile39_groups = profile39_groups_from_rows(
        profile39_rows,
        query_weight=profile39_query_weight,
    )
    profile39_train_groups, profile39_eval_groups = split_profile39_groups(
        profile39_groups,
        eval_fraction=eval_fraction,
        seed=split_seed,
    )

    train_groups = [*canonical_train_groups, *profile39_train_groups]
    eval_groups = [*canonical_eval_groups, *profile39_eval_groups]
    if not train_groups:
        raise ValueError("mixed profile specialist has no rankable training groups")

    try:
        import lightgbm as lgb
    except ModuleNotFoundError as err:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "mixed profile specialist training requires lightgbm; install project requirements first"
        ) from err

    grouped_train = _flatten_groups(train_groups)
    train_group_sizes = [len(group.rows) for group in train_groups]
    features = tuple(feature_order or runtime_feature_order(row.feature_values for row in grouped_train))
    x = feature_matrix([row.feature_values for row in grouped_train], features)
    y = np.asarray(
        [mixed_profile_lambdarank_label(row.regret) for row in grouped_train],
        dtype="int32",
    )
    item_weights = _item_weights(train_groups)

    random_state = int(hashlib.sha256(str(split_seed).encode("utf-8")).hexdigest()[:8], 16)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=iterations,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        deterministic=True,
        verbosity=-1,
    )
    ranker.fit(x, y, group=train_group_sizes, sample_weight=item_weights)
    booster = ranker.booster_
    importances = _feature_importance_map(
        features,
        booster.feature_importance(importance_type="gain"),
    )

    source_mix = {
        "train": _source_mix_counts(train_groups),
        "eval": _source_mix_counts(eval_groups),
    }
    filtering = {
        "canonical68_profile": {
            "train": _group_filter_stats(
                raw_group_count=_raw_canonical_group_count(canonical_train_rows),
                groups=canonical_train_groups,
            ),
            "eval": _group_filter_stats(
                raw_group_count=_raw_canonical_group_count(canonical_eval_rows),
                groups=canonical_eval_groups,
            ),
        },
        "profile39": {
            "all": _group_filter_stats(
                raw_group_count=_raw_profile39_group_count(profile39_rows),
                groups=profile39_groups,
            ),
            "train_group_count": len(profile39_train_groups),
            "eval_group_count": len(profile39_eval_groups),
        },
    }

    artifact = {
        "artifact_schema_version": 3,
        "version": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "scorer_version": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "runtime_policy": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "target": TARGET_PROFILE_MIXED_TRANSFORM_REGRET,
        "objective": MIXED_PROFILE_OBJECTIVE,
        "training_mode": MIXED_PROFILE_TRAINING_MODE,
        "selection_target": "inverse_mixed_profile_transform_regret_rank",
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "schema": "mixed_canonical68_profile_and_profile39",
        "features": list(features),
        "runtime_feature_contract_version": RUNTIME_FEATURE_CONTRACT_VERSION,
        "model_data": booster.model_to_string(),
        "feature_importances": importances,
        "calibration": {"type": "none", "params": {}},
        "source_mix": source_mix,
        "mixed_profile_filtering": filtering,
        "training_data_counts": {
            "row_count": len(grouped_train),
            "sample_group_count": len(train_group_sizes),
            "eval_row_count": sum(len(group.rows) for group in eval_groups),
            "eval_sample_group_count": len(eval_groups),
            "candidate_count": len(candidates),
        },
        "split_ids": {
            "profile39_seed": split_seed,
            "profile39_eval_fraction": eval_fraction,
            "profile39_train_group_count": len(profile39_train_groups),
            "profile39_eval_group_count": len(profile39_eval_groups),
        },
        "sample_weighting": {
            "strategy": "mixed_profile_source_query_weighting",
            "canonical_query_weight_multiplier": canonical_query_weight_multiplier,
            "profile39_query_weight": profile39_query_weight,
            "row_count": len(grouped_train),
            "query_count": len(train_group_sizes),
            "mean_weight": float(np.mean(item_weights)) if item_weights.size else 0.0,
            "max_weight": float(np.max(item_weights)) if item_weights.size else 0.0,
        },
        "lightgbm_params": {
            "objective": "lambdarank",
            "n_estimators": iterations,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "random_state_source": split_seed,
            "deterministic": True,
        },
    }
    artifact_path = write_json(output_dir / MIXED_PROFILE_SCORER_ARTIFACT, artifact)
    training_rows_path = _write_mixed_rows_csv(
        train_groups,
        output_dir / MIXED_PROFILE_TRAINING_ROWS_CSV,
        split="train",
        features=features,
    )
    eval_rows_path = _write_mixed_rows_csv(
        eval_groups,
        output_dir / MIXED_PROFILE_EVAL_ROWS_CSV,
        split="eval",
        features=features,
    )
    importances_path = _write_feature_importance_csv(
        output_dir / MIXED_PROFILE_FEATURE_IMPORTANCES_CSV,
        importances,
    )

    metrics: dict[str, T.Any] = {
        "artifact": str(artifact_path),
        "training_rows": str(training_rows_path),
        "eval_rows": str(eval_rows_path),
        "feature_importances": str(importances_path),
        "candidate_count": len(candidates),
        "candidates": list(candidates),
        "feature_count": len(features),
        "runtime_feature_contract_version": RUNTIME_FEATURE_CONTRACT_VERSION,
        "target": TARGET_PROFILE_MIXED_TRANSFORM_REGRET,
        "objective": MIXED_PROFILE_OBJECTIVE,
        "training_mode": MIXED_PROFILE_TRAINING_MODE,
        "runtime_policy": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "source_mix": source_mix,
        "condition_mix": {
            "train": _condition_counts(train_groups),
            "eval": _condition_counts(eval_groups),
        },
        "mixed_profile_filtering": filtering,
        "profile39_report": profile39_mix_report(profile39_rows),
        "split_seed": split_seed,
        "eval_fraction": eval_fraction,
        "training_data_counts": artifact["training_data_counts"],
        "split_ids": artifact["split_ids"],
        "sample_weighting": artifact["sample_weighting"],
        "lightgbm_params": artifact["lightgbm_params"],
        "selection_metrics": {
            "train": _selection_metrics(train_groups, features=features, booster=booster),
            "eval": _selection_metrics(eval_groups, features=features, booster=booster),
        },
    }
    metrics_path = write_json(output_dir / MIXED_PROFILE_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


__all__ = [
    "CANONICAL68_PROFILE_SOURCE",
    "MIN_MIXED_PROFILE_REGRET_GAP",
    "MIXED_PROFILE_FEATURE_IMPORTANCES_CSV",
    "MIXED_PROFILE_METRICS_JSON",
    "MIXED_PROFILE_OBJECTIVE",
    "MIXED_PROFILE_REGRET_LABEL_CLAMP",
    "MIXED_PROFILE_SCORER_ARTIFACT",
    "MIXED_PROFILE_TRAINING_MODE",
    "MIXED_PROFILE_TRAINING_ROWS_CSV",
    "MIXED_PROFILE_EVAL_ROWS_CSV",
    "MixedProfileRankGroup",
    "MixedProfileRankRow",
    "PROFILE39_SOURCE",
    "SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE",
    "TARGET_PROFILE_MIXED_TRANSFORM_REGRET",
    "canonical68_profile_groups_from_rows",
    "mixed_profile_lambdarank_label",
    "profile39_groups_from_rows",
    "split_profile39_groups",
    "train_mixed_profile_specialist",
]
