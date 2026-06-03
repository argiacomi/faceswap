#!/usr/bin/env python3
"""Train a profile specialist from partial-schema 39-point rows (#218/#219).

This is the separate partial-schema training path requested for
``learned_quality_v3_profile``: it consumes the 39-point profile rows produced by
:mod:`lib.landmarks.ensemble.profile39_dataset` (visible-side error + 39-point
transform cost/regret) and trains a LambdaRank ranker on ``profile39_transform
_regret``. It writes its own artifact and report so the 39-point objective is
never mixed into the canonical-68 scorer target.
"""

from __future__ import annotations

import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.profile39_dataset import Profile39Row, profile39_mix_report
from lib.landmarks.ensemble.runtime_features import (
    RUNTIME_FEATURE_CONTRACT_VERSION,
    runtime_feature_order,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import feature_matrix
from lib.landmarks.ensemble.scorer_target_config import (
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    SCORE_SEMANTICS_PREDICTED_COST,
    TARGET_PROFILE39_TRANSFORM_REGRET,
)
from lib.landmarks.pipeline_conventions import write_json

PROFILE39_SCORER_ARTIFACT = "runtime_resolver_scorer_v3_profile39.json"
PROFILE39_METRICS_JSON = "runtime_resolver_scorer_v3_profile39_metrics.json"
PROFILE39_TARGET = TARGET_PROFILE39_TRANSFORM_REGRET
PROFILE39_OBJECTIVE = "lambdarank_profile39_transform_regret"
SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE = "learned_quality_v3_profile"

#: Maximum profile-39 regret mapped into LambdaRank relevance labels.
PROFILE39_REGRET_LABEL_CLAMP = 1.0


def _profile39_label(regret: float) -> int:
    """Return relevance where lower 39-point regret ranks higher."""
    if PROFILE39_REGRET_LABEL_CLAMP <= 0.0:
        return 0
    clipped = min(max(float(regret), 0.0), PROFILE39_REGRET_LABEL_CLAMP)
    return round((1.0 - clipped / PROFILE39_REGRET_LABEL_CLAMP) * 30.0)


def _feature_order(rows: T.Sequence[Profile39Row]) -> tuple[str, ...]:
    ordered: tuple[str, ...] = runtime_feature_order(row.feature_values for row in rows)
    return ordered


def grouped_profile39_rows(
    rows: T.Sequence[Profile39Row],
) -> tuple[list[list[Profile39Row]], dict[str, T.Any]]:
    """Group rows per sample, keeping only groups with rankable regret variation."""
    groups: dict[str, list[Profile39Row]] = {}
    for row in rows:
        groups.setdefault(row.sample_id, []).append(row)
    valid_groups: list[list[Profile39Row]] = []
    single_candidate_groups = 0
    flat_regret_groups = 0
    for sample_id in sorted(groups):
        group = [row for row in groups[sample_id] if row.profile39_rankable]
        if len(group) < 2:
            single_candidate_groups += 1
            continue
        regrets = [row.profile39_transform_regret for row in group]
        if max(regrets) - min(regrets) <= 0.0:
            flat_regret_groups += 1
            continue
        valid_groups.append(sorted(group, key=lambda row: row.candidate_name))
    stats = {
        "total_group_count": len(groups),
        "rankable_group_count": len(valid_groups),
        "single_candidate_group_count": single_candidate_groups,
        "flat_regret_group_count": flat_regret_groups,
        "rankable_row_count": sum(len(group) for group in valid_groups),
    }
    return valid_groups, stats


def train_profile39_specialist(
    rows: T.Sequence[Profile39Row],
    *,
    output_dir: Path,
    candidates: T.Sequence[str],
    learning_rate: float = 0.05,
    iterations: int = 150,
    num_leaves: int = 31,
    split_seed: int = 42,
) -> dict[str, T.Any]:
    """Train the 39-point profile specialist and write its artifact + report.

    Returns metrics. Raises ``ValueError`` when there are no rankable 39-point
    groups so the caller can record a skip status instead of producing an empty
    artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, filter_stats = grouped_profile39_rows(rows)
    if not groups:
        raise ValueError("no rankable profile39 groups after filtering")

    try:
        import lightgbm as lgb
    except ModuleNotFoundError as err:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "profile39 specialist training requires lightgbm; install project requirements first"
        ) from err

    grouped_rows = [row for group in groups for row in group]
    group_sizes = [len(group) for group in groups]
    features = _feature_order(grouped_rows)
    x = feature_matrix([row.feature_values for row in grouped_rows], features)
    y = np.asarray([_profile39_label(row.profile39_transform_regret) for row in grouped_rows])

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=iterations,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=split_seed,
        deterministic=True,
        verbosity=-1,
    )
    ranker.fit(x, y.astype("int32"), group=group_sizes)
    booster = ranker.booster_
    importances = {
        feature: float(value)
        for feature, value in sorted(
            zip(features, booster.feature_importance(importance_type="gain"), strict=True),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
    }

    artifact = {
        "artifact_schema_version": 3,
        "version": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "scorer_version": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "runtime_policy": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "target": PROFILE39_TARGET,
        "objective": PROFILE39_OBJECTIVE,
        "training_mode": "grouped_lambdarank_profile39",
        "selection_target": "inverse_profile39_transform_regret_rank",
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "schema": "2d_39_partial",
        "features": list(features),
        "runtime_feature_contract_version": RUNTIME_FEATURE_CONTRACT_VERSION,
        "model_data": booster.model_to_string(),
        "feature_importances": importances,
        "calibration": {"type": "none", "params": {}},
        "profile39_filtering": filter_stats,
        "training_data_counts": {
            "row_count": len(grouped_rows),
            "sample_group_count": len(group_sizes),
            "candidate_count": len(candidates),
        },
    }
    artifact_path = write_json(output_dir / PROFILE39_SCORER_ARTIFACT, artifact)

    metrics: dict[str, T.Any] = {
        "artifact": str(artifact_path),
        "target": PROFILE39_TARGET,
        "objective": PROFILE39_OBJECTIVE,
        "schema": "2d_39_partial",
        "runtime_policy": SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
        "feature_count": len(features),
        "profile39_filtering": filter_stats,
        "profile39_report": profile39_mix_report(rows),
    }
    metrics_path = write_json(output_dir / PROFILE39_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


__all__ = [
    "PROFILE39_METRICS_JSON",
    "PROFILE39_OBJECTIVE",
    "PROFILE39_SCORER_ARTIFACT",
    "PROFILE39_TARGET",
    "grouped_profile39_rows",
    "train_profile39_specialist",
]
