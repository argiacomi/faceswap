#!/usr/bin/env python3
"""Reusable runtime resolver scorer training implementation."""

from __future__ import annotations

import csv
import hashlib
import shutil
import typing as T
from collections import defaultdict
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverScorer,
    feature_matrix,
    sigmoid,
    write_runtime_resolver_scorer,
)
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
    DEFAULT_COLLAPSE_COST_PENALTY,
    DEFAULT_FAILURE_COST_PENALTY,
    DEFAULT_LARGE_COST_THRESHOLD,
    DEFAULT_REGRET_NORMALIZER,
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    MODEL_TYPE_LINEAR_REGRESSION,
    MODEL_TYPE_LOGISTIC_REGRESSION,
    REGRESSION_TARGETS,
    SCORE_SEMANTICS_PREDICTED_COST,
    SCORE_SEMANTICS_PREDICTED_RISK,
    SCORER_TARGETS,
    TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
    TARGET_NORMALIZED_REGRET,
    TARGET_ORACLE_REGRET,
    TARGET_SELECTION_COST,
)
from lib.landmarks.ensemble.scorer_targets import (
    TaggedRow,
    scorer_candidate_table_rows,
    source_quality_rows,
    tagged_quality_rows,
    untag_quality_rows,
)
from lib.landmarks.pipeline_conventions import (
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
    write_json,
)

SCORER_ARTIFACT = "runtime_resolver_scorer.json"
SCORER_V2_ARTIFACT = "runtime_resolver_scorer_v2.json"
SCORERS_DIR = "scorers"
SCORER_SUITE_METRICS_JSON = "metrics.json"
SCORER_SUITE_SENTINEL_JSON = ".scorer_training_complete.json"
TRAINING_ROWS_CSV = "runtime_resolver_scorer_training_rows.csv"
EVAL_ROWS_CSV = "runtime_resolver_scorer_eval_rows.csv"
TRAINING_CANDIDATE_TABLE_CSV = "candidate_table.csv"
TRAINING_METRICS_JSON = "runtime_resolver_scorer_training_metrics.json"
TRAINING_V2_METRICS_JSON = "runtime_resolver_scorer_v2_training_metrics.json"
SCORER_CONDITION_REPORT_CSV = "scorer_report_by_condition.csv"
SCORER_REGION_REPORT_CSV = "scorer_report_by_region.csv"

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
    feature_names = sorted({name for row, _source in rows for name in row.feature_values})
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
    """Return stable feature order for scorer training."""
    names: set[str] = set()
    for row in rows:
        names.update(row.feature_values)
    preferred = [
        "candidate_is_single_model",
        "candidate_is_fusion",
        "cloud_area_ratio",
        "hull_area_ratio",
        "points_outside_expanded_bbox_fraction",
        "eye_mouth_order_valid_after_deroll",
        "roi_center_consensus_distance",
        "landmark_consensus_distance",
        "roll_degrees",
        "yaw_degrees",
        "roll_delta_to_consensus",
        "yaw_delta_to_consensus",
        "candidate_yaw_disagreement",
        "max_disagreement_px",
        "has_geometry_veto",
    ]
    ordered = [name for name in preferred if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return tuple(ordered)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return float(np.log(clipped / (1.0 - clipped)))


def fit_logistic(
    x_raw: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    learning_rate: float,
    iterations: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a weighted logistic-regression scorer using numpy only."""
    if x_raw.shape[0] == 0:
        raise ValueError("cannot train scorer on an empty matrix")
    weights = (
        np.ones(x_raw.shape[0], dtype="float64")
        if sample_weight is None
        else np.asarray(sample_weight, dtype="float64")
    )
    weights = np.where(weights > 0.0, weights, 1.0)
    weight_sum = float(np.sum(weights))
    positive_rate = float(np.sum(weights * y) / max(weight_sum, 1e-9))
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return np.zeros(x_raw.shape[1], dtype="float64"), _logit(positive_rate)

    mean = np.average(x_raw, axis=0, weights=weights)
    centered = x_raw - mean
    std = np.sqrt(np.average(np.square(centered), axis=0, weights=weights))
    std = np.where(std < 1e-8, 1.0, std)
    x = centered / std
    coef = np.zeros(x.shape[1], dtype="float64")
    intercept = _logit(positive_rate)
    for _ in range(iterations):
        linear = x @ coef + intercept
        pred = np.asarray([sigmoid(float(item)) for item in linear], dtype="float64")
        error = (pred - y) * weights
        grad_coef = (x.T @ error) / weight_sum + (l2 * coef)
        grad_intercept = float(np.sum(error) / weight_sum)
        coef -= learning_rate * grad_coef
        intercept -= learning_rate * grad_intercept
    raw_coef = coef / std
    raw_intercept = float(intercept - np.sum((coef * mean) / std))
    return raw_coef, raw_intercept


def fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a weighted ridge linear regressor using numpy only."""
    if x.shape[0] == 0:
        raise ValueError("cannot train scorer on an empty matrix")
    weights = (
        np.ones(x.shape[0], dtype="float64")
        if sample_weight is None
        else np.asarray(sample_weight, dtype="float64")
    )
    weights = np.where(weights > 0.0, weights, 1.0)
    design = np.column_stack([np.ones(x.shape[0], dtype="float64"), x])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    penalty = np.eye(design.shape[1], dtype="float64") * l2
    penalty[0, 0] = 0.0
    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_y
    try:
        params = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return params[1:].astype("float64"), float(params[0])


def scorer_target_value(row: CandidateQualityRow, target: str) -> float:
    """Return the configured scorer target value for one training row."""
    if target == TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP:
        return float(row.candidate_failure_or_high_gap)
    if target == TARGET_NORMALIZED_REGRET:
        return float(row.normalized_regret)
    if target == TARGET_ORACLE_REGRET:
        return max(float(row.candidate_nme - row.oracle_nme), 0.0)
    if target == TARGET_SELECTION_COST:
        return float(row.selection_cost)
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


def scorer_sample_weight(row: CandidateQualityRow, source: str = "") -> float:
    """Return sample weight for one candidate row.

    Normal: 1x
    Profile: 2x
    Occlusion: 2x
    Profile + occlusion: 4x
    Production failure: 5x minimum
    """
    label = _hard_case_split_label(row, source)
    weight = HARD_CASE_SAMPLE_WEIGHTS[label]
    if source == SOURCE_PRODUCTION_VALIDATED and bool(row.failure_label):
        weight = max(weight, HARD_CASE_SAMPLE_WEIGHTS["production_failure"])
    weight += _crop_breaking_weight_bonus(row)
    if row.candidate_failure_or_high_gap:
        weight += 1.0
    return float(weight)


def scorer_sample_weighting_stats(tagged_rows: T.Sequence[TaggedRow]) -> dict[str, T.Any]:
    weights = np.asarray(
        [scorer_sample_weight(row, source) for row, source in tagged_rows],
        dtype="float64",
    )
    by_split: dict[str, dict[str, T.Any]] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, source in tagged_rows:
        grouped[_hard_case_split_label(row, source)].append(scorer_sample_weight(row, source))
    for split in ("normal", "profile", "occlusion", "profile_occlusion", "production_failure"):
        values = grouped.get(split, [])
        by_split[split] = {
            "row_count": len(values),
            "mean_weight": float(np.mean(values)) if values else 0.0,
            "max_weight": float(np.max(values)) if values else 0.0,
        }
    return {
        "strategy": "hard_case_weighting_single_scorer",
        "weights": HARD_CASE_SAMPLE_WEIGHTS,
        "row_count": int(weights.size),
        "mean_weight": float(np.mean(weights)) if weights.size else 0.0,
        "max_weight": float(np.max(weights)) if weights.size else 0.0,
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


def scorer_row_metrics(
    scorer: RuntimeResolverScorer,
    rows: T.Sequence[CandidateQualityRow],
) -> dict[str, T.Any]:
    """Return standard scorer metrics for scorer rows."""
    if not rows:
        metrics: dict[str, T.Any] = {
            "row_count": 0,
            "target": scorer.target,
            "model_type": scorer.model_type,
        }
        if scorer.model_type == MODEL_TYPE_LOGISTIC_REGRESSION:
            metrics.update(
                {
                    "positive_count": 0,
                    "positive_rate": 0.0,
                    "accuracy_at_0_5": 0.0,
                    "log_loss": 0.0,
                }
            )
        else:
            metrics.update({"target_mean": 0.0, "mae": 0.0, "mse": 0.0, "rmse": 0.0})
        return metrics
    labels = np.asarray(
        [scorer_target_value(row, scorer.target) for row in rows],
        dtype="float64",
    )
    scores = np.asarray(
        [scorer.score_feature_map(row.feature_values) for row in rows],
        dtype="float64",
    )
    if scorer.model_type == MODEL_TYPE_LINEAR_REGRESSION:
        errors = scores - labels
        mse = float(np.mean(np.square(errors))) if labels.size else 0.0
        return {
            "row_count": len(rows),
            "target": scorer.target,
            "model_type": scorer.model_type,
            "target_mean": float(labels.mean()) if labels.size else 0.0,
            "mae": float(np.mean(np.abs(errors))) if labels.size else 0.0,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
    predicted = scores >= 0.5
    accuracy = float(np.mean(predicted == labels)) if labels.size else 0.0
    loss = -np.mean(
        labels * np.log(np.clip(scores, 1e-8, 1.0))
        + (1.0 - labels) * np.log(np.clip(1.0 - scores, 1e-8, 1.0))
    )
    return {
        "row_count": len(rows),
        "target": scorer.target,
        "model_type": scorer.model_type,
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "accuracy_at_0_5": accuracy,
        "log_loss": float(loss) if labels.size else 0.0,
    }


def _sample_group_key(row: CandidateQualityRow, source: str) -> tuple[str, str, str, str]:
    return source, row.dataset, row.condition, row.sample_id


def _lambdarank_label(row: CandidateQualityRow) -> int:
    """Return an integer relevance label where higher means better candidate quality."""
    if row.is_oracle:
        return 30
    clipped_cost = min(max(float(row.selection_cost), 0.0), 3.0)
    return max(0, min(29, int(round((1.0 - clipped_cost / 3.0) * 29.0))))


def _lambdarank_weight(row: CandidateQualityRow, source: str) -> float:
    """Return v2 item weight using the same hard-case policy as v1 scorers."""
    return scorer_sample_weight(row, source)


def _grouped_rows(rows: T.Sequence[TaggedRow]) -> tuple[list[TaggedRow], list[int]]:
    groups: dict[tuple[str, str, str, str], list[TaggedRow]] = {}
    for tagged in rows:
        row, source = tagged
        groups.setdefault(_sample_group_key(row, source), []).append(tagged)
    ordered: list[TaggedRow] = []
    group_sizes: list[int] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda tagged: tagged[0].candidate_name)
        ordered.extend(group)
        group_sizes.append(len(group))
    return ordered, group_sizes


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


def _train_linear_or_logistic_from_tagged_rows(
    *,
    tagged_rows: T.Sequence[TaggedRow],
    train_tagged_rows: T.Sequence[TaggedRow],
    eval_tagged_rows: T.Sequence[TaggedRow],
    candidates: T.Sequence[str],
    output_dir: Path,
    target: str,
    failure_threshold: float,
    high_gap_threshold: float,
    l2: float,
    learning_rate: float,
    iterations: int,
    eval_fraction: float,
    split_seed: int,
    allow_image_backfill: bool,
    gt_hard_resolver_metadata: Path | None,
    candidate_table_path: Path | None = None,
) -> dict[str, T.Any]:
    """Train a v1/v1.1 scorer from prebuilt scorer rows."""

    if target not in SCORER_TARGETS:
        raise ValueError(f"target must be one of {SCORER_TARGETS}, got {target!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = untag_quality_rows(tagged_rows)
    train_rows = untag_quality_rows(train_tagged_rows)
    eval_rows = untag_quality_rows(eval_tagged_rows)
    features = feature_order(train_rows)
    x = feature_matrix([row.feature_values for row in train_rows], features)
    y = np.asarray(
        [scorer_target_value(row, target) for row in train_rows],
        dtype="float64",
    )
    train_sample_weights = np.asarray(
        [scorer_sample_weight(row, source) for row, source in train_tagged_rows],
        dtype="float64",
    )
    if target in REGRESSION_TARGETS:
        coefficients, intercept = fit_linear_regression(
            x,
            y,
            l2=l2,
            sample_weight=train_sample_weights,
        )
        model_type = MODEL_TYPE_LINEAR_REGRESSION
        score_semantics = SCORE_SEMANTICS_PREDICTED_COST
        version = "continuous_regret_v1_1"
        if target == TARGET_ORACLE_REGRET:
            selection_target = "oracle_regret"
            objective = "minimize_candidate_oracle_regret"
            training_mode = "oracle_regret_regression"
            runtime_policy = "learned_quality_v1_1"
        else:
            selection_target = "continuous_regret"
            objective = "minimize_candidate_selection_regret"
            training_mode = "continuous_selection_cost"
            runtime_policy = "learned_quality_v1_1"
    else:
        coefficients, intercept = fit_logistic(
            x,
            y,
            l2=l2,
            learning_rate=learning_rate,
            iterations=iterations,
            sample_weight=train_sample_weights,
        )
        model_type = MODEL_TYPE_LOGISTIC_REGRESSION
        score_semantics = SCORE_SEMANTICS_PREDICTED_RISK
        version = "learned_quality_v1"
        selection_target = "binary_failure_or_high_gap"
        objective = "minimize_candidate_failure_risk"
        training_mode = "binary_failure_or_high_gap"
        runtime_policy = "learned_quality_v1"

    scorer = RuntimeResolverScorer(
        features=features,
        coefficients=tuple(float(item) for item in coefficients),
        intercept=float(intercept),
        model_type=model_type,
        target=target,
        score_semantics=score_semantics,
        higher_is_better=False,
        failure_threshold=failure_threshold,
        calibration={"type": "none", "params": {}},
        version=version,
        selection_target=selection_target,
        objective=objective,
        training_mode=training_mode,
        runtime_policy=runtime_policy,
    )
    scorer_path = write_runtime_resolver_scorer(scorer, output_dir / SCORER_ARTIFACT)
    rows_path = write_tagged_rows_csv(train_tagged_rows, output_dir / TRAINING_ROWS_CSV)
    eval_rows_path = write_tagged_rows_csv(eval_tagged_rows, output_dir / EVAL_ROWS_CSV)

    condition_report_rows = _training_report_rows(tagged_rows, group_by="condition")
    region_report_rows = _training_report_rows(tagged_rows, group_by="region")
    condition_report_path = write_training_report_csv(
        condition_report_rows,
        output_dir / SCORER_CONDITION_REPORT_CSV,
    )
    region_report_path = write_training_report_csv(
        region_report_rows,
        output_dir / SCORER_REGION_REPORT_CSV,
    )

    metrics = scorer_row_metrics(scorer, rows)
    target_stats = target_distribution_stats(rows, target=target)
    train_metrics = scorer_row_metrics(scorer, train_rows)
    eval_metrics = scorer_row_metrics(scorer, eval_rows)
    production_eval_metrics = scorer_row_metrics(
        scorer, source_quality_rows(eval_tagged_rows, SOURCE_PRODUCTION_VALIDATED)
    )
    gt_eval_metrics = scorer_row_metrics(
        scorer, source_quality_rows(eval_tagged_rows, SOURCE_GT_HARD)
    )
    metrics.update(
        {
            "artifact": str(scorer_path),
            "training_rows": str(rows_path),
            "eval_rows": str(eval_rows_path),
            "candidate_table": "" if candidate_table_path is None else str(candidate_table_path),
            "candidate_count": len(candidates),
            "candidates": list(candidates),
            "feature_count": len(features),
            "target": target,
            "model_type": model_type,
            "score_semantics": score_semantics,
            "higher_is_better": False,
            "target_stats": target_stats,
            "sample_weighting": scorer_sample_weighting_stats(train_tagged_rows),
            "scorer_report_by_condition": str(condition_report_path),
            "scorer_report_by_region": str(region_report_path),
            "failure_threshold": failure_threshold,
            "high_gap_threshold": high_gap_threshold,
            "normalized_regret_clamp": DEFAULT_REGRET_NORMALIZER,
            "failure_cost_penalty": DEFAULT_FAILURE_COST_PENALTY,
            "collapse_cost_penalty": DEFAULT_COLLAPSE_COST_PENALTY,
            "l2": l2,
            "split_seed": split_seed,
            "eval_fraction": eval_fraction,
            "allow_image_backfill": allow_image_backfill,
            "gt_hard_resolver_metadata": (
                "" if gt_hard_resolver_metadata is None else str(gt_hard_resolver_metadata)
            ),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "production_only_eval_metrics": production_eval_metrics,
            "gt_hard_only_eval_metrics": gt_eval_metrics,
        }
    )
    metrics_path = write_json(output_dir / TRAINING_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def _train_lambdarank_from_tagged_rows(
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
) -> dict[str, T.Any]:
    """Train learned_quality_v2 from prebuilt scorer rows."""

    try:
        import lightgbm as lgb
    except ModuleNotFoundError as err:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "learned_quality_v2 training requires lightgbm; install project requirements first"
        ) from err

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_train, train_group_sizes = _grouped_rows(train_tagged_rows)
    train_rows = untag_quality_rows(grouped_train)
    features = feature_order(train_rows)
    x = feature_matrix([row.feature_values for row in train_rows], features)
    y = np.asarray([_lambdarank_label(row) for row in train_rows], dtype="int32")
    item_weights = np.asarray(
        [_lambdarank_weight(row, source) for row, source in grouped_train],
        dtype="float64",
    )
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
        "artifact_schema_version": 2,
        "version": "learned_quality_v2",
        "scorer_version": "learned_quality_v2",
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "target": TARGET_SELECTION_COST,
        "objective": "lambdarank_inverse_regret",
        "training_mode": "grouped_lambdarank",
        "selection_target": "inverse_selection_cost_rank",
        "runtime_policy": "learned_quality_v2",
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "failure_threshold": failure_threshold,
        "features": list(features),
        "model_data": booster.model_to_string(),
        "training_data_counts": {
            "row_count": len(train_rows),
            "sample_group_count": len(train_group_sizes),
            "eval_row_count": len(eval_tagged_rows),
            "candidate_count": len(candidates),
        },
        "split_ids": {
            "seed": split_seed,
            "eval_fraction": eval_fraction,
            "train_group_count": len(train_group_sizes),
        },
        "feature_importances": importances,
        "calibration": {"type": "none", "params": {}},
        "sample_weighting": scorer_sample_weighting_stats(grouped_train),
        "lightgbm_params": {
            "objective": "lambdarank",
            "n_estimators": iterations,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": split_seed,
            "deterministic": True,
        },
    }
    artifact_path = write_json(output_dir / SCORER_V2_ARTIFACT, artifact)
    rows_path = write_tagged_rows_csv(grouped_train, output_dir / TRAINING_ROWS_CSV)
    eval_rows_path = write_tagged_rows_csv(eval_tagged_rows, output_dir / EVAL_ROWS_CSV)
    importances_path = _write_feature_importance_csv(
        output_dir / "runtime_resolver_scorer_v2_feature_importances.csv",
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
        "target": TARGET_SELECTION_COST,
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "split_seed": split_seed,
        "eval_fraction": eval_fraction,
        "training_data_counts": artifact["training_data_counts"],
        "split_ids": artifact["split_ids"],
        "sample_weighting": artifact["sample_weighting"],
        "lightgbm_params": artifact["lightgbm_params"],
    }
    metrics_path = write_json(output_dir / TRAINING_V2_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


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
    l2: float = 0.001,
    learning_rate: float = 0.1,
    iterations: int = 1500,
    eval_fraction: float = 0.20,
    split_seed: int = 42,
    allow_image_backfill: bool = False,
    v2_learning_rate: float = 0.05,
    v2_iterations: int = 150,
    v2_num_leaves: int = 31,
) -> dict[str, T.Any]:
    """Train all learned runtime resolver scorers from one canonical row split."""

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
            "candidates": list(candidates),
            "failure_threshold": failure_threshold,
            "high_gap_threshold": high_gap_threshold,
            "outlier_threshold": outlier_threshold,
            "eval_fraction": eval_fraction,
            "split_seed": split_seed,
            "allow_image_backfill": allow_image_backfill,
        },
    )

    candidate_table_path = write_candidate_table_csv(
        scorer_candidate_table_rows(contexts),
        output_dir / TRAINING_CANDIDATE_TABLE_CSV,
    )
    binary_dir = output_dir / "v1_binary"
    continuous_dir = output_dir / "v1_1_selection_cost"
    oracle_regret_dir = output_dir / "v1_1_oracle_regret"
    v2_dir = output_dir / "v2_lambdarank"
    scorers_dir = output_dir / SCORERS_DIR
    scorers_dir.mkdir(parents=True, exist_ok=True)

    binary_metrics = _train_linear_or_logistic_from_tagged_rows(
        tagged_rows=tagged_rows,
        train_tagged_rows=train_tagged_rows,
        eval_tagged_rows=eval_tagged_rows,
        candidates=candidates,
        output_dir=binary_dir,
        target=TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
        failure_threshold=failure_threshold,
        high_gap_threshold=high_gap_threshold,
        l2=l2,
        learning_rate=learning_rate,
        iterations=iterations,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        allow_image_backfill=allow_image_backfill,
        gt_hard_resolver_metadata=gt_hard_resolver_metadata,
        candidate_table_path=candidate_table_path,
    )
    continuous_metrics = _train_linear_or_logistic_from_tagged_rows(
        tagged_rows=tagged_rows,
        train_tagged_rows=train_tagged_rows,
        eval_tagged_rows=eval_tagged_rows,
        candidates=candidates,
        output_dir=continuous_dir,
        target=TARGET_SELECTION_COST,
        failure_threshold=failure_threshold,
        high_gap_threshold=high_gap_threshold,
        l2=l2,
        learning_rate=learning_rate,
        iterations=iterations,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        allow_image_backfill=allow_image_backfill,
        gt_hard_resolver_metadata=gt_hard_resolver_metadata,
        candidate_table_path=candidate_table_path,
    )
    oracle_regret_metrics = _train_linear_or_logistic_from_tagged_rows(
        tagged_rows=tagged_rows,
        train_tagged_rows=train_tagged_rows,
        eval_tagged_rows=eval_tagged_rows,
        candidates=candidates,
        output_dir=oracle_regret_dir,
        target=TARGET_ORACLE_REGRET,
        failure_threshold=failure_threshold,
        high_gap_threshold=high_gap_threshold,
        l2=l2,
        learning_rate=learning_rate,
        iterations=iterations,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        allow_image_backfill=allow_image_backfill,
        gt_hard_resolver_metadata=gt_hard_resolver_metadata,
        candidate_table_path=candidate_table_path,
    )
    v2_metrics = _train_lambdarank_from_tagged_rows(
        train_tagged_rows=train_tagged_rows,
        eval_tagged_rows=eval_tagged_rows,
        candidates=candidates,
        output_dir=v2_dir,
        failure_threshold=failure_threshold,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        learning_rate=v2_learning_rate,
        iterations=v2_iterations,
        num_leaves=v2_num_leaves,
    )

    canonical_binary = scorers_dir / "learned_quality_v1.json"
    canonical_continuous = scorers_dir / "learned_quality_v1_1.json"
    canonical_oracle_regret = scorers_dir / "learned_quality_v1_1_oracle_regret.json"
    canonical_v2 = scorers_dir / "learned_quality_v2.json"
    shutil.copy2(binary_dir / SCORER_ARTIFACT, canonical_binary)
    shutil.copy2(continuous_dir / SCORER_ARTIFACT, canonical_continuous)
    shutil.copy2(oracle_regret_dir / SCORER_ARTIFACT, canonical_oracle_regret)
    shutil.copy2(v2_dir / SCORER_V2_ARTIFACT, canonical_v2)

    metrics = {
        "artifact_schema_version": 1,
        "scorer_dataset": dataset_manifest,
        "scorers": {
            "learned_quality_v1": {
                **binary_metrics,
                "canonical_artifact": str(canonical_binary),
            },
            "learned_quality_v1_1": {
                **continuous_metrics,
                "canonical_artifact": str(canonical_continuous),
            },
            "learned_quality_v1_1_oracle_regret": {
                **oracle_regret_metrics,
                "canonical_artifact": str(canonical_oracle_regret),
            },
            "learned_quality_v2": {
                **v2_metrics,
                "canonical_artifact": str(canonical_v2),
            },
        },
        "candidate_table": str(candidate_table_path),
        "compatibility_artifacts": {
            "legacy_per_scorer_training_rows": [
                str(binary_dir / TRAINING_ROWS_CSV),
                str(continuous_dir / TRAINING_ROWS_CSV),
                str(oracle_regret_dir / TRAINING_ROWS_CSV),
                str(v2_dir / TRAINING_ROWS_CSV),
            ],
            "legacy_per_scorer_eval_rows": [
                str(binary_dir / EVAL_ROWS_CSV),
                str(continuous_dir / EVAL_ROWS_CSV),
                str(oracle_regret_dir / EVAL_ROWS_CSV),
                str(v2_dir / EVAL_ROWS_CSV),
            ],
            "candidate_table": str(candidate_table_path),
            "note": (
                "These files are kept for compatibility. New consumers should use "
                "scorer_dataset/rows.csv and scorer_dataset/manifest.json."
            ),
        },
        "candidate_table_status": "compatibility_derived_from_scorer_contexts",
        "split_seed": split_seed,
        "eval_fraction": eval_fraction,
        "candidates": list(candidates),
    }
    metrics_path = write_json(scorers_dir / SCORER_SUITE_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    metrics["artifact"] = str(canonical_continuous)
    return metrics


def train_runtime_resolver_scorer_v2(
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
    learning_rate: float = 0.05,
    iterations: int = 150,
    num_leaves: int = 31,
) -> dict[str, T.Any]:
    """Train learned_quality_v2 with LightGBM LambdaRank grouped by sample."""
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as err:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "learned_quality_v2 training requires lightgbm; install project requirements first"
        ) from err

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
    )
    tagged_rows = tagged_quality_rows(contexts, high_gap_threshold=high_gap_threshold)
    if not tagged_rows:
        raise ValueError("no scorer training rows were loaded")
    train_tagged_rows, eval_tagged_rows = split_tagged_rows(
        tagged_rows,
        eval_fraction=eval_fraction,
        seed=split_seed,
    )
    grouped_train, train_group_sizes = _grouped_rows(train_tagged_rows)
    train_rows = untag_quality_rows(grouped_train)
    features = feature_order(train_rows)
    x = feature_matrix([row.feature_values for row in train_rows], features)
    y = np.asarray([_lambdarank_label(row) for row in train_rows], dtype="int32")
    item_weights = np.asarray(
        [_lambdarank_weight(row, source) for row, source in grouped_train],
        dtype="float64",
    )
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
        "artifact_schema_version": 2,
        "version": "learned_quality_v2",
        "scorer_version": "learned_quality_v2",
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "target": TARGET_SELECTION_COST,
        "objective": "lambdarank_inverse_regret",
        "training_mode": "grouped_lambdarank",
        "selection_target": "inverse_selection_cost_rank",
        "runtime_policy": "learned_quality_v2",
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "failure_threshold": failure_threshold,
        "features": list(features),
        "model_data": booster.model_to_string(),
        "training_data_counts": {
            "row_count": len(train_rows),
            "sample_group_count": len(train_group_sizes),
            "eval_row_count": len(eval_tagged_rows),
            "candidate_count": len(candidates),
        },
        "split_ids": {
            "seed": split_seed,
            "eval_fraction": eval_fraction,
            "train_group_count": len(train_group_sizes),
        },
        "feature_importances": importances,
        "calibration": {"type": "none", "params": {}},
        "sample_weighting": {
            "base": 1.0,
            "candidate_failure_bonus": 2.0,
            "hard_pose_bucket_bonus": 0.5,
            "production_validated_bonus": 0.25,
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
    artifact_path = write_json(output_dir / SCORER_V2_ARTIFACT, artifact)
    rows_path = write_tagged_rows_csv(grouped_train, output_dir / TRAINING_ROWS_CSV)
    eval_rows_path = write_tagged_rows_csv(eval_tagged_rows, output_dir / EVAL_ROWS_CSV)
    importances_path = _write_feature_importance_csv(
        output_dir / "runtime_resolver_scorer_v2_feature_importances.csv",
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
        "target": TARGET_SELECTION_COST,
        "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
        "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
        "higher_is_better": False,
        "split_seed": split_seed,
        "eval_fraction": eval_fraction,
        "training_data_counts": artifact["training_data_counts"],
        "split_ids": artifact["split_ids"],
        "sample_weighting": artifact["sample_weighting"],
        "lightgbm_params": artifact["lightgbm_params"],
    }
    metrics_path = write_json(output_dir / TRAINING_V2_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


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
    eval_fraction: float = 0.20,
    split_seed: int = 42,
    allow_image_backfill: bool = False,
    target: str = TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
) -> dict[str, T.Any]:
    """Train the scorer and write the portable artifact plus diagnostics."""
    if target not in SCORER_TARGETS:
        raise ValueError(f"target must be one of {SCORER_TARGETS}, got {target!r}")
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
    )
    tagged_rows = tagged_quality_rows(contexts, high_gap_threshold=high_gap_threshold)
    if not tagged_rows:
        raise ValueError("no scorer training rows were loaded")
    candidate_rows = scorer_candidate_table_rows(contexts)
    train_tagged_rows, eval_tagged_rows = split_tagged_rows(
        tagged_rows,
        eval_fraction=eval_fraction,
        seed=split_seed,
    )
    rows = untag_quality_rows(tagged_rows)
    train_rows = untag_quality_rows(train_tagged_rows)
    eval_rows = untag_quality_rows(eval_tagged_rows)
    features = feature_order(train_rows)
    x = feature_matrix([row.feature_values for row in train_rows], features)
    y = np.asarray(
        [scorer_target_value(row, target) for row in train_rows],
        dtype="float64",
    )
    if target in REGRESSION_TARGETS:
        coefficients, intercept = fit_linear_regression(x, y, l2=l2)
        model_type = MODEL_TYPE_LINEAR_REGRESSION
        score_semantics = SCORE_SEMANTICS_PREDICTED_COST
        version = "continuous_regret_v1_1"
        selection_target = "continuous_regret"
        objective = "minimize_candidate_selection_regret"
        training_mode = "continuous_selection_cost"
        # The v1.1 continuous scorer is selected at runtime by policy
        # "learned_quality_v1_1"; the v1 binary classifier ships under
        # "learned_quality_v1". Writing the wrong runtime_policy lets the
        # production bundle install this artifact under a manifest slot it
        # was not trained for and the runtime validation catches that.
        runtime_policy = "learned_quality_v1_1"
    else:
        coefficients, intercept = fit_logistic(
            x,
            y,
            l2=l2,
            learning_rate=learning_rate,
            iterations=iterations,
        )
        model_type = MODEL_TYPE_LOGISTIC_REGRESSION
        score_semantics = SCORE_SEMANTICS_PREDICTED_RISK
        version = "learned_quality_v1"
        selection_target = "binary_failure_or_high_gap"
        objective = "minimize_candidate_failure_risk"
        training_mode = "binary_failure_or_high_gap"
        runtime_policy = "learned_quality_v1"
    scorer = RuntimeResolverScorer(
        features=features,
        coefficients=tuple(float(item) for item in coefficients),
        intercept=float(intercept),
        model_type=model_type,
        target=target,
        score_semantics=score_semantics,
        higher_is_better=False,
        failure_threshold=failure_threshold,
        calibration={"type": "none", "params": {}},
        version=version,
        selection_target=selection_target,
        objective=objective,
        training_mode=training_mode,
        runtime_policy=runtime_policy,
    )
    scorer_path = write_runtime_resolver_scorer(scorer, output_dir / SCORER_ARTIFACT)
    rows_path = write_tagged_rows_csv(train_tagged_rows, output_dir / TRAINING_ROWS_CSV)
    eval_rows_path = write_tagged_rows_csv(eval_tagged_rows, output_dir / EVAL_ROWS_CSV)
    candidate_table_path = write_candidate_table_csv(
        candidate_rows,
        output_dir / TRAINING_CANDIDATE_TABLE_CSV,
    )
    metrics = scorer_row_metrics(scorer, rows)
    target_stats = target_distribution_stats(rows, target=target)
    train_metrics = scorer_row_metrics(scorer, train_rows)
    eval_metrics = scorer_row_metrics(scorer, eval_rows)
    production_eval_metrics = scorer_row_metrics(
        scorer, source_quality_rows(eval_tagged_rows, SOURCE_PRODUCTION_VALIDATED)
    )
    gt_eval_metrics = scorer_row_metrics(
        scorer, source_quality_rows(eval_tagged_rows, SOURCE_GT_HARD)
    )
    metrics.update(
        {
            "artifact": str(scorer_path),
            "training_rows": str(rows_path),
            "eval_rows": str(eval_rows_path),
            "candidate_table": str(candidate_table_path),
            "candidate_count": len(candidates),
            "candidates": list(candidates),
            "feature_count": len(features),
            "target": target,
            "model_type": model_type,
            "score_semantics": score_semantics,
            "higher_is_better": False,
            "target_stats": target_stats,
            "failure_threshold": failure_threshold,
            "high_gap_threshold": high_gap_threshold,
            "normalized_regret_clamp": DEFAULT_REGRET_NORMALIZER,
            "failure_cost_penalty": DEFAULT_FAILURE_COST_PENALTY,
            "collapse_cost_penalty": DEFAULT_COLLAPSE_COST_PENALTY,
            "l2": l2,
            "split_seed": split_seed,
            "eval_fraction": eval_fraction,
            "allow_image_backfill": allow_image_backfill,
            "gt_hard_resolver_metadata": (
                "" if gt_hard_resolver_metadata is None else str(gt_hard_resolver_metadata)
            ),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "production_only_eval_metrics": production_eval_metrics,
            "gt_hard_only_eval_metrics": gt_eval_metrics,
        }
    )
    metrics_path = write_json(output_dir / TRAINING_METRICS_JSON, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


__all__ = [
    "SCORER_ARTIFACT",
    "SCORER_V2_ARTIFACT",
    "SCORERS_DIR",
    "SCORER_SUITE_METRICS_JSON",
    "SCORER_SUITE_SENTINEL_JSON",
    "TRAINING_CANDIDATE_TABLE_CSV",
    "TRAINING_METRICS_JSON",
    "TRAINING_V2_METRICS_JSON",
    "TRAINING_ROWS_CSV",
    "EVAL_ROWS_CSV",
    "feature_order",
    "fit_linear_regression",
    "fit_logistic",
    "scorer_target_value",
    "scorer_row_metrics",
    "split_tagged_rows",
    "target_distribution_stats",
    "train_runtime_resolver_scorer",
    "train_runtime_resolver_scorer_suite",
    "train_runtime_resolver_scorer_v2",
    "write_tagged_rows_csv",
]
