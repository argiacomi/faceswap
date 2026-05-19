#!/usr/bin/env python3
"""Shared data builders for runtime resolver scorer training/evaluation."""

from __future__ import annotations

import csv
import json
import logging
import typing as T
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.datasets.manifest_io import (
    LandmarkSample,
    bbox_from_truth_fallback,
    load_manifest,
)
from lib.landmarks.ensemble.runtime_resolver import (
    CandidateMetrics,
    CandidateRecord,
    ModelPrediction,
    RuntimeResolverConfig,
    _circular_median,
    _max_landmark_consensus_px,
    _metric_for_candidate,
    _populate_consensus_geometry,
    _shape_reasons,
    build_candidates,
    infer_runtime_bucket,
    resolve_runtime,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import candidate_feature_map
from lib.landmarks.ensemble.strategies import canonical_strategy
from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.evaluation.nme_metrics import evaluate_prediction

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_THRESHOLD: float = 0.08
DEFAULT_HIGH_GAP_THRESHOLD: float = 0.01
DEFAULT_OUTLIER_THRESHOLD: float = 3.5
DEFAULT_RESOLVER_CANDIDATES: tuple[str, ...] = (
    "hrnet",
    "spiga",
    "orformer",
    "static_weighted",
    "static_weighted_downweight",
    "static_weighted_hard_drop",
    "weighted_median",
)
DEFAULT_SCORER_CANDIDATES: tuple[str, ...] = DEFAULT_RESOLVER_CANDIDATES
CANDIDATE_TABLE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "dataset",
    "condition",
    "candidate",
    "nme",
    "failure",
    "is_oracle",
    "gap_vs_oracle",
    "cloud_area_ratio",
    "hull_area_ratio",
    "points_outside_expanded_bbox_fraction",
    "eye_mouth_order_valid_after_deroll",
    "roi_center_consensus_distance",
    "landmark_consensus_distance",
    "roll",
    "yaw",
    "roll_delta_to_consensus",
    "yaw_delta_to_consensus",
    "max_disagreement_px",
    "candidate_yaw_disagreement",
    "runtime_bucket",
    "geometry_veto_reasons",
)


@dataclass(frozen=True)
class CandidateQualityRow:
    """One sample/candidate row for scorer training or policy evaluation."""

    sample_id: str
    dataset: str
    condition: str
    candidate_name: str
    candidate_nme: float
    failure_label: bool
    is_oracle: bool
    was_selected_by_current_policy: bool
    gap_vs_oracle: float
    runtime_bucket: str
    risk_route: str
    feature_values: dict[str, float]
    selected_by_current_policy: str
    oracle: str
    geometry_veto_reasons: tuple[str, ...]

    def to_csv_row(self) -> dict[str, T.Any]:
        """Return stable CSV fields plus a JSON feature payload."""
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "condition": self.condition,
            "candidate_name": self.candidate_name,
            "candidate_nme": self.candidate_nme,
            "failure_label": int(self.failure_label),
            "is_oracle": int(self.is_oracle),
            "was_selected_by_current_policy": int(self.was_selected_by_current_policy),
            "gap_vs_oracle": self.gap_vs_oracle,
            "runtime_bucket": self.runtime_bucket,
            "risk_route": self.risk_route,
            "geometry_veto_reasons": "|".join(self.geometry_veto_reasons),
            "selected_by_current_policy": self.selected_by_current_policy,
            "oracle": self.oracle,
            "features_json": json.dumps(self.feature_values, sort_keys=True),
        }


@dataclass(frozen=True)
class SampleCandidateContext:
    """Resolved candidates and row data for one manifest sample."""

    sample_id: str
    dataset: str
    condition: str
    candidates: tuple[CandidateRecord, ...]
    metrics: dict[str, CandidateMetrics]
    nme_by_candidate: dict[str, float]
    failure_by_candidate: dict[str, bool]
    runtime_bucket: str
    risk_route: str
    current_policy_choice: str
    oracle: str
    model_predictions_available: dict[str, bool]
    roll_estimate: float | None
    yaw_estimate: float | None
    candidate_yaw_disagreement: float | None
    max_disagreement_px: float


def parse_candidates(
    value: str | None, weights: T.Mapping[str, T.Sequence[float]]
) -> tuple[str, ...]:
    """Parse CLI candidate list with a stable default."""
    if value:
        return tuple(item.strip() for item in value.split(",") if item.strip())
    ordered = [name for name in DEFAULT_SCORER_CANDIDATES if name in weights or _is_strategy(name)]
    ordered.extend(name for name in weights if name not in ordered)
    return tuple(dict.fromkeys(ordered))


def _is_strategy(name: str) -> bool:
    try:
        canonical_strategy(name)
    except (KeyError, ValueError):
        return False
    return True


def _load_truth(sample: LandmarkSample) -> np.ndarray:
    return np.load(sample.landmarks).astype("float32")


def _candidate_config(
    weights: T.Mapping[str, T.Sequence[float]],
    *,
    outlier_threshold: float,
) -> RuntimeResolverConfig:
    return RuntimeResolverConfig(
        policy="roll_aware_veto",
        weights=weights,
        general_strategy="static_weighted",
        hard_case_strategy="static_weighted_downweight",
        secondary_hard_case_strategy="static_weighted_hard_drop",
        fallback_strategy="plain_average",
        outlier_threshold=outlier_threshold,
    )


def _read_predictions(
    cache: DiskPredictionCache,
    sample: LandmarkSample,
    *,
    models: T.Iterable[str],
) -> list[ModelPrediction]:
    available = set(cache.available_models(sample.sample_id))
    missing = sorted(set(models) - available)
    if missing:
        logger.warning("sample %s missing cached models: %s", sample.sample_id, missing)
    return [
        ModelPrediction(model, cache.read(sample.sample_id, model).landmarks)
        for model in sorted(set(models))
        if model in available
    ]


def _nme(
    landmarks: np.ndarray,
    truth: np.ndarray,
    *,
    sample: LandmarkSample,
    failure_threshold: float,
) -> tuple[float, bool]:
    metrics = evaluate_prediction(
        landmarks,
        truth,
        normalizer=sample.normalizer,
        failure_threshold=failure_threshold,
        visibility=sample.visibility,
    )
    return float(metrics["nme"]), bool(metrics["failure"])


def build_sample_context(
    sample: LandmarkSample,
    *,
    cache: DiskPredictionCache,
    requested_candidates: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]],
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> SampleCandidateContext:
    """Build candidates, diagnostics, labels, and current-policy choice for one sample."""
    single_models = {name for name in requested_candidates if not _is_strategy(name)}
    model_names = tuple(dict.fromkeys((*weights.keys(), *single_models)))
    predictions = _read_predictions(cache, sample, models=model_names)
    if not predictions:
        raise FileNotFoundError(f"sample {sample.sample_id!r} has no cached model predictions")
    config = _candidate_config(weights, outlier_threshold=outlier_threshold)
    all_candidates = build_candidates(predictions, config)
    by_name = {candidate.name: candidate for candidate in all_candidates}
    missing = [name for name in requested_candidates if name not in by_name]
    if missing:
        logger.warning("sample %s could not build candidates: %s", sample.sample_id, missing)
    candidates = tuple(by_name[name] for name in requested_candidates if name in by_name)
    if not candidates:
        raise ValueError(f"sample {sample.sample_id!r} produced no requested candidates")
    truth = _load_truth(sample)
    reference_bbox = sample.face_bbox or bbox_from_truth_fallback(truth)
    metrics = {
        candidate.name: _metric_for_candidate(candidate, reference_bbox=reference_bbox)
        for candidate in candidates
    }
    _populate_consensus_geometry(candidates, metrics, reference_bbox=reference_bbox)
    roll_estimate = _circular_median(
        [metric.roll_degrees for metric in metrics.values() if metric.roll_degrees is not None]
    )
    yaw_estimate = _circular_median(
        [metric.yaw_degrees for metric in metrics.values() if metric.yaw_degrees is not None]
    )
    max_disagreement_px = _max_landmark_consensus_px(candidates)
    bucket_result = infer_runtime_bucket(
        image_crop=None,
        crop_to_frame_matrix=None,
        detector_bbox=reference_bbox,
        candidates=candidates,
        metrics=metrics,
        yaw_estimate=yaw_estimate,
        roll_estimate=roll_estimate,
        max_disagreement_px=max_disagreement_px,
        hard_roll_degrees=config.hard_roll_degrees,
    )
    for name, metric in metrics.items():
        metric.geometry_veto_reasons = _shape_reasons(bucket_result.bucket, name, metric)

    nme_by_candidate: dict[str, float] = {}
    failure_by_candidate: dict[str, bool] = {}
    for candidate in candidates:
        nme, failure = _nme(
            candidate.landmarks,
            truth,
            sample=sample,
            failure_threshold=failure_threshold,
        )
        nme_by_candidate[candidate.name] = nme
        failure_by_candidate[candidate.name] = failure
    current = resolve_runtime(
        predictions,
        config,
        detector_bbox=reference_bbox,
    )
    oracle = min(nme_by_candidate, key=lambda name: nme_by_candidate[name])
    current_policy_choice = (
        current.selected_candidate if current.selected_candidate in nme_by_candidate else oracle
    )
    hard_bucket = bucket_result.bucket not in {"frontal", "intermediate", "no_pose"}
    vetoed = {name for name, metric in metrics.items() if metric.geometry_veto_reasons}
    risk_route = (
        "high_risk"
        if hard_bucket or max_disagreement_px > config.hard_disagreement_px or vetoed
        else "low_risk"
    )
    return SampleCandidateContext(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        condition=sample.condition or bucket_result.bucket or "unknown",
        candidates=candidates,
        metrics=metrics,
        nme_by_candidate=nme_by_candidate,
        failure_by_candidate=failure_by_candidate,
        runtime_bucket=bucket_result.bucket,
        risk_route=risk_route,
        current_policy_choice=current_policy_choice,
        oracle=oracle,
        model_predictions_available={prediction.model: True for prediction in predictions},
        roll_estimate=roll_estimate,
        yaw_estimate=yaw_estimate,
        candidate_yaw_disagreement=bucket_result.features.get("candidate_yaw_disagreement"),
        max_disagreement_px=max_disagreement_px,
    )


def rows_for_context(
    context: SampleCandidateContext,
    *,
    high_gap_threshold: float = DEFAULT_HIGH_GAP_THRESHOLD,
) -> list[CandidateQualityRow]:
    """Expand a sample context into one row per candidate."""
    rows: list[CandidateQualityRow] = []
    oracle_nme = context.nme_by_candidate[context.oracle]
    for candidate in context.candidates:
        metric = context.metrics[candidate.name]
        gap = context.nme_by_candidate[candidate.name] - oracle_nme
        rows.append(
            CandidateQualityRow(
                sample_id=context.sample_id,
                dataset=context.dataset,
                condition=context.condition,
                candidate_name=candidate.name,
                candidate_nme=context.nme_by_candidate[candidate.name],
                failure_label=bool(
                    context.failure_by_candidate[candidate.name] or gap > high_gap_threshold
                ),
                is_oracle=candidate.name == context.oracle,
                was_selected_by_current_policy=(candidate.name == context.current_policy_choice),
                gap_vs_oracle=gap,
                runtime_bucket=context.runtime_bucket,
                risk_route=context.risk_route,
                feature_values=candidate_feature_map(
                    candidate,
                    metric,
                    runtime_bucket=context.runtime_bucket,
                    risk_route=context.risk_route,
                    model_predictions_available=context.model_predictions_available,
                    roll_estimate=context.roll_estimate,
                    yaw_estimate=context.yaw_estimate,
                    candidate_yaw_disagreement=context.candidate_yaw_disagreement,
                    max_disagreement_px=context.max_disagreement_px,
                ),
                selected_by_current_policy=context.current_policy_choice,
                oracle=context.oracle,
                geometry_veto_reasons=tuple(metric.geometry_veto_reasons),
            )
        )
    return rows


def candidate_table_rows_for_context(
    context: SampleCandidateContext,
) -> list[dict[str, T.Any]]:
    """Expand a sample context into canonical candidate diagnostic table rows."""
    rows: list[dict[str, T.Any]] = []
    oracle_nme = context.nme_by_candidate[context.oracle]
    for candidate in context.candidates:
        metric = context.metrics[candidate.name]
        nme = context.nme_by_candidate[candidate.name]
        rows.append(
            {
                "sample_id": context.sample_id,
                "dataset": context.dataset,
                "condition": context.condition,
                "candidate": candidate.name,
                "nme": nme,
                "failure": int(context.failure_by_candidate[candidate.name]),
                "is_oracle": int(candidate.name == context.oracle),
                "gap_vs_oracle": nme - oracle_nme,
                "cloud_area_ratio": metric.cloud_area_ratio,
                "hull_area_ratio": metric.hull_area_ratio,
                "points_outside_expanded_bbox_fraction": (
                    metric.points_outside_expanded_bbox_fraction
                ),
                "eye_mouth_order_valid_after_deroll": (
                    None
                    if metric.eye_mouth_order_valid_after_deroll is None
                    else int(metric.eye_mouth_order_valid_after_deroll)
                ),
                "roi_center_consensus_distance": metric.roi_center_consensus_distance,
                "landmark_consensus_distance": metric.landmark_consensus_distance,
                "roll": metric.roll_degrees,
                "yaw": metric.yaw_degrees,
                "roll_delta_to_consensus": (
                    None
                    if metric.roll_degrees is None or context.roll_estimate is None
                    else abs(metric.roll_degrees - context.roll_estimate)
                ),
                "yaw_delta_to_consensus": (
                    None
                    if metric.yaw_degrees is None or context.yaw_estimate is None
                    else abs(metric.yaw_degrees - context.yaw_estimate)
                ),
                "max_disagreement_px": context.max_disagreement_px,
                "candidate_yaw_disagreement": context.candidate_yaw_disagreement,
                "runtime_bucket": context.runtime_bucket,
                "geometry_veto_reasons": "|".join(metric.geometry_veto_reasons),
            }
        )
    return rows


def candidate_table_rows(
    contexts: T.Sequence[SampleCandidateContext],
) -> list[dict[str, T.Any]]:
    """Return canonical diagnostic table rows for all sample contexts."""
    rows: list[dict[str, T.Any]] = []
    for context in contexts:
        rows.extend(candidate_table_rows_for_context(context))
    return rows


def load_contexts(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str] | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> list[SampleCandidateContext]:
    """Load all sample contexts for one manifest/cache pair."""
    weights = load_weights(weights_path)
    requested = tuple(candidates or parse_candidates(None, weights))
    cache = DiskPredictionCache(cache_dir)
    contexts: list[SampleCandidateContext] = []
    for sample in load_manifest(manifest_path):
        try:
            contexts.append(
                build_sample_context(
                    sample,
                    cache=cache,
                    requested_candidates=requested,
                    weights=weights,
                    failure_threshold=failure_threshold,
                    outlier_threshold=outlier_threshold,
                )
            )
        except (FileNotFoundError, ValueError) as err:
            logger.warning("skipping sample %s: %s", sample.sample_id, err)
    return contexts


def write_candidate_table_csv(rows: T.Sequence[dict[str, T.Any]], path: Path) -> Path:
    """Write the canonical candidate diagnostic table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANDIDATE_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in CANDIDATE_TABLE_COLUMNS})
    return path


def write_candidate_table_parquet(rows: T.Sequence[dict[str, T.Any]], path: Path) -> Path:
    """Write the canonical candidate diagnostic table to parquet when available."""
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "parquet export requires pandas plus a parquet engine such as pyarrow; "
            "use --output-csv in this environment or install the optional parquet stack"
        ) from err
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{name: row.get(name) for name in CANDIDATE_TABLE_COLUMNS} for row in rows],
        columns=list(CANDIDATE_TABLE_COLUMNS),
    )
    frame.to_parquet(path, index=False)
    return path


def export_candidate_table(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str] | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> list[dict[str, T.Any]]:
    """Build the canonical candidate diagnostic table for one manifest/cache pair."""
    contexts = load_contexts(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=candidates,
        failure_threshold=failure_threshold,
        outlier_threshold=outlier_threshold,
    )
    return candidate_table_rows(contexts)


def write_rows_csv(rows: T.Sequence[CandidateQualityRow], path: Path) -> Path:
    """Write scorer training rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = sorted({name for row in rows for name in row.feature_values})
    fieldnames = [
        "sample_id",
        "dataset",
        "condition",
        "candidate_name",
        "candidate_nme",
        "failure_label",
        "is_oracle",
        "was_selected_by_current_policy",
        "gap_vs_oracle",
        "runtime_bucket",
        "risk_route",
        "geometry_veto_reasons",
        "selected_by_current_policy",
        "oracle",
        "features_json",
        *feature_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row.to_csv_row(),
                    **{name: row.feature_values.get(name, 0.0) for name in feature_names},
                }
            )
    return path


__all__ = [
    "CANDIDATE_TABLE_COLUMNS",
    "CandidateQualityRow",
    "DEFAULT_RESOLVER_CANDIDATES",
    "SampleCandidateContext",
    "build_sample_context",
    "candidate_table_rows",
    "candidate_table_rows_for_context",
    "export_candidate_table",
    "load_contexts",
    "parse_candidates",
    "rows_for_context",
    "write_candidate_table_csv",
    "write_candidate_table_parquet",
    "write_rows_csv",
]
