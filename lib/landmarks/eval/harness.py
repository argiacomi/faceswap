#!/usr/bin/env python3
"""Cache-driven landmark quality harness."""

from __future__ import annotations

import csv
import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.weights import load_weights, weights_matrix_for_models
from lib.landmarks.eval.metrics import (
    evaluate_prediction,
    inter_model_disagreement,
    summarize_errors,
)
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.fusion import FusionResult, plain_average, static_weighted
from lib.landmarks.schema import CANONICAL_SCHEMA, LandmarkPrediction


@dataclass(frozen=True)
class LandmarkSample:
    """One landmark evaluation manifest entry."""

    sample_id: str
    image: str
    landmarks: str
    dataset: str = ""
    condition: str = ""
    normalizer: float | None = None


def load_manifest(path: str | Path) -> list[LandmarkSample]:
    """Load a landmark manifest JSON file."""
    manifest = Path(path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    base = manifest.parent
    samples = []
    for entry in payload.get("samples", payload.get("scenarios", [])):
        landmarks = str(entry.get("landmarks") or entry.get("ground_truth") or "")
        if not landmarks:
            raise ValueError(f"manifest entry {entry!r} missing landmarks path")
        samples.append(
            LandmarkSample(
                sample_id=str(entry.get("sample_id") or entry.get("id") or entry.get("name")),
                image=str((base / str(entry.get("image", ""))).resolve()),
                landmarks=str((base / landmarks).resolve()),
                dataset=str(entry.get("dataset", "")),
                condition=str(entry.get("condition", entry.get("scenario", ""))),
                normalizer=entry.get("normalizer"),
            )
        )
    return samples


def _load_truth(sample: LandmarkSample) -> np.ndarray:
    return np.load(sample.landmarks).astype("float32")


def _append_grouped(
    groups: dict[str, dict[str, list[float]]],
    group_name: str,
    label: str,
    value: float,
) -> None:
    """Append one metric value into a nested report group."""
    if not group_name:
        group_name = "unspecified"
    groups.setdefault(group_name, {}).setdefault(label, []).append(value)


def _scenario_bucket(sample: LandmarkSample) -> str:
    """Return the dataset/scenario bucket used for macro overall scoring."""
    dataset = sample.dataset or "unspecified"
    condition = sample.condition or "unspecified"
    return f"{dataset}:{condition}"


def _summarize_group(
    groups: dict[str, dict[str, list[float]]],
    *,
    failure_threshold: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Summarize grouped NME values."""
    return {
        group: {
            label: summarize_errors(values, failure_threshold=failure_threshold)
            for label, values in sorted(labels.items())
        }
        for group, labels in sorted(groups.items())
    }


def _summarize_bucket_weighted(
    bucket_errors: dict[str, dict[str, list[float]]],
    *,
    failure_threshold: float,
) -> dict[str, dict[str, T.Any]]:
    """Summarize labels using equal-weighted scenario buckets.

    Each dataset/condition bucket contributes one equally weighted score for each
    label, regardless of how many samples are present in that bucket. This avoids
    large or easy datasets hiding failures in smaller hard-condition buckets.
    The returned ``nme`` and ``overall_nme`` are intentionally identical so older
    downstream code that sorts by ``nme`` uses the scenario-bucket aggregate.
    """
    bucket_summaries = _summarize_group(bucket_errors, failure_threshold=failure_threshold)
    labels = sorted({label for values in bucket_errors.values() for label in values})
    output: dict[str, dict[str, T.Any]] = {}
    for label in labels:
        summaries = [
            labels_for_bucket[label]
            for labels_for_bucket in bucket_summaries.values()
            if label in labels_for_bucket
        ]
        if not summaries:
            continue
        output[label] = {
            "count": float(sum(summary["count"] for summary in summaries)),
            "scenario_bucket_count": float(len(summaries)),
            "overall_nme": float(np.mean([summary["nme"] for summary in summaries])),
            "nme": float(np.mean([summary["nme"] for summary in summaries])),
            "auc": float(np.mean([summary["auc"] for summary in summaries])),
            "failure_rate": float(np.mean([summary["failure_rate"] for summary in summaries])),
            "aggregation": "equal_weighted_scenario_buckets",
            "bucket_weighting": "equal_per_dataset_condition",
        }
    return output


def _fuse_variant(
    variant: str,
    predictions: T.Sequence[LandmarkPrediction],
    model_names: T.Sequence[str],
    weights: dict[str, list[float]],
    *,
    outlier_threshold: float,
) -> tuple[FusionResult, int]:
    """Fuse one ensemble variant and return its rejected landmark count."""
    if variant == "plain_average":
        return plain_average(predictions, outlier_method="none"), 0

    matrix = weights_matrix_for_models(weights, tuple(model_names))
    if variant == "static_weighted":
        return static_weighted(predictions, matrix, outlier_method="none"), 0

    variant_methods = {
        "static_weighted_none": "none",
        "static_weighted_outliers": "hard_drop",
        "static_weighted_hard_drop": "hard_drop",
        "static_weighted_downweight": "downweight",
        "weighted_median": "weighted_median",
    }
    if variant in variant_methods:
        fused = static_weighted(
            predictions,
            matrix,
            outlier_threshold=outlier_threshold,
            outlier_method=variant_methods[variant],
        )
        return (
            FusionResult(
                points=fused.points,
                schema=CANONICAL_SCHEMA,
                strategy=variant,
                weights=fused.weights,
                sources=fused.sources,
                kept_indices=fused.kept_indices,
                rejected_indices=fused.rejected_indices,
                rejected_landmarks=fused.rejected_landmarks,
            ),
            fused.rejected_landmarks,
        )

    raise ValueError(f"Unknown ensemble variant '{variant}'")


def run_quality_harness(
    manifest_path: str | Path,
    cache_dir: str | Path,
    *,
    models: T.Sequence[str] | None = None,
    variants: T.Sequence[str] = ("plain_average",),
    weights_path: str | Path | None = None,
    output_dir: str | Path = "outputs/landmark_quality",
    failure_threshold: float = 0.08,
    outlier_threshold: float = 3.5,
) -> dict[str, T.Any]:
    """Evaluate cached model predictions and ensemble variants.

    ``outlier_threshold`` is measured in robust z-score units for outlier-aware
    variants.
    """
    samples = load_manifest(manifest_path)
    cache = DiskPredictionCache(cache_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_weights = load_weights(weights_path) if weights_path else {}
    rows: list[dict[str, T.Any]] = []
    per_landmark_rows: list[dict[str, T.Any]] = []
    per_region_rows: list[dict[str, T.Any]] = []
    ensemble_regressions: list[dict[str, T.Any]] = []
    grouped_errors: dict[str, list[float]] = {}
    dataset_errors: dict[str, dict[str, list[float]]] = {}
    condition_errors: dict[str, dict[str, list[float]]] = {}
    scenario_bucket_errors: dict[str, dict[str, list[float]]] = {}
    region_errors: dict[str, dict[str, list[float]]] = {}
    threshold_failed = False

    for sample in samples:
        truth = _load_truth(sample)
        model_names = tuple(models or cache.available_models(sample.sample_id))
        cached_models = cache.available_models(sample.sample_id)
        missing_models = [name for name in model_names if name not in cached_models]
        if missing_models:
            raise FileNotFoundError(
                f"missing cached predictions for {sample.sample_id}: {missing_models}"
            )
        predictions = {name: cache.read(sample.sample_id, name).landmarks for name in model_names}
        if not predictions:
            raise FileNotFoundError(f"no cached predictions for {sample.sample_id}")
        disagreement = inter_model_disagreement(predictions)
        single_scores: dict[str, float] = {}
        for name, points in predictions.items():
            metrics = evaluate_prediction(
                points,
                truth,
                normalizer=sample.normalizer,
                failure_threshold=failure_threshold,
            )
            threshold_failed = threshold_failed or bool(metrics["failure"])
            grouped_errors.setdefault(name, []).append(float(metrics["nme"]))
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset": sample.dataset,
                    "condition": sample.condition,
                    "model": name,
                    "variant": "single",
                    "nme": metrics["nme"],
                    "failure": metrics["failure"],
                    "disagreement": disagreement.get(name, 0.0),
                    "rejected_landmarks": 0,
                    "best_single_model": "",
                    "best_single_nme": "",
                    "delta_vs_best_single": "",
                }
            )
            single_scores[name] = float(metrics["nme"])
            _append_detail_rows(
                per_landmark_rows,
                per_region_rows,
                sample=sample,
                label=name,
                model=name,
                variant="single",
                metrics=metrics,
            )
            _append_grouped(dataset_errors, sample.dataset, name, float(metrics["nme"]))
            _append_grouped(condition_errors, sample.condition, name, float(metrics["nme"]))
            _append_grouped(
                scenario_bucket_errors,
                _scenario_bucket(sample),
                name,
                float(metrics["nme"]),
            )
            for region, error in metrics["per_region_error"].items():
                _append_grouped(region_errors, region, name, float(error))
        best_single_model = None
        best_single_nme = None
        if single_scores:
            best_single_model, best_single_nme = min(
                single_scores.items(), key=lambda item: item[1]
            )
        prediction_items = [cache.read(sample.sample_id, name) for name in predictions]
        if len(prediction_items) >= 2:
            for variant in variants:
                fused, rejected_count = _fuse_variant(
                    variant,
                    prediction_items,
                    tuple(predictions),
                    static_weights,
                    outlier_threshold=outlier_threshold,
                )
                metrics = evaluate_prediction(
                    fused.points,
                    truth,
                    normalizer=sample.normalizer,
                    failure_threshold=failure_threshold,
                )
                threshold_failed = threshold_failed or bool(metrics["failure"])
                grouped_errors.setdefault(variant, []).append(float(metrics["nme"]))
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "dataset": sample.dataset,
                        "condition": sample.condition,
                        "model": "ensemble",
                        "variant": variant,
                        "nme": metrics["nme"],
                        "failure": metrics["failure"],
                        "disagreement": 0.0,
                        "rejected_landmarks": rejected_count,
                        "best_single_model": best_single_model or "",
                        "best_single_nme": "" if best_single_nme is None else best_single_nme,
                        "delta_vs_best_single": (
                            ""
                            if best_single_nme is None
                            else float(metrics["nme"]) - best_single_nme
                        ),
                    }
                )
                _append_detail_rows(
                    per_landmark_rows,
                    per_region_rows,
                    sample=sample,
                    label=variant,
                    model="ensemble",
                    variant=variant,
                    metrics=metrics,
                )
                if best_single_nme is not None and float(metrics["nme"]) > best_single_nme:
                    ensemble_regressions.append(
                        {
                            "sample_id": sample.sample_id,
                            "dataset": sample.dataset,
                            "condition": sample.condition,
                            "variant": variant,
                            "nme": metrics["nme"],
                            "best_single_model": best_single_model,
                            "best_single_nme": best_single_nme,
                            "delta_vs_best_single": float(metrics["nme"]) - best_single_nme,
                        }
                    )
                _append_grouped(dataset_errors, sample.dataset, variant, float(metrics["nme"]))
                _append_grouped(condition_errors, sample.condition, variant, float(metrics["nme"]))
                _append_grouped(
                    scenario_bucket_errors,
                    _scenario_bucket(sample),
                    variant,
                    float(metrics["nme"]),
                )
                for region, error in metrics["per_region_error"].items():
                    _append_grouped(region_errors, region, variant, float(error))

    pooled_overall = {
        name: summarize_errors(values, failure_threshold=failure_threshold)
        for name, values in sorted(grouped_errors.items())
    }
    scenario_weighted_overall = _summarize_bucket_weighted(
        scenario_bucket_errors,
        failure_threshold=failure_threshold,
    )
    scenario_buckets = _summarize_group(
        scenario_bucket_errors,
        failure_threshold=failure_threshold,
    )
    summary = {
        "overall": scenario_weighted_overall,
        "overall_nme_aggregation": "equal_weighted_dataset_condition_scenario_buckets",
        "overall_pooled": pooled_overall,
        "overall_scenario_weighted": scenario_weighted_overall,
        "scenario_buckets": scenario_buckets,
        "datasets": _summarize_group(dataset_errors, failure_threshold=failure_threshold),
        "conditions": _summarize_group(condition_errors, failure_threshold=failure_threshold),
        "regions": {
            region: {label: float(np.mean(values)) for label, values in sorted(labels.items())}
            for region, labels in sorted(region_errors.items())
        },
        "rows": rows,
        "threshold_failed": threshold_failed,
    }
    best_summary = _best_variant_summary(
        grouped_errors,
        variants=variants,
        failure_threshold=failure_threshold,
        conditions=summary["conditions"],
        threshold_failed=threshold_failed,
        overall_metrics=scenario_weighted_overall,
    )
    summary["best_variant"] = best_summary
    summary["best_single_model"] = best_summary["best_single_model"]
    summary["ensemble_deltas_vs_best_single"] = best_summary["ensemble_deltas_vs_best_single"]
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "sample_id",
            "dataset",
            "condition",
            "model",
            "variant",
            "nme",
            "failure",
            "disagreement",
            "rejected_landmarks",
            "best_single_model",
            "best_single_nme",
            "delta_vs_best_single",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_csv(out_dir / "per_landmark_error.csv", per_landmark_rows)
    _write_csv(out_dir / "per_region_error.csv", per_region_rows)
    _write_condition_csv(out_dir / "per_condition_error.csv", condition_errors, failure_threshold)
    _write_condition_csv(
        out_dir / "per_scenario_bucket_error.csv",
        scenario_bucket_errors,
        failure_threshold,
        condition_column="scenario_bucket",
    )
    _write_csv(
        out_dir / "ensemble_regressions.csv",
        ensemble_regressions,
        fieldnames=[
            "sample_id",
            "dataset",
            "condition",
            "variant",
            "nme",
            "best_single_model",
            "best_single_nme",
            "delta_vs_best_single",
        ],
    )
    (out_dir / "best_variant_summary.json").write_text(
        json.dumps(best_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _append_detail_rows(
    per_landmark_rows: list[dict[str, T.Any]],
    per_region_rows: list[dict[str, T.Any]],
    *,
    sample: LandmarkSample,
    label: str,
    model: str,
    variant: str,
    metrics: dict[str, T.Any],
) -> None:
    """Append per-landmark and per-region metric rows for one prediction."""
    for index, error in enumerate(metrics["per_landmark_error"]):
        per_landmark_rows.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "condition": sample.condition,
                "model": model,
                "variant": variant,
                "label": label,
                "landmark": index,
                "error": error,
            }
        )
    for region, error in metrics["per_region_error"].items():
        per_region_rows.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "condition": sample.condition,
                "model": model,
                "variant": variant,
                "label": label,
                "region": region,
                "error": error,
            }
        )


def _best_variant_summary(
    grouped_errors: dict[str, list[float]],
    *,
    variants: T.Sequence[str],
    failure_threshold: float,
    conditions: dict[str, dict[str, dict[str, float]]],
    threshold_failed: bool,
    overall_metrics: dict[str, dict[str, T.Any]] | None = None,
) -> dict[str, T.Any]:
    """Return the best aggregate model/variant summary."""
    summaries = overall_metrics or {
        label: summarize_errors(values, failure_threshold=failure_threshold)
        for label, values in grouped_errors.items()
        if values
    }
    variant_names = set(variants)
    singles = {
        label: metrics for label, metrics in summaries.items() if label not in variant_names
    }
    ensembles = {label: metrics for label, metrics in summaries.items() if label in variant_names}
    best_single_label, best_single_metrics = min(
        singles.items(),
        key=lambda item: item[1]["nme"],
        default=("", {}),
    )
    best_variant_label, best_variant_metrics = min(
        ensembles.items(),
        key=lambda item: item[1]["nme"],
        default=("", {}),
    )
    deltas = {
        label: metrics["nme"] - best_single_metrics.get("nme", 0.0)
        for label, metrics in sorted(ensembles.items())
        if best_single_label
    }
    if not summaries:
        return {
            "label": "",
            "metrics": {},
            "best_single_model": "",
            "best_single": {},
            "best_variant": "",
            "best_variant_metrics": {},
            "ensemble_deltas_vs_best_single": {},
            "failure_rate_by_condition": {},
            "threshold_failed": threshold_failed,
            "overall_nme_aggregation": "equal_weighted_dataset_condition_scenario_buckets",
        }
    label, metrics = min(summaries.items(), key=lambda item: item[1]["nme"])
    return {
        "label": label,
        "metrics": metrics,
        "best_single_model": best_single_label,
        "best_single": best_single_metrics,
        "best_variant": best_variant_label,
        "best_variant_metrics": best_variant_metrics,
        "ensemble_deltas_vs_best_single": deltas,
        "failure_rate_by_condition": conditions,
        "threshold_failed": threshold_failed,
        "overall_nme_aggregation": "equal_weighted_dataset_condition_scenario_buckets",
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, T.Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    """Write arbitrary dictionaries as CSV."""
    if fieldnames is None and rows:
        fieldnames = sorted({key for row in rows for key in row})
    if fieldnames is None:
        fieldnames = []
    if not rows:
        path.write_text(",".join(fieldnames) + ("\n" if fieldnames else ""), encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_condition_csv(
    path: Path,
    condition_errors: dict[str, dict[str, list[float]]],
    failure_threshold: float,
    *,
    condition_column: str = "condition",
) -> None:
    """Write condition-level summary metrics."""
    rows = []
    for condition, labels in sorted(condition_errors.items()):
        for label, values in sorted(labels.items()):
            rows.append(
                {
                    condition_column: condition,
                    "label": label,
                    **summarize_errors(values, failure_threshold=failure_threshold),
                }
            )
    _write_csv(path, rows)
