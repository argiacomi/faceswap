#!/usr/bin/env python3
"""Cache-driven landmark quality harness."""

from __future__ import annotations

import csv
import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.outliers import reject_outliers, weighted_median
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
        return plain_average(predictions), 0

    matrix = weights_matrix_for_models(weights, tuple(model_names))
    if variant == "static_weighted":
        return static_weighted(predictions, matrix), 0

    stack = np.stack([prediction.canonical_68().points for prediction in predictions], axis=0)
    method = "hard_drop"
    if variant == "static_weighted_downweight":
        method = "downweight"
    if variant in {"static_weighted_outliers", "static_weighted_downweight"}:
        rejection = reject_outliers(
            stack,
            matrix,
            model_names=model_names,
            threshold=outlier_threshold,
            method=method,
        )
        points = (stack * rejection.weights[..., None]).sum(axis=0).astype("float32")
        return (
            FusionResult(
                points=points,
                schema=CANONICAL_SCHEMA,
                strategy=variant,
                weights=rejection.weights,
                sources=tuple(model_names),
                kept_indices=tuple(range(len(model_names))),
            ),
            len(rejection.rejected),
        )

    if variant == "weighted_median":
        points = weighted_median(stack, matrix)
        return (
            FusionResult(
                points=points,
                schema=CANONICAL_SCHEMA,
                strategy=variant,
                weights=matrix,
                sources=tuple(model_names),
                kept_indices=tuple(range(len(model_names))),
            ),
            0,
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
    """Evaluate cached model predictions and ensemble variants."""
    samples = load_manifest(manifest_path)
    cache = DiskPredictionCache(cache_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_weights = load_weights(weights_path) if weights_path else {}
    rows: list[dict[str, T.Any]] = []
    grouped_errors: dict[str, list[float]] = {}
    dataset_errors: dict[str, dict[str, list[float]]] = {}
    condition_errors: dict[str, dict[str, list[float]]] = {}
    region_errors: dict[str, dict[str, list[float]]] = {}

    for sample in samples:
        truth = _load_truth(sample)
        model_names = tuple(models or cache.available_models(sample.sample_id))
        predictions = {
            name: cache.read(sample.sample_id, name).landmarks
            for name in model_names
            if name in cache.available_models(sample.sample_id)
        }
        disagreement = inter_model_disagreement(predictions)
        for name, points in predictions.items():
            metrics = evaluate_prediction(
                points,
                truth,
                normalizer=sample.normalizer,
                failure_threshold=failure_threshold,
            )
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
                }
            )
            _append_grouped(dataset_errors, sample.dataset, name, float(metrics["nme"]))
            _append_grouped(condition_errors, sample.condition, name, float(metrics["nme"]))
            for region, error in metrics["per_region_error"].items():
                _append_grouped(region_errors, region, name, float(error))
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
                    }
                )
                _append_grouped(dataset_errors, sample.dataset, variant, float(metrics["nme"]))
                _append_grouped(condition_errors, sample.condition, variant, float(metrics["nme"]))
                for region, error in metrics["per_region_error"].items():
                    _append_grouped(region_errors, region, variant, float(error))

    summary = {
        "overall": {
            name: summarize_errors(values, failure_threshold=failure_threshold)
            for name, values in sorted(grouped_errors.items())
        },
        "datasets": _summarize_group(dataset_errors, failure_threshold=failure_threshold),
        "conditions": _summarize_group(condition_errors, failure_threshold=failure_threshold),
        "regions": {
            region: {label: float(np.mean(values)) for label, values in sorted(labels.items())}
            for region, labels in sorted(region_errors.items())
        },
        "rows": rows,
    }
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
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary
