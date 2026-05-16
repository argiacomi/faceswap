#!/usr/bin/env python3
"""Evaluate profile-aware alignment metrics for AFLW/profile manifests (#76).

Reads a manifest plus a populated prediction cache, optionally a static-weight
JSON or promoted setup, and writes:

- ``aflw_profile_metrics.json``: per-label aggregates plus per-sample rows
- ``aflw_profile_metrics.csv``: one row per (sample, label) with scalar metrics
- ``aflw_region_failures.csv``: one row per (sample, label, region) with the
  region failure flag

The harness is cache-only: it never invokes landmark adapters. Faceswap
manifests must include a face bbox per sample (AFLW2000-3D already populates
``metadata.face_bbox`` from landmark extrema); samples without a usable bbox
are skipped with an audit row so downstream regression tracking stays honest.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import typing as T
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.ensemble.weights import (
    load_weights,
    weights_matrix_for_models,
)
from lib.landmarks.eval.harness import LandmarkSample, load_manifest
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.eval.profile_metrics import (
    DEFAULT_NORMALIZER,
    DEFAULT_PCK_THRESHOLDS,
    DEFAULT_PRIORITY_FAILURE_REGIONS,
    DEFAULT_REGION_FAILURE_THRESHOLD,
    DEFAULT_REGION_WEIGHTS,
    NORMALIZERS,
    PROFILE_OBJECTIVE,
    REGION_INDICES,
    ProfileAggregate,
    ProfileSampleMetrics,
    aggregate_profile_samples,
    evaluate_profile_sample,
)
from lib.landmarks.fusion import (
    FusionResult,
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.rejection import weighted_median
from lib.landmarks.schema import CANONICAL_SCHEMA, LandmarkPrediction

logger = logging.getLogger(__name__)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_pck_thresholds(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _fuse_variant(
    variant: str,
    predictions: T.Sequence[LandmarkPrediction],
    models: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]] | None,
    *,
    outlier_threshold: float,
) -> np.ndarray:
    """Return fused 68×2 points for one variant of cached predictions."""
    canonical = canonical_strategy(variant)
    method = strategy_outlier_method(canonical)
    threshold = outlier_threshold if strategy_uses_threshold(canonical) else 3.5

    if not strategy_requires_weights(canonical):
        return plain_average(predictions, outlier_method=method, outlier_threshold=threshold).points

    if weights is None:
        raise ValueError(f"variant {variant!r} requires a static weights file")
    matrix = weights_matrix_for_models(weights, tuple(models))
    if canonical == "weighted_median":
        stack = np.stack(
            [prediction.canonical_68().points for prediction in predictions], axis=0
        )
        normalized = normalize_weight_matrix(
            matrix, model_count=stack.shape[0], landmark_count=stack.shape[1]
        )
        return weighted_median(stack, normalized)
    return static_weighted(
        predictions,
        matrix,
        outlier_method=method,
        outlier_threshold=threshold,
    ).points


def _load_truth(sample: LandmarkSample) -> np.ndarray:
    return np.load(sample.landmarks).astype("float32")


def _bbox(sample: LandmarkSample) -> tuple[float, float, float, float] | None:
    """Return the sample bbox, defaulting to landmark extrema when missing."""
    if sample.face_bbox is not None:
        return sample.face_bbox
    try:
        truth = _load_truth(sample)
    except Exception:
        return None
    left, top = np.min(truth, axis=0)
    right, bottom = np.max(truth, axis=0)
    return (float(left), float(top), float(right), float(bottom))


def _evaluate_label(
    sample: LandmarkSample,
    points: np.ndarray,
    *,
    label: str,
    normalizer_method: str,
    region_weights: T.Mapping[str, float],
    region_failure_threshold: float,
    priority_failure_regions: T.Sequence[str],
    pck_thresholds: T.Sequence[float],
) -> ProfileSampleMetrics | None:
    """Wrap ``evaluate_profile_sample`` with a skip-on-missing-bbox guard."""
    bbox = _bbox(sample)
    if bbox is None:
        logger.warning(
            "[profile] skipping %s: no face_bbox available for normalization", sample.sample_id
        )
        return None
    truth = _load_truth(sample)
    return evaluate_profile_sample(
        points,
        truth,
        sample_id=sample.sample_id,
        face_bbox=bbox,
        visibility=sample.visibility,
        normalizer_method=normalizer_method,
        region_weights=region_weights,
        region_failure_threshold=region_failure_threshold,
        priority_failure_regions=priority_failure_regions,
        pck_thresholds=pck_thresholds,
    )


def evaluate_manifest(
    manifest_path: str | Path,
    cache_dir: str | Path,
    *,
    models: T.Sequence[str],
    variants: T.Sequence[str] = (),
    weights_path: str | Path | None = None,
    outlier_threshold: float = 3.5,
    normalizer_method: str = DEFAULT_NORMALIZER,
    region_weights: T.Mapping[str, float] = DEFAULT_REGION_WEIGHTS,
    region_failure_threshold: float = DEFAULT_REGION_FAILURE_THRESHOLD,
    priority_failure_regions: T.Sequence[str] = DEFAULT_PRIORITY_FAILURE_REGIONS,
    pck_thresholds: T.Sequence[float] = DEFAULT_PCK_THRESHOLDS,
) -> dict[str, T.Any]:
    """Evaluate every model + ensemble variant against the profile objective.

    Returns a JSON-ready payload with per-label aggregates and per-sample rows.
    Single models are always evaluated; ``variants`` adds optional ensemble
    fusion variants on top.
    """
    samples = load_manifest(manifest_path)
    cache = DiskPredictionCache(cache_dir)
    weights = load_weights(weights_path) if weights_path else None
    per_label_samples: dict[str, list[ProfileSampleMetrics]] = {}
    rows: list[dict[str, T.Any]] = []
    skipped: list[str] = []

    for sample in samples:
        # Surface adapter availability up-front for a clear error.
        available = cache.available_models(sample.sample_id)
        missing = [name for name in models if name not in available]
        if missing:
            raise FileNotFoundError(
                f"sample {sample.sample_id!r} is missing cached predictions for {missing}"
            )

        predictions = {name: cache.read(sample.sample_id, name) for name in models}
        prediction_items = [predictions[name] for name in models]

        for model_name in models:
            metrics = _evaluate_label(
                sample,
                predictions[model_name].landmarks,
                label=model_name,
                normalizer_method=normalizer_method,
                region_weights=region_weights,
                region_failure_threshold=region_failure_threshold,
                priority_failure_regions=priority_failure_regions,
                pck_thresholds=pck_thresholds,
            )
            if metrics is None:
                skipped.append(sample.sample_id)
                continue
            per_label_samples.setdefault(model_name, []).append(metrics)
            rows.append(_row_from_metrics(sample, model_name, "single", metrics))

        for variant in variants:
            fused_points = _fuse_variant(
                variant,
                prediction_items,
                models,
                weights,
                outlier_threshold=outlier_threshold,
            )
            metrics = _evaluate_label(
                sample,
                fused_points,
                label=variant,
                normalizer_method=normalizer_method,
                region_weights=region_weights,
                region_failure_threshold=region_failure_threshold,
                priority_failure_regions=priority_failure_regions,
                pck_thresholds=pck_thresholds,
            )
            if metrics is None:
                continue
            per_label_samples.setdefault(variant, []).append(metrics)
            rows.append(_row_from_metrics(sample, "ensemble", variant, metrics))

    aggregates: dict[str, ProfileAggregate] = {
        label: aggregate_profile_samples(
            label,
            samples_list,
            region_weights=region_weights,
            priority_failure_regions=priority_failure_regions,
            pck_thresholds=pck_thresholds,
        )
        for label, samples_list in per_label_samples.items()
    }

    best_single = min(
        (
            agg
            for label, agg in aggregates.items()
            if label in models
        ),
        key=lambda agg: agg.overall_score,
        default=None,
    )
    regression_rate_vs_best_single: dict[str, float] = {}
    if best_single is not None:
        baseline = best_single.overall_score
        for label, agg in aggregates.items():
            regression_rate_vs_best_single[label] = float(
                max(agg.overall_score - baseline, 0.0)
            )

    return {
        "objective": PROFILE_OBJECTIVE,
        "normalizer_method": normalizer_method,
        "pck_thresholds": list(pck_thresholds),
        "region_failure_threshold": region_failure_threshold,
        "priority_failure_regions": list(priority_failure_regions),
        "region_weights": {key: float(value) for key, value in region_weights.items()},
        "best_single_label": best_single.label if best_single else "",
        "best_single_overall_score": (
            float(best_single.overall_score) if best_single is not None else 0.0
        ),
        "aggregates": {label: _aggregate_payload(agg) for label, agg in aggregates.items()},
        "regression_rate_vs_best_single": regression_rate_vs_best_single,
        "rows": rows,
        "skipped_sample_ids": skipped,
    }


def _aggregate_payload(aggregate: ProfileAggregate) -> dict[str, T.Any]:
    return {
        "label": aggregate.label,
        "sample_count": aggregate.sample_count,
        "overall_score": aggregate.overall_score,
        "weighted_region_score": aggregate.weighted_region_score,
        "region_failure_rate": aggregate.region_failure_rate,
        "p90_visible_error": aggregate.p90_visible_error,
        "pck_at": dict(aggregate.pck_at),
        "per_region_error": dict(aggregate.per_region_error),
        "per_region_failure_rate": dict(aggregate.per_region_failure_rate),
    }


def _row_from_metrics(
    sample: LandmarkSample,
    model: str,
    variant: str,
    metrics: ProfileSampleMetrics,
) -> dict[str, T.Any]:
    row = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "condition": sample.condition,
        "model": model,
        "variant": variant,
        "normalizer": metrics.normalizer,
        "visible_landmark_count": metrics.visible_landmark_count,
        "weighted_region_score": metrics.weighted_region_score,
        "p90_visible_error": metrics.p90_visible_error,
        "overall_score": metrics.overall_score,
    }
    for region, value in metrics.per_region_error.items():
        row[f"region_error_{region}"] = value
        row[f"region_failure_{region}"] = metrics.region_failures.get(region, False)
    for threshold, value in metrics.pck_at.items():
        row[f"pck@{threshold}"] = value
    return row


def _write_outputs(output_dir: Path, payload: dict[str, T.Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "aflw_profile_metrics.json"
    csv_path = output_dir / "aflw_profile_metrics.csv"
    failures_path = output_dir / "aflw_region_failures.csv"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = payload["rows"]
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
    else:
        fieldnames = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failure_rows: list[dict[str, T.Any]] = []
    for row in rows:
        for region in REGION_INDICES:
            failure_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "model": row["model"],
                    "variant": row["variant"],
                    "region": region,
                    "region_error": row.get(f"region_error_{region}", 0.0),
                    "region_failed": row.get(f"region_failure_{region}", False),
                }
            )
    failure_fieldnames = [
        "sample_id",
        "model",
        "variant",
        "region",
        "region_error",
        "region_failed",
    ]
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument(
        "--variants",
        default="",
        help="Optional ensemble fusion variants to score alongside single models.",
    )
    parser.add_argument("--weights", default="")
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument(
        "--normalizer",
        default=DEFAULT_NORMALIZER,
        choices=NORMALIZERS,
        help="Face-bbox normalizer for point error; default is the diagonal.",
    )
    parser.add_argument(
        "--region-failure-threshold",
        type=float,
        default=DEFAULT_REGION_FAILURE_THRESHOLD,
    )
    parser.add_argument(
        "--pck-thresholds",
        default=",".join(f"{t:.2f}" for t in DEFAULT_PCK_THRESHOLDS),
    )
    parser.add_argument(
        "--priority-failure-regions",
        default=",".join(DEFAULT_PRIORITY_FAILURE_REGIONS),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    payload = evaluate_manifest(
        args.manifest,
        args.cache_dir,
        models=_parse_csv(args.models),
        variants=_parse_csv(args.variants),
        weights_path=args.weights or None,
        outlier_threshold=args.outlier_threshold,
        normalizer_method=args.normalizer,
        region_failure_threshold=args.region_failure_threshold,
        pck_thresholds=_parse_pck_thresholds(args.pck_thresholds),
        priority_failure_regions=_parse_csv(args.priority_failure_regions),
    )
    _write_outputs(Path(args.output_dir), payload)
    print(
        f"Wrote profile metrics for {len(payload['aggregates'])} labels "
        f"to {args.output_dir}"
    )
    if payload["best_single_label"]:
        print(
            f"  best single: {payload['best_single_label']} "
            f"(score={payload['best_single_overall_score']:.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
