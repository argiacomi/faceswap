#!/usr/bin/env python3
"""Evaluate GT-derived alignment-geometry metrics for landmark predictions (#76).

Reads a manifest and a populated prediction cache and writes:

- ``geometry_metrics.json``  — per-label aggregates + per-sample rows
- ``geometry_metrics.csv``   — one row per (sample, label)
- ``per_region_geometry.csv``— one row per (sample, label, region)
- ``catastrophic_geometry_failures.csv`` — flagged samples only
- ``worst_geometry_failures/worst_samples.json`` — top-N worst samples per
  failure mode (image rendering / contact-sheet pass is a follow-up)

The harness is cache-only: it never invokes landmark adapters. Optional
ensemble fusion variants can be scored alongside the single models if a
static-weight JSON is supplied.
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

from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.eval.geometry_metrics import (
    GEOMETRY_OBJECTIVE,
    REGION_DEFINITIONS,
    GeometrySampleMetrics,
    aggregate_geometry_samples,
    evaluate_geometry_sample,
)
from lib.landmarks.eval.geometry_signals import alignment_summary
from lib.landmarks.eval.harness import LandmarkSample, load_manifest
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction

logger = logging.getLogger(__name__)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


# Canonical fusion helper now lives in lib.landmarks.search.fusion_variants.
from lib.landmarks.search.fusion_variants import fuse_variant as _fuse_variant_impl


def _fuse_variant(
    variant: str,
    predictions: T.Sequence[LandmarkPrediction],
    models: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]] | None,
    *,
    outlier_threshold: float,
) -> np.ndarray:
    """Return fused 68×2 points for a variant of cached predictions.

    Thin compatibility shim — the canonical implementation lives in
    :mod:`lib.landmarks.search.fusion_variants`.
    """
    return _fuse_variant_impl(
        variant,
        predictions,
        models=models,
        weights=weights,
        outlier_threshold=outlier_threshold,
    )


def _bbox_for_sample(sample: LandmarkSample) -> tuple[float, float, float, float] | None:
    """Resolve a usable bbox via the canonical manifest helper."""
    from lib.landmarks.manifest import bbox_for_sample

    return bbox_for_sample(sample, allow_truth_fallback=True)


def _truth_landmarks(sample: LandmarkSample) -> np.ndarray:
    return np.load(sample.landmarks).astype("float32")


def evaluate_manifest(
    manifest_path: str | Path,
    cache_dir: str | Path,
    *,
    models: T.Sequence[str],
    variants: T.Sequence[str] = (),
    weights_path: str | Path | None = None,
    outlier_threshold: float = 3.5,
    aligned_size: int = 512,
    region_failure_threshold: float = 0.05,
) -> dict[str, T.Any]:
    """Evaluate every requested model + ensemble variant against GT geometry."""
    samples = load_manifest(manifest_path)
    cache = DiskPredictionCache(cache_dir)
    weights = load_weights(weights_path) if weights_path else None

    per_label_samples: dict[str, list[GeometrySampleMetrics]] = {}
    rows: list[dict[str, T.Any]] = []
    skipped: list[str] = []

    for sample in samples:
        available = cache.available_models(sample.sample_id)
        missing = [name for name in models if name not in available]
        if missing:
            raise FileNotFoundError(
                f"sample {sample.sample_id!r} is missing cached predictions for {missing}"
            )

        bbox = _bbox_for_sample(sample)
        if bbox is None:
            logger.warning(
                "[geometry] skipping %s: no face_bbox and no truth file", sample.sample_id
            )
            skipped.append(sample.sample_id)
            continue
        truth = _truth_landmarks(sample)
        # Build the GT-side AlignedFace summary once per sample; every
        # candidate evaluated below reuses it instead of re-running Umeyama
        # + solvePnP on the same GT cloud.
        truth_summary = alignment_summary(truth, size=aligned_size)
        predictions = {name: cache.read(sample.sample_id, name) for name in models}
        prediction_items = [predictions[name] for name in models]

        for model_name in models:
            metrics = evaluate_geometry_sample(
                predictions[model_name].landmarks,
                truth,
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                condition=sample.condition,
                bbox=bbox,
                visibility=sample.visibility,
                aligned_size=aligned_size,
                region_failure_threshold=region_failure_threshold,
                truth_summary=truth_summary,
            )
            per_label_samples.setdefault(model_name, []).append(metrics)
            rows.append(_csv_row(metrics, model=model_name, variant="single"))

        for variant in variants:
            fused = _fuse_variant(
                variant,
                prediction_items,
                models,
                weights,
                outlier_threshold=outlier_threshold,
            )
            metrics = evaluate_geometry_sample(
                fused,
                truth,
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                condition=sample.condition,
                bbox=bbox,
                visibility=sample.visibility,
                aligned_size=aligned_size,
                region_failure_threshold=region_failure_threshold,
                truth_summary=truth_summary,
            )
            per_label_samples.setdefault(variant, []).append(metrics)
            rows.append(_csv_row(metrics, model="ensemble", variant=variant))

    aggregates = {
        label: aggregate_geometry_samples(label, samples_list)
        for label, samples_list in per_label_samples.items()
    }
    best_single = min(
        (agg for label, agg in aggregates.items() if label in models),
        key=lambda agg: agg.overall_score,
        default=None,
    )

    return {
        "objective": GEOMETRY_OBJECTIVE,
        "aligned_size": int(aligned_size),
        "region_failure_threshold": float(region_failure_threshold),
        "best_single_label": best_single.label if best_single else "",
        "best_single_overall_score": (
            float(best_single.overall_score) if best_single is not None else 0.0
        ),
        "aggregates": {label: agg.to_payload() for label, agg in aggregates.items()},
        "regression_vs_best_single": (
            {
                label: max(agg.overall_score - best_single.overall_score, 0.0)
                for label, agg in aggregates.items()
            }
            if best_single is not None
            else {}
        ),
        "rows": rows,
        "per_label_samples": {
            label: [metrics.to_payload() for metrics in samples_list]
            for label, samples_list in per_label_samples.items()
        },
        "skipped_sample_ids": skipped,
    }


def _csv_row(metrics: GeometrySampleMetrics, *, model: str, variant: str) -> dict[str, T.Any]:
    row = {
        "sample_id": metrics.sample_id,
        "dataset": metrics.dataset,
        "condition": metrics.condition,
        "model": model,
        "variant": variant,
        "overall_score": metrics.overall_score,
        "catastrophic": metrics.catastrophic_flags.any,
        "cloud_collapse": metrics.catastrophic_flags.cloud_collapse,
        "eye_mouth_flip": metrics.catastrophic_flags.eye_mouth_flip,
        "points_outside_bbox": metrics.points_outside_bbox,
        "scale_delta": metrics.matrix_delta.scale_delta,
        "relative_scale_delta": metrics.relative_scale_delta,
        "rotation_degrees_delta": metrics.matrix_delta.rotation_degrees_delta,
        "translation_normalized": metrics.matrix_delta.translation_normalized_distance,
        "roi_iou": metrics.roi_delta.iou,
        "roi_center_normalized": metrics.roi_delta.center_normalized_distance,
        "hull_iou": metrics.hull_iou,
        "pitch_delta_degrees": metrics.pose_delta.pitch_delta_degrees,
        "yaw_delta_degrees": metrics.pose_delta.yaw_delta_degrees,
        "roll_delta_degrees": metrics.pose_delta.roll_delta_degrees,
        "average_distance_delta": metrics.average_distance_delta,
    }
    for region, value in metrics.per_region_error.items():
        row[f"region_error_{region}"] = value
        row[f"region_failure_{region}"] = metrics.per_region_failure.get(region, False)
    if metrics.roi_diagnostics is not None:
        roi = metrics.roi_diagnostics
        row.update(
            {
                "bbox_source": roi.bbox_source,
                "bbox_aspect_ratio": roi.bbox_aspect_ratio,
                "bbox_diagonal": roi.bbox_diagonal,
                "aligned_roi_aspect_ratio": roi.aligned_roi_aspect_ratio,
                "aligned_crop_visible_hull_iou": roi.aligned_crop_visible_hull_iou,
                "landmarks_inside_aligned_crop_fraction": (
                    roi.landmarks_inside_aligned_crop_fraction
                ),
                "landmarks_outside_detector_bbox_fraction": (
                    roi.landmarks_outside_detector_bbox_fraction
                ),
                "aligned_crop_misses_visible_face": roi.aligned_crop_misses_visible_face,
            }
        )
    return row


def _write_outputs(output_dir: Path, payload: dict[str, T.Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometry_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = payload["rows"]
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with (output_dir / "geometry_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    region_rows: list[dict[str, T.Any]] = []
    for row in rows:
        for region in REGION_DEFINITIONS:
            region_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row.get("dataset", ""),
                    "condition": row.get("condition", ""),
                    "model": row["model"],
                    "variant": row["variant"],
                    "region": region,
                    "region_error": row.get(f"region_error_{region}", 0.0),
                    "region_failed": row.get(f"region_failure_{region}", False),
                }
            )
    with (output_dir / "per_region_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "dataset",
                "condition",
                "model",
                "variant",
                "region",
                "region_error",
                "region_failed",
            ],
        )
        writer.writeheader()
        writer.writerows(region_rows)

    catastrophic_rows = [row for row in rows if row.get("catastrophic")]
    with (output_dir / "catastrophic_geometry_failures.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catastrophic_rows)

    # Worst-failure index (image contact sheets are a follow-up).
    worst_dir = output_dir / "worst_geometry_failures"
    worst_dir.mkdir(parents=True, exist_ok=True)
    worst_payload = _worst_failures_index(payload, limit=20)
    (worst_dir / "worst_samples.json").write_text(
        json.dumps(worst_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _worst_failures_index(payload: dict[str, T.Any], *, limit: int = 20) -> dict[str, T.Any]:
    """Return a JSON-friendly index of the worst samples per failure mode."""
    by_label: dict[str, list[dict[str, T.Any]]] = {}
    for label, sample_rows in payload.get("per_label_samples", {}).items():
        ranked = sorted(sample_rows, key=lambda row: row.get("overall_score", 0.0), reverse=True)
        by_label[label] = ranked[:limit]
    return {
        "objective": payload.get("objective"),
        "limit": int(limit),
        "by_label": by_label,
    }


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
    parser.add_argument("--aligned-size", type=int, default=512)
    parser.add_argument("--region-failure-threshold", type=float, default=0.05)
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
        aligned_size=args.aligned_size,
        region_failure_threshold=args.region_failure_threshold,
    )
    output_dir = Path(args.output_dir)
    _write_outputs(output_dir, payload)
    print(
        f"Wrote alignment-geometry metrics for {len(payload['aggregates'])} labels to {output_dir}"
    )
    if payload["best_single_label"]:
        print(
            f"  best single: {payload['best_single_label']} "
            f"(score={payload['best_single_overall_score']:.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
