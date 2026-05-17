#!/usr/bin/env python3
"""GT-geometry signal validation report (#80).

Reads a manifest + populated prediction cache, evaluates every candidate
(single models + named ensemble variants when weights are supplied) against
GT-derived ``AlignedFace`` geometry, and writes:

- ``candidate_index.csv``           — one row per (sample, candidate)
- ``signal_validation_report.json`` — precision/recall/AUC per signal
- ``selector_ablations.csv``        — oracle-match rates per selector
- ``signal_validation_report.csv``  — same as above in flat CSV form
- ``selector_ablations.json``       — same selector data as JSON

The report answers two questions:

1. Which signal best identifies candidates that deviate from the per-sample
   oracle's GT-derived geometry?
2. Which selector picks the oracle's choice most often, on this manifest?
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

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.evaluation.geometry_metrics import evaluate_geometry_sample
from lib.landmarks.evaluation.geometry_signals import alignment_summary
from lib.landmarks.evaluation.harness import LandmarkSample, load_manifest
from lib.landmarks.evaluation.nme_metrics import evaluate_prediction
from lib.landmarks.evaluation.signal_validation import (
    CandidateRecord,
    candidate_record_from_geometry,
    evaluate_selectors,
    evaluate_signals,
    tag_oracle,
)

logger = logging.getLogger(__name__)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bbox_for_sample(sample: LandmarkSample) -> tuple[float, float, float, float] | None:
    """Resolve a usable bbox via the canonical manifest helper."""
    from lib.landmarks.datasets.manifest_io import bbox_for_sample

    return bbox_for_sample(sample, allow_truth_fallback=True)


# Canonical fusion helper now lives in lib.landmarks.core.fusion_variants.
from lib.landmarks.core.fusion_variants import fuse_variant as _fuse_variant_impl


def _fuse_variant(
    variant: str,
    predictions: T.Sequence[LandmarkPrediction],
    models: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]] | None,
    *,
    outlier_threshold: float,
) -> np.ndarray:
    """Compatibility shim delegating to :func:`lib.landmarks.core.fusion_variants.fuse_variant`."""
    return _fuse_variant_impl(
        variant,
        predictions,
        models=models,
        weights=weights,
        outlier_threshold=outlier_threshold,
    )


def build_candidate_records(
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str],
    variants: T.Sequence[str] = (),
    weights_path: str | Path | None = None,
    outlier_threshold: float = 3.5,
    aligned_size: int = 512,
    region_failure_threshold: float = 0.05,
) -> list[CandidateRecord]:
    """Run every candidate per sample and return the per-row records."""
    samples = load_manifest(manifest_path)
    cache = DiskPredictionCache(cache_dir)
    weights = load_weights(weights_path) if weights_path else None
    records: list[CandidateRecord] = []
    for sample in samples:
        bbox = _bbox_for_sample(sample)
        if bbox is None:
            logger.warning("[signals] skipping %s: no usable bbox", sample.sample_id)
            continue
        truth = np.load(sample.landmarks).astype("float32")
        # Build the GT-side AlignedFace summary once per sample so every
        # downstream candidate (single models + ensemble variants) skips
        # the redundant Umeyama + solvePnP pass on the same GT cloud.
        truth_summary = alignment_summary(truth, size=aligned_size)
        predictions = {name: cache.read(sample.sample_id, name) for name in models}
        prediction_items = [predictions[name] for name in models]
        hard_slice = sample.condition  # build_hard_alignment_validation copies this
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
            nme = float(
                evaluate_prediction(
                    predictions[model_name].landmarks,
                    truth,
                    normalizer=sample.normalizer,
                )["nme"]
            )
            records.append(
                candidate_record_from_geometry(
                    metrics,
                    candidate_label=model_name,
                    nme=nme,
                    is_baseline=True,
                    hard_slice=hard_slice,
                )
            )
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
            nme = float(evaluate_prediction(fused, truth, normalizer=sample.normalizer)["nme"])
            records.append(
                candidate_record_from_geometry(
                    metrics,
                    candidate_label=variant,
                    nme=nme,
                    is_baseline=False,
                    hard_slice=hard_slice,
                )
            )
    return tag_oracle(records)


def _write_candidate_index(path: Path, records: T.Sequence[CandidateRecord]) -> None:
    fieldnames = [
        "sample_id",
        "dataset",
        "condition",
        "hard_slice",
        "candidate_label",
        "is_baseline",
        "is_oracle",
        "geometry_score",
        "nme",
        "transform_normalized",
        "crop_center_normalized",
        "roll_degrees_delta",
        "hull_iou",
        "catastrophic",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "dataset": record.dataset,
                    "condition": record.condition,
                    "hard_slice": record.hard_slice,
                    "candidate_label": record.candidate_label,
                    "is_baseline": record.is_baseline,
                    "is_oracle": record.is_oracle,
                    "geometry_score": record.geometry_score,
                    "nme": record.nme,
                    "transform_normalized": record.transform_normalized,
                    "crop_center_normalized": record.crop_center_normalized,
                    "roll_degrees_delta": record.roll_degrees_delta,
                    "hull_iou": record.hull_iou,
                    "catastrophic": record.catastrophic,
                }
            )


def _write_signal_report(path_json: Path, path_csv: Path, reports: list, margin: float) -> None:
    payload = {
        "bad_candidate_margin": float(margin),
        "signals": [r.to_payload() for r in reports],
    }
    path_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with path_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["name", "direction", "threshold", "precision", "recall", "auc"]
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "name": report.name,
                    "direction": report.direction,
                    "threshold": report.threshold,
                    "precision": report.precision,
                    "recall": report.recall,
                    "auc": report.auc,
                }
            )


def _write_selector_report(path_json: Path, path_csv: Path, reports: list) -> None:
    payload = {"selectors": [r.to_payload() for r in reports]}
    path_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with path_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "sample_count",
                "oracle_match_rate",
                "mean_score_gap_vs_oracle",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "name": report.name,
                    "sample_count": report.sample_count,
                    "oracle_match_rate": report.oracle_match_rate,
                    "mean_score_gap_vs_oracle": report.mean_score_gap_vs_oracle,
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument("--variants", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument("--aligned-size", type=int, default=512)
    parser.add_argument("--region-failure-threshold", type=float, default=0.05)
    parser.add_argument(
        "--bad-candidate-margin",
        type=float,
        default=0.05,
        help="How much a candidate must exceed the per-sample oracle to count as bad.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.75,
        help="Per-signal threshold quantile used to compute precision/recall.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    records = build_candidate_records(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        models=_parse_csv(args.models),
        variants=_parse_csv(args.variants),
        weights_path=args.weights or None,
        outlier_threshold=args.outlier_threshold,
        aligned_size=args.aligned_size,
        region_failure_threshold=args.region_failure_threshold,
    )
    if not records:
        logger.error("[signals] no candidate records produced; check manifest + cache")
        return 1

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_candidate_index(output / "candidate_index.csv", records)

    signal_reports = evaluate_signals(
        records,
        margin=args.bad_candidate_margin,
        threshold_quantile=args.threshold_quantile,
    )
    selector_reports = evaluate_selectors(records)

    _write_signal_report(
        output / "signal_validation_report.json",
        output / "signal_validation_report.csv",
        signal_reports,
        args.bad_candidate_margin,
    )
    _write_selector_report(
        output / "selector_ablations.json",
        output / "selector_ablations.csv",
        selector_reports,
    )

    best_signal = max(signal_reports, key=lambda r: r.auc, default=None)
    best_selector = max(selector_reports, key=lambda r: r.oracle_match_rate, default=None)
    if best_signal:
        print(
            f"Best signal: {best_signal.name} (AUC={best_signal.auc:.4f}, "
            f"precision={best_signal.precision:.4f}, recall={best_signal.recall:.4f})"
        )
    if best_selector:
        print(
            f"Best selector: {best_selector.name} "
            f"(oracle_match_rate={best_selector.oracle_match_rate:.4f}, "
            f"mean_gap={best_selector.mean_score_gap_vs_oracle:.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
