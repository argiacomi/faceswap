#!/usr/bin/env python3
"""Backfill image-aware runtime resolver metadata into a landmark manifest."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import typing as T
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.datasets.manifest_io import (
    LandmarkSample,
    bbox_from_truth_fallback,
    load_manifest,
)
from lib.landmarks.ensemble.promoted_setup import load_promoted_setup
from lib.landmarks.ensemble.runtime_resolver import ModelPrediction, RuntimeResolverConfig
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    image_aware_runtime_result,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("backfill_runtime_resolver_metadata")


def _trace(message: str, *args: T.Any) -> None:
    """Log at Faceswap TRACE level when available."""
    trace = getattr(logger, "trace", None)
    if callable(trace):
        trace(message, *args)


def _json_safe(value: T.Any) -> T.Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _models(value: str | T.Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value)


def _read_predictions(
    cache: DiskPredictionCache,
    sample: LandmarkSample,
    *,
    models: T.Sequence[str],
) -> list[ModelPrediction]:
    available = set(cache.available_models(sample.sample_id))
    missing = [model for model in models if model not in available]
    if missing:
        raise FileNotFoundError(f"sample {sample.sample_id} missing cached models: {missing}")
    return [
        ModelPrediction(model, cache.read(sample.sample_id, model).landmarks) for model in models
    ]


def _runtime_config(
    *,
    setup_path: Path,
    weights_path: Path,
) -> tuple[RuntimeResolverConfig, float]:
    setup = load_promoted_setup(setup_path, load_weights=False)
    weights = load_weights(weights_path)
    return (
        RuntimeResolverConfig(
            policy="roll_aware_veto",
            weights=weights,
            general_strategy=setup.strategy,
            hard_case_strategy="static_weighted_downweight",
            secondary_hard_case_strategy="static_weighted_hard_drop",
            fallback_strategy="plain_average",
            outlier_threshold=3.5 if setup.outlier_threshold is None else setup.outlier_threshold,
        ),
        setup.crop_scale,
    )


def _bbox_for_sample(sample: LandmarkSample) -> tuple[float, float, float, float]:
    if sample.face_bbox is not None:
        return sample.face_bbox
    truth = np.load(sample.landmarks).astype("float32")
    bbox = bbox_from_truth_fallback(truth)
    if bbox is None:
        raise ValueError(f"sample {sample.sample_id} has no usable face bbox")
    return bbox


def _backfilled_metadata(metadata: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    payload = dict(metadata)
    payload["runtime_bucket_source"] = "image_aware_backfill"
    payload["bucket"] = payload.get("runtime_bucket") or payload.get("bucket")
    payload.setdefault("candidate_scores", {})
    return _json_safe(payload)


def backfill_runtime_resolver_metadata(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    setup_path: Path,
    output_path: Path,
    models: T.Sequence[str],
    crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
) -> dict[str, T.Any]:
    """Write a copy of ``manifest_path`` with image-aware runtime metadata."""
    raw_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries_key = "samples" if "samples" in raw_payload else "scenarios"
    entries = raw_payload.get(entries_key, [])
    if not isinstance(entries, list):
        raise ValueError(f"manifest {manifest_path} must contain a list under {entries_key!r}")

    samples = {sample.sample_id: sample for sample in load_manifest(manifest_path)}
    config, crop_scale = _runtime_config(setup_path=setup_path, weights_path=weights_path)
    cache = DiskPredictionCache(cache_dir)
    model_names = _models(models)
    if not model_names:
        raise ValueError("at least one model is required")
    logger.info(
        "Backfilling runtime resolver metadata: manifest=%s samples=%d models=%s crop_scale=%s",
        manifest_path,
        len(entries),
        ",".join(model_names),
        crop_scale,
    )

    updated = 0
    for entry in entries:
        sample_id = str(entry.get("sample_id") or entry.get("id") or entry.get("name"))
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(f"could not resolve manifest sample {sample_id!r}")
        bbox = _bbox_for_sample(sample)
        predictions = _read_predictions(cache, sample, models=model_names)
        logger.debug(
            "Backfilling sample=%s image=%s bbox=%s models=%s",
            sample.sample_id,
            sample.image,
            bbox,
            [prediction.model for prediction in predictions],
        )
        result = image_aware_runtime_result(
            sample,
            predictions=predictions,
            config=config,
            detector_bbox=bbox,
            crop_scale=crop_scale,
            crop_size=crop_size,
        )
        logger.debug(
            "Backfilled sample=%s bucket=%s selected=%s fallback=%s",
            sample.sample_id,
            result.metadata.get("runtime_bucket"),
            result.selected_candidate,
            result.metadata.get("fallback_reason"),
        )
        _trace("Backfilled sample=%s metadata=%s", sample.sample_id, result.metadata)
        metadata = entry.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            entry["metadata"] = metadata
        existing = metadata.get("landmark_ensemble", {})
        if not isinstance(existing, dict):
            existing = {}
        metadata["landmark_ensemble"] = {
            **existing,
            **_backfilled_metadata(result.metadata),
        }
        updated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(raw_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"sample_count": len(entries), "updated_count": updated, "output": str(output_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_IMAGE_BACKFILL_CROP_SIZE)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    report = backfill_runtime_resolver_metadata(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        weights_path=args.weights,
        setup_path=args.setup,
        output_path=args.output,
        models=_models(args.models),
        crop_size=args.crop_size,
    )
    logger.info("Backfilled runtime metadata for %d samples", report["updated_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
