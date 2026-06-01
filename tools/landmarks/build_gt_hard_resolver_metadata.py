#!/usr/bin/env python3
"""Build a frozen GT-hard resolver metadata sidecar from a landmark manifest."""

from __future__ import annotations

import argparse
import logging
import sys
import typing as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.datasets.manifest_io import LandmarkSample, load_manifest
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    DEFAULT_OUTLIER_THRESHOLD,
    build_sample_context,
    parse_candidates,
)
from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.pipeline_conventions import (
    METADATA_SOURCES,
    SOURCE_GT_HARD,
    face_index_for_sample,
    load_resolver_metadata_sidecar,
    normalize_source_label,
    validate_resolver_metadata_for_manifest,
    write_jsonl,
)

logger = logging.getLogger("build_gt_hard_resolver_metadata")


def _json_safe(value: T.Any) -> T.Any:
    """Convert numpy/scalar-like values to deterministic JSON-safe payloads."""
    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - numpy is required by callers
        np = None  # type: ignore[assignment]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _sample_without_manifest_runtime_metadata(sample: LandmarkSample) -> LandmarkSample:
    """Return a sample copy that cannot reuse stale manifest runtime metadata."""
    metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    metadata.pop("landmark_ensemble", None)
    metadata.pop("resolver_metadata", None)
    for key in list(metadata):
        if str(key).startswith("landmark_ensemble_"):
            metadata.pop(key, None)
    return LandmarkSample(
        sample_id=sample.sample_id,
        image=sample.image,
        landmarks=sample.landmarks,
        dataset=sample.dataset,
        condition=sample.condition,
        conditions=sample.conditions,
        normalizer=sample.normalizer,
        face_bbox=sample.face_bbox,
        visibility=sample.visibility,
        metadata=metadata,
    )


def _resolver_metadata_row(context: T.Any, sample: LandmarkSample) -> dict[str, T.Any]:
    """Return one resolver_metadata.jsonl row for a GT-hard sample context."""
    face_index = face_index_for_sample(sample)
    condition = str(
        getattr(context, "condition", "")
        or sample.condition
        or context.runtime_bucket
        or "unknown"
    )
    hard_case_tags = list(getattr(context, "hard_case_tags", ()) or ())
    runtime_features: dict[str, T.Any] = {
        "candidate_yaw_disagreement": context.candidate_yaw_disagreement,
        "max_disagreement_px": context.max_disagreement_px,
        "landmark_pose_roll": context.roll_estimate,
        "landmark_pose_yaw": context.yaw_estimate,
        "model_predictions_available": context.model_predictions_available,
    }
    runtime_features = {key: value for key, value in runtime_features.items() if value is not None}
    resolver = {
        "runtime_bucket": context.runtime_bucket,
        "bucket": context.runtime_bucket,
        "runtime_bucket_source": context.runtime_bucket_source,
        "runtime_bucket_features": runtime_features,
        "condition": condition,
        "hard_case_tags": hard_case_tags,
        "selected_candidate": context.current_policy_choice,
        "risk_route": context.risk_route,
        "candidate_yaw_disagreement": context.candidate_yaw_disagreement,
        "max_disagreement_px": context.max_disagreement_px,
        "roll_estimate": context.roll_estimate,
        "yaw_estimate": context.yaw_estimate,
        "model_predictions_available": context.model_predictions_available,
    }
    resolver = {key: value for key, value in resolver.items() if value is not None}
    landmark_ensemble = {
        **resolver,
        "resolver": dict(resolver),
    }
    return _json_safe(  # type: ignore[no-any-return]
        {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image),
            "face_index": face_index,
            "condition": condition,
            "runtime_bucket": context.runtime_bucket,
            "hard_case_tags": hard_case_tags,
            "landmark_ensemble": landmark_ensemble,
        }
    )


def build_gt_hard_resolver_metadata(
    *,
    manifest: Path,
    cache_dir: Path,
    weights: Path,
    candidates: T.Sequence[str] | None,
    output: Path,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    allow_image_backfill: bool = False,
    image_backfill_crop_scale: float = DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    image_backfill_crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    source: str = SOURCE_GT_HARD,
) -> list[dict[str, T.Any]]:
    """Run the runtime resolver context builder and write a complete GT-hard sidecar."""
    source = normalize_source_label(source)
    loaded_weights = load_weights(weights)
    requested_candidates = tuple(candidates or parse_candidates(None, loaded_weights))
    cache = DiskPredictionCache(cache_dir)
    rows: list[dict[str, T.Any]] = []
    failures: list[str] = []
    samples = load_manifest(manifest)
    for sample in samples:
        try:
            context_sample = _sample_without_manifest_runtime_metadata(sample)
            context = build_sample_context(
                context_sample,
                cache=cache,
                requested_candidates=requested_candidates,
                weights=loaded_weights,
                source="",
                resolver_metadata=None,
                failure_threshold=failure_threshold,
                outlier_threshold=outlier_threshold,
                allow_image_backfill=allow_image_backfill,
                image_backfill_crop_scale=image_backfill_crop_scale,
                image_backfill_crop_size=image_backfill_crop_size,
            )
            rows.append(_resolver_metadata_row(context, sample))
        except Exception as err:  # noqa: BLE001
            message = f"{sample.sample_id}: {type(err).__name__}: {err}"
            failures.append(message)
            logger.warning("Failed GT-hard resolver metadata row: %s", message)
    if failures:
        detail = "; ".join(failures[:10])
        raise RuntimeError(
            f"failed to build resolver metadata for {len(failures)} GT-hard sample(s): {detail}"
        )
    write_jsonl(output, rows)
    validate_resolver_metadata_for_manifest(
        manifest,
        load_resolver_metadata_sidecar(output),
        source=source,
        require_complete=True,
    )
    logger.info("Wrote %d GT-hard resolver metadata row(s) to %s", len(rows), output)
    return rows


def _parse_candidates_arg(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument(
        "--allow-image-backfill",
        action="store_true",
        help="Use image-aware runtime bucket inference while ignoring manifest-stored runtime metadata.",
    )
    parser.add_argument(
        "--image-backfill-crop-scale",
        type=float,
        default=DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    )
    parser.add_argument(
        "--image-backfill-crop-size",
        type=int,
        default=DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    )
    parser.add_argument("--source", choices=METADATA_SOURCES, default=SOURCE_GT_HARD)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    build_gt_hard_resolver_metadata(
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        weights=args.weights,
        candidates=_parse_candidates_arg(args.candidates),
        output=args.output,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
        allow_image_backfill=args.allow_image_backfill,
        image_backfill_crop_scale=args.image_backfill_crop_scale,
        image_backfill_crop_size=args.image_backfill_crop_size,
        source=args.source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_gt_hard_resolver_metadata", "main", "_sample_without_manifest_runtime_metadata"]
