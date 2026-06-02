#!/usr/bin/env python3
"""Library helper for fitting static landmark ensemble weights."""

from __future__ import annotations

import logging
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.metrics import per_landmark_error
from lib.landmarks.datasets.manifest_io import filter_canonical_68_samples
from lib.landmarks.ensemble.hard_condition_taxonomy import (
    derive_hard_condition_taxonomy,
    weight_bucket_from_pose,
)
from lib.landmarks.ensemble.weights import (
    FUSION_REGION_INDICES,
    MODEL_NAMES,
    normalize_region_weights,
    weights_from_errors,
)
from lib.landmarks.evaluation.geometry_signals import alignment_summary
from lib.landmarks.evaluation.harness import load_manifest

logger = logging.getLogger(__name__)

#: Default minimum number of fit samples a bucket needs before it gets its own
#: weight set; sparser buckets fall back to the global weights at runtime.
DEFAULT_BUCKET_MIN_SAMPLES: int = 200


def compute_static_weights(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str] = MODEL_NAMES,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute static weights and mean validation errors from cached predictions."""
    model_names = tuple(model.strip() for model in models if model.strip())
    if not model_names:
        raise ValueError("at least one model is required")
    cache = DiskPredictionCache(cache_dir)
    samples = filter_canonical_68_samples(
        load_manifest(manifest_path), context="static weight fit"
    )
    if not samples:
        raise ValueError("manifest contains no validation samples")

    errors: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    for sample in samples:
        truth = np.load(sample.landmarks).astype("float32")
        for model in model_names:
            prediction = cache.read(sample.sample_id, model)
            errors[model].append(per_landmark_error(prediction.landmarks, truth))
    mean_errors = {
        model: np.stack(model_errors, axis=0).mean(axis=0).astype("float32").tolist()
        for model, model_errors in errors.items()
    }
    return weights_from_errors(mean_errors), mean_errors


def _pose_from_truth(truth: np.ndarray) -> tuple[float | None, float | None]:
    """Return (yaw, roll) degrees from ground-truth landmarks, or (None, None).

    Pose estimation runs Faceswap's alignment; degenerate landmark clouds raise
    and fall back to ``(None, None)`` so the sample lands in the frontal bucket
    rather than aborting the whole fit.
    """
    try:
        summary = alignment_summary(truth.astype("float32", copy=False))
    except (ValueError, np.linalg.LinAlgError) as err:
        logger.debug("pose estimation failed for bucket fit: %s", err)
        return None, None
    return float(summary.yaw), float(summary.roll)


def compute_bucket_weights(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str] = MODEL_NAMES,
    *,
    min_samples: int = DEFAULT_BUCKET_MIN_SAMPLES,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    """Fit global and per-bucket static weights from cached predictions.

    Each sample is assigned to exactly one coarse weight bucket from its
    ground-truth pose (:func:`weight_bucket_from_pose`) and occlusion taxonomy.
    Buckets with at least ``min_samples`` fit samples get their own normalized
    ``{model: [68]}`` weights; sparser buckets are omitted so the runtime falls
    back to the global weights.

    Returns ``(global_weights, global_mean_errors, bucket_weights)`` where
    ``bucket_weights`` is a ``{bucket: {model: [68]}}`` mapping.
    """
    model_names = tuple(model.strip() for model in models if model.strip())
    if not model_names:
        raise ValueError("at least one model is required")
    if min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    cache = DiskPredictionCache(cache_dir)
    samples = filter_canonical_68_samples(
        load_manifest(manifest_path), context="bucket weight fit"
    )
    if not samples:
        raise ValueError("manifest contains no validation samples")

    global_errors: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    bucket_errors: dict[str, dict[str, list[np.ndarray]]] = {}
    for sample in samples:
        truth = np.load(sample.landmarks).astype("float32")
        per_model_error = {
            model: per_landmark_error(cache.read(sample.sample_id, model).landmarks, truth)
            for model in model_names
        }
        for model, error in per_model_error.items():
            global_errors[model].append(error)

        yaw, roll = _pose_from_truth(truth)
        taxonomy = derive_hard_condition_taxonomy(
            sample,
            runtime_bucket="",
            yaw_estimate=yaw,
            roll_estimate=roll,
        )
        occluded = "occlusion" in taxonomy.hard_case_tags
        bucket = weight_bucket_from_pose(yaw, roll, occluded=occluded)
        bucket_slot = bucket_errors.setdefault(bucket, {model: [] for model in model_names})
        for model, error in per_model_error.items():
            bucket_slot[model].append(error)

    global_mean_errors = {
        model: np.stack(errors, axis=0).mean(axis=0).astype("float32").tolist()
        for model, errors in global_errors.items()
    }
    global_weights = weights_from_errors(global_mean_errors)

    bucket_weights: dict[str, dict[str, list[float]]] = {}
    for bucket, slot in bucket_errors.items():
        sample_count = len(next(iter(slot.values())))
        if sample_count < min_samples:
            logger.debug(
                "bucket %s has %d samples (< %d); falling back to global weights",
                bucket,
                sample_count,
                min_samples,
            )
            continue
        mean_errors = {
            model: np.stack(errors, axis=0).mean(axis=0).astype("float32").tolist()
            for model, errors in slot.items()
        }
        bucket_weights[bucket] = weights_from_errors(mean_errors)

    return global_weights, global_mean_errors, bucket_weights


def compute_region_weights(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str] = MODEL_NAMES,
    *,
    epsilon: float = 1e-6,
) -> dict[str, dict[str, float]]:
    """Fit per-region inverse-error fusion weights from cached predictions.

    Each region's per-model weight is the inverse of that model's mean
    validation error averaged over the region's landmark indices, normalized so
    the region's model weights sum to one. The result is a compact
    ``{region: {model: weight}}`` table consumed by ``region_weights_to_matrix``
    at fusion time (Phase 5 #9).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")
    _, mean_errors = compute_static_weights(manifest_path, cache_dir, models)
    region_table: dict[str, dict[str, float]] = {}
    for region, indices in FUSION_REGION_INDICES.items():
        inverse: dict[str, float] = {}
        for model, errors in mean_errors.items():
            region_error = float(np.mean([errors[index] for index in indices]))
            inverse[model] = 1.0 / max(region_error, epsilon)
        region_table[region] = inverse
    return normalize_region_weights(region_table)


__all__ = [
    "DEFAULT_BUCKET_MIN_SAMPLES",
    "compute_bucket_weights",
    "compute_region_weights",
    "compute_static_weights",
]
