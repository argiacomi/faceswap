#!/usr/bin/env python3
"""Static reliability weights for landmark ensembles."""

from __future__ import annotations

import json
import typing as T
from pathlib import Path

import numpy as np

MODEL_NAMES = ("hrnet", "spiga", "orformer")
LANDMARK_COUNT = 68

#: 6-region partition for region-level fusion (Phase 5 #9). The index sets mirror
#: ``lib.landmarks.evaluation.profile_metrics.REGION_INDICES`` (a drift-guard test
#: keeps them aligned) and cover all 68 canonical landmarks exactly once, so a
#: per-region weight broadcast is equivalent to assembling the fused face
#: region-by-region.
FUSION_REGION_INDICES: dict[str, tuple[int, ...]] = {
    "visible_jaw": tuple(range(0, 17)),
    "brows": tuple(range(17, 27)),
    "nose": tuple(range(27, 36)),
    "visible_eye": tuple(range(36, 48)),
    "mouth_outer": tuple(range(48, 60)),
    "mouth_inner": tuple(range(60, 68)),
}
FUSION_REGIONS: tuple[str, ...] = tuple(FUSION_REGION_INDICES)


def default_weights(
    models: T.Sequence[str] = MODEL_NAMES,
    *,
    landmark_count: int = LANDMARK_COUNT,
) -> dict[str, list[float]]:
    """Return equal per-landmark weights for every model."""
    value = 1.0 / len(models)
    return {model: [value] * landmark_count for model in models}


def normalize_static_weights(
    weights: T.Mapping[str, T.Sequence[float]],
    *,
    landmark_count: int = LANDMARK_COUNT,
) -> dict[str, list[float]]:
    """Normalize a model->per-landmark weight mapping so each landmark sums to one."""
    if not weights:
        raise ValueError("weights cannot be empty")
    model_names = tuple(weights)
    matrix = np.asarray([weights[name] for name in model_names], dtype="float32")
    if matrix.shape != (len(model_names), landmark_count):
        raise ValueError(
            f"weights must have shape {(len(model_names), landmark_count)}, got {matrix.shape}"
        )
    if np.any(matrix < 0):
        raise ValueError("weights cannot contain negative values")
    totals = matrix.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("every landmark must have at least one non-zero weight")
    normalized = matrix / totals[None, :]
    return {
        model: normalized[idx].astype("float32").tolist() for idx, model in enumerate(model_names)
    }


def normalize_bucket_weights(
    bucket_weights: T.Mapping[str, T.Mapping[str, T.Sequence[float]]],
    *,
    landmark_count: int = LANDMARK_COUNT,
) -> dict[str, dict[str, list[float]]]:
    """Normalize a ``{bucket: {model: per-landmark weights}}`` mapping.

    Each bucket's weight set is normalized independently with
    :func:`normalize_static_weights`. Empty input returns an empty mapping so
    callers can persist "no per-bucket weights" without special-casing.
    """
    return {
        bucket: normalize_static_weights(weights, landmark_count=landmark_count)
        for bucket, weights in bucket_weights.items()
    }


def normalize_region_weights(
    region_weights: T.Mapping[str, T.Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Normalize a ``{region: {model: weight}}`` table so each region sums to one.

    Unknown regions are rejected; every region's per-model weights must be
    non-negative with a positive sum. Empty input returns an empty mapping.
    """
    normalized: dict[str, dict[str, float]] = {}
    for region, column in region_weights.items():
        if region not in FUSION_REGION_INDICES:
            raise ValueError(
                f"unknown fusion region {region!r}; supported regions: "
                + ", ".join(FUSION_REGIONS)
            )
        values = {model: float(weight) for model, weight in column.items()}
        if any(weight < 0 for weight in values.values()):
            raise ValueError(f"region {region!r} weights cannot be negative")
        total = sum(values.values())
        if total <= 0:
            raise ValueError(f"region {region!r} must have at least one positive weight")
        normalized[region] = {model: weight / total for model, weight in values.items()}
    return normalized


def region_weights_to_matrix(
    region_weights: T.Mapping[str, T.Mapping[str, float]],
    models: T.Sequence[str],
    *,
    default_weight: float = 1.0,
    landmark_count: int = LANDMARK_COUNT,
) -> np.ndarray:
    """Broadcast a ``{region: {model: weight}}`` table into a normalized matrix.

    Each region's per-model scalar weight is broadcast across that region's
    landmark indices, producing a ``(models, landmarks)`` matrix that is then
    column-normalized via :func:`normalize_static_weights`. Regions or models
    missing from the table fall back to ``default_weight`` (equal trust).
    """
    matrix: np.ndarray = np.full((len(models), landmark_count), default_weight, dtype="float32")
    for region, indices in FUSION_REGION_INDICES.items():
        column = region_weights.get(region, {})
        for model_index, model in enumerate(models):
            matrix[model_index, list(indices)] = float(column.get(model, default_weight))
    normalized = normalize_static_weights(
        {model: matrix[idx].tolist() for idx, model in enumerate(models)},
        landmark_count=landmark_count,
    )
    return np.asarray([normalized[model] for model in models], dtype="float32")  # type: ignore[no-any-return]


def weights_matrix_for_models(
    weights: T.Mapping[str, T.Sequence[float]],
    models: T.Sequence[str],
    *,
    default_weight: float = 1.0,
    landmark_count: int = LANDMARK_COUNT,
) -> np.ndarray:
    """Return a normalized ``(models, landmarks)`` weight matrix."""
    selected = {
        model: list(weights.get(model, [default_weight] * landmark_count)) for model in models
    }
    normalized = normalize_static_weights(selected, landmark_count=landmark_count)
    return np.asarray([normalized[model] for model in models], dtype="float32")  # type: ignore[no-any-return]


def weights_from_errors(
    errors: T.Mapping[str, T.Sequence[float]],
    *,
    epsilon: float = 1e-6,
    landmark_count: int = LANDMARK_COUNT,
) -> dict[str, list[float]]:
    """Convert per-model per-landmark error arrays into inverse-error weights."""
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")
    if not errors:
        raise ValueError("errors cannot be empty")
    model_names = tuple(errors)
    matrix = np.asarray([errors[name] for name in model_names], dtype="float32")
    if matrix.shape != (len(model_names), landmark_count):
        raise ValueError(
            f"errors must have shape {(len(model_names), landmark_count)}, got {matrix.shape}"
        )
    if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("errors must be finite non-negative values")
    inverse = 1.0 / np.maximum(matrix, epsilon)
    return normalize_static_weights(
        {model: inverse[idx].tolist() for idx, model in enumerate(model_names)},
        landmark_count=landmark_count,
    )


def load_weights(path: str | Path) -> dict[str, list[float]]:
    """Load static weights from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("weights", payload)
    if not isinstance(raw, dict):
        raise ValueError("weights file must contain an object")
    return normalize_static_weights(raw)


def load_optional_weight_blocks(
    path: str | Path,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, float]]]:
    """Return ``(bucket_weights, region_weights)`` from a weights JSON file.

    v1 artifacts (and any file lacking the optional blocks) yield empty mappings.
    Used by the scorer-data generator so a retrain enumerates the same per-bucket
    (#8) and region-level (#9) fusion candidates the runtime resolver produces.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_buckets = payload.get("bucket_weights") or {}
    raw_regions = payload.get("region_weights") or {}
    bucket_weights = {
        str(bucket): {str(model): [float(v) for v in column] for model, column in columns.items()}
        for bucket, columns in raw_buckets.items()
    }
    region_weights = {
        str(region): {str(model): float(weight) for model, weight in columns.items()}
        for region, columns in raw_regions.items()
    }
    return bucket_weights, region_weights


def save_weights(path: str | Path, weights: T.Mapping[str, T.Sequence[float]]) -> None:
    """Write normalized static weights to JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "2d_68", "weights": normalize_static_weights(weights)}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
