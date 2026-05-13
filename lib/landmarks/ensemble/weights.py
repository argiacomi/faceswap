#!/usr/bin/env python3
"""Static reliability weights for landmark ensembles."""

from __future__ import annotations

import json
import typing as T
from pathlib import Path

import numpy as np

MODEL_NAMES = ("hrnet", "spiga", "orformer")
LANDMARK_COUNT = 68


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
    return np.asarray([normalized[model] for model in models], dtype="float32")


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


def save_weights(path: str | Path, weights: T.Mapping[str, T.Sequence[float]]) -> None:
    """Write normalized static weights to JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "2d_68", "weights": normalize_static_weights(weights)}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
