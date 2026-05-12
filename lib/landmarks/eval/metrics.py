#!/usr/bin/env python3
"""Evaluation metrics for landmark harness reports."""

from __future__ import annotations

import typing as T

import numpy as np

from lib.align.constants import LANDMARK_PARTS, LandmarkType
from lib.landmarks.metrics import (
    auc,
    failure_rate,
    normalized_mean_error,
    per_landmark_error,
)
from lib.landmarks.schema import normalize_landmarks


def region_errors(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return mean point error by Faceswap 68-point region."""
    errors = per_landmark_error(predicted, target)
    output: dict[str, float] = {}
    for name, (start, end, _is_polygon) in LANDMARK_PARTS[LandmarkType.LM_2D_68].items():
        output[name] = float(errors[start:end].mean())
    return output


def inter_model_disagreement(predictions: T.Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return mean distance from each model to the cross-model median."""
    if not predictions:
        return {}
    names = tuple(predictions)
    stack = np.stack([normalize_landmarks(predictions[name]) for name in names], axis=0)
    median = np.median(stack, axis=0)
    return {
        name: float(np.linalg.norm(stack[idx] - median, axis=1).mean())
        for idx, name in enumerate(names)
    }


def evaluate_prediction(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    normalizer: float | None = None,
    failure_threshold: float = 0.08,
) -> dict[str, T.Any]:
    """Return core scalar and per-landmark metrics for one prediction."""
    pred = normalize_landmarks(predicted)
    truth = normalize_landmarks(target)
    point_errors = per_landmark_error(pred, truth)
    nme = normalized_mean_error(pred, truth, normalizer=normalizer)
    return {
        "nme": nme,
        "failure": bool(nme > failure_threshold),
        "per_landmark_error": point_errors.astype("float32").tolist(),
        "per_region_error": region_errors(pred, truth),
    }


def summarize_errors(
    errors: T.Sequence[float], *, failure_threshold: float = 0.08
) -> dict[str, float]:
    """Summarize a sequence of NME values."""
    values = np.asarray(errors, dtype="float32")
    if values.size == 0:
        return {"count": 0.0, "nme": 0.0, "auc": 0.0, "failure_rate": 0.0}
    return {
        "count": float(values.size),
        "nme": float(values.mean()),
        "auc": auc(values, threshold=failure_threshold),
        "failure_rate": failure_rate(values, threshold=failure_threshold),
    }
