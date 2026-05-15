#!/usr/bin/env python3
"""Promoted landmark ensemble setup artifact writers (#69 / #71).

The ``best_setup.json`` / ``best_weights.json`` schema v1 is the contract
between offline candidate search (#69) and the runtime extract aligner (#66).
This module owns the writer half so #69 produces well-formed artifacts; the
strict loader and validator land in #71 and consume the same constants.
"""

from __future__ import annotations

import json
import typing as T
from pathlib import Path

from lib.landmarks.ensemble.strategies import canonical_strategy, validate_threshold
from lib.landmarks.ensemble.weights import normalize_static_weights
from lib.landmarks.schema import CANONICAL_SCHEMA

ARTIFACT_SCHEMA_VERSION: int = 1
WEIGHTS_FILENAME: str = "best_weights.json"
SETUP_FILENAME: str = "best_setup.json"
PROMOTION_REPORT_FILENAME: str = "promotion_report.md"


def write_best_weights(
    path: str | Path,
    weights: T.Mapping[str, T.Sequence[float]],
    *,
    models: T.Sequence[str],
    schema: str = CANONICAL_SCHEMA,
) -> Path:
    """Write a ``best_weights.json`` v1 artifact.

    Each model weight list must contain exactly 68 non-negative values; every
    landmark column must sum to a positive value before normalization. The
    payload is normalized in place so per-landmark columns sum to 1.0.
    """
    if schema != CANONICAL_SCHEMA:
        raise ValueError(f"only {CANONICAL_SCHEMA!r} schema is supported, got {schema!r}")
    ordered = {model: list(weights[model]) for model in models}
    normalized = normalize_static_weights(ordered)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "schema": schema,
        "models": list(models),
        "weights": normalized,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_best_setup(
    path: str | Path,
    *,
    candidate_id: str,
    models: T.Sequence[str],
    strategy: str,
    outlier_threshold: float | None,
    weight_generator_name: str,
    weight_generator_params: T.Mapping[str, T.Any],
    crop_scale: float,
    bbox_source: str,
    regression_epsilon_nme: float,
    reproducibility: T.Mapping[str, T.Any],
    fit: T.Mapping[str, T.Any],
    selection_metrics: T.Mapping[str, T.Any],
    report_metrics: T.Mapping[str, T.Any] | None = None,
    evaluation_log_path: str = "",
    weights_path: str = WEIGHTS_FILENAME,
    schema: str = CANONICAL_SCHEMA,
) -> Path:
    """Write a ``best_setup.json`` v1 artifact.

    Validates the strategy + threshold combination before writing so an
    incompatible setup cannot reach disk. ``weights_path`` is stored
    verbatim; callers should pass a path relative to the setup file's
    directory so promoted bundles are portable.
    """
    if schema != CANONICAL_SCHEMA:
        raise ValueError(f"only {CANONICAL_SCHEMA!r} schema is supported, got {schema!r}")
    canonical = canonical_strategy(strategy)
    validate_threshold(canonical, outlier_threshold)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, T.Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "schema": schema,
        "candidate_id": str(candidate_id),
        "models": list(models),
        "strategy": canonical,
        "outlier_threshold": outlier_threshold,
        "regression_epsilon_nme": float(regression_epsilon_nme),
        "crop_scale": float(crop_scale),
        "bbox_source": str(bbox_source),
        "weights_path": str(weights_path),
        "weight_generator": {
            "name": str(weight_generator_name),
            "params": {key: _scalar(value) for key, value in weight_generator_params.items()},
        },
        "reproducibility": _json_ready(reproducibility),
        "fit": _json_ready(fit),
        "selection_metrics": _json_ready(selection_metrics),
        "report_metrics": _json_ready(report_metrics or {}),
        "evaluation_log_path": str(evaluation_log_path),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _scalar(value: T.Any) -> T.Any:
    """Coerce numpy scalars and other numbers to plain JSON-friendly values."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()  # numpy scalars
        except Exception:  # pragma: no cover - defensive fallback
            return value
    return value


def _json_ready(value: T.Any) -> T.Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return _scalar(value)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PROMOTION_REPORT_FILENAME",
    "SETUP_FILENAME",
    "WEIGHTS_FILENAME",
    "write_best_setup",
    "write_best_weights",
]
