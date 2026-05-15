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
from dataclasses import dataclass, field
from pathlib import Path

from lib.landmarks.ensemble.strategies import (
    CANONICAL_STRATEGIES,
    canonical_strategy,
    validate_threshold,
)
from lib.landmarks.ensemble.weights import LANDMARK_COUNT, normalize_static_weights
from lib.landmarks.schema import CANONICAL_SCHEMA

ARTIFACT_SCHEMA_VERSION: int = 1
WEIGHTS_FILENAME: str = "best_weights.json"
SETUP_FILENAME: str = "best_setup.json"
PROMOTION_REPORT_FILENAME: str = "promotion_report.md"

SETUP_REQUIRED_KEYS: tuple[str, ...] = (
    "artifact_schema_version",
    "schema",
    "candidate_id",
    "models",
    "strategy",
    "outlier_threshold",
    "weights_path",
    "weight_generator",
    "reproducibility",
    "fit",
    "selection_metrics",
    "report_metrics",
)
WEIGHTS_REQUIRED_KEYS: tuple[str, ...] = (
    "artifact_schema_version",
    "schema",
    "models",
    "weights",
)


class PromotedSetupError(ValueError):
    """Raised when a promoted setup or weights artifact fails strict validation.

    Inherits from ``ValueError`` so callers that catch the broader error type
    (including legacy code) still get the failure; the dedicated subclass lets
    the runtime extract plugin distinguish artifact-load failures from other
    validation errors.
    """


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


@dataclass(frozen=True)
class PromotedSetup:
    """Validated in-memory view of a ``best_setup.json`` artifact."""

    artifact_schema_version: int
    schema: str
    candidate_id: str
    models: tuple[str, ...]
    strategy: str
    outlier_threshold: float | None
    weight_generator: dict[str, T.Any]
    crop_scale: float
    bbox_source: str
    regression_epsilon_nme: float
    reproducibility: dict[str, T.Any]
    fit: dict[str, T.Any]
    selection_metrics: dict[str, T.Any]
    report_metrics: dict[str, T.Any]
    evaluation_log_path: str
    weights_path: str
    weights: dict[str, list[float]] = field(default_factory=dict)

    def has_weights(self) -> bool:
        return bool(self.weights)


def validate_setup_payload(payload: T.Mapping[str, T.Any]) -> None:
    """Strictly validate a parsed ``best_setup.json`` dict.

    Raises :class:`PromotedSetupError` with an actionable message on any
    failure. Validations:

    - ``artifact_schema_version`` must equal :data:`ARTIFACT_SCHEMA_VERSION`
      (unsupported future versions including ``2`` are rejected here).
    - ``schema`` must equal ``'2d_68'``.
    - All :data:`SETUP_REQUIRED_KEYS` must be present.
    - ``strategy`` must be a canonical strategy name; aliases are not allowed
      in promoted artifacts (offline tooling resolves them before promotion).
    - ``outlier_threshold`` must be ``null`` for non-threshold strategies and
      a positive number for threshold-using strategies.
    - The setup must NOT include the legacy ``outlier_method`` field; the
      strategy is the single source of truth.
    - ``weight_generator`` must be an object with a non-empty ``name``.
    """
    if not isinstance(payload, T.Mapping):
        raise PromotedSetupError(
            f"setup payload must be a JSON object, got {type(payload).__name__}"
        )
    missing = [key for key in SETUP_REQUIRED_KEYS if key not in payload]
    if missing:
        raise PromotedSetupError(f"setup payload missing required keys: {missing}")
    if "outlier_method" in payload:
        raise PromotedSetupError(
            "setup payload contains forbidden 'outlier_method'; strategy is the "
            "single source of truth"
        )
    version = payload["artifact_schema_version"]
    if version != ARTIFACT_SCHEMA_VERSION:
        raise PromotedSetupError(
            f"unsupported artifact_schema_version {version!r}; this loader supports "
            f"version {ARTIFACT_SCHEMA_VERSION}"
        )
    schema_value = payload["schema"]
    if schema_value != CANONICAL_SCHEMA:
        raise PromotedSetupError(
            f"unsupported schema {schema_value!r}; only {CANONICAL_SCHEMA!r} is supported"
        )
    strategy_value = payload["strategy"]
    if strategy_value not in CANONICAL_STRATEGIES:
        raise PromotedSetupError(
            f"strategy {strategy_value!r} is not a canonical strategy; supported: "
            + ", ".join(CANONICAL_STRATEGIES)
        )
    threshold = payload["outlier_threshold"]
    try:
        validate_threshold(strategy_value, threshold)
    except ValueError as err:
        raise PromotedSetupError(str(err)) from err
    weight_generator = payload["weight_generator"]
    if not isinstance(weight_generator, T.Mapping):
        raise PromotedSetupError("weight_generator must be a JSON object")
    if not weight_generator.get("name"):
        raise PromotedSetupError("weight_generator.name must be a non-empty string")
    models = payload["models"]
    if not isinstance(models, list) or not models:
        raise PromotedSetupError("models must be a non-empty list of strings")
    if len(set(models)) != len(models):
        raise PromotedSetupError(f"models contains duplicates: {models}")


def validate_weights_payload(
    payload: T.Mapping[str, T.Any], *, expected_models: T.Sequence[str] | None = None
) -> None:
    """Strictly validate a parsed ``best_weights.json`` dict.

    When ``expected_models`` is provided every model must appear in
    ``weights`` and the ordering must match the setup's ``models`` list.
    """
    if not isinstance(payload, T.Mapping):
        raise PromotedSetupError(
            f"weights payload must be a JSON object, got {type(payload).__name__}"
        )
    missing = [key for key in WEIGHTS_REQUIRED_KEYS if key not in payload]
    if missing:
        raise PromotedSetupError(f"weights payload missing required keys: {missing}")
    version = payload["artifact_schema_version"]
    if version != ARTIFACT_SCHEMA_VERSION:
        raise PromotedSetupError(
            f"unsupported weights artifact_schema_version {version!r}; this loader "
            f"supports version {ARTIFACT_SCHEMA_VERSION}"
        )
    schema_value = payload["schema"]
    if schema_value != CANONICAL_SCHEMA:
        raise PromotedSetupError(
            f"unsupported weights schema {schema_value!r}; only {CANONICAL_SCHEMA!r} is supported"
        )
    models = payload["models"]
    weights = payload["weights"]
    if not isinstance(models, list) or not models:
        raise PromotedSetupError("weights.models must be a non-empty list of strings")
    if not isinstance(weights, dict):
        raise PromotedSetupError("weights must be a JSON object")
    missing_columns = [model for model in models if model not in weights]
    if missing_columns:
        raise PromotedSetupError(f"weights payload missing columns for models: {missing_columns}")
    for model in models:
        column = weights[model]
        if not isinstance(column, list) or len(column) != LANDMARK_COUNT:
            raise PromotedSetupError(
                f"weights[{model!r}] must be a list of exactly {LANDMARK_COUNT} values, "
                f"got length {len(column) if isinstance(column, list) else 'n/a'}"
            )
        for index, value in enumerate(column):
            if not isinstance(value, (int, float)):
                raise PromotedSetupError(
                    f"weights[{model!r}][{index}] must be numeric, got {type(value).__name__}"
                )
            if value < 0:
                raise PromotedSetupError(
                    f"weights[{model!r}][{index}] must be non-negative, got {value!r}"
                )
    if expected_models is not None and tuple(models) != tuple(expected_models):
        raise PromotedSetupError(
            f"weights.models {tuple(models)!r} do not match setup.models {tuple(expected_models)!r}"
        )


def load_promoted_setup(
    setup_path: str | Path,
    *,
    load_weights: bool = True,
) -> PromotedSetup:
    """Load and strictly validate a ``best_setup.json`` (and its weights file).

    Raises :class:`PromotedSetupError` for any validation failure. When
    ``load_weights`` is true, the referenced weights JSON is loaded, validated
    against the setup, and attached to the returned :class:`PromotedSetup`.
    Relative ``weights_path`` values resolve against the setup file's directory.
    """
    path = Path(setup_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as err:
        raise PromotedSetupError(f"could not read setup file {path}: {err}") from err
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as err:
        raise PromotedSetupError(f"setup file {path} is not valid JSON: {err}") from err
    validate_setup_payload(payload)

    weights: dict[str, list[float]] = {}
    if load_weights:
        weights_relative = str(payload["weights_path"])
        weights_path = Path(weights_relative)
        if not weights_path.is_absolute():
            weights_path = path.parent / weights_path
        try:
            weights_raw = weights_path.read_text(encoding="utf-8")
        except OSError as err:
            raise PromotedSetupError(f"could not read weights file {weights_path}: {err}") from err
        try:
            weights_payload = json.loads(weights_raw)
        except json.JSONDecodeError as err:
            raise PromotedSetupError(
                f"weights file {weights_path} is not valid JSON: {err}"
            ) from err
        validate_weights_payload(weights_payload, expected_models=payload["models"])
        weights = {
            model: [float(value) for value in weights_payload["weights"][model]]
            for model in payload["models"]
        }

    return PromotedSetup(
        artifact_schema_version=int(payload["artifact_schema_version"]),
        schema=str(payload["schema"]),
        candidate_id=str(payload["candidate_id"]),
        models=tuple(str(model) for model in payload["models"]),
        strategy=str(payload["strategy"]),
        outlier_threshold=(
            None if payload["outlier_threshold"] is None else float(payload["outlier_threshold"])
        ),
        weight_generator=dict(payload["weight_generator"]),
        crop_scale=float(payload.get("crop_scale", 1.6)),
        bbox_source=str(payload.get("bbox_source", "manifest")),
        regression_epsilon_nme=float(payload.get("regression_epsilon_nme", 0.001)),
        reproducibility=dict(payload["reproducibility"]),
        fit=dict(payload["fit"]),
        selection_metrics=dict(payload["selection_metrics"]),
        report_metrics=dict(payload["report_metrics"]),
        evaluation_log_path=str(payload.get("evaluation_log_path", "")),
        weights_path=str(payload["weights_path"]),
        weights=weights,
    )


def ensure_compatible_adapters(setup: PromotedSetup, available_adapters: T.Sequence[str]) -> None:
    """Validate that every model in the setup is present in ``available_adapters``.

    Used by the runtime extract plugin (#66) to hard-fail before extraction
    when the promoted setup expects a model that is not loaded.
    """
    available = set(available_adapters)
    missing = [model for model in setup.models if model not in available]
    if missing:
        raise PromotedSetupError(
            f"promoted setup expects models {missing!r} that are not present in the "
            f"loaded adapters {sorted(available)!r}"
        )


def strategy_supported_by_runtime(strategy: str, supported: T.Sequence[str]) -> None:
    """Raise :class:`PromotedSetupError` if ``strategy`` is not in ``supported``."""
    if strategy not in supported:
        raise PromotedSetupError(
            f"promoted setup uses strategy {strategy!r} that the runtime does not "
            f"support; supported: {', '.join(supported)}"
        )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PROMOTION_REPORT_FILENAME",
    "PromotedSetup",
    "PromotedSetupError",
    "SETUP_FILENAME",
    "SETUP_REQUIRED_KEYS",
    "WEIGHTS_FILENAME",
    "WEIGHTS_REQUIRED_KEYS",
    "ensure_compatible_adapters",
    "load_promoted_setup",
    "strategy_supported_by_runtime",
    "validate_setup_payload",
    "validate_weights_payload",
    "write_best_setup",
    "write_best_weights",
]
