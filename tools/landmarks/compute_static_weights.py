#!/usr/bin/env python3
"""Compute static per-landmark ensemble weights from cached predictions.

Legacy diagnostic CLI. Candidate for merge into `search_ensemble_setup.py`
when the unified promotion flow fully replaces standalone static fitting.
"""

from __future__ import annotations

import argparse
import json
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.metrics import per_landmark_error
from lib.landmarks.ensemble.weight_generators import (
    GENERATOR_NAMES,
    ErrorTable,
    WeightFitResult,
    get_generator,
)
from lib.landmarks.ensemble.weights import (
    MODEL_NAMES,
    normalize_static_weights,
)
from lib.landmarks.evaluation.harness import load_manifest
from tools.landmarks.pipeline_conventions import STATIC_WEIGHTS_FILENAME, write_json

DEFAULT_OUTPUT = f"configs/ensemble/{STATIC_WEIGHTS_FILENAME}"
DEFAULT_GENERATOR = "inverse_mean_error"


def _build_error_table(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str],
) -> ErrorTable:
    """Load samples and assemble a per-model error table from cached predictions."""
    model_names = tuple(model.strip() for model in models if model.strip())
    if not model_names:
        raise ValueError("at least one model is required")
    cache = DiskPredictionCache(cache_dir)
    samples = load_manifest(manifest_path)
    if not samples:
        raise ValueError("manifest contains no validation samples")

    errors: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    for sample in samples:
        truth = np.load(sample.landmarks).astype("float32")
        for model in model_names:
            prediction = cache.read(sample.sample_id, model)
            errors[model].append(per_landmark_error(prediction.landmarks, truth))
    return ErrorTable.from_samples(errors)


def fit_static_weights(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str] = MODEL_NAMES,
    *,
    generator: str = DEFAULT_GENERATOR,
    generator_params: T.Mapping[str, T.Any] | None = None,
) -> tuple[WeightFitResult, ErrorTable]:
    """Fit static weights with the chosen generator and return diagnostics.

    Returns the generator's :class:`WeightFitResult` plus the assembled
    :class:`ErrorTable` so callers (CLI, candidate search) can persist both the
    selected weights and the underlying error statistics.
    """
    table = _build_error_table(manifest_path, cache_dir, models)
    generator_cls = type(get_generator(generator))
    instance = generator_cls(**dict(generator_params or {}))
    result = instance.fit(table, models=table.models)
    return result, table


def compute_static_weights(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: T.Sequence[str] = MODEL_NAMES,
    *,
    generator: str = DEFAULT_GENERATOR,
    generator_params: T.Mapping[str, T.Any] | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute static weights and mean validation errors from cached predictions.

    Returns ``(weights, mean_errors)`` for parity with the legacy public API.
    Use :func:`fit_static_weights` when generator name and diagnostics are
    needed alongside the weights.
    """
    result, table = fit_static_weights(
        manifest_path,
        cache_dir,
        models,
        generator=generator,
        generator_params=generator_params,
    )
    mean_errors = {
        name: matrix.mean(axis=0).astype("float32").tolist()
        for name, matrix in table.errors.items()
    }
    return result.weights, mean_errors


def save_weight_artifact(
    output: str | Path,
    result: WeightFitResult,
) -> None:
    """Write generated weights to JSON with embedded generator metadata.

    The on-disk schema is the same shape consumed by
    :func:`lib.landmarks.ensemble.weights.load_weights` so existing readers
    continue to work; the additional ``generator`` block is optional metadata
    new readers (#71 promoted artifacts) can consume.
    """
    payload = {
        "schema": "2d_68",
        "weights": normalize_static_weights(result.weights),
        "generator": {
            "name": result.name,
            "diagnostics": result.diagnostics,
        },
    }
    write_json(Path(output), payload)


def _parse_generator_params(raw: str | None) -> dict[str, T.Any]:
    """Parse ``key=value`` pairs into a generator constructor kwargs dict."""
    if not raw:
        return {}
    params: dict[str, T.Any] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"generator param {chunk!r} is missing '='")
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"generator param {chunk!r} has empty key")
        try:
            params[key] = float(value)
        except ValueError as err:
            raise ValueError(f"generator param {key}={value!r} must be numeric") from err
    return params


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--generator",
        default=DEFAULT_GENERATOR,
        choices=GENERATOR_NAMES,
        help="Static weight generator to use (default: inverse_mean_error).",
    )
    parser.add_argument(
        "--generator-params",
        default=None,
        help=(
            "Comma-separated 'key=value' overrides forwarded to the generator "
            "constructor (e.g. 'epsilon=1e-5,per_landmark_weight=0.6')."
        ),
    )
    args = parser.parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    params = _parse_generator_params(args.generator_params)
    result, table = fit_static_weights(
        args.manifest,
        args.cache_dir,
        models,
        generator=args.generator,
        generator_params=params,
    )
    save_weight_artifact(args.output, result)
    if args.report:
        weight_matrix = np.asarray([result.weights[model] for model in result.weights])
        dominant = weight_matrix.argmax(axis=0)
        payload = {
            "generator": {"name": result.name, "diagnostics": result.diagnostics},
            "models": list(result.weights),
            "mean_errors": {
                name: matrix.mean(axis=0).astype("float32").tolist()
                for name, matrix in table.errors.items()
            },
            "dominant_model_by_landmark": dominant.astype(int).tolist(),
        }
        write_json(Path(args.report), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
