#!/usr/bin/env python3
"""Write fixture predictions into the landmark disk cache."""

from __future__ import annotations

import argparse
import json
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.eval.harness import load_manifest
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction


def _split_models(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_prediction_roots(values: T.Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--prediction-root must be in model=path form")
        model, root = value.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError("--prediction-root model name cannot be empty")
        roots[model] = Path(root).expanduser()
    return roots


def _manifest_prediction(entry: dict[str, T.Any], model: str) -> str:
    for key in ("predictions", "model_predictions", "prediction_fixtures", "fixtures"):
        predictions = entry.get(key)
        if isinstance(predictions, dict) and model in predictions:
            return str(predictions[model])
    model_key = f"{model}_prediction"
    if model_key in entry:
        return str(entry[model_key])
    return ""


def _load_manifest_entries(path: str | Path) -> list[dict[str, T.Any]]:
    manifest = Path(path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(entries, list):
        raise ValueError("manifest samples must be a list")
    return [entry for entry in entries if isinstance(entry, dict)]


def _prediction_path(
    *,
    sample_id: str,
    model: str,
    manifest_path: Path,
    entry: dict[str, T.Any],
    prediction_roots: T.Mapping[str, Path],
) -> Path:
    manifest_prediction = _manifest_prediction(entry, model)
    if manifest_prediction:
        return (manifest_path.parent / manifest_prediction).resolve()
    root = prediction_roots.get(model)
    if root is None:
        raise FileNotFoundError(f"no prediction path for sample '{sample_id}' model '{model}'")
    candidates = (
        root / f"{sample_id}.npy",
        root / sample_id / f"{model}.npy",
        root / model / f"{sample_id}.npy",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing prediction for sample '{sample_id}' model '{model}' under {root}"
    )


def _write_prediction(
    cache: DiskPredictionCache,
    sample_id: str,
    model: str,
    prediction_path: Path,
    *,
    schema: str,
    coordinate_space: str,
    checkpoint: str,
    refresh: bool,
) -> bool:
    points = np.load(str(prediction_path)).astype("float32")
    prediction = LandmarkPrediction(
        landmarks=points,
        schema=schema,
        model_name=model,
        source_landmark_count=points.shape[0],
        coordinate_space=coordinate_space,
    )
    config = {
        "schema": schema,
        "coordinate_space": coordinate_space,
        "source": str(prediction_path),
    }
    fresh = cache.is_fresh(sample_id, prediction, checkpoint=checkpoint, config=config)
    cache.write(
        sample_id,
        prediction,
        checkpoint=checkpoint,
        config=config,
        refresh=refresh,
    )
    return refresh or not fresh


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--prediction-root", action="append", default=[])
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--prediction", default="")
    parser.add_argument("--schema", default="2d_68")
    parser.add_argument("--coordinate-space", default="frame")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    cache = DiskPredictionCache(args.cache_dir)

    if args.manifest:
        if not args.models:
            raise SystemExit("--models is required with --manifest")
        roots = _parse_prediction_roots(args.prediction_root)
        manifest_path = Path(args.manifest).expanduser().resolve()
        samples = load_manifest(manifest_path)
        entries = {
            str(entry.get("sample_id") or entry.get("id") or entry.get("name")): entry
            for entry in _load_manifest_entries(manifest_path)
        }
        written = 0
        reused = 0
        for sample in samples:
            entry = entries.get(sample.sample_id, {})
            for model in _split_models(args.models):
                path = _prediction_path(
                    sample_id=sample.sample_id,
                    model=model,
                    manifest_path=manifest_path,
                    entry=entry,
                    prediction_roots=roots,
                )
                changed = _write_prediction(
                    cache,
                    sample.sample_id,
                    model,
                    path,
                    schema=args.schema,
                    coordinate_space=args.coordinate_space,
                    checkpoint=args.checkpoint,
                    refresh=args.refresh,
                )
                written += int(changed)
                reused += int(not changed)
        print(f"Cached predictions: wrote={written} reused={reused}")
        return 0

    if not (args.sample_id and args.model and args.prediction):
        raise SystemExit("--sample-id, --model, and --prediction are required without --manifest")
    changed = _write_prediction(
        cache,
        args.sample_id,
        args.model,
        Path(args.prediction).expanduser().resolve(),
        schema=args.schema,
        coordinate_space=args.coordinate_space,
        checkpoint=args.checkpoint,
        refresh=args.refresh,
    )
    message = (
        "Cached predictions: wrote=1 reused=0"
        if changed
        else "Cached predictions: wrote=0 reused=1"
    )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
