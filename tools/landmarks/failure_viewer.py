#!/usr/bin/env python3
"""Create landmark failure overlays and worst-first contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import typing as T
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.eval.harness import _fuse_variant, load_manifest
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.eval.visualize import (
    write_contact_sheet,
    write_debug_records,
    write_overlay,
)

DEFAULT_MODELS = ("hrnet", "spiga", "orformer")


def _parse_csv_list(value: str, default: T.Sequence[str]) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or tuple(default)


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "sample"


def _load_metrics_rows(path: str | Path) -> list[dict[str, T.Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _write_records(
    records: list[dict[str, T.Any]],
    output_path: Path,
    *,
    fieldnames: T.Sequence[str] | None = None,
) -> None:
    output_path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output_path.with_suffix(".csv")
    if not records:
        csv_path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames or sorted({key for record in records for key in record}))
    with csv_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _legacy_contact_sheet(args: argparse.Namespace) -> int:
    rows = sorted(
        _load_metrics_rows(args.metrics),
        key=lambda row: row.get("nme", 0),
        reverse=True,
    )
    images = [f"{args.debug_dir}/{row['sample_id']}.png" for row in rows[: args.limit]]
    write_contact_sheet(images, args.output or "outputs/landmark_debug/contact_sheet.png")
    return 0


def _row_label(row: T.Mapping[str, T.Any]) -> str:
    variant = str(row.get("variant", ""))
    if variant and variant != "single":
        return variant
    return str(row.get("model", "model"))


def _fused_for_row(
    row: T.Mapping[str, T.Any],
    predictions: dict[str, np.ndarray],
    cache: DiskPredictionCache,
    sample_id: str,
    *,
    models: T.Sequence[str],
    weights: dict[str, list[float]],
    outlier_threshold: float,
) -> tuple[np.ndarray | None, dict[str, list[int]], int]:
    variant = str(row.get("variant", ""))
    if variant in ("", "single"):
        return None, {}, 0
    available = tuple(name for name in models if name in predictions)
    if len(available) < 2:
        return None, {}, 0
    cached_predictions = [cache.read(sample_id, name) for name in available]
    fused, rejected_count = _fuse_variant(
        variant,
        cached_predictions,
        available,
        weights,
        outlier_threshold=outlier_threshold,
    )
    rejected: dict[str, list[int]] = {}
    for model_index, model_name in enumerate(available):
        indexes = np.flatnonzero(fused.weights[model_index] <= 0).astype(int).tolist()
        if indexes:
            rejected[model_name] = indexes
    return fused.points, rejected, rejected_count


def _write_case_overlay(
    row: T.Mapping[str, T.Any],
    rank: int,
    *,
    samples: dict[str, T.Any],
    cache: DiskPredictionCache,
    models: T.Sequence[str],
    weights: dict[str, list[float]],
    output_dir: Path,
    outlier_threshold: float,
) -> dict[str, T.Any] | None:
    sample_id = str(row.get("sample_id", ""))
    sample = samples.get(sample_id)
    if sample is None:
        return None
    truth = np.load(sample.landmarks).astype("float32")
    predictions: dict[str, np.ndarray] = {"ground_truth": truth}
    available_models = cache.available_models(sample_id)
    cached_model_names = tuple(name for name in models if name in available_models)
    for name in cached_model_names:
        predictions[name] = cache.read(sample_id, name).landmarks
    fused, rejected, rejected_count = _fused_for_row(
        row,
        predictions,
        cache,
        sample_id,
        models=cached_model_names,
        weights=weights,
        outlier_threshold=outlier_threshold,
    )
    if fused is not None:
        predictions["fused"] = fused
    label = _row_label(row)
    overlay_path = (
        output_dir
        / "overlays"
        / f"{rank:03d}_{_safe_filename(sample_id)}_{_safe_filename(label)}.png"
    )
    write_overlay(sample.image, predictions, overlay_path, rejected_landmarks=rejected)
    return {
        "sample_id": sample_id,
        "dataset": sample.dataset,
        "condition": sample.condition,
        "model": row.get("model", ""),
        "variant": row.get("variant", ""),
        "nme": row.get("nme", ""),
        "best_single_model": row.get("best_single_model", ""),
        "best_single_nme": row.get("best_single_nme", ""),
        "delta_vs_best_single": row.get("delta_vs_best_single", ""),
        "rejected_landmarks": rejected_count or row.get("rejected_landmarks", 0),
        "rejected_model_landmarks": rejected,
        "overlay": str(overlay_path),
    }


def _write_contact_if_possible(paths: list[str], output_path: Path) -> None:
    if not paths:
        return
    write_contact_sheet(paths, output_path)


def _run_rich_failure_viewer(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir or args.debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = DiskPredictionCache(args.cache_dir)
    models = _parse_csv_list(args.models, DEFAULT_MODELS)
    weights = load_weights(args.weights) if args.weights else {}
    samples = {sample.sample_id: sample for sample in load_manifest(args.manifest)}
    rows = _load_metrics_rows(args.metrics)
    worst_rows = sorted(rows, key=lambda row: float(row.get("nme", 0)), reverse=True)[: args.limit]
    regression_rows = [
        row
        for row in rows
        if str(row.get("model")) == "ensemble"
        and row.get("delta_vs_best_single") not in ("", None)
        and float(row.get("delta_vs_best_single", 0)) > 0
    ][: args.limit]

    debug_records: list[dict[str, T.Any]] = []
    worst_records: list[dict[str, T.Any]] = []
    for rank, row in enumerate(worst_rows, start=1):
        record = _write_case_overlay(
            row,
            rank,
            samples=samples,
            cache=cache,
            models=models,
            weights=weights,
            output_dir=output_dir,
            outlier_threshold=args.outlier_threshold,
        )
        if record is not None:
            worst_records.append(record)
            debug_records.append(record)

    regression_records: list[dict[str, T.Any]] = []
    for rank, row in enumerate(regression_rows, start=1):
        record = _write_case_overlay(
            row,
            rank,
            samples=samples,
            cache=cache,
            models=models,
            weights=weights,
            output_dir=output_dir / "ensemble_regressions",
            outlier_threshold=args.outlier_threshold,
        )
        if record is not None:
            regression_records.append(record)
            debug_records.append(record)

    write_debug_records(debug_records, output_dir)
    _write_records(worst_records, output_dir / "worst_cases.json")
    _write_records(regression_records, output_dir / "ensemble_regressions.json")
    _write_contact_if_possible(
        [str(record["overlay"]) for record in worst_records],
        output_dir / "worst_contact_sheet.png",
    )
    _write_contact_if_possible(
        [str(record["overlay"]) for record in regression_records],
        output_dir / "ensemble_regressions_contact_sheet.png",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--debug-dir", default="outputs/landmark_debug")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--weights", default="")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    args = parser.parse_args(argv)
    if args.limit < 1:
        raise ValueError("--limit must be greater than zero")
    if not args.manifest:
        return _legacy_contact_sheet(args)
    return _run_rich_failure_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
