#!/usr/bin/env python3
"""Canonical scorer-row dataset helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

from lib.landmarks.ensemble.runtime_resolver_scorer_data import CandidateQualityRow
from lib.landmarks.pipeline_conventions import write_json

TaggedRow = tuple[CandidateQualityRow, str]

SCORER_DATASET_DIR = "scorer_dataset"
SCORER_ROWS_CSV = "rows.csv"
SCORER_ROWS_PARQUET = "rows.parquet"
SCORER_DATASET_MANIFEST_JSON = "manifest.json"


@dataclass(frozen=True)
class ScorerDataset:
    """Rows and metadata loaded from the canonical scorer dataset."""

    rows: tuple[dict[str, T.Any], ...]
    train_rows: tuple[dict[str, T.Any], ...]
    eval_rows: tuple[dict[str, T.Any], ...]
    feature_names: tuple[str, ...]
    manifest: dict[str, T.Any]
    rows_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_names(rows: T.Sequence[TaggedRow]) -> tuple[str, ...]:
    return tuple(sorted({name for row, _source in rows for name in row.feature_values}))


def _fieldnames(feature_names: T.Sequence[str]) -> list[str]:
    base_fieldnames = [
        "split",
        "source",
        "sample_id",
        "face_index",
        "dataset",
        "condition",
        "candidate_name",
        "candidate_nme",
        "oracle_nme",
        "regret_vs_oracle",
        "normalized_regret",
        "failure_label",
        "large_regret_label",
        "candidate_failure_or_high_gap",
        "selection_cost",
        "transform_cost_v3",
        "transform_oracle_cost_v3",
        "transform_regret_v3",
        "transform_oracle_candidate_v3",
        "transform_oracle_gap_v3",
        "rankable_v3",
        "hard_invalid_v3",
        "hard_invalid_reasons_v3",
        "soft_structural_penalty_v3",
        "is_oracle",
        "was_selected_by_current_policy",
        "gap_vs_oracle",
        "runtime_bucket",
        "runtime_bucket_source",
        "hard_case_tags",
        "risk_route",
        "geometry_veto_reasons",
        "selected_by_current_policy",
        "selected_candidate_missing_from_eval",
        "oracle",
        "features_json",
    ]
    dynamic = [name for name in feature_names if name not in base_fieldnames]
    return [*base_fieldnames, *dynamic]


def _write_rows_csv(
    *,
    train_rows: T.Sequence[TaggedRow],
    eval_rows: T.Sequence[TaggedRow],
    path: Path,
    feature_names: T.Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(feature_names))
        writer.writeheader()
        for split, rows in (("train", train_rows), ("eval", eval_rows)):
            for row, source in rows:
                writer.writerow(
                    {
                        "split": split,
                        "source": source,
                        **row.to_csv_row(),
                        **{name: row.feature_values.get(name, 0.0) for name in feature_names},
                    }
                )
    return path


def write_scorer_dataset(
    *,
    train_rows: T.Sequence[TaggedRow],
    eval_rows: T.Sequence[TaggedRow],
    output_dir: Path,
    inputs: T.Mapping[str, T.Any],
    config: T.Mapping[str, T.Any],
) -> dict[str, T.Any]:
    """Write canonical scorer rows plus a manifest and return manifest payload."""

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [*train_rows, *eval_rows]
    feature_names = _feature_names(all_rows)
    rows_path = _write_rows_csv(
        train_rows=train_rows,
        eval_rows=eval_rows,
        path=output_dir / SCORER_ROWS_CSV,
        feature_names=feature_names,
    )
    train_samples = {
        (source, row.dataset, row.sample_id, row.face_index) for row, source in train_rows
    }
    eval_samples = {
        (source, row.dataset, row.sample_id, row.face_index) for row, source in eval_rows
    }
    manifest = {
        "artifact_schema_version": 1,
        "rows": SCORER_ROWS_CSV,
        "rows_sha256": _sha256_file(rows_path),
        "row_count": len(all_rows),
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "train_sample_count": len(train_samples),
        "eval_sample_count": len(eval_samples),
        "feature_count": len(feature_names),
        "features": list(feature_names),
        "inputs": dict(inputs),
        "config": dict(config),
    }
    manifest_path = write_json(output_dir / SCORER_DATASET_MANIFEST_JSON, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["rows_path"] = str(rows_path)
    return manifest


def _coerce_csv_value(value: str) -> T.Any:
    raw = value.strip()
    if raw == "":
        return ""
    if raw in {"0", "1"}:
        return int(raw)
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return value


def _load_rows_csv(path: Path) -> tuple[dict[str, T.Any], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, T.Any]] = []
        for row in reader:
            parsed = {key: _coerce_csv_value(value or "") for key, value in row.items()}
            features_json = parsed.get("features_json")
            if isinstance(features_json, str) and features_json:
                try:
                    parsed["features"] = json.loads(features_json)
                except json.JSONDecodeError:
                    parsed["features"] = {}
            else:
                parsed["features"] = {}
            rows.append(parsed)
    return tuple(rows)


def resolve_scorer_dataset_path(path: Path) -> tuple[Path, Path, dict[str, T.Any]]:
    """Return `(dataset_dir, rows_path, manifest)` for any supported dataset path."""

    if path.is_dir():
        dataset_dir = path
        manifest_path = dataset_dir / SCORER_DATASET_MANIFEST_JSON
    elif path.name == SCORER_DATASET_MANIFEST_JSON:
        dataset_dir = path.parent
        manifest_path = path
    else:
        dataset_dir = path.parent
        manifest_path = dataset_dir / SCORER_DATASET_MANIFEST_JSON

    manifest: dict[str, T.Any] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = loaded

    rows_name = str(manifest.get("rows") or SCORER_ROWS_CSV)
    rows_path = (
        path
        if path.is_file() and path.name != SCORER_DATASET_MANIFEST_JSON
        else dataset_dir / rows_name
    )
    return dataset_dir, rows_path, manifest


def read_scorer_dataset(path: Path) -> ScorerDataset:
    """Read a canonical scorer dataset from a dir, manifest, or rows CSV."""

    _dataset_dir, rows_path, manifest = resolve_scorer_dataset_path(path)
    rows = _load_rows_csv(rows_path)
    train_rows = tuple(row for row in rows if str(row.get("split") or "") == "train")
    eval_rows = tuple(row for row in rows if str(row.get("split") or "") == "eval")
    feature_names = tuple(str(name) for name in manifest.get("features", ()) or ())
    if not feature_names and rows:
        feature_names = tuple(sorted((rows[0].get("features") or {}).keys()))
    return ScorerDataset(
        rows=rows,
        train_rows=train_rows,
        eval_rows=eval_rows,
        feature_names=feature_names,
        manifest=manifest,
        rows_path=rows_path,
    )


__all__ = [
    "SCORER_DATASET_DIR",
    "SCORER_DATASET_MANIFEST_JSON",
    "SCORER_ROWS_CSV",
    "SCORER_ROWS_PARQUET",
    "ScorerDataset",
    "read_scorer_dataset",
    "resolve_scorer_dataset_path",
    "write_scorer_dataset",
]
