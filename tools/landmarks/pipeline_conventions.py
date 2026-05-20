#!/usr/bin/env python3
"""Shared path, source-label, and sidecar conventions for landmark tools."""

from __future__ import annotations

import csv
import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

from lib.landmarks.datasets.manifest_io import LandmarkSample, load_manifest

SOURCE_GT_HARD = "gt_hard"
SOURCE_PRODUCTION_VALIDATED = "production_validated"
SOURCE_PRODUCTION_EXTRACTION = "production_extraction"
METADATA_SOURCES = (
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
    SOURCE_PRODUCTION_EXTRACTION,
)

MANIFEST_FILENAME = "manifest.json"
PREDICTION_CACHE_DIRNAME = "cache"
RESOLVER_METADATA_JSONL = "resolver_metadata.jsonl"
SCORER_REPORT_DIRNAME = "scorer_reports"
DEBUG_DIRNAME = "debug"
DEBUG_JSON_SUFFIX = ".json"
DEBUG_JSONL_SUFFIX = ".jsonl"

SCORER_METRICS_JSON = "scorer_metrics.json"
SCORER_POLICY_REPORT_JSON = "scorer_policy_report.json"
SCORER_HELDOUT_POLICY_REPORT_JSON = "scorer_policy_eval_report.json"
SCORER_POLICY_REPORT_CSV = "scorer_policy_report.csv"
SCORER_WORST_SAMPLES_JSON = "scorer_worst_samples.json"
SCORER_FEATURE_IMPORTANCE_CSV = "scorer_feature_importance.csv"


@dataclass(frozen=True)
class ManifestCachePair:
    """Explicit manifest/cache pair for a scorer or promotion source."""

    source: str
    manifest_path: Path
    cache_dir: Path

    def __post_init__(self) -> None:
        if self.source not in METADATA_SOURCES:
            raise ValueError(
                f"source must be one of {METADATA_SOURCES}, got {self.source!r}"
            )


def normalize_source_label(value: str) -> str:
    """Return a canonical metadata source label."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "production": SOURCE_PRODUCTION_VALIDATED,
        "prod": SOURCE_PRODUCTION_VALIDATED,
        "production_validated_manifest": SOURCE_PRODUCTION_VALIDATED,
        "prod_extract": SOURCE_PRODUCTION_EXTRACTION,
        "production_extract": SOURCE_PRODUCTION_EXTRACTION,
        "gt": SOURCE_GT_HARD,
        "hard": SOURCE_GT_HARD,
        "gt_hard_validation": SOURCE_GT_HARD,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in METADATA_SOURCES:
        raise ValueError(f"unknown landmark metadata source label {value!r}")
    return normalized


def metadata_key(sample_id: str, face_index: int | None = 0) -> tuple[str, int]:
    """Return the stable resolver sidecar lookup key."""
    return str(sample_id).strip(), int(face_index or 0)


def face_index_from_metadata(metadata: T.Mapping[str, T.Any] | None) -> int:
    """Return the normalized face index from manifest or sidecar metadata."""
    if not isinstance(metadata, T.Mapping):
        return 0
    value = metadata.get("face_index", metadata.get("face", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError) as err:
        raise ValueError(f"face_index must be an integer-compatible value, got {value!r}") from err


def face_index_for_sample(sample: LandmarkSample) -> int:
    """Return the face index for a loaded manifest sample."""
    return face_index_from_metadata(sample.metadata)


def expected_metadata_keys(samples: T.Iterable[LandmarkSample]) -> set[tuple[str, int]]:
    """Return the resolver sidecar keys expected by a manifest."""
    return {metadata_key(sample.sample_id, face_index_for_sample(sample)) for sample in samples}


def runtime_bucket_from_resolver_metadata(row: T.Mapping[str, T.Any]) -> str | None:
    """Return runtime bucket from a resolver metadata sidecar row."""
    le = row.get("landmark_ensemble")
    if not isinstance(le, T.Mapping):
        return None

    resolver = le.get("resolver")
    if not isinstance(resolver, T.Mapping):
        resolver = {}

    bucket = (
        le.get("runtime_bucket")
        or le.get("bucket")
        or resolver.get("runtime_bucket")
        or resolver.get("bucket")
    )
    return str(bucket) if bucket else None


def load_resolver_metadata_sidecar(path: Path | None) -> dict[tuple[str, int], dict[str, T.Any]]:
    """Load resolver metadata JSONL keyed by `(sample_id, face_index)`.

    Each non-empty line must contain `sample_id`; `face_index` defaults to 0.
    Duplicate keys fail fast because they make runtime bucket provenance
    ambiguous.
    """
    if path is None:
        return {}

    records: dict[tuple[str, int], dict[str, T.Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_num} resolver metadata row must be a JSON object")
            sample_id = row.get("sample_id")
            if not sample_id:
                raise ValueError(f"{path}:{line_num} missing sample_id")
            key = metadata_key(str(sample_id), face_index_from_metadata(row))
            if key in records:
                raise ValueError(f"{path}:{line_num} duplicate resolver metadata key {key}")
            records[key] = row
    return records


def validate_resolver_metadata_for_samples(
    samples: T.Sequence[LandmarkSample],
    metadata: T.Mapping[tuple[str, int], T.Mapping[str, T.Any]],
    *,
    source: str,
    require_complete: bool,
) -> None:
    """Validate resolver sidecar keys and bucket payloads against a manifest."""
    source = normalize_source_label(source)
    expected = expected_metadata_keys(samples)
    observed = set(metadata)
    missing = sorted(expected - observed)
    extras = sorted(observed - expected)
    if extras:
        raise ValueError(
            f"{source} resolver metadata contains {len(extras)} key(s) not present in manifest; "
            f"examples: {extras[:10]}"
        )
    if require_complete and missing:
        raise ValueError(
            f"{source} resolver metadata missing {len(missing)} manifest key(s); "
            f"examples: {missing[:10]}"
        )
    for key, row in metadata.items():
        if key not in expected:
            continue
        if runtime_bucket_from_resolver_metadata(row) is None:
            raise ValueError(f"{source} resolver metadata key {key} has no runtime bucket")


def validate_resolver_metadata_for_manifest(
    manifest_path: Path,
    metadata: T.Mapping[tuple[str, int], T.Mapping[str, T.Any]],
    *,
    source: str,
    require_complete: bool,
) -> None:
    """Load a manifest and validate resolver sidecar metadata against it."""
    validate_resolver_metadata_for_samples(
        load_manifest(manifest_path),
        metadata,
        source=source,
        require_complete=require_complete,
    )


def require_manifest_cache_pair(
    *,
    source: str,
    manifest_path: Path | None,
    cache_dir: Path | None,
) -> ManifestCachePair | None:
    """Return a normalized manifest/cache pair or fail on half-specified inputs."""
    source = normalize_source_label(source)
    if manifest_path is None and cache_dir is None:
        return None
    if manifest_path is None or cache_dir is None:
        raise ValueError(f"{source} manifest/cache inputs must be supplied together")
    return ManifestCachePair(source=source, manifest_path=manifest_path, cache_dir=cache_dir)


def write_json(path: Path, payload: T.Any) -> Path:
    """Write a deterministic JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: T.Iterable[T.Mapping[str, T.Any]]) -> Path:
    """Write a deterministic JSONL artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def write_csv(path: Path, rows: T.Sequence[T.Mapping[str, T.Any]], fieldnames: T.Sequence[str] | None = None) -> Path:
    """Write rows to CSV with stable field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (list(rows[0]) if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        if names:
            writer.writeheader()
            writer.writerows(rows)
    return path


__all__ = [
    "DEBUG_DIRNAME",
    "DEBUG_JSON_SUFFIX",
    "DEBUG_JSONL_SUFFIX",
    "MANIFEST_FILENAME",
    "METADATA_SOURCES",
    "PREDICTION_CACHE_DIRNAME",
    "RESOLVER_METADATA_JSONL",
    "SCORER_FEATURE_IMPORTANCE_CSV",
    "SCORER_HELDOUT_POLICY_REPORT_JSON",
    "SCORER_METRICS_JSON",
    "SCORER_POLICY_REPORT_CSV",
    "SCORER_POLICY_REPORT_JSON",
    "SCORER_REPORT_DIRNAME",
    "SCORER_WORST_SAMPLES_JSON",
    "SOURCE_GT_HARD",
    "SOURCE_PRODUCTION_EXTRACTION",
    "SOURCE_PRODUCTION_VALIDATED",
    "ManifestCachePair",
    "expected_metadata_keys",
    "face_index_for_sample",
    "face_index_from_metadata",
    "load_resolver_metadata_sidecar",
    "metadata_key",
    "normalize_source_label",
    "require_manifest_cache_pair",
    "runtime_bucket_from_resolver_metadata",
    "validate_resolver_metadata_for_manifest",
    "validate_resolver_metadata_for_samples",
    "write_csv",
    "write_json",
    "write_jsonl",
]
