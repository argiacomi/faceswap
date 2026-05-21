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
METADATA_SOURCES = (SOURCE_GT_HARD, SOURCE_PRODUCTION_VALIDATED, SOURCE_PRODUCTION_EXTRACTION)

MANIFEST_FILENAME = "manifest.json"
REPORT_MANIFEST_FILENAME = "report_manifest.json"
RUN_SUMMARY_JSON = "run_summary.json"
SPLITS_FILENAME = "splits.json"
DATASET_AUDIT_JSON = "dataset_audit.json"
PREDICTION_CACHE_DIRNAME = "cache"
RESOLVER_METADATA_JSONL = "resolver_metadata.jsonl"
SCORER_REPORT_DIRNAME = "scorer_reports"
DEBUG_DIRNAME = "debug"
DEBUG_JSON_SUFFIX = ".json"
DEBUG_JSONL_SUFFIX = ".jsonl"

STATIC_WEIGHTS_FILENAME = "static_landmark_weights.json"
BEST_SETUP_FILENAME = "best_setup.json"
BEST_WEIGHTS_FILENAME = "best_weights.json"
CANDIDATE_RESULTS_CSV = "candidate_results.csv"
CANDIDATE_RESULTS_JSON = "candidate_results.json"
PROMOTION_REPORT_MD = "promotion_report.md"
NO_PROMOTION_JSON = "no_promotion.json"

GEOMETRY_METRICS_JSON = "geometry_metrics.json"
GEOMETRY_METRICS_CSV = "geometry_metrics.csv"
PER_REGION_GEOMETRY_CSV = "per_region_geometry.csv"
CATASTROPHIC_GEOMETRY_FAILURES_CSV = "catastrophic_geometry_failures.csv"
WORST_GEOMETRY_FAILURES_DIRNAME = "worst_geometry_failures"
WORST_SAMPLES_JSON = "worst_samples.json"

SIGNAL_CANDIDATE_INDEX_CSV = "candidate_index.csv"
SIGNAL_VALIDATION_REPORT_JSON = "signal_validation_report.json"
SIGNAL_VALIDATION_REPORT_CSV = "signal_validation_report.csv"
SELECTOR_ABLATIONS_JSON = "selector_ablations.json"
SELECTOR_ABLATIONS_CSV = "selector_ablations.csv"

AFLW_PROFILE_METRICS_JSON = "aflw_profile_metrics.json"
AFLW_PROFILE_METRICS_CSV = "aflw_profile_metrics.csv"
AFLW_REGION_FAILURES_CSV = "aflw_region_failures.csv"

PRODUCTION_PROMOTION_REPORT_JSON = "production_promotion_report.json"
PRODUCTION_PROMOTION_REPORT_MD = "production_promotion_report.md"
PRODUCTION_PER_BUCKET_CSV = "production_per_bucket_metrics.csv"
PRODUCTION_POLICY_FAILURES_CSV = "production_policy_failures.csv"
PRODUCTION_WORST_SAMPLES_JSON = "production_worst_samples.json"

FAILURE_WORST_CASES_JSON = "worst_cases.json"
FAILURE_ENSEMBLE_REGRESSIONS_JSON = "ensemble_regressions.json"
FAILURE_WORST_CONTACT_SHEET = "worst_contact_sheet.png"
FAILURE_ENSEMBLE_REGRESSIONS_CONTACT_SHEET = "ensemble_regressions_contact_sheet.png"

SCORER_METRICS_JSON = "scorer_metrics.json"
SCORER_POLICY_REPORT_JSON = "scorer_policy_report.json"
SCORER_HELDOUT_POLICY_REPORT_JSON = "scorer_policy_eval_report.json"
SCORER_POLICY_REPORT_CSV = "scorer_policy_report.csv"
SCORER_WORST_SAMPLES_JSON = "scorer_worst_samples.json"
SCORER_FEATURE_IMPORTANCE_CSV = "scorer_feature_importance.csv"


class ResolverMetadataValidationError(RuntimeError, ValueError):
    """Resolver sidecar validation error compatible with legacy callers."""


@dataclass(frozen=True)
class ManifestCachePair:
    """Explicit manifest/cache pair for a scorer or promotion source."""

    source: str
    manifest_path: Path
    cache_dir: Path

    def __post_init__(self) -> None:
        if self.source not in METADATA_SOURCES:
            raise ValueError(f"source must be one of {METADATA_SOURCES}, got {self.source!r}")


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
    """Load resolver metadata JSONL keyed by `(sample_id, face_index)`."""
    if path is None:
        return {}
    records: dict[tuple[str, int], dict[str, T.Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ResolverMetadataValidationError(
                    f"{path}:{line_num} resolver metadata row must be a JSON object"
                )
            sample_id = row.get("sample_id")
            if not sample_id:
                raise ResolverMetadataValidationError(f"{path}:{line_num} missing sample_id")
            key = metadata_key(str(sample_id), face_index_from_metadata(row))
            if key in records:
                raise ResolverMetadataValidationError(
                    f"{path}:{line_num} duplicate resolver metadata key {key}"
                )
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
        raise ResolverMetadataValidationError(
            f"{source} resolver metadata contains {len(extras)} key(s) not present in manifest; examples: {extras[:10]}"
        )
    if require_complete and missing:
        detail = f"{source} resolver metadata missing {len(missing)} manifest key(s); examples: {missing[:10]}"
        if source == SOURCE_GT_HARD:
            raise ResolverMetadataValidationError(
                "GT-hard sample missing stored resolver metadata. " + detail
            )
        raise ResolverMetadataValidationError(detail)
    for key, row in metadata.items():
        if key not in expected:
            continue
        if runtime_bucket_from_resolver_metadata(row) is None:
            raise ResolverMetadataValidationError(
                f"{source} resolver metadata key {key} has no runtime bucket"
            )


def validate_resolver_metadata_for_manifest(
    manifest_path: Path,
    metadata: T.Mapping[tuple[str, int], T.Mapping[str, T.Any]],
    *,
    source: str,
    require_complete: bool,
) -> None:
    """Load a manifest and validate resolver sidecar metadata against it."""
    validate_resolver_metadata_for_samples(
        load_manifest(manifest_path), metadata, source=source, require_complete=require_complete
    )


def require_manifest_cache_pair(
    *, source: str, manifest_path: Path | None, cache_dir: Path | None
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


def write_csv(
    path: Path, rows: T.Sequence[T.Mapping[str, T.Any]], fieldnames: T.Sequence[str] | None = None
) -> Path:
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
    "AFLW_PROFILE_METRICS_CSV",
    "AFLW_PROFILE_METRICS_JSON",
    "AFLW_REGION_FAILURES_CSV",
    "BEST_SETUP_FILENAME",
    "BEST_WEIGHTS_FILENAME",
    "CANDIDATE_RESULTS_CSV",
    "CANDIDATE_RESULTS_JSON",
    "CATASTROPHIC_GEOMETRY_FAILURES_CSV",
    "DATASET_AUDIT_JSON",
    "DEBUG_DIRNAME",
    "DEBUG_JSON_SUFFIX",
    "DEBUG_JSONL_SUFFIX",
    "FAILURE_ENSEMBLE_REGRESSIONS_CONTACT_SHEET",
    "FAILURE_ENSEMBLE_REGRESSIONS_JSON",
    "FAILURE_WORST_CASES_JSON",
    "FAILURE_WORST_CONTACT_SHEET",
    "GEOMETRY_METRICS_CSV",
    "GEOMETRY_METRICS_JSON",
    "MANIFEST_FILENAME",
    "METADATA_SOURCES",
    "NO_PROMOTION_JSON",
    "PREDICTION_CACHE_DIRNAME",
    "PER_REGION_GEOMETRY_CSV",
    "PRODUCTION_PER_BUCKET_CSV",
    "PRODUCTION_POLICY_FAILURES_CSV",
    "PRODUCTION_PROMOTION_REPORT_JSON",
    "PRODUCTION_PROMOTION_REPORT_MD",
    "PRODUCTION_WORST_SAMPLES_JSON",
    "PROMOTION_REPORT_MD",
    "REPORT_MANIFEST_FILENAME",
    "RESOLVER_METADATA_JSONL",
    "RUN_SUMMARY_JSON",
    "SCORER_FEATURE_IMPORTANCE_CSV",
    "SCORER_HELDOUT_POLICY_REPORT_JSON",
    "SCORER_METRICS_JSON",
    "SCORER_POLICY_REPORT_CSV",
    "SCORER_POLICY_REPORT_JSON",
    "SCORER_REPORT_DIRNAME",
    "SCORER_WORST_SAMPLES_JSON",
    "SELECTOR_ABLATIONS_CSV",
    "SELECTOR_ABLATIONS_JSON",
    "SIGNAL_CANDIDATE_INDEX_CSV",
    "SIGNAL_VALIDATION_REPORT_CSV",
    "SIGNAL_VALIDATION_REPORT_JSON",
    "SOURCE_GT_HARD",
    "SOURCE_PRODUCTION_EXTRACTION",
    "SOURCE_PRODUCTION_VALIDATED",
    "SPLITS_FILENAME",
    "STATIC_WEIGHTS_FILENAME",
    "WORST_GEOMETRY_FAILURES_DIRNAME",
    "WORST_SAMPLES_JSON",
    "ManifestCachePair",
    "ResolverMetadataValidationError",
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
