#!/usr/bin/env python3
"""Post-build audit polish helpers for landmark quality datasets."""

from __future__ import annotations

import json
import logging
import statistics
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.datasets.sources import OFFICIAL_SOURCE_NOTES

logger = logging.getLogger(__name__)
LANDMARK_AUDIT_FEATURES = (
    "bbox_area",
    "bbox_width",
    "bbox_height",
    "landmark_x_span",
    "landmark_y_span",
    "landmark_centroid_x",
    "landmark_centroid_y",
)


def _read_json(path: Path, default: T.Any) -> T.Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: T.Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_entries(output_dir: str | Path) -> tuple[dict[str, T.Any], list[dict[str, T.Any]]]:
    """Return manifest payload and entries, accepting legacy manifest keys."""
    root = Path(output_dir)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.json not found: {manifest}")
    payload = _read_json(manifest, {})
    entries = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(entries, list):
        raise ValueError("manifest.json must contain a samples or scenarios list")
    return payload, [dict(entry) for entry in entries if isinstance(entry, dict)]


def rewrite_manifest(
    output_dir: str | Path, payload: dict[str, T.Any], entries: list[dict[str, T.Any]]
) -> None:
    """Rewrite manifest entries after post-processing."""
    root = Path(output_dir)
    if "samples" in payload:
        payload["samples"] = entries
    elif "scenarios" in payload:
        payload["scenarios"] = entries
    else:
        payload["samples"] = entries
    _write_json(root / "manifest.json", payload)


def entry_path(value: str, output_dir: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(output_dir) / path


def entry_conditions(entry: dict[str, T.Any]) -> tuple[str, ...]:
    raw = entry.get("conditions")
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, (list, tuple)):
        values = tuple(str(item) for item in raw if str(item))
    else:
        values = (str(entry.get("condition", "default")),)
    return tuple(dict.fromkeys(value for value in values if value)) or ("default",)


def source_key(entry: dict[str, T.Any]) -> tuple[str, str]:
    source = entry.get("source", {}) if isinstance(entry.get("source"), dict) else {}
    dataset = str(source.get("dataset") or entry.get("dataset") or "")
    source_id = (
        source.get("source_id")
        or source.get("image_id")
        or source.get("sample_id")
        or entry.get("image")
        or entry.get("sample_id")
        or entry.get("name")
        or ""
    )
    return dataset, str(source_id)


def source_key_json(entry: dict[str, T.Any]) -> list[str]:
    return list(source_key(entry))


def duplicate_source_audit(entries: list[dict[str, T.Any]]) -> list[dict[str, T.Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for entry in entries:
        key = source_key(entry)
        for condition in entry_conditions(entry):
            grouped.setdefault(key, []).append(
                {
                    "sample_id": str(entry.get("sample_id") or entry.get("name") or ""),
                    "condition": condition,
                }
            )
    duplicates: list[dict[str, T.Any]] = []
    for key, refs in grouped.items():
        groups = sorted({ref["condition"] for ref in refs})
        if len(groups) > 1:
            duplicates.append(
                {"source_key": list(key), "condition_groups": groups, "entries": refs}
            )
    return sorted(duplicates, key=lambda item: tuple(item["source_key"]))


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"min": ordered[0], "median": float(statistics.median(ordered)), "max": ordered[-1]}


def _landmark_features(entry: dict[str, T.Any], output_dir: Path) -> dict[str, float]:
    features: dict[str, float] = {}
    metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
    bbox = metadata.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        features.update({"bbox_width": width, "bbox_height": height, "bbox_area": width * height})
    landmark_value = str(entry.get("landmarks", ""))
    landmark_path = entry_path(landmark_value, output_dir) if landmark_value else None
    if landmark_path is not None and landmark_path.is_file():
        points = np.load(str(landmark_path)).astype("float32")
        if points.ndim == 2 and points.shape[1] >= 2 and points.size:
            features.update(
                {
                    "landmark_x_span": float(np.max(points[:, 0]) - np.min(points[:, 0])),
                    "landmark_y_span": float(np.max(points[:, 1]) - np.min(points[:, 1])),
                    "landmark_centroid_x": float(np.mean(points[:, 0])),
                    "landmark_centroid_y": float(np.mean(points[:, 1])),
                }
            )
    return features


def _shortfall_groups(audit: dict[str, T.Any]) -> list[dict[str, T.Any]]:
    existing = audit.get("shortfall_groups")
    if isinstance(existing, list):
        return existing
    raw = audit.get("condition_shortfalls", {})
    if not isinstance(raw, dict):
        return []
    result = []
    for group, payload in sorted(raw.items()):
        result.append(
            {"scenario_group": group, **payload}
            if isinstance(payload, dict)
            else {"scenario_group": group, "shortfall": payload}
        )
    return result


def enrich_dataset_audit(output_dir: str | Path, *, allow_overlap: bool = False) -> Path:
    """Write AutoMask-donor-parity fields into landmark dataset_audit.json."""
    root = Path(output_dir)
    _payload, entries = manifest_entries(root)
    audit_path = root / "dataset_audit.json"
    audit = _read_json(audit_path, {})
    count_per_condition: dict[str, int] = {}
    count_per_dataset: dict[str, int] = {}
    count_per_source_schema: dict[str, int] = {}
    selected_by_group: dict[str, list[list[str]]] = {}
    feature_values: dict[str, list[float]] = {feature: [] for feature in LANDMARK_AUDIT_FEATURES}
    feature_values_by_group: dict[str, dict[str, list[float]]] = {}

    for entry in entries:
        dataset = str(entry.get("dataset") or entry.get("source", {}).get("dataset") or "")
        count_per_dataset[dataset] = count_per_dataset.get(dataset, 0) + 1
        schema = str(entry.get("source_schema", "2d_68"))
        count_per_source_schema[schema] = count_per_source_schema.get(schema, 0) + 1
        features = _landmark_features(entry, root)
        for condition in entry_conditions(entry):
            count_per_condition[condition] = count_per_condition.get(condition, 0) + 1
            selected_by_group.setdefault(condition, []).append(source_key_json(entry))
            feature_values_by_group.setdefault(
                condition, {name: [] for name in LANDMARK_AUDIT_FEATURES}
            )
            for feature, value in features.items():
                feature_values[feature].append(value)
                feature_values_by_group[condition][feature].append(value)

    duplicates = audit.get("duplicate_source_ids")
    if not isinstance(duplicates, list):
        duplicates = duplicate_source_audit(entries)
    unique_sources = {tuple(source_key_json(entry)) for entry in entries}
    feature_stats = {
        feature: stats
        for feature, values in feature_values.items()
        if (stats := _stats(values)) is not None
    }
    feature_stats_by_group = {
        group: {
            feature: stats
            for feature, values in by_feature.items()
            if (stats := _stats(values)) is not None
        }
        for group, by_feature in sorted(feature_values_by_group.items())
    }
    audit.update(
        {
            "schema_version": audit.get("schema_version", 1),
            "total_entries": len(entries),
            "count_per_condition": dict(sorted(count_per_condition.items())),
            "condition_counts": audit.get(
                "condition_counts", dict(sorted(count_per_condition.items()))
            ),
            "count_per_scenario_group": dict(sorted(count_per_condition.items())),
            "count_per_dataset": dict(sorted(count_per_dataset.items())),
            "count_per_source_schema": dict(sorted(count_per_source_schema.items())),
            "count_per_suite": {"landmark_quality": len(entries)},
            "unique_source_count": len(unique_sources),
            "duplicate_source_ids": duplicates,
            "overlap": {
                "allow_overlap": allow_overlap,
                "has_overlap": bool(duplicates),
                "duplicate_count": len(duplicates),
            },
            "rejected_candidates_due_to_exclusivity": audit.get(
                "rejected_candidates_due_to_exclusivity", []
            ),
            "shortfall_groups": _shortfall_groups(audit),
            "feature_stats": feature_stats,
            "feature_stats_by_group": feature_stats_by_group,
            "selected_source_ids_per_group": {
                group: values for group, values in sorted(selected_by_group.items())
            },
            "landmark_quality_entry_count": len(entries),
            "audit_schema_family": "automask_dataset_audit_v1",
        }
    )
    _write_json(audit_path, audit)
    logger.info("Polished landmark dataset audit: %s", audit_path)
    return audit_path


def write_source_notes(output_dir: str | Path) -> Path:
    """Write landmark-specific provenance/source notes."""
    root = Path(output_dir)
    _payload, entries = manifest_entries(root)
    datasets = sorted(
        {
            str(entry.get("dataset") or entry.get("source", {}).get("dataset") or "")
            for entry in entries
        }
    )
    display_names = {
        "wflw": "WFLW",
        "cofw": "COFW",
        "merl-rav": "MERL-RAV",
        "aflw2000-3d": "AFLW2000-3D",
    }
    lines = [
        "# Landmark quality dataset source notes",
        "",
        "This directory was populated by `tools/landmarks/build_quality_dataset.py`.",
        "",
        "Review upstream dataset terms before use or redistribution. Do not commit "
        "generated images, landmarks, overlays, or manifests unless licensing has been "
        "reviewed.",
        "",
        "## Audit files",
        "",
        "`dataset_audit.json` follows the AutoMask donor audit shape where applicable: "
        "counts, overlap metadata, selected source IDs, shortfalls, rejected candidates, "
        "and feature statistics.",
        "",
        "## Dataset notes",
        "",
    ]
    for dataset in datasets or ["unknown"]:
        display = display_names.get(dataset, dataset or "unknown")
        lines.extend(
            [
                f"### {display}",
                "",
                OFFICIAL_SOURCE_NOTES.get(
                    display, "Review source-specific licensing and provenance before use."
                ),
                "",
            ]
        )
    path = root / "SOURCE_NOTES.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logger.info("Polished landmark source notes: %s", path)
    return path


def polish_landmark_dataset_artifacts(
    output_dir: str | Path, *, allow_overlap: bool = False
) -> dict[str, Path]:
    """Polish audit and source-note artifacts after build or audit-only runs."""
    return {
        "audit": enrich_dataset_audit(output_dir, allow_overlap=allow_overlap),
        "source_notes": write_source_notes(output_dir),
    }
