#!/usr/bin/env python3
"""Landmark quality dataset manifest helpers."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.datasets.sources import (
    DEFAULT_CACHE_DIR,
    DatasetSourceSpec,
    extract_archive_to_temp,
    is_archive,
    resolve_dataset_source,
)
from lib.landmarks.schema import normalize_landmarks

logger = logging.getLogger(__name__)
SUPPORTED_DATASETS = ("wflw", "cofw", "directory")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
MANIFEST_MODES = ("replace", "merge")
WFLW_SOURCE = DatasetSourceSpec(
    dataset="WFLW",
    cache_subdir="wflw",
    canonical_archive="wflw.zip",
    cache_aliases=("WFLW.zip", "WFLW.tar.gz", "WFLW.tgz"),
    extracted_aliases=("WFLW", "WFLW_images"),
    manual_hint=(
        "Provide --wflw-annotations and --image-root, or place a WFLW archive/extracted "
        "dataset under .fs_cache/landmark_quality/wflw."
    ),
)
COFW_SOURCE = DatasetSourceSpec(
    dataset="COFW",
    cache_subdir="cofw",
    canonical_archive="cofw.zip",
    cache_aliases=("COFW.zip", "COFW.tar.gz", "COFW.tgz", "cofw_68.json"),
    extracted_aliases=("COFW", "cofw"),
    manual_hint=(
        "Provide --cofw-json, or place cofw_68.json/an archive/extracted dataset under "
        ".fs_cache/landmark_quality/cofw."
    ),
)


def validate_manifest_mode(value: str) -> str:
    """Return a valid manifest write mode."""
    mode = value.strip().lower()
    if mode not in MANIFEST_MODES:
        raise ValueError(f"manifest mode must be one of {MANIFEST_MODES}, got {value!r}")
    return mode


def validate_samples_per_scenario(value: int | str | None) -> int | None:
    """Return a non-negative scenario sample limit or ``None`` for all samples."""
    if value is None:
        return None
    count = int(value)
    if count < 0:
        raise ValueError("samples_per_scenario must be non-negative")
    return None if count == 0 else count


def build_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    scenario: str = "default",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
) -> Path:
    """Build a simple manifest from ``*.npy`` landmarks and matching images."""
    dataset_name = dataset.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset '{dataset}'")
    src = Path(source_dir)
    scenario_groups = _explicit_scenario_groups(scenarios)
    samples: list[dict[str, T.Any]] = []
    for landmarks in sorted(src.glob("*.npy")):
        image = _matching_image(landmarks)
        if image is None:
            continue
        samples.append(
            {
                "sample_id": landmarks.stem,
                "dataset": dataset_name,
                "condition": scenario,
                "image": str(image.resolve()),
                "landmarks": str(landmarks.resolve()),
                "source": {"dataset": dataset_name, "source_id": landmarks.stem},
            }
        )
    return _write_manifest_and_audit(
        _filter_samples(samples, scenario_groups, samples_per_scenario),
        Path(output_dir),
        dataset_name,
        scenario,
        manifest_mode=manifest_mode,
        allow_overlap=allow_overlap,
        write_overlays=write_overlays,
        scenario_groups=scenario_groups,
    )


def build_directory_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str = "directory",
    scenario: str = "default",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
) -> Path:
    """Build a manifest from a directory tree of ``*.npy`` landmarks."""
    dataset_name = dataset.lower()
    src = Path(source_dir)
    scenario_groups = _explicit_scenario_groups(scenarios)
    samples = []
    for landmarks in sorted(src.rglob("*.npy")):
        image = _matching_image(landmarks)
        if image is None:
            continue
        sample_id = landmarks.relative_to(src).with_suffix("").as_posix()
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "condition": scenario,
                "image": str(image.resolve()),
                "landmarks": str(landmarks.resolve()),
                "source": {"dataset": dataset_name, "source_id": sample_id},
            }
        )
    return _write_manifest_and_audit(
        _filter_samples(samples, scenario_groups, samples_per_scenario),
        Path(output_dir),
        dataset_name,
        scenario,
        manifest_mode=manifest_mode,
        allow_overlap=allow_overlap,
        write_overlays=write_overlays,
        scenario_groups=scenario_groups,
    )


def build_wflw_manifest(
    annotation_file: str | Path | None,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "default",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
) -> Path:
    """Build a WFLW manifest from 98-point annotations."""
    cleanup: contextlib.AbstractContextManager[Path] | None = None
    if annotation_file is None:
        resolved = resolve_dataset_source(
            WFLW_SOURCE,
            cache_dir=cache_dir,
            source_dir=source_dir,
            source_zip=source_zip,
            download_url=download_url,
            force_download=force_download,
            no_download=no_download,
        )
        cleanup = _source_root(resolved)
        root = cleanup.__enter__()
        annotations = _find_wflw_annotation(root)
        inferred_image_root = _find_wflw_image_root(root)
    else:
        annotations = Path(annotation_file)
        inferred_image_root = annotations.parent
    try:
        if not annotations.is_file():
            raise FileNotFoundError(f"WFLW annotation file not found: {annotations}")
        root = inferred_image_root if image_root is None else Path(image_root)
        scenario_groups = _explicit_scenario_groups(scenarios)
        samples = []
        for index, line in enumerate(annotations.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 197:
                raise ValueError(f"WFLW line {index + 1} has too few fields")
            points = np.asarray([float(value) for value in parts[:196]], dtype="float32")
            image_rel = parts[-1]
            sample_id = Path(image_rel).with_suffix("").as_posix()
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset": "wflw",
                    "condition": scenario,
                    "image": str((root / image_rel).resolve()),
                    "source_schema": "2d_98",
                    "source": {"dataset": "wflw", "source_id": image_rel},
                    "points": normalize_landmarks(points.reshape(98, 2), source_schema="2d_98"),
                }
            )
        return _write_manifest_and_audit(
            _filter_samples(samples, scenario_groups, samples_per_scenario),
            Path(output_dir),
            "wflw",
            scenario,
            manifest_mode=manifest_mode,
            allow_overlap=allow_overlap,
            write_overlays=write_overlays,
            scenario_groups=scenario_groups,
        )
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)


def build_cofw_manifest(
    source_json: str | Path | None,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "default",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
) -> Path:
    """Build a COFW manifest from a simple JSON export.

    Expected input shape is ``{"samples": [{"sample_id", "landmarks", "image",
    "conditions"}]}`` or a bare list with the same item shape.
    """
    cleanup: contextlib.AbstractContextManager[Path] | None = None
    if source_json is None:
        resolved = resolve_dataset_source(
            COFW_SOURCE,
            cache_dir=cache_dir,
            source_dir=source_dir,
            source_zip=source_zip,
            download_url=download_url,
            force_download=force_download,
            no_download=no_download,
        )
        if resolved.is_file() and not is_archive(resolved):
            source = resolved
            root = source.parent
        else:
            cleanup = _source_root(resolved)
            root = cleanup.__enter__()
            source = _find_cofw_json(root)
    else:
        source = Path(source_json)
        root = source.parent
    try:
        if not source.is_file():
            raise FileNotFoundError(f"COFW JSON not found: {source}")
        image_base = root if image_root is None else Path(image_root)
        payload = json.loads(source.read_text(encoding="utf-8"))
        entries = payload.get("samples", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError("COFW JSON must contain a list or a 'samples' list")
        scenario_groups = _explicit_scenario_groups(scenarios)
        samples = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"COFW entry {index + 1} must be an object")
            raw_points = entry.get("ground_truth", entry.get("landmarks"))
            if raw_points is None:
                raise ValueError(f"COFW entry {index + 1} missing landmarks")
            points = np.asarray(raw_points, dtype="float32")
            conditions = dict(entry.get("conditions", {}))
            condition = str(conditions.get("scenario", scenario))
            image_value = str(entry.get("image", ""))
            image_path = Path(image_value)
            if image_value and not image_path.is_absolute():
                image_value = str((image_base / image_path).resolve())
            sample_id = str(entry.get("sample_id") or entry.get("id") or index)
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset": "cofw",
                    "condition": condition,
                    "image": image_value,
                    "source": {"dataset": "cofw", "source_id": image_value or sample_id},
                    "points": normalize_landmarks(points),
                }
            )
        return _write_manifest_and_audit(
            _filter_samples(samples, scenario_groups, samples_per_scenario),
            Path(output_dir),
            "cofw",
            scenario,
            manifest_mode=manifest_mode,
            allow_overlap=allow_overlap,
            write_overlays=write_overlays,
            scenario_groups=scenario_groups,
        )
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)


def audit_existing_manifest(
    output_dir: str | Path,
    *,
    allow_overlap: bool = False,
) -> Path:
    """Audit an existing manifest without regenerating fixtures."""
    out = Path(output_dir)
    manifest = out / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.json not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = list(payload.get("samples", []))
    dataset = str(payload.get("dataset", "mixed"))
    duplicate_sources = validate_no_cross_group_source_overlap(
        entries, allow_overlap=allow_overlap
    )
    audit = out / "dataset_audit.json"
    audit.write_text(
        json.dumps(
            _build_audit(
                entries,
                out,
                dataset,
                "default",
                duplicate_sources,
                allow_overlap=allow_overlap,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_source_notes(out)
    logger.info("Wrote landmark dataset audit: %s", audit)
    return audit


@contextlib.contextmanager
def _source_root(source: Path) -> T.Iterator[Path]:
    """Yield an extracted root directory for a source archive or directory."""
    if source.is_dir():
        yield source
    else:
        with extract_archive_to_temp(source) as root:
            yield root


def _find_wflw_annotation(root: Path) -> Path:
    """Find the best WFLW 98-point annotation file inside ``root``."""
    candidates = [path for path in root.rglob("*.txt") if "98pt" in path.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            f"No WFLW 98-point annotation file found under {root}. "
            "Pass --wflw-annotations to point at it explicitly."
        )
    return sorted(candidates, key=lambda p: ("test" not in p.name.lower(), len(p.parts), p.name))[
        0
    ]


def _find_wflw_image_root(root: Path) -> Path:
    """Return likely WFLW image root for relative annotation image paths."""
    for name in ("WFLW_images", "images", "Images"):
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return sorted(matches, key=lambda p: len(p.parts))[0]
    return root


def _find_cofw_json(root: Path) -> Path:
    """Find a COFW JSON export inside ``root``."""
    candidates = sorted(
        root.rglob("*.json"), key=lambda p: ("cofw" not in p.name.lower(), len(p.parts), p.name)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No COFW JSON export found under {root}. Pass --cofw-json to point at it explicitly."
        )
    return candidates[0]


def _explicit_scenario_groups(scenarios: T.Sequence[str] | None = None) -> tuple[str, ...]:
    """Return explicitly requested scenario groups.

    ``--scenario`` supplies the default condition for unlabelled datasets and
    should not filter dataset-provided labels. Only ``--scenarios`` filters.
    """
    if scenarios is None:
        return ()
    return tuple(dict.fromkeys(str(value).strip() for value in scenarios if str(value).strip()))


def _filter_samples(
    samples: T.Sequence[dict[str, T.Any]],
    scenario_groups: T.Sequence[str],
    samples_per_scenario: int | None = None,
) -> list[dict[str, T.Any]]:
    """Filter by explicit scenario groups and optionally cap samples per group."""
    limit = validate_samples_per_scenario(samples_per_scenario)
    allowed = set(scenario_groups)
    counts: dict[str, int] = {}
    filtered: list[dict[str, T.Any]] = []
    for sample in samples:
        condition = str(sample.get("condition", "default"))
        if allowed and condition not in allowed:
            continue
        if limit is not None and counts.get(condition, 0) >= limit:
            continue
        filtered.append(dict(sample))
        counts[condition] = counts.get(condition, 0) + 1
    return filtered


def _replacement_groups(
    entries: T.Sequence[dict[str, T.Any]],
    scenario: str,
    scenario_groups: T.Sequence[str],
) -> tuple[str, ...]:
    """Return manifest groups that should be replaced during merge."""
    if scenario_groups:
        return tuple(scenario_groups)
    generated = tuple(dict.fromkeys(str(entry.get("condition", scenario)) for entry in entries))
    return generated or (scenario,)


def _matching_image(path: Path) -> Path | None:
    """Return the matching image for a landmark path, if present."""
    for ext in IMAGE_EXTS:
        candidate = path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _safe_filename(value: str) -> str:
    """Return a readable filename-safe sample identifier."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "sample"


def _entry_path(value: str, output_dir: Path) -> Path:
    """Return a manifest path as absolute or relative to ``output_dir``."""
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


def _source_key_for_entry(entry: dict[str, T.Any]) -> tuple[str, str]:
    """Return a stable source key for overlap detection."""
    source = entry.get("source", {})
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


def duplicate_source_audit(entries: T.Sequence[dict[str, T.Any]]) -> list[dict[str, T.Any]]:
    """Return source keys used by more than one condition group."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for entry in entries:
        key = _source_key_for_entry(entry)
        grouped.setdefault(key, []).append(
            {
                "sample_id": str(entry.get("sample_id") or entry.get("name") or ""),
                "condition": str(entry.get("condition") or entry.get("scenario") or ""),
            }
        )
    duplicates: list[dict[str, T.Any]] = []
    for key, refs in grouped.items():
        conditions = sorted({ref["condition"] for ref in refs})
        if len(conditions) > 1:
            duplicates.append(
                {"source_key": list(key), "condition_groups": conditions, "entries": refs}
            )
    return sorted(duplicates, key=lambda item: tuple(item["source_key"]))


def validate_no_cross_group_source_overlap(
    entries: T.Sequence[dict[str, T.Any]],
    *,
    allow_overlap: bool,
) -> list[dict[str, T.Any]]:
    """Validate source-key uniqueness across condition groups."""
    duplicates = duplicate_source_audit(entries)
    if duplicates and not allow_overlap:
        formatted = "; ".join(
            f"{tuple(item['source_key'])}: {', '.join(item['condition_groups'])}"
            for item in duplicates
        )
        raise ValueError("Manifest contains cross-group duplicate sources: " + formatted)
    return duplicates


def _load_existing_manifest_entries(manifest_path: Path) -> list[dict[str, T.Any]]:
    """Load existing manifest samples if present."""
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("samples", []))


def _merge_manifest_entries(
    manifest_path: Path,
    new_entries: T.Sequence[dict[str, T.Any]],
    *,
    dataset: str,
    scenario_groups: T.Sequence[str],
) -> list[dict[str, T.Any]]:
    """Merge entries by replacing the generated dataset/scenario slice."""
    replaced = set(scenario_groups)
    kept = [
        entry
        for entry in _load_existing_manifest_entries(manifest_path)
        if not (
            str(entry.get("dataset")) == dataset
            and str(entry.get("condition", "default")) in replaced
        )
    ]
    return sorted(
        kept + list(new_entries),
        key=lambda entry: (
            str(entry.get("dataset", "")),
            str(entry.get("condition", "")),
            str(entry.get("sample_id", "")),
        ),
    )


def _copy_image_if_possible(image_value: str, scenario_dir: Path) -> str | None:
    """Copy an existing source image into the generated fixture directory."""
    if not image_value:
        return None
    image_path = Path(image_value)
    if not image_path.is_file():
        return None
    suffix = image_path.suffix if image_path.suffix.lower() in IMAGE_EXTS else ".png"
    target = scenario_dir / f"frame{suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target)
    return target.name


def _write_landmark_overlay(image_path: Path, points: np.ndarray, output_path: Path) -> None:
    """Write a simple landmark visual audit overlay."""
    try:
        import cv2  # pylint:disable=import-outside-toplevel
    except ImportError:
        logger.warning("OpenCV unavailable; skipping landmark overlay for %s", image_path)
        return
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("Unable to read image for landmark overlay: %s", image_path)
        return
    for index, pt in enumerate(points):
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        if index % 5 == 0:
            cv2.putText(
                image,
                str(index),
                (x + 2, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Unable to write landmark overlay: {output_path}")


def _copy_landmarks_if_possible(landmarks_value: str, scenario_dir: Path) -> str | None:
    """Copy an existing landmark file into the generated fixture directory."""
    if not landmarks_value:
        return None
    landmarks_path = Path(landmarks_value)
    if not landmarks_path.is_file():
        return None
    target = scenario_dir / "landmarks.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(landmarks_path, target)
    return target.name


def _prepare_manifest_entry(
    sample: dict[str, T.Any],
    output_dir: Path,
    *,
    write_overlays: bool,
) -> dict[str, T.Any]:
    """Prepare one manifest entry and generated fixture files."""
    entry = dict(sample)
    sample_id = str(entry["sample_id"])
    scenario_dir = output_dir / "generated" / _safe_filename(sample_id)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    image_name = _copy_image_if_possible(str(entry.get("image", "")), scenario_dir)
    if image_name is not None:
        entry["image"] = str((scenario_dir / image_name).relative_to(output_dir))

    points = entry.pop("points", None)
    if points is None:
        landmark_name = _copy_landmarks_if_possible(str(entry.get("landmarks", "")), scenario_dir)
        if landmark_name is not None:
            entry["landmarks"] = str((scenario_dir / landmark_name).relative_to(output_dir))
            if write_overlays and image_name is not None:
                landmarks = np.load(str(scenario_dir / landmark_name)).astype("float32")
                overlay_path = scenario_dir / "overlays" / "landmarks_gt.png"
                _write_landmark_overlay(scenario_dir / image_name, landmarks, overlay_path)
                entry.setdefault("metadata", {})["overlay"] = str(overlay_path.relative_to(output_dir))
        return entry

    landmarks = np.asarray(points, dtype="float32")
    landmarks_path = scenario_dir / "landmarks.npy"
    np.save(str(landmarks_path), landmarks)
    entry["landmarks"] = str(landmarks_path.relative_to(output_dir))

    if write_overlays and image_name is not None:
        overlay_path = scenario_dir / "overlays" / "landmarks_gt.png"
        _write_landmark_overlay(scenario_dir / image_name, landmarks, overlay_path)
        entry.setdefault("metadata", {})["overlay"] = str(overlay_path.relative_to(output_dir))
    return entry


def _build_audit(
    manifest_samples: T.Sequence[dict[str, T.Any]],
    output_dir: Path,
    dataset: str,
    scenario: str,
    duplicate_sources: list[dict[str, T.Any]] | None = None,
    *,
    allow_overlap: bool = False,
) -> dict[str, T.Any]:
    """Build a dataset audit payload for landmark manifests."""
    condition_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    source_schema_counts: dict[str, int] = {}
    selected_by_group: dict[str, list[list[str]]] = {}
    missing_images: list[str] = []
    missing_landmarks: list[str] = []
    invalid_landmarks: list[dict[str, T.Any]] = []
    sample_ids = [str(sample.get("sample_id", "")) for sample in manifest_samples]
    duplicate_ids = sorted(
        {sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1}
    )

    for sample in manifest_samples:
        condition = str(sample.get("condition", scenario))
        sample_dataset = str(sample.get("dataset", dataset))
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        dataset_counts[sample_dataset] = dataset_counts.get(sample_dataset, 0) + 1
        source_schema = str(sample.get("source_schema", "2d_68"))
        source_schema_counts[source_schema] = source_schema_counts.get(source_schema, 0) + 1
        selected_by_group.setdefault(condition, []).append(list(_source_key_for_entry(sample)))
        image = str(sample.get("image", ""))
        if image and not _entry_path(image, output_dir).is_file():
            missing_images.append(image)
        landmarks = str(sample.get("landmarks", ""))
        landmark_path = _entry_path(landmarks, output_dir)
        if not landmark_path.is_file():
            missing_landmarks.append(landmarks)
            continue
        try:
            shape = tuple(np.load(str(landmark_path)).shape)
        except (OSError, ValueError) as err:
            invalid_landmarks.append({"sample_id": sample.get("sample_id", ""), "error": str(err)})
            continue
        shape_key = "x".join(str(part) for part in shape)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        if shape != (68, 2):
            invalid_landmarks.append(
                {"sample_id": sample.get("sample_id", ""), "shape": shape_key}
            )

    if duplicate_sources is None:
        duplicate_sources = duplicate_source_audit(manifest_samples)
    return {
        "schema_version": 1,
        "dataset": dataset,
        "total_entries": len(manifest_samples),
        "condition_counts": dict(sorted(condition_counts.items())),
        "count_per_dataset": dict(sorted(dataset_counts.items())),
        "count_per_source_schema": dict(sorted(source_schema_counts.items())),
        "landmark_shape_counts": dict(sorted(shape_counts.items())),
        "missing_images": missing_images,
        "missing_landmarks": missing_landmarks,
        "invalid_landmarks": invalid_landmarks,
        "duplicate_sample_ids": duplicate_ids,
        "duplicate_source_ids": duplicate_sources,
        "overlap": {
            "allow_overlap": allow_overlap,
            "has_overlap": bool(duplicate_sources),
            "duplicate_count": len(duplicate_sources),
        },
        "selected_source_ids_per_group": {
            group: values for group, values in sorted(selected_by_group.items())
        },
        "supported_datasets": SUPPORTED_DATASETS,
    }


def _write_source_notes(output_dir: Path) -> None:
    """Write generated source/licensing notes next to generated manifests."""
    notes = output_dir / "SOURCE_NOTES.md"
    if notes.is_file():
        return
    notes.write_text(
        "# Landmark quality dataset source notes\n\n"
        "This directory was populated by `tools/landmarks/build_quality_dataset.py`.\n\n"
        "The builder resolves sources from explicit CLI paths, `.fs_cache/landmark_quality`, "
        "or configured download URLs. Review upstream dataset terms before use or redistribution."
        "\n\nDo not commit generated images, annotations, or manifests unless licensing has been "
        "reviewed.\n",
        encoding="utf-8",
    )


def _write_manifest_and_audit(
    samples: T.Sequence[dict[str, T.Any]],
    output_dir: Path,
    dataset: str,
    scenario: str,
    *,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
    scenario_groups: T.Sequence[str] = (),
) -> Path:
    """Write manifest, audit, source notes, and generated fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    mode = validate_manifest_mode(manifest_mode)
    prepared = [
        _prepare_manifest_entry(sample, output_dir, write_overlays=write_overlays)
        for sample in samples
    ]
    groups = _replacement_groups(prepared, scenario, scenario_groups)
    final_entries = (
        _merge_manifest_entries(manifest, prepared, dataset=dataset, scenario_groups=groups)
        if mode == "merge"
        else sorted(
            prepared,
            key=lambda entry: (
                str(entry.get("dataset", "")),
                str(entry.get("condition", "")),
                str(entry.get("sample_id", "")),
            ),
        )
    )
    duplicates = validate_no_cross_group_source_overlap(
        final_entries, allow_overlap=allow_overlap
    )
    manifest.write_text(
        json.dumps(
            {"dataset": dataset, "samples": final_entries},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(
            _build_audit(
                final_entries,
                output_dir,
                dataset,
                scenario,
                duplicates,
                allow_overlap=allow_overlap,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_source_notes(output_dir)
    logger.info(
        "Wrote landmark manifest: %s entries=%d mode=%s", manifest, len(final_entries), mode
    )
    return manifest
