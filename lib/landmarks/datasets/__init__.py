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

from lib.landmarks.core.schema import normalize_landmarks
from lib.landmarks.datasets.sources import (
    DEFAULT_CACHE_DIR,
    EXTRACTED_DIR_NAME,
    WFLW_OFFICIAL_SOURCE,
    DatasetSourceSpec,
    extract_archive_to_temp,
    is_archive,
    resolve_dataset_source,
    resolve_wflw_official_source,
)

logger = logging.getLogger(__name__)
SUPPORTED_DATASETS = ("wflw", "cofw", "merl-rav", "aflw2000-3d", "directory")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
MANIFEST_MODES = ("replace", "merge")
WFLW_ATTRIBUTE_NAMES = ("pose", "expression", "illumination", "makeup", "occlusion", "blur")
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


def _wflw_official_cache_present(cache_dir: str | Path) -> bool:
    """Return whether the cache contains official WFLW multipart artifacts."""
    cache_root = Path(cache_dir) / WFLW_OFFICIAL_SOURCE.cache_root_name
    if any((cache_root / part.archive_name).exists() for part in WFLW_OFFICIAL_SOURCE.parts):
        return True
    extracted = cache_root / EXTRACTED_DIR_NAME
    return any((extracted / name).exists() for name in ("WFLW_annotations", "WFLW_images"))


def _resolve_wflw_source(
    *,
    source_dir: str | Path | None,
    source_zip: str | Path | None,
    cache_dir: str | Path,
    download_url: str | None,
    force_download: bool,
    no_download: bool,
) -> Path:
    """Resolve WFLW sources, preferring complete official multipart cache parts."""
    if source_dir is None and source_zip is None and download_url is None:
        try:
            return resolve_wflw_official_source(
                cache_dir=cache_dir,
                force_download=False,
                no_download=True,
            )
        except FileNotFoundError as err:
            if _wflw_official_cache_present(cache_dir):
                raise FileNotFoundError(
                    "Incomplete WFLW official multipart cache. Expected both "
                    "WFLW_annotations.tar.gz and WFLW_images.zip, or a complete "
                    "extracted cache with WFLW_annotations and WFLW_images. Remove "
                    "the partial cache or provide --source-dir/--source-zip."
                ) from err
    return resolve_dataset_source(
        WFLW_SOURCE,
        cache_dir=cache_dir,
        source_dir=source_dir,
        source_zip=source_zip,
        download_url=download_url,
        force_download=force_download,
        no_download=no_download,
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
MERL_RAV_SOURCE = DatasetSourceSpec(
    dataset="MERL-RAV",
    cache_subdir="merl-rav",
    canonical_archive="merl-rav.zip",
    cache_aliases=("MERL-RAV.zip", "MERL_RAV.zip", "merl_rav.json"),
    extracted_aliases=("MERL-RAV", "MERL_RAV", "merl-rav", "merl_rav"),
    manual_hint=(
        "Provide --source-dir/--source-zip containing a JSON-style manifest or "
        "a directory tree of image/npy landmark pairs."
    ),
)
AFLW2000_3D_SOURCE = DatasetSourceSpec(
    dataset="AFLW2000-3D",
    cache_subdir="aflw2000-3d",
    canonical_archive="aflw2000-3d.zip",
    cache_aliases=("AFLW2000-3D.zip", "AFLW2000_3D.zip", "aflw2000_3d.json"),
    extracted_aliases=("AFLW2000-3D", "AFLW2000_3D", "aflw2000-3d", "aflw2000_3d"),
    manual_hint=(
        "Provide --source-dir/--source-zip containing a JSON-style manifest or "
        "a directory tree of image/npy landmark pairs."
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
    if not src.is_dir():
        raise FileNotFoundError(f"{dataset_name} source directory not found: {src}")
    scenario_groups = _explicit_scenario_groups(scenarios)
    condition = _fallback_condition_label(scenario)
    samples: list[dict[str, T.Any]] = []
    for landmarks in sorted(src.glob("*.npy")):
        image = _matching_image(landmarks)
        if image is None:
            continue
        samples.append(
            {
                "sample_id": landmarks.stem,
                "dataset": dataset_name,
                "condition": condition,
                "conditions": (condition,),
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
    if not src.is_dir():
        raise FileNotFoundError(f"{dataset_name} source directory not found: {src}")
    scenario_groups = _explicit_scenario_groups(scenarios)
    condition = _fallback_condition_label(scenario)
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
                "condition": condition,
                "conditions": (condition,),
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


def build_merl_rav_manifest(
    output_dir: str | Path,
    *,
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
    """Build a MERL-RAV manifest from JSON-style manifests or image/npy directories."""
    return _build_json_or_directory_dataset_manifest(
        "merl-rav",
        MERL_RAV_SOURCE,
        output_dir,
        source_dir=source_dir,
        source_zip=source_zip,
        cache_dir=cache_dir,
        download_url=download_url,
        force_download=force_download,
        no_download=no_download,
        scenario=scenario,
        scenarios=scenarios,
        samples_per_scenario=samples_per_scenario,
        manifest_mode=manifest_mode,
        allow_overlap=allow_overlap,
        write_overlays=write_overlays,
    )


def build_aflw2000_3d_manifest(
    output_dir: str | Path,
    *,
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
    """Build an AFLW2000-3D manifest from JSON-style manifests or image/npy directories."""
    return _build_json_or_directory_dataset_manifest(
        "aflw2000-3d",
        AFLW2000_3D_SOURCE,
        output_dir,
        source_dir=source_dir,
        source_zip=source_zip,
        cache_dir=cache_dir,
        download_url=download_url,
        force_download=force_download,
        no_download=no_download,
        scenario=scenario,
        scenarios=scenarios,
        samples_per_scenario=samples_per_scenario,
        manifest_mode=manifest_mode,
        allow_overlap=allow_overlap,
        write_overlays=write_overlays,
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
        resolved = _resolve_wflw_source(
            source_dir=source_dir,
            source_zip=source_zip,
            cache_dir=cache_dir,
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
        parsed_rows = []
        image_counts: dict[str, int] = {}
        for index, line in enumerate(annotations.read_text(encoding="utf-8").splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            points, bbox, attributes, image_rel = _parse_wflw_line(stripped, index + 1)
            parsed_rows.append((points, bbox, attributes, image_rel))
            image_counts[image_rel] = image_counts.get(image_rel, 0) + 1

        samples = []
        image_occurrences: dict[str, int] = {}
        for points, bbox, attributes, image_rel in parsed_rows:
            image_occurrences[image_rel] = image_occurrences.get(image_rel, 0) + 1
            sample_id = _wflw_sample_id(
                image_rel,
                image_occurrences[image_rel],
                image_counts[image_rel],
            )
            condition_labels = _labels_from_wflw_attributes(attributes)
            condition = (
                condition_labels[0] if condition_labels else _fallback_condition_label(scenario)
            )
            canonical_68 = normalize_landmarks(points.reshape(98, 2), source_schema="2d_98")
            metadata: dict[str, T.Any] = {
                "bbox": bbox,
                "attributes": attributes,
                "image_id": image_rel,
            }
            visibility = _visibility_from_jaw_asymmetry(canonical_68)
            if visibility is not None:
                metadata["visibility"] = visibility
                metadata["visibility_source"] = "wflw_jaw_asymmetry"
            entry: dict[str, T.Any] = {
                "sample_id": sample_id,
                "dataset": "wflw",
                "condition": condition,
                "conditions": condition_labels or (condition,),
                "image": str((root / image_rel).resolve()),
                "source_schema": "2d_98",
                "source": {"dataset": "wflw", "source_id": sample_id},
                "metadata": metadata,
                "points": canonical_68,
            }
            if visibility is not None:
                entry["visibility"] = visibility
            samples.append(entry)
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


def _wflw_sample_id(image_rel: str, occurrence: int, total_occurrences: int) -> str:
    """Return a stable WFLW face sample id for an annotation row."""
    base = Path(image_rel).with_suffix("").as_posix()
    if total_occurrences <= 1:
        return base
    return f"{base}#face-{occurrence:02d}"


# Asymmetry threshold above which one side of the jawline is considered
# self-occluded. Calibrated against the WFLW test split: median asymmetry
# is ~0.25 for non-pose conditions (faces are rarely perfectly centered)
# and ~0.90 for the ``pose`` condition. 0.50 cleanly isolates the right
# tail of pose-flagged samples without firing on slightly-tilted faces
# elsewhere in the dataset.
DEFAULT_WFLW_ASYMMETRY_THRESHOLD: float = 0.50


def _visibility_from_jaw_asymmetry(
    canonical_68: np.ndarray,
    *,
    asymmetry_threshold: float = DEFAULT_WFLW_ASYMMETRY_THRESHOLD,
) -> list[bool] | None:
    """Derive per-landmark visibility from 68-point jawline asymmetry.

    Compares the horizontal distance from the nose tip (index 30) to each
    ear-side jawline endpoint (indices 0 and 16). On a frontal face these
    distances are roughly equal; under heavy yaw the far-side compresses
    toward the nose, producing a strong asymmetry. When the asymmetry
    exceeds ``asymmetry_threshold``, the far half of the jawline is
    flagged as self-occluded.

    Returns ``None`` for frontal-ish faces so the manifest stays compact
    and the harness's existing ``visibility=None`` path applies to those
    samples unchanged.
    """
    if canonical_68.shape != (68, 2):
        return None
    nose_x = float(canonical_68[30, 0])
    right_jaw_x = float(canonical_68[0, 0])  # subject's right ear (viewer's left)
    left_jaw_x = float(canonical_68[16, 0])  # subject's left ear (viewer's right)
    right_dist = abs(nose_x - right_jaw_x)
    left_dist = abs(left_jaw_x - nose_x)
    total = right_dist + left_dist
    if total <= 0:
        return None
    asymmetry = (right_dist - left_dist) / total
    if abs(asymmetry) < asymmetry_threshold:
        return None
    visibility = [True] * 68
    if asymmetry < 0:
        # Subject's right side compressed → head turned toward subject's
        # right → right-side jawline (indices 0..7) is self-occluded.
        for idx in range(0, 8):
            visibility[idx] = False
    else:
        # Subject's left side compressed → head turned toward subject's
        # left → left-side jawline (indices 9..16) is self-occluded.
        for idx in range(9, 17):
            visibility[idx] = False
    return visibility


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
            raw_points = _entry_landmark_values(entry)
            if raw_points is None:
                raise ValueError(f"COFW entry {index + 1} missing landmarks")
            points, source_schema = _normalize_points_for_manifest(raw_points)
            metadata = _entry_metadata(entry)
            conditions = _entry_conditions(entry)
            condition_labels = _condition_labels_from_metadata(
                conditions,
                metadata,
                default=scenario,
            )
            condition = condition_labels[0]
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
                    "conditions": condition_labels,
                    "image": image_value,
                    "source_schema": source_schema,
                    "source": {"dataset": "cofw", "source_id": image_value or sample_id},
                    "metadata": metadata,
                    "points": points,
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


def _build_json_or_directory_dataset_manifest(
    dataset_name: str,
    source_spec: DatasetSourceSpec,
    output_dir: str | Path,
    *,
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
    """Build a manifest from a JSON-style manifest, falling back to directory pairs."""
    resolved = resolve_dataset_source(
        source_spec,
        cache_dir=cache_dir,
        source_dir=source_dir,
        source_zip=source_zip,
        download_url=download_url,
        force_download=force_download,
        no_download=no_download,
    )
    cleanup: contextlib.AbstractContextManager[Path] | None = None
    try:
        if resolved.is_file() and not is_archive(resolved):
            source = resolved
            root = source.parent
        else:
            cleanup = _source_root(resolved)
            root = cleanup.__enter__()
            source = _find_dataset_json(root, dataset_name)
        if source is None:
            return build_directory_manifest(
                root,
                output_dir,
                dataset=dataset_name,
                scenario=scenario,
                scenarios=scenarios,
                samples_per_scenario=samples_per_scenario,
                manifest_mode=manifest_mode,
                allow_overlap=allow_overlap,
                write_overlays=write_overlays,
            )
        image_base = root
        scenario_groups = _explicit_scenario_groups(scenarios)
        samples: list[dict[str, T.Any]] = []
        for index, entry in enumerate(_parse_json_entries(source, dataset_name)):
            raw_points = _entry_landmark_values(entry)
            if raw_points is None:
                raise ValueError(f"{dataset_name} entry {index + 1} missing landmarks")
            points, source_schema = _normalize_points_for_manifest(raw_points)
            metadata = _entry_metadata(entry)
            condition_labels = _condition_labels_from_metadata(
                _entry_conditions(entry),
                metadata,
                default=scenario,
            )
            image_value = str(entry.get("image", entry.get("image_path", "")))
            image_path = Path(image_value)
            if image_value and not image_path.is_absolute():
                image_value = str((image_base / image_path).resolve())
            sample_id = str(
                entry.get("sample_id")
                or entry.get("id")
                or Path(image_value).with_suffix("").name
                or index
            )
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "condition": condition_labels[0],
                    "conditions": condition_labels,
                    "image": image_value,
                    "source_schema": source_schema,
                    "source": {"dataset": dataset_name, "source_id": image_value or sample_id},
                    "metadata": metadata,
                    "points": points,
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
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)


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


def _find_dataset_json(root: Path, dataset: str) -> Path | None:
    """Find a JSON-style manifest for ``dataset`` under ``root``."""
    dataset_tokens = {dataset.lower(), dataset.lower().replace("-", "_")}
    candidates = [
        path
        for path in root.rglob("*.json")
        if path.name != "dataset_audit.json" and path.name != "metrics.json"
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda p: (
            p.name != "manifest.json",
            not any(token in p.stem.lower() for token in dataset_tokens),
            len(p.parts),
            p.name,
        ),
    )[0]


def _parse_json_entries(source: Path, dataset: str) -> list[dict[str, T.Any]]:
    """Return sample entries from a JSON-style dataset manifest."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    entries = (
        payload.get("samples", payload.get("entries", payload))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(entries, list):
        raise ValueError(f"{dataset} JSON must contain a list or a 'samples' list")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{dataset} JSON entries must be objects")
    return list(entries)


def _entry_landmark_values(entry: dict[str, T.Any]) -> T.Any:
    """Return landmark data from common JSON manifest keys."""
    for key in ("ground_truth", "landmarks", "points", "pts"):
        if key in entry:
            return entry[key]
    return None


def _normalize_points_for_manifest(raw_points: T.Any) -> tuple[np.ndarray, str]:
    """Normalize 2D/3D landmark arrays into canonical 68x2 manifest points."""
    array = np.asarray(raw_points, dtype="float32")
    if array.ndim == 1:
        if array.size % 3 == 0 and array.size // 3 == 68:
            array = array.reshape((68, 3))
        elif array.size % 2 == 0:
            array = array.reshape((-1, 2))
    if array.ndim != 2:
        raise ValueError(f"landmarks must be a 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("landmarks contain NaN or infinite values")
    if array.shape == (68, 3):
        return np.ascontiguousarray(array[:, :2], dtype="float32"), "3d_68"
    if array.shape[1] == 3:
        array = array[:, :2]
    if array.shape == (98, 2):
        return normalize_landmarks(array, source_schema="2d_98"), "2d_98"
    if array.shape == (68, 2):
        return normalize_landmarks(array, source_schema="2d_68"), "2d_68"
    return normalize_landmarks(array), f"2d_{array.shape[0]}"


def _entry_metadata(entry: dict[str, T.Any]) -> dict[str, T.Any]:
    """Copy non-structural JSON metadata for manifest preservation."""
    metadata = dict(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {}
    for key in (
        "occlusion",
        "occlusions",
        "occluded",
        "visibility",
        "visible",
        "attributes",
        "pose",
        "expression",
        "illumination",
        "makeup",
        "blur",
    ):
        if key in entry:
            metadata[key] = entry[key]
    return metadata


def _entry_conditions(entry: dict[str, T.Any]) -> dict[str, T.Any]:
    """Return condition metadata from a JSON entry."""
    raw = entry.get("conditions", {})
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        return {"labels": list(raw)}
    if raw:
        return {"scenario": raw}
    return {}


def _truthy_condition(value: T.Any) -> bool:
    """Return whether a metadata value marks a condition as present."""
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no", "clean"}
    if isinstance(value, (list, tuple, set)):
        return any(_truthy_condition(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).size and np.any(value))
    return bool(value)


def _normalize_condition_label(value: T.Any) -> str:
    """Normalize user/dataset condition labels while preserving their meaning."""
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or "default"


def _fallback_condition_label(value: T.Any) -> str:
    """Return the condition label for unlabelled samples."""
    return _normalize_condition_label(value)


def _condition_labels_from_metadata(
    conditions: dict[str, T.Any],
    metadata: dict[str, T.Any],
    *,
    default: str,
) -> tuple[str, ...]:
    """Build normalized condition labels from explicit and occlusion metadata."""
    labels: list[str] = []
    raw_labels = conditions.get("labels")
    if isinstance(raw_labels, (list, tuple, set)):
        labels.extend(_normalize_condition_label(label) for label in raw_labels)
    elif raw_labels:
        labels.append(_normalize_condition_label(raw_labels))
    scenario = conditions.get("scenario") or conditions.get("condition")
    if scenario:
        labels.append(_normalize_condition_label(scenario))
    for key, value in {**conditions, **metadata}.items():
        normalized_key = _normalize_condition_label(key)
        if normalized_key in {"scenario", "condition", "labels"}:
            continue
        if normalized_key in {"occlusion", "occlusions", "occluded", "visibility", "visible"}:
            if normalized_key in {"visibility", "visible"}:
                visible_values = np.asarray(value).ravel()
                if _truthy_condition(value) and not all(bool(item) for item in visible_values):
                    labels.append("occlusion")
            elif _truthy_condition(value):
                labels.append("occlusion")
        elif normalized_key in WFLW_ATTRIBUTE_NAMES and _truthy_condition(value):
            labels.append(normalized_key)
    labels = [
        _normalize_condition_label(label) for label in labels if _normalize_condition_label(label)
    ]
    if not labels:
        labels = [_fallback_condition_label(default)]
    return tuple(dict.fromkeys(labels))


def _parse_wflw_line(
    line: str,
    line_number: int,
) -> tuple[np.ndarray, list[float], dict[str, int], str]:
    """Parse one WFLW annotation row, including bbox and six official attributes."""
    parts = line.split()
    if len(parts) < 197:
        raise ValueError(f"WFLW line {line_number} has too few fields")
    try:
        points = np.asarray([float(value) for value in parts[:196]], dtype="float32")
    except ValueError as err:
        raise ValueError(f"WFLW line {line_number} has invalid landmark coordinates") from err
    bbox: list[float] = []
    if len(parts) >= 201:
        try:
            bbox = [float(value) for value in parts[196:200]]
        except ValueError as err:
            raise ValueError(f"WFLW line {line_number} has invalid bbox values") from err
    attributes = dict.fromkeys(WFLW_ATTRIBUTE_NAMES, 0)
    if len(parts) >= 207:
        try:
            values = [int(float(value)) for value in parts[200:206]]
        except ValueError as err:
            raise ValueError(f"WFLW line {line_number} has invalid attribute values") from err
        attributes = dict(zip(WFLW_ATTRIBUTE_NAMES, values, strict=True))
        image_rel = " ".join(parts[206:])
    else:
        image_rel = parts[-1]
    if not image_rel:
        raise ValueError(f"WFLW line {line_number} is missing image path")
    return points, bbox, attributes, image_rel


def _labels_from_wflw_attributes(attributes: dict[str, int]) -> tuple[str, ...]:
    """Return active normalized WFLW condition labels."""
    return tuple(name for name in WFLW_ATTRIBUTE_NAMES if int(attributes.get(name, 0)) != 0)


def _explicit_scenario_groups(scenarios: T.Sequence[str] | None = None) -> tuple[str, ...]:
    """Return explicitly requested scenario groups.

    ``--scenario`` supplies the default condition for unlabelled datasets and
    should not filter dataset-provided labels. Only ``--scenarios`` filters.
    """
    if scenarios is None:
        return ()
    return tuple(
        dict.fromkeys(
            _normalize_condition_label(value) for value in scenarios if str(value).strip()
        )
    )


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
        condition_labels = _entry_condition_labels(sample, "default")
        if allowed and not set(condition_labels).intersection(allowed):
            continue
        primary = condition_labels[0]
        if limit is not None and counts.get(primary, 0) >= limit:
            continue
        filtered.append(dict(sample))
        for condition in condition_labels:
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
    generated = tuple(
        dict.fromkeys(
            label for entry in entries for label in _entry_condition_labels(entry, scenario)
        )
    )
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


def _entry_condition_labels(entry: dict[str, T.Any], default: str) -> tuple[str, ...]:
    """Return normalized condition labels for a manifest entry."""
    raw = entry.get("conditions")
    labels: list[str] = []
    if isinstance(raw, (list, tuple, set)):
        labels.extend(_normalize_condition_label(value) for value in raw)
    elif raw:
        labels.append(_normalize_condition_label(raw))
    condition = entry.get("condition", default)
    if condition:
        labels.append(_normalize_condition_label(condition))
    labels = [label for label in labels if label]
    return tuple(dict.fromkeys(labels)) or (_normalize_condition_label(default),)


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
            and set(_entry_condition_labels(entry, "default")).intersection(replaced)
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
    image_name = image_path.name
    if image_path.suffix.lower() not in IMAGE_EXTS:
        image_name = f"{image_path.stem}.png"
    target = scenario_dir / image_name
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
                entry.setdefault("metadata", {})["overlay"] = str(
                    overlay_path.relative_to(output_dir)
                )
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
    expected_condition_groups: T.Sequence[str] = (),
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
    invalid_metric_samples: list[dict[str, T.Any]] = []
    sample_ids = [str(sample.get("sample_id", "")) for sample in manifest_samples]
    duplicate_ids = sorted(
        {sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1}
    )

    for sample in manifest_samples:
        condition_labels = _entry_condition_labels(sample, scenario)
        sample_dataset = str(sample.get("dataset", dataset))
        for label in condition_labels:
            condition_counts[label] = condition_counts.get(label, 0) + 1
        dataset_counts[sample_dataset] = dataset_counts.get(sample_dataset, 0) + 1

        source_schema = str(sample.get("source_schema", "2d_68"))
        source_schema_counts[source_schema] = source_schema_counts.get(source_schema, 0) + 1

        for label in condition_labels:
            selected_by_group.setdefault(label, []).append(list(_source_key_for_entry(sample)))

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

        normalizer_value = sample.get("normalizer")
        if normalizer_value is None:
            invalid_metric_samples.append(
                {
                    "sample_id": sample.get("sample_id", ""),
                    "error": "missing normalizer",
                }
            )
        else:
            try:
                normalizer = float(normalizer_value)
            except (TypeError, ValueError):
                invalid_metric_samples.append(
                    {
                        "sample_id": sample.get("sample_id", ""),
                        "normalizer": normalizer_value,
                        "error": "non-numeric normalizer",
                    }
                )
            else:
                if not np.isfinite(normalizer) or normalizer <= 0.0:
                    invalid_metric_samples.append(
                        {
                            "sample_id": sample.get("sample_id", ""),
                            "normalizer": normalizer,
                            "error": "normalizer must be finite and > 0",
                        }
                    )

        visibility_value = sample.get("visibility")
        if visibility_value is not None:
            try:
                visible = np.asarray(visibility_value, dtype=bool).ravel()
            except (TypeError, ValueError):
                invalid_metric_samples.append(
                    {
                        "sample_id": sample.get("sample_id", ""),
                        "visibility": visibility_value,
                        "error": "invalid visibility mask",
                    }
                )
            else:
                if visible.size and not bool(np.any(visible)):
                    invalid_metric_samples.append(
                        {
                            "sample_id": sample.get("sample_id", ""),
                            "error": "visibility mask has no true landmarks",
                        }
                    )

        metadata = sample.get("metadata", {})
        if isinstance(metadata, dict) and "face_bbox" in metadata:
            try:
                bbox = np.asarray(metadata["face_bbox"], dtype="float64")
            except (TypeError, ValueError):
                invalid_metric_samples.append(
                    {
                        "sample_id": sample.get("sample_id", ""),
                        "face_bbox": metadata.get("face_bbox"),
                        "error": "non-numeric face_bbox",
                    }
                )
            else:
                if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
                    invalid_metric_samples.append(
                        {
                            "sample_id": sample.get("sample_id", ""),
                            "face_bbox": metadata.get("face_bbox"),
                            "error": "face_bbox must be four finite values",
                        }
                    )

    if duplicate_sources is None:
        duplicate_sources = duplicate_source_audit(manifest_samples)

    expected_conditions = tuple(
        dict.fromkeys(_normalize_condition_label(group) for group in expected_condition_groups)
    )
    condition_shortfalls = {
        group: {"count": condition_counts.get(group, 0), "minimum": 1}
        for group in expected_conditions
        if condition_counts.get(group, 0) < 1
    }

    return {
        "schema_version": 1,
        "dataset": dataset,
        "total_entries": len(manifest_samples),
        "condition_counts": dict(sorted(condition_counts.items())),
        "condition_shortfalls": condition_shortfalls,
        "count_per_dataset": dict(sorted(dataset_counts.items())),
        "count_per_source_schema": dict(sorted(source_schema_counts.items())),
        "landmark_shape_counts": dict(sorted(shape_counts.items())),
        "missing_images": missing_images,
        "missing_landmarks": missing_landmarks,
        "invalid_landmarks": invalid_landmarks,
        "invalid_metric_samples": invalid_metric_samples,
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
    duplicates = validate_no_cross_group_source_overlap(final_entries, allow_overlap=allow_overlap)
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
                expected_condition_groups=scenario_groups,
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
