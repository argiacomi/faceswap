#!/usr/bin/env python3
"""Menpo2D dataset integration from MenpoBenchmark."""

from __future__ import annotations

import typing as T
from pathlib import Path

from lib.landmarks.datasets.menpo_benchmark import build_menpo_benchmark_manifest
from lib.landmarks.datasets.sources import DEFAULT_CACHE_DIR, DatasetSourceSpec

# MenpoBenchmark README:
# https://github.com/jiankangdeng/MenpoBenchmark
# Menpo2D Google Drive file id from:
# https://drive.google.com/file/d/1CUqs0n135lye6J6RM5FQXT_DIT45dKvP/view
MENPO2D_GOOGLE_DRIVE_FILE_ID = "1CUqs0n135lye6J6RM5FQXT_DIT45dKvP"

MENPO2D_SOURCE = DatasetSourceSpec(
    dataset="Menpo2D",
    cache_subdir="menpo2d",
    canonical_archive="Menpo2D.zip",
    cache_aliases=("menpo2d.zip", "Menpo2D.tar.gz", "menpo2d.tgz"),
    extracted_aliases=("Menpo2D", "menpo2d"),
    google_drive_file_id=MENPO2D_GOOGLE_DRIVE_FILE_ID,
    manual_hint=(
        "Menpo2D is distributed by MenpoBenchmark via Google Drive. "
        "Install the optional Google Drive downloader dependency if needed, "
        "or place Menpo2D.zip/extracted Menpo2D under .fs_cache/landmark_quality/menpo2d."
    ),
)


def build_menpo2d_manifest(
    output_dir: str | Path,
    *,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "menpo2d",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
    include_39pt_profile: bool = True,
) -> Path:
    return build_menpo_benchmark_manifest(
        dataset_name="menpo2d",
        source_spec=MENPO2D_SOURCE,
        output_dir=output_dir,
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
        include_39pt_profile=include_39pt_profile,
    )
