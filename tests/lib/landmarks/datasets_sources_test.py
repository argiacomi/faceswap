#!/usr/bin/env python3
"""Tests for landmark dataset source resolution and cached extraction."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from lib.landmarks.datasets import build_merl_rav_manifest, build_wflw_manifest
from lib.landmarks.datasets.sources import DatasetSourceSpec, resolve_dataset_source


def _points_98_row(image_rel: str = "images/sample.jpg") -> str:
    """Return one minimal WFLW 98-point annotation row."""
    coords = []
    for index in range(98):
        coords.extend((float(index), float(index + 100)))
    return (
        " ".join(str(value) for value in coords)
        + " 1 2 3 4"
        + " 0 0 0 0 0 0"
        + f" {image_rel}\n"
    )


def _write_wflw_archive(path: Path) -> None:
    """Write a tiny WFLW-like archive."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("WFLW/WFLW_annotations/list_98pt_rect_attr_test.txt", _points_98_row())
        zf.writestr("WFLW/WFLW_images/images/sample.jpg", b"not-used-by-manifest-build")


def test_resolve_dataset_source_reuses_cached_extraction(tmp_path: Path) -> None:
    """A fresh extracted cache directory is preferred without network access."""
    cache_dir = tmp_path / "cache"
    extracted = cache_dir / "demo" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "sentinel.txt").write_text("ok", encoding="utf-8")
    spec = DatasetSourceSpec(dataset="Demo", cache_subdir="demo", canonical_archive="demo.zip")

    resolved = resolve_dataset_source(spec, cache_dir=cache_dir, no_download=True)

    assert resolved == extracted


def test_resolve_dataset_source_downloads_and_extracts_archive(tmp_path: Path) -> None:
    """Download fallback stores and extracts archives into the dataset cache."""
    source_archive = tmp_path / "source.zip"
    _write_wflw_archive(source_archive)
    cache_dir = tmp_path / "cache"
    spec = DatasetSourceSpec(
        dataset="Demo",
        cache_subdir="demo",
        canonical_archive="demo.zip",
        url=source_archive.as_uri(),
    )

    resolved = resolve_dataset_source(spec, cache_dir=cache_dir)
    second = resolve_dataset_source(spec, cache_dir=cache_dir, no_download=True)

    assert resolved == cache_dir / "demo" / "extracted"
    assert second == resolved
    assert (cache_dir / "demo" / "demo.zip").is_file()
    assert (resolved / ".source.json").is_file()
    assert (resolved / "WFLW" / "WFLW_annotations" / "list_98pt_rect_attr_test.txt").is_file()


def test_resolve_dataset_source_rejects_bad_checksum(tmp_path: Path) -> None:
    """Configured checksums are enforced for downloaded archives."""
    source_archive = tmp_path / "source.zip"
    _write_wflw_archive(source_archive)
    spec = DatasetSourceSpec(
        dataset="Demo",
        cache_subdir="demo",
        canonical_archive="demo.zip",
        url=source_archive.as_uri(),
        sha256="0" * 64,
    )

    with pytest.raises(OSError, match="checksum mismatch"):
        resolve_dataset_source(spec, cache_dir=tmp_path / "cache")


def test_build_wflw_manifest_from_downloaded_archive(tmp_path: Path) -> None:
    """WFLW can build from a configured archive URL without explicit annotation paths."""
    source_archive = tmp_path / "wflw.zip"
    _write_wflw_archive(source_archive)
    output_dir = tmp_path / "out"

    manifest_path = build_wflw_manifest(
        None,
        output_dir,
        cache_dir=tmp_path / "cache",
        download_url=source_archive.as_uri(),
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "wflw"
    assert len(payload["samples"]) == 1
    assert (output_dir / "dataset_audit.json").is_file()


def test_merl_rav_without_aflw_images_fails_clearly_when_download_disabled(tmp_path: Path) -> None:
    """MERL-RAV does not pretend to be fully downloadable without AFLW images."""
    with pytest.raises(FileNotFoundError, match="AFLW images"):
        build_merl_rav_manifest(
            tmp_path / "out",
            cache_dir=tmp_path / "cache",
            no_download=True,
        )
