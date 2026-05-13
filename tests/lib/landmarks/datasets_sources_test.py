#!/usr/bin/env python3
"""Tests for landmark dataset source resolution and cached extraction."""

from __future__ import annotations

import json
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from lib.landmarks.datasets import build_merl_rav_manifest, build_wflw_manifest
from lib.landmarks.datasets.sources import (
    DatasetSourceSpec,
    MultiDatasetSourceSpec,
    MultiSourcePart,
    resolve_dataset_source,
    resolve_multipart_dataset_source,
)
from tools.landmarks import build_quality_dataset


def _points_98_row(image_rel: str = "images/sample.jpg") -> str:
    """Return one minimal WFLW 98-point annotation row."""
    coords = []
    for index in range(98):
        coords.extend((float(index), float(index + 100)))
    return (
        " ".join(str(value) for value in coords) + " 1 2 3 4" + " 0 0 0 0 0 0" + f" {image_rel}\n"
    )


def _write_wflw_archive(path: Path) -> None:
    """Write a tiny WFLW-like archive."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("WFLW/WFLW_annotations/list_98pt_rect_attr_test.txt", _points_98_row())
        zf.writestr("WFLW/WFLW_images/images/sample.jpg", b"not-used-by-manifest-build")


def _write_named_archive(path: Path, filename: str, content: str) -> None:
    """Write a tiny archive with one named text file."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(filename, content)


def _write_tar_gz(path: Path, files: dict[str, str]) -> None:
    """Write a tiny tar.gz archive with string files."""
    staging = path.parent / f"{path.name}.staging"
    staging.mkdir()
    try:
        for name, content in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        with tarfile.open(path, "w:gz") as tf:
            for file_path in sorted(staging.rglob("*")):
                if file_path.is_file():
                    tf.add(file_path, arcname=file_path.relative_to(staging).as_posix())
    finally:
        for child in sorted(staging.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        if staging.exists():
            staging.rmdir()


def _write_wflw_annotations_archive(path: Path, *, image_rel: str = "images/sample.jpg") -> None:
    """Write a WFLW annotations tarball."""
    _write_tar_gz(
        path,
        {"WFLW_annotations/list_98pt_rect_attr_test.txt": _points_98_row(image_rel)},
    )


def _write_wflw_images_archive(path: Path) -> None:
    """Write a WFLW images zip archive."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("WFLW_images/images/sample.jpg", b"not-used-by-manifest-build")


def _wflw_multipart_spec(annotations: Path, images: Path) -> MultiDatasetSourceSpec:
    """Return a test-local WFLW multipart source spec."""
    return MultiDatasetSourceSpec(
        dataset="WFLW",
        cache_subdir="wflw",
        parts=(
            MultiSourcePart(
                name="annotations",
                archive_name="WFLW_annotations.tar.gz",
                url=annotations.as_uri(),
            ),
            MultiSourcePart(
                name="images",
                archive_name="WFLW_images.zip",
                url=images.as_uri(),
            ),
        ),
    )


def test_resolve_dataset_source_reuses_cached_extraction(tmp_path: Path) -> None:
    """A cached extracted directory is used when no archive can make it stale."""
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


def test_resolve_dataset_source_refreshes_stale_extraction_when_archive_changes(
    tmp_path: Path,
) -> None:
    """A stale extracted cache is not trusted when the cached archive changed."""
    cache_dir = tmp_path / "cache"
    archive = cache_dir / "demo" / "demo.zip"
    archive.parent.mkdir(parents=True)
    spec = DatasetSourceSpec(dataset="Demo", cache_subdir="demo", canonical_archive="demo.zip")
    _write_named_archive(archive, "old.txt", "old")
    os.utime(archive, (100, 100))

    first = resolve_dataset_source(spec, cache_dir=cache_dir, no_download=True)
    assert (first / "old.txt").is_file()

    _write_named_archive(archive, "new.txt", "newer-content")
    os.utime(archive, (200, 200))
    second = resolve_dataset_source(spec, cache_dir=cache_dir, no_download=True)

    assert second == first
    assert not (second / "old.txt").exists()
    assert (second / "new.txt").read_text(encoding="utf-8") == "newer-content"


def test_resolve_multipart_dataset_source_downloads_extracts_and_reuses(tmp_path: Path) -> None:
    """Multipart sources download all parts and extract them into one cache root."""
    annotations = tmp_path / "annotations.tar.gz"
    images = tmp_path / "images.zip"
    _write_wflw_annotations_archive(annotations)
    _write_wflw_images_archive(images)
    cache_dir = tmp_path / "cache"
    spec = _wflw_multipart_spec(annotations, images)

    resolved = resolve_multipart_dataset_source(spec, cache_dir=cache_dir)
    second = resolve_multipart_dataset_source(spec, cache_dir=cache_dir, no_download=True)

    assert resolved == cache_dir / "wflw" / "extracted"
    assert second == resolved
    assert (cache_dir / "wflw" / "WFLW_annotations.tar.gz").is_file()
    assert (cache_dir / "wflw" / "WFLW_images.zip").is_file()
    assert (resolved / "WFLW_annotations" / "list_98pt_rect_attr_test.txt").is_file()
    assert (resolved / "WFLW_images" / "images" / "sample.jpg").is_file()


def test_resolve_multipart_dataset_source_refreshes_when_either_archive_changes(
    tmp_path: Path,
) -> None:
    """Multipart extraction freshness depends on every cached source archive."""
    annotations = tmp_path / "annotations.tar.gz"
    images = tmp_path / "images.zip"
    _write_wflw_annotations_archive(annotations, image_rel="images/sample.jpg")
    _write_wflw_images_archive(images)
    cache_dir = tmp_path / "cache"
    spec = _wflw_multipart_spec(annotations, images)
    first = resolve_multipart_dataset_source(spec, cache_dir=cache_dir)
    assert "images/sample.jpg" in (
        first / "WFLW_annotations" / "list_98pt_rect_attr_test.txt"
    ).read_text(encoding="utf-8")

    cached_annotations = cache_dir / "wflw" / "WFLW_annotations.tar.gz"
    _write_wflw_annotations_archive(cached_annotations, image_rel="images/changed.jpg")
    os.utime(cached_annotations, (300, 300))
    second = resolve_multipart_dataset_source(spec, cache_dir=cache_dir, no_download=True)

    assert second == first
    assert "images/changed.jpg" in (
        second / "WFLW_annotations" / "list_98pt_rect_attr_test.txt"
    ).read_text(encoding="utf-8")


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


def test_build_quality_dataset_wflw_download_official_uses_multipart_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WFLW official CLI flag resolves annotations and images from multipart cache."""
    annotations = tmp_path / "annotations.tar.gz"
    images = tmp_path / "images.zip"
    _write_wflw_annotations_archive(annotations)
    _write_wflw_images_archive(images)
    import lib.landmarks.datasets.sources as sources

    monkeypatch.setattr(sources, "WFLW_OFFICIAL_SOURCE", _wflw_multipart_spec(annotations, images))
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    build_quality_dataset.main(
        [
            "--dataset",
            "wflw",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            "--wflw-download-official",
        ]
    )

    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "wflw"
    assert len(payload["samples"]) == 1
    assert (cache_dir / "wflw" / "WFLW_annotations.tar.gz").is_file()
    assert (cache_dir / "wflw" / "WFLW_images.zip").is_file()
    assert (cache_dir / "wflw" / "extracted" / ".source.json").is_file()


def test_merl_rav_without_aflw_images_fails_clearly_when_download_disabled(tmp_path: Path) -> None:
    """MERL-RAV does not pretend to be fully downloadable without AFLW images."""
    with pytest.raises(FileNotFoundError, match="AFLW images"):
        build_merl_rav_manifest(
            tmp_path / "out",
            cache_dir=tmp_path / "cache",
            no_download=True,
        )
