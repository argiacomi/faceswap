#!/usr/bin/env python3
"""Tests for 300W dataset manifest building."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.w300 import (
    W300_DVE_SHA1,
    W300_DVE_URL,
    W300_OFFICIAL_PAGE,
    W300_OFFICIAL_PART_NAMES,
    build_300w_manifest,
)


def _points_68() -> np.ndarray:
    """Return deterministic 68-point landmarks."""
    return np.stack(
        (
            np.linspace(10, 77, 68, dtype="float32"),
            np.linspace(20, 87, 68, dtype="float32"),
        ),
        axis=1,
    )


def _write_png(path: Path) -> None:
    """Write a tiny valid PNG image."""
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype="uint8")
    image[..., 2] = 255
    assert cv2.imwrite(str(path), image)


def _write_pts(path: Path, points: np.ndarray) -> None:
    """Write a standard Menpo/300W .pts annotation file."""
    rows = ["version: 1", "n_points: 68", "{"]
    rows.extend(f"{float(x):.4f} {float(y):.4f}" for x, y in points)
    rows.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_300w_manifest_from_pts_and_images(tmp_path: Path) -> None:
    """300W builder emits canonical 68-point samples with split labels."""
    root = tmp_path / "300w"
    points = _points_68()
    _write_png(root / "ibug" / "face.jpg")
    _write_pts(root / "ibug" / "face.pts", points)
    _write_png(root / "helen" / "common.jpg")
    _write_pts(root / "helen" / "common.pts", points + 1)

    manifest = build_300w_manifest(root / "out", source_dir=root)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit = json.loads((root / "out" / "dataset_audit.json").read_text(encoding="utf-8"))
    samples = payload["samples"]

    assert payload["dataset"] == "300w"
    assert {sample["condition"] for sample in samples} == {"challenging", "common"}
    assert {sample["source_schema"] for sample in samples} == {"2d_68"}
    assert audit["count_per_source_schema"] == {"2d_68": 2}
    assert audit["landmark_shape_counts"] == {"68x2": 2}
    for sample in samples:
        landmarks = np.load(root / "out" / sample["landmarks"])
        assert landmarks.shape == (68, 2)
        assert sample["metadata"]["face_bbox_source"] == "300w_68_landmark_extrema"
        assert sample["metadata"]["dve_source_url"] == W300_DVE_URL
        assert sample["metadata"]["dve_sha1"] == W300_DVE_SHA1
        assert sample["metadata"]["ibug_source_page"] == W300_OFFICIAL_PAGE


def test_build_300w_manifest_from_official_split_cache(tmp_path: Path) -> None:
    """Official iBUG split zip parts are combined and extracted from cache."""
    source = tmp_path / "source"
    points = _points_68()
    _write_png(source / "ibug" / "face.jpg")
    _write_pts(source / "ibug" / "face.pts", points)

    archive = tmp_path / "300w.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source).as_posix())

    payload = archive.read_bytes()
    cache_root = tmp_path / "cache" / "300w"
    cache_root.mkdir(parents=True)
    chunk_size = max(1, len(payload) // len(W300_OFFICIAL_PART_NAMES))
    offset = 0
    for index, name in enumerate(W300_OFFICIAL_PART_NAMES):
        if index == len(W300_OFFICIAL_PART_NAMES) - 1:
            chunk = payload[offset:]
        else:
            chunk = payload[offset : offset + chunk_size]
        offset += len(chunk)
        (cache_root / name).write_bytes(chunk)

    manifest = build_300w_manifest(
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        no_download=True,
    )

    payload_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload_json["samples"]) == 1
    assert (cache_root / "300w.zip").is_file()
    assert (cache_root / "extracted" / "ibug" / "face.pts").is_file()


def test_build_300w_manifest_reports_source_setup_when_no_download(tmp_path: Path) -> None:
    """Missing sources produce actionable DVE and iBUG setup guidance."""
    with pytest.raises(FileNotFoundError) as exc:
        build_300w_manifest(
            tmp_path / "out",
            cache_dir=tmp_path / "cache",
            no_download=True,
        )

    message = str(exc.value)
    assert W300_DVE_URL in message
    assert W300_DVE_SHA1 in message
    assert "300w.tar.gz" in message
    assert W300_OFFICIAL_PART_NAMES[0] in message


def test_build_300w_manifest_reports_official_missing_parts(tmp_path: Path) -> None:
    """Missing iBUG split parts produce actionable setup guidance."""
    cache_root = tmp_path / "cache" / "300w"
    cache_root.mkdir(parents=True)
    (cache_root / W300_OFFICIAL_PART_NAMES[0]).write_bytes(b"partial")

    with pytest.raises(FileNotFoundError) as exc:
        build_300w_manifest(
            tmp_path / "out",
            cache_dir=tmp_path / "cache",
            no_download=True,
        )

    message = str(exc.value)
    assert "Missing iBUG split parts" in message
    assert W300_OFFICIAL_PAGE in message
    assert W300_OFFICIAL_PART_NAMES[1] in message
