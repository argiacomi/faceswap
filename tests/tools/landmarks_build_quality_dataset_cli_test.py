#!/usr/bin/env python3
"""Tests for landmark quality dataset source resolution and CLI orchestration."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets import build_cofw_manifest, build_wflw_manifest
from lib.landmarks.datasets.sources import safe_tar_extractall, safe_zip_extractall
from tools.landmarks import build_quality_dataset


def _points_68() -> np.ndarray:
    """Return deterministic 68-point landmarks."""
    return np.stack(
        (
            np.linspace(0, 67, 68, dtype="float32"),
            np.linspace(10, 77, 68, dtype="float32"),
        ),
        axis=1,
    )


def _points_98() -> np.ndarray:
    """Return deterministic 98-point landmarks."""
    return np.stack(
        (
            np.linspace(0, 97, 98, dtype="float32"),
            np.linspace(100, 197, 98, dtype="float32"),
        ),
        axis=1,
    )


def test_cofw_builder_uses_cached_json_without_manual_source(tmp_path: Path) -> None:
    """COFW falls back to the standard cache before attempting download."""
    cache_json = tmp_path / "cache" / "cofw" / "cofw_68.json"
    cache_json.parent.mkdir(parents=True)
    cache_json.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "cached_face",
                        "image": "face.png",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "cached"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_cofw_manifest(
        None,
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        no_download=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text())

    assert payload["samples"][0]["sample_id"] == "cached_face"
    assert payload["samples"][0]["condition"] == "cached"
    assert audit["count_per_source_schema"] == {"2d_68": 1}
    assert audit["landmark_shape_counts"] == {"68x2": 1}


def test_wflw_builder_uses_cached_archive(tmp_path: Path) -> None:
    """WFLW can resolve an archive from the standard cache waterfall."""
    archive = tmp_path / "cache" / "wflw" / "wflw.zip"
    archive.parent.mkdir(parents=True)
    points = _points_98()
    annotation = " ".join(str(value) for value in points.reshape(-1)) + " images/sample.jpg\n"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("WFLW/list_98pt_rect_attr_test.txt", annotation)
        zf.writestr("WFLW/WFLW_images/images/sample.jpg", b"fake image bytes")

    manifest = build_wflw_manifest(
        None,
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        no_download=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text())

    assert payload["dataset"] == "wflw"
    assert np.load(tmp_path / "out" / payload["samples"][0]["landmarks"]).shape == (68, 2)
    assert audit["count_per_source_schema"] == {"2d_98": 1}
    assert audit["missing_images"] == []


def test_cli_runs_from_cofw_cache_without_source_dir(tmp_path: Path) -> None:
    """CLI no longer requires --source-dir when the dataset exists in cache."""
    cache_json = tmp_path / "cache" / "cofw" / "cofw_68.json"
    cache_json.parent.mkdir(parents=True)
    cache_json.write_text(
        json.dumps([{"sample_id": "face", "image": "", "landmarks": _points_68().tolist()}]),
        encoding="utf-8",
    )

    rc = build_quality_dataset.main(
        [
            "--dataset",
            "cofw",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-download",
        ]
    )

    assert rc == 0
    assert (tmp_path / "out" / "manifest.json").is_file()
    assert (tmp_path / "out" / "SOURCE_NOTES.md").is_file()


def test_cli_rejects_conflicting_source_modes(tmp_path: Path) -> None:
    """Ambiguous source modes are rejected before any build work starts."""
    with pytest.raises(ValueError, match="Pass only one source mode"):
        build_quality_dataset.main(
            [
                "--dataset",
                "wflw",
                "--wflw-annotations",
                str(tmp_path / "wflw.txt"),
                "--source-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_no_download_failure_mentions_manual_hint(tmp_path: Path) -> None:
    """Offline mode fails with a useful manual-source hint when cache is missing."""
    with pytest.raises(FileNotFoundError, match="Download disabled by --no-download"):
        build_cofw_manifest(
            None,
            tmp_path / "out",
            cache_dir=tmp_path / "missing_cache",
            no_download=True,
        )


def test_safe_zip_blocks_path_traversal(tmp_path: Path) -> None:
    """Zip extraction blocks path traversal."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../evil.txt", "pwned")
    with zipfile.ZipFile(archive, "r") as zf:
        with pytest.raises(ValueError, match="path traversal"):
            safe_zip_extractall(zf, tmp_path / "dest")


def _make_tar(path: Path, member_name: str, *, symlink: bool = False) -> Path:
    """Create a small tar archive for safe extraction tests."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tf:
        info = tarfile.TarInfo(member_name)
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        else:
            content = b"data"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    path.write_bytes(buffer.getvalue())
    return path


def test_safe_tar_blocks_path_traversal(tmp_path: Path) -> None:
    """Tar extraction blocks path traversal."""
    archive = _make_tar(tmp_path / "evil.tar", "../../evil.txt")
    with tarfile.open(archive, "r:*") as tf:
        with pytest.raises(ValueError, match="path traversal"):
            safe_tar_extractall(tf, tmp_path / "dest")


def test_safe_tar_blocks_symlink(tmp_path: Path) -> None:
    """Tar extraction blocks symlinks."""
    archive = _make_tar(tmp_path / "evil_link.tar", "bad_link", symlink=True)
    with tarfile.open(archive, "r:*") as tf:
        with pytest.raises(ValueError, match="symlink"):
            safe_tar_extractall(tf, tmp_path / "dest")
