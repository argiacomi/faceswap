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

from lib.landmarks.datasets import (
    audit_existing_manifest,
    build_cofw_manifest,
    build_manifest,
    build_wflw_manifest,
)
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


def _write_png(path: Path) -> None:
    """Write a tiny valid PNG for fixture-copy and overlay tests."""
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((96, 96, 3), dtype="uint8")
    image[..., 1] = 128
    assert cv2.imwrite(str(path), image)


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


def test_scenario_filter_and_samples_per_scenario(tmp_path: Path) -> None:
    """Explicit scenario filters and sample caps are applied per condition."""
    source = tmp_path / "cofw.json"
    source.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "occ1",
                        "image": "",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "occluded"},
                    },
                    {
                        "sample_id": "occ2",
                        "image": "",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "occluded"},
                    },
                    {
                        "sample_id": "clean",
                        "image": "",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "clean"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_cofw_manifest(
        source,
        tmp_path / "out",
        scenarios=("occluded",),
        samples_per_scenario=1,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text())

    assert [sample["sample_id"] for sample in payload["samples"]] == ["occ1"]
    assert audit["condition_counts"] == {"occluded": 1}


def test_manifest_merge_replaces_only_selected_dataset_slice(tmp_path: Path) -> None:
    """Merge mode keeps unrelated entries and replaces the selected scenario slice."""
    first = tmp_path / "first.json"
    first.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "keep",
                        "image": "",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "keep"},
                    },
                    {
                        "sample_id": "old",
                        "image": "",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "replace"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    second = tmp_path / "second.json"
    second.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "new",
                        "image": "",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "replace"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    build_cofw_manifest(first, tmp_path / "out")
    manifest = build_cofw_manifest(
        second,
        tmp_path / "out",
        scenarios=("replace",),
        manifest_mode="merge",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert [(sample["condition"], sample["sample_id"]) for sample in payload["samples"]] == [
        ("keep", "keep"),
        ("replace", "new"),
    ]


def test_prepared_manifest_writes_relative_generated_fixtures_and_overlay(tmp_path: Path) -> None:
    """Prepared directory manifests copy images/landmarks into portable generated paths."""
    source = tmp_path / "source"
    source.mkdir()
    _write_png(source / "sample.png")
    np.save(str(source / "sample.npy"), _points_68())

    manifest = build_manifest(
        source,
        tmp_path / "out",
        dataset="cofw",
        write_overlays=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    assert not Path(sample["image"]).is_absolute()
    assert not Path(sample["landmarks"]).is_absolute()
    assert (tmp_path / "out" / sample["image"]).is_file()
    assert (tmp_path / "out" / sample["landmarks"]).is_file()
    assert (tmp_path / "out" / sample["metadata"]["overlay"]).is_file()


def test_audit_only_rewrites_dataset_audit(tmp_path: Path) -> None:
    """Audit-only mode validates an existing manifest without regenerating fixtures."""
    source = tmp_path / "source"
    source.mkdir()
    _write_png(source / "sample.png")
    np.save(str(source / "sample.npy"), _points_68())
    build_manifest(source, tmp_path / "out", dataset="cofw")
    (tmp_path / "out" / "dataset_audit.json").unlink()

    rc = build_quality_dataset.main(
        [
            "--dataset",
            "cofw",
            "--output-dir",
            str(tmp_path / "out"),
            "--audit-only",
        ]
    )
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text())

    assert rc == 0
    assert audit["total_entries"] == 1
    assert audit["landmark_shape_counts"] == {"68x2": 1}


def test_audit_only_rejects_cross_group_overlap(tmp_path: Path) -> None:
    """Cross-group source overlap is rejected unless explicitly allowed."""
    output = tmp_path / "out"
    output.mkdir()
    np.save(str(output / "landmarks.npy"), _points_68())
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "cofw",
                "samples": [
                    {
                        "sample_id": "a",
                        "dataset": "cofw",
                        "condition": "one",
                        "landmarks": "landmarks.npy",
                        "source": {"dataset": "cofw", "source_id": "same"},
                    },
                    {
                        "sample_id": "b",
                        "dataset": "cofw",
                        "condition": "two",
                        "landmarks": "landmarks.npy",
                        "source": {"dataset": "cofw", "source_id": "same"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cross-group duplicate"):
        audit_existing_manifest(output)

    audit_path = audit_existing_manifest(output, allow_overlap=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["overlap"] == {"allow_overlap": True, "duplicate_count": 1, "has_overlap": True}


def test_safe_zip_blocks_path_traversal(tmp_path: Path) -> None:
    """Zip extraction blocks path traversal."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../evil.txt", "pwned")

    with zipfile.ZipFile(archive, "r") as zf, pytest.raises(ValueError, match="path traversal"):
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
    with tarfile.open(archive, "r:*") as tf, pytest.raises(ValueError, match="path traversal"):
        safe_tar_extractall(tf, tmp_path / "dest")


def test_safe_tar_blocks_symlink(tmp_path: Path) -> None:
    """Tar extraction blocks symlinks."""
    archive = _make_tar(tmp_path / "evil_link.tar", "bad_link", symlink=True)
    with tarfile.open(archive, "r:*") as tf, pytest.raises(ValueError, match="symlink"):
        safe_tar_extractall(tf, tmp_path / "dest")
