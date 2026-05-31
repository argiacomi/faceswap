#!/usr/bin/env python3
"""Tests for COFW-68 source materialization."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.cofw68 import build_cofw68_json_from_sources, resolve_cofw68_json


def _points_68(offset: float) -> np.ndarray:
    """Return deterministic 68-point landmarks."""
    return np.stack(  # type: ignore[no-any-return]
        (
            np.linspace(10 + offset, 77 + offset, 68, dtype="float32"),
            np.linspace(20 + offset, 87 + offset, 68, dtype="float32"),
        ),
        axis=1,
    )


def test_build_cofw68_json_from_color_mat_and_annotation_mats(tmp_path: Path) -> None:
    """COFW-68 materialization joins Caltech images and 68-point annotation mats."""
    scipy_io = pytest.importorskip("scipy.io")
    color_root = tmp_path / "color"
    annotation_root = tmp_path / "annotations" / "COFW68_Data"
    test_annotations = annotation_root / "test_annotations"
    color_root.mkdir(parents=True)
    test_annotations.mkdir(parents=True)

    images = np.empty((2, 1), dtype=object)  # type: ignore[var-annotated]
    images[0, 0] = np.full((8, 9, 3), 32, dtype="uint8")
    images[1, 0] = np.full((10, 7, 3), 64, dtype="uint8")
    scipy_io.savemat(str(color_root / "COFW_test_color.mat"), {"IsT": images})

    scipy_io.savemat(
        str(test_annotations / "1_points.mat"),
        {"Points": _points_68(0), "Occ": np.zeros((1, 68), dtype="uint8")},
    )
    scipy_io.savemat(
        str(test_annotations / "2_points.mat"),
        {"Points": _points_68(5), "Occ": np.ones((1, 68), dtype="uint8")},
    )
    scipy_io.savemat(
        str(annotation_root / "cofw68_test_bboxes.mat"),
        {"bboxes": np.asarray([[1, 2, 11, 12], [3, 4, 13, 14]], dtype="float32")},
    )

    output = build_cofw68_json_from_sources(
        color_root=color_root,
        annotation_root=annotation_root,
        output_json=tmp_path / "cofw_68.json",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    samples = payload["samples"]
    assert payload["dataset"] == "cofw_68"
    assert [sample["sample_id"] for sample in samples] == ["cofw68/0001", "cofw68/0002"]
    assert Path(samples[0]["image"]).is_file()
    assert np.asarray(samples[0]["landmarks"], dtype="float32").shape == (68, 2)
    assert samples[0]["face_bbox"] == [1.0, 2.0, 12.0, 14.0]
    assert samples[0]["metadata"]["face_bbox"] == [1.0, 2.0, 12.0, 14.0]
    assert samples[0]["metadata"]["face_bbox_raw"] == [1.0, 2.0, 11.0, 12.0]
    assert samples[0]["metadata"]["face_bbox_format"] == "ltrb"
    assert samples[0]["metadata"]["face_bbox_raw_format"] == "xywh"
    assert samples[0]["metadata"]["face_bbox_source"] == "cofw68_benchmark"
    assert samples[0]["visibility"] == [True] * 68
    assert samples[1]["conditions"] == {"occlusion": True}
    assert samples[1]["metadata"]["occlusion"] == [1] * 68
    assert samples[1]["visibility"] == [False] * 68


def test_resolve_cofw68_json_uses_cached_archives_when_download_disabled(
    tmp_path: Path,
) -> None:
    """Cached COFW source archives can materialize cofw_68.json offline."""
    scipy_io = pytest.importorskip("scipy.io")
    staging = tmp_path / "staging"
    color_root = staging / "color"
    annotation_root = staging / "annotations" / "COFW68_Data"
    test_annotations = annotation_root / "test_annotations"
    color_root.mkdir(parents=True)
    test_annotations.mkdir(parents=True)
    images = np.empty((1, 1), dtype=object)  # type: ignore[var-annotated]
    images[0, 0] = np.full((8, 9, 3), 32, dtype="uint8")
    scipy_io.savemat(str(color_root / "COFW_test_color.mat"), {"IsT": images})
    scipy_io.savemat(
        str(test_annotations / "1_points.mat"),
        {"Points": _points_68(0), "Occ": np.zeros((1, 68), dtype="uint8")},
    )

    cache_root = tmp_path / "cache" / "cofw"
    cache_root.mkdir(parents=True)
    with zipfile.ZipFile(cache_root / "COFW_color.zip", "w") as zf:
        zf.write(color_root / "COFW_test_color.mat", "COFW_test_color.mat")
    with zipfile.ZipFile(cache_root / "cofw68-benchmark-master.zip", "w") as zf:
        zf.write(test_annotations / "1_points.mat", "COFW68_Data/test_annotations/1_points.mat")

    output = resolve_cofw68_json(cache_dir=tmp_path / "cache", no_download=True)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"] == "cofw_68"
    assert len(payload["samples"]) == 1
    assert Path(payload["samples"][0]["image"]).is_file()
