#!/usr/bin/env python3
"""Tests for landmark dataset manifest builders."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.landmarks.datasets import (
    build_aflw2000_3d_manifest,
    build_cofw_manifest,
    build_directory_manifest,
    build_manifest,
    build_merl_rav_manifest,
    build_wflw_manifest,
)


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


def test_directory_builders_reject_missing_source_dirs(tmp_path: Path) -> None:
    """Directory-style manifest builders should not silently write empty manifests."""
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="source directory not found"):
        build_manifest(missing, tmp_path / "out-flat", dataset="directory")
    with pytest.raises(FileNotFoundError, match="source directory not found"):
        build_directory_manifest(missing, tmp_path / "out-recursive")


def test_wflw_builder_preserves_attributes_and_filters_by_normalized_label(
    tmp_path: Path,
) -> None:
    """WFLW official attributes become metadata and condition labels."""
    points = _points_98()
    row = (
        " ".join(str(value) for value in points.reshape(-1))
        + " 1 2 3 4"
        + " 1 0 1 0 1 0"
        + " images/sample.jpg\n"
    )
    annotation = tmp_path / "list_98pt_rect_attr_test.txt"
    annotation.write_text(row, encoding="utf-8")

    manifest_path = build_wflw_manifest(
        annotation,
        tmp_path / "out",
        scenarios=("Occlusion",),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text())
    sample = payload["samples"][0]

    assert sample["conditions"] == ["pose", "illumination", "occlusion"]
    assert sample["metadata"]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert sample["metadata"]["attributes"] == {
        "blur": 0,
        "expression": 0,
        "illumination": 1,
        "makeup": 0,
        "occlusion": 1,
        "pose": 1,
    }
    assert audit["condition_counts"] == {"illumination": 1, "occlusion": 1, "pose": 1}
    assert audit["condition_shortfalls"] == {}
    assert np.load(tmp_path / "out" / sample["landmarks"]).shape == (68, 2)


def test_wflw_builder_writes_visual_mapping_overlay(tmp_path: Path) -> None:
    """The WFLW 98-to-68 mapping is inspectable through generated GT overlays."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((256, 256, 3), dtype="uint8"))
    points = _points_98()
    row = (
        " ".join(str(value) for value in points.reshape(-1))
        + " 1 2 3 4"
        + " 0 0 0 0 0 0"
        + " images/sample.jpg\n"
    )
    annotation = tmp_path / "list_98pt_rect_attr_test.txt"
    annotation.write_text(row, encoding="utf-8")

    manifest_path = build_wflw_manifest(annotation, tmp_path / "out", write_overlays=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    overlay_path = tmp_path / "out" / sample["metadata"]["overlay"]

    assert sample["source_schema"] == "2d_98"
    assert sample["metadata"]["overlay"].endswith("landmarks_gt.png")
    assert overlay_path.is_file()
    assert cv2.imread(str(overlay_path), cv2.IMREAD_COLOR) is not None
    assert np.load(tmp_path / "out" / sample["landmarks"]).shape == (68, 2)


def test_wflw_builder_disambiguates_multiple_faces_in_same_image(tmp_path: Path) -> None:
    """WFLW may annotate multiple distinct faces in one source image."""
    points = _points_98()
    rows = []
    for attr_values in ("1 0 0 0 0 0", "0 0 0 0 1 0"):
        rows.append(
            " ".join(str(value) for value in points.reshape(-1))
            + " 1 2 3 4"
            + f" {attr_values}"
            + " images/group.jpg\n"
        )
    annotation = tmp_path / "list_98pt_rect_attr_test.txt"
    annotation.write_text("".join(rows), encoding="utf-8")

    manifest_path = build_wflw_manifest(annotation, tmp_path / "out")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text())
    sample_ids = [sample["sample_id"] for sample in payload["samples"]]
    source_ids = [sample["source"]["source_id"] for sample in payload["samples"]]

    assert sorted(sample_ids) == ["images/group#face-01", "images/group#face-02"]
    assert sorted(source_ids) == sorted(sample_ids)
    assert audit["duplicate_source_ids"] == []


def test_cofw_builder_preserves_occlusion_metadata(tmp_path: Path) -> None:
    """COFW JSON occlusion metadata contributes normalized coverage labels."""
    source = tmp_path / "cofw.json"
    source.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "face",
                        "image": "face.png",
                        "landmarks": _points_68().tolist(),
                        "conditions": {"scenario": "Hard Occlusion"},
                        "occlusion": [False, True, False],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest_path = build_cofw_manifest(source, tmp_path / "out")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text())
    sample = payload["samples"][0]

    assert sample["condition"] == "hard_occlusion"
    assert sample["conditions"] == ["hard_occlusion", "occlusion"]
    assert sample["metadata"]["occlusion"] == [False, True, False]
    assert audit["condition_counts"] == {"hard_occlusion": 1, "occlusion": 1}


def test_merl_rav_json_manifest_builder(tmp_path: Path) -> None:
    """MERL-RAV MVP builder accepts JSON-style sample manifests."""
    source = tmp_path / "merl_rav.json"
    source.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "merl",
                        "image_path": "merl.jpg",
                        "points": _points_68().tolist(),
                        "conditions": {"labels": ["Yaw Left", "Blur"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest_path = build_merl_rav_manifest(tmp_path / "out", source_dir=tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["dataset"] == "merl-rav"
    assert payload["samples"][0]["conditions"] == ["yaw_left", "blur"]


def test_aflw2000_3d_json_manifest_uses_xy_landmarks(tmp_path: Path) -> None:
    """AFLW2000-3D MVP builder projects 68x3 JSON landmarks to canonical xy."""
    points = np.concatenate(
        (_points_68(), np.ones((68, 1), dtype="float32")),
        axis=1,
    )
    source = tmp_path / "manifest.json"
    source.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "aflw",
                        "image": "aflw.jpg",
                        "landmarks": points.tolist(),
                        "conditions": ["Large Pose"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest_path = build_aflw2000_3d_manifest(tmp_path / "out", source_dir=tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    landmarks = np.load(tmp_path / "out" / sample["landmarks"])

    assert payload["dataset"] == "aflw2000-3d"
    assert sample["source_schema"] == "3d_68"
    assert sample["conditions"] == ["large_pose"]
    np.testing.assert_array_equal(landmarks, _points_68())
