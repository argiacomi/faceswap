#!/usr/bin/env python3
"""Tests for landmark dataset manifest builders."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.landmarks.datasets import (
    _visibility_from_jaw_asymmetry,
    build_aflw2000_3d_manifest,
    build_cofw_manifest,
    build_directory_manifest,
    build_manifest,
    build_merl_rav_manifest,
    build_wflw_manifest,
)
from lib.landmarks.datasets.multipie import _parse_line, build_multipie_manifest


def _points_68() -> np.ndarray:
    """Return deterministic 68-point landmarks."""
    return np.stack(  # type: ignore[no-any-return]
        (
            np.linspace(0, 67, 68, dtype="float32"),
            np.linspace(10, 77, 68, dtype="float32"),
        ),
        axis=1,
    )


def _points_98() -> np.ndarray:
    """Return deterministic 98-point landmarks."""
    return np.stack(  # type: ignore[no-any-return]
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
    rows = [
        (
            " ".join(str(value) for value in points.reshape(-1))
            + " 1 2 3 4"
            + f" {attr_values}"
            + " images/group.jpg\n"
        )
        for attr_values in ("1 0 0 0 0 0", "0 0 0 0 1 0")
    ]
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


def _frontal_canonical_68() -> np.ndarray:
    """Return synthetic 68-point landmarks where the nose sits between equally-spaced ear jaws."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[0] = [0.0, 50.0]
    points[16] = [100.0, 50.0]
    points[30] = [50.0, 30.0]  # nose tip exactly at midpoint
    return points


def _profile_right_canonical_68() -> np.ndarray:
    """Return a 68-point cloud where the subject's right side is compressed (head turned to subject's right)."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    # Right jaw close to nose, left jaw far from nose.
    points[0] = [40.0, 50.0]
    points[16] = [100.0, 50.0]
    points[30] = [50.0, 30.0]
    return points


def _profile_left_canonical_68() -> np.ndarray:
    """Return a 68-point cloud where the subject's left side is compressed (head turned to subject's left)."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[0] = [0.0, 50.0]
    points[16] = [60.0, 50.0]
    points[30] = [50.0, 30.0]
    return points


def test_visibility_from_jaw_asymmetry_returns_none_for_frontal() -> None:
    """Frontal-ish faces (asymmetry below threshold) emit no visibility entry."""
    assert _visibility_from_jaw_asymmetry(_frontal_canonical_68()) is None


def test_visibility_from_jaw_asymmetry_masks_subject_right_when_compressed() -> None:
    """Heavy yaw toward subject's right occludes the right-side jawline (indices 0..7)."""
    visibility = _visibility_from_jaw_asymmetry(_profile_right_canonical_68())
    assert visibility is not None
    assert len(visibility) == 68
    assert visibility[0:8] == [False] * 8
    assert visibility[8] is True  # chin is visible
    assert visibility[9:17] == [True] * 8


def test_visibility_from_jaw_asymmetry_masks_subject_left_when_compressed() -> None:
    """Heavy yaw toward subject's left occludes the left-side jawline (indices 9..16)."""
    visibility = _visibility_from_jaw_asymmetry(_profile_left_canonical_68())
    assert visibility is not None
    assert visibility[0:9] == [True] * 9
    assert visibility[9:17] == [False] * 8


def test_visibility_from_jaw_asymmetry_returns_none_for_degenerate_nose_at_jaw() -> None:
    """If the nose collapses onto a jaw extreme (total distance 0), no visibility is emitted."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[0] = [10.0, 0.0]
    points[16] = [10.0, 0.0]
    points[30] = [10.0, 0.0]
    assert _visibility_from_jaw_asymmetry(points) is None


def test_visibility_from_jaw_asymmetry_rejects_wrong_shape() -> None:
    """Non-68-point input returns ``None`` rather than crashing."""
    assert _visibility_from_jaw_asymmetry(np.zeros((10, 2), dtype="float32")) is None


def test_wflw_builder_emits_visibility_when_jawline_is_asymmetric(tmp_path: Path) -> None:
    """When the WFLW landmark cloud produces an asymmetric canonical jawline, visibility is recorded."""
    from lib.align.constants import MAP_2D_68, LandmarkType

    # Build a 98-point cloud that lands the canonical jaw/nose triple in a
    # clearly-profile layout (asymmetry ≈ 0.67), comfortably above the 0.50
    # threshold. We touch only the three 98-indices that map to canonical
    # jaw[0]/jaw[16]/nose[30] and leave the rest at innocuous values.
    mapping = MAP_2D_68[LandmarkType.LM_2D_98]
    points = np.zeros((98, 2), dtype="float32")  # type: ignore[var-annotated]
    points[:, 0] = np.linspace(0, 97, 98, dtype="float32")
    points[mapping[0]] = [40.0, 50.0]  # subject-right ear close to nose
    points[mapping[16]] = [100.0, 50.0]  # subject-left ear far from nose
    points[mapping[30]] = [50.0, 30.0]  # nose tip
    row = (
        " ".join(str(value) for value in points.reshape(-1))
        + " 0 0 100 100"
        + " 1 0 0 0 0 0"
        + " images/sample.jpg\n"
    )
    annotation = tmp_path / "list_98pt_rect_attr_test.txt"
    annotation.write_text(row, encoding="utf-8")

    manifest_path = build_wflw_manifest(annotation, tmp_path / "out")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    visibility = sample.get("visibility") or sample["metadata"].get("visibility")
    assert visibility is not None
    assert len(visibility) == 68
    assert sample["metadata"]["visibility_source"] == "wflw_jaw_asymmetry"
    # Subject-right side compressed → indices 0..7 masked out (8 landmarks).
    occluded_count = sum(1 for value in visibility if not value)
    assert occluded_count == 8


def test_wflw_builder_omits_visibility_for_symmetric_jawline(tmp_path: Path) -> None:
    """A 98-point cloud whose canonical-68 jaw/nose layout is symmetric emits no visibility entry."""
    from lib.align.constants import MAP_2D_68, LandmarkType

    # The 98→68 mapping picks specific 98-indices for canonical jaw[0]/jaw[16]/nose[30].
    # We build a 98-point cloud that's deliberately symmetric in canonical-68 space
    # by anchoring those three indices around a common nose X, leaving the other
    # 95 points at innocuous values that the asymmetry helper never inspects.
    mapping = MAP_2D_68[LandmarkType.LM_2D_98]
    points = np.zeros((98, 2), dtype="float32")  # type: ignore[var-annotated]
    points[:, 0] = np.linspace(0, 97, 98, dtype="float32")  # arbitrary X for non-jaw
    points[mapping[0]] = [0.0, 50.0]  # canonical jaw[0]
    points[mapping[16]] = [100.0, 50.0]  # canonical jaw[16]
    points[mapping[30]] = [50.0, 30.0]  # canonical nose tip — exactly centered
    row = (
        " ".join(str(value) for value in points.reshape(-1))
        + " 0 0 100 100"
        + " 0 0 0 0 0 0"
        + " images/sample.jpg\n"
    )
    annotation = tmp_path / "list_98pt_rect_attr_test.txt"
    annotation.write_text(row, encoding="utf-8")

    manifest_path = build_wflw_manifest(annotation, tmp_path / "out")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    assert "visibility" not in sample
    assert "visibility" not in sample["metadata"]


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


def _points_39() -> np.ndarray:
    """Return deterministic 39-point profile landmarks."""
    return np.stack(  # type: ignore[no-any-return]
        (
            np.linspace(0, 38, 39, dtype="float32"),
            np.linspace(20, 58, 39, dtype="float32"),
        ),
        axis=1,
    )


def _multipie_list_line(image_rel: str, points: np.ndarray) -> str:
    """Build a MenpoBenchmark list-file line: path, 4 bbox, 5 ref points, dense points."""
    header = [0.0, 0.0, 256.0, 256.0, *([10.0] * 10)]
    values = header + points.reshape(-1).tolist()
    return " ".join([image_rel, *(f"{value:g}" for value in values)])


def test_multipie_parse_line_splits_header_and_dense() -> None:
    """_parse_line drops the 4 bbox + 5 reference points and keeps the dense landmarks."""
    p68 = _points_68()
    semifrontal = _parse_line(_multipie_list_line("image/semifrontal/x.jpg", p68))
    assert semifrontal is not None
    image_rel, bbox, parsed = semifrontal
    assert image_rel == "image/semifrontal/x.jpg"
    assert bbox == (0.0, 0.0, 256.0, 256.0)
    assert parsed.shape == (68, 2)
    np.testing.assert_allclose(parsed, p68)

    p39 = _points_39()
    profile = _parse_line(_multipie_list_line("image/profile/y.jpg", p39))
    assert profile is not None
    _, profile_bbox, parsed_profile = profile
    assert profile_bbox == (0.0, 0.0, 256.0, 256.0)
    assert parsed_profile.shape == (39, 2)
    np.testing.assert_allclose(parsed_profile, p39)

    assert _parse_line("") is None
    assert _parse_line("image/semifrontal/x.jpg 1 2 3 4 5") is None


def _write_multipie_cache(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Create a minimal MenpoBenchmark MultiPIE layout under ``root``."""
    base = root / "MultiPIE"
    (base / "image" / "semifrontal").mkdir(parents=True)
    (base / "image" / "profile").mkdir(parents=True)
    sf_rel = "image/semifrontal/001_01_01_041_14.jpg"
    pf_rel = "image/profile/024_01_01_090_03.jpg"
    blank: np.ndarray = np.zeros((256, 256, 3), dtype="uint8")
    assert cv2.imwrite(str(base / sf_rel), blank)
    assert cv2.imwrite(str(base / pf_rel), blank)
    p68, p39 = _points_68(), _points_39()
    (base / "MultiPIE_semifrontal_train.txt").write_text(
        _multipie_list_line(sf_rel, p68) + "\n", encoding="utf-8"
    )
    (base / "MultiPIE_profile_train.txt").write_text(
        _multipie_list_line(pf_rel, p39) + "\n", encoding="utf-8"
    )
    return p68, p39


def test_multipie_builder_parses_menpo_list_files(tmp_path: Path) -> None:
    """The MultiPIE builder parses the flat MenpoBenchmark list files into samples."""
    _, p39 = _write_multipie_cache(tmp_path)

    manifest_path = build_multipie_manifest(
        tmp_path / "out", source_dir=tmp_path, no_download=True
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["dataset"] == "multipie"
    by_schema = {sample["source_schema"]: sample for sample in payload["samples"]}
    assert set(by_schema) == {"2d_68", "2d_39"}

    semifrontal = by_schema["2d_68"]
    profile = by_schema["2d_39"]
    assert "semifrontal" in semifrontal["conditions"]
    assert "profile" in profile["conditions"]
    assert profile["face_bbox"] == [0.0, 0.0, 256.0, 256.0]
    assert profile["metadata"]["face_bbox"] == [0.0, 0.0, 256.0, 256.0]
    assert profile["metadata"]["face_bbox_source"] == "multipie_menpobenchmark_detection"
    assert np.load(tmp_path / "out" / semifrontal["landmarks"]).shape == (68, 2)
    np.testing.assert_allclose(np.load(tmp_path / "out" / profile["landmarks"]), p39)


def test_multipie_builder_can_exclude_39pt_profile(tmp_path: Path) -> None:
    """``include_39pt_profile=False`` keeps only the 68-point semifrontal samples."""
    _write_multipie_cache(tmp_path)

    manifest_path = build_multipie_manifest(
        tmp_path / "out",
        source_dir=tmp_path,
        no_download=True,
        include_39pt_profile=False,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schemas = {sample["source_schema"] for sample in payload["samples"]}
    assert schemas == {"2d_68"}
