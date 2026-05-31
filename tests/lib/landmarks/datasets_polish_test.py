#!/usr/bin/env python3
"""Tests for landmark dataset audit polish and overlay generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.polish import polish_landmark_dataset_artifacts
from tools.landmarks import build_quality_dataset


def _points_68() -> np.ndarray:
    """Return deterministic 68-point landmarks."""
    return np.stack(  # type: ignore[no-any-return]
        (
            np.linspace(10, 74, 68, dtype="float32"),
            np.linspace(12, 76, 68, dtype="float32"),
        ),
        axis=1,
    )


def _points_98() -> np.ndarray:
    """Return deterministic 98-point WFLW landmarks."""
    return np.stack(  # type: ignore[no-any-return]
        (
            np.linspace(20, 216, 98, dtype="float32"),
            np.linspace(30, 226, 98, dtype="float32"),
        ),
        axis=1,
    )


def _write_png(path: Path) -> None:
    """Write a tiny valid PNG."""
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((96, 96, 3), dtype="uint8")  # type: ignore[var-annotated]
    image[..., 1] = 128
    assert cv2.imwrite(str(path), image)


def test_cli_writes_indexed_region_overlays_and_donor_audit_fields(tmp_path: Path) -> None:
    """CLI polish writes indexed/region overlays and AutoMask-parity audit keys."""
    source = tmp_path / "source"
    source.mkdir()
    _write_png(source / "sample.png")
    cofw_json = source / "cofw.json"
    cofw_json.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample",
                        "image": "sample.png",
                        "landmarks": _points_68().tolist(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rc = build_quality_dataset.main(
        [
            "--dataset",
            "cofw",
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--write-overlays",
        ]
    )

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "out" / "dataset_audit.json").read_text(encoding="utf-8"))
    sample = manifest["samples"][0]
    overlays = sample["metadata"]["overlays"]

    assert rc == 0
    assert (tmp_path / "out" / overlays["indexed"]).is_file()
    assert (tmp_path / "out" / overlays["regions"]).is_file()
    assert audit["audit_schema_family"] == "automask_dataset_audit_v1"
    assert audit["count_per_scenario_group"] == {"default": 1}
    assert audit["count_per_suite"] == {"landmark_quality": 1}
    assert audit["count_per_condition"] == {"default": 1}
    assert audit["unique_source_count"] == 1
    assert audit["shortfall_groups"] == []
    assert audit["rejected_candidates_due_to_exclusivity"] == []
    assert audit["landmark_quality_entry_count"] == 1
    assert "feature_stats" in audit
    assert "landmark_x_span" in audit["feature_stats"]
    assert "selected_source_ids_per_group" in audit


def test_cli_rewrites_rich_source_notes(tmp_path: Path) -> None:
    """Source notes include dataset-specific provenance sections after polish."""
    source = tmp_path / "source"
    source.mkdir()
    _write_png(source / "sample.png")
    cofw_json = source / "cofw.json"
    cofw_json.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample",
                        "image": "sample.png",
                        "landmarks": _points_68().tolist(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    build_quality_dataset.main(
        [
            "--dataset",
            "cofw",
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    notes = (tmp_path / "out" / "SOURCE_NOTES.md").read_text(encoding="utf-8")
    assert "Landmark quality dataset source notes" in notes
    assert "AutoMask donor audit shape" in notes
    assert "### COFW" in notes
    assert "COFW color images" in notes


def test_polish_normalizes_wflw_annotation_bbox_for_prediction_roi(tmp_path: Path) -> None:
    """WFLW annotation bbox is preserved separately from validation face_bbox."""
    output = tmp_path / "out"
    output.mkdir()
    landmark_path = output / "landmarks.npy"
    points = _points_98()
    np.save(str(landmark_path), points)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "wflw",
                "samples": [
                    {
                        "sample_id": "sample",
                        "dataset": "wflw",
                        "condition": "default",
                        "conditions": ["default"],
                        "image": "sample.jpg",
                        "landmarks": "landmarks.npy",
                        "source_schema": "2d_98",
                        "metadata": {"bbox": [1000, 1000, 2000, 2000]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output / "dataset_audit.json").write_text("{}\n", encoding="utf-8")

    polish_landmark_dataset_artifacts(output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["samples"][0]["metadata"]
    assert "bbox" not in metadata
    assert metadata["wflw_annotation_bbox"] == [1000, 1000, 2000, 2000]
    assert metadata["face_bbox_source"] == "wflw_98_landmark_extrema"
    np.testing.assert_allclose(
        metadata["face_bbox"],
        [
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ],
    )
