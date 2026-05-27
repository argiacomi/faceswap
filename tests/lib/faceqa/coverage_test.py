#!/usr/bin/env python3
"""Tests for FaceQA coverage reports."""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np

from lib.align.faceset_qa import FaceQAFile, FaceQARecord, load, sidecar_path
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.coverage import compute_coverage, records_from_alignments
from lib.serializer import get_serializer
from tools.faceqa_coverage.faceqa_coverage import Faceqa_Coverage


def _face(width: int = 80, height: int = 90) -> FileAlignments:
    """Return a minimal alignments face."""
    landmarks = np.zeros((68, 2), dtype="float32")
    return FileAlignments(x=0, y=0, w=width, h=height, landmarks_xy=landmarks)


def test_compute_coverage_counts_unknown_fields() -> None:
    """Missing QA fields should bucket as unknown rather than crashing."""
    report = compute_coverage([FaceQARecord(frame="frame.png", face_index=0)])

    assert report.total_faces == 1
    assert report.usable_faces == 1
    assert report.bucket_counts["blur"]["unknown"] == 1
    assert report.bucket_counts["pose"]["unknown"] == 1
    assert report.bucket_counts["resolution"]["unknown"] == 1


def test_compute_coverage_uses_sidecar_risk_metadata() -> None:
    """Duplicate and identity sidecar fields should affect usable counts."""
    records = [
        FaceQARecord(
            frame="frame.png",
            face_index=0,
            blur_score=0.5,
            duplicate_keep_recommendation="prune_candidate",
            identity_quality_flag="outlier",
            resolution=[40, 40],
            average_distance=0.5,
        )
    ]
    report = compute_coverage(records, exclude_duplicates=True, exclude_outliers=True)

    assert report.usable_faces == 0
    assert report.bucket_counts["blur"]["unusable"] == 1
    assert report.bucket_counts["resolution"]["tiny"] == 1
    assert report.bucket_counts["misalignment"]["extreme"] == 1
    assert report.duplicate_ratio == 1.0
    assert report.identity_outlier_ratio == 1.0


def test_sidecar_load_ignores_unknown_fields(tmp_path) -> None:
    """Future additive sidecar fields should not break this reader."""
    path = tmp_path / "alignments_faceset_qa.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "future_root": True,
                "faces": [
                    {
                        "frame": "frame.png",
                        "face_index": 0,
                        "yaw": 10.0,
                        "future_face_field": "ignored",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    qa_file = load(str(path))

    assert isinstance(qa_file, FaceQAFile)
    assert qa_file.faces[0].yaw == 10.0


def test_faceqa_coverage_tool_writes_reports_from_alignments(tmp_path) -> None:
    """The tool should write JSON and Markdown reports from alignments alone."""
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                "frame_000001.png": AlignmentsEntry(faces=[_face()]).to_dict(),
            },
        },
    )
    output_json = tmp_path / "coverage.json"
    output_md = tmp_path / "coverage.md"
    args = Namespace(
        alignments=str(alignments),
        sidecar=None,
        output_json=str(output_json),
        output_markdown=str(output_md),
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
    )

    Faceqa_Coverage(args).process()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["alignments"] == str(alignments)
    assert payload["total_faces"] == 1
    assert "pose" in payload["coverage"]
    assert "blur" in payload["coverage"]
    assert output_md.read_text(encoding="utf-8").startswith("# FaceQA Coverage Report")


def test_records_from_alignments_prefers_spiga_pose_metadata(tmp_path) -> None:
    """FaceQA should prefer native SPIGA pose over landmark-derived pose."""
    face = _face()
    face.metadata = {
        "spiga": {
            "pose": {
                "yaw": 42.0,
                "pitch": -8.0,
                "roll": 3.0,
                "source": "spiga",
                "model": "spiga",
                "delta": {"yaw": 2.0, "pitch": -1.0, "roll": 0.5},
            }
        }
    }
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {"frame_000001.png": AlignmentsEntry(faces=[face]).to_dict()},
        },
    )

    records = records_from_alignments(alignments)

    assert records[0].yaw == 42.0
    assert records[0].pitch == -8.0
    assert records[0].roll == 3.0
    assert records[0].pose_source == "spiga"
    assert records[0].pose_model == "spiga"
    assert records[0].pose_delta_yaw == 2.0


def test_sidecar_path() -> None:
    """Default FaceQA sidecar path should be derived from alignments stem."""
    assert sidecar_path("/tmp/project/alignments.fsa") == "/tmp/project/alignments_faceset_qa.json"
