#!/usr/bin/env python3
"""Tests for FaceQA training loss diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lib.align.objects import PNGAlignments, PNGHeader, PNGSource
from lib.training.data.collate import BatchMeta
from lib.training.faceqa_diagnostics import FaceQALossDiagnostics, FaceQASampleMetadata
from lib.training.loss import BatchLoss


def _header(*, faceqa: dict[str, object] | None = None) -> PNGHeader:
    metadata = {} if faceqa is None else {"faceqa": faceqa}
    return PNGHeader(
        alignments=PNGAlignments(
            x=0,
            y=0,
            w=256,
            h=128,
            landmarks_xy=np.zeros((68, 2), dtype=np.float32),
            mask={},
            metadata=metadata,
        ),
        source=PNGSource(
            alignments_version=2.4,
            original_filename="face_000001.png",
            face_index=2,
            source_filename="frame_000001.png",
            source_is_video=False,
            source_frame_dims=(720, 1280),
        ),
    )


def _loss(values: list[float]) -> BatchLoss:
    tensor = torch.tensor(values, dtype=torch.float32)
    return BatchLoss(unweighted=[{"mae": tensor}], weighted=[{"mae": tensor}])


def test_faceqa_sample_metadata_extracts_embedded_buckets() -> None:
    """PNG-embedded FaceQA metadata should become compact training metadata."""
    sample = FaceQASampleMetadata.from_png_header(
        "a",
        "/faces/a/face_000001.png",
        _header(
            faceqa={
                "pose": {"yaw": 42.0, "pitch": -20.0, "roll": 0.0, "source": "spiga"},
                "image_metrics": {
                    "blur_score": 0.2,
                    "mean_luminance": 220.0,
                    "contrast": 20.0,
                    "left_right_ratio": 1.0,
                    "top_bottom_ratio": 1.0,
                    "color_warmth": 0.0,
                    "provenance": "frame_aligned_crop",
                },
                "identity": {"quality_flag": "outlier"},
                "duplicate_cluster_id": "cluster-1",
                "expression_bucket": "smile",
                "mask_qa_ref": "components",
            }
        ),
    )

    assert sample.has_faceqa is True
    assert sample.side == "A"
    assert sample.source_file == "frame_000001.png"
    assert sample.source_id == "frame_000001.png:2"
    assert sample.yaw_pose_bucket == "right_profile"
    assert sample.pitch_bucket == "down"
    assert sample.expression_bucket == "smile"
    assert sample.blur_bucket == "unusable"
    assert sample.resolution_bucket == "ok"
    assert sample.mask_qa_bucket == "present"
    assert sample.duplicate_bucket == "duplicate"
    assert sample.identity_outlier_bucket == "outlier"


def test_faceqa_sample_metadata_missing_faceqa_is_placeholder() -> None:
    """Training samples without FaceQA metadata should not activate diagnostics."""
    sample = FaceQASampleMetadata.from_png_header("b", "/faces/b/face.png", _header())

    assert sample.has_faceqa is False
    assert sample.side == "B"
    assert sample.source_file == "frame_000001.png"
    assert sample.yaw_pose_bucket == "unknown"


def test_batch_meta_keeps_faceqa_out_of_tensor_dict() -> None:
    """Distributed training should only scatter tensor metadata."""
    sample = FaceQASampleMetadata.missing("a", "face.png")
    meta = BatchMeta(mask_face=[torch.ones((2, 2, 1, 4, 4))], faceqa=[[sample], [sample]])

    side_meta = meta[1]

    assert side_meta.faceqa == [[sample]]
    assert "faceqa" not in meta.tensor_dict()
    assert meta.to("cpu").faceqa == [[sample], [sample]]


def test_faceqa_loss_diagnostics_aggregates_and_writes_jsonl(tmp_path: Path) -> None:
    """Per-sample losses should update rolling bucket summaries and JSONL detail."""
    jsonl = tmp_path / "faceqa_training_diagnostics.jsonl"
    diagnostics = FaceQALossDiagnostics(jsonl_path=str(jsonl))
    metadata = [
        [
            FaceQASampleMetadata(
                side="A",
                filename="a1.png",
                source_file="src1.png",
                source_id="src1.png:0",
                face_index=0,
                has_faceqa=True,
                yaw_pose_bucket="frontal",
            ),
            FaceQASampleMetadata(
                side="A",
                filename="a2.png",
                source_file="src2.png",
                source_id="src2.png:0",
                face_index=0,
                has_faceqa=True,
                yaw_pose_bucket="right_profile",
            ),
        ],
        [
            FaceQASampleMetadata(
                side="B",
                filename="b1.png",
                source_file="src3.png",
                source_id="src3.png:0",
                face_index=0,
                has_faceqa=True,
                yaw_pose_bucket="frontal",
            ),
            FaceQASampleMetadata.missing("B", "b2.png"),
        ],
    ]

    logs = diagnostics.update([_loss([1.0, 3.0]), _loss([2.0, 4.0])], metadata, iteration=10)

    assert logs["metadata_coverage"] == pytest.approx(0.75)
    assert "ab_reconstruction_imbalance" in logs
    payload = json.loads(jsonl.read_text(encoding="utf-8").strip())
    assert payload["metadata_samples"] == 3
    assert payload["total_samples"] == 4
    assert any(
        item["dimension"] == "yaw_pose" and item["bucket"] == "right_profile"
        for item in payload["worst_current"]
    )
    assert any(item["dimension"] == "source_id" for item in payload["worst_sources"])


def test_faceqa_loss_diagnostics_no_metadata_fallback(tmp_path: Path) -> None:
    """No FaceQA metadata should produce no logs or JSON side effects."""
    jsonl = tmp_path / "faceqa_training_diagnostics.jsonl"
    diagnostics = FaceQALossDiagnostics(jsonl_path=str(jsonl))
    metadata = [[FaceQASampleMetadata.missing("A", "a.png")]]

    assert diagnostics.update([_loss([1.0])], metadata, iteration=1) == {}
    assert not jsonl.exists()
