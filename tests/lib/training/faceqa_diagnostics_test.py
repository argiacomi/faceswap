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
from lib.training.data.data_set import TrainSet
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


def _train_set_stub(*, include_faceqa: bool, faceqa_index: object | None = None) -> TrainSet:
    """Return a TrainSet shell for testing FaceQA metadata helpers."""
    dataset = object.__new__(TrainSet)
    dataset._include_faceqa = include_faceqa
    dataset._faceqa_index = faceqa_index
    dataset._faceqa_cache = {}
    dataset._side = "A"
    return dataset


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


def test_faceqa_loss_diagnostics_ranks_sources_before_truncating(tmp_path: Path) -> None:
    """Worst source buckets should not be dropped by low-cardinality buckets."""
    jsonl = tmp_path / "faceqa_training_diagnostics.jsonl"
    diagnostics = FaceQALossDiagnostics(jsonl_path=str(jsonl), worst_limit=1)
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
            )
        ]
    ]

    diagnostics.update([_loss([9.0])], metadata, iteration=1)

    payload = json.loads(jsonl.read_text(encoding="utf-8").strip())
    assert len(payload["worst_sources"]) == 1
    assert payload["worst_sources"][0]["dimension"] in {"source_file", "source_id"}


def test_faceqa_loss_diagnostics_rejects_sample_count_mismatch() -> None:
    """Diagnostics should fail loudly if losses and metadata no longer align."""
    diagnostics = FaceQALossDiagnostics()
    metadata = [[FaceQASampleMetadata.missing("A", "a.png")]]

    with pytest.raises(ValueError, match="sample count mismatch"):
        diagnostics.update([_loss([1.0, 2.0])], metadata, iteration=1)


def test_faceqa_loss_diagnostics_rejects_side_count_mismatch() -> None:
    """Diagnostics should fail loudly if side counts no longer align."""
    diagnostics = FaceQALossDiagnostics()
    metadata = [[FaceQASampleMetadata.missing("A", "a.png")]]

    with pytest.raises(ValueError, match="side count mismatch"):
        diagnostics.update([_loss([1.0]), _loss([2.0])], metadata, iteration=1)


def test_faceqa_loss_diagnostics_writes_bare_jsonl_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare JSONL filenames should not call makedirs('') and fail."""
    monkeypatch.chdir(tmp_path)
    diagnostics = FaceQALossDiagnostics(jsonl_path="faceqa_training_diagnostics.jsonl")
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
            )
        ]
    ]

    diagnostics.update([_loss([1.0])], metadata, iteration=1)

    assert (tmp_path / "faceqa_training_diagnostics.jsonl").exists()


def test_faceqa_loss_diagnostics_tensorboard_keys_are_stable() -> None:
    """TensorBoard scalars should be bucket-labelled, not positional top-N slots."""
    diagnostics = FaceQALossDiagnostics()
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
            )
        ]
    ]

    logs = diagnostics.update([_loss([1.0])], metadata, iteration=1)

    assert "worst_current_loss_1" not in logs
    assert logs["bucket/A/yaw_pose/frontal/ema"] == pytest.approx(1.0)


def test_train_set_faceqa_metadata_disabled_skips_header_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled FaceQA diagnostics should not parse FaceQA headers."""

    def fail_from_png_header(*_args: object, **_kwargs: object) -> FaceQASampleMetadata:
        raise AssertionError("from_png_header should not be called")

    monkeypatch.setattr(
        FaceQASampleMetadata,
        "from_png_header",
        staticmethod(fail_from_png_header),
    )
    dataset = _train_set_stub(include_faceqa=False)

    sample = dataset._faceqa_metadata(0, "face.png", _header(faceqa={"pose": {"yaw": 0.0}}))

    assert sample.has_faceqa is False
    assert sample.side == "A"


def test_train_set_faceqa_metadata_caches_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled FaceQA diagnostics should cache parsed metadata per dataset index."""
    calls = 0

    def fake_from_png_header(
        side: str,
        filename: str,
        _header_obj: PNGHeader,
    ) -> FaceQASampleMetadata:
        nonlocal calls
        calls += 1
        return FaceQASampleMetadata(
            side=side.upper(),
            filename=filename,
            source_file=f"src_{calls}.png",
            source_id=f"src_{calls}.png:0",
            face_index=0,
            has_faceqa=True,
        )

    monkeypatch.setattr(
        FaceQASampleMetadata,
        "from_png_header",
        staticmethod(fake_from_png_header),
    )
    dataset = _train_set_stub(include_faceqa=True)

    first = dataset._faceqa_metadata(1, "face.png", _header(faceqa={"pose": {"yaw": 0.0}}))
    second = dataset._faceqa_metadata(1, "face.png", _header(faceqa={"pose": {"yaw": 45.0}}))
    third = dataset._faceqa_metadata(2, "face2.png", _header(faceqa={"pose": {"yaw": 90.0}}))

    assert first is second
    assert third is not first
    assert calls == 2


def test_train_set_faceqa_metadata_uses_fallback_index_when_header_missing() -> None:
    """FaceQA diagnostics should fall back to an enriched alignments index."""

    class _FallbackIndex:
        def lookup(
            self,
            source_file: str,
            face_index: int,
            *,
            filename: str,
        ) -> FaceQASampleMetadata | None:
            assert source_file == "frame_000001.png"
            assert face_index == 2
            assert filename == "/faces/a/face_000001.png"
            return FaceQASampleMetadata(
                side="A",
                filename=filename,
                source_file=source_file,
                source_id=f"{source_file}:{face_index}",
                face_index=face_index,
                has_faceqa=True,
                yaw_pose_bucket="frontal",
            )

    dataset = _train_set_stub(include_faceqa=True, faceqa_index=_FallbackIndex())

    sample = dataset._faceqa_metadata(0, "/faces/a/face_000001.png", _header())

    assert sample.has_faceqa is True
    assert sample.source_file == "frame_000001.png"
    assert sample.face_index == 2
    assert sample.yaw_pose_bucket == "frontal"


def test_train_set_faceqa_metadata_fallback_miss_returns_placeholder() -> None:
    """Missing header metadata and missing fallback should retain the no-metadata behavior."""

    class _FallbackIndex:
        def lookup(
            self,
            _source_file: str,
            _face_index: int,
            *,
            filename: str,
        ) -> FaceQASampleMetadata | None:
            assert filename == "/faces/a/face_000001.png"
            return None

    dataset = _train_set_stub(include_faceqa=True, faceqa_index=_FallbackIndex())

    sample = dataset._faceqa_metadata(0, "/faces/a/face_000001.png", _header())

    assert sample.has_faceqa is False
    assert sample.source_file == "frame_000001.png"
    assert sample.face_index == 2
