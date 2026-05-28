#!/usr/bin/env python3
"""Integration smoke tests for the unified tools/faceqa Faceqa dispatcher."""

from __future__ import annotations

import json
from argparse import Namespace

import cv2
import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer
from tools.faceqa.faceqa import Faceqa


def _face_with_embedding(
    *,
    embedding: np.ndarray,
    width: int = 256,
    height: int = 256,
) -> FileAlignments:
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    face = FileAlignments(x=0, y=0, w=width, h=height, landmarks_xy=landmarks)
    face.identity = {"insightface": embedding.astype(np.float32)}
    return face


def _save_alignments(path, frames: dict[str, list[FileAlignments]]) -> None:
    get_serializer("compressed").save(
        str(path),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                frame: AlignmentsEntry(faces=faces).to_dict() for frame, faces in frames.items()
            },
        },
    )


def _write_face_image(path) -> None:
    cv2.imwrite(str(path), np.full((96, 96, 3), 80, dtype=np.uint8))


def _duplicates_args(
    *,
    alignments,
    faces_dir,
    output_dir,
    duplicates_report=None,
) -> Namespace:
    return Namespace(
        mode="duplicates",
        alignments=str(alignments),
        faces_dir=str(faces_dir),
        output_dir=str(output_dir),
        sidecar=None,
        frames_dir=None,
        output_json=None,
        output_markdown=None,
        output_prefix="source_target_compatibility",
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
        identity_model=None,
        similarity_threshold=0.85,
        temporal_window=-1,
        duplicates_report=duplicates_report,
        symlink=False,
        contact_sheet_tile_size=128,
        contact_sheet_cols=2,
        contact_sheet_format="png",
        source_alignments=None,
        target_alignments=None,
        source_sidecar=None,
        target_sidecar=None,
        source_frames_dir=None,
        target_frames_dir=None,
    )


def test_faceqa_duplicates_mode_produces_full_output_tree(tmp_path) -> None:
    """The duplicates mode should write JSON, manifests, sorted folders and contact sheets."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [
                _face_with_embedding(embedding=embedding, width=512, height=512),
            ],
            "frame_000002.png": [
                _face_with_embedding(embedding=embedding + 0.01, width=64, height=64),
            ],
            "frame_000050.png": [
                _face_with_embedding(embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")),
            ],
        },
    )
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    for name in ("frame_000001_0.png", "frame_000002_0.png", "frame_000050_0.png"):
        _write_face_image(faces_dir / name)
    output_dir = tmp_path / "out"

    Faceqa(
        _duplicates_args(alignments=alignments, faces_dir=faces_dir, output_dir=output_dir)
    ).process()

    json_report = output_dir / "faceqa_duplicates.json"
    assert json_report.is_file()
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["total_faces"] == 3
    assert payload["records"], "expected per-face records"

    assert (output_dir / "keep").is_dir()
    assert (output_dir / "review").is_dir()
    assert (output_dir / "prune_candidate").is_dir()
    assert (output_dir / "keep.csv").is_file()
    assert (output_dir / "keep.jsonl").is_file()
    assert (output_dir / "prune_candidates.csv").is_file()
    assert (output_dir / "prune_candidates.jsonl").is_file()
    contact_sheets = list((output_dir / "contact_sheets").iterdir())
    assert contact_sheets, "expected at least one contact sheet for the multi-face cluster"


def test_faceqa_duplicates_mode_reuses_existing_report(tmp_path) -> None:
    """Supplying --duplicates-report skips clustering and consumes the report directly."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face_with_embedding(embedding=embedding)],
            "frame_000002.png": [_face_with_embedding(embedding=embedding + 0.01)],
        },
    )
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    _write_face_image(faces_dir / "frame_000001_0.png")
    _write_face_image(faces_dir / "frame_000002_0.png")

    first_output = tmp_path / "first"
    Faceqa(
        _duplicates_args(alignments=alignments, faces_dir=faces_dir, output_dir=first_output)
    ).process()
    report_path = first_output / "faceqa_duplicates.json"
    assert report_path.is_file()

    # Now re-run with the existing report as input → no recompute.
    second_output = tmp_path / "second"
    Faceqa(
        _duplicates_args(
            alignments=alignments,
            faces_dir=faces_dir,
            output_dir=second_output,
            duplicates_report=str(report_path),
        )
    ).process()
    # The second run does NOT need to write a new JSON, but the sorted folders
    # and manifests should match what the first run produced.
    assert (second_output / "keep").is_dir()
    assert (second_output / "prune_candidate").is_dir()
    assert (second_output / "keep.csv").is_file()


def test_faceqa_dispatch_rejects_unknown_mode(tmp_path) -> None:
    """An unknown --mode should raise FaceswapError, not silently no-op."""
    from lib.utils import FaceswapError

    args = _duplicates_args(
        alignments=tmp_path / "missing.fsa",
        faces_dir=tmp_path,
        output_dir=tmp_path,
    )
    args.mode = "nope"
    try:
        Faceqa(args).process()
    except FaceswapError as err:
        assert "Unknown FaceQA mode" in str(err)
    else:
        raise AssertionError("Expected FaceswapError for unknown mode")
