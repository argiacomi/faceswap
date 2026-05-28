#!/usr/bin/env python3
"""Integration smoke tests for the unified tools/faceqa Faceqa dispatcher."""

from __future__ import annotations

import json
from argparse import Namespace

import cv2
import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer
from lib.utils import FaceswapError
from tools.faceqa.faceqa import Faceqa


def _face() -> FileAlignments:
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    return FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=landmarks)


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


def _base_args(**overrides) -> Namespace:
    defaults = dict(
        mode="coverage",
        alignments=None,
        sidecar=None,
        frames_dir=None,
        faces_dir=None,
        output_json=None,
        output_markdown=None,
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
        suggest_pruning=False,
        prune_aggressiveness="balanced",
        prune_output_dir=None,
        source_alignments=None,
        target_alignments=None,
        source_sidecar=None,
        target_sidecar=None,
        compatibility_output_dir=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_coverage_mode_runs_without_pruning(tmp_path, monkeypatch) -> None:
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(alignments, {"frame_000001.png": [_face()]})
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        output_json=str(tmp_path / "coverage.json"),
        output_markdown=str(tmp_path / "coverage.md"),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)

    Faceqa(args).process()

    payload = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert payload["total_faces"] == 1
    # Without --suggest-pruning the block stays empty.
    assert payload["pruning_suggestions"] == {}


def test_coverage_mode_with_suggest_pruning_emits_redundancy(tmp_path, monkeypatch) -> None:
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {f"frame_{i:06d}.png": [_face()] for i in range(1, 6)},
    )
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        suggest_pruning=True,
        prune_aggressiveness="balanced",
        output_json=str(tmp_path / "coverage.json"),
        output_markdown=str(tmp_path / "coverage.md"),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)

    Faceqa(args).process()

    payload = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    pruning = payload["pruning_suggestions"]
    assert pruning["aggressiveness"] == "balanced"
    assert pruning["total_faces"] == 5
    assert {"keep_count", "review_count", "prune_candidate_count"}.issubset(pruning)
    md = (tmp_path / "coverage.md").read_text(encoding="utf-8")
    assert "## Pruning Suggestions" in md


def test_prune_output_dir_emits_sorted_folders_and_contact_sheets(tmp_path, monkeypatch) -> None:
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {f"frame_{i:06d}.png": [_face()] for i in range(1, 6)},
    )
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    for i in range(1, 6):
        _write_face_image(faces_dir / f"frame_{i:06d}_0.png")
    prune_dir = tmp_path / "prune"
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        faces_dir=str(faces_dir),
        suggest_pruning=True,
        prune_output_dir=str(prune_dir),
        output_json=str(tmp_path / "coverage.json"),
        output_markdown=str(tmp_path / "coverage.md"),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)

    Faceqa(args).process()

    assert (prune_dir / "faceqa_redundancy.json").is_file()
    assert (prune_dir / "keep.csv").is_file()
    assert (prune_dir / "keep.jsonl").is_file()
    assert (prune_dir / "prune_candidates.csv").is_file()
    assert (prune_dir / "prune_candidates.jsonl").is_file()
    assert (prune_dir / "keep").is_dir()
    assert (prune_dir / "review").is_dir()
    assert (prune_dir / "prune_candidate").is_dir()
    assert (prune_dir / "redundancy_mapping.csv").is_file()
    contact_sheets = list((prune_dir / "contact_sheets").iterdir())
    # Multiple identical faces should cluster into one multi-face cluster → one sheet.
    assert len(contact_sheets) >= 1


def test_prune_output_dir_requires_faces_dir(tmp_path, monkeypatch) -> None:
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(alignments, {"frame_000001.png": [_face()]})
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        suggest_pruning=True,
        prune_output_dir=str(tmp_path / "prune"),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)

    try:
        Faceqa(args).process()
    except FaceswapError as err:
        assert "--faces-dir" in str(err)
    else:
        raise AssertionError("Expected FaceswapError when --faces-dir is missing")


def test_unknown_mode_raises(tmp_path) -> None:
    args = _base_args(mode="duplicates", alignments=str(tmp_path / "nope.fsa"))
    try:
        Faceqa(args).process()
    except FaceswapError as err:
        assert "Unknown FaceQA mode" in str(err)
    else:
        raise AssertionError("Expected FaceswapError for unknown mode")
