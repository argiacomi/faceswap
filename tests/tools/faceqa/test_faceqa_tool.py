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
from tools.faceqa.cli import FaceqaArgs
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
        frames_dir=None,
        faces_dir=None,
        output_dir=None,
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
        suggest_pruning=False,
        sort_prune=False,
        contact_sheets=False,
        keep_originals=False,
        prune_aggressiveness="balanced",
        source_alignments=None,
        target_alignments=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _argument_by_dest(dest: str) -> dict:
    for argument in FaceqaArgs.get_argument_list():
        if argument.get("dest") == dest:
            return argument
    raise AssertionError(f"Argument with dest={dest!r} not found")


def test_cli_pruning_flags_match_final_contract() -> None:
    """CLI should expose move-by-default sort pruning with --keep copy mode."""
    arguments = FaceqaArgs.get_argument_list()
    all_opts = {opt for argument in arguments for opt in argument.get("opts", ())}

    assert "--sort-prune" in _argument_by_dest("sort_prune")["opts"]
    assert "--contact-sheets" in _argument_by_dest("contact_sheets")["opts"]
    keep_argument = _argument_by_dest("keep_originals")
    assert keep_argument["opts"] == ("--keep",)
    assert keep_argument["default"] is False
    assert "--no-keep" not in all_opts
    assert "--frames-dir" in _argument_by_dest("frames_dir")["opts"]
    assert "--faces-dir" in _argument_by_dest("faces_dir")["opts"]


def test_coverage_mode_runs_without_pruning(tmp_path, monkeypatch) -> None:
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(alignments, {"frame_000001.png": [_face()]})
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)
    monkeypatch.setattr(Faceqa, "_metrics_backfiller", lambda _: None)

    Faceqa(args).process()

    payload = json.loads(
        (tmp_path / "alignments_faceqa_coverage.json").read_text(encoding="utf-8")
    )
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
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)
    monkeypatch.setattr(Faceqa, "_metrics_backfiller", lambda _: None)

    Faceqa(args).process()

    payload = json.loads(
        (tmp_path / "alignments_faceqa_coverage.json").read_text(encoding="utf-8")
    )
    pruning = payload["pruning_suggestions"]
    assert pruning["aggressiveness"] == "balanced"
    assert pruning["total_faces"] == 5
    assert {"keep_count", "review_count", "prune_candidate_count"}.issubset(pruning)
    md = (tmp_path / "alignments_faceqa_coverage.md").read_text(encoding="utf-8")
    assert "## Pruning Suggestions" in md


def test_sort_prune_and_contact_sheets_move_by_default(tmp_path, monkeypatch) -> None:
    """Default sort-prune moves originals into buckets inside --faces-dir."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {f"frame_{i:06d}.png": [_face()] for i in range(1, 6)},
    )
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    for i in range(1, 6):
        _write_face_image(faces_dir / f"frame_{i:06d}_0.png")
    output_dir = tmp_path / "out"
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        faces_dir=str(faces_dir),
        suggest_pruning=True,
        sort_prune=True,
        contact_sheets=True,
        output_dir=str(output_dir),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)
    monkeypatch.setattr(Faceqa, "_metrics_backfiller", lambda _: None)

    Faceqa(args).process()

    # Default sort-prune is destructive: originals move into bucket folders
    # inside the faces directory.
    assert (faces_dir / "keep").is_dir()
    assert (faces_dir / "review").is_dir()
    assert (faces_dir / "prune_candidate").is_dir()
    assert (faces_dir / "redundancy_mapping.csv").is_file()
    assert not any(faces_dir.glob("frame_*.png"))

    # Contact sheets remain review artifacts under output_dir/pruning/.
    pruning_dir = output_dir / "pruning"
    contact_sheets = list((pruning_dir / "contact_sheets").iterdir())
    assert len(contact_sheets) >= 1

    # The old per-bucket CSV/JSONL manifests and the standalone
    # faceqa_redundancy.json file are gone — the coverage JSON now carries
    # the recommendations.
    assert not (faces_dir / "faceqa_redundancy.json").exists()
    assert not (faces_dir / "keep.csv").exists()
    assert not (faces_dir / "keep.jsonl").exists()
    assert not (faces_dir / "prune_candidates.csv").exists()
    assert not (faces_dir / "prune_candidates.jsonl").exists()

    coverage_payload = json.loads(
        (output_dir / "alignments_faceqa_coverage.json").read_text(encoding="utf-8")
    )
    assert coverage_payload["pruning_suggestions"]["total_faces"] == 5


def test_sort_prune_keep_copies_to_output_dir(tmp_path, monkeypatch) -> None:
    """--keep should preserve originals and copy sorted buckets under output_dir."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {f"frame_{i:06d}.png": [_face()] for i in range(1, 4)},
    )
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    originals = [faces_dir / f"frame_{i:06d}_0.png" for i in range(1, 4)]
    for path in originals:
        _write_face_image(path)
    output_dir = tmp_path / "out"
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        faces_dir=str(faces_dir),
        suggest_pruning=True,
        sort_prune=True,
        keep_originals=True,
        output_dir=str(output_dir),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)
    monkeypatch.setattr(Faceqa, "_metrics_backfiller", lambda _: None)

    Faceqa(args).process()

    pruning_dir = output_dir / "pruning"
    assert (pruning_dir / "keep").is_dir()
    assert (pruning_dir / "review").is_dir()
    assert (pruning_dir / "prune_candidate").is_dir()
    assert (pruning_dir / "redundancy_mapping.csv").is_file()
    assert all(path.exists() for path in originals)


def test_sort_prune_requires_faces_dir(tmp_path, monkeypatch) -> None:
    """--sort-prune without --faces-dir must error rather than silently skip."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(alignments, {"frame_000001.png": [_face()]})
    args = _base_args(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        suggest_pruning=True,
        sort_prune=True,
        output_dir=str(tmp_path / "out"),
    )
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _: lambda __, ___: None)
    monkeypatch.setattr(Faceqa, "_metrics_backfiller", lambda _: None)

    try:
        Faceqa(args).process()
    except FaceswapError as err:
        assert "--faces-dir" in str(err)
    else:
        raise AssertionError("Expected FaceswapError when --faces-dir is missing")


def test_coverage_mode_without_frames_dir_errors(tmp_path) -> None:
    """Coverage now REQUIRES --frames-dir: identity + SPIGA + image metrics all need frames."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(alignments, {"frame_000001.png": [_face()]})
    args = _base_args(
        alignments=str(alignments),
        frames_dir=None,
        output_dir=str(tmp_path),
    )

    try:
        Faceqa(args).process()
    except FaceswapError as err:
        assert "--frames-dir" in str(err)
    else:
        raise AssertionError("Expected FaceswapError when --frames-dir missing")


def test_suggest_pruning_requires_frames_dir(tmp_path, monkeypatch) -> None:
    """--suggest-pruning without --frames-dir must error rather than skip backfill."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(alignments, {"frame_000001.png": [_face()]})
    args = _base_args(
        alignments=str(alignments),
        suggest_pruning=True,
        # frames_dir intentionally None.
        output_json=str(tmp_path / "coverage.json"),
        output_markdown=str(tmp_path / "coverage.md"),
    )

    try:
        Faceqa(args).process()
    except FaceswapError as err:
        assert "--frames-dir" in str(err)
    else:
        raise AssertionError("Expected FaceswapError when --frames-dir missing")


def test_unknown_mode_raises(tmp_path) -> None:
    args = _base_args(mode="duplicates", alignments=str(tmp_path / "nope.fsa"))
    try:
        Faceqa(args).process()
    except FaceswapError as err:
        assert "Unknown FaceQA mode" in str(err)
    else:
        raise AssertionError("Expected FaceswapError for unknown mode")
