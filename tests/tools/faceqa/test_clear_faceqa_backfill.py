#!/usr/bin/env python3
"""Tests for ``tools/faceqa/clear_faceqa_backfill.py``.

Covers the per-face clear primitives, the CLI flag combinations
(``--pose`` / ``--identity`` / ``--all`` / neither), and the end-to-end
``process_file`` dry-run vs ``--write`` behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer
from tools.faceqa.clear_faceqa_backfill import (
    IDENTITY_RECORD_FIELDS,
    POSE_RECORD_FIELDS,
    backup_path,
    clear_face,
    clear_identity_metadata,
    clear_pose_metadata,
    find_fsa_files,
    main,
    process_file,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _face_with(
    *,
    pose: bool = True,
    identity: bool = True,
) -> FileAlignments:
    """Build a ``FileAlignments`` with SPIGA pose + identity metadata.

    Per-flag toggles let individual tests assert that the unrelated mode
    isn't touched (e.g. ``--pose`` must not nuke identity data).
    """
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    face = FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=landmarks)
    metadata: dict = {}
    if pose:
        metadata["spiga"] = {
            "pose": {"yaw": 5.0, "pitch": -3.0, "roll": 1.0},
            "landmarks_98": [[0.0, 0.0]],
        }
        metadata["faceqa"] = metadata.get("faceqa", {})
        metadata["faceqa"].update(
            {
                "pose": {"source": "spiga"},
                "yaw": 5.0,
                "pitch": -3.0,
                "roll": 1.0,
                "pose_source": "spiga_backfill",
                "pose_model": "spiga",
            }
        )
        metadata["pose"] = {"yaw": 5.0, "source": "spiga"}
        metadata["landmark_ensemble"] = {
            "adapter_pose": {"spiga": {"yaw": 5.0}, "synergy": {"yaw": 4.5}},
        }
    if identity:
        face.identity["arcface"] = np.ones(512, dtype="float32")
        face.identity["insightface"] = np.ones(512, dtype="float32")
        metadata["faceqa"] = metadata.get("faceqa", {})
        metadata["faceqa"].update(
            {
                "identity": {"score": 0.97, "model": "arcface"},
                "identity_score": 0.97,
                "identity_model": "arcface",
                "identity_quality_flag": "inlier",
                "identity_final_decision": "inlier",
            }
        )
    face.metadata = metadata
    return face


def _write_alignments(path, frames: dict[str, list[FileAlignments]]) -> None:
    get_serializer("compressed").save(
        str(path),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                frame: AlignmentsEntry(faces=faces).to_dict() for frame, faces in frames.items()
            },
        },
    )


# ---------------------------------------------------------------------------
# Per-face helpers
# ---------------------------------------------------------------------------


def test_clear_pose_metadata_drops_spiga_blocks() -> None:
    face = _face_with()
    removed = clear_pose_metadata(face.metadata)

    assert removed > 0
    # ``metadata['spiga']['pose']`` is removed but the SPIGA block itself
    # survives because non-pose SPIGA payloads (``landmarks_98`` etc.) are
    # unrelated to backfill and must not be touched.
    assert "pose" not in face.metadata.get("spiga", {})
    assert face.metadata["spiga"]["landmarks_98"] == [[0.0, 0.0]]
    assert "pose" not in face.metadata
    # ``faceqa`` survives because identity data is still there.
    assert "pose" not in face.metadata["faceqa"]
    for field in POSE_RECORD_FIELDS:
        assert field not in face.metadata["faceqa"]
    # Ensemble adapter pose: only the SPIGA branch is removed; synergy stays.
    adapter_pose = face.metadata["landmark_ensemble"]["adapter_pose"]
    assert "spiga" not in adapter_pose
    assert adapter_pose == {"synergy": {"yaw": 4.5}}
    # Identity untouched.
    assert "arcface" in face.identity
    assert "identity" in face.metadata["faceqa"]


def test_clear_pose_metadata_skips_non_spiga_pose_block() -> None:
    """The top-level ``metadata['pose']`` block survives when it doesn't
    look SPIGA-flavoured — we only touch SPIGA-derived pose data."""
    face = _face_with(pose=False, identity=False)
    face.metadata = {"pose": {"yaw": 1.0, "source": "synergy"}}

    removed = clear_pose_metadata(face.metadata)

    assert removed == 0
    assert face.metadata == {"pose": {"yaw": 1.0, "source": "synergy"}}


def test_clear_identity_metadata_drops_embeddings_and_decisions() -> None:
    face = _face_with()
    removed = clear_identity_metadata(face, face.metadata)

    assert removed > 0
    assert face.identity == {}
    assert "identity" not in face.metadata.get("faceqa", {})
    for field in IDENTITY_RECORD_FIELDS:
        assert field not in face.metadata.get("faceqa", {})
    # Pose untouched.
    assert face.metadata["spiga"]["pose"] == {"yaw": 5.0, "pitch": -3.0, "roll": 1.0}
    assert face.metadata["faceqa"]["pose_source"] == "spiga_backfill"


def test_clear_face_pose_only() -> None:
    face = _face_with()
    removed = clear_face(face, clear_pose=True, clear_identity=False)

    assert removed > 0
    assert face.identity != {}  # identity untouched
    assert "pose" not in face.metadata.get("spiga", {})  # SPIGA pose cleared
    assert "pose" not in face.metadata  # top-level SPIGA-flavoured pose cleared


def test_clear_face_identity_only() -> None:
    face = _face_with()
    removed = clear_face(face, clear_pose=False, clear_identity=True)

    assert removed > 0
    assert face.identity == {}
    assert face.metadata["spiga"]["pose"]["yaw"] == 5.0  # pose untouched


def test_clear_face_both_modes() -> None:
    face = _face_with()
    removed = clear_face(face, clear_pose=True, clear_identity=True)

    assert removed > 0
    assert face.identity == {}
    assert "pose" not in face.metadata.get("spiga", {})
    # ``faceqa`` envelope dict is emptied entirely now that both modes ran.
    assert "faceqa" not in face.metadata


def test_clear_face_skips_when_metadata_missing() -> None:
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    face = FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=landmarks)
    face.metadata = None  # type: ignore[assignment]

    assert clear_face(face, clear_pose=True, clear_identity=True) == 0


# ---------------------------------------------------------------------------
# Filesystem driver
# ---------------------------------------------------------------------------


def test_process_file_dry_run_does_not_write(tmp_path) -> None:
    alignments = tmp_path / "alignments.fsa"
    _write_alignments(alignments, {"frame_000001.png": [_face_with()]})
    before = alignments.read_bytes()

    changed_faces, removed_fields = process_file(
        alignments,
        write=False,
        backup=True,
        clear_pose=True,
        clear_identity=True,
    )

    assert changed_faces == 1
    assert removed_fields > 0
    assert alignments.read_bytes() == before  # dry-run preserved bytes


def test_process_file_write_creates_backup(tmp_path) -> None:
    alignments = tmp_path / "alignments.fsa"
    _write_alignments(alignments, {"frame_000001.png": [_face_with()]})

    process_file(
        alignments,
        write=True,
        backup=True,
        clear_pose=True,
        clear_identity=True,
    )

    bak = alignments.with_suffix(alignments.suffix + ".bak")
    assert bak.exists()


def test_backup_path_uniquifies_collisions(tmp_path) -> None:
    target = tmp_path / "alignments.fsa"
    target.write_text("a")
    (tmp_path / "alignments.fsa.bak").write_text("b")
    (tmp_path / "alignments.fsa.bak1").write_text("c")

    candidate = backup_path(target)

    assert candidate.name == "alignments.fsa.bak2"


def test_find_fsa_files_recursive_default(tmp_path) -> None:
    (tmp_path / "top.fsa").write_text("a")
    nested = tmp_path / "deeper"
    nested.mkdir()
    (nested / "inner.fsa").write_text("b")
    (tmp_path / "other.txt").write_text("c")

    recursive = find_fsa_files(tmp_path, recursive=True)
    top_only = find_fsa_files(tmp_path, recursive=False)

    assert {p.name for p in recursive} == {"top.fsa", "inner.fsa"}
    assert {p.name for p in top_only} == {"top.fsa"}


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_main_errors_when_no_clear_flag(tmp_path) -> None:
    """At least one of ``--pose`` / ``--identity`` / ``--all`` is required."""
    with pytest.raises(SystemExit):
        main([str(tmp_path)])


def test_main_dry_run_pose_only(tmp_path, capsys) -> None:
    alignments = tmp_path / "alignments.fsa"
    _write_alignments(alignments, {"frame_000001.png": [_face_with()]})
    before = alignments.read_bytes()

    rc = main([str(tmp_path), "--pose"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out
    assert "clearing pose" in out
    assert "identity" not in out.split("clearing ", 1)[1].split("\n", 1)[0]
    # Dry-run leaves the file untouched.
    assert alignments.read_bytes() == before


def test_main_write_all_clears_both(tmp_path) -> None:
    alignments = tmp_path / "alignments.fsa"
    _write_alignments(alignments, {"frame_000001.png": [_face_with()]})

    rc = main([str(tmp_path), "--all", "--write", "--no-backup"])

    assert rc == 0
    # Reload and verify both modes cleared.
    raw = get_serializer("compressed").load(str(alignments))
    data_entry = next(iter(raw["__data__"].values()))
    cleared_face = data_entry["faces"][0]
    assert cleared_face.get("identity", {}) == {}
    cleared_metadata = cleared_face.get("metadata") or {}
    assert "pose" not in cleared_metadata.get("spiga", {})
    assert "faceqa" not in cleared_metadata


def test_main_pose_only_preserves_identity(tmp_path) -> None:
    alignments = tmp_path / "alignments.fsa"
    _write_alignments(alignments, {"frame_000001.png": [_face_with()]})

    rc = main([str(tmp_path), "--pose", "--write", "--no-backup"])

    assert rc == 0
    raw = get_serializer("compressed").load(str(alignments))
    cleared_face = next(iter(raw["__data__"].values()))["faces"][0]
    cleared_metadata = cleared_face.get("metadata") or {}
    assert "pose" not in cleared_metadata.get("spiga", {})
    # Identity embeddings intact.
    identity = cleared_face.get("identity", {})
    assert "arcface" in identity
    assert "insightface" in identity
