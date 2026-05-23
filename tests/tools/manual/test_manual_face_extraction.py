#!/usr/bin/env python3
"""Tests for the Qt-neutral Manual Tool Extract Faces service (#111)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.manual.face_extraction import (
    ExtractFacesResult,
    FaceExtractionRequest,
    extract_faces,
)
from tools.manual.session import ManualEditableAlignments, ManualSession


def _write_frame(path: Path, value: int = 80) -> None:
    """Write a small valid source frame."""
    image = np.full((96, 96, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _session_with_face(tmp_path: Path) -> ManualSession:
    """Create a one-frame Manual session with a single landmark-bearing face."""
    _write_frame(tmp_path / "frame_000.png")
    session = ManualSession.create(frames=str(tmp_path))
    editable = ManualEditableAlignments()
    # Provide some non-degenerate landmarks so AlignedFace returns a real crop.
    landmarks = tuple((30.0 + i * 0.5, 40.0 + i * 0.4) for i in range(68))
    editable.add_face(0, (10.0, 10.0, 60.0, 60.0), landmarks=landmarks)
    session.alignments_handle().persist(
        editable, frame_names=[frame.name for frame in session.frame_list]
    )
    return session


def test_extract_faces_no_alignments_file_returns_empty_result(tmp_path: Path) -> None:
    """A session without an alignments file produces a no-op result."""
    _write_frame(tmp_path / "frame_000.png")
    session = ManualSession.create(frames=str(tmp_path))
    result = extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(tmp_path / "out"),
        ),
    )
    assert isinstance(result, ExtractFacesResult)
    assert result.faces_written == 0
    assert result.frames_processed == 0


def test_extract_faces_image_folder_writes_named_pngs(tmp_path: Path) -> None:
    """Image-folder extract writes ``{stem}_{face_index}.png`` to the output folder."""
    session = _session_with_face(tmp_path)
    output = tmp_path / "extracted"
    result = extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(output),
        ),
    )
    assert result.faces_written == 1
    assert result.frames_processed == 1
    assert (output / "frame_000_0.png").is_file()


def test_extract_faces_embeds_alignments_and_source_metadata(tmp_path: Path) -> None:
    """Extracted PNGs carry alignments + source metadata in the PNG header."""
    from lib.image import png_read_meta

    session = _session_with_face(tmp_path)
    output = tmp_path / "extracted"
    extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(output),
        ),
    )
    meta = png_read_meta((output / "frame_000_0.png").read_bytes())
    assert meta is not None
    assert meta.source.source_filename == "frame_000.png"
    assert meta.source.face_index == 0
    assert meta.source.source_is_video is False
    assert meta.source.original_filename == "frame_000_0.png"
    assert meta.alignments is not None


def test_extract_faces_progress_callback_fires_per_frame(tmp_path: Path) -> None:
    """Progress callback receives one ``(done, total, message)`` tuple per frame."""
    session = _session_with_face(tmp_path)
    output = tmp_path / "extracted"
    progress: list[tuple[int, int, str]] = []
    extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(output),
        ),
        progress=lambda done, total, message: progress.append((done, total, message)),
    )
    assert len(progress) == 1
    done, total, message = progress[0]
    assert (done, total) == (1, 1)
    assert "frame_000.png" in message


def test_extract_faces_cancel_predicate_aborts_between_frames(tmp_path: Path) -> None:
    """A cancel predicate returning True halts extraction at the next frame boundary."""
    session = _session_with_face(tmp_path)
    output = tmp_path / "extracted"
    cancel_calls = [True]
    result = extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(output),
        ),
        is_cancelled=lambda: cancel_calls[0],
    )
    assert result.cancelled is True
    assert result.faces_written == 0


def test_extract_faces_skips_missing_source_frame(tmp_path: Path) -> None:
    """A missing source frame is logged and skipped, not raised."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    alignments = handle.open()
    alignments.data["missing.png"] = alignments.data.pop("frame_000.png")
    output = tmp_path / "extracted"
    result = extract_faces(
        FaceExtractionRequest(handle=handle, session=session, output_folder=str(output)),
    )
    assert result.skipped_frames == 1
    assert result.faces_written == 0
    assert any("missing.png" in err for err in result.errors)


def test_extract_faces_reader_failure_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader exception aborts cleanly and surfaces to the caller."""
    session = _session_with_face(tmp_path)
    output = tmp_path / "extracted"

    def boom(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("reader exploded")

    monkeypatch.setattr("lib.image.read_image", boom)

    with pytest.raises(RuntimeError, match="reader exploded"):
        extract_faces(
            FaceExtractionRequest(
                handle=session.alignments_handle(),
                session=session,
                output_folder=str(output),
            ),
        )


# ---------------------------------------------------------------------------
# #116 — Extract uses live editable state instead of persisted alignments
# ---------------------------------------------------------------------------


def test_extract_faces_editable_targets_replace_disk_alignments(tmp_path: Path) -> None:
    """``editable_targets`` bypass ``alignments.data`` entirely.

    Persist one face on frame_000, then pass an editable snapshot that
    references only frame_001 (a frame the disk file has never seen).
    Extract should write frame_001's face and skip frame_000 — proving the
    snapshot is the source of truth, not disk.
    """
    from tools.manual.face_extraction import EditableFaceSpec

    session = _session_with_face(tmp_path)
    _write_frame(tmp_path / "frame_001.png")
    # Re-create the session so frame_list picks up frame_001 too.
    session = ManualSession.create(frames=str(tmp_path))
    landmarks = np.asarray([(40.0 + i * 0.5, 40.0 + i * 0.5) for i in range(68)], dtype=np.float32)
    targets = (
        (
            "frame_001.png",
            (
                EditableFaceSpec(
                    face_index=0,
                    bbox=(10, 10, 60, 60),
                    landmarks_xy=landmarks,
                ),
            ),
        ),
    )
    output = tmp_path / "extracted"
    result = extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(output),
            editable_targets=targets,
        ),
    )
    assert result.faces_written == 1
    assert (output / "frame_001_0.png").is_file()
    # frame_000 — the only face on disk — must NOT be extracted.
    assert not (output / "frame_000_0.png").is_file()


def test_extract_faces_editable_targets_reflect_unsaved_delete(tmp_path: Path) -> None:
    """An empty editable snapshot yields an empty result, even with disk faces."""
    session = _session_with_face(tmp_path)
    output = tmp_path / "extracted"
    result = extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(output),
            editable_targets=(),
        ),
    )
    assert result.faces_written == 0
    assert result.frames_processed == 0


def test_extract_faces_editable_targets_carry_unsaved_bbox_into_metadata(
    tmp_path: Path,
) -> None:
    """The PNG metadata reflects the editable bbox, not the persisted one."""
    from lib.image import png_read_meta
    from tools.manual.face_extraction import EditableFaceSpec

    session = _session_with_face(tmp_path)
    # Editable bbox shifted ten pixels right/down from the persisted one.
    landmarks = np.asarray([(45.0 + i * 0.5, 50.0 + i * 0.4) for i in range(68)], dtype=np.float32)
    targets = (
        (
            "frame_000.png",
            (
                EditableFaceSpec(
                    face_index=0,
                    bbox=(20, 20, 60, 60),
                    landmarks_xy=landmarks,
                ),
            ),
        ),
    )
    output = tmp_path / "extracted"
    extract_faces(
        FaceExtractionRequest(
            handle=session.alignments_handle(),
            session=session,
            output_folder=str(output),
            editable_targets=targets,
        ),
    )
    meta = png_read_meta((output / "frame_000_0.png").read_bytes())
    assert meta is not None
    # Source frame name + face_index are still correct.
    assert meta.source.source_filename == "frame_000.png"
    assert meta.source.face_index == 0
    # The alignments embed the *editable* bbox, not the persisted (10, 10).
    assert meta.alignments.x == 20
    assert meta.alignments.y == 20
