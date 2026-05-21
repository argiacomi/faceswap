#!/usr/bin/env python3
"""Tests for GUI-neutral Manual Tool session state."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from tools.manual.session import (
    ManualAlignmentsHandle,
    ManualEditorState,
    ManualSession,
    ManualVideoMetadata,
)


def test_manual_session_discovers_image_frames(tmp_path: Path) -> None:
    """Image-folder sessions expose sorted GUI-neutral frame metadata."""
    (tmp_path / "b.jpg").write_bytes(b"jpg")
    (tmp_path / "a.png").write_bytes(b"png")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    session = ManualSession.create(frames=str(tmp_path))

    assert session.has_images
    assert session.frame_count == 2
    assert [frame.name for frame in session.frame_list] == ["a.png", "b.jpg"]


def test_manual_session_rejects_missing_input(tmp_path: Path) -> None:
    """A clear validation error is raised before any GUI is constructed."""
    with pytest.raises(ValueError, match="does not exist"):
        ManualSession.create(frames=str(tmp_path / "missing"))


def test_manual_session_rejects_non_video_file(tmp_path: Path) -> None:
    """A non-video file is rejected before any GUI is constructed."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError, match="not a supported video file"):
        ManualSession.create(frames=str(file_path))


def test_manual_session_from_cli_values_uses_switch_keys(tmp_path: Path) -> None:
    """Qt command-panel values can initialize a session without tkinter."""
    frame = tmp_path / "frame_000001.png"
    alignments = tmp_path / "alignments.fsa"
    frame.write_bytes(b"png")
    alignments.write_text("{}", encoding="utf-8")

    session = ManualSession.from_cli_values(
        {
            "-f": str(tmp_path),
            "-a": str(alignments),
            "-t": True,
            "-s": True,
        }
    )

    assert session.frames == str(tmp_path.resolve())
    assert session.alignments_path == str(alignments.resolve())
    assert session.thumb_regenerate
    assert session.single_process


def test_manual_session_from_namespace(tmp_path: Path) -> None:
    """The Tk entry point can build a session from a Namespace."""
    (tmp_path / "frame.png").write_bytes(b"png")
    namespace = Namespace(
        frames=str(tmp_path),
        alignments_path="",
        thumb_regenerate=False,
        single_process=True,
    )
    session = ManualSession.from_namespace(namespace)
    assert session.frames == str(tmp_path.resolve())
    assert session.alignments_path is None
    assert not session.thumb_regenerate
    assert session.single_process


def test_alignments_handle_image_folder_defaults(tmp_path: Path) -> None:
    """Alignments default to alignments.fsa next to image input folder."""
    (tmp_path / "frame.png").write_bytes(b"png")
    session = ManualSession.create(frames=str(tmp_path))
    handle = session.alignments_handle()

    assert isinstance(handle, ManualAlignmentsHandle)
    assert handle.folder == str(tmp_path.resolve())
    assert handle.filename == "alignments.fsa"
    assert handle.path == str(tmp_path.resolve() / "alignments.fsa")
    assert handle.exists is False
    assert handle.has_thumbnails() is False
    assert handle.video_metadata() is None


def test_alignments_handle_video_default_name(tmp_path: Path) -> None:
    """Alignments default to <video>_alignments.fsa for video input."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"vid")
    session = ManualSession.create(frames=str(video))
    handle = session.alignments_handle()

    assert handle.filename == "clip_alignments.fsa"
    assert handle.folder == str(tmp_path.resolve())
    assert session.is_video_input


def test_alignments_handle_honors_explicit_path(tmp_path: Path) -> None:
    """An explicit alignments path is preserved through the session."""
    (tmp_path / "frame.png").write_bytes(b"png")
    alignments = tmp_path / "custom" / "alignments.fsa"
    alignments.parent.mkdir()
    alignments.write_text("{}", encoding="utf-8")
    session = ManualSession.create(
        frames=str(tmp_path),
        alignments_path=str(alignments),
    )
    handle = session.alignments_handle()

    assert handle.path == str(alignments.resolve())
    assert handle.exists is True


def test_video_metadata_is_neutral_dataclass() -> None:
    """The video metadata view is GUI-neutral and reports validity."""
    meta = ManualVideoMetadata(pts_time=(0, 1, 2), keyframes=(0, 2))
    assert meta.frame_count == 3
    assert meta.is_valid
    assert ManualVideoMetadata(pts_time=(), keyframes=()).is_valid is False


def test_editor_state_subscribe_and_set() -> None:
    """Editor state emits to subscribers only when values change."""
    state = ManualEditorState()
    seen: list[int] = []
    unsubscribe = state.subscribe("frame_index", seen.append)

    state.set("frame_index", 0)  # No change.
    state.set("frame_index", 5)
    state.set("frame_index", 5)  # Repeat suppressed.
    state.set("frame_index", 7)
    unsubscribe()
    state.set("frame_index", 9)  # Listener no longer fires.

    assert seen == [5, 7]
    assert state.frame_index == 9


def test_editor_state_rejects_unknown_field() -> None:
    """Editor state guards typo-prone field names."""
    state = ManualEditorState()
    with pytest.raises(ValueError):
        state.set("frame_idx", 1)
    with pytest.raises(ValueError):
        state.subscribe("frame_idx", lambda _value: None)


def test_session_needs_thumbnail_regeneration_when_missing(tmp_path: Path) -> None:
    """Sessions report thumbnail-regeneration need when no cache exists."""
    (tmp_path / "frame.png").write_bytes(b"png")
    session = ManualSession.create(frames=str(tmp_path))
    assert session.needs_thumbnail_regeneration() is True

    session_force = ManualSession.create(frames=str(tmp_path), thumb_regenerate=True)
    assert session_force.needs_thumbnail_regeneration() is True


def test_session_create_editor_state_returns_fresh_state(tmp_path: Path) -> None:
    """Each call returns an independent editor state instance."""
    (tmp_path / "frame.png").write_bytes(b"png")
    session = ManualSession.create(frames=str(tmp_path))
    first = session.create_editor_state()
    second = session.create_editor_state()

    assert first is not second
    first.set("frame_index", 4)
    assert second.frame_index == 0
