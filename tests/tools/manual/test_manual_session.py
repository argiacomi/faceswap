#!/usr/bin/env python3
"""Tests for GUI-neutral Manual Tool session state."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.manual.session import ManualSession


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
