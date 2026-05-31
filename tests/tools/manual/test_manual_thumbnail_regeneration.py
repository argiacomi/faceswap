#!/usr/bin/env python3
"""Tests for the Qt-neutral Manual Tool thumbnail-regeneration service."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.manual.session import ManualEditableAlignments, ManualSession


def _write_frame(path: Path, value: int = 80) -> None:
    """Write a small valid source frame."""
    image = np.full((64, 64, 3), value, dtype=np.uint8)  # type: ignore[var-annotated]
    assert cv2.imwrite(str(path), image)


def _session_with_face(tmp_path: Path, *, thumb_regenerate: bool = False) -> ManualSession:
    """Create a one-frame Manual session with one persisted face."""
    _write_frame(tmp_path / "frame_000.png")
    session = ManualSession.create(frames=str(tmp_path), thumb_regenerate=thumb_regenerate)
    editable = ManualEditableAlignments()
    editable.add_face(0, (8.0, 8.0, 32.0, 32.0))
    session.alignments_handle().persist(
        editable, frame_names=[frame.name for frame in session.frame_list]
    )
    return session


def test_regenerate_thumbnails_populates_missing_cache(tmp_path: Path) -> None:
    """``regenerate_thumbnails`` writes thumbs and ``has_thumbnails`` reports True."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()

    # Pre-condition: persistence stripped thumb payloads so the cache is empty.
    assert not handle.faces_for_frame_name("frame_000.png", frame_index=0)[0].has_image
    assert not handle.has_thumbnails()

    generated = handle.regenerate_thumbnails(session)
    assert generated == 1
    assert handle.has_thumbnails()

    fresh = session.alignments_handle()
    assert fresh.faces_for_frame_name("frame_000.png", frame_index=0)[0].has_image


def test_regenerate_thumbnails_refreshes_existing_payload(tmp_path: Path) -> None:
    """Calling ``regenerate_thumbnails`` again refreshes existing thumbs."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    handle.regenerate_thumbnails(session)
    before = handle.faces_for_frame_name("frame_000.png", frame_index=0)[0].thumbnail_jpeg

    # Modify the source frame so the regenerated thumbnail must differ.
    _write_frame(tmp_path / "frame_000.png", value=180)
    forced = ManualSession.create(frames=str(tmp_path), thumb_regenerate=True)
    forced_handle = forced.alignments_handle()
    forced_handle.regenerate_thumbnails(forced)

    after = forced_handle.faces_for_frame_name("frame_000.png", frame_index=0)[0].thumbnail_jpeg
    assert after
    assert after != before


def test_regenerate_thumbnails_progress_callback(tmp_path: Path) -> None:
    """Progress callback fires once per frame with (done, total, message)."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    progress: list[tuple[int, int, str]] = []

    generated = handle.regenerate_thumbnails(
        session,
        progress=lambda done, total, message: progress.append((done, total, message)),
    )

    assert generated == 1
    assert progress == [(1, 1, "Generated thumbnails for frame_000.png")]


def test_regenerate_thumbnails_restores_payload_when_reader_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader failure rolls back the in-memory thumb payload."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    handle.regenerate_thumbnails(session)
    alignments = handle.open()
    original = alignments.data["frame_000.png"].faces[0].thumb
    assert original is not None

    def _raise_reader(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("reader unavailable")

    monkeypatch.setattr("lib.image.read_image", _raise_reader)

    with pytest.raises(RuntimeError, match="reader unavailable"):
        handle.regenerate_thumbnails(session)

    assert alignments.data["frame_000.png"].faces[0].thumb is original


def test_regenerate_thumbnails_skips_missing_source_frame(tmp_path: Path) -> None:
    """Frames whose source image is missing are reported and skipped."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    alignments = handle.open()
    alignments.data["missing.png"] = alignments.data.pop("frame_000.png")
    progress: list[tuple[int, int, str]] = []

    generated = handle.regenerate_thumbnails(
        session,
        progress=lambda done, total, message: progress.append((done, total, message)),
    )

    assert generated == 0
    assert progress == [(1, 1, "Skipped missing.png")]


def test_has_thumbnails_does_not_regenerate(tmp_path: Path) -> None:
    """``has_thumbnails`` is a pure read — it must not regenerate implicitly."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()

    # No payload yet; calling has_thumbnails should leave it that way.
    assert handle.has_thumbnails() is False
    entries = handle.faces_for_frame_name("frame_000.png", frame_index=0)
    assert entries and not entries[0].has_image
