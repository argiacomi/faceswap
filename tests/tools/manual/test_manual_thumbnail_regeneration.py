#!/usr/bin/env python3
"""Tests for Qt Manual Tool thumbnail generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools.manual.session import ManualEditableAlignments, ManualSession


def _write_frame(path: Path, value: int = 80) -> None:
    """Write a small valid source frame."""
    image = np.full((64, 64, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _session_with_face(tmp_path: Path, *, thumb_regenerate: bool = False) -> ManualSession:
    """Create a one-frame Manual session with one face entry."""
    _write_frame(tmp_path / "frame_000.png")
    session = ManualSession.create(frames=str(tmp_path), thumb_regenerate=thumb_regenerate)
    editable = ManualEditableAlignments()
    editable.add_face(0, (8.0, 8.0, 32.0, 32.0))
    session.alignments_handle().persist(editable, frame_names=[frame.name for frame in session.frame_list])
    return session


def test_missing_thumbnail_cache_is_generated(tmp_path: Path) -> None:
    """A missing cache is generated and persisted through has_thumbnails()."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()

    assert not handle.faces_for_frame_name("frame_000.png", frame_index=0)[0].has_image

    assert handle.has_thumbnails()

    generated = handle.faces_for_frame_name("frame_000.png", frame_index=0)[0]
    assert generated.has_image
    fresh = session.alignments_handle()
    assert fresh.faces_for_frame_name("frame_000.png", frame_index=0)[0].has_image


def test_forced_thumbnail_generation_refreshes_existing_cache(tmp_path: Path) -> None:
    """thumb_regenerate=True refreshes an existing cache."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    assert handle.has_thumbnails()
    before = handle.faces_for_frame_name("frame_000.png", frame_index=0)[0].thumbnail_jpeg

    _write_frame(tmp_path / "frame_000.png", value=180)
    forced = ManualSession.create(frames=str(tmp_path), thumb_regenerate=True)
    forced_handle = forced.alignments_handle()

    assert forced_handle.has_thumbnails()
    after = forced_handle.faces_for_frame_name("frame_000.png", frame_index=0)[0].thumbnail_jpeg
    assert after
    assert after != before


def test_thumbnail_generation_progress_callback(tmp_path: Path) -> None:
    """The neutral generation API exposes deterministic progress callbacks."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    progress: list[tuple[int, int, str]] = []

    generated = handle.regenerate_thumbnails(progress.append)

    assert generated == 1
    assert progress == [(1, 1, "Generated thumbnails for frame_000.png")]


def test_thumbnail_generation_skips_missing_source_frame(tmp_path: Path) -> None:
    """Missing source frames are skipped without changing other entries."""
    session = _session_with_face(tmp_path)
    handle = session.alignments_handle()
    alignments = handle.open()
    alignments.data["missing.png"] = alignments.data.pop("frame_000.png")
    progress: list[tuple[int, int, str]] = []

    generated = handle.regenerate_thumbnails(progress.append)

    assert generated == 0
    assert progress == [(1, 1, "Skipped missing.png")]
