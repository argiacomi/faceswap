#!/usr/bin/env python3
"""Native Qt Manual Tool persistence tests.

Covers the save-success path, save-failure handling and cancel-close
behavior for ``ManualToolWindow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QCloseEvent, QColor, QPixmap
from PySide6.QtWidgets import QMessageBox

from tools.manual.qt import ManualToolWindow
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 3) -> ManualSession:
    """Write ``count`` small PNG fixtures and return a session."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(64, 48)
        pixmap.fill(QColor("#3366ff"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path) -> ManualToolWindow:
    """Return a ManualToolWindow after its startup worker has drained."""
    window = ManualToolWindow(_session_with_frames(folder))
    qtbot.addWidget(window)
    # Startup completion refreshes the face panel from the alignments file.
    # Saving before that signal drains can race with Alignments.save(), so the
    # refresh reads a partially-written compressed stream.
    qtbot.waitUntil(lambda: window._startup_worker is None, timeout=5000)
    qtbot.wait(0)
    return window


def test_save_persists_edits_to_alignments_file(qtbot, tmp_path: Path) -> None:
    """A successful save writes the editable model into the alignments file on disk."""
    from lib.align import Alignments

    window = _make_window(qtbot, tmp_path)
    window.editable_alignments.add_face(0, (10.0, 12.0, 30.0, 30.0))
    window.editable_alignments.add_face(1, (5.0, 6.0, 20.0, 22.0))

    # Save is now scheduled on a worker thread (#115) — True means scheduled.
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)
    assert window.editor_state.unsaved is False

    reloaded = Alignments(str(tmp_path), "alignments.fsa")
    assert len(reloaded.data["frame_000.png"].faces) == 1
    face_a = reloaded.data["frame_000.png"].faces[0]
    assert (int(face_a.x), int(face_a.y), int(face_a.w), int(face_a.h)) == (
        10,
        12,
        30,
        30,
    )
    assert len(reloaded.data["frame_001.png"].faces) == 1


def test_save_failure_keeps_session_dirty(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If persistence raises, save() returns False and leaves edits in memory."""
    window = _make_window(qtbot, tmp_path)
    window.editable_alignments.add_face(0, (10.0, 12.0, 30.0, 30.0))
    assert window.editor_state.unsaved is True

    def _boom(*_args: object, **_kwargs: object) -> int:
        raise OSError("simulated disk failure")

    # Patch the handle's persist hook so we exercise the error path
    # without coupling to lib.align internals.
    monkeypatch.setattr(window._alignments_handle, "persist", _boom)

    # save() now returns True ("scheduled"); the failure surfaces through the
    # worker's failed signal so wait for the worker to finish before checking
    # dirty state.
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)
    assert window.editor_state.unsaved is True
    assert window.editable_alignments.face_count(0) == 1
    assert window.editable_alignments.can_undo is True


def test_close_with_unsaved_changes_can_be_cancelled(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the unsaved-changes prompt keeps the window open."""
    window = _make_window(qtbot, tmp_path)
    window.mark_dirty(True)

    # Override the conftest auto-yes patch so this test exercises the cancel
    # path: the QMessageBox returns No, which closeEvent should treat as
    # "do not close".
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *_a, **_kw: QMessageBox.StandardButton.No),
    )

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window.editor_state.unsaved is True


def test_close_with_unsaved_changes_accepts_when_confirmed(
    qtbot,
    tmp_path: Path,
) -> None:
    """Confirming the unsaved-changes prompt closes the window (default auto-Yes)."""
    window = _make_window(qtbot, tmp_path)
    window.mark_dirty(True)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is True


def test_save_after_persist_refreshes_face_panel(qtbot, tmp_path: Path) -> None:
    """After persistence the face panel reflects faces stored on disk."""
    window = _make_window(qtbot, tmp_path)
    window.editable_alignments.add_face(0, (10.0, 10.0, 30.0, 30.0))
    window.editable_alignments.add_face(0, (40.0, 10.0, 20.0, 20.0))

    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    # The face panel reads from the alignments handle; after a successful
    # save both newly-added faces should be visible.
    assert len(window.face_panel.faces) == 2


def test_revert_current_frame_restores_saved_alignments(qtbot, tmp_path: Path) -> None:
    """Revert reloads the current frame from disk rather than only undoing history."""
    from lib.align import Alignments

    window = _make_window(qtbot, tmp_path)
    window.editable_alignments.add_face(0, (10.0, 12.0, 30.0, 30.0))
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    window.editable_alignments.resize_face(0, 0, (40.0, 42.0, 20.0, 20.0))
    assert window.editor_state.unsaved is True
    window.revert_current_frame()

    assert window.editable_alignments.faces(0)[0].bbox == (10.0, 12.0, 30.0, 30.0)
    assert window.editor_state.unsaved is False
    reloaded = Alignments(str(tmp_path), "alignments.fsa")
    assert int(reloaded.data["frame_000.png"].faces[0].x) == 10


def test_save_preserves_unrelated_mask_identity_and_metadata(
    qtbot,
    tmp_path: Path,
) -> None:
    """Saving bbox/landmark edits keeps unrelated face payloads intact."""
    import zlib

    import numpy as np

    from lib.align import Alignments
    from lib.align.objects import MaskAlignmentsFile

    window = _make_window(qtbot, tmp_path)
    face_index = window.editable_alignments.add_face(0, (10.0, 12.0, 30.0, 30.0))
    window.editable_alignments._persisted_mask_blobs[(0, face_index, "custom")] = (
        MaskAlignmentsFile(
            mask=zlib.compress(np.zeros((128, 128), dtype=np.uint8).tobytes()),
            affine_matrix=np.array([[1.0, 0.0, -10.0], [0.0, 1.0, -12.0]], dtype=np.float32),
            interpolator=1,
            stored_size=128,
            stored_centering="head",
        )
    )

    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)
    window.editable_alignments.resize_face(0, 0, (10.0, 12.0, 30.0, 30.0))
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    reloaded = Alignments(str(tmp_path), "alignments.fsa")
    assert "custom" in reloaded.data["frame_000.png"].faces[0].mask


def test_save_preserves_multiple_faces_masks_identities_and_frame_metadata(
    qtbot,
    tmp_path: Path,
) -> None:
    """A save round-trip keeps unrelated .fsa payloads across faces and frames."""
    import zlib

    import numpy as np

    from lib.align import Alignments
    from lib.align.objects import MaskAlignmentsFile

    window = _make_window(qtbot, tmp_path)
    window.editable_alignments.add_face(
        0,
        (10.0, 12.0, 30.0, 30.0),
        landmarks=((11.0, 13.0), (20.0, 24.0)),
    )
    window.editable_alignments.add_face(
        0,
        (40.0, 15.0, 20.0, 24.0),
        landmarks=((41.0, 16.0), (50.0, 30.0)),
    )
    window.editable_alignments.add_face(1, (6.0, 8.0, 18.0, 20.0))
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    alignments = window._alignments_handle.open()
    frame0 = alignments.data["frame_000.png"]
    frame1 = alignments.data["frame_001.png"]
    frame0.video_meta = {"pts_time": 1234, "keyframe": 1}
    frame1.video_meta = {"pts_time": 5678, "keyframe": 0}
    for face_index, face in enumerate(frame0.faces):
        face.identity[f"id-{face_index}"] = np.full((4,), float(face_index + 1), dtype=np.float32)
        face.metadata["plugin"] = {"face": face_index, "keep": True}
        face.mask["components"] = MaskAlignmentsFile(
            mask=zlib.compress(np.full((128, 128), face_index + 1, dtype=np.uint8).tobytes()),
            affine_matrix=np.array(
                [[0.25, 0.0, float(face.x)], [0.0, 0.25, float(face.y)]],
                dtype=np.float32,
            ),
            interpolator=1,
            stored_size=128,
            stored_centering="head",
        )
        face.mask["extended"] = MaskAlignmentsFile(
            mask=zlib.compress(np.full((128, 128), 20 + face_index, dtype=np.uint8).tobytes()),
            affine_matrix=np.array(
                [[0.25, 0.0, float(face.x)], [0.0, 0.25, float(face.y)]],
                dtype=np.float32,
            ),
            interpolator=1,
            stored_size=128,
            stored_centering="head",
        )
    frame1.faces[0].identity["id-next"] = np.ones((4,), dtype=np.float32)
    frame1.faces[0].metadata["frame"] = "next"
    alignments.save()

    assert window.editable_alignments.update_landmark(0, 0, 0, 12.0, 14.0) is True
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    reloaded = Alignments(str(tmp_path), "alignments.fsa")
    saved_frame0 = reloaded.data["frame_000.png"]
    saved_frame1 = reloaded.data["frame_001.png"]
    assert saved_frame0.video_meta == {"pts_time": 1234, "keyframe": 1}
    assert saved_frame1.video_meta == {"pts_time": 5678, "keyframe": 0}
    assert len(saved_frame0.faces) == 2
    for face_index, face in enumerate(saved_frame0.faces):
        assert set(face.mask) == {"components", "extended"}
        assert f"id-{face_index}" in face.identity
        assert face.metadata["plugin"] == {"face": face_index, "keep": True}
    assert "id-next" in saved_frame1.faces[0].identity
    assert saved_frame1.faces[0].metadata["frame"] == "next"
