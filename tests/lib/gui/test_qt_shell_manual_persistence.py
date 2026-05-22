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

from lib.gui.qt_shell.manual_tool import ManualToolWindow
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 3) -> ManualSession:
    """Write ``count`` small PNG fixtures and return a session."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(64, 48)
        pixmap.fill(QColor("#3366ff"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def test_save_persists_edits_to_alignments_file(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A successful save writes the editable model into the alignments file on disk."""
    from lib.align import Alignments

    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 12.0, 30.0, 30.0))
    window.editable_alignments.add_face(1, (5.0, 6.0, 20.0, 22.0))

    assert window.save() is True

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


def test_save_failure_keeps_session_dirty(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If persistence raises, save() returns False and leaves edits in memory."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 12.0, 30.0, 30.0))
    assert window.editor_state.unsaved is True

    def _boom(*_args: object, **_kwargs: object) -> int:
        raise OSError("simulated disk failure")

    # Patch the handle's persist hook so we exercise the error path
    # without coupling to lib.align internals.
    monkeypatch.setattr(window._alignments_handle, "persist", _boom)

    assert window.save() is False
    assert window.editor_state.unsaved is True
    assert window.editable_alignments.face_count(0) == 1
    assert window.editable_alignments.can_undo is True


def test_close_with_unsaved_changes_can_be_cancelled(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the unsaved-changes prompt keeps the window open."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
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


def test_close_with_unsaved_changes_accepts_when_confirmed(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """Confirming the unsaved-changes prompt closes the window (default auto-Yes)."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.mark_dirty(True)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is True


def test_save_after_persist_refreshes_face_panel(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """After persistence the face panel reflects faces stored on disk."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 10.0, 30.0, 30.0))
    window.editable_alignments.add_face(0, (40.0, 10.0, 20.0, 20.0))

    assert window.save() is True

    # The face panel reads from the alignments handle; after a successful
    # save both newly-added faces should be visible.
    assert len(window.face_panel.faces) == 2
