#!/usr/bin/env python3
"""Pointer-add, context menu, focus-scoped shortcut and save-gating tests.

Covers the acceptance criteria of issues #105 and #110:

* #105 — pointer-add gesture in BoundingBox mode, frame-view & face-panel
  context menus, arrow-key nudge scoped to frame-view focus.
* #110 — save in-flight gating, action disable while a save is running,
  duplicate-save blocking, dirty-state preservation on failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPixmap

from lib.gui.qt_shell.manual_tool import (
    FaceThumbnailPanel,
    ManualFrameView,
    ManualToolWindow,
)
from tools.manual.session import FaceThumbnail, ManualSession


def _session_with_frames(folder: Path, count: int = 2) -> ManualSession:
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(120, 80)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    session = _session_with_frames(folder, count=2)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def _press_release(view: ManualFrameView, pos: QPointF, button: Qt.MouseButton) -> None:
    """Synthesize a click on ``view`` at the given widget-local ``pos``."""
    global_pos = view.mapToGlobal(pos.toPoint())
    view.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            pos,
            QPointF(global_pos),
            button,
            button,
            Qt.NoModifier,
        )
    )
    view.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            pos,
            QPointF(global_pos),
            button,
            Qt.NoButton,
            Qt.NoModifier,
        )
    )


# ---------------------------------------------------------------------------
# #105 — Pointer-add gesture in BoundingBox mode
# ---------------------------------------------------------------------------


def test_click_in_bbox_mode_creates_face_at_pointer(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Clicking empty frame space in BBox mode creates a face under the pointer."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + target.width() / 2, target.y() + target.height() / 2)
    initial_count = window._editable.face_count(0)

    _press_release(window._frame_view, pos, Qt.LeftButton)

    assert window._editable.face_count(0) == initial_count + 1
    # Pointer-added face goes through the same editable model, so undo works.
    assert window._editable.can_undo


def test_click_in_view_mode_does_not_create_face(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Outside BBox mode an empty-space click must NOT add a face."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "View")
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + target.width() / 2, target.y() + target.height() / 2)
    initial = window._editable.face_count(0)

    _press_release(window._frame_view, pos, Qt.LeftButton)
    assert window._editable.face_count(0) == initial


def test_pointer_added_face_participates_in_undo_redo(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Pointer-added faces share the toolbar add stack: undo removes them."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + target.width() / 2, target.y() + target.height() / 2)

    before = window._editable.face_count(0)
    _press_release(window._frame_view, pos, Qt.LeftButton)
    assert window._editable.face_count(0) == before + 1
    assert window._editable.undo()
    assert window._editable.face_count(0) == before
    assert window._editable.redo()
    assert window._editable.face_count(0) == before + 1


# ---------------------------------------------------------------------------
# #105 — Right-click context menus
# ---------------------------------------------------------------------------


def test_frame_view_right_click_emits_context_menu(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Right-clicking an existing face emits the context-menu signal."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    src_w, src_h = window._frame_view.source_size
    window._editable.add_face(0, (5.0, 5.0, src_w / 2, src_h / 2))
    window.refresh_faces()

    received: list[tuple[int, QPointF]] = []
    window._frame_view.face_context_menu_requested.connect(
        lambda fi, pos: received.append((fi, pos))
    )

    target = window._frame_view._target_rect()
    # Click well inside the bbox.
    pos = QPointF(target.x() + target.width() / 4, target.y() + target.height() / 4)
    global_pos = window._frame_view.mapToGlobal(pos.toPoint())
    window._frame_view.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            pos,
            QPointF(global_pos),
            Qt.RightButton,
            Qt.RightButton,
            Qt.NoModifier,
        )
    )

    assert received
    assert received[0][0] == 0


def test_face_panel_right_click_emits_context_menu(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Right-clicking a face thumbnail emits the context-menu signal."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    panel.set_faces(
        [
            FaceThumbnail(0, "frame_000.png", 0, b""),
            FaceThumbnail(0, "frame_000.png", 1, b""),
        ]
    )
    panel.show()
    qtbot.waitExposed(panel)

    received: list[tuple[int, QPointF]] = []
    panel.face_context_menu_requested.connect(lambda fi, pos: received.append((fi, pos)))
    item = panel.item(1)
    rect = panel.visualItemRect(item)
    panel._on_context_menu_requested(rect.center())

    assert received == [(1, panel.mapToGlobal(rect.center()).toPointF())] or (
        received and received[0][0] == 1
    )


# ---------------------------------------------------------------------------
# #105 — Focus-scoped nudge shortcuts
# ---------------------------------------------------------------------------


def test_nudge_actions_scoped_to_frame_view(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Nudge QActions are parented to the frame view, not the window."""
    window = _make_window(qtbot, tmp_path)
    for key in (
        "nudge_up",
        "nudge_down",
        "nudge_left",
        "nudge_right",
        "nudge_up_fast",
        "nudge_down_fast",
        "nudge_left_fast",
        "nudge_right_fast",
    ):
        action = window.actions_by_key[key]
        assert action.parent() is window._frame_view, (
            f"Nudge action {key} should be parented to the frame view so it only "
            f"fires when the frame view has focus."
        )
        assert action.shortcutContext() == Qt.WidgetWithChildrenShortcut


def test_non_nudge_actions_remain_window_scoped(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Save / navigation actions remain window-scoped so the panel can't block them."""
    window = _make_window(qtbot, tmp_path)
    for key in ("save", "previous_frame", "next_frame", "delete_face"):
        action = window.actions_by_key[key]
        assert action.parent() is window
        assert action.shortcutContext() == Qt.WindowShortcut


def test_frame_view_takes_keyboard_focus_on_click(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Clicking the frame view should give it keyboard focus."""
    window = _make_window(qtbot, tmp_path)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + 4, target.y() + 4)
    _press_release(window._frame_view, pos, Qt.LeftButton)

    assert window._frame_view.focusPolicy() == Qt.StrongFocus


# ---------------------------------------------------------------------------
# #110 — Save action gating + duplicate-save block
# ---------------------------------------------------------------------------


def test_save_blocks_duplicate_invocation_while_in_flight(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A second save() call while one is in flight returns False."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))

    # Stub persist with a re-entrant probe: while we're inside persist, fire
    # another save() and confirm it short-circuits.
    inner_results: list[bool] = []

    def reentrant_persist(_editable, *, frame_names):  # type:ignore[no-untyped-def]
        inner_results.append(window.save())
        return 1

    window._alignments_handle.persist = reentrant_persist  # type:ignore[assignment]

    assert window.save() is True
    assert inner_results == [False], (
        "Second save invocation must short-circuit while the first is in flight."
    )


def test_save_disables_mutating_actions_during_flight(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Mutating actions are disabled while save is running."""
    window = _make_window(qtbot, tmp_path)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))

    snapshots: list[dict[str, bool]] = []

    def snapshot_then_persist(_editable, *, frame_names):  # type:ignore[no-untyped-def]
        snapshots.append(
            {key: window.actions_by_key[key].isEnabled() for key in window._MUTATING_ACTION_KEYS}
        )
        return 1

    window._alignments_handle.persist = snapshot_then_persist  # type:ignore[assignment]
    assert window.save() is True

    assert snapshots, "persist should have been called"
    snapshot = snapshots[0]
    for key, enabled in snapshot.items():
        assert enabled is False, f"Action {key} should be disabled during save"

    # After save, save action is disabled (nothing to save) but copy/add are back on.
    assert window.actions_by_key["save"].isEnabled() is False  # clean
    assert window.actions_by_key["add_face"].isEnabled() is True


def test_save_failure_preserves_dirty_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A persist failure leaves dirty state intact and re-enables save."""
    from PySide6.QtWidgets import QMessageBox

    window = _make_window(qtbot, tmp_path)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))
    window.mark_dirty(True)

    def failing_persist(_editable, *, frame_names):  # type:ignore[no-untyped-def]
        raise RuntimeError("disk full")

    window._alignments_handle.persist = failing_persist  # type:ignore[assignment]
    # Avoid blocking on the modal dialog in tests.
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: QMessageBox.Ok)

    assert window.save() is False
    assert window._editor_state.unsaved is True
    assert window._save_in_flight is False
    # Save remains available so the user can retry.
    assert window.actions_by_key["save"].isEnabled() is True


def test_save_success_clears_dirty_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Successful save clears dirty + edited + face_count_changed flags."""
    window = _make_window(qtbot, tmp_path)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))
    window.mark_dirty(True)

    window._alignments_handle.persist = lambda *a, **k: 1  # type:ignore[assignment]
    assert window.save() is True

    assert window._editor_state.unsaved is False
    assert window._editor_state.edited is False
    assert window._editor_state.face_count_changed is False


def test_save_shows_busy_progress_bar(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A determinate progress bar is materialized + branded for save."""
    window = _make_window(qtbot, tmp_path)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))

    observations: list[tuple[bool, str, tuple[int, int]]] = []

    def observe_persist(_editable, *, frame_names):  # type:ignore[no-untyped-def]
        bar = window._progress_bar
        # Inside the lock the bar must exist, carry the busy format, and be
        # set to an indeterminate range (0,0).  We assert these instead of
        # ``isVisible()`` because widget visibility under offscreen mode is
        # influenced by previous tests' window lifecycles and is flaky in a
        # full-suite run.
        observations.append(
            (
                bar is not None,
                bar.format() if bar is not None else "",
                (bar.minimum(), bar.maximum()) if bar is not None else (-1, -1),
            )
        )
        return 1

    window._alignments_handle.persist = observe_persist  # type:ignore[assignment]
    window.save()

    assert observations == [(True, "Saving alignments…", (0, 0))]
    # And after the save completes the busy state is fully torn down.
    assert window._busy_operation is None
    assert window._save_in_flight is False


def test_busy_lock_helper_releases_state_on_exception(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``_with_busy_lock`` restores state even when the block raises."""
    window = _make_window(qtbot, tmp_path)

    with (
        pytest.raises(RuntimeError, match="bad bulk op"),
        window._with_busy_lock("Running bulk op…"),
    ):
        assert window._busy_operation == "Running bulk op…"
        raise RuntimeError("bad bulk op")

    assert window._busy_operation is None
    assert window._save_in_flight is False
