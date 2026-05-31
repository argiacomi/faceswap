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

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication

from tools.manual.aligner_service import ManualAlignerService
from tools.manual.qt import (
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


class _InertAlignerBackend:
    """Small aligner double for interaction tests.

    These tests exercise pointer, context-menu and save behavior; they should
    not load real aligner plugins just because BoundingBox controls are shown.
    """

    def align(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        return np.zeros((68, 2), dtype=np.float32)

    def set_normalization(self, method: str) -> None:
        return None


def _inert_aligner_service() -> ManualAlignerService:
    """Return a deterministic, model-free aligner service for GUI interaction tests."""
    return ManualAlignerService(
        available=lambda: ("HRNet",),
        default=lambda: "HRNet",
        factory=lambda _aligner, _normalization: _InertAlignerBackend(),
    )


def _wait_for_frame_view_ready(qtbot, window: ManualToolWindow, *, timeout: int = 3000) -> None:
    """Wait until the frame view has both a source image and a usable target rect."""
    # Offscreen splitter layouts can leave the frame view at 0x0 even after
    # the source pixmap has loaded. Synthetic mouse tests need a stable
    # widget-space target rect, so force the frame pane to have geometry
    # before polling `_target_rect()`.
    window._frame_view.setMinimumSize(240, 160)
    window._frame_view.updateGeometry()
    qtbot.wait(0)

    def _ready() -> bool:
        if window._frame_view.source_size == (0, 0):
            return False
        if window._frame_view.width() <= 0 or window._frame_view.height() <= 0:
            return False
        rect = window._frame_view._target_rect()
        return rect.width() > 0 and rect.height() > 0

    qtbot.waitUntil(_ready, timeout=timeout)


def _make_window(qtbot, folder: Path) -> ManualToolWindow:
    session = _session_with_frames(folder, count=2)
    window = ManualToolWindow(session, aligner_service=_inert_aligner_service())
    # Interaction tests are about pointer/context/save behavior, not #104
    # auto-align.  Keep auto-run disabled so pointer-add is a single undoable
    # edit and these tests cannot touch model/plugin loading paths.
    window._editor_state.set("aligner_auto_run", False)
    window._frame_view.setMinimumSize(240, 160)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    _wait_for_frame_view_ready(qtbot, window)
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
            Qt.NoModifier,  # type: ignore[attr-defined]
        )
    )
    view.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            pos,
            QPointF(global_pos),
            button,
            Qt.NoButton,  # type: ignore[attr-defined]
            Qt.NoModifier,  # type: ignore[attr-defined]
        )
    )


def _source_to_widget(window: ManualToolWindow, sx: float, sy: float) -> QPointF:
    """Map a source-image point into the frame-view widget."""
    view = window._frame_view
    target = view._target_rect()
    src_w, src_h = view.source_size
    return QPointF(
        target.x() + target.width() * (sx / src_w),
        target.y() + target.height() * (sy / src_h),
    )


def _mouse_event(
    event_type: QEvent.Type,
    view: ManualFrameView,
    pos: QPointF,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> QMouseEvent:
    """Build a frame-view-local mouse event."""
    return QMouseEvent(
        event_type,
        pos,
        QPointF(view.mapToGlobal(pos.toPoint())),
        button,
        buttons,
        Qt.NoModifier,  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# #105 — Pointer-add gesture in BoundingBox mode
# ---------------------------------------------------------------------------


def test_click_in_bbox_mode_creates_face_at_pointer(qtbot, tmp_path: Path) -> None:
    """Clicking empty frame space in BBox mode creates a face under the pointer."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    _wait_for_frame_view_ready(qtbot, window)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + target.width() / 2, target.y() + target.height() / 2)
    initial_count = window._editable.face_count(0)

    _press_release(window._frame_view, pos, Qt.LeftButton)  # type: ignore[attr-defined]

    assert window._editable.face_count(0) == initial_count + 1
    # Pointer-added face goes through the same editable model, so undo works.
    assert window._editable.can_undo


def test_bbox_add_centers_default_box_without_source_clamp(qtbot, tmp_path: Path) -> None:
    """A BBox click near the frame edge centers the default square on the click."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    _wait_for_frame_view_ready(qtbot, window)

    pos = _source_to_widget(window, 5.0, 5.0)

    _press_release(window._frame_view, pos, Qt.LeftButton)  # type: ignore[attr-defined]

    face = window._editable.faces(0)[0]
    assert face.bbox == pytest.approx((-5.0, -5.0, 20.0, 20.0))


def test_entering_bbox_mode_does_not_passively_preload_aligner(qtbot, tmp_path: Path) -> None:
    """BBox controls becoming visible must not touch production aligner loading."""
    window = _make_window(qtbot, tmp_path)

    window._editor_state.set("editor_mode", "BoundingBox")

    assert window._aligner_load_worker is None
    assert window._aligner_load_target is None
    assert window._aligner_loaded_targets == set()


def test_click_in_view_mode_does_not_create_face(qtbot, tmp_path: Path) -> None:
    """Outside BBox mode an empty-space click must NOT add a face."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "View")
    _wait_for_frame_view_ready(qtbot, window)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + target.width() / 2, target.y() + target.height() / 2)
    initial = window._editable.face_count(0)

    _press_release(window._frame_view, pos, Qt.LeftButton)  # type: ignore[attr-defined]
    assert window._editable.face_count(0) == initial


def test_pointer_added_face_participates_in_undo_redo(qtbot, tmp_path: Path) -> None:
    """Pointer-added faces share the toolbar add stack: undo removes them."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    _wait_for_frame_view_ready(qtbot, window)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + target.width() / 2, target.y() + target.height() / 2)

    before = window._editable.face_count(0)
    _press_release(window._frame_view, pos, Qt.LeftButton)  # type: ignore[attr-defined]
    assert window._editable.face_count(0) == before + 1
    assert window._editable.undo()
    assert window._editable.face_count(0) == before
    assert window._editable.redo()
    assert window._editable.face_count(0) == before + 1


# ---------------------------------------------------------------------------
# #105 — Right-click context menus
# ---------------------------------------------------------------------------


def test_frame_view_right_click_emits_context_menu(qtbot, tmp_path: Path) -> None:
    """Right-clicking an existing face emits the context-menu signal."""
    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    _wait_for_frame_view_ready(qtbot, window)
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
            Qt.RightButton,  # type: ignore[attr-defined]
            Qt.RightButton,  # type: ignore[attr-defined]
            Qt.NoModifier,  # type: ignore[attr-defined]
        )
    )

    assert received
    assert received[0][0] == 0


def test_face_panel_right_click_emits_context_menu(qtbot, tmp_path: Path) -> None:
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


def test_nudge_actions_scoped_to_frame_view(qtbot, tmp_path: Path) -> None:
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
        assert action.shortcutContext() == Qt.WidgetWithChildrenShortcut  # type: ignore[attr-defined]


def test_non_nudge_actions_remain_window_scoped(qtbot, tmp_path: Path) -> None:
    """Save / navigation actions remain window-scoped so the panel can't block them."""
    window = _make_window(qtbot, tmp_path)
    for key in ("save", "previous_frame", "next_frame", "delete_face"):
        action = window.actions_by_key[key]
        assert action.parent() is window
        assert action.shortcutContext() == Qt.WindowShortcut  # type: ignore[attr-defined]


def test_frame_view_takes_keyboard_focus_on_click(qtbot, tmp_path: Path) -> None:
    """Clicking the frame view should give it keyboard focus."""
    window = _make_window(qtbot, tmp_path)
    _wait_for_frame_view_ready(qtbot, window)

    target = window._frame_view._target_rect()
    pos = QPointF(target.x() + 4, target.y() + 4)
    _press_release(window._frame_view, pos, Qt.LeftButton)  # type: ignore[attr-defined]

    assert window._frame_view.focusPolicy() == Qt.StrongFocus  # type: ignore[attr-defined]


def test_bbox_hover_cursor_uses_non_active_visible_face_handles(
    qtbot,
    tmp_path: Path,
) -> None:
    """BBox hover hit-testing follows every visible handle, not only active face."""
    window = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 30.0, 30.0))
    window._editable.add_face(0, (70.0, 10.0, 30.0, 30.0))
    window.refresh_faces()
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")
    pos = _source_to_widget(window, 100.0, 40.0)

    window._frame_view.mouseMoveEvent(
        _mouse_event(
            QEvent.Type.MouseMove,
            window._frame_view,
            pos,
            Qt.NoButton,  # type: ignore[attr-defined]
            Qt.NoButton,  # type: ignore[attr-defined]
        )
    )

    assert window._frame_view.hovered_face_index == 1
    assert window._frame_view.cursor().shape() == Qt.SizeFDiagCursor  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# #133 — Frame interaction stops playback before editing
# ---------------------------------------------------------------------------


def test_frame_view_left_click_stops_playback(qtbot, tmp_path: Path) -> None:
    """A direct frame click freezes playback before selection/edit dispatch."""
    window = _make_window(qtbot, tmp_path)
    _wait_for_frame_view_ready(qtbot, window)
    window.toggle_play()
    assert window._play_timer.isActive() is True

    target = window._frame_view._target_rect()
    pos = QPointF(target.center())
    _press_release(window._frame_view, pos, Qt.LeftButton)  # type: ignore[attr-defined]

    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False


def test_bbox_drag_stops_playback_and_does_not_advance_mid_drag(
    qtbot,
    tmp_path: Path,
) -> None:
    """Starting a BBox drag stops the timer before it can advance rows."""
    window = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window.refresh_faces()
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")
    start = _source_to_widget(window, 20.0, 20.0)

    window.toggle_play()
    assert window._play_timer.isActive() is True
    window._frame_view.mousePressEvent(
        _mouse_event(
            QEvent.Type.MouseButtonPress,
            window._frame_view,
            start,
            Qt.LeftButton,  # type: ignore[attr-defined]
            Qt.LeftButton,  # type: ignore[attr-defined]
        )
    )
    qtbot.wait(80)

    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False
    assert window._thumbnail_panel.currentRow() == 0
    window._frame_view.mouseReleaseEvent(
        _mouse_event(
            QEvent.Type.MouseButtonRelease,
            window._frame_view,
            start,
            Qt.LeftButton,  # type: ignore[attr-defined]
            Qt.NoButton,  # type: ignore[attr-defined]
        )
    )


def test_mask_paint_stops_playback_and_keeps_frame_mid_stroke(
    qtbot,
    tmp_path: Path,
) -> None:
    """Mask paint press stops playback before the paint stroke starts."""
    window = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window.refresh_faces()
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "Mask")
    start = _source_to_widget(window, 20.0, 20.0)

    window.toggle_play()
    assert window._play_timer.isActive() is True
    window._frame_view.mousePressEvent(
        _mouse_event(
            QEvent.Type.MouseButtonPress,
            window._frame_view,
            start,
            Qt.LeftButton,  # type: ignore[attr-defined]
            Qt.LeftButton,  # type: ignore[attr-defined]
        )
    )
    qtbot.wait(80)

    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False
    assert window._thumbnail_panel.currentRow() == 0
    window._frame_view.mouseReleaseEvent(
        _mouse_event(
            QEvent.Type.MouseButtonRelease,
            window._frame_view,
            start,
            Qt.LeftButton,  # type: ignore[attr-defined]
            Qt.NoButton,  # type: ignore[attr-defined]
        )
    )


def test_frame_view_right_click_stops_playback(qtbot, tmp_path: Path) -> None:
    """Right-click context-menu interaction also freezes playback first."""
    window = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window.refresh_faces()
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")
    pos = _source_to_widget(window, 20.0, 20.0)

    window.toggle_play()
    assert window._play_timer.isActive() is True
    window._frame_view.mousePressEvent(
        _mouse_event(
            QEvent.Type.MouseButtonPress,
            window._frame_view,
            pos,
            Qt.RightButton,  # type: ignore[attr-defined]
            Qt.RightButton,  # type: ignore[attr-defined]
        )
    )

    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False
    popup = QApplication.activePopupWidget()
    if popup is not None:
        popup.close()


# ---------------------------------------------------------------------------
# #110 — Save action gating + duplicate-save block
# ---------------------------------------------------------------------------


def _slow_persist(release, observations: list, modified: int = 1):
    """Persist stub that pauses on the worker thread until ``release`` fires.

    Used by the #115 async-save tests to assert main-thread state while the
    worker is mid-persist.  ``observations`` is a list the stub appends to so
    the test can verify what the worker saw under the busy lock.
    """

    def _stub(_editable, *, frame_names):
        observations.append(frame_names)
        release.wait(timeout=5.0)
        return modified

    return _stub


def test_save_blocks_duplicate_invocation_while_in_flight(qtbot, tmp_path: Path) -> None:
    """A second save() call while one is in flight returns False.

    Save is now async (#115) so re-entry is detected by ``_save_in_flight``,
    not by a re-entrant persist callback. Pause the worker mid-persist on the
    main thread and try to schedule a second save — it must short-circuit.
    """
    import threading

    window = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    _wait_for_frame_view_ready(qtbot, window)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))

    release = threading.Event()
    observations: list = []
    window._alignments_handle.persist = _slow_persist(release, observations)  # type: ignore[method-assign]

    try:
        assert window.save() is True
        qtbot.waitUntil(lambda: bool(observations), timeout=3000)
        # First save is mid-persist on the worker thread — try another.
        assert window.save() is False, (
            "Second save() must short-circuit while the first is in flight."
        )
    finally:
        release.set()
        qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)


def test_save_disables_mutating_actions_during_flight(qtbot, tmp_path: Path) -> None:
    """Mutating actions are disabled while save is running."""
    import threading

    window = _make_window(qtbot, tmp_path)
    _wait_for_frame_view_ready(qtbot, window)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))

    release = threading.Event()
    observations: list = []
    window._alignments_handle.persist = _slow_persist(release, observations)  # type: ignore[method-assign]

    try:
        assert window.save() is True
        # Wait for the worker to start persisting (it has entered the busy lock
        # because the worker construction + start happens after the lock).
        qtbot.waitUntil(lambda: bool(observations), timeout=3000)
        # Now the worker is paused inside persist — check action state from main.
        for key in window._MUTATING_ACTION_KEYS:
            assert window.actions_by_key[key].isEnabled() is False, (
                f"Action {key} should be disabled during save"
            )
    finally:
        release.set()
        qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    # After save, save action is disabled (nothing to save) but add is back on.
    assert window.actions_by_key["save"].isEnabled() is False  # clean
    assert window.actions_by_key["add_face"].isEnabled() is True


def test_save_failure_preserves_dirty_state(qtbot, tmp_path: Path) -> None:
    """A persist failure leaves dirty state intact and re-enables save."""
    window = _make_window(qtbot, tmp_path)
    _wait_for_frame_view_ready(qtbot, window)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))
    window.mark_dirty(True)

    def failing_persist(_editable, *, frame_names):
        raise RuntimeError("persist failed")

    window._alignments_handle.persist = failing_persist  # type: ignore[method-assign]

    # save() returns True (scheduled); the failure surfaces through the worker.  # type: ignore[method-assign]
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    assert window._editor_state.unsaved is True
    assert window._save_in_flight is False
    # Save remains available so the user can retry.
    assert window.actions_by_key["save"].isEnabled() is True


def test_save_success_clears_dirty_state(qtbot, tmp_path: Path) -> None:
    """Successful save clears dirty + edited + face_count_changed flags."""
    window = _make_window(qtbot, tmp_path)
    _wait_for_frame_view_ready(qtbot, window)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))
    window.mark_dirty(True)

    window._alignments_handle.persist = lambda *a, **k: 1  # type: ignore[method-assign]
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    assert window._editor_state.unsaved is False
    assert window._editor_state.edited is False
    assert window._editor_state.face_count_changed is False


def test_save_shows_busy_progress_bar(qtbot, tmp_path: Path) -> None:
    """A determinate progress bar is materialized + branded for save.

    Under async save, observability of the busy state belongs on the main
    thread — the worker just persists. So we pause persist mid-flight and
    snapshot the bar from the main thread.
    """
    import threading

    window = _make_window(qtbot, tmp_path)
    _wait_for_frame_view_ready(qtbot, window)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))

    release = threading.Event()
    observations: list = []
    window._alignments_handle.persist = _slow_persist(release, observations)  # type: ignore[method-assign]

    try:
        assert window.save() is True
        qtbot.waitUntil(lambda: bool(observations), timeout=3000)
        bar = window._progress_bar
        assert bar is not None
        assert bar.format() == "Saving alignments…"
        assert (bar.minimum(), bar.maximum()) == (0, 0)
    finally:
        release.set()
        qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    # And after the save completes the busy state is fully torn down.
    assert window._busy_operation is None
    assert window._save_in_flight is False


def test_busy_lock_helper_releases_state_on_exception(qtbot, tmp_path: Path) -> None:
    """``_with_busy_lock`` restores state even when the block raises."""
    window = _make_window(qtbot, tmp_path)

    with (
        pytest.raises(RuntimeError, match="bulk op failed"),
        window._with_busy_lock("Running bulk op…"),
    ):
        assert window._busy_operation == "Running bulk op…"
        raise RuntimeError("bulk op failed")

    assert window._busy_operation is None
    assert window._save_in_flight is False


# ---------------------------------------------------------------------------
# #115 — busy/progress feedback is visible BEFORE persistence begins
# ---------------------------------------------------------------------------


def test_save_busy_state_painted_before_persistence_completes(qtbot, tmp_path: Path) -> None:
    """The busy state + disabled actions are observable before persist returns.

    Under async save (#115), persistence runs on a worker thread, so the
    main-thread event loop has a chance to repaint between the schedule and
    the worker's completion.  We assert that *while the worker is still
    inside persist*:

    * ``_save_in_flight`` is True.
    * ``_busy_operation == "Saving alignments…"``.
    * The progress bar exists, is indeterminate (range 0,0) and carries the
      ``Saving alignments…`` format.
    * Mutating actions are disabled.

    All four invariants must hold *before* persistence completes — that's the
    parity-with-Tk concern the issue is closing.
    """
    import threading

    window = _make_window(qtbot, tmp_path)
    _wait_for_frame_view_ready(qtbot, window)
    window._editable.add_face(0, (5.0, 5.0, 20.0, 20.0))

    release = threading.Event()
    observations: list = []
    window._alignments_handle.persist = _slow_persist(release, observations)  # type: ignore[method-assign]

    try:
        assert window.save() is True
        # Wait for the worker to actually start persisting.
        qtbot.waitUntil(lambda: bool(observations), timeout=3000)
        assert window._save_in_flight is True
        assert window._busy_operation == "Saving alignments…"
        bar = window._progress_bar
        assert bar is not None
        assert bar.format() == "Saving alignments…"
        assert (bar.minimum(), bar.maximum()) == (0, 0)
        for key in window._MUTATING_ACTION_KEYS:
            assert window.actions_by_key[key].isEnabled() is False, (
                f"Action {key} must be disabled before persistence completes"
            )
    finally:
        release.set()
        qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    assert window._save_in_flight is False
    assert window._busy_operation is None
