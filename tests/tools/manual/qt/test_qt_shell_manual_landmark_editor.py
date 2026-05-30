#!/usr/bin/env python3
"""Native Qt Manual Tool Landmark editor (F4) tests.

Covers #103:
* Overlay hit-test + marquee helpers.
* Single-point drag → ``landmark_move_requested``.
* Marquee selection → ``landmarks_select_requested``.
* Group move on a prior selection → ``landmarks_move_requested``.
* Frame-view ↔ editable model wiring (dirty state, undo/redo).
* Magnify action (fit active face).
* No-active-face and no-landmark edge cases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPixmap

from tools.manual.qt import (
    ManualFrameOverlay,
    ManualFrameView,
    ManualToolWindow,
)
from tools.manual.session import ManualSession

# ---------------------------------------------------------------------------
# Pure overlay helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _silence_expected_pose_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="lib.align.pose")

def test_landmark_at_returns_closest_index_within_tolerance() -> None:
    """``landmark_at`` picks the closest landmark, not the first matching."""
    landmarks = ((0.0, 0.0), (10.0, 10.0), (10.5, 10.5))
    # Point (10.6, 10.6) is closer to index 2 than to index 1.
    assert ManualFrameOverlay.landmark_at(landmarks, (10.6, 10.6), tolerance=2.0) == 2
    # Point well outside the radius returns None.
    assert ManualFrameOverlay.landmark_at(landmarks, (200.0, 200.0), tolerance=2.0) is None
    # Empty landmark list returns None instead of raising.
    assert ManualFrameOverlay.landmark_at((), (0.0, 0.0), tolerance=2.0) is None


def test_landmarks_in_rect_returns_indices_inclusive() -> None:
    """``landmarks_in_rect`` returns every index whose coords fall inside the rect."""
    landmarks = ((1.0, 1.0), (5.0, 5.0), (10.0, 10.0))
    rect = QRectF(0.0, 0.0, 6.0, 6.0)
    assert ManualFrameOverlay.landmarks_in_rect(landmarks, rect) == (0, 1)
    # A zero-size rect returns an empty tuple.
    assert ManualFrameOverlay.landmarks_in_rect(landmarks, QRectF(0, 0, 0, 0)) == ()


# ---------------------------------------------------------------------------
# Host wiring
# ---------------------------------------------------------------------------


def _session_with_frames(folder: Path, count: int = 1) -> ManualSession:
    """Write ``count`` small PNG fixtures and return a session."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(200, 200)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    session = _session_with_frames(folder)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    return window


def _enter_landmark_mode(window: ManualToolWindow) -> None:
    """Switch the window to F4 landmark mode and ensure overlay state is fresh."""
    window._editor_state.set("editor_mode", "Landmarks")


def _seed_face_with_landmarks(window: ManualToolWindow) -> int:
    """Add a face with a handful of landmarks and return its face_index."""
    landmarks = ((50.0, 50.0), (60.0, 60.0), (70.0, 70.0), (80.0, 80.0))
    return window._editable.add_face(0, (40.0, 40.0, 60.0, 60.0), landmarks=landmarks) or 0


def _source_to_widget(view: ManualFrameView, sx: float, sy: float) -> QPointF:
    """Project a source-pixel point to widget coordinates via the live target rect."""
    target = view._target_rect()  # noqa: SLF001 - exposed for tests
    src_w, src_h = view.source_size
    return QPointF(
        target.x() + target.width() * (sx / src_w),
        target.y() + target.height() * (sy / src_h),
    )


def _drag(view: ManualFrameView, start: QPointF, end: QPointF) -> None:
    """Synthesize a press → move → release sequence."""
    global_start = view.mapToGlobal(start.toPoint())
    global_end = view.mapToGlobal(end.toPoint())
    view.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start,
            QPointF(global_start),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    view.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            end,
            QPointF(global_end),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    view.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            end,
            QPointF(global_end),
            Qt.LeftButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
    )


def test_single_landmark_drag_updates_editable_model(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Dragging a landmark moves only that point in the editable model."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    start = _source_to_widget(view, 50.0, 50.0)
    end = _source_to_widget(view, 55.0, 53.0)
    _drag(view, start, end)

    landmarks = window._editable.faces(0)[0].landmarks
    # Landmark 0 should now sit at ~(55, 53); others unchanged.
    assert abs(landmarks[0][0] - 55.0) < 0.5
    assert abs(landmarks[0][1] - 53.0) < 0.5
    assert landmarks[1] == (60.0, 60.0)
    assert window._editor_state.edited is True


def test_landmark_hover_uses_large_anchor_label_and_hidden_cursor(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Hover near a point highlights it, labels it, and hides the native cursor."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    # Point is 6 source pixels away from landmark 0: outside the visible
    # dot but inside the larger invisible grab/hover anchor.
    window._frame_view._update_hover_cursor(_source_to_widget(window._frame_view, 56.0, 50.0))

    assert window._frame_view.landmark_hover == {"face_index": face_index, "landmark_index": 0}
    assert window._overlay.hovered_landmark == (face_index, 0)
    assert window._frame_view.cursor().shape() == Qt.BlankCursor


def test_marquee_selects_landmarks_inside_rect(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """An empty-space drag inside the bbox selects every landmark inside the rect."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    # Start in empty space (inside bbox but away from any landmark), end past
    # landmarks 0 and 1 only.
    start = _source_to_widget(view, 45.0, 90.0)
    end = _source_to_widget(view, 65.0, 45.0)
    _drag(view, start, end)

    selected = window._overlay.selected_landmarks
    assert 0 in selected
    assert 1 in selected
    # Landmark 2 (at 70,70) and 3 (at 80,80) are outside the marquee.
    assert 2 not in selected
    assert 3 not in selected


def test_marquee_crossing_multiple_faces_clears_selection(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """A marquee containing landmarks from multiple faces is rejected like Tk."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editable.add_face(
        0,
        (100.0, 100.0, 60.0, 60.0),
        landmarks=((110.0, 80.0), (120.0, 85.0)),
    )
    window._editor_state.set("face_index", face_index)
    window._overlay.set_selected_landmarks((0, 1))

    view = window._frame_view
    _drag(view, _source_to_widget(view, 45.0, 95.0), _source_to_widget(view, 125.0, 45.0))

    assert window._overlay.selected_landmarks == frozenset()


def test_group_move_translates_only_selected_landmarks(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """After a marquee selection, dragging a selected point moves the group."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    # Pre-seed the selection set with two indices.
    window._overlay.set_selected_landmarks((0, 1))

    view = window._frame_view
    # Drag from landmark 0's position by (10, 5).
    start = _source_to_widget(view, 50.0, 50.0)
    end = _source_to_widget(view, 60.0, 55.0)
    _drag(view, start, end)

    landmarks = window._editable.faces(0)[0].landmarks
    # Landmarks 0 and 1 moved by ~(10, 5); 2 and 3 unchanged.
    assert abs(landmarks[0][0] - 60.0) < 0.5
    assert abs(landmarks[0][1] - 55.0) < 0.5
    assert abs(landmarks[1][0] - 70.0) < 0.5
    assert abs(landmarks[1][1] - 65.0) < 0.5
    assert landmarks[2] == (70.0, 70.0)
    assert landmarks[3] == (80.0, 80.0)


def test_zoomed_landmark_drag_uses_inverse_view_transform(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """After auto-magnify, widget drags still land in source coordinates."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)
    _enter_landmark_mode(window)
    assert window._frame_view.zoom > 1.0

    view = window._frame_view
    _drag(view, _source_to_widget(view, 50.0, 50.0), _source_to_widget(view, 58.0, 57.0))

    landmarks = window._editable.faces(0)[0].landmarks
    assert landmarks[0][0] == pytest.approx(58.0, abs=0.5)
    assert landmarks[0][1] == pytest.approx(57.0, abs=0.5)


def test_editor_mode_and_frame_switch_clear_landmark_temp_state(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Selections, hover labels and drags clear when editor/frame changes."""
    session = _session_with_frames(tmp_path, count=2)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)
    _enter_landmark_mode(window)
    window._overlay.set_selected_landmarks((0, 1))
    window._frame_view._update_hover_cursor(_source_to_widget(window._frame_view, 56.0, 50.0))
    assert window._overlay.selected_landmarks
    assert window._overlay.hovered_landmark == (face_index, 0)

    window._editor_state.set("editor_mode", "View")

    assert window._overlay.selected_landmarks == frozenset()
    assert window._overlay.hovered_landmark is None
    assert window._frame_view.landmark_hover is None

    window._overlay.set_selected_landmarks((0, 1))
    window._editor_state.set("editor_mode", "Landmarks")
    window._thumbnail_panel.setCurrentRow(1)
    assert window._overlay.selected_landmarks == frozenset()
    assert window._overlay.hovered_landmark is None


def test_landmark_edit_participates_in_undo_history(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A landmark drag is undoable + redoable through the editable model."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    _drag(
        view,
        _source_to_widget(view, 50.0, 50.0),
        _source_to_widget(view, 55.0, 53.0),
    )

    assert window._editable.can_undo is True
    assert window._editable.undo() is True
    landmarks = window._editable.faces(0)[0].landmarks
    assert landmarks[0] == (50.0, 50.0)
    assert window._editable.redo() is True
    landmarks = window._editable.faces(0)[0].landmarks
    assert abs(landmarks[0][0] - 55.0) < 0.5


def test_landmark_mode_with_no_active_face_falls_back_to_pan(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path
) -> None:
    """Landmark gestures are no-ops when there is no active face."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    # No face added; face_index defaults to 0 but the editable model has no face.
    view = window._frame_view
    moves: list = []
    view.landmark_move_requested.connect(lambda *args: moves.append(args))

    _drag(
        view,
        _source_to_widget(view, 50.0, 50.0),
        _source_to_widget(view, 60.0, 60.0),
    )
    assert moves == []
    # And the editable model is untouched.
    assert window._editable.face_count(0) == 0


def test_magnify_active_face_zooms_and_pans(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``magnify_active_face`` zooms into the bbox of the active face."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    before_zoom = window._frame_view._zoom  # noqa: SLF001
    assert window.magnify_active_face() is True
    after_zoom = window._frame_view._zoom  # noqa: SLF001
    assert after_zoom > before_zoom


def test_magnify_active_face_toggles_back_to_previous_view(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path
) -> None:
    """The Landmark magnify command restores the prior zoom/pan on second trigger."""
    window = _make_window(qtbot, tmp_path)
    _enter_landmark_mode(window)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    before = window._frame_view.view_state()
    assert window.magnify_active_face() is True
    assert window._frame_view.zoom > float(before["zoom"])

    assert window.magnify_active_face() is True
    restored = window._frame_view.view_state()
    assert abs(float(restored["zoom"]) - float(before["zoom"])) < 0.001
    assert restored["offset"] == before["offset"]


def test_entering_landmarks_auto_magnifies_and_leaving_restores(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path
) -> None:
    """Entering Landmark mode auto-fits the active face and View restores it."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face_with_landmarks(window)
    window._editor_state.set("face_index", face_index)

    before = window._frame_view.view_state()
    window._editor_state.set("editor_mode", "Landmarks")
    assert window._frame_view.zoom > float(before["zoom"])

    window._editor_state.set("editor_mode", "View")
    restored = window._frame_view.view_state()
    assert abs(float(restored["zoom"]) - float(before["zoom"])) < 0.001
    assert restored["offset"] == before["offset"]


def test_magnify_with_no_active_face_is_noop_and_surfaces_status(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path
) -> None:
    """``magnify_active_face`` returns False + shows a status message when nothing is active."""
    window = _make_window(qtbot, tmp_path)
    # No face added.
    assert window.magnify_active_face() is False
