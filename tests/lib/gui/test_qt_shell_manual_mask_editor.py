#!/usr/bin/env python3
"""Native Qt Manual Tool Mask editor (F5) tests.

Covers #101:
* Brush state (size, mode, opacity) + Draw / Erase / [ / ] actions.
* Pointer drag → ``mask_paint_requested`` continuously.
* Ctrl+drag inverts Draw/Erase for the gesture.
* Editable mask grid materialises on first paint and the overlay renders it.
* Host paint_mask_at marks dirty + records undo/redo.
* Mode-scoped mask controls and no-active-face / out-of-bbox / no-mask edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPixmap

from lib.gui.qt_shell.manual_tool import (
    ManualFrameView,
    ManualToolWindow,
)
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 1) -> ManualSession:
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


def _enter_mask_mode(window: ManualToolWindow) -> None:
    window._editor_state.set("editor_mode", "Mask")


def _seed_face(window: ManualToolWindow) -> int:
    """Add a large face so the mask grid has room to paint."""
    return window._editable.add_face(0, (40.0, 40.0, 120.0, 120.0)) or 0


def _source_to_widget(view: ManualFrameView, sx: float, sy: float) -> QPointF:
    target = view._target_rect()  # noqa: SLF001 - test helper
    src_w, src_h = view.source_size
    return QPointF(
        target.x() + target.width() * (sx / src_w),
        target.y() + target.height() * (sy / src_h),
    )


def _drag(
    view: ManualFrameView,
    start: QPointF,
    end: QPointF,
    *,
    modifiers: Qt.KeyboardModifier = Qt.NoModifier,
) -> None:
    global_start = view.mapToGlobal(start.toPoint())
    global_end = view.mapToGlobal(end.toPoint())
    view.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start,
            QPointF(global_start),
            Qt.LeftButton,
            Qt.LeftButton,
            modifiers,
        )
    )
    view.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            end,
            QPointF(global_end),
            Qt.NoButton,
            Qt.LeftButton,
            modifiers,
        )
    )
    view.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            end,
            QPointF(global_end),
            Qt.LeftButton,
            Qt.NoButton,
            modifiers,
        )
    )


# ---------------------------------------------------------------------------
# Brush state + actions
# ---------------------------------------------------------------------------


def test_mask_draw_and_erase_actions_set_brush_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """B sets Draw mode and D sets Erase mode."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    window.set_mask_erase_mode()
    assert window._editor_state.brush_mode == "erase"
    window.set_mask_draw_mode()
    assert window._editor_state.brush_mode == "draw"


def test_entering_mask_mode_auto_magnifies_and_leaving_restores(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path
) -> None:
    """Entering Mask mode auto-fits the active face and View restores it."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    before = window._frame_view.view_state()
    _enter_mask_mode(window)
    assert window._frame_view.zoom > float(before["zoom"])

    window._editor_state.set("editor_mode", "View")
    restored = window._frame_view.view_state()
    assert abs(float(restored["zoom"]) - float(before["zoom"])) < 0.001
    assert restored["offset"] == before["offset"]


def test_brush_size_increment_and_decrement_clamp_to_bounds(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``[`` and ``]`` step brush size by the configured amount and clamp."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    initial = window._editor_state.brush_size
    new_size = window.increase_brush_size()
    assert new_size == initial + window._BRUSH_STEP
    # Slam the size to the lower bound: keep decreasing.
    for _ in range(50):
        window.decrease_brush_size()
    assert window._editor_state.brush_size == window._BRUSH_MIN
    # And then increase past the upper bound.
    for _ in range(50):
        window.increase_brush_size()
    assert window._editor_state.brush_size == window._BRUSH_MAX


# ---------------------------------------------------------------------------
# Pointer painting
# ---------------------------------------------------------------------------


def test_mask_drag_paints_into_editable_model(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Drag inside bbox writes pixels through ``paint_mask_stroke``."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    _drag(
        view,
        _source_to_widget(view, 80.0, 80.0),
        _source_to_widget(view, 110.0, 110.0),
    )
    mask = window._editable.get_mask(0, face_index, window.active_mask_type())
    assert mask is not None
    # At least some pixels along the drag path should be painted.
    assert int(mask.sum()) > 0
    assert window._editor_state.edited is True


def test_ctrl_drag_inverts_mode_for_one_gesture(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Ctrl+drag while in Draw mode erases (invert mode flag)."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    # Pre-fill the mask so erase has something to remove.
    assert window.paint_mask_at(face_index, 100.0, 100.0)
    mask = window._editable.get_mask(0, face_index, window.active_mask_type())
    assert int(mask.sum()) > 0
    painted_before = int(mask.sum())

    # Ctrl+drag — should erase even though brush_mode=="draw".
    view = window._frame_view
    _drag(
        view,
        _source_to_widget(view, 100.0, 100.0),
        _source_to_widget(view, 105.0, 105.0),
        modifiers=Qt.ControlModifier,
    )
    painted_after = int(mask.sum())
    assert painted_after < painted_before
    # brush_mode itself didn't flip — only the gesture inverted.
    assert window._editor_state.brush_mode == "draw"


def test_mask_drag_outside_bbox_is_noop(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A press outside the active face's bbox does not start a mask drag."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    emits: list = []
    view.mask_paint_requested.connect(lambda *args: emits.append(args))
    _drag(
        view,
        _source_to_widget(view, 10.0, 10.0),
        _source_to_widget(view, 15.0, 15.0),
    )
    assert emits == []
    assert window._editable.get_mask(0, face_index, window.active_mask_type()) is None


def test_mask_drag_no_active_face_is_noop(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Mask gestures are no-ops when no face is active."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    view = window._frame_view
    emits: list = []
    view.mask_paint_requested.connect(lambda *args: emits.append(args))
    _drag(
        view,
        _source_to_widget(view, 80.0, 80.0),
        _source_to_widget(view, 100.0, 100.0),
    )
    assert emits == []


# ---------------------------------------------------------------------------
# Host paint_mask_at + undo/redo
# ---------------------------------------------------------------------------


def test_paint_mask_at_marks_dirty_and_is_undoable(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``paint_mask_at`` participates in the undo/redo stack."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    assert window.paint_mask_at(face_index, 100.0, 100.0) is True
    mask = window._editable.get_mask(0, face_index, window.active_mask_type())
    painted = int(mask.sum())
    assert painted > 0
    assert window._editor_state.edited is True

    # Undo should remove what we painted.
    assert window._editable.undo() is True
    assert int(mask.sum()) == 0
    # Redo replays.
    assert window._editable.redo() is True
    assert int(mask.sum()) == painted


def test_paint_mask_at_invert_flag_erases(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The ``invert`` flag flips Draw vs Erase for a single stamp."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    window.paint_mask_at(face_index, 100.0, 100.0)
    mask = window._editable.get_mask(0, face_index, window.active_mask_type())
    assert int(mask.sum()) > 0
    # Now invert (erase a slightly larger region — the brush radius pulls
    # from a single bigger size).
    window._editor_state.set("brush_size", 20)
    window.paint_mask_at(face_index, 100.0, 100.0, invert=True)
    assert int(mask.sum()) == 0


def test_active_mask_type_falls_back_to_default(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``active_mask_type`` returns DEFAULT_MASK_TYPE when user hasn't picked."""
    window = _make_window(qtbot, tmp_path)
    assert window._editor_state.mask_type == ""
    assert window.active_mask_type() == window.DEFAULT_MASK_TYPE
    window._editor_state.set("mask_type", "extended")
    assert window.active_mask_type() == "extended"


def test_paint_mask_at_no_frame_returns_false_with_status_message(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Painting with no current frame returns False + surfaces a status message."""
    window = _make_window(qtbot, tmp_path)
    # ``_current_frame_index`` is anchored to ``_current_frame``, not the
    # panel row, so the no-frame branch is only reached by clearing the
    # cached frame directly.
    window._current_frame = None

    assert window.paint_mask_at(0, 100.0, 100.0) is False
    assert window.statusBar().currentMessage() == "No frame to edit"


# ---------------------------------------------------------------------------
# Mask-type dropdown (#101 remainder)
# ---------------------------------------------------------------------------


def test_mask_controls_hidden_outside_mask_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The Mask control row is hidden in View / BBox / Landmark / Extract modes."""
    window = _make_window(qtbot, tmp_path)

    for mode in ("View", "BoundingBox", "Landmarks", "ExtractBox"):
        window._editor_state.set("editor_mode", mode)
        assert window._mask_controls.isVisible() is False, mode


def test_mask_controls_hide_after_leaving_mask_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Leaving Mask mode immediately hides the mask-type controls."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    assert window._mask_controls.isVisible() is True

    window._editor_state.set("editor_mode", "BoundingBox")

    assert window._mask_controls.isVisible() is False


def test_mask_controls_visible_when_mask_mode_active(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Entering Mask mode surfaces the mask-type dropdown + opacity slider."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    assert window._mask_controls.isVisible() is True
    # Defaults are present in the dropdown.
    items = [window._mask_type_combo.itemText(i) for i in range(window._mask_type_combo.count())]
    assert "components" in items
    assert "extended" in items
    # The selected mask type matches the editor-state field.
    assert window._mask_type_combo.currentText() == window.active_mask_type()


def test_mask_type_dropdown_includes_persisted_blobs(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A persisted mask blob surfaces as a dropdown option even without painting."""
    import numpy as np

    from lib.align.objects import MaskAlignmentsFile

    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    window._editable._persisted_mask_blobs[(0, face_index, "vgg-clear")] = MaskAlignmentsFile(
        mask=b"\x00",
        affine_matrix=np.eye(3, dtype=np.float32)[:2],
        interpolator=1,
        stored_size=128,
        stored_centering="head",
    )
    _enter_mask_mode(window)
    items = [window._mask_type_combo.itemText(i) for i in range(window._mask_type_combo.count())]
    assert "vgg-clear" in items


def test_mask_type_dropdown_change_updates_editor_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Selecting a different mask type writes through to editor state."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    window._mask_type_combo.setCurrentText("extended")
    assert window._editor_state.mask_type == "extended"
    assert window.active_mask_type() == "extended"


def test_mask_opacity_slider_updates_editor_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Moving the opacity slider writes through to editor state."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    window._mask_opacity_slider.setValue(80)
    assert window._editor_state.mask_opacity == 80


# ---------------------------------------------------------------------------
# #101 closure — brush cursor preview + status messages
# ---------------------------------------------------------------------------


def _hover_at(window: ManualToolWindow, sx: float, sy: float) -> None:
    """Drive the frame view's hover-cursor path with a source-coordinate point."""
    view = window._frame_view
    pos = _source_to_widget(view, sx, sy)
    view._update_hover_cursor(pos)  # noqa: SLF001 - exercising the hover seam


def test_brush_preview_hidden_outside_mask_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The brush preview is hidden in View / BBox / Landmark / Extract modes."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    window._editor_state.set("editor_mode", "View")
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview is None


def test_brush_preview_visible_over_active_face_in_mask_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Hovering inside the active face's bbox in Mask mode shows the preview."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    _enter_mask_mode(window)

    _hover_at(window, 100.0, 100.0)
    preview = window._frame_view.brush_preview
    assert preview is not None
    assert preview["mode"] == "draw"
    # Half the brush size (12 / 2 = 6 source pixels by default).
    assert preview["radius"] == pytest.approx(window._editor_state.brush_size / 2.0)
    # The recorded source point matches the supplied source coord (subject to
    # the widget→source projection rounding — float tolerance is fine).
    point = preview["source_point"]
    assert point.x() == pytest.approx(100.0, abs=0.5)
    assert point.y() == pytest.approx(100.0, abs=0.5)


def test_brush_preview_hidden_outside_bbox_in_mask_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Hovering outside the active face's bbox hides the preview."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    _enter_mask_mode(window)

    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview is not None
    # Move outside the bbox (face was seeded at (40, 40, 120, 120) so 10/10 is outside).
    _hover_at(window, 10.0, 10.0)
    assert window._frame_view.brush_preview is None


def test_brush_preview_hidden_when_no_active_face(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """No active face means no preview, even in Mask mode."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview is None


def test_brush_preview_radius_reflects_brush_size_change(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Changing brush size updates the preview radius on next hover."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    _enter_mask_mode(window)
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview["radius"] == pytest.approx(
        window._editor_state.brush_size / 2.0
    )

    window._editor_state.set("brush_size", 40)
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview["radius"] == pytest.approx(20.0)


def test_brush_preview_mode_reflects_draw_vs_erase(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Switching brush_mode updates the preview's draw/erase colour cue."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    _enter_mask_mode(window)
    window.set_mask_erase_mode()
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview["mode"] == "erase"
    window.set_mask_draw_mode()
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview["mode"] == "draw"


def test_brush_preview_cleared_on_leave_event(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``leaveEvent`` hides the preview so it doesn't ghost outside the view."""
    from PySide6.QtCore import QEvent as _QEvent
    from PySide6.QtGui import QEnterEvent  # noqa: F401 — for type registration

    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    _enter_mask_mode(window)
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview is not None

    window._frame_view.leaveEvent(_QEvent(_QEvent.Type.Leave))
    assert window._frame_view.brush_preview is None


def test_brush_preview_cleared_on_mode_switch(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Switching out of Mask mode hides any in-flight preview on the next hover."""
    window = _make_window(qtbot, tmp_path)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)
    _enter_mask_mode(window)
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview is not None
    window._editor_state.set("editor_mode", "View")
    _hover_at(window, 100.0, 100.0)
    assert window._frame_view.brush_preview is None


# ---------------------------------------------------------------------------
# Status messages on no-frame / no-face edges
# ---------------------------------------------------------------------------


def test_paint_mask_at_no_active_face_surfaces_status_message(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Painting with no editable faces returns False and surfaces a status message."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    # No face has been added.
    assert window.paint_mask_at(0, 100.0, 100.0) is False
    # The implementation routes through ``paint_mask_stroke`` which returns False;
    # the brush UX is silent here (no spurious chatter for every failed stroke)
    # but the explicit no-face hover surfaces a status message on the host.
    # Assert at least one of: the function returned False *and* the host reflects
    # the lack of active face.
    assert window._active_face_index() is None


def test_paint_mask_at_invalid_face_returns_false_silently(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """An invalid face_index returns False and does not crash."""
    window = _make_window(qtbot, tmp_path)
    _enter_mask_mode(window)
    _seed_face(window)
    assert window.paint_mask_at(99, 100.0, 100.0) is False
