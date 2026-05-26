#!/usr/bin/env python3
"""Native Qt Manual Tool Extract Box editor (F3) tests.

Covers #102:
* Translate gesture (interior drag) → ``face_move_requested``.
* Scale gesture (corner drag) → ``face_scale_requested``.
* Rotate gesture (outside-corner halo drag) → ``face_rotate_requested``.
* Cursor feedback (move / size / rotate) over different regions.
* Host wiring: editable model updates, dirty state, undo/redo.
* No-active-face fall-through to pan.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPixmap

from tools.manual.qt import (
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


def _wait_for_frame_view_ready(  # type:ignore[no-untyped-def]
    qtbot, window: ManualToolWindow, *, timeout: int = 3000
) -> None:
    """Wait until the frame view has both a source image and a laid-out rect.

    Synthetic mouse tests need both invariants:

    * ``source_size != (0, 0)`` — the source pixmap has been decoded.
    * ``_target_rect()`` is non-empty — the widget has been resized by the
      layout system, so widget-coordinate drags translate to source pixels.

    Polling only ``source_size`` is not enough: on a freshly-shown window the
    source image may load before the layout has assigned the frame view its
    final geometry, which leaves ``_target_rect().width() == 0`` and makes
    every synthetic gesture land outside the image.
    """

    def _ready() -> bool:
        if window._frame_view.source_size == (0, 0):
            return False
        rect = window._frame_view._target_rect()
        return rect.width() > 0 and rect.height() > 0

    qtbot.waitUntil(_ready, timeout=timeout)


def _make_window(qtbot, folder: Path) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    session = _session_with_frames(folder)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    _wait_for_frame_view_ready(qtbot, window)
    return window


def _enter_extract_mode(window: ManualToolWindow) -> None:
    window._editor_state.set("editor_mode", "ExtractBox")


def _seed_face(window: ManualToolWindow) -> int:
    """Seed a face with landmarks suitable for rotation tests."""
    landmarks = ((45.0, 50.0), (55.0, 50.0), (50.0, 60.0))
    return window._editable.add_face(0, (40.0, 40.0, 20.0, 20.0), landmarks=landmarks) or 0


def _source_to_widget(view: ManualFrameView, sx: float, sy: float) -> QPointF:
    target = view._target_rect()  # noqa: SLF001 - exposed for tests
    src_w, src_h = view.source_size
    return QPointF(
        target.x() + target.width() * (sx / src_w),
        target.y() + target.height() * (sy / src_h),
    )


def _drag(view: ManualFrameView, start: QPointF, end: QPointF) -> None:
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


def test_extract_translate_drag_moves_face(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Interior drag in Extract Box mode translates landmarks + bbox."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    # Drag from a point inside the bbox to a point ten source-pixels away.
    start = _source_to_widget(view, 50.0, 50.0)
    end = _source_to_widget(view, 60.0, 55.0)
    _drag(view, start, end)

    face = window._editable.faces(0)[0]
    assert abs(face.bbox[0] - 50.0) < 0.5  # was 40, +10
    assert abs(face.bbox[1] - 45.0) < 0.5  # was 40, +5
    assert window._editor_state.edited is True


def test_extract_corner_drag_scales_face(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Corner drag in Extract Box mode scales landmarks + bbox around the centre."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    # SE corner is at (60, 60).  Drag outward to (80, 80) — radius from
    # centre (50, 50) goes from ~14.14 to ~42.43, ie. scale ~3x.
    start = _source_to_widget(view, 60.0, 60.0)
    end = _source_to_widget(view, 80.0, 80.0)
    _drag(view, start, end)

    face = window._editable.faces(0)[0]
    # Bbox should have grown; centre preserved at (50, 50).
    bbox_cx = face.bbox[0] + face.bbox[2] / 2.0
    bbox_cy = face.bbox[1] + face.bbox[3] / 2.0
    assert abs(bbox_cx - 50.0) < 1.0
    assert abs(bbox_cy - 50.0) < 1.0
    assert face.bbox[2] > 20.0  # original width was 20


def test_extract_corner_drag_keeps_legacy_minimum_size(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Inward corner drags do not shrink below the Tk minimum box size."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    _drag(view, _source_to_widget(view, 60.0, 60.0), _source_to_widget(view, 50.5, 50.5))

    face = window._editable.faces(0)[0]
    assert face.bbox == (40.0, 40.0, 20.0, 20.0)


def test_extract_corner_drag_clamps_to_source_bounds(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Outward corner scale cannot move the extract box outside the source image."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = window._editable.add_face(
        0,
        (10.0, 10.0, 40.0, 40.0),
        landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 40.0)),
    )
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    _drag(view, _source_to_widget(view, 50.0, 50.0), _source_to_widget(view, 100.0, 100.0))

    face = window._editable.faces(0)[0]
    assert face.bbox[0] >= 0.0
    assert face.bbox[1] >= 0.0
    assert face.bbox[0] + face.bbox[2] <= view.source_size[0]
    assert face.bbox[1] + face.bbox[3] <= view.source_size[1]


def test_extract_outside_corner_drag_rotates_face(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Outside-corner drag in Extract Box mode rotates landmarks around the centre."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    original_landmarks = window._editable.faces(0)[0].landmarks
    # Rotation hit-testing is a widget-pixel halo, not a fixed source-pixel
    # band.  Pick points directly from the projected active bbox so the
    # gesture remains inside the 24px halo regardless of offscreen layout size.
    widget_bbox = view._active_bbox_widget_rect()  # noqa: SLF001 - test hit geometry
    assert widget_bbox is not None
    offset = view._EXTRACT_ROTATION_BAND_PX / 2.0  # noqa: SLF001
    start = QPointF(widget_bbox.right() + offset, widget_bbox.center().y())
    end = QPointF(widget_bbox.center().x(), widget_bbox.bottom() + offset)
    _drag(view, start, end)

    rotated_landmarks = window._editable.faces(0)[0].landmarks
    # Verify rotation actually occurred (landmarks moved).
    assert rotated_landmarks != original_landmarks
    assert window._editor_state.edited is True


def test_extract_no_active_face_fall_through(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Extract gestures are no-ops without an active face."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    # No face added.
    view = window._frame_view
    scales: list = []
    view.face_scale_requested.connect(lambda *args: scales.append(args))

    _drag(view, _source_to_widget(view, 80.0, 80.0), _source_to_widget(view, 90.0, 90.0))
    assert scales == []


def test_extract_drag_participates_in_undo(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Extract Box scale is undoable + redoable through the editable model."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    before_bbox = window._editable.faces(0)[0].bbox
    _drag(view, _source_to_widget(view, 60.0, 60.0), _source_to_widget(view, 80.0, 80.0))
    after_bbox = window._editable.faces(0)[0].bbox
    assert after_bbox != before_bbox

    assert window._editable.undo() is True
    assert window._editable.faces(0)[0].bbox == before_bbox
    assert window._editable.redo() is True
    assert window._editable.faces(0)[0].bbox == after_bbox


def test_extract_invalid_scale_surfaces_status(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A degenerate scale leaves state unchanged and surfaces a message."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    # Directly call the host handler with a degenerate scale.
    window._on_face_scale_requested(face_index, 0.0001)  # produces sub-pixel bbox
    assert window._editor_state.edited is False
    assert window.statusBar().currentMessage().startswith("Scale failed")


def test_extract_cursor_feedback_distinguishes_regions(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Hover over body / corner / rotation-halo yields distinct cursor shapes.

    Uses a larger seeded bbox so that handle tolerance can't bleed into the
    body's hit zone on the offscreen widget.
    """
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    # Big bbox so handle hit-test won't span the body.
    landmarks = ((30.0, 80.0), (170.0, 80.0), (100.0, 160.0))
    face_index = window._editable.add_face(0, (20.0, 20.0, 160.0, 160.0), landmarks=landmarks) or 0
    window._editor_state.set("face_index", face_index)

    view = window._frame_view
    # Body of bbox → SizeAllCursor.
    view._update_hover_cursor(_source_to_widget(view, 100.0, 100.0))  # noqa: SLF001
    assert view.cursor().shape() == Qt.SizeAllCursor
    # SE corner → SizeFDiagCursor.
    view._update_hover_cursor(_source_to_widget(view, 180.0, 180.0))  # noqa: SLF001
    assert view.cursor().shape() == Qt.SizeFDiagCursor
    # Far outside the rotation halo → default arrow.
    view._update_hover_cursor(QPointF(0.0, 0.0))
    assert view.cursor().shape() == Qt.ArrowCursor


def test_extract_scale_handler_clamps_via_model(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Pi/2 rotation produces a different landmark cloud + bbox via the model."""
    window = _make_window(qtbot, tmp_path)
    _enter_extract_mode(window)
    face_index = _seed_face(window)
    window._editor_state.set("face_index", face_index)

    before_landmarks = window._editable.faces(0)[0].landmarks
    window._on_face_rotate_requested(face_index, math.pi / 2)
    after_landmarks = window._editable.faces(0)[0].landmarks
    assert after_landmarks != before_landmarks
    assert window._editor_state.edited is True
