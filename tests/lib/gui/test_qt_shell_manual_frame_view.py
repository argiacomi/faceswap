#!/usr/bin/env python3
"""Native Qt Manual Tool frame viewer tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtTest import QTest

from lib.gui.qt_shell.manual_tool import FrameViewport, ManualFrameView, ManualToolWindow
from tools.manual.session import ManualFrame, ManualSession


def _write_fixture_frame(path: Path, color: str = "blue", size: int = 32) -> ManualFrame:
    """Write a PNG fixture to ``path`` and return a ManualFrame metadata entry."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    assert pixmap.save(str(path), "PNG"), f"Could not write fixture: {path}"
    return ManualFrame(index=0, name=path.name, path=str(path))


def _session_with_frames(folder: Path, count: int = 3) -> ManualSession:
    """Write ``count`` colored PNG fixtures into ``folder`` and return a session."""
    colors = ["#ff5544", "#44ff88", "#5588ff", "#ffaa00", "#aa00ff"]
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(48, 32)
        pixmap.fill(QColor(colors[index % len(colors)]))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _mouse_event(
    event_type: QEvent.Type,
    view: ManualFrameView,
    pos: QPointF,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> QMouseEvent:
    """Return a non-deprecated synthetic mouse event for direct event-handler tests.

    Most tests use :mod:`QTest` helpers, but the bbox editor drag tests call the
    widget handlers directly so they can inspect emitted model signals.  PySide6
    deprecates the shorter QMouseEvent constructor; this helper supplies local,
    scene and global positions explicitly.
    """
    global_pos = QPointF(view.mapToGlobal(pos.toPoint()))
    return QMouseEvent(event_type, pos, pos, global_pos, button, buttons, Qt.NoModifier)


def test_frame_view_loads_fixture_image(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """ManualFrameView loads a fixture frame and reports its source size."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 160)
    frame = _write_fixture_frame(tmp_path / "frame.png", color="#00ff00", size=24)

    captured: list[ManualFrame] = []
    view.frame_loaded.connect(captured.append)

    assert view.load_frame(frame) is True
    assert view.has_frame is True
    assert view.source_size == (24, 24)
    assert captured == [frame]


def test_frame_view_zoom_clamped_to_bounds(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Zoom in/out are clamped between 1x and 20x."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 160)
    view.load_frame(_write_fixture_frame(tmp_path / "f.png"))

    for _ in range(40):
        view.zoom_in()
    assert view.zoom == pytest.approx(20.0)

    for _ in range(40):
        view.zoom_out()
    assert view.zoom == pytest.approx(1.0)


def test_frame_view_pan_clamped_while_zoomed(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Drag-pan while zoomed updates the offset and stays within the clamped margin."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 160)
    view.load_frame(_write_fixture_frame(tmp_path / "f.png"))
    view.zoom_in()
    view.zoom_in()
    view.zoom_in()
    assert view.zoom > 1.0

    start = QPoint(100, 80)
    moved = QPoint(180, 130)

    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(view, moved)
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, moved)

    offset = view.offset
    assert offset.x() != 0.0 or offset.y() != 0.0

    # Drag the view by an absurd amount; offset must remain finite and clamped.
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, QPoint(100, 80))
    QTest.mouseMove(view, QPoint(5000, 5000))
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, QPoint(5000, 5000))
    clamped = view.offset

    # Pan offset never exceeds the widget dimensions when zoomed; this guards the clamp.
    assert abs(clamped.x()) <= view.width()
    assert abs(clamped.y()) <= view.height()


def test_frame_view_wheel_zooms(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Mouse wheel scrolls zoom in/out within bounds."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 160)
    view.load_frame(_write_fixture_frame(tmp_path / "f.png"))

    wheel_up = QWheelEvent(
        QPointF(100, 80),
        QPointF(100, 80),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    view.wheelEvent(wheel_up)
    assert view.zoom > 1.0
    previous = view.zoom

    wheel_down = QWheelEvent(
        QPointF(100, 80),
        QPointF(100, 80),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    view.wheelEvent(wheel_down)
    assert view.zoom < previous


def test_frame_view_reset_view_returns_to_baseline(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Reset view restores zoom=1 and zero offset."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 160)
    view.load_frame(_write_fixture_frame(tmp_path / "f.png"))
    view.zoom_in()
    view.zoom_in()
    view.reset_view()
    assert view.zoom == pytest.approx(1.0)
    assert view.offset.x() == pytest.approx(0.0)
    assert view.offset.y() == pytest.approx(0.0)


def test_frame_view_clear_frame_resets_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """clear_frame drops the pixmap and shows the empty message."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 160)
    view.load_frame(_write_fixture_frame(tmp_path / "f.png"))
    view.clear_frame("cleared")
    assert view.has_frame is False
    assert view.zoom == pytest.approx(1.0)
    assert view.source_size == (0, 0)


def test_frame_view_overlay_invoked(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Overlay callables receive the painter and viewport during paint."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 160)
    view.load_frame(_write_fixture_frame(tmp_path / "f.png"))

    received: list[FrameViewport] = []

    def overlay(_painter: QPainter, viewport: FrameViewport) -> None:
        received.append(viewport)

    remove = view.add_overlay(overlay)

    # grab() returns a QPixmap of the widget and drives paintEvent deterministically.
    view.grab()

    assert received, "Overlay was not invoked"
    last = received[-1]
    assert last.source_size == view.source_size
    assert last.zoom == pytest.approx(view.zoom)
    assert last.target_rect.width() > 0

    remove()
    received.clear()
    view.grab()
    assert received == [], "Overlay still invoked after removal"


def test_frame_view_click_emits_source_coordinates(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Left-click inside the painted frame emits source-image coordinates."""
    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(200, 200)
    view.load_frame(_write_fixture_frame(tmp_path / "f.png", size=32))

    captured: list[QPointF] = []
    view.clicked_at.connect(captured.append)

    # Click roughly at center → expect coordinates near the middle of the source image.
    centre = QPoint(view.width() // 2, view.height() // 2)
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, centre)
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, centre)

    assert len(captured) == 1
    source_point = captured[0]
    assert 12 <= source_point.x() <= 20
    assert 12 <= source_point.y() <= 20


def test_manual_tool_window_navigation_emits_frame_changed(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path
) -> None:
    """ManualToolWindow advances frames and emits frame_changed."""
    session = _session_with_frames(tmp_path, count=3)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    captured: list[int] = []
    window.frame_changed.connect(captured.append)

    # First selection happens automatically on load; clear and navigate explicitly.
    captured.clear()
    window._next_frame()  # noqa: SLF001 - exercise navigation slot
    window._next_frame()  # noqa: SLF001
    window._previous_frame()  # noqa: SLF001

    assert captured == [1, 2, 1]
    assert window.frame_view.has_frame is True


def test_frame_view_drag_inside_bbox_emits_move(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Dragging from inside an active bbox emits face_move_requested on release."""
    from PySide6.QtCore import QRectF

    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(100, 100)
    view.show()

    # Load a 100x100 fixture so source coords map 1:1 to widget coords.
    fixture = tmp_path / "f.png"
    pixmap = QPixmap(100, 100)
    pixmap.fill(QColor("#888"))
    assert pixmap.save(str(fixture), "PNG")
    view.load_frame(ManualFrame(index=0, name=fixture.name, path=str(fixture)))

    view.install_editor_seams(
        active_face_provider=lambda: 0,
        active_bbox_provider=lambda: QRectF(20.0, 20.0, 30.0, 30.0),
    )

    moves: list[tuple[int, float, float]] = []
    view.face_move_requested.connect(lambda idx, dx, dy: moves.append((idx, dx, dy)))

    # Press inside the bbox at (30, 30), drag to (50, 40), release.
    target = view._target_rect()  # internal helper — fixture maps 1:1 once shown
    start_widget = QPointF(target.x() + 30.0, target.y() + 30.0)
    end_widget = QPointF(target.x() + 50.0, target.y() + 40.0)
    press = _mouse_event(
        QEvent.Type.MouseButtonPress,
        view,
        start_widget,
        Qt.LeftButton,
        Qt.LeftButton,
    )
    move = _mouse_event(
        QEvent.Type.MouseMove,
        view,
        end_widget,
        Qt.NoButton,
        Qt.LeftButton,
    )
    release = _mouse_event(
        QEvent.Type.MouseButtonRelease,
        view,
        end_widget,
        Qt.LeftButton,
        Qt.NoButton,
    )
    view.mousePressEvent(press)
    view.mouseMoveEvent(move)
    view.mouseReleaseEvent(release)

    assert len(moves) == 1
    face_index, dx, dy = moves[0]
    assert face_index == 0
    # Source-to-widget is 1:1 because target_rect spans full widget size.
    assert dx == pytest.approx(20.0, abs=0.5)
    assert dy == pytest.approx(10.0, abs=0.5)


def test_frame_view_drag_se_handle_emits_resize(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Dragging the SE handle emits face_resize_requested with grown bbox."""
    from PySide6.QtCore import QRectF

    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(100, 100)
    view.show()

    fixture = tmp_path / "f.png"
    pixmap = QPixmap(100, 100)
    pixmap.fill(QColor("#888"))
    assert pixmap.save(str(fixture), "PNG")
    view.load_frame(ManualFrame(index=0, name=fixture.name, path=str(fixture)))
    view.install_editor_seams(
        active_face_provider=lambda: 0,
        active_bbox_provider=lambda: QRectF(20.0, 20.0, 30.0, 30.0),
    )

    captured: list[tuple[int, QRectF]] = []
    view.face_resize_requested.connect(lambda idx, rect: captured.append((idx, QRectF(rect))))

    target = view._target_rect()
    # SE corner of source bbox at (50,50) — also widget coords here.
    start_widget = QPointF(target.x() + 50.0, target.y() + 50.0)
    end_widget = QPointF(target.x() + 70.0, target.y() + 65.0)
    press = _mouse_event(
        QEvent.Type.MouseButtonPress,
        view,
        start_widget,
        Qt.LeftButton,
        Qt.LeftButton,
    )
    move = _mouse_event(
        QEvent.Type.MouseMove,
        view,
        end_widget,
        Qt.NoButton,
        Qt.LeftButton,
    )
    release = _mouse_event(
        QEvent.Type.MouseButtonRelease,
        view,
        end_widget,
        Qt.LeftButton,
        Qt.NoButton,
    )
    view.mousePressEvent(press)
    view.mouseMoveEvent(move)
    view.mouseReleaseEvent(release)

    assert len(captured) == 1
    idx, rect = captured[0]
    assert idx == 0
    # Bbox should have grown by ~20px wide and 15px tall, anchored at (20,20).
    assert rect.x() == pytest.approx(20.0, abs=0.5)
    assert rect.y() == pytest.approx(20.0, abs=0.5)
    assert rect.width() == pytest.approx(50.0, abs=0.5)
    assert rect.height() == pytest.approx(45.0, abs=0.5)


def test_frame_view_hover_cursor_switches_on_handle(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Hovering an active-face resize handle yields a diagonal sizing cursor."""
    from PySide6.QtCore import QRectF

    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(100, 100)
    view.show()

    fixture = tmp_path / "f.png"
    pixmap = QPixmap(100, 100)
    pixmap.fill(QColor("#888"))
    assert pixmap.save(str(fixture), "PNG")
    view.load_frame(ManualFrame(index=0, name=fixture.name, path=str(fixture)))
    view.install_editor_seams(
        active_face_provider=lambda: 0,
        active_bbox_provider=lambda: QRectF(20.0, 20.0, 30.0, 30.0),
    )

    target = view._target_rect()
    # Hover SE corner -> SizeFDiagCursor; inside body -> SizeAllCursor.
    view._update_hover_cursor(QPointF(target.x() + 50.0, target.y() + 50.0))
    assert view.cursor().shape() == Qt.SizeFDiagCursor
    view._update_hover_cursor(QPointF(target.x() + 30.0, target.y() + 30.0))
    assert view.cursor().shape() == Qt.SizeAllCursor


def test_manual_window_arrow_key_nudges_active_face(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The Up arrow keyboard shortcut translates the active face one source pixel."""
    session = _session_with_frames(tmp_path, count=1)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    window.editable_alignments.add_face(0, (50.0, 50.0, 30.0, 30.0))
    window.editor_state.set("face_index", 0)

    assert window.nudge_up_one() is True
    face = window.editable_alignments.faces(0)[0]
    assert face.bbox == (50.0, 49.0, 30.0, 30.0)

    assert window.nudge_right_one() is True
    assert window.editable_alignments.faces(0)[0].bbox == (51.0, 49.0, 30.0, 30.0)

    assert window.nudge_down_fast() is True
    assert window.editable_alignments.faces(0)[0].bbox == (51.0, 59.0, 30.0, 30.0)
