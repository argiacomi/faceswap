#!/usr/bin/env python3
"""Native Qt Manual Tool frame viewer tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent

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

    press = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(start),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    move = QMouseEvent(
        QEvent.MouseMove,
        QPointF(moved),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    release = QMouseEvent(
        QEvent.MouseButtonRelease,
        QPointF(moved),
        Qt.LeftButton,
        Qt.NoButton,
        Qt.NoModifier,
    )

    view.mousePressEvent(press)
    view.mouseMoveEvent(move)
    view.mouseReleaseEvent(release)

    offset = view.offset
    assert offset.x() != 0.0 or offset.y() != 0.0

    # Drag the view by an absurd amount; offset must remain finite and clamped.
    far_press = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(QPoint(100, 80)),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    far_move = QMouseEvent(
        QEvent.MouseMove,
        QPointF(QPoint(5000, 5000)),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    view.mousePressEvent(far_press)
    view.mouseMoveEvent(far_move)
    view.mouseReleaseEvent(release)
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
    press = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(centre),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    view.mousePressEvent(press)
    release = QMouseEvent(
        QEvent.MouseButtonRelease,
        QPointF(centre),
        Qt.LeftButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    view.mouseReleaseEvent(release)

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


def test_video_image_decoder_handles_empty_array() -> None:
    """The BGR→QImage helper returns a null image for empty/None input."""
    import numpy as np

    from lib.gui.qt_shell.manual_tool import _bgr_array_to_qimage

    assert _bgr_array_to_qimage(None).isNull()
    assert _bgr_array_to_qimage(np.zeros((0, 0, 3), dtype=np.uint8)).isNull()


def test_video_image_decoder_converts_bgr() -> None:
    """A small BGR uint8 array converts to a non-null QImage with matching size."""
    import numpy as np

    from lib.gui.qt_shell.manual_tool import _bgr_array_to_qimage

    frame = np.full((4, 5, 3), 200, dtype=np.uint8)
    image = _bgr_array_to_qimage(frame)
    assert image.isNull() is False
    assert image.width() == 5
    assert image.height() == 4
    assert image.format() == QImage.Format_BGR888
