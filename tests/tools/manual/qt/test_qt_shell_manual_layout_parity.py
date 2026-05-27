#!/usr/bin/env python3
"""Qt Manual Tool legacy shell-layout parity tests (#126)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QToolBar, QToolButton, QWidget

from tools.manual.qt import ManualFrameView, ManualToolWindow
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 2) -> ManualSession:
    """Write small image fixtures and return a Manual Tool session."""
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(96, 64)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    """Return a shown ManualToolWindow with startup drained."""
    window = ManualToolWindow(_session_with_frames(folder))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._startup_worker is None, timeout=5000)  # noqa:SLF001
    return window


def test_default_shell_hides_non_legacy_panels(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The default Qt shell omits the toolbar, metadata, status row and extra panels."""
    window = _make_window(qtbot, tmp_path)

    toolbar = window.findChild(QToolBar, "qt-manual-toolbar")
    assert toolbar is not None
    assert toolbar.isVisible() is False
    assert window.findChild(QLabel, "qt-manual-session-metadata").isVisible() is False
    assert window._status_label.isVisible() is False  # noqa:SLF001
    assert window._thumbnail_panel.isVisible() is False  # noqa:SLF001
    assert window._face_panel.isVisible() is False  # noqa:SLF001
    assert window.findChild(QLabel, "qt-manual-face-grid-size-label") is None


def test_top_editor_area_uses_legacy_rail_canvas_context_shape(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """The top area is left rail, center canvas/transport, and fixed right panel."""
    window = _make_window(qtbot, tmp_path)

    assert window.findChild(QWidget, "qt-manual-editor-area").isVisible() is True
    rail = window.findChild(QWidget, "qt-manual-editor-rail")
    canvas = window.findChild(QWidget, "qt-manual-frame-canvas")
    context = window.findChild(QWidget, "qt-manual-context-panel")
    transport_row = window.findChild(QWidget, "qt-manual-transport-button-row")

    assert rail is not None
    assert rail.width() == 42
    assert canvas is not None
    assert context is not None
    assert context.minimumWidth() >= 240
    assert context.maximumWidth() <= 300
    assert transport_row is not None
    assert window._face_grid_size_combo.parent() is transport_row  # noqa:SLF001


def test_left_rail_contains_editor_and_frame_actions(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Editor modes and copy/revert actions live in the vertical left rail."""
    window = _make_window(qtbot, tmp_path)
    rail = window.findChild(QWidget, "qt-manual-editor-rail")
    assert rail is not None

    for key in (
        "set_view_mode",
        "set_boundingbox_mode",
        "set_extractbox_mode",
        "set_landmarks_mode",
        "set_mask_mode",
        "copy_prev_face",
        "copy_next_face",
        "revert_frame",
    ):
        button = rail.findChild(QToolButton, f"qt-manual-rail-action-{key}")
        assert button is not None, key
        assert button.toolButtonStyle() == Qt.ToolButtonIconOnly


def test_optional_rail_actions_track_active_editor(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Mode-specific rail actions are visible only for the active editor."""
    window = _make_window(qtbot, tmp_path)
    rail = window.findChild(QWidget, "qt-manual-editor-rail")
    assert rail is not None

    zoom = rail.findChild(QToolButton, "qt-manual-rail-action-zoom_in")
    draw = rail.findChild(QToolButton, "qt-manual-rail-action-mask_draw")
    assert zoom is not None
    assert draw is not None

    assert zoom.isVisible() is True
    assert draw.isVisible() is False

    window.set_editor_mask()

    assert zoom.isVisible() is False
    assert draw.isVisible() is True


def test_transport_row_contains_filter_save_extract_and_face_size(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """The second transport row carries navigation, filter, extract/save and face size."""
    window = _make_window(qtbot, tmp_path)
    row = window.findChild(QWidget, "qt-manual-transport-button-row")
    assert row is not None

    for key in ("play_pause", "first_frame", "previous_frame", "next_frame", "last_frame"):
        assert row.findChild(QToolButton, f"qt-manual-rail-action-{key}") is not None
    assert window._filter_combo.parent() is window._filter_controls  # noqa:SLF001
    assert window._filter_threshold_slider.isVisible() is False  # noqa:SLF001
    assert row.findChild(QToolButton, "qt-manual-rail-action-extract_faces") is not None
    assert row.findChild(QToolButton, "qt-manual-rail-action-save") is not None
    assert window._face_grid_size_combo.parent() is row  # noqa:SLF001

    window.editor_state.set("filter_mode", "Misaligned Faces")

    assert window._filter_threshold_slider.isVisible() is True  # noqa:SLF001


def test_frame_view_paints_black_centered_gutters(qtbot) -> None:  # type:ignore[no-untyped-def]
    """The frame canvas uses black gutters and centers the source image."""

    def assert_color_close(actual: QColor, expected: QColor, tolerance: int = 4) -> None:
        assert abs(actual.red() - expected.red()) <= tolerance
        assert abs(actual.green() - expected.green()) <= tolerance
        assert abs(actual.blue() - expected.blue()) <= tolerance
        assert actual.alpha() == expected.alpha()

    view = ManualFrameView()
    qtbot.addWidget(view)
    view.resize(300, 200)
    image = QImage(200, 100, QImage.Format_RGB32)
    image.fill(QColor("#446699"))
    assert view.set_image(image)

    view.show()
    qtbot.waitExposed(view)
    pixmap = view.grab().toImage()

    assert pixmap.pixelColor(2, 2) == QColor("#000000")
    assert_color_close(pixmap.pixelColor(150, 100), QColor("#446699"))
