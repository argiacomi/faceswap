#!/usr/bin/env python3
"""Cross-frame face grid regressions for the Qt Manual Tool (#108)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QPixmap, QWheelEvent
from PySide6.QtWidgets import QApplication

from lib.gui.qt_shell.manual_tool import ManualToolWindow
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 5) -> ManualSession:
    """Create a small image-folder Manual session with ``count`` frames."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(96, 72)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path, count: int = 5) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    """Return a shown ManualToolWindow with startup fully drained."""
    window = ManualToolWindow(_session_with_frames(folder, count=count))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._startup_worker is None, timeout=5000)
    return window


def _seed_face(
    window: ManualToolWindow,
    frame_index: int,
    *,
    landmarks: tuple[tuple[float, float], ...] = (),
) -> int:
    """Seed one editable face on ``frame_index``."""
    face_offset = window.editable_alignments.face_count(frame_index) * 4.0
    return window.editable_alignments.add_face(
        frame_index,
        (10.0 + face_offset, 10.0, 24.0, 24.0),
        landmarks=landmarks,
    )


def test_face_grid_lists_faces_across_filtered_session(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The grid displays every visible face, not only current-frame faces."""
    window = _make_window(qtbot, tmp_path, count=5)

    _seed_face(window, 0)
    _seed_face(window, 2)
    _seed_face(window, 4)

    assert window.face_grid_panel.entry_keys() == ((0, 0), (2, 0), (4, 0))
    assert window.face_grid_panel.count() == 3


def test_face_grid_respects_active_filter(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Grid contents are derived from ``filtered_frame_indices()``."""
    window = _make_window(qtbot, tmp_path, count=5)
    _seed_face(window, 1)
    _seed_face(window, 3)
    _seed_face(window, 3)

    window.editor_state.set("filter_mode", "Has Face(s)")
    assert window.filtered_frame_indices() == (1, 3)
    assert window.face_grid_panel.entry_keys() == ((1, 0), (3, 0), (3, 1))

    window.editor_state.set("filter_mode", "Single Face")
    assert window.face_grid_panel.entry_keys() == ((1, 0),)

    window.editor_state.set("filter_mode", "Multiple Faces")
    assert window.face_grid_panel.entry_keys() == ((3, 0), (3, 1))

    window.editor_state.set("filter_mode", "No Faces")
    assert window.face_grid_panel.entry_keys() == ()
    assert window.face_grid_panel.item(0).text() == "No faces match current filter"


def test_face_grid_size_control_updates_layout_and_preserves_active(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Legacy face-size choices resize icons and keep the active face selected."""
    window = _make_window(qtbot, tmp_path, count=3)
    _seed_face(window, 2)
    window._on_face_grid_activated(2, 0)
    old_icon = window.face_grid_panel.iconSize()
    old_grid = window.face_grid_panel.gridSize()

    window._face_grid_size_combo.setCurrentText("Extra Large")

    assert window.editor_state.faces_size == "Extra Large"
    assert window.face_grid_panel.iconSize().width() > old_icon.width()
    assert window.face_grid_panel.gridSize().width() > old_grid.width()
    assert window.face_grid_panel.active_entry() == (2, 0)


def test_face_grid_click_navigates_frame_and_face(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Clicking a grid thumbnail loads its source frame and selects the face."""
    window = _make_window(qtbot, tmp_path, count=4)
    _seed_face(window, 0)
    _seed_face(window, 3)
    _seed_face(window, 3)
    panel = window.face_grid_panel
    item = panel.item_for(3, 1)
    assert item is not None

    panel.scrollToItem(item)
    qtbot.wait(0)
    qtbot.mouseClick(panel.viewport(), Qt.LeftButton, pos=panel.visualItemRect(item).center())

    assert window.editor_state.frame_index == 3
    assert window._thumbnail_panel.currentRow() == 3
    assert window.editor_state.face_index == 1
    assert panel.active_entry() == (3, 1)


def test_face_grid_active_and_hover_state_are_testable(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Active frame/face and hover state are exposed through item state."""
    window = _make_window(qtbot, tmp_path, count=4)
    _seed_face(window, 2)
    _seed_face(window, 2)
    panel = window.face_grid_panel

    window._on_face_grid_activated(2, 1)
    active_item = panel.item_for(2, 1)
    assert active_item is not None
    assert panel.active_entry() == (2, 1)
    assert active_item.isSelected() is True

    hover_item = panel.item_for(2, 0)
    assert hover_item is not None
    panel.scrollToItem(hover_item)
    qtbot.wait(0)
    qtbot.mouseMove(panel.viewport(), panel.visualItemRect(hover_item).center())

    assert panel.hovered_entry() == (2, 0)
    assert panel.item_for(2, 0).data(Qt.UserRole + 3) is True
    assert window.statusBar().currentMessage() == "Frame 3 / Face 1"


def test_face_grid_keyboard_and_mouse_wheel_scroll(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Focused grid handles list keyboard navigation and mouse-wheel scrolling."""
    window = _make_window(qtbot, tmp_path, count=18)
    for frame_index in range(18):
        _seed_face(window, frame_index)
        _seed_face(window, frame_index)
    panel = window.face_grid_panel
    panel.setFixedSize(150, 110)
    window._face_grid_size_combo.setCurrentText("Extra Large")
    panel.setFocus(Qt.OtherFocusReason)
    qtbot.waitUntil(lambda: panel.verticalScrollBar().maximum() > 0, timeout=3000)

    assert panel.currentRow() <= 0
    qtbot.keyClick(panel, Qt.Key_Down)
    assert panel.currentRow() > 0

    panel.verticalScrollBar().setValue(0)
    wheel_event = QWheelEvent(
        QPointF(20.0, 20.0),
        QPointF(panel.viewport().mapToGlobal(QPoint(20, 20))),
        QPoint(0, 0),
        QPoint(0, -240),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(panel.viewport(), wheel_event)
    assert panel.verticalScrollBar().value() > 0


def test_face_grid_delete_removes_non_current_face_and_refreshes(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Grid deletion works for a face outside the current frame."""
    window = _make_window(qtbot, tmp_path, count=4)
    _seed_face(window, 0)
    _seed_face(window, 3)
    window._thumbnail_panel.setCurrentRow(0)

    window._delete_face_at(3, 0)

    assert window.editable_alignments.face_count(3) == 0
    assert window.editor_state.unsaved is True
    assert (3, 0) not in window.face_grid_panel.entry_keys()


def test_face_grid_overlay_render_requests_track_mask_and_mesh(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Thumbnail render state carries F9/F10 mesh and mask overlay settings."""
    window = _make_window(qtbot, tmp_path, count=2)
    _seed_face(window, 0, landmarks=((10.0, 10.0), (18.0, 18.0), (26.0, 10.0)))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    window.editable_alignments._mask_state[(0, 0, "components")] = mask

    window.editor_state.set("mask_opacity", 40)
    window.editor_state.set("annotation_mode", "Mask")
    mask_request = window.face_grid_panel.render_requests()[0]

    assert mask_request.show_mask is True
    assert mask_request.mask_type == "components"
    assert mask_request.mask_opacity == 40

    window.editor_state.set("annotation_mode", "Mesh")
    mesh_request = window.face_grid_panel.render_requests()[0]

    assert mesh_request.show_mesh is True
    assert mesh_request.show_mask is False


def test_face_grid_refreshes_after_non_current_edit_with_filter(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Non-current edits refresh both filtered navigation and grid contents."""
    window = _make_window(qtbot, tmp_path, count=4)
    window._thumbnail_panel.setCurrentRow(0)
    window.editor_state.set("filter_mode", "Has Face(s)")
    assert window.face_grid_panel.entry_keys() == ()

    _seed_face(window, 2)

    assert window.filtered_frame_indices() == (2,)
    assert window._thumbnail_panel.currentRow() == 2
    assert window.face_grid_panel.entry_keys() == ((2, 0),)
