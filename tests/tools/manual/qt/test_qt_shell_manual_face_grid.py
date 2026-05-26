#!/usr/bin/env python3
"""Cross-frame face grid regressions for the Qt Manual Tool (#108)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QApplication, QToolButton

from lib.align import LANDMARK_PARTS, LandmarkType
from lib.align.constants import MEAN_FACE
from tools.manual.qt import (
    FaceGridThumbnailRenderer,
    ManualToolWindow,
)
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


def test_face_grid_wraps_to_available_width(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The wrapping layout adapts column count to the viewport width."""
    window = _make_window(qtbot, tmp_path, count=8)
    for frame_index in range(8):
        _seed_face(window, frame_index)
    panel = window.face_grid_panel
    window._face_grid_size_combo.setCurrentText("Tiny")

    def _columns_in_first_row() -> int:
        panel.doItemsLayout()
        qtbot.wait(0)
        first_top: int | None = None
        columns = 0
        for row in range(panel.count()):
            item = panel.item(row)
            if item is None:
                continue
            rect = panel.visualItemRect(item)
            if first_top is None:
                first_top = rect.top()
            if rect.top() == first_top:
                columns += 1
        return columns

    panel.setFixedSize(panel.gridSize().width() + 24, 240)
    narrow_columns = _columns_in_first_row()
    panel.setFixedSize((panel.gridSize().width() * 3) + 24, 240)
    wide_columns = _columns_in_first_row()

    assert narrow_columns == 1
    assert wide_columns > narrow_columns


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
    """Focused grid handles list keyboard paging and mouse-wheel scrolling."""
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
    panel.setCurrentRow(0)
    panel.verticalScrollBar().setValue(0)

    qtbot.keyClick(panel, Qt.Key_PageDown)
    page_down_row = panel.currentRow()
    page_down_scroll = panel.verticalScrollBar().value()
    assert page_down_row > 0 or page_down_scroll > 0

    qtbot.keyClick(panel, Qt.Key_PageUp)
    assert (
        panel.currentRow() < page_down_row or panel.verticalScrollBar().value() < page_down_scroll
    )

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


def test_face_grid_right_click_menu_delete_path(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type:ignore[no-untyped-def]
    """Right-clicking a grid thumbnail opens a Delete Face menu wired to deletion."""
    import PySide6.QtWidgets as qtwidgets

    class _FakeSignal:
        def __init__(self) -> None:
            self.callback = None

        def connect(self, callback):  # type:ignore[no-untyped-def]
            self.callback = callback

        def emit(self) -> None:
            assert self.callback is not None
            self.callback()

    class _FakeAction:
        def __init__(self, text: str) -> None:
            self.text = text
            self.triggered = _FakeSignal()

    class _FakeMenu:
        instances = []

        def __init__(self, parent=None) -> None:  # type:ignore[no-untyped-def]
            self.parent = parent
            self.actions: list[_FakeAction] = []
            self.aboutToHide = _FakeSignal()
            self.popup_point = None
            self.__class__.instances.append(self)

        def addAction(self, text: str) -> _FakeAction:  # noqa:N802
            action = _FakeAction(text)
            self.actions.append(action)
            return action

        def popup(self, point) -> None:  # type:ignore[no-untyped-def]
            self.popup_point = point

        def deleteLater(self) -> None:  # noqa:N802
            return None

    monkeypatch.setattr(qtwidgets, "QMenu", _FakeMenu)
    window = _make_window(qtbot, tmp_path, count=4)
    _seed_face(window, 3)
    panel = window.face_grid_panel
    item = panel.item_for(3, 0)
    assert item is not None

    panel.scrollToItem(item)
    qtbot.wait(0)
    panel.customContextMenuRequested.emit(panel.visualItemRect(item).center())

    assert len(_FakeMenu.instances) == 1
    menu = _FakeMenu.instances[0]
    assert menu.actions[0].text == "Delete Face"
    assert menu.popup_point is not None

    menu.actions[0].triggered.emit()

    assert window.editable_alignments.face_count(3) == 0
    assert window.editor_state.unsaved is True
    assert window.face_grid_panel.entry_keys() == ()


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
    jaw = tuple(
        (float(7.0 + index * 1.25), float(26.0 + abs(index - 8) * 0.5)) for index in range(17)
    )
    core = tuple(
        (float(point[0] * 24.0 + 10.0), float(point[1] * 24.0 + 10.0))
        for point in MEAN_FACE[LandmarkType.LM_2D_51]
    )
    landmarks = jaw + core
    _seed_face(window, 0, landmarks=landmarks)
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

    groups = FaceGridThumbnailRenderer.mesh_groups_for_entry(
        window.face_grid_panel.entries[0],
        window.face_grid_panel.iconSize().width(),
    )
    legacy_parts = LANDMARK_PARTS[LandmarkType.LM_2D_68]
    assert len(groups["polygon"]) == sum(
        1 for _name, (_start, _end, fill) in legacy_parts.items() if fill
    )
    assert len(groups["line"]) == sum(
        1 for _name, (_start, _end, fill) in legacy_parts.items() if not fill
    )


def test_face_grid_mini_rail_toggles_mesh_and_mask_independently(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """F9/F10 and the mini rail expose independent thumbnail annotation toggles."""
    window = _make_window(qtbot, tmp_path, count=2)
    _seed_face(window, 0)
    mesh_button = window.findChild(QToolButton, "qt-manual-face-grid-mesh-toggle")
    mask_button = window.findChild(QToolButton, "qt-manual-face-grid-mask-toggle")

    assert mesh_button is not None
    assert mask_button is not None

    window.toggle_mesh_annotation()
    mesh_request = window.face_grid_panel.render_requests()[0]
    assert mesh_button.isChecked() is True
    assert mask_button.isChecked() is False
    assert mesh_request.show_mesh is True
    assert mesh_request.show_mask is False

    window.toggle_mask_annotation()
    both_request = window.face_grid_panel.render_requests()[0]
    assert mesh_button.isChecked() is True
    assert mask_button.isChecked() is True
    assert both_request.show_mesh is True
    assert both_request.show_mask is True

    window.toggle_mesh_annotation()
    mask_only_request = window.face_grid_panel.render_requests()[0]
    assert mesh_button.isChecked() is False
    assert mask_button.isChecked() is True
    assert mask_only_request.show_mesh is False
    assert mask_only_request.show_mask is True


def test_face_grid_delete_key_removes_multi_selection_across_frames(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Extended grid selection deletes all selected faces as one undoable edit."""
    window = _make_window(qtbot, tmp_path, count=4)
    _seed_face(window, 0)
    _seed_face(window, 1)
    _seed_face(window, 1)
    _seed_face(window, 3)
    panel = window.face_grid_panel
    selected = (panel.item_for(0, 0), panel.item_for(1, 1), panel.item_for(3, 0))
    assert all(item is not None for item in selected)
    panel.setCurrentItem(selected[0])
    for item in selected:
        panel.selectionModel().select(
            panel.indexFromItem(item),
            QItemSelectionModel.Select,
        )
    assert panel.selected_entry_keys() == ((0, 0), (1, 1), (3, 0))
    panel.setFocus(Qt.OtherFocusReason)

    panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))

    assert window.editable_alignments.face_count(0) == 0
    assert window.editable_alignments.face_count(1) == 1
    assert window.editable_alignments.face_count(3) == 0
    assert window.face_grid_panel.entry_keys() == ((1, 0),)
    assert window.editor_state.unsaved is True

    assert window.undo_edit() is True
    assert window.editable_alignments.face_count(0) == 1
    assert window.editable_alignments.face_count(1) == 2
    assert window.editable_alignments.face_count(3) == 1


def test_face_grid_refreshes_after_non_current_edit_with_filter(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Non-current edits refresh filtered results and grid contents without navigating."""
    window = _make_window(qtbot, tmp_path, count=4)
    window._thumbnail_panel.setCurrentRow(0)
    window.editor_state.set("filter_mode", "Has Face(s)")
    assert window.face_grid_panel.entry_keys() == ()

    _seed_face(window, 2)

    assert window.filtered_frame_indices() == (2,)
    assert window._thumbnail_panel.currentRow() == 0
    assert window.face_grid_panel.entry_keys() == ((2, 0),)
