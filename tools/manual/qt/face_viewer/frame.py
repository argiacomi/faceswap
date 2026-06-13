#!/usr/bin/env python3
"""Cross-frame face grid widget for the Qt Manual Tool."""

from __future__ import annotations

import logging
import typing as T

from PySide6.QtCore import (
    QPoint,
    QPointF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QKeyEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

logger = logging.getLogger(__name__)
from .viewport import (
    _FACE_GRID_ACTIVE_FACE_ROLE,
    _FACE_GRID_ACTIVE_FRAME_ROLE,
    _FACE_GRID_ENTRY_ROLE,
    _FACE_GRID_HOVER_ROLE,
    _FACE_GRID_SIZES,
    FaceGridEntry,
    FaceGridRenderRequest,
    FaceGridThumbnailRenderer,
)


class CrossFrameFaceGridPanel(QListWidget):
    """Scrollable, wrapping grid of visible faces across the filtered session."""

    face_activated = Signal(int, int)
    """Emits ``(frame_index, face_index)`` when a thumbnail is clicked/accepted."""
    face_hovered = Signal(int, int)
    """Emits ``(frame_index, face_index)`` when hover moves onto a face."""
    face_context_menu_requested = Signal(int, int, QPointF)
    """Emits ``(frame_index, face_index, global_pos)`` for right-click menus."""
    face_delete_requested = Signal(int, int)
    """Emits ``(frame_index, face_index)`` when Delete is pressed in the grid."""
    faces_delete_requested = Signal(object)
    """Emits ordered ``((frame_index, face_index), ...)`` for multi-delete."""

    def __init__(
        self,
        renderer: FaceGridThumbnailRenderer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-cross-frame-face-grid")
        self.setFlow(QListView.LeftToRight)
        self.setViewMode(QListView.IconMode)
        self.setMovement(QListView.Static)
        self.setResizeMode(QListView.Adjust)
        self.setWrapping(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setFocusPolicy(Qt.StrongFocus)  # type: ignore[attr-defined]
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setSpacing(1)
        self.setMinimumHeight(96)
        self.setUniformItemSizes(True)
        self._renderer = renderer
        self._entries: tuple[FaceGridEntry, ...] = ()
        self._face_size_name = "Medium"
        self._active: tuple[int, int] | None = None
        self._hovered: tuple[int, int] | None = None
        self._show_mesh = False
        self._show_mask = False
        self._mask_type = ""
        self._mask_opacity = 50
        self._render_requests: tuple[FaceGridRenderRequest, ...] = ()
        self.itemClicked.connect(self._on_item_clicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self.customContextMenuRequested.connect(self._on_context_menu_requested)
        self.set_face_size(self._face_size_name)

    @property
    def entries(self) -> tuple[FaceGridEntry, ...]:
        """Return the currently visible grid entries."""
        return self._entries

    def entry_keys(self) -> tuple[tuple[int, int], ...]:
        """Return ``(frame_index, face_index)`` keys for all real entries."""
        return tuple((entry.frame_index, entry.face_index) for entry in self._entries)

    def render_requests(self) -> tuple[FaceGridRenderRequest, ...]:
        """Return the render-state records from the most recent population."""
        return self._render_requests

    def active_entry(self) -> tuple[int, int] | None:
        """Return the currently active ``(frame, face)`` pair."""
        return self._active

    def hovered_entry(self) -> tuple[int, int] | None:
        """Return the currently hovered ``(frame, face)`` pair."""
        return self._hovered

    def selected_entry_keys(self) -> tuple[tuple[int, int], ...]:
        """Return selected real entries in visible grid order."""
        selected = set()
        for item in self.selectedItems():
            entry = item.data(_FACE_GRID_ENTRY_ROLE)
            if isinstance(entry, FaceGridEntry):
                selected.add((entry.frame_index, entry.face_index))
        return tuple(key for key in self.entry_keys() if key in selected)

    def face_size_name(self) -> str:
        """Return the active legacy size label."""
        return self._face_size_name

    def set_face_size(self, size_name: str) -> None:
        """Apply one legacy thumbnail size and relayout the wrapping grid."""
        if size_name not in _FACE_GRID_SIZES:
            size_name = "Medium"
        self._face_size_name = size_name
        size = _FACE_GRID_SIZES[size_name]
        self.setIconSize(QSize(size, size))
        self.setGridSize(QSize(size + 4, size + 4))
        for row in range(self.count()):
            item = self.item(row)
            if item is not None:
                item.setSizeHint(self.gridSize())

    def set_overlay_state(
        self,
        *,
        show_mesh: bool,
        show_mask: bool,
        mask_type: str,
        mask_opacity: int,
    ) -> None:
        """Set annotation flags used for the next thumbnail render."""
        self._show_mesh = bool(show_mesh)
        self._show_mask = bool(show_mask)
        self._mask_type = str(mask_type)
        self._mask_opacity = int(mask_opacity)

    def set_entries(self, entries: T.Iterable[FaceGridEntry]) -> None:
        """Populate the grid with one item per visible face."""
        current = self._active
        self._entries = tuple(entries)
        requests: list[FaceGridRenderRequest] = []
        self.blockSignals(True)
        try:
            self.clear()
            if not self._entries:
                item = QListWidgetItem("No faces match current filter")
                item.setFlags(Qt.NoItemFlags)  # type: ignore[attr-defined]
                item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                self.addItem(item)
                self._render_requests = ()
                return
            icon_size = self.iconSize().width()
            for entry in self._entries:
                item = QListWidgetItem()
                item.setToolTip(f"Frame {entry.frame_index + 1} / Face {entry.face_index + 1}")
                item.setData(_FACE_GRID_ENTRY_ROLE, entry)
                item.setSizeHint(self.gridSize())
                request = FaceGridRenderRequest(
                    frame_index=entry.frame_index,
                    face_index=entry.face_index,
                    icon_size=icon_size,
                    show_mesh=self._show_mesh,
                    show_mask=self._show_mask,
                    mask_type=self._mask_type,
                    mask_opacity=self._mask_opacity,
                    is_misaligned=entry.is_misaligned,
                )
                requests.append(request)
                item.setIcon(
                    self._renderer.render(
                        entry,
                        icon_size=icon_size,
                        show_mesh=self._show_mesh,
                        show_mask=self._show_mask,
                        mask_type=self._mask_type,
                        mask_opacity=self._mask_opacity,
                    )
                )
                self.addItem(item)
        finally:
            self._render_requests = tuple(requests)
            self.blockSignals(False)
        if current is not None:
            self.select_face(*current)
        self._apply_item_states()

    def item_for(self, frame_index: int, face_index: int) -> QListWidgetItem | None:
        """Return the item matching ``frame_index`` and ``face_index``."""
        for row in range(self.count()):
            item = self.item(row)
            entry = item.data(_FACE_GRID_ENTRY_ROLE) if item is not None else None
            if (
                isinstance(entry, FaceGridEntry)
                and entry.frame_index == frame_index
                and entry.face_index == face_index
            ):
                return item  # type: ignore[no-any-return]
        return None

    def select_face(self, frame_index: int, face_index: int) -> bool:
        """Select the item matching ``frame_index`` and ``face_index``."""
        item = self.item_for(frame_index, face_index)
        if item is None:
            return False
        self.setCurrentItem(item)
        self.scrollToItem(item, QAbstractItemView.PositionAtCenter)
        return True

    def set_active(self, frame_index: int, face_index: int) -> None:
        """Mark the active frame/face for visible grid highlighting."""
        active = (
            (int(frame_index), int(face_index)) if frame_index >= 0 and face_index >= 0 else None
        )
        self._active = active
        if active is not None:
            self.select_face(*active)
        self._apply_item_states()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        entry = item.data(_FACE_GRID_ENTRY_ROLE)
        if isinstance(entry, FaceGridEntry):
            self.face_activated.emit(entry.frame_index, entry.face_index)

    def _on_context_menu_requested(self, position: QPoint) -> None:
        item = self.itemAt(position)
        entry = item.data(_FACE_GRID_ENTRY_ROLE) if item is not None else None
        if isinstance(entry, FaceGridEntry):
            self.face_context_menu_requested.emit(
                entry.frame_index,
                entry.face_index,
                QPointF(self.mapToGlobal(position)),
            )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa:N802
        item = self.itemAt(event.position().toPoint())
        entry = item.data(_FACE_GRID_ENTRY_ROLE) if item is not None else None
        hovered = (
            (entry.frame_index, entry.face_index) if isinstance(entry, FaceGridEntry) else None
        )
        if hovered != self._hovered:
            self._hovered = hovered
            self._apply_item_states()
            if hovered is not None:
                self.face_hovered.emit(*hovered)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: T.Any) -> None:  # noqa:N802
        self._hovered = None
        self._apply_item_states()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa:N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # type: ignore[attr-defined]
            entry = self._current_entry()
            if entry is not None:
                self.face_activated.emit(entry.frame_index, entry.face_index)
                event.accept()
                return
        if event.key() == Qt.Key_Delete:  # type: ignore[attr-defined]
            selected = self.selected_entry_keys()
            if selected:
                self.faces_delete_requested.emit(selected)
                event.accept()
                return
            entry = self._current_entry()
            if entry is not None:
                self.face_delete_requested.emit(entry.frame_index, entry.face_index)
                event.accept()
                return
        super().keyPressEvent(event)

    def _current_entry(self) -> FaceGridEntry | None:
        item = self.currentItem()
        entry = item.data(_FACE_GRID_ENTRY_ROLE) if item is not None else None
        return entry if isinstance(entry, FaceGridEntry) else None

    def _apply_item_states(self) -> None:
        for row in range(self.count()):
            item = self.item(row)
            if item is None:
                continue
            entry = item.data(_FACE_GRID_ENTRY_ROLE)
            if not isinstance(entry, FaceGridEntry):
                continue
            key = (entry.frame_index, entry.face_index)
            is_active_face = self._active == key
            is_active_frame = self._active is not None and self._active[0] == entry.frame_index
            is_hovered = self._hovered == key
            item.setData(_FACE_GRID_ACTIVE_FRAME_ROLE, is_active_frame)
            item.setData(_FACE_GRID_ACTIVE_FACE_ROLE, is_active_face)
            item.setData(_FACE_GRID_HOVER_ROLE, is_hovered)
            if is_active_face:
                item.setSelected(True)
                item.setBackground(QBrush(QColor("#3f4f5c")))
                item.setForeground(QBrush(QColor("#ffffff")))
            elif is_hovered:
                item.setBackground(QBrush(QColor("#34414a")))
                item.setForeground(QBrush(QColor("#ffffff")))
            elif is_active_frame:
                item.setBackground(QBrush(QColor("#27323a")))
                item.setForeground(QBrush(QColor("#d7e3ea")))
            else:
                item.setBackground(QBrush())
                item.setForeground(QBrush())
