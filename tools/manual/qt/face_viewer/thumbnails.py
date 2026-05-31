#!/usr/bin/env python3
"""Frame and face thumbnail widgets for the Qt Manual Tool face viewer."""

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
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QListView,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

from tools.manual.session import (
    FaceThumbnail,
    ManualFrame,
)

logger = logging.getLogger(__name__)


class ManualThumbnailPanel(QListWidget):
    """Frame thumbnail/selection panel placeholder for Qt Manual Tool."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-thumbnail-panel")
        self.setMinimumSize(0, 96)

    def set_frames(self, frames: tuple[ManualFrame, ...]) -> None:
        """Display selectable entries for available frames."""
        self.clear()
        if not frames:
            item = QListWidgetItem("No image frames available yet")
            item.setFlags(Qt.NoItemFlags)  # type: ignore[attr-defined]
            self.addItem(item)
            return
        for frame in frames:
            item = QListWidgetItem(f"{frame.index + 1}: {frame.name}")
            item.setData(Qt.UserRole, frame.index)  # type: ignore[attr-defined]
            self.addItem(item)


def _decode_jpeg_to_qimage(payload: bytes) -> QImage:
    """Decode a JPEG byte string into a QImage. Returns a null image on failure."""
    if not payload:
        return QImage()
    image = QImage()
    if not image.loadFromData(payload, "JPG"):  # type: ignore[arg-type]
        # Some thumbnails are JPEG-encoded but without a JPG hint; let Qt sniff.
        image = QImage()
        if not image.loadFromData(payload):
            return QImage()
    return image


class FaceThumbnailPanel(QListWidget):
    """Detected-face thumbnail panel for the currently selected frame.

    Renders the JPEG thumbnails stored in the Manual Tool alignments file as a
    horizontally-scrollable strip of selectable items. The active selection is
    surfaced through :attr:`face_selected` for the host window to wire into the
    shared editor state.
    """

    face_selected = Signal(int)
    """Emits the face_index of the currently active entry, or -1 when cleared."""
    face_context_menu_requested = Signal(int, QPointF)
    """Emits ``(face_index, global_pos)`` when a face item is right-clicked."""

    _ICON_SIZE = 72

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-face-thumbnail-panel")
        self.setFlow(QListView.LeftToRight)
        self.setViewMode(QListView.IconMode)
        self.setMovement(QListView.Static)
        self.setResizeMode(QListView.Adjust)
        self.setWrapping(False)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setIconSize(QSize(self._ICON_SIZE, self._ICON_SIZE))
        self.setSpacing(6)
        self.setMinimumHeight(self._ICON_SIZE + 36)
        self.setUniformItemSizes(True)
        self._faces: tuple[FaceThumbnail, ...] = ()
        self._placeholder = self._build_placeholder_icon()
        self.currentRowChanged.connect(self._on_row_changed)
        self.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self.customContextMenuRequested.connect(self._on_context_menu_requested)

    def _on_context_menu_requested(self, position: QPoint) -> None:
        """Emit ``face_context_menu_requested`` for the right-clicked item."""
        item = self.itemAt(position)
        if item is None:
            return
        face_index = item.data(Qt.UserRole)  # type: ignore[attr-defined]
        if not isinstance(face_index, int):
            return
        self.face_context_menu_requested.emit(face_index, QPointF(self.mapToGlobal(position)))

    @property
    def faces(self) -> tuple[FaceThumbnail, ...]:
        """Return the currently displayed face entries."""
        return self._faces

    def set_faces(self, faces: T.Iterable[FaceThumbnail]) -> None:
        """Refresh the panel with the given face entries.

        When ``faces`` is empty, emits ``face_selected(-1)`` so the host
        window can clear stale active-face state instead of leaving it
        pointed at a face from the previous frame.
        """
        self._faces = tuple(faces)
        self.blockSignals(True)
        try:
            self.clear()
            if not self._faces:
                placeholder = QListWidgetItem("No faces in frame")
                placeholder.setFlags(Qt.NoItemFlags)  # type: ignore[attr-defined]
                placeholder.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                self.addItem(placeholder)
            else:
                for face in self._faces:
                    item = QListWidgetItem(f"Face {face.face_index + 1}")
                    item.setData(Qt.UserRole, face.face_index)  # type: ignore[attr-defined]
                    item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                    item.setIcon(self._icon_for(face))
                    item.setSizeHint(QSize(self._ICON_SIZE + 24, self._ICON_SIZE + 28))
                    self.addItem(item)
        finally:
            self.blockSignals(False)
        if self._faces:
            self.setCurrentRow(0)
        else:
            self.face_selected.emit(-1)

    def select_face(self, face_index: int) -> bool:
        """Select the row matching ``face_index``. Returns ``True`` on success."""
        for row in range(self.count()):
            item = self.item(row)
            if item is None:
                continue
            data = item.data(Qt.UserRole)  # type: ignore[attr-defined]
            if isinstance(data, int) and data == face_index:
                self.setCurrentRow(row)
                return True
        return False

    def active_face(self) -> FaceThumbnail | None:
        """Return the currently selected face entry, or ``None`` when empty."""
        row = self.currentRow()
        if row < 0 or row >= len(self._faces):
            return None
        return self._faces[row]

    def _icon_for(self, face: FaceThumbnail) -> QIcon:
        """Decode a JPEG thumbnail into a QIcon, falling back to the placeholder."""
        image = _decode_jpeg_to_qimage(face.thumbnail_jpeg)
        if image.isNull():
            return self._placeholder
        return QIcon(QPixmap.fromImage(image))

    def _build_placeholder_icon(self) -> QIcon:
        """Build a generic placeholder icon for missing/undecodable thumbnails."""
        pixmap = QPixmap(self._ICON_SIZE, self._ICON_SIZE)
        pixmap.fill(QColor("#222"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#888"))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "?")  # type: ignore[attr-defined]
        painter.end()
        return QIcon(pixmap)

    def _on_row_changed(self, row: int) -> None:
        """Translate row changes into ``face_selected`` events."""
        if row < 0 or row >= len(self._faces):
            self.face_selected.emit(-1)
            return
        self.face_selected.emit(self._faces[row].face_index)
