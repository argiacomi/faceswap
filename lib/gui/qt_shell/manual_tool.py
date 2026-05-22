#!/usr/bin/env python3
"""Native Qt Manual Tool shell.

This is the first Qt-native surface for the Manual Tool migration.  It provides
startup validation, frame display with zoom/pan/overlay seams, thumbnail/
selection, navigation, dirty-state lifecycle, async video-frame loading and a
legacy fallback launcher without importing ``tkinter`` on the Qt path.
"""

from __future__ import annotations

import logging
import subprocess
import typing as T

from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.theme import QtTheme, icon_for_action
from lib.gui.services.command_builder import CommandBuilder
from tools.manual.session import (
    FaceThumbnail,
    ManualAlignmentsHandle,
    ManualEditableAlignments,
    ManualEditorState,
    ManualFrame,
    ManualSession,
    ManualVideoMetadata,
)

logger = logging.getLogger(__name__)

OverlayPainter = T.Callable[[QPainter, "FrameViewport"], None]


class FrameViewport(T.NamedTuple):
    """Snapshot of a frame view geometry passed to overlay painters."""

    source_size: tuple[int, int]
    """(width, height) of the source frame in pixels."""
    target_rect: QRectF
    """Destination rectangle in widget coordinates where the frame is drawn."""
    zoom: float
    """Effective zoom factor relative to the fit-to-widget baseline."""

    def source_to_widget(self, sx: float, sy: float) -> QPointF:
        """Translate a source-image point into widget-coordinate space."""
        src_w, src_h = self.source_size
        if src_w <= 0 or src_h <= 0:
            return QPointF(self.target_rect.x(), self.target_rect.y())
        rx = sx / src_w
        ry = sy / src_h
        return QPointF(
            self.target_rect.x() + rx * self.target_rect.width(),
            self.target_rect.y() + ry * self.target_rect.height(),
        )

    def source_rect_to_widget(self, x: float, y: float, w: float, h: float) -> QRectF:
        """Translate a source-image rect into widget-coordinate space."""
        top_left = self.source_to_widget(x, y)
        bottom_right = self.source_to_widget(x + w, y + h)
        return QRectF(top_left, bottom_right)


class ManualFrameOverlay:
    """Stateful painter for editable bounding boxes + landmarks.

    The overlay is registered with :meth:`ManualFrameView.add_overlay` and
    draws one rectangle per :class:`EditableFace` returned by the bound
    :class:`ManualEditableAlignments`.  The active face (passed via
    :meth:`set_active`) is rendered with a contrasting accent and a halo so
    follow-up editor surfaces can build on the same hit-target.
    """

    _DEFAULT_COLOR = QColor("#3aa0ff")
    _ACTIVE_COLOR = QColor("#ffb000")
    _LANDMARK_COLOR = QColor("#ffffff")
    _LANDMARK_RADIUS = 2.0

    def __init__(
        self,
        model: ManualEditableAlignments,
        *,
        frame_index_provider: T.Callable[[], int],
    ) -> None:
        self._model = model
        self._frame_index_provider = frame_index_provider
        self._active_face: int | None = None

    def set_active(self, face_index: int | None) -> None:
        """Mark a face as the active selection for highlight rendering."""
        self._active_face = face_index

    @property
    def active_face(self) -> int | None:
        """Return the currently highlighted ``face_index`` (or ``None``)."""
        return self._active_face

    def __call__(self, painter: QPainter, viewport: FrameViewport) -> None:
        """Draw the overlay during :meth:`ManualFrameView.paintEvent`."""
        frame_index = self._frame_index_provider()
        faces = self._model.faces(frame_index)
        if not faces:
            return
        pen_width = max(1.0, 1.5 * (1.0 / max(viewport.zoom, 0.001)) * viewport.zoom)
        for face in faces:
            is_active = face.face_index == self._active_face
            painter.save()
            rect = viewport.source_rect_to_widget(*face.bbox)
            color = self._ACTIVE_COLOR if is_active else self._DEFAULT_COLOR
            pen = QPen(color)
            pen.setWidthF(pen_width + (1.0 if is_active else 0.0))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            if face.landmarks:
                painter.setBrush(QBrush(self._LANDMARK_COLOR))
                painter.setPen(Qt.NoPen)
                for lx, ly in face.landmarks:
                    point = viewport.source_to_widget(lx, ly)
                    painter.drawEllipse(
                        point,
                        self._LANDMARK_RADIUS,
                        self._LANDMARK_RADIUS,
                    )
            painter.restore()


class ManualFrameView(QWidget):
    """Manual Tool frame display with pan/zoom and overlay seams."""

    view_changed = Signal()
    frame_loaded = Signal(ManualFrame)
    clicked_at = Signal(QPointF)
    """Emitted with source-image coordinates when the view is left-clicked."""

    _MIN_ZOOM = 1.0
    _MAX_ZOOM = 20.0
    _ZOOM_STEP = 1.25

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-frame-view")
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)
        self._source: QPixmap = QPixmap()
        self._zoom: float = 1.0
        self._offset: QPointF = QPointF(0.0, 0.0)
        self._drag_anchor: QPoint | None = None
        self._drag_origin: QPointF = QPointF(0.0, 0.0)
        self._empty_message: str = "No frame selected"
        self._overlays: list[OverlayPainter] = []

    # ---- public API ----
    @property
    def zoom(self) -> float:
        """Return current zoom factor."""
        return self._zoom

    @property
    def offset(self) -> QPointF:
        """Return current pan offset (widget-pixel translation of the target rect)."""
        return QPointF(self._offset)

    @property
    def has_frame(self) -> bool:
        """Return whether a frame pixmap is loaded."""
        return not self._source.isNull()

    @property
    def source_size(self) -> tuple[int, int]:
        """Return (width, height) of the current source frame, or (0, 0) when empty."""
        if self._source.isNull():
            return (0, 0)
        return (self._source.width(), self._source.height())

    def load_frame(self, frame: ManualFrame) -> bool:
        """Load and display a source frame image from disk."""
        pixmap = QPixmap(frame.path)
        if pixmap.isNull():
            self.clear_frame(f"Could not load frame: {frame.name}")
            return False
        return self._set_pixmap(pixmap, frame)

    def set_image(self, image: QImage, frame: ManualFrame | None = None) -> bool:
        """Load and display a pre-decoded QImage (e.g. from a video frame loader)."""
        if image.isNull():
            self.clear_frame("Could not load frame: empty image")
            return False
        return self._set_pixmap(QPixmap.fromImage(image), frame)

    def clear_frame(self, message: str = "No frame selected") -> None:
        """Clear the current frame."""
        self._source = QPixmap()
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._empty_message = message
        self.update()
        self.view_changed.emit()

    def zoom_in(self) -> None:
        """Zoom into the frame display."""
        self._set_zoom(self._zoom * self._ZOOM_STEP)

    def zoom_out(self) -> None:
        """Zoom out of the frame display."""
        self._set_zoom(self._zoom / self._ZOOM_STEP)

    def reset_view(self) -> None:
        """Reset zoom + pan and redraw."""
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self.update()
        self.view_changed.emit()

    def add_overlay(self, painter: OverlayPainter) -> T.Callable[[], None]:
        """Register a callable invoked after the frame is painted.

        ``painter(qpainter, viewport)`` receives the active ``QPainter`` and the
        current :class:`FrameViewport` describing the source size, on-widget
        target rect and zoom.  Returns an unregister callable.
        """
        self._overlays.append(painter)

        def _remove() -> None:
            if painter in self._overlays:
                self._overlays.remove(painter)
            self.update()

        self.update()
        return _remove

    def viewport(self) -> FrameViewport | None:
        """Return the current viewport snapshot, or ``None`` when empty."""
        if self._source.isNull():
            return None
        return FrameViewport(
            source_size=(self._source.width(), self._source.height()),
            target_rect=self._target_rect(),
            zoom=self._zoom,
        )

    # ---- Qt event handlers ----
    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa:N802
        """Redraw on resize."""
        super().resizeEvent(event)
        self._clamp_offset()
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa:N802
        """Zoom with mouse wheel."""
        if self._source.isNull():
            event.ignore()
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa:N802
        """Start drag-pan on left click; capture click position for click signal."""
        if self._source.isNull() or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        position = event.position()
        self._drag_anchor = position.toPoint()
        self._drag_origin = QPointF(self._offset)
        self.setCursor(Qt.ClosedHandCursor)
        source_point = self._widget_to_source(position)
        if source_point is not None:
            self.clicked_at.emit(source_point)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa:N802
        """Update pan offset while dragging."""
        position = event.position()
        if self._drag_anchor is None:
            self._update_hover_cursor(position)
            super().mouseMoveEvent(event)
            return
        delta = position - QPointF(self._drag_anchor)
        self._offset = QPointF(
            self._drag_origin.x() + delta.x(),
            self._drag_origin.y() + delta.y(),
        )
        self._clamp_offset()
        self.update()
        self.view_changed.emit()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa:N802
        """End drag-pan."""
        if self._drag_anchor is None:
            super().mouseReleaseEvent(event)
            return
        self._drag_anchor = None
        self._update_hover_cursor(event.position())
        event.accept()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa:N802
        """Draw the source frame at the current zoom+offset, then overlays."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101418"))
        if self._source.isNull():
            painter.setPen(QColor("#cccccc"))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._empty_message)
            return
        target = self._target_rect()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._source, QRectF(self._source.rect()))
        viewport = FrameViewport(
            source_size=(self._source.width(), self._source.height()),
            target_rect=target,
            zoom=self._zoom,
        )
        for overlay in tuple(self._overlays):
            try:
                overlay(painter, viewport)
            except Exception:  # pragma: no cover - defensive against bad overlay callables
                logger.exception("Overlay painter raised; continuing")

    # ---- internals ----
    def _set_pixmap(self, pixmap: QPixmap, frame: ManualFrame | None) -> bool:
        """Install a new source pixmap and reset view state."""
        self._source = pixmap
        self._offset = QPointF(0.0, 0.0)
        self.update()
        self.view_changed.emit()
        if frame is not None:
            self.frame_loaded.emit(frame)
        return True

    def _set_zoom(self, zoom: float) -> None:
        """Clamp and apply zoom."""
        clamped = max(self._MIN_ZOOM, min(self._MAX_ZOOM, zoom))
        if clamped == self._zoom:
            return
        self._zoom = clamped
        self._clamp_offset()
        self.update()
        self.view_changed.emit()

    def _fit_size(self) -> tuple[float, float]:
        """Return the (width, height) the source frame uses at zoom=1 inside the widget."""
        if self._source.isNull() or self.width() <= 0 or self.height() <= 0:
            return (0.0, 0.0)
        source_aspect = self._source.width() / self._source.height()
        widget_aspect = self.width() / self.height()
        if source_aspect >= widget_aspect:
            width = float(self.width())
            height = width / source_aspect
        else:
            height = float(self.height())
            width = height * source_aspect
        return (width, height)

    def _target_rect(self) -> QRectF:
        """Compute the destination rect for the source frame at the current zoom+offset."""
        fit_w, fit_h = self._fit_size()
        width = fit_w * self._zoom
        height = fit_h * self._zoom
        x = (self.width() - width) / 2.0 + self._offset.x()
        y = (self.height() - height) / 2.0 + self._offset.y()
        return QRectF(x, y, width, height)

    def _clamp_offset(self) -> None:
        """Clamp pan offset so the source frame cannot be dragged fully off-screen."""
        fit_w, fit_h = self._fit_size()
        margin_x = max(0.0, (fit_w * self._zoom - self.width()) / 2.0)
        margin_y = max(0.0, (fit_h * self._zoom - self.height()) / 2.0)
        self._offset = QPointF(
            max(-margin_x, min(margin_x, self._offset.x())),
            max(-margin_y, min(margin_y, self._offset.y())),
        )

    def _widget_to_source(self, point: QPointF) -> QPointF | None:
        """Translate a widget-coordinate point to source-image pixel coordinates."""
        target = self._target_rect()
        if not target.contains(point) or target.width() <= 0 or target.height() <= 0:
            return None
        src_w, src_h = self.source_size
        rel_x = (point.x() - target.x()) / target.width()
        rel_y = (point.y() - target.y()) / target.height()
        return QPointF(rel_x * src_w, rel_y * src_h)

    def _update_hover_cursor(self, position: QPointF) -> None:
        """Show a hand cursor when hovering a pannable region."""
        if self._source.isNull() or self._zoom <= self._MIN_ZOOM:
            self.setCursor(Qt.ArrowCursor)
            return
        if self._target_rect().contains(position):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)


class _VideoFrameWorker(QObject):
    """Worker that lives on a QThread and serves video frames on demand."""

    count_ready = Signal(int)
    frame_ready = Signal(int, str, QImage)
    load_failed = Signal(int, str)

    def __init__(
        self,
        path: str,
        video_meta_data: dict[str, list[int]] | None,
    ) -> None:
        super().__init__()
        self._path = path
        self._video_meta_data = video_meta_data
        self._loader: T.Any | None = None

    def initialize(self) -> None:
        """Open the underlying SingleFrameLoader and emit the frame count."""
        from lib.image import SingleFrameLoader

        try:
            self._loader = SingleFrameLoader(self._path, video_meta_data=self._video_meta_data)
        except Exception as err:  # pragma: no cover - exercised through integration
            self.load_failed.emit(-1, f"Could not open video: {err}")
            return
        self.count_ready.emit(int(self._loader.count))

    def fetch(self, index: int) -> None:
        """Decode and emit the requested frame index as a QImage."""
        if self._loader is None:
            self.load_failed.emit(index, "Video frame loader not initialized")
            return
        try:
            filename, frame = self._loader.image_from_index(index)
        except Exception as err:  # pragma: no cover - exercised through integration
            self.load_failed.emit(index, str(err))
            return
        image = _bgr_array_to_qimage(frame)
        if image.isNull():
            self.load_failed.emit(index, "Decoded frame produced an empty image")
            return
        self.frame_ready.emit(index, filename, image)


def _bgr_array_to_qimage(frame: T.Any) -> QImage:
    """Convert a numpy BGR uint8 array (from SingleFrameLoader) to a QImage."""
    if frame is None or frame.size == 0:
        return QImage()
    height, width = frame.shape[:2]
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    if channels == 1:
        image = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
    elif channels == 3:
        # SingleFrameLoader returns BGR; Qt expects RGB, so swap into Format_BGR888.
        image = QImage(frame.data, width, height, width * 3, QImage.Format_BGR888)
    else:
        # 4-channel BGRA fallback.
        image = QImage(frame.data, width, height, width * 4, QImage.Format_ARGB32)
    # The numpy buffer is owned by the caller; copy to detach.
    return image.copy()


class VideoFrameProvider(QObject):
    """Async video frame provider for the native Qt Manual Tool.

    The provider owns a worker QObject moved onto a private QThread.  Consumers
    call :meth:`request_frame` and listen on :attr:`frame_ready` (or
    :attr:`load_failed`) for results, keeping the UI thread responsive while
    individual frames decode.
    """

    count_ready = Signal(int)
    frame_ready = Signal(int, str, QImage)
    load_failed = Signal(int, str)
    _request = Signal(int)
    _start_init = Signal()

    def __init__(
        self,
        path: str,
        video_meta_data: dict[str, list[int]] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = _VideoFrameWorker(path, video_meta_data)
        self._worker.moveToThread(self._thread)
        self._worker.count_ready.connect(self.count_ready)
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.load_failed.connect(self.load_failed)
        self._start_init.connect(self._worker.initialize)
        self._request.connect(self._worker.fetch)
        self._thread.start()

    def start(self) -> None:
        """Open the video on the worker thread."""
        self._start_init.emit()

    def request_frame(self, index: int) -> None:
        """Ask the worker thread to decode and emit the given frame index."""
        self._request.emit(int(index))

    def shutdown(self) -> None:
        """Stop the worker thread and release resources."""
        if not self._thread.isRunning():
            return
        self._thread.quit()
        self._thread.wait(2000)


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
            item.setFlags(Qt.NoItemFlags)
            self.addItem(item)
            return
        for frame in frames:
            item = QListWidgetItem(f"{frame.index + 1}: {frame.name}")
            item.setData(Qt.UserRole, frame.index)
            self.addItem(item)


def _decode_jpeg_to_qimage(payload: bytes) -> QImage:
    """Decode a JPEG byte string into a QImage. Returns a null image on failure."""
    if not payload:
        return QImage()
    image = QImage()
    if not image.loadFromData(payload, "JPG"):
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
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setIconSize(QSize(self._ICON_SIZE, self._ICON_SIZE))
        self.setSpacing(6)
        self.setMinimumHeight(self._ICON_SIZE + 36)
        self.setUniformItemSizes(True)
        self._faces: tuple[FaceThumbnail, ...] = ()
        self._placeholder = self._build_placeholder_icon()
        self.currentRowChanged.connect(self._on_row_changed)

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
                placeholder.setFlags(Qt.NoItemFlags)
                placeholder.setTextAlignment(Qt.AlignCenter)
                self.addItem(placeholder)
            else:
                for face in self._faces:
                    item = QListWidgetItem(f"Face {face.face_index + 1}")
                    item.setData(Qt.UserRole, face.face_index)
                    item.setTextAlignment(Qt.AlignCenter)
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
            data = item.data(Qt.UserRole)
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
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "?")
        painter.end()
        return QIcon(pixmap)

    def _on_row_changed(self, row: int) -> None:
        """Translate row changes into ``face_selected`` events."""
        if row < 0 or row >= len(self._faces):
            self.face_selected.emit(-1)
            return
        self.face_selected.emit(self._faces[row].face_index)


class ManualAction(T.NamedTuple):
    """Static metadata for one Manual Tool editor action.

    The legacy Tk Manual Tool wires editor commands through Tk widgets and key
    bindings; the Qt shell mirrors the same surface through a registry so each
    action has a single source of truth for label, icon, shortcut, tooltip and
    availability semantics.  ``handler`` is the unbound method name on
    :class:`ManualToolWindow` that performs the action when triggered.

    Icons reuse the existing Qt theme icon cache where available.  Actions
    whose ``icon`` does not yet resolve are documented inline below.
    """

    key: str
    label: str
    handler: str
    shortcut: tuple[str, ...] = ()
    icon: str | None = None
    tooltip: str = ""
    separator_before: bool = False


# Manual Tool action registry. The first six actions cover the legacy file/
# navigation surface. The remaining entries register editor operations
# (add/delete/copy/revert), the filter cycle, the F1-F5 editor modes, the
# F9/F10 annotation toggles and the legacy-tool fallback.  Icons marked
# "(missing)" in the comments fall back to a null QIcon — replace them with
# real PNGs in lib/gui/.cache/icons/ when art lands.
MANUAL_ACTIONS: tuple[ManualAction, ...] = (
    ManualAction(
        key="save",
        label="Save",
        handler="save",
        shortcut=("Ctrl+S",),
        icon="save",
        tooltip="Save edits to the alignments file (Ctrl+S)",
    ),
    ManualAction(
        key="revert_frame",
        label="Revert Frame",
        handler="revert_current_frame",
        shortcut=("R",),
        icon="reload",
        tooltip="Revert edits in the current frame (R)",
        separator_before=True,
    ),
    ManualAction(
        key="first_frame",
        label="First Frame",
        handler="goto_first_frame",
        shortcut=("Home",),
        icon="beginning",
        tooltip="Jump to the first frame (Home)",
        separator_before=True,
    ),
    ManualAction(
        key="previous_frame",
        label="Previous",
        handler="_previous_frame",
        shortcut=("Z", "Left"),
        icon="prev",
        tooltip="Previous frame (Z)",
    ),
    ManualAction(
        key="next_frame",
        label="Next",
        handler="_next_frame",
        shortcut=("X", "Right"),
        icon="next",
        tooltip="Next frame (X)",
    ),
    ManualAction(
        key="last_frame",
        label="Last Frame",
        handler="goto_last_frame",
        shortcut=("End",),
        icon="end",
        tooltip="Jump to the last frame (End)",
    ),
    ManualAction(
        key="play_pause",
        label="Play / Pause",
        handler="toggle_play",
        shortcut=("Space",),
        icon="play",
        tooltip="Play / pause playback (Space)",
    ),
    ManualAction(
        key="copy_prev_face",
        label="Copy From Previous",
        handler="copy_prev_face",
        shortcut=("C",),
        icon="copy_prev",
        tooltip="Copy alignment from the previous frame (C)",
        separator_before=True,
    ),
    ManualAction(
        key="copy_next_face",
        label="Copy From Next",
        handler="copy_next_face",
        shortcut=("V",),
        icon="copy_next",
        tooltip="Copy alignment from the next frame (V)",
    ),
    ManualAction(
        key="delete_face",
        label="Delete Face",
        handler="delete_active_face",
        shortcut=("Delete", "Backspace"),
        icon="clear",  # No dedicated trash icon yet; reusing the "clear" glyph.
        tooltip="Delete the active face (Delete)",
    ),
    ManualAction(
        key="add_face",
        label="Add Face",
        handler="add_face_at_center",
        shortcut=("Ctrl+N",),
        icon=None,  # (missing) — no dedicated add-face icon in the cache.
        tooltip="Add a new face at the center of the current frame (Ctrl+N)",
    ),
    ManualAction(
        key="undo_edit",
        label="Undo",
        handler="undo_edit",
        shortcut=("Ctrl+Z",),
        icon=None,  # (missing) — no dedicated undo icon in the cache.
        tooltip="Undo the last edit (Ctrl+Z)",
        separator_before=True,
    ),
    ManualAction(
        key="redo_edit",
        label="Redo",
        handler="redo_edit",
        shortcut=("Ctrl+Shift+Z", "Ctrl+Y"),
        icon=None,  # (missing) — no dedicated redo icon in the cache.
        tooltip="Redo the last undone edit (Ctrl+Shift+Z)",
    ),
    ManualAction(
        key="cycle_filter",
        label="Filter",
        handler="cycle_filter_mode",
        shortcut=("F",),
        icon=None,  # (missing) — no filter icon in the cache.
        tooltip="Cycle the navigation filter mode (F)",
        separator_before=True,
    ),
    ManualAction(
        key="set_view_mode",
        label="View",
        handler="set_editor_view",
        shortcut=("F1",),
        icon="view",
        tooltip="Switch to View editor (F1)",
        separator_before=True,
    ),
    ManualAction(
        key="set_boundingbox_mode",
        label="Bounding Box",
        handler="set_editor_boundingbox",
        shortcut=("F2",),
        icon="boundingbox",
        tooltip="Switch to Bounding Box editor (F2)",
    ),
    ManualAction(
        key="set_extractbox_mode",
        label="Extract Box",
        handler="set_editor_extractbox",
        shortcut=("F3",),
        icon="extractbox",
        tooltip="Switch to Extract Box editor (F3)",
    ),
    ManualAction(
        key="set_landmarks_mode",
        label="Landmarks",
        handler="set_editor_landmarks",
        shortcut=("F4",),
        icon="landmarks",
        tooltip="Switch to Landmarks editor (F4)",
    ),
    ManualAction(
        key="set_mask_mode",
        label="Mask",
        handler="set_editor_mask",
        shortcut=("F5",),
        icon="mask",
        tooltip="Switch to Mask editor (F5)",
    ),
    ManualAction(
        key="cycle_annotation",
        label="Annotation",
        handler="cycle_annotation_display",
        shortcut=("F9",),
        icon=None,  # (missing) — no annotation icon in the cache.
        tooltip="Cycle annotation display (F9)",
        separator_before=True,
    ),
    ManualAction(
        key="zoom_in",
        label="Zoom In",
        handler="zoom_in",
        shortcut=("=", "+"),
        icon="zoom",
        tooltip="Zoom in on the frame view (+)",
        separator_before=True,
    ),
    ManualAction(
        key="zoom_out",
        label="Zoom Out",
        handler="zoom_out",
        shortcut=("-",),
        icon=None,  # (missing) — no dedicated zoom-out icon.
        tooltip="Zoom out of the frame view (-)",
    ),
    ManualAction(
        key="reset_view",
        label="Reset View",
        handler="reset_view",
        shortcut=("0",),
        icon=None,  # (missing) — reuse default until art lands.
        tooltip="Reset zoom and pan (0)",
    ),
    ManualAction(
        key="legacy_tool",
        label="Open Legacy Tool",
        handler="launch_legacy",
        shortcut=(),
        icon=None,  # (missing) — no dedicated legacy icon.
        tooltip="Launch the legacy Tk Manual Tool",
        separator_before=True,
    ),
)


class _ManualStartupTask(QObject):
    """Worker that performs Manual Tool session preparation off the UI thread.

    Stages reported via :attr:`progress`:
        * ``"open"``      – Open the alignments file (parses on first access).
        * ``"seed"``      – Seed the editable model from the alignments file.
        * ``"thumbs"``    – Check the thumbnail cache state.
        * ``"complete"``  – All stages finished.

    The worker is intentionally tiny and synchronous inside :meth:`run` — Qt
    moves it to a dedicated QThread, so blocking IO inside this object does
    not freeze the window.
    """

    progress = Signal(str, str)
    """Emitted with ``(stage, message)`` as each stage starts."""
    completed = Signal(bool, str)
    """Emitted with ``(has_thumbnails, summary)`` on success."""
    failed = Signal(str)
    """Emitted with a user-facing error message on failure."""
    _start = Signal()

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
    ) -> None:
        super().__init__()
        self._handle = handle
        self._editable = editable
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule the worker on its owning thread."""
        self._start.emit()

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Execute the staged startup work."""
        try:
            if not self._handle.exists:
                self.progress.emit("complete", "No alignments file yet — ready")
                self.completed.emit(False, "No alignments file yet")
                return

            self.progress.emit("open", "Opening alignments file…")
            alignments = self._handle.open()
            face_count = sum(len(entry.faces) for entry in alignments.data.values())

            self.progress.emit("thumbs", "Checking thumbnail cache…")
            has_thumbnails = self._handle.has_thumbnails()

            summary = f"Loaded {face_count} face(s) across {len(alignments.data)} frame(s)"
            self.progress.emit("complete", summary)
            self.completed.emit(has_thumbnails, summary)
        except Exception as err:  # noqa: BLE001 - surface any startup failure
            logger.exception("Manual Tool startup failed")
            self.failed.emit(str(err))


class ManualStartupWorker(QObject):
    """Owns the QThread that runs :class:`_ManualStartupTask`.

    The window subscribes to :attr:`progress`, :attr:`completed` and
    :attr:`failed` for user-facing feedback.  Calling :meth:`stop` is safe at
    any time — it gracefully terminates the thread even if the user closes
    the window mid-load.
    """

    progress = Signal(str, str)
    completed = Signal(bool, str)
    failed = Signal(str)

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualStartupTask(handle, editable)
        self._task.moveToThread(self._thread)
        self._task.progress.connect(self.progress)
        self._task.completed.connect(self.completed)
        self._task.failed.connect(self.failed)
        # Auto-quit the QThread as soon as the task reports a terminal
        # state so the thread is fully stopped before the parent window
        # destructor runs.  Without this, Qt aborts on the QThread
        # destructor under pytest teardown when stop() was not called.
        self._task.completed.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background processing on the worker thread."""
        self._task.kick_off()

    def stop(self) -> None:
        """Stop the worker thread and wait for it to exit."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)


class ManualToolWindow(QMainWindow):
    """Qt-native Manual Tool window with legacy fallback support."""

    dirty_changed = Signal(bool)
    frame_changed = Signal(int)
    action_triggered = Signal(str)
    """Emitted with the :class:`ManualAction.key` whenever an action fires.

    Editor-mode and stubbed editing actions (copy/revert/delete) connect
    follow-up tickets through this signal without needing direct method hooks.
    """

    def __init__(
        self,
        session: ManualSession,
        *,
        legacy_args: list[str] | None = None,
        parent: QWidget | None = None,
        console_logger: T.Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-tool-window")
        self.setWindowTitle("Faceswap Manual Tool")
        self._session = session
        self._editor_state = session.create_editor_state()
        self._video_metadata: ManualVideoMetadata | None = None
        self._video_provider: VideoFrameProvider | None = None
        self._video_frames: list[ManualFrame] = []
        self._legacy_args = legacy_args or []
        self._current_frame: ManualFrame | None = None
        self._frame_view = ManualFrameView()
        self._thumbnail_panel = ManualThumbnailPanel()
        self._face_panel = FaceThumbnailPanel()
        self._status_label = QLabel()
        self._metadata_label = QLabel()
        self._progress_bar: QProgressBar | None = None
        self._console_logger = console_logger
        self._startup_worker: ManualStartupWorker | None = None
        self._startup_complete = False
        self._actions: dict[str, QAction] = {}
        # One cached alignments handle for the lifetime of this window so we
        # are not reopening the alignments file on every face refresh.
        self._alignments_handle = session.alignments_handle()
        self._editable = ManualEditableAlignments()
        self._editable.subscribe(self._on_editable_changed)
        self._overlay = ManualFrameOverlay(
            self._editable,
            frame_index_provider=self._current_frame_index,
        )
        self._editor_state.subscribe("unsaved", self._on_unsaved_changed)
        self._editor_state.subscribe("face_index", lambda _v: self._sync_actions())
        self._editor_state.subscribe(
            "face_index", lambda value: self._overlay.set_active(int(value))
        )
        self._editor_state.subscribe("face_index", lambda _v: self._frame_view.update())
        self._editor_state.subscribe("frame_index", lambda _v: self._sync_actions())
        self._editor_state.subscribe("editor_mode", self._on_editor_mode_changed)
        self._build_ui()
        self._connect_signals()
        self._load_session()
        self._frame_view.add_overlay(self._overlay)
        self._start_background_startup()

    @property
    def session(self) -> ManualSession:
        """Return the backing GUI-neutral session."""
        return self._session

    @property
    def is_dirty(self) -> bool:
        """Return whether the Qt Manual Tool has unsaved edits."""
        return self._editor_state.unsaved

    @property
    def editor_state(self) -> ManualEditorState:
        """Return the shared GUI-neutral editor state."""
        return self._editor_state

    @property
    def video_metadata(self) -> ManualVideoMetadata | None:
        """Return cached neutral video metadata, if loaded."""
        return self._video_metadata

    @property
    def frame_view(self) -> ManualFrameView:
        """Return the embedded frame view widget."""
        return self._frame_view

    @property
    def face_panel(self) -> FaceThumbnailPanel:
        """Return the per-frame face thumbnail panel widget."""
        return self._face_panel

    @property
    def editable_alignments(self) -> ManualEditableAlignments:
        """Return the GUI-neutral editable alignment model.

        Exposed so callers (tests, follow-up integration code) can seed it
        from a real alignments file or assert edit/undo behavior without
        scraping internal state.
        """
        return self._editable

    @property
    def frame_overlay(self) -> ManualFrameOverlay:
        """Return the editable-alignments overlay painter."""
        return self._overlay

    @property
    def actions_by_key(self) -> dict[str, QAction]:
        """Return the registered Manual Tool actions keyed by :class:`ManualAction.key`.

        Exposed so that follow-up integrations and tests can drive specific
        actions (and assert their enabled state) without scraping the toolbar.
        """
        return dict(self._actions)

    @property
    def video_provider(self) -> VideoFrameProvider | None:
        """Return the async video frame provider for video inputs (or ``None``)."""
        return self._video_provider

    @classmethod
    def from_command_values(
        cls,
        values: T.Mapping[str, object],
        *,
        builder: CommandBuilder,
        parent: QWidget | None = None,
        console_logger: T.Callable[[str], None] | None = None,
    ) -> ManualToolWindow:
        """Create a Manual Tool window from command-panel values."""
        session = ManualSession.from_cli_values(values)
        legacy_args = builder.build("tools", "manual", values, generate=False)
        return cls(
            session,
            legacy_args=legacy_args,
            parent=parent,
            console_logger=console_logger,
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa:N802
        """Prompt before closing when the editor has unsaved changes."""
        if self._editor_state.unsaved:
            answer = QMessageBox.question(
                self,
                "Unsaved Manual Tool Changes",
                "Close the Manual Tool and discard unsaved changes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        if self._video_provider is not None:
            self._video_provider.shutdown()
            self._video_provider = None
        if self._startup_worker is not None:
            self._startup_worker.stop()
            self._startup_worker = None
        event.accept()

    def mark_dirty(self, dirty: bool = True) -> None:
        """Set dirty state and update action availability."""
        self._editor_state.set("unsaved", dirty)

    def save(self) -> bool:
        """Persist editable alignment edits back to the alignments file.

        On success: clears dirty state, drops the editable undo/redo history
        (so the user cannot undo past the save point) and refreshes the face
        thumbnail panel against the freshly-written alignments file.

        On failure: leaves the editor state dirty, logs the exception and
        surfaces the error to the user through both the status bar and a
        modal dialog so a silent save-failure is not possible.
        """
        if self._current_frame_index() < 0 and not any(
            self._editable.face_count(i) for i in range(self._session.frame_count)
        ):
            self.statusBar().showMessage("Nothing to save", 3000)
            return True
        try:
            frame_names = self._frame_names_for_persist()
            modified = self._alignments_handle.persist(self._editable, frame_names=frame_names)
        except Exception as err:  # noqa:BLE001 - surface any persist failure
            message = f"Manual Tool save failed: {err}"
            logger.exception("Manual Tool save failed")
            self.statusBar().showMessage(message, 7000)
            QMessageBox.critical(self, "Manual Tool Save", message)
            return False
        self._editable.clear_history()
        self.mark_dirty(False)
        self._editor_state.set("edited", False)
        self._editor_state.set("face_count_changed", False)
        self.refresh_faces()
        self._frame_view.update()
        self.statusBar().showMessage(f"Saved {modified} frame(s) to alignments file", 5000)
        return True

    def _frame_name_for_index(self, frame_index: int) -> str | None:
        """Resolve one editable frame_index to its on-disk frame name.

        Mirrors :meth:`_frame_names_for_persist` but for a single index — used
        when refreshing the face panel where we just need the current frame's
        name for thumbnail lookup, not the whole mapping.
        """
        mapping = self._frame_names_for_persist()
        if callable(mapping):
            return mapping(frame_index)
        if 0 <= frame_index < len(mapping):
            return mapping[frame_index]
        return None

    def _frame_names_for_persist(
        self,
    ) -> list[str] | T.Callable[[int], str | None]:
        """Return a frame-name mapping ordered to match the editable model.

        Image-folder sessions return the source frame list (sparse alignments
        still align to the source ordering); video sessions return a callable
        that synthesizes the Faceswap-standard dummy frame name on demand, so
        edits on frames the alignments file has *never* seen before (the
        common case for fresh video input) still resolve to a writable name.
        Falls back to the alignments file's existing keys when neither source
        is available.
        """
        if self._session.has_images:
            return [frame.name for frame in self._session.frame_list]
        if self._session.is_video_input:
            return self._session.frame_name_for_index
        return list(self._alignments_handle.sorted_frame_names())

    def launch_legacy(self) -> bool:
        """Launch the existing Tk Manual Tool as a fallback subprocess."""
        if not self._legacy_args:
            QMessageBox.warning(self, "Legacy Manual Tool", "Legacy command is unavailable.")
            return False
        try:
            subprocess.Popen(self._legacy_args)  # noqa:S603 - args are built internally
        except OSError as err:
            QMessageBox.critical(self, "Legacy Manual Tool", str(err))
            return False
        self.statusBar().showMessage("Launched legacy Manual Tool", 5000)
        return True

    def goto_first_frame(self) -> None:
        """Select the first available frame in the thumbnail panel."""
        if self._thumbnail_panel.count() == 0:
            return
        self._thumbnail_panel.setCurrentRow(0)

    def goto_last_frame(self) -> None:
        """Select the last available frame in the thumbnail panel."""
        count = self._thumbnail_panel.count()
        if count == 0:
            return
        self._thumbnail_panel.setCurrentRow(count - 1)

    def toggle_play(self) -> None:
        """Toggle the editor playback flag.  Concrete playback lands in a follow-up."""
        self._editor_state.set("is_playing", not self._editor_state.is_playing)

    def revert_current_frame(self) -> None:
        """Revert editable edits recorded against the current frame only.

        Edits on other frames stay on the undo stack so unrelated work is
        not silently discarded.  Dirty state is recomputed from the
        remaining undo records rather than blanket-cleared.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("Nothing to revert", 3000)
            return
        reverted = self._editable.revert_frame(frame_index)
        self.mark_dirty(self._editable.can_undo)
        if not self._editable.can_undo:
            self._editor_state.set("edited", False)
        if reverted:
            self.statusBar().showMessage(f"Reverted {reverted} edit(s) in this frame", 5000)
        else:
            self.statusBar().showMessage("Nothing to revert in this frame", 3000)

    def copy_prev_face(self) -> bool:
        """Copy the editable faces from the previous frame onto the current one."""
        return self._copy_faces_from(self._current_frame_index() - 1, "previous")

    def copy_next_face(self) -> bool:
        """Copy the editable faces from the next frame onto the current one."""
        return self._copy_faces_from(self._current_frame_index() + 1, "next")

    def _copy_faces_from(self, source_frame_index: int, direction: str) -> bool:
        """Replace current frame faces with the source frame's editable faces.

        Operates entirely on :class:`ManualEditableAlignments` so each
        deletion/addition lands on the shared undo stack and the overlay
        repaints itself via the existing listener.  Returns ``False`` and
        surfaces a status message when there is nothing to copy.
        """
        current_index = self._current_frame_index()
        if current_index < 0:
            return False
        source_faces = self._editable.faces(source_frame_index)
        if not source_faces:
            self.statusBar().showMessage(f"No faces in the {direction} frame to copy", 5000)
            return False
        for face_index in reversed(range(self._editable.face_count(current_index))):
            self._editable.delete_face(current_index, face_index)
        for face in source_faces:
            self._editable.add_face(current_index, face.bbox, landmarks=face.landmarks)
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        if self._editable.face_count(current_index):
            self._editor_state.set("face_index", 0)
        self.statusBar().showMessage(
            f"Copied {len(source_faces)} face(s) from the {direction} frame", 5000
        )
        return True

    def delete_active_face(self) -> None:
        """Delete the active face from the editable alignment model."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return
        face_index = self._editor_state.face_index
        if not self._editable.delete_face(frame_index, face_index):
            self.statusBar().showMessage("No face selected to delete", 5000)
            return
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        new_count = self._editable.face_count(frame_index)
        if new_count == 0:
            # No face left to operate on; flag "no active selection".
            self._editor_state.set("face_index", -1)
        else:
            self._editor_state.set("face_index", min(face_index, new_count - 1))
        self.statusBar().showMessage(f"Deleted face index {face_index}", 5000)

    def add_face_at_center(
        self,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> int | None:
        """Add a new face. Defaults to a centered fixed-size bbox.

        ``bbox`` is provided in source-image coordinates. The convenience
        default places a 64x64 box centred in the source frame so the action
        is usable without a pointer-driven add gesture.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return None
        if bbox is None:
            src_w, src_h = self._frame_view.source_size
            if src_w <= 0 or src_h <= 0:
                self.statusBar().showMessage("Cannot add face: frame not loaded", 5000)
                return None
            size = float(min(64, max(8, min(src_w, src_h) // 4)))
            bbox = (src_w / 2 - size / 2, src_h / 2 - size / 2, size, size)
        try:
            new_index = self._editable.add_face(frame_index, bbox)
        except ValueError as err:
            self.statusBar().showMessage(str(err), 5000)
            return None
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        self._editor_state.set("face_index", new_index)
        self.statusBar().showMessage(f"Added face index {new_index}", 5000)
        return new_index

    def nudge_active_face(self, dx: float, dy: float) -> bool:
        """Translate the active face's bbox + landmarks by ``(dx, dy)`` pixels."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return False
        face_index = self._editor_state.face_index
        if not self._editable.move_face(frame_index, face_index, dx, dy):
            return False
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        return True

    def undo_edit(self) -> bool:
        """Reverse the last edit (if any)."""
        if not self._editable.undo():
            self.statusBar().showMessage("Nothing to undo", 3000)
            return False
        self.statusBar().showMessage("Undid last edit", 3000)
        return True

    def redo_edit(self) -> bool:
        """Re-apply the most recently undone edit (if any)."""
        if not self._editable.redo():
            self.statusBar().showMessage("Nothing to redo", 3000)
            return False
        self.statusBar().showMessage("Redid last edit", 3000)
        return True

    def cycle_filter_mode(self) -> None:
        """Advance to the next filter mode in the legacy rotation order."""
        order = (
            "All Frames",
            "Has Face(s)",
            "No Faces",
            "Single Face",
            "Multiple Faces",
            "Misaligned Faces",
        )
        current = self._editor_state.filter_mode or order[0]
        try:
            index = order.index(current)
        except ValueError:
            index = -1
        self._editor_state.set("filter_mode", order[(index + 1) % len(order)])
        self.statusBar().showMessage(f"Filter: {self._editor_state.filter_mode}", 5000)

    def cycle_annotation_display(self) -> None:
        """Cycle annotation overlays in the legacy rotation order."""
        order = ("None", "Mesh", "Mask", "Landmarks")
        current = self._editor_state.annotation_mode or order[0]
        try:
            index = order.index(current)
        except ValueError:
            index = -1
        self._editor_state.set("annotation_mode", order[(index + 1) % len(order)])
        self.statusBar().showMessage(f"Annotation: {self._editor_state.annotation_mode}", 5000)

    def set_editor_view(self) -> None:
        """Activate the View editor mode."""
        self._editor_state.set("editor_mode", "View")

    def set_editor_boundingbox(self) -> None:
        """Activate the Bounding Box editor mode."""
        self._editor_state.set("editor_mode", "BoundingBox")

    def set_editor_extractbox(self) -> None:
        """Activate the Extract Box editor mode."""
        self._editor_state.set("editor_mode", "ExtractBox")

    def set_editor_landmarks(self) -> None:
        """Activate the Landmarks editor mode."""
        self._editor_state.set("editor_mode", "Landmarks")

    def set_editor_mask(self) -> None:
        """Activate the Mask editor mode."""
        self._editor_state.set("editor_mode", "Mask")

    def zoom_in(self) -> None:
        """Zoom into the embedded frame view (action handler)."""
        self._frame_view.zoom_in()

    def zoom_out(self) -> None:
        """Zoom out of the embedded frame view (action handler)."""
        self._frame_view.zoom_out()

    def reset_view(self) -> None:
        """Reset the embedded frame view (action handler)."""
        self._frame_view.reset_view()

    def _build_ui(self) -> None:
        """Build Manual Tool widgets."""
        self.resize(980, 680)
        self._build_toolbar()
        status = QStatusBar()
        self.setStatusBar(status)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        self._metadata_label.setObjectName("qt-manual-session-metadata")
        self._metadata_label.setWordWrap(True)
        left_layout.addWidget(self._metadata_label)
        left_layout.addWidget(self._frame_view, 1)
        left_layout.addWidget(self._status_label)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("qt-manual-main-splitter")
        splitter.addWidget(left)
        splitter.addWidget(self._face_panel)
        splitter.addWidget(self._thumbnail_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        """Build the Manual Tool action toolbar from :data:`MANUAL_ACTIONS`."""
        toolbar = QToolBar("Manual Tool")
        toolbar.setObjectName("qt-manual-toolbar")
        self.addToolBar(toolbar)
        theme = QtTheme.default()
        for spec in MANUAL_ACTIONS:
            if spec.separator_before:
                toolbar.addSeparator()
            action = QAction(spec.label, self)
            action.setObjectName(f"qt-manual-action-{spec.key}")
            if spec.tooltip:
                action.setToolTip(spec.tooltip)
                action.setStatusTip(spec.tooltip)
            if spec.icon:
                icon = icon_for_action(theme, spec.icon)
                if not icon.isNull():
                    action.setIcon(icon)
            shortcuts = [QKeySequence(text) for text in spec.shortcut]
            if shortcuts:
                action.setShortcuts(shortcuts)
                action.setShortcutContext(Qt.WindowShortcut)
            handler = getattr(self, spec.handler)
            action.triggered.connect(self._make_action_dispatch(spec.key, handler))
            self.addAction(action)  # Register shortcut on the window too.
            toolbar.addAction(action)
            self._actions[spec.key] = action

    def _make_action_dispatch(
        self, key: str, handler: T.Callable[[], object]
    ) -> T.Callable[[], None]:
        """Return a closure that invokes ``handler`` and emits :attr:`action_triggered`."""

        def _dispatch(_checked: bool = False) -> None:
            self.action_triggered.emit(key)
            try:
                handler()
            except Exception:  # pragma: no cover - defensive; surface in logs
                logger.exception("Manual Tool action %s raised", key)

        return _dispatch

    def _connect_signals(self) -> None:
        """Connect selection and dirty-state signals."""
        self._thumbnail_panel.currentItemChanged.connect(self._thumbnail_selected)
        self._thumbnail_panel.currentRowChanged.connect(lambda _row: self._sync_actions())
        self._face_panel.face_selected.connect(self._on_face_selected)
        self._frame_view.clicked_at.connect(self._on_frame_clicked)
        self.dirty_changed.connect(
            lambda dirty: self.statusBar().showMessage(
                "Manual Tool has unsaved changes" if dirty else "Manual Tool ready",
                5000,
            )
        )

    def _load_session(self) -> None:
        """Populate the Qt Manual Tool from a neutral session.

        Only renders cheap session metadata + frame discovery here.  The
        expensive alignments-file parse and thumbnail-cache check run in
        :class:`ManualStartupWorker` and refresh the metadata label once
        they complete.
        """
        frame_summary: str
        if self._session.has_images:
            frame_summary = str(self._session.frame_count)
        elif self._session.is_video_input:
            frame_summary = "video input (loading…)"
        else:
            frame_summary = "video input"
        thumbs_state = "regenerate forced" if self._session.thumb_regenerate else "loading…"
        self._metadata_label.setText(
            "\n".join(
                (
                    f"Input: {self._session.frames}",
                    f"Alignments: {self._alignments_handle.path}",
                    f"Frames: {frame_summary}",
                    f"Thumbnails: {thumbs_state}",
                )
            )
        )
        # Seed the editable model from any existing on-disk alignments
        # before the first frame is selected so the overlay + face panel show
        # the persisted faces immediately.  This open() blocks the UI for the
        # duration of an Alignments load; that is acceptable because the
        # alternative (empty panel until the worker completes) breaks core
        # Manual Tool behavior on an existing alignments file.
        try:
            # Image-folder sessions seed against their source filenames so
            # sparse alignments (an entry only for ``frame_010.png``) attach
            # to the matching frame index — not lexicographic position 0.
            # Video sessions fall back to the alignment-keys ordering since
            # those keys *are* the canonical frame order.
            if self._session.has_images:
                self._editable.seed_from_handle(
                    self._alignments_handle,
                    frame_names=[frame.name for frame in self._session.frame_list],
                )
            else:
                self._editable.seed_from_handle(self._alignments_handle)
        except Exception:  # noqa: BLE001 - re-surfaced by the startup worker
            logger.exception("Manual Tool synchronous seed failed")
        if self._session.has_images:
            self._thumbnail_panel.set_frames(self._session.frame_list)
            self._thumbnail_panel.setCurrentRow(0)
        elif self._session.is_video_input:
            # Load video metadata synchronously before the provider starts
            # so the worker (a) does not race the provider for the same
            # alignments-file open and (b) the provider sees any persisted
            # pts_time / keyframes payload from disk on first launch.
            try:
                self._video_metadata = self._alignments_handle.video_metadata()
            except Exception:  # noqa: BLE001 - re-surfaced by the startup worker
                logger.exception("Manual Tool video metadata load failed")
                self._video_metadata = None
            self._start_video_provider()
        else:
            self._frame_view.clear_frame(
                "Video input detected. Frame extraction will be wired in a follow-up."
            )
        self._status_label.setText("Manual Tool starting…")
        self._sync_actions()

    def _start_background_startup(self) -> None:
        """Kick off ``ManualStartupWorker`` for async alignments preparation."""
        self._progress_bar = self._build_progress_bar()
        if self._progress_bar is not None:
            self.statusBar().addPermanentWidget(self._progress_bar)
            self._progress_bar.show()
        self._startup_worker = ManualStartupWorker(
            self._alignments_handle, self._editable, parent=self
        )
        self._startup_worker.progress.connect(self._on_startup_progress)
        self._startup_worker.completed.connect(self._on_startup_completed)
        self._startup_worker.failed.connect(self._on_startup_failed)
        logger.info("Manual Tool startup worker scheduled")
        self._emit_console("Manual Tool: preparing session…")
        self._startup_worker.start()

    def _build_progress_bar(self) -> QProgressBar:
        """Return a Qt-native indeterminate progress bar for the status bar."""
        bar = QProgressBar()
        bar.setObjectName("qt-manual-startup-progress")
        bar.setRange(0, 0)
        bar.setMaximumWidth(160)
        bar.setTextVisible(False)
        bar.hide()
        return bar

    def _on_startup_progress(self, stage: str, message: str) -> None:
        """Surface intermediate startup messages to the status bar + console."""
        logger.debug("Manual Tool startup [%s]: %s", stage, message)
        self.statusBar().showMessage(message)
        self._emit_console(f"Manual Tool [{stage}]: {message}")

    def _on_startup_completed(self, has_thumbnails: bool, summary: str) -> None:
        """Finalize UI state once the startup worker reports success."""
        self._startup_complete = True
        thumbs_state = (
            "regenerate forced"
            if self._session.thumb_regenerate
            else ("cached" if has_thumbnails else "needs generation")
        )
        self._metadata_label.setText(
            "\n".join(
                (
                    f"Input: {self._session.frames}",
                    f"Alignments: {self._alignments_handle.path}",
                    f"Frames: {self._frame_summary_text()}",
                    f"Thumbnails: {thumbs_state}",
                )
            )
        )
        if self._session.is_video_input:
            self._video_metadata = self._alignments_handle.video_metadata()
        self._status_label.setText("Manual Tool ready")
        self.statusBar().showMessage(summary, 5000)
        self._hide_progress_bar()
        logger.info("Manual Tool startup complete: %s", summary)
        self._emit_console(f"Manual Tool ready — {summary}")
        # The editable model may now hold seeded faces; repaint + refresh.
        self.refresh_faces()
        self._frame_view.update()
        self._sync_actions()

    def _on_startup_failed(self, message: str) -> None:
        """Surface startup failures through status, console, log and dialog."""
        self._startup_complete = False
        self._status_label.setText("Manual Tool startup failed")
        self.statusBar().showMessage(message, 7000)
        self._hide_progress_bar()
        logger.error("Manual Tool startup failed: %s", message)
        self._emit_console(f"Manual Tool startup failed: {message}")
        QMessageBox.critical(self, "Manual Tool Startup", message)

    def _hide_progress_bar(self) -> None:
        """Hide and detach the indeterminate startup progress widget."""
        if self._progress_bar is not None:
            self._progress_bar.hide()
            self.statusBar().removeWidget(self._progress_bar)
            self._progress_bar = None

    def _emit_console(self, message: str) -> None:
        """Forward a user-facing message to the host shell console, if any."""
        if self._console_logger is None:
            return
        try:
            self._console_logger(message)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Manual Tool console logger raised")

    def _frame_summary_text(self) -> str:
        """Return the metadata summary text for the loaded frame source."""
        if self._session.has_images:
            return str(self._session.frame_count)
        if self._video_metadata is not None and self._video_metadata.is_valid:
            return f"{self._video_metadata.frame_count} (video)"
        return "video input"

    def _start_video_provider(self) -> None:
        """Initialize the async video frame provider for video sessions."""
        self._frame_view.clear_frame("Loading video frames…")
        meta_dict: dict[str, list[int]] | None
        if self._video_metadata is not None and self._video_metadata.is_valid:
            meta_dict = {
                "pts_time": list(self._video_metadata.pts_time),
                "keyframes": list(self._video_metadata.keyframes),
            }
        else:
            meta_dict = None
        self._video_provider = VideoFrameProvider(
            self._session.frames,
            video_meta_data=meta_dict,
            parent=self,
        )
        self._video_provider.count_ready.connect(self._on_video_count_ready)
        self._video_provider.frame_ready.connect(self._on_video_frame_ready)
        self._video_provider.load_failed.connect(self._on_video_load_failed)
        self._video_provider.start()

    def _on_video_count_ready(self, count: int) -> None:
        """Populate the thumbnail list with video frame placeholders."""
        if count <= 0:
            self._frame_view.clear_frame("Video reported zero frames")
            return
        self._video_frames = [
            ManualFrame(index=idx, name=f"frame_{idx:06d}", path=self._session.frames)
            for idx in range(count)
        ]
        self._thumbnail_panel.set_frames(tuple(self._video_frames))
        self._thumbnail_panel.setCurrentRow(0)

    def _on_video_frame_ready(self, index: int, filename: str, image: QImage) -> None:
        """Display a video frame that was decoded on the worker thread."""
        if (
            self._current_frame is not None
            and self._current_frame.index == index
            and self._session.is_video_input
        ):
            self._frame_view.set_image(image, self._current_frame)
            self._status_label.setText(f"Frame {index + 1}: {filename}")

    def _on_video_load_failed(self, index: int, message: str) -> None:
        """Surface a video load failure in the status label and console."""
        logger.warning("Manual Tool video frame %s failed: %s", index, message)
        self._status_label.setText(f"Failed to load frame {index}: {message}")

    def _on_unsaved_changed(self, dirty: bool) -> None:
        """Forward editor-state unsaved changes to UI signals."""
        self.dirty_changed.emit(bool(dirty))
        self._sync_actions()

    def _on_face_selected(self, face_index: int) -> None:
        """Propagate active face selection to the shared editor state.

        A negative ``face_index`` (emitted when the panel is cleared) is
        propagated as ``-1`` so editor state and the overlay do not keep
        pointing at a face that no longer exists.
        """
        self._editor_state.set("face_index", face_index if face_index >= 0 else -1)
        self._sync_actions()

    def _current_frame_index(self) -> int:
        """Return the sorted-frame index of the active frame, or -1 if none."""
        return -1 if self._current_frame is None else self._current_frame.index

    def _on_editable_changed(self, frame_index: int) -> None:
        """React to any change in the editable alignment model.

        The face panel mirrors the alignments-file thumbnails (read-only) and
        emits ``face_selected(-1)`` whenever it is empty.  Refresh it first,
        then resync ``editor_state.face_index`` against the editable model so
        a programmatic add does not leave us pointed at ``-1``.  Dirty state
        is derived from the model's undo stack so any edit (including ones
        made on a non-current frame) marks the session unsaved, and undoing
        back to the saved snapshot drops the dirty flag automatically.
        """
        self.mark_dirty(self._editable.can_undo)
        if frame_index != self._current_frame_index():
            return
        self.refresh_faces()
        self._frame_view.update()
        face_count = self._editable.face_count(frame_index)
        active = self._editor_state.face_index
        if face_count > 0 and active < 0:
            self._editor_state.set("face_index", 0)
        elif face_count == 0 and active >= 0:
            self._editor_state.set("face_index", -1)
        self._sync_actions()

    def _on_frame_clicked(self, point: QPointF) -> None:
        """Hit-test the editable model at the clicked source-image point."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return
        face_index = self._editable.hit_test(frame_index, point.x(), point.y())
        if face_index is None:
            return
        self._editor_state.set("face_index", face_index)
        self._face_panel.select_face(face_index)

    def refresh_faces(self) -> None:
        """Rebuild the face panel from the editable model + persisted thumbs.

        The editable model is the source of truth for what's currently on
        screen — add/delete/move edits must appear in the panel before
        Save runs.  Thumbnails come from the alignments file when the face
        is unchanged from its persisted bbox; otherwise we surface a
        placeholder so a stale JPEG is never shown next to a moved bbox.
        """
        if self._current_frame is None:
            self._face_panel.set_faces(())
            return
        frame_index = self._current_frame.index
        editable_faces = self._editable.faces(frame_index)
        if not editable_faces:
            self._face_panel.set_faces(())
            return
        frame_name = self._frame_name_for_index(frame_index) or self._current_frame.name
        # Look up persisted thumbnails by frame *name* — the editable model is
        # anchored to the source frame list, but the alignments file may be
        # sparse, so sorted-index lookups (``faces_for_frame``) can attach the
        # wrong frame's thumbnail to a given editable index.
        persisted = {
            entry.face_index: entry
            for entry in self._alignments_handle.faces_for_frame_name(
                frame_name, frame_index=frame_index
            )
        }
        entries = []
        for face in editable_faces:
            previous = persisted.get(face.face_index)
            entries.append(
                FaceThumbnail(
                    frame_index=frame_index,
                    frame_name=frame_name,
                    face_index=face.face_index,
                    thumbnail_jpeg=previous.thumbnail_jpeg if previous else b"",
                )
            )
        self._face_panel.set_faces(entries)

    def _thumbnail_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Display the selected source frame."""
        if current is None:
            return
        index = current.data(Qt.UserRole)
        if not isinstance(index, int):
            return
        if self._session.has_images:
            if index >= len(self._session.frame_list):
                return
            frame = self._session.frame_list[index]
            self._current_frame = frame
            self._editor_state.set("frame_index", frame.index)
            self.frame_changed.emit(frame.index)
            if self._frame_view.load_frame(frame):
                self._status_label.setText(
                    f"Frame {frame.index + 1} of {self._session.frame_count}: {frame.name}"
                )
            self.refresh_faces()
        elif self._video_frames:
            if index >= len(self._video_frames):
                return
            frame = self._video_frames[index]
            self._current_frame = frame
            self._editor_state.set("frame_index", frame.index)
            self.frame_changed.emit(frame.index)
            if self._video_provider is not None:
                self._video_provider.request_frame(frame.index)
                self._status_label.setText(
                    f"Loading frame {frame.index + 1} of {len(self._video_frames)}"
                )
            self.refresh_faces()

    def _previous_frame(self) -> None:
        """Select previous frame."""
        row = self._thumbnail_panel.currentRow()
        if row > 0:
            self._thumbnail_panel.setCurrentRow(row - 1)

    def _next_frame(self) -> None:
        """Select next frame."""
        row = self._thumbnail_panel.currentRow()
        if 0 <= row < self._thumbnail_panel.count() - 1:
            self._thumbnail_panel.setCurrentRow(row + 1)

    def _sync_actions(self) -> None:
        """Update action availability from session, selection and dirty state."""
        if not self._actions:
            return
        frame_total = self._thumbnail_panel.count()
        current_row = self._thumbnail_panel.currentRow()
        has_frames = frame_total > 0
        not_first = has_frames and current_row > 0
        not_last = has_frames and 0 <= current_row < frame_total - 1
        frame_index = self._current_frame_index()
        editable_count = self._editable.face_count(frame_index) if frame_index >= 0 else 0
        face_selected = self._editor_state.face_index >= 0 and (
            editable_count > 0 or bool(self._face_panel.faces)
        )
        has_frame_loaded = self._frame_view.has_frame
        editor_mode = self._editor_state.editor_mode

        availability: dict[str, bool] = {
            "save": self._editor_state.unsaved,
            "revert_frame": self._editor_state.edited
            or self._editor_state.unsaved
            or self._editable.can_undo,
            "first_frame": not_first,
            "previous_frame": not_first,
            "next_frame": not_last,
            "last_frame": not_last,
            "play_pause": has_frames,
            "copy_prev_face": not_first,
            "copy_next_face": not_last,
            "delete_face": face_selected and editable_count > 0,
            "add_face": has_frame_loaded,
            "undo_edit": self._editable.can_undo,
            "redo_edit": self._editable.can_redo,
            "cycle_filter": has_frames,
            "cycle_annotation": has_frames,
            "set_view_mode": editor_mode != "View",
            "set_boundingbox_mode": editor_mode != "BoundingBox",
            "set_extractbox_mode": editor_mode != "ExtractBox",
            "set_landmarks_mode": editor_mode != "Landmarks",
            "set_mask_mode": editor_mode != "Mask",
            "zoom_in": has_frames,
            "zoom_out": has_frames,
            "reset_view": has_frames,
            "legacy_tool": bool(self._legacy_args),
        }
        for key, enabled in availability.items():
            action = self._actions.get(key)
            if action is not None:
                action.setEnabled(enabled)

    def _on_editor_mode_changed(self, _mode: object) -> None:
        """Refresh action availability when the editor mode flips."""
        self._sync_actions()


__all__ = [
    "MANUAL_ACTIONS",
    "FaceThumbnailPanel",
    "FrameViewport",
    "ManualAction",
    "ManualFrameOverlay",
    "ManualFrameView",
    "ManualThumbnailPanel",
    "ManualToolWindow",
    "VideoFrameProvider",
]
