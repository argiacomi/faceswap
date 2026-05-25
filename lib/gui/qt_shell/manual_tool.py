#!/usr/bin/env python3
"""Native Qt Manual Tool shell.

This is the first Qt-native surface for the Manual Tool migration.  It provides
startup validation, frame display with zoom/pan/overlay seams, thumbnail/
selection, navigation, dirty-state lifecycle, async video-frame loading and a
legacy fallback launcher without importing ``tkinter`` on the Qt path.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import typing as T
from dataclasses import dataclass

from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QIcon,
    QImage,
    QIntValidator,
    QKeyEvent,
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
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
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

_FACE_GRID_SIZES: dict[str, int] = {
    "Tiny": 48,
    "Small": 64,
    "Medium": 96,
    "Large": 128,
    "Extra Large": 160,
}

_FACE_GRID_ENTRY_ROLE = Qt.UserRole
_FACE_GRID_ACTIVE_FRAME_ROLE = Qt.UserRole + 1
_FACE_GRID_ACTIVE_FACE_ROLE = Qt.UserRole + 2
_FACE_GRID_HOVER_ROLE = Qt.UserRole + 3


@dataclass(frozen=True)
class FaceGridEntry:
    """One visible face in the filtered-session cross-frame grid."""

    frame_index: int
    frame_name: str
    face_index: int
    thumbnail: FaceThumbnail | None
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class FaceGridRenderRequest:
    """Testable record of one thumbnail render path and overlay state."""

    frame_index: int
    face_index: int
    icon_size: int
    show_mesh: bool
    show_mask: bool
    mask_type: str
    mask_opacity: int


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
    _LANDMARK_SELECTED_COLOR = QColor("#ffb000")
    _LANDMARK_RADIUS = 2.0
    _LANDMARK_SELECTED_RADIUS = 3.5
    _HANDLE_SIZE = 7.0  # square handle side length in widget pixels
    _HANDLE_FILL = QColor("#ffffff")
    _HANDLE_EDGE = QColor("#11191c")
    # Source-coordinate offsets for the 8 resize handles drawn on the active
    # face, expressed as (anchor_x_fraction, anchor_y_fraction) relative to
    # the bbox. ``0.5`` is the midpoint. Order matches Qt's
    # ``size*Cursor`` cursor naming so a hit-test can drive cursors directly.
    HANDLE_OFFSETS: T.ClassVar[tuple[tuple[str, float, float], ...]] = (
        ("nw", 0.0, 0.0),
        ("n", 0.5, 0.0),
        ("ne", 1.0, 0.0),
        ("e", 1.0, 0.5),
        ("se", 1.0, 1.0),
        ("s", 0.5, 1.0),
        ("sw", 0.0, 1.0),
        ("w", 0.0, 0.5),
    )

    def __init__(
        self,
        model: ManualEditableAlignments,
        *,
        frame_index_provider: T.Callable[[], int],
    ) -> None:
        self._model = model
        self._frame_index_provider = frame_index_provider
        self._active_face: int | None = None
        # Selected landmark indices for the active face (Landmark editor, #103).
        self._selected_landmarks: frozenset[int] = frozenset()
        # Mask editor (#101) render seam — the host installs callables that
        # return the active mask type / opacity / colour for the overlay.
        self._mask_type_provider: T.Callable[[], str] | None = None
        self._mask_opacity_provider: T.Callable[[], int] | None = None
        self._mask_show_provider: T.Callable[[], bool] | None = None

    def set_active(self, face_index: int | None) -> None:
        """Mark a face as the active selection for highlight rendering."""
        self._active_face = face_index
        # Clear the landmark selection whenever the active face changes; the
        # selection set is per-face and would otherwise leak across switches.
        self._selected_landmarks = frozenset()

    @property
    def active_face(self) -> int | None:
        """Return the currently highlighted ``face_index`` (or ``None``)."""
        return self._active_face

    def set_selected_landmarks(self, indices: T.Iterable[int]) -> None:
        """Set the landmark selection set used to render highlight points."""
        self._selected_landmarks = frozenset(int(i) for i in indices)

    @property
    def selected_landmarks(self) -> frozenset[int]:
        """Return the active face's currently selected landmark indices."""
        return self._selected_landmarks

    def install_mask_render_seam(
        self,
        *,
        mask_type_provider: T.Callable[[], str],
        mask_opacity_provider: T.Callable[[], int],
        mask_show_provider: T.Callable[[], bool],
    ) -> None:
        """Hook the Mask editor render seam (#101).

        ``mask_type_provider`` returns the mask type to display for the
        active face.  ``mask_opacity_provider`` returns ``0..100``.
        ``mask_show_provider`` toggles the overlay layer on/off so other
        editor modes (View / BBox / Landmark / Extract) don't render
        masks unintentionally.
        """
        self._mask_type_provider = mask_type_provider
        self._mask_opacity_provider = mask_opacity_provider
        self._mask_show_provider = mask_show_provider

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
                painter.setPen(Qt.NoPen)
                selected = self._selected_landmarks if is_active else frozenset()
                for lm_index, (lx, ly) in enumerate(face.landmarks):
                    point = viewport.source_to_widget(lx, ly)
                    if lm_index in selected:
                        painter.setBrush(QBrush(self._LANDMARK_SELECTED_COLOR))
                        radius = self._LANDMARK_SELECTED_RADIUS
                    else:
                        painter.setBrush(QBrush(self._LANDMARK_COLOR))
                        radius = self._LANDMARK_RADIUS
                    painter.drawEllipse(point, radius, radius)
            if is_active:
                self._paint_mask_overlay(painter, viewport, face)
                self._draw_handles(painter, rect)
            painter.restore()

    def _paint_mask_overlay(
        self,
        painter: QPainter,
        viewport: FrameViewport,
        face: T.Any,
    ) -> None:
        """Render the active face's mask as a colour-tinted alpha layer (#101).

        The mask is stored at ``MASK_STORED_SIZE`` in stored-space and is
        projected onto the face's source-coordinate bbox via the same
        axis-aligned mapping ``paint_mask_stroke`` uses.  Opacity is
        ``mask_opacity_provider() / 100`` so the user can blend the
        overlay against the underlying frame.
        """
        if (
            self._mask_type_provider is None
            or self._mask_opacity_provider is None
            or self._mask_show_provider is None
            or not self._mask_show_provider()
        ):
            return
        mask_type = self._mask_type_provider()
        if not mask_type:
            return
        frame_index = self._frame_index_provider()
        mask = self._model.get_mask(frame_index, int(face.face_index), mask_type)
        if mask is None:
            return
        opacity_pct = max(0, min(100, int(self._mask_opacity_provider())))
        if opacity_pct == 0:
            return
        # Build a QImage from the uint8 mask with the editor's colour tint.
        try:
            mask_array = mask
            if mask_array.size == 0:
                return
            height, width = mask_array.shape[:2]
            from PySide6.QtGui import QImage

            argb = bytearray(width * height * 4)
            tint_r, tint_g, tint_b = (255, 80, 80)
            alpha_factor = opacity_pct / 100.0
            mv = memoryview(argb)
            data = mask_array.tobytes()
            for i, value in enumerate(data):
                base = i * 4
                if value == 0:
                    mv[base : base + 4] = b"\x00\x00\x00\x00"
                    continue
                a = int(min(255, value * alpha_factor))
                # Qt QImage in Format_ARGB32 is BGRA in memory on little-endian.
                mv[base] = tint_b
                mv[base + 1] = tint_g
                mv[base + 2] = tint_r
                mv[base + 3] = a
            image = QImage(bytes(argb), width, height, width * 4, QImage.Format_ARGB32)
        except Exception:  # pragma: no cover - defensive
            return
        rect = viewport.source_rect_to_widget(*face.bbox)
        painter.save()
        painter.setOpacity(1.0)
        painter.drawImage(rect, image)
        painter.restore()

    def _draw_handles(self, painter: QPainter, bbox: QRectF) -> None:
        """Draw eight square resize handles around the active bbox."""
        painter.save()
        edge_pen = QPen(self._HANDLE_EDGE)
        edge_pen.setWidthF(1.0)
        painter.setPen(edge_pen)
        painter.setBrush(QBrush(self._HANDLE_FILL))
        for _name, fx, fy in self.HANDLE_OFFSETS:
            cx = bbox.x() + bbox.width() * fx
            cy = bbox.y() + bbox.height() * fy
            half = self._HANDLE_SIZE / 2.0
            painter.drawRect(QRectF(cx - half, cy - half, self._HANDLE_SIZE, self._HANDLE_SIZE))
        painter.restore()

    @classmethod
    def handle_at(
        cls,
        bbox: QRectF,
        widget_point: QPointF,
        tolerance: float = 0.0,
    ) -> str | None:
        """Return the handle name (``nw``/``n``/...) under ``widget_point``.

        ``tolerance`` adds extra pixels around each handle for forgiving hit
        targets when the source-image zoom is small. Returns ``None`` when
        the point misses every handle.
        """
        half = cls._HANDLE_SIZE / 2.0 + max(0.0, tolerance)
        for name, fx, fy in cls.HANDLE_OFFSETS:
            cx = bbox.x() + bbox.width() * fx
            cy = bbox.y() + bbox.height() * fy
            if (
                cx - half <= widget_point.x() <= cx + half
                and cy - half <= widget_point.y() <= cy + half
            ):
                return name
        return None

    @classmethod
    def landmark_at(
        cls,
        landmarks: T.Sequence[tuple[float, float]],
        source_point: tuple[float, float],
        *,
        tolerance: float = 0.0,
    ) -> int | None:
        """Return the index of the landmark within ``tolerance`` of ``source_point``.

        Coordinates are in source pixels (no widget mapping). When multiple
        landmarks fall inside the tolerance radius the *closest* index wins,
        so users in dense regions (eye/lip contours) get the expected point.
        Returns ``None`` when nothing is within reach.
        """
        if not landmarks:
            return None
        sx, sy = source_point
        radius = cls._LANDMARK_RADIUS + max(0.0, tolerance)
        radius_sq = radius * radius
        best_index: int | None = None
        best_dist_sq = radius_sq
        for index, (lx, ly) in enumerate(landmarks):
            dx = lx - sx
            dy = ly - sy
            dist_sq = dx * dx + dy * dy
            if dist_sq <= best_dist_sq:
                best_dist_sq = dist_sq
                best_index = index
        return best_index

    @staticmethod
    def landmarks_in_rect(
        landmarks: T.Sequence[tuple[float, float]],
        source_rect: QRectF,
    ) -> tuple[int, ...]:
        """Return indices of every landmark inside ``source_rect`` (inclusive).

        Used by the Landmark editor's marquee selection gesture (#103).
        An empty rect (zero width or height) returns an empty tuple.
        """
        if source_rect.width() <= 0.0 or source_rect.height() <= 0.0:
            return ()
        x0 = source_rect.x()
        y0 = source_rect.y()
        x1 = x0 + source_rect.width()
        y1 = y0 + source_rect.height()
        return tuple(
            index for index, (lx, ly) in enumerate(landmarks) if x0 <= lx <= x1 and y0 <= ly <= y1
        )


class ManualFrameView(QWidget):
    """Manual Tool frame display with pan/zoom and overlay seams."""

    view_changed = Signal()
    frame_loaded = Signal(ManualFrame)
    clicked_at = Signal(QPointF)
    """Emitted with source-image coordinates when the view is left-clicked."""
    face_move_requested = Signal(int, float, float)
    """Emitted at the end of a bbox drag: (face_index, dx, dy) in source pixels."""
    face_resize_requested = Signal(int, QRectF)
    """Emitted at the end of a handle drag: (face_index, new_bbox in source pixels)."""
    face_add_requested = Signal(QRectF)
    """Emitted at the end of an empty-space drag/click in BBox add mode."""
    face_context_menu_requested = Signal(int, QPointF)
    """Emitted on right-click over an existing face: (face_index, global pos)."""
    landmark_move_requested = Signal(int, int, float, float)
    """``(face_index, landmark_index, new_source_x, new_source_y)`` on single-point release."""
    landmarks_move_requested = Signal(int, object, float, float)
    """``(face_index, tuple(indices), dx, dy)`` on group-move release."""
    landmarks_select_requested = Signal(int, object)
    """``(face_index, tuple(indices))`` on marquee release.  Empty tuple clears."""
    face_scale_requested = Signal(int, float)
    """``(face_index, scale_factor)`` on Extract Box corner-drag release (#102)."""
    face_rotate_requested = Signal(int, float)
    """``(face_index, angle_radians)`` on Extract Box rotation-zone drag release (#102)."""
    mask_paint_requested = Signal(int, float, float, bool)
    """``(face_index, source_x, source_y, invert)`` during a Mask editor stroke (#101).

    Emitted on press *and* every move so painting is continuous; the host
    handler stamps the brush at each emit.  ``invert`` is ``True`` when
    ``Ctrl`` is held to flip Draw/Erase for the duration of one gesture.
    """

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
        # ``StrongFocus`` lets the frame view receive keyboard focus on click
        # so frame-view-scoped shortcuts (e.g. arrow-key nudge) fire only when
        # the user is interacting with the frame, not the face panel.
        self.setFocusPolicy(Qt.StrongFocus)
        self._source: QPixmap = QPixmap()
        self._zoom: float = 1.0
        self._offset: QPointF = QPointF(0.0, 0.0)
        self._drag_anchor: QPoint | None = None
        self._drag_origin: QPointF = QPointF(0.0, 0.0)
        self._empty_message: str = "No frame selected"
        self._overlays: list[OverlayPainter] = []
        # Editor seam: when an active face + bbox provider are installed the
        # frame view drives bbox move/resize gestures and emits face_*_requested
        # signals on commit. Without these the view falls back to pure pan.
        self._active_bbox_provider: T.Callable[[], QRectF | None] | None = None
        self._active_face_provider: T.Callable[[], int | None] | None = None
        # ``add_mode_provider`` returns ``True`` when an empty-space click/drag
        # should create a new face (BoundingBox editor). ``face_hit_provider``
        # maps a source point to an existing face_index for right-click context
        # menus.  Both default to disabled so existing tests keep their
        # pure-pan behavior.
        self._add_mode_provider: T.Callable[[], bool] | None = None
        self._face_hit_provider: T.Callable[[float, float], int | None] | None = None
        # Landmark editor seams (#103) — when installed, landmark-mode pointer
        # gestures hit-test individual points and emit landmark_*_requested.
        self._landmark_mode_provider: T.Callable[[], bool] | None = None
        self._landmark_provider: T.Callable[[], T.Sequence[tuple[float, float]] | None] | None = (
            None
        )
        self._landmark_selection_provider: T.Callable[[], frozenset[int]] | None = None
        self._edit_drag_mode: str | None = None
        # Modes: "move" | "resize" | "add" | "landmark" | "landmark_group"
        # | "landmark_marquee"
        self._edit_drag_handle: str | None = None
        self._edit_drag_source_anchor: QPointF | None = None
        self._edit_drag_original_bbox: QRectF | None = None
        self._edit_drag_current_bbox: QRectF | None = None
        # Landmark-drag state — populated by ``_begin_landmark_drag``.
        self._landmark_drag_index: int | None = None
        self._landmark_drag_indices: tuple[int, ...] = ()
        self._landmark_drag_origins: tuple[tuple[float, float], ...] = ()
        # Extract Box editor (#102) state — set by ``_begin_extract_drag``.
        self._extract_mode_provider: T.Callable[[], bool] | None = None
        self._extract_drag_center: QPointF | None = None
        self._extract_drag_start_radius: float = 0.0
        self._extract_drag_start_angle: float = 0.0
        self._extract_drag_scale: float = 1.0
        self._extract_drag_angle: float = 0.0
        # Mask editor (#101) seam.  Active when ``editor_mode == "Mask"``;
        # press + drag inside the active face's bbox emits
        # :attr:`mask_paint_requested` continuously for the host to stamp.
        self._mask_mode_provider: T.Callable[[], bool] | None = None
        self._mask_drag_active: bool = False
        self._mask_drag_invert: bool = False
        # Brush preview (#101 closure) — when ``_brush_provider`` returns a
        # ``(radius_source_px, mode)`` tuple while the mask editor is active
        # *and* the pointer is inside the active face's bbox, the view draws
        # a circular cursor preview so the user can see brush size + mode
        # before clicking.  Hidden in any other state.
        self._brush_provider: T.Callable[[], tuple[float, str] | None] | None = None
        self._brush_preview_source_point: QPointF | None = None
        self._brush_preview_radius: float = 0.0
        self._brush_preview_mode: str = "draw"

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

    def current_frame_array(self) -> T.Any:
        """Return the current source frame as a numpy ``(H, W, 3)`` uint8 array.

        Used by the Aligner integration (#104) to feed the pipeline without
        re-reading the file from disk.  Returns ``None`` when no frame is
        loaded.  The conversion goes through ``QImage`` so the same path
        works for both image-folder and video sessions.
        """
        if self._source.isNull():
            return None
        try:
            import numpy as np

            image = self._source.toImage().convertToFormat(QImage.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.constBits()
            buffer = bytes(ptr)
            # ``Format_RGB888`` rows are padded to 4-byte multiples; strip the
            # padding by selecting the meaningful width × 3 prefix per row.
            row_stride = image.bytesPerLine()
            array = np.frombuffer(buffer, dtype=np.uint8)
            array = array.reshape(height, row_stride)[:, : width * 3]
            return array.reshape(height, width, 3).copy()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Manual Tool: current_frame_array conversion failed")
            return None

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

    def magnify_to_source_rect(self, source_rect: QRectF) -> bool:
        """Zoom + pan so ``source_rect`` fills the viewport (Landmark magnify).

        ``source_rect`` is in source-image pixel coordinates.  Returns
        ``False`` when no source is loaded or the rect is degenerate.
        """
        src_w, src_h = self.source_size
        if src_w <= 0 or src_h <= 0 or source_rect.width() <= 0.0 or source_rect.height() <= 0.0:
            return False
        fit_w, fit_h = self._fit_size()
        if fit_w <= 0.0 or fit_h <= 0.0:
            return False
        # Choose the zoom that maps the source-rect's longer side to the widget
        # axis it fills (matching Qt's ``KeepAspectRatio`` semantics).
        zoom_x = (src_w / source_rect.width()) * (self.width() / fit_w if fit_w else 1.0)
        zoom_y = (src_h / source_rect.height()) * (self.height() / fit_h if fit_h else 1.0)
        zoom = min(zoom_x, zoom_y)
        clamped = max(self._MIN_ZOOM, min(self._MAX_ZOOM, zoom))
        self._zoom = clamped
        # Reset pan, then center the source-rect under the viewport.  ``_target_rect``
        # already accounts for the centred offset, so we compute the
        # source-rect's centre in widget coordinates and shift by the
        # difference from the widget centre.
        self._offset = QPointF(0.0, 0.0)
        target = self._target_rect()
        center_x = target.x() + target.width() * (
            (source_rect.x() + source_rect.width() / 2.0) / src_w
        )
        center_y = target.y() + target.height() * (
            (source_rect.y() + source_rect.height() / 2.0) / src_h
        )
        self._offset = QPointF(
            self.width() / 2.0 - center_x,
            self.height() / 2.0 - center_y,
        )
        self._clamp_offset()
        self.update()
        self.view_changed.emit()
        return True

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

    def install_editor_seams(
        self,
        *,
        active_face_provider: T.Callable[[], int | None],
        active_bbox_provider: T.Callable[[], QRectF | None],
        add_mode_provider: T.Callable[[], bool] | None = None,
        face_hit_provider: T.Callable[[float, float], int | None] | None = None,
    ) -> None:
        """Install editor callbacks so pointer drags drive face move/resize.

        ``active_face_provider`` returns the currently selected ``face_index``
        (or ``None`` when nothing is selected) and ``active_bbox_provider``
        returns its source-coordinate :class:`QRectF`. Without these the view
        falls back to plain pan behaviour, preserving existing test contracts.

        ``add_mode_provider`` (optional) returns ``True`` when an empty-space
        click/drag should create a new face — set by the host window when the
        Bounding Box editor is active.

        ``face_hit_provider`` (optional) maps a source-image point to an
        existing face_index so right-clicking a face emits
        :attr:`face_context_menu_requested` for the host to surface a context
        menu (Delete Face, etc.).
        """
        self._active_face_provider = active_face_provider
        self._active_bbox_provider = active_bbox_provider
        self._add_mode_provider = add_mode_provider
        self._face_hit_provider = face_hit_provider

    def install_landmark_seams(
        self,
        *,
        landmark_mode_provider: T.Callable[[], bool],
        landmark_provider: T.Callable[[], T.Sequence[tuple[float, float]] | None],
        landmark_selection_provider: T.Callable[[], frozenset[int]],
    ) -> None:
        """Install Landmark editor seams (#103).

        ``landmark_mode_provider`` returns ``True`` while the editor mode is
        ``"Landmarks"`` — gating point-drag, marquee, and group-move gestures.
        ``landmark_provider`` returns the active face's landmark sequence (or
        ``None`` when no face is active).  ``landmark_selection_provider``
        returns the currently selected landmark indices so group-move uses
        the same set the marquee defined.
        """
        self._landmark_mode_provider = landmark_mode_provider
        self._landmark_provider = landmark_provider
        self._landmark_selection_provider = landmark_selection_provider

    def install_extract_seams(
        self,
        *,
        extract_mode_provider: T.Callable[[], bool],
    ) -> None:
        """Install Extract Box editor seam (#102).

        ``extract_mode_provider`` returns ``True`` while the editor mode is
        ``"ExtractBox"``.  When active, the bbox handle/body hit-tests fan
        out into translate / scale / rotate drags instead of the bbox
        move/resize used in the View / BoundingBox editors.
        """
        self._extract_mode_provider = extract_mode_provider

    def install_mask_seams(
        self,
        *,
        mask_mode_provider: T.Callable[[], bool],
        brush_provider: T.Callable[[], tuple[float, str] | None] | None = None,
    ) -> None:
        """Install Mask editor seams (#101).

        ``mask_mode_provider`` returns ``True`` while ``editor_mode == "Mask"``.
        When active, press + drag inside the active face's bbox emits
        :attr:`mask_paint_requested` continuously so the host can stamp
        each brush position.  ``Ctrl+drag`` flips Draw/Erase for the
        duration of the gesture (legacy Tk Manual Tool parity).

        ``brush_provider`` (optional) returns ``(radius_source_px, mode)``
        — when present the frame view draws a circular cursor preview at
        the pointer while in Mask mode + over an active face's bbox.
        ``mode`` is ``"draw"`` or ``"erase"``.
        """
        self._mask_mode_provider = mask_mode_provider
        self._brush_provider = brush_provider

    @property
    def brush_preview(self) -> dict | None:
        """Return the visible brush preview state for tests/inspection.

        ``None`` means no preview is currently shown.  When visible, the
        dict contains ``source_point`` (QPointF), ``radius`` (source px),
        and ``mode`` (``"draw"`` or ``"erase"``).
        """
        if self._brush_preview_source_point is None:
            return None
        return {
            "source_point": QPointF(self._brush_preview_source_point),
            "radius": float(self._brush_preview_radius),
            "mode": str(self._brush_preview_mode),
        }

    def edit_drag_preview_bbox(self) -> QRectF | None:
        """Return the in-progress drag bbox in source coordinates, if any.

        Tests and overlays use this to render a preview rectangle before the
        drag is committed.
        """
        if self._edit_drag_current_bbox is None:
            return None
        return QRectF(self._edit_drag_current_bbox)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa:N802
        """Pointer-down dispatch: context menu, resize handle, bbox move, add, or pan."""
        if self._source.isNull():
            super().mousePressEvent(event)
            return
        position = event.position()
        source_point = self._widget_to_source(position)
        if event.button() == Qt.RightButton:
            if self._emit_context_menu(source_point, event.globalPosition()):
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        # Always emit clicked_at so selection on empty-space click is preserved.
        if source_point is not None:
            self.clicked_at.emit(source_point)
        # Mask editor (#101) takes precedence in mask mode so the user can
        # paint continuously inside the active face's bbox.  Ctrl+drag
        # inverts Draw/Erase for the gesture (Tk parity).
        if self._begin_mask_drag(position, source_point, event):
            event.accept()
            return
        # Landmark editor (#103) takes precedence in landmark mode so the user
        # can drag individual points or marquee-select inside the bbox.
        if self._begin_landmark_drag(position, source_point, event):
            event.accept()
            return
        # Extract Box editor (#102) maps the bbox handles + halo to
        # translate/scale/rotate drags instead of bbox move/resize.
        if self._begin_extract_drag(position):
            event.accept()
            return
        if self._begin_edit_drag(position):
            event.accept()
            return
        if self._begin_add_drag(position, source_point):
            event.accept()
            return
        # No edit/add drag available — start pan.
        self._drag_anchor = position.toPoint()
        self._drag_origin = QPointF(self._offset)
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa:N802
        """Dispatch pointer motion to active drag mode or update hover cursor."""
        position = event.position()
        if self._mask_drag_active:
            self._continue_mask_drag(position)
            event.accept()
            return
        if self._edit_drag_mode is not None:
            self._update_edit_drag(position)
            event.accept()
            return
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
        """Commit an active edit drag or end the pan drag."""
        if self._mask_drag_active:
            self._end_mask_drag()
            self._update_hover_cursor(event.position())
            event.accept()
            return
        if self._edit_drag_mode is not None:
            self._commit_edit_drag()
            self._update_hover_cursor(event.position())
            event.accept()
            return
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
        self._paint_add_preview(painter, viewport)
        self._paint_brush_preview(painter, viewport)

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
        """Cursor priority: resize handle > bbox body > pannable > default."""
        # Always refresh the Mask brush preview before falling through to
        # the cursor-shape ladder.  ``_update_brush_preview`` schedules a
        # repaint when the preview state changes.
        self._update_brush_preview(position)
        if self._source.isNull():
            self.setCursor(Qt.ArrowCursor)
            return
        in_extract_mode = bool(self._extract_mode_provider and self._extract_mode_provider())
        # Hover an active-face resize handle?
        widget_bbox = self._active_bbox_widget_rect()
        if widget_bbox is not None:
            handle = ManualFrameOverlay.handle_at(widget_bbox, position, tolerance=2.0)
            if handle is not None:
                # Extract Box scale-on-corner reuses the diag/ver/hor sizing
                # cursors so the user sees the same shape they would for a
                # normal resize; the distinction is in the dispatch table.
                self.setCursor(self._cursor_for_handle(handle))
                return
            if widget_bbox.contains(position):
                self.setCursor(Qt.SizeAllCursor)
                return
            if in_extract_mode:
                # Outside the bbox but inside the rotation halo (#102).
                halo = QRectF(
                    widget_bbox.x() - self._EXTRACT_ROTATION_BAND_PX,
                    widget_bbox.y() - self._EXTRACT_ROTATION_BAND_PX,
                    widget_bbox.width() + 2 * self._EXTRACT_ROTATION_BAND_PX,
                    widget_bbox.height() + 2 * self._EXTRACT_ROTATION_BAND_PX,
                )
                if halo.contains(position):
                    # No native Qt rotate cursor — closed-hand reads as
                    # "I will turn this".
                    self.setCursor(Qt.ClosedHandCursor)
                    return
        if self._zoom > self._MIN_ZOOM and self._target_rect().contains(position):
            self.setCursor(Qt.OpenHandCursor)
            return
        self.setCursor(Qt.ArrowCursor)

    @staticmethod
    def _cursor_for_handle(handle: str) -> Qt.CursorShape:
        """Map a handle name to a Qt sizing cursor."""
        if handle in ("nw", "se"):
            return Qt.SizeFDiagCursor
        if handle in ("ne", "sw"):
            return Qt.SizeBDiagCursor
        if handle in ("n", "s"):
            return Qt.SizeVerCursor
        return Qt.SizeHorCursor

    def _update_brush_preview(self, position: QPointF) -> None:
        """Track the Mask brush preview state for the pointer position.

        Visibility rules (#101 closure):

        * Hidden when the Mask editor is not active.
        * Hidden when the brush provider isn't installed or refuses to
          return a brush spec.
        * Hidden when no source frame is loaded.
        * Hidden when no active face exists or the pointer falls outside
          its source-coordinate bbox.
        * Hidden during an in-progress mask drag (the painted stroke is
          already visible feedback).

        Otherwise: records the source-space pointer position, radius and
        draw/erase mode, and schedules a repaint so the new circle lands.
        """
        previous = self._brush_preview_source_point is not None
        cleared = self._clear_brush_preview_if_no_active_modes()
        if cleared and previous:
            self.update()
            return
        if cleared:
            return
        source_point = self._widget_to_source(position)
        bbox = self._active_bbox_source_rect()
        if source_point is None or bbox is None or not bbox.contains(source_point):
            if previous:
                self._brush_preview_source_point = None
                self.update()
            return
        assert (
            self._brush_provider is not None
        )  # narrowed by _clear_brush_preview_if_no_active_modes
        spec = self._brush_provider()
        if spec is None:
            if previous:
                self._brush_preview_source_point = None
                self.update()
            return
        radius, mode = spec
        self._brush_preview_source_point = QPointF(source_point)
        self._brush_preview_radius = max(0.5, float(radius))
        self._brush_preview_mode = str(mode)
        self.update()

    def _clear_brush_preview_if_no_active_modes(self) -> bool:
        """Wipe brush preview when Mask mode / providers are inactive.

        Returns ``True`` when the preview was cleared (caller may need to
        request a repaint).
        """
        if (
            self._mask_mode_provider is None
            or not self._mask_mode_provider()
            or self._brush_provider is None
            or self._source.isNull()
            or self._mask_drag_active
        ):
            if self._brush_preview_source_point is not None:
                self._brush_preview_source_point = None
            return True
        return False

    def leaveEvent(self, event: T.Any) -> None:  # noqa:N802 - Qt slot
        """Hide the brush preview when the pointer leaves the view (#101)."""
        if self._brush_preview_source_point is not None:
            self._brush_preview_source_point = None
            self.update()
        super().leaveEvent(event)

    def _active_bbox_source_rect(self) -> QRectF | None:
        """Resolve the current active face's source-coordinate bbox."""
        if self._active_face_provider is None or self._active_bbox_provider is None:
            return None
        if self._active_face_provider() is None:
            return None
        return self._active_bbox_provider()

    def _active_bbox_widget_rect(self) -> QRectF | None:
        """Project the active source bbox into widget coordinates."""
        source = self._active_bbox_source_rect()
        if source is None or self._source.isNull():
            return None
        viewport = FrameViewport(
            source_size=(self._source.width(), self._source.height()),
            target_rect=self._target_rect(),
            zoom=self._zoom,
        )
        return viewport.source_rect_to_widget(
            source.x(), source.y(), source.width(), source.height()
        )

    _EXTRACT_ROTATION_BAND_PX = 24.0
    """Widget-pixel band outside the bbox that triggers rotation drags."""

    def _begin_mask_drag(
        self,
        position: QPointF,
        source_point: QPointF | None,
        event: QMouseEvent,
    ) -> bool:
        """Try to start a Mask editor stroke (#101).

        Requires the active face to have a bbox covering ``source_point``.
        Press + every subsequent move emit :attr:`mask_paint_requested` so
        the host can stamp the brush continuously.  ``Ctrl`` held at press
        inverts Draw/Erase for the duration of the gesture.
        """
        if self._mask_mode_provider is None or not self._mask_mode_provider():
            return False
        if source_point is None:
            return False
        if self._active_face_provider is None:
            return False
        face_index = self._active_face_provider()
        if face_index is None:
            return False
        bbox = self._active_bbox_provider() if self._active_bbox_provider else None
        if bbox is None or not bbox.contains(source_point):
            return False
        self._mask_drag_active = True
        self._mask_drag_invert = bool(event.modifiers() & Qt.ControlModifier)
        self.mask_paint_requested.emit(
            int(face_index),
            float(source_point.x()),
            float(source_point.y()),
            bool(self._mask_drag_invert),
        )
        return True

    def _continue_mask_drag(self, position: QPointF) -> bool:
        """Forward a move event during an in-flight Mask stroke."""
        if not self._mask_drag_active:
            return False
        source_point = self._widget_to_source(position)
        if source_point is None:
            return True  # swallow moves that exit the source rect
        face_index = self._active_face_provider() if self._active_face_provider else None
        if face_index is None:
            return True
        self.mask_paint_requested.emit(
            int(face_index),
            float(source_point.x()),
            float(source_point.y()),
            bool(self._mask_drag_invert),
        )
        return True

    def _end_mask_drag(self) -> None:
        """Tear down Mask drag state on release."""
        self._mask_drag_active = False
        self._mask_drag_invert = False

    def _begin_extract_drag(self, position: QPointF) -> bool:
        """Try to start an Extract Box translate/scale/rotate drag (#102).

        Hit-test priority (closest action first):

        1. A corner handle → scale uniformly around the bbox centre.
        2. A point inside the bbox body → translate (landmarks + bbox).
        3. A point inside a ``_EXTRACT_ROTATION_BAND_PX`` halo outside the
           bbox → rotate landmarks around the centre.
        4. Otherwise return ``False`` so the view can pan instead.
        """
        import math

        if self._extract_mode_provider is None or not self._extract_mode_provider():
            return False
        if self._active_face_provider is None or self._active_bbox_provider is None:
            return False
        face_index = self._active_face_provider()
        if face_index is None:
            return False
        widget_bbox = self._active_bbox_widget_rect()
        source_bbox = self._active_bbox_source_rect()
        if widget_bbox is None or source_bbox is None:
            return False
        source_point = self._widget_to_source(position)
        if source_point is None:
            return False
        QPointF(
            widget_bbox.x() + widget_bbox.width() / 2.0,
            widget_bbox.y() + widget_bbox.height() / 2.0,
        )
        centre_source = QPointF(
            source_bbox.x() + source_bbox.width() / 2.0,
            source_bbox.y() + source_bbox.height() / 2.0,
        )
        handle = ManualFrameOverlay.handle_at(widget_bbox, position, tolerance=2.0)
        # Scale uses only corner handles (nw/ne/sw/se).  Mid-edge handles
        # also map to scale here (uniform scale by closest corner) so the
        # whole 8-handle ring is responsive.
        if handle is not None:
            radius = math.hypot(
                source_point.x() - centre_source.x(),
                source_point.y() - centre_source.y(),
            )
            if radius <= 0.0:
                return False
            self._edit_drag_mode = "extract_scale"
            self._edit_drag_handle = handle
            self._extract_drag_center = centre_source
            self._extract_drag_start_radius = radius
            self._extract_drag_scale = 1.0
            self._edit_drag_source_anchor = source_point
            self._edit_drag_original_bbox = QRectF(source_bbox)
            self._edit_drag_current_bbox = QRectF(source_bbox)
            return True
        if widget_bbox.contains(position):
            self._edit_drag_mode = "extract_translate"
            self._edit_drag_handle = None
            self._edit_drag_source_anchor = source_point
            self._edit_drag_original_bbox = QRectF(source_bbox)
            self._edit_drag_current_bbox = QRectF(source_bbox)
            return True
        # Outside the bbox but within the rotation halo → rotate.
        halo = QRectF(
            widget_bbox.x() - self._EXTRACT_ROTATION_BAND_PX,
            widget_bbox.y() - self._EXTRACT_ROTATION_BAND_PX,
            widget_bbox.width() + 2 * self._EXTRACT_ROTATION_BAND_PX,
            widget_bbox.height() + 2 * self._EXTRACT_ROTATION_BAND_PX,
        )
        if not halo.contains(position):
            return False
        start_angle = math.atan2(
            source_point.y() - centre_source.y(),
            source_point.x() - centre_source.x(),
        )
        self._edit_drag_mode = "extract_rotate"
        self._edit_drag_handle = None
        self._extract_drag_center = centre_source
        self._extract_drag_start_angle = start_angle
        self._extract_drag_angle = 0.0
        self._edit_drag_source_anchor = source_point
        self._edit_drag_original_bbox = QRectF(source_bbox)
        self._edit_drag_current_bbox = QRectF(source_bbox)
        return True

    def _begin_edit_drag(self, position: QPointF) -> bool:
        """Try to start a move/resize drag on the active face. Returns True on hit."""
        if self._active_face_provider is None:
            return False
        face_index = self._active_face_provider()
        if face_index is None:
            return False
        widget_bbox = self._active_bbox_widget_rect()
        source_bbox = self._active_bbox_source_rect()
        if widget_bbox is None or source_bbox is None:
            return False
        source_point = self._widget_to_source(position)
        if source_point is None:
            return False
        handle = ManualFrameOverlay.handle_at(widget_bbox, position, tolerance=2.0)
        if handle is not None:
            self._edit_drag_mode = "resize"
            self._edit_drag_handle = handle
        elif widget_bbox.contains(position):
            self._edit_drag_mode = "move"
            self._edit_drag_handle = None
        else:
            return False
        self._edit_drag_source_anchor = source_point
        self._edit_drag_original_bbox = QRectF(source_bbox)
        self._edit_drag_current_bbox = QRectF(source_bbox)
        return True

    def _update_edit_drag(self, position: QPointF) -> None:
        """Apply pointer motion to the active drag preview.

        Handles bbox move/resize/add plus the new landmark, landmark-group,
        and marquee modes (#103).  All previews update incrementally and
        request a repaint so the view + overlays stay in sync.
        """
        source_point = self._widget_to_source(position)
        if source_point is None or self._edit_drag_source_anchor is None:
            return
        anchor = self._edit_drag_source_anchor
        dx = source_point.x() - anchor.x()
        dy = source_point.y() - anchor.y()
        mode = self._edit_drag_mode
        if mode == "landmark":
            # Single-point drag: the overlay reads the live coord from
            # ``landmark_drag_preview`` so we just store the new position.
            self._edit_drag_current_bbox = QRectF(source_point.x(), source_point.y(), 0.0, 0.0)
            self.update()
            return
        if mode == "landmark_group":
            # Group move: overlay reads ``landmark_drag_preview`` for each
            # selected index — we just store the delta in the current bbox.
            self._edit_drag_current_bbox = QRectF(dx, dy, 0.0, 0.0)
            self.update()
            return
        if mode == "landmark_marquee":
            rect = QRectF(anchor, source_point).normalized()
            self._edit_drag_current_bbox = rect
            self.update()
            return
        if mode == "extract_translate":
            origin = self._edit_drag_original_bbox
            if origin is not None:
                self._edit_drag_current_bbox = QRectF(
                    origin.x() + dx,
                    origin.y() + dy,
                    origin.width(),
                    origin.height(),
                )
            self.update()
            return
        if mode == "extract_scale":
            self._update_extract_scale(source_point)
            self.update()
            return
        if mode == "extract_rotate":
            self._update_extract_rotate(source_point)
            self.update()
            return
        if self._edit_drag_original_bbox is None:
            return
        if mode == "move":
            origin = self._edit_drag_original_bbox
            self._edit_drag_current_bbox = QRectF(
                origin.x() + dx,
                origin.y() + dy,
                origin.width(),
                origin.height(),
            )
        elif mode == "add":
            self._edit_drag_current_bbox = QRectF(anchor, source_point).normalized()
        else:
            self._edit_drag_current_bbox = self._resize_bbox(
                self._edit_drag_original_bbox, self._edit_drag_handle or "", dx, dy
            )
        self.update()

    def _update_extract_scale(self, source_point: QPointF) -> None:
        """Track the live scale factor + preview bbox for an Extract Box scale drag."""
        import math

        if (
            self._extract_drag_center is None
            or self._extract_drag_start_radius <= 0.0
            or self._edit_drag_original_bbox is None
        ):
            return
        radius = math.hypot(
            source_point.x() - self._extract_drag_center.x(),
            source_point.y() - self._extract_drag_center.y(),
        )
        scale = max(0.05, radius / self._extract_drag_start_radius)
        self._extract_drag_scale = scale
        original = self._edit_drag_original_bbox
        cx = self._extract_drag_center.x()
        cy = self._extract_drag_center.y()
        new_w = original.width() * scale
        new_h = original.height() * scale
        self._edit_drag_current_bbox = QRectF(cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)

    def _update_extract_rotate(self, source_point: QPointF) -> None:
        """Track the live rotation delta for an Extract Box rotation drag.

        The preview bbox isn't rotated visually (the displayed rect stays
        axis-aligned) — the rotation only takes effect on commit because
        ``rotate_face`` is the source of truth.
        """
        import math

        if self._extract_drag_center is None:
            return
        angle = math.atan2(
            source_point.y() - self._extract_drag_center.y(),
            source_point.x() - self._extract_drag_center.x(),
        )
        self._extract_drag_angle = angle - self._extract_drag_start_angle

    @staticmethod
    def _resize_bbox(original: QRectF, handle: str, dx: float, dy: float) -> QRectF:
        """Compute the new bbox for a resize drag in source coordinates.

        Enforces a minimum 1px dimension so the bbox can't collapse, and lets
        opposing handles cross (the user expects the box to flip rather than
        get stuck) by normalising via :meth:`QRectF.normalized`.
        """
        x = original.x()
        y = original.y()
        w = original.width()
        h = original.height()
        # West edges move x; east edges move width; analogous for vertical.
        if "w" in handle:
            x += dx
            w -= dx
        if "e" in handle:
            w += dx
        if "n" in handle:
            y += dy
            h -= dy
        if "s" in handle:
            h += dy
        rect = QRectF(x, y, w, h).normalized()
        return QRectF(rect.x(), rect.y(), max(1.0, rect.width()), max(1.0, rect.height()))

    def _commit_edit_drag(self) -> None:
        """Emit the appropriate signal then clear the in-progress drag state."""
        original = self._edit_drag_original_bbox
        current = self._edit_drag_current_bbox
        mode = self._edit_drag_mode
        anchor = self._edit_drag_source_anchor
        landmark_index = self._landmark_drag_index
        landmark_indices = self._landmark_drag_indices
        extract_scale = self._extract_drag_scale
        extract_angle = self._extract_drag_angle
        face_index = self._active_face_provider() if self._active_face_provider else None
        self._reset_edit_drag()
        if mode == "add":
            self._emit_add_request(anchor, current)
            self.update()
            return
        if mode == "landmark":
            self._emit_landmark_move(face_index, landmark_index, anchor, current)
            self.update()
            return
        if mode == "landmark_group":
            self._emit_landmark_group_move(face_index, landmark_indices, current)
            self.update()
            return
        if mode == "landmark_marquee":
            self._emit_landmark_marquee(face_index, current)
            self.update()
            return
        if mode == "extract_translate":
            if face_index is not None and original is not None and current is not None:
                dx = current.x() - original.x()
                dy = current.y() - original.y()
                if dx != 0.0 or dy != 0.0:
                    self.face_move_requested.emit(int(face_index), float(dx), float(dy))
            self.update()
            return
        if mode == "extract_scale":
            if face_index is not None and extract_scale != 1.0:
                self.face_scale_requested.emit(int(face_index), float(extract_scale))
            self.update()
            return
        if mode == "extract_rotate":
            if face_index is not None and extract_angle != 0.0:
                self.face_rotate_requested.emit(int(face_index), float(extract_angle))
            self.update()
            return
        if face_index is None or original is None or current is None:
            self.update()
            return
        if mode == "move":
            dx = current.x() - original.x()
            dy = current.y() - original.y()
            if dx == 0.0 and dy == 0.0:
                self.update()
                return
            self.face_move_requested.emit(face_index, float(dx), float(dy))
        elif mode == "resize":
            if (
                current.x() == original.x()
                and current.y() == original.y()
                and current.width() == original.width()
                and current.height() == original.height()
            ):
                self.update()
                return
            self.face_resize_requested.emit(face_index, QRectF(current))
        self.update()

    def _emit_landmark_move(
        self,
        face_index: int | None,
        landmark_index: int | None,
        anchor: QPointF | None,
        current: QRectF | None,
    ) -> None:
        """Emit ``landmark_move_requested`` for a single-point drag commit."""
        if face_index is None or landmark_index is None or current is None or anchor is None:
            return
        if current.x() == anchor.x() and current.y() == anchor.y():
            return
        self.landmark_move_requested.emit(
            int(face_index), int(landmark_index), float(current.x()), float(current.y())
        )

    def _emit_landmark_group_move(
        self,
        face_index: int | None,
        indices: tuple[int, ...],
        current: QRectF | None,
    ) -> None:
        """Emit ``landmarks_move_requested`` for a group-drag commit."""
        if face_index is None or not indices or current is None:
            return
        dx = current.x()
        dy = current.y()
        if dx == 0.0 and dy == 0.0:
            return
        self.landmarks_move_requested.emit(int(face_index), tuple(indices), float(dx), float(dy))

    def _emit_landmark_marquee(
        self,
        face_index: int | None,
        current: QRectF | None,
    ) -> None:
        """Emit ``landmarks_select_requested`` with indices inside the marquee.

        A zero-size marquee (click without drag) clears the selection — that
        mirrors the Tk Manual Tool's behaviour and gives the user a way to
        deselect everything by clicking on empty space inside the bbox.
        """
        if face_index is None:
            return
        if (
            current is None
            or current.width() <= 0.0
            or current.height() <= 0.0
            or self._landmark_provider is None
        ):
            self.landmarks_select_requested.emit(int(face_index), ())
            return
        landmarks = self._landmark_provider() or ()
        indices = ManualFrameOverlay.landmarks_in_rect(landmarks, current)
        self.landmarks_select_requested.emit(int(face_index), indices)

    def _emit_add_request(self, anchor: QPointF | None, current: QRectF | None) -> None:
        """Emit ``face_add_requested`` for a committed add gesture.

        A click without motion (current is None or rect is below the
        ``_ADD_MIN_DRAG`` threshold) becomes a small default-sized box centred
        at the click anchor. An actual drag emits the normalized rectangle.
        """
        if anchor is None:
            return
        src_w, src_h = self.source_size
        if src_w <= 0 or src_h <= 0:
            return
        if current is None or (
            current.width() < self._ADD_MIN_DRAG and current.height() < self._ADD_MIN_DRAG
        ):
            default_size = max(8.0, min(64.0, min(src_w, src_h) / 6.0))
            half = default_size / 2.0
            rect = QRectF(
                max(0.0, min(src_w - default_size, anchor.x() - half)),
                max(0.0, min(src_h - default_size, anchor.y() - half)),
                default_size,
                default_size,
            )
        else:
            rect = QRectF(current).normalized()
            rect = QRectF(
                max(0.0, min(src_w - 1.0, rect.x())),
                max(0.0, min(src_h - 1.0, rect.y())),
                max(1.0, min(src_w - rect.x(), rect.width())),
                max(1.0, min(src_h - rect.y(), rect.height())),
            )
        self.face_add_requested.emit(rect)

    _ADD_MIN_DRAG = 4.0
    """Source-pixel threshold below which an add-drag is treated as a click."""

    def _begin_add_drag(self, position: QPointF, source_point: QPointF | None) -> bool:
        """Try to start an add drag if the add-mode provider returns True."""
        if self._add_mode_provider is None or not self._add_mode_provider():
            return False
        if source_point is None or not self._target_rect().contains(position):
            return False
        self._edit_drag_mode = "add"
        self._edit_drag_handle = None
        self._edit_drag_source_anchor = QPointF(source_point)
        # ``original_bbox`` is unused for add but the helpers guard on it; use
        # a zero-sized placeholder anchored at the click point.
        self._edit_drag_original_bbox = QRectF(source_point.x(), source_point.y(), 0.0, 0.0)
        self._edit_drag_current_bbox = QRectF(source_point.x(), source_point.y(), 0.0, 0.0)
        self.setCursor(Qt.CrossCursor)
        return True

    _LANDMARK_HIT_TOLERANCE = 4.0
    """Extra source pixels around each landmark for forgiving point hit-test."""

    def _begin_landmark_drag(
        self,
        position: QPointF,
        source_point: QPointF | None,
        event: QMouseEvent,
    ) -> bool:
        """Try to start a single-point, group, or marquee landmark drag (#103).

        Pre-conditions:

        * Landmark editor mode is active (``_landmark_mode_provider`` returns
          True) — otherwise the bbox edit/add path runs as before.
        * The active face has landmarks — clicking on a face without landmarks
          falls back to bbox move/resize.
        """
        if (
            self._landmark_mode_provider is None
            or self._landmark_provider is None
            or self._landmark_selection_provider is None
            or not self._landmark_mode_provider()
        ):
            return False
        if source_point is None:
            return False
        if self._active_face_provider is None:
            return False
        face_index = self._active_face_provider()
        if face_index is None:
            return False
        landmarks = self._landmark_provider()
        if not landmarks:
            return False
        # 1) Point hit-test first — drag a single landmark, or a group when
        #    that landmark is part of the current selection.
        hit_index = ManualFrameOverlay.landmark_at(
            landmarks,
            (source_point.x(), source_point.y()),
            tolerance=self._LANDMARK_HIT_TOLERANCE,
        )
        if hit_index is not None:
            selection = self._landmark_selection_provider()
            self._edit_drag_source_anchor = QPointF(source_point)
            self._landmark_drag_index = hit_index
            if hit_index in selection and len(selection) > 1:
                self._edit_drag_mode = "landmark_group"
                self._landmark_drag_indices = tuple(sorted(selection))
                self._landmark_drag_origins = tuple(
                    (landmarks[idx][0], landmarks[idx][1]) for idx in self._landmark_drag_indices
                )
            else:
                self._edit_drag_mode = "landmark"
                self._landmark_drag_indices = (hit_index,)
                self._landmark_drag_origins = ((landmarks[hit_index][0], landmarks[hit_index][1]),)
            return True
        # 2) Empty-space drag inside the active face's bbox starts a marquee.
        #    Outside the bbox we fall through so the user can still pan.
        bbox = self._active_bbox_provider() if self._active_bbox_provider else None
        if bbox is None or not bbox.contains(source_point):
            return False
        self._edit_drag_mode = "landmark_marquee"
        self._edit_drag_source_anchor = QPointF(source_point)
        self._edit_drag_original_bbox = QRectF(source_point.x(), source_point.y(), 0.0, 0.0)
        self._edit_drag_current_bbox = QRectF(source_point.x(), source_point.y(), 0.0, 0.0)
        self.setCursor(Qt.CrossCursor)
        return True

    def _emit_context_menu(self, source_point: QPointF | None, global_position: QPointF) -> bool:
        """Hit-test the right-click and emit ``face_context_menu_requested``.

        Returns ``True`` when a face was hit (and the host is expected to open
        a menu); ``False`` lets the caller fall through to the base handler.
        """
        if self._face_hit_provider is None or source_point is None:
            return False
        face_index = self._face_hit_provider(source_point.x(), source_point.y())
        if face_index is None:
            return False
        self.face_context_menu_requested.emit(face_index, QPointF(global_position))
        return True

    def _reset_edit_drag(self) -> None:
        """Clear all in-progress edit-drag state without emitting signals."""
        self._edit_drag_mode = None
        self._edit_drag_handle = None
        self._edit_drag_source_anchor = None
        self._edit_drag_original_bbox = None
        self._edit_drag_current_bbox = None
        self._landmark_drag_index = None
        self._landmark_drag_indices = ()
        self._landmark_drag_origins = ()
        self._extract_drag_center = None
        self._extract_drag_start_radius = 0.0
        self._extract_drag_start_angle = 0.0
        self._extract_drag_scale = 1.0
        self._extract_drag_angle = 0.0

    def _paint_brush_preview(self, painter: QPainter, viewport: FrameViewport) -> None:
        """Draw a circular brush preview at the pointer in Mask mode (#101).

        Only paints when ``_brush_preview_source_point`` is set — the hover
        path keeps that state in sync with the pointer position, the
        active face's bbox, the editor mode and the brush size.  The
        circle's stroke colour distinguishes Draw (cyan) from Erase
        (magenta) at a glance.
        """
        point = self._brush_preview_source_point
        if point is None:
            return
        widget_point = viewport.source_to_widget(point.x(), point.y())
        radius_widget = max(1.0, float(self._brush_preview_radius) * float(viewport.zoom))
        painter.save()
        colour = QColor("#88d4ff") if self._brush_preview_mode == "draw" else QColor("#ff66cc")
        pen = QPen(colour)
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(widget_point, radius_widget, radius_widget)
        painter.restore()

    def _paint_add_preview(self, painter: QPainter, viewport: FrameViewport) -> None:
        """Draw a dashed preview rect for in-progress add or marquee gestures."""
        mode = self._edit_drag_mode
        if mode not in ("add", "landmark_marquee"):
            return
        source = self._edit_drag_current_bbox
        if source is None:
            return
        if source.width() <= 0.0 and source.height() <= 0.0:
            return
        widget_rect = viewport.source_rect_to_widget(
            source.x(), source.y(), source.width(), source.height()
        )
        painter.save()
        pen = QPen(QColor("#88d4ff"))
        pen.setStyle(Qt.DashLine)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(widget_rect)
        painter.restore()


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
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setIconSize(QSize(self._ICON_SIZE, self._ICON_SIZE))
        self.setSpacing(6)
        self.setMinimumHeight(self._ICON_SIZE + 36)
        self.setUniformItemSizes(True)
        self._faces: tuple[FaceThumbnail, ...] = ()
        self._placeholder = self._build_placeholder_icon()
        self.currentRowChanged.connect(self._on_row_changed)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu_requested)

    def _on_context_menu_requested(self, position: QPoint) -> None:
        """Emit ``face_context_menu_requested`` for the right-clicked item."""
        item = self.itemAt(position)
        if item is None:
            return
        face_index = item.data(Qt.UserRole)
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


class FaceGridThumbnailRenderer:
    """Compose cross-frame grid thumbnail icons with optional mask/mesh overlays."""

    _MASK_TINT = QColor(255, 80, 80)
    _MESH_PEN = QColor("#ffb000")
    _LANDMARK_FILL = QColor("#ffffff")

    def __init__(self, editable: ManualEditableAlignments) -> None:
        self._editable = editable

    def render(
        self,
        entry: FaceGridEntry,
        *,
        icon_size: int,
        show_mesh: bool,
        show_mask: bool,
        mask_type: str,
        mask_opacity: int,
    ) -> QIcon:
        """Return an icon for ``entry`` with the requested annotation layers."""
        base = self._base_pixmap(entry, icon_size)
        painter = QPainter(base)
        if show_mask:
            self._paint_mask(painter, entry, icon_size, mask_type, mask_opacity)
        if show_mesh:
            self._paint_mesh(painter, entry, icon_size)
        painter.end()
        return QIcon(base)

    def _base_pixmap(self, entry: FaceGridEntry, icon_size: int) -> QPixmap:
        image = (
            _decode_jpeg_to_qimage(entry.thumbnail.thumbnail_jpeg)
            if entry.thumbnail is not None
            else QImage()
        )
        if not image.isNull():
            return QPixmap.fromImage(
                image.scaled(
                    icon_size,
                    icon_size,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
            ).copy(0, 0, icon_size, icon_size)
        pixmap = QPixmap(icon_size, icon_size)
        pixmap.fill(QColor("#222"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#888"))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "?")
        painter.end()
        return pixmap

    def _paint_mask(
        self,
        painter: QPainter,
        entry: FaceGridEntry,
        icon_size: int,
        mask_type: str,
        mask_opacity: int,
    ) -> None:
        if not mask_type:
            return
        opacity_pct = max(0, min(100, int(mask_opacity)))
        if opacity_pct == 0:
            return
        mask = self._editable.get_mask(entry.frame_index, entry.face_index, mask_type)
        if mask is None:
            return
        try:
            if mask.size == 0:
                return
            height, width = mask.shape[:2]
            argb = bytearray(width * height * 4)
            tint = self._MASK_TINT
            alpha_factor = opacity_pct / 100.0
            data = mask.tobytes()
            mv = memoryview(argb)
            for idx, value in enumerate(data):
                base = idx * 4
                if value == 0:
                    mv[base : base + 4] = b"\x00\x00\x00\x00"
                    continue
                mv[base] = tint.blue()
                mv[base + 1] = tint.green()
                mv[base + 2] = tint.red()
                mv[base + 3] = int(min(255, value * alpha_factor))
            image = QImage(bytes(argb), width, height, width * 4, QImage.Format_ARGB32)
        except Exception:  # pragma: no cover - defensive thumbnail path
            return
        painter.drawImage(QRectF(0.0, 0.0, float(icon_size), float(icon_size)), image)

    def _paint_mesh(self, painter: QPainter, entry: FaceGridEntry, icon_size: int) -> None:
        if not entry.landmarks:
            return
        x, y, width, height = entry.bbox
        if width <= 0.0 or height <= 0.0:
            return
        points = [
            QPointF(
                max(0.0, min(float(icon_size), ((lx - x) / width) * icon_size)),
                max(0.0, min(float(icon_size), ((ly - y) / height) * icon_size)),
            )
            for lx, ly in entry.landmarks
        ]
        if not points:
            return
        painter.save()
        pen = QPen(self._MESH_PEN)
        pen.setWidthF(max(1.0, icon_size / 80.0))
        painter.setPen(pen)
        for first, second in zip(points, points[1:], strict=False):
            painter.drawLine(first, second)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._LANDMARK_FILL))
        radius = max(1.0, icon_size / 42.0)
        for point in points:
            painter.drawEllipse(point, radius, radius)
        painter.restore()


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
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setSpacing(6)
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
        self.setContextMenuPolicy(Qt.CustomContextMenu)
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
        self.setGridSize(QSize(size + 34, size + 42))
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
                item.setFlags(Qt.NoItemFlags)
                item.setTextAlignment(Qt.AlignCenter)
                self.addItem(item)
                self._render_requests = ()
                return
            icon_size = self.iconSize().width()
            for entry in self._entries:
                item = QListWidgetItem(f"{entry.frame_index + 1}:{entry.face_index + 1}")
                item.setData(_FACE_GRID_ENTRY_ROLE, entry)
                item.setTextAlignment(Qt.AlignCenter)
                item.setSizeHint(self.gridSize())
                request = FaceGridRenderRequest(
                    frame_index=entry.frame_index,
                    face_index=entry.face_index,
                    icon_size=icon_size,
                    show_mesh=self._show_mesh,
                    show_mask=self._show_mask,
                    mask_type=self._mask_type,
                    mask_opacity=self._mask_opacity,
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
                return item
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
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            entry = self._current_entry()
            if entry is not None:
                self.face_activated.emit(entry.frame_index, entry.face_index)
                event.accept()
                return
        if event.key() == Qt.Key_Delete:
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
                item.setSelected(False)
                item.setBackground(QBrush(QColor("#34414a")))
                item.setForeground(QBrush(QColor("#ffffff")))
            elif is_active_frame:
                item.setSelected(False)
                item.setBackground(QBrush(QColor("#27323a")))
                item.setForeground(QBrush(QColor("#d7e3ea")))
            else:
                item.setSelected(False)
                item.setBackground(QBrush())
                item.setForeground(QBrush())


class ManualTransportBar(QWidget):
    """Transport controls below the frame view (slider + jump entry + counter).

    The widget is deliberately driven by ``set_total`` / ``set_position`` calls
    and emits ``position_changed`` for any user-driven motion (slider drag,
    jump-entry submit).  The host window owns the actual frame index — this is
    purely UI glue.

    Layout::

        [ slider ─────────────────────────────── ] [Frame: 12 / 250] [Go: 12]

    """

    position_changed = Signal(int)
    """Emitted when the user moves the slider or submits a jump-to value."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-transport-bar")
        self._total = 0
        self._suppress_signal = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setObjectName("qt-manual-transport-slider")
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)
        self._slider.setTracking(True)
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self._slider, 1)

        self._counter = QLabel("Frame: – / 0")
        self._counter.setObjectName("qt-manual-transport-counter")
        layout.addWidget(self._counter)

        self._jump_label = QLabel("Go:")
        layout.addWidget(self._jump_label)
        self._jump_entry = QLineEdit()
        self._jump_entry.setObjectName("qt-manual-transport-jump")
        self._jump_entry.setFixedWidth(60)
        # Jump entry is 1-based to match the visible ``Frame: X / N`` counter.
        # Validator range follows: ``[1, total]`` when total > 0, else ``[0, 0]``
        # so the entry stays uneditable when no frames are loaded.
        self._jump_entry.setValidator(QIntValidator(0, 0, self))
        self._jump_entry.setPlaceholderText("#")
        # ``editingFinished`` already fires on Return *and* on focus-out, so
        # there is no need to also connect ``returnPressed`` — doing both
        # makes a single Return keystroke emit ``position_changed`` twice
        # (issue #119 task 3).  ``editingFinished`` covers both paths.
        self._jump_entry.editingFinished.connect(self._on_jump_submit)
        layout.addWidget(self._jump_entry)

    @property
    def slider(self) -> QSlider:
        """Return the underlying transport slider widget (for tests/styling)."""
        return self._slider

    @property
    def jump_entry(self) -> QLineEdit:
        """Return the jump-to-frame line edit (for tests)."""
        return self._jump_entry

    @property
    def counter_label(self) -> QLabel:
        """Return the ``Frame: X / N`` counter label (for tests/styling)."""
        return self._counter

    def set_total(self, total: int) -> None:
        """Resize the transport range to ``total`` frames (0 disables the bar)."""
        self._total = max(0, int(total))
        last = max(0, self._total - 1)
        self._slider.setEnabled(self._total > 0)
        self._slider.setRange(0, last)
        self._jump_entry.setEnabled(self._total > 0)
        validator = self._jump_entry.validator()
        if isinstance(validator, QIntValidator):
            # 1-based validator: when no frames are loaded keep the entry pinned
            # to (0, 0) so it rejects all input; otherwise accept ``1..total``.
            if self._total > 0:
                validator.setRange(1, self._total)
            else:
                validator.setRange(0, 0)
        self._update_counter(self._slider.value())

    def set_position(self, position: int) -> None:
        """Snap the slider + counter to ``position`` without emitting signals."""
        if self._total <= 0:
            return
        clamped = max(0, min(int(position), self._total - 1))
        self._suppress_signal = True
        try:
            self._slider.setValue(clamped)
        finally:
            self._suppress_signal = False
        self._update_counter(clamped)

    def _on_slider_value_changed(self, value: int) -> None:
        """User dragged the slider — emit unless we're applying a programmatic snap."""
        self._update_counter(value)
        if not self._suppress_signal:
            self.position_changed.emit(int(value))

    def _on_jump_submit(self) -> None:
        """Parse a 1-based jump entry, clamp to ``[1, total]``, emit on change.

        The visible counter is ``Frame: X / N`` (1-based), so the jump entry
        accepts the same numbering: typing ``1`` selects the first frame,
        ``N`` selects the last.  The emitted ``position_changed`` signal
        carries the underlying 0-based transport position.
        """
        text = self._jump_entry.text().strip()
        current_one_based = self._slider.value() + 1 if self._total > 0 else 0
        if not text or self._total <= 0:
            self._jump_entry.setText(str(current_one_based) if self._total > 0 else "")
            return
        try:
            value = int(text)
        except ValueError:
            self._jump_entry.setText(str(current_one_based))
            return
        clamped_one_based = max(1, min(value, self._total))
        position = clamped_one_based - 1
        if position != self._slider.value():
            self.position_changed.emit(position)
        self._jump_entry.setText(str(clamped_one_based))

    def _update_counter(self, position: int) -> None:
        """Refresh the ``Frame: X / N`` label and the 1-based jump-entry text.

        The jump entry only refreshes when it is *not* focused so the user
        can keep typing without their input being clobbered by an in-flight
        slider move triggered by playback.
        """
        if self._total <= 0:
            self._counter.setText("Frame: – / 0")
            if not self._jump_entry.hasFocus():
                self._jump_entry.setText("")
            return
        one_based = position + 1
        self._counter.setText(f"Frame: {one_based} / {self._total}")
        if not self._jump_entry.hasFocus():
            self._jump_entry.setText(str(one_based))


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
    toolbar_visible: bool = True
    """When ``False``, the action is registered for its shortcut but not added
    to the toolbar — used by keyboard-only commands (nudge arrows, etc.)."""
    focus_scope: str = "window"
    """Where the shortcut listens. ``"window"`` (default) fires from anywhere
    in the Manual Tool. ``"frame_view"`` parents the action to the frame view
    with :data:`Qt.WidgetWithChildrenShortcut` so the panel can claim arrow
    keys for its own navigation when focused."""


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
        shortcut=("Z",),
        icon="prev",
        tooltip="Previous frame (Z)",
    ),
    ManualAction(
        key="next_frame",
        label="Next",
        handler="_next_frame",
        shortcut=("X",),
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
        key="magnify_active_face",
        label="Magnify Active Face",
        handler="magnify_active_face",
        shortcut=("M",),
        icon=None,
        tooltip="Fit the active face's bbox to the frame view (M)",
    ),
    # Mask editor (F5, #101) — Draw / Erase actions + brush size shortcuts.
    ManualAction(
        key="mask_draw",
        label="Mask: Draw",
        handler="set_mask_draw_mode",
        shortcut=("B",),
        icon=None,
        tooltip="Switch the Mask editor to Draw mode (B)",
    ),
    ManualAction(
        key="mask_erase",
        label="Mask: Erase",
        handler="set_mask_erase_mode",
        shortcut=("D",),
        icon=None,
        tooltip="Switch the Mask editor to Erase mode (D)",
    ),
    ManualAction(
        key="brush_size_increase",
        label="Brush Size +",
        handler="increase_brush_size",
        shortcut=("]",),
        icon=None,
        tooltip="Increase Mask brush size (])",
    ),
    ManualAction(
        key="brush_size_decrease",
        label="Brush Size -",
        handler="decrease_brush_size",
        shortcut=("[",),
        icon=None,
        tooltip="Decrease Mask brush size ([)",
    ),
    ManualAction(
        key="nudge_up",
        label="Nudge Up",
        handler="nudge_up_one",
        shortcut=("Up",),
        icon=None,
        tooltip="Nudge the active face up one source pixel (Up)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_down",
        label="Nudge Down",
        handler="nudge_down_one",
        shortcut=("Down",),
        icon=None,
        tooltip="Nudge the active face down one source pixel (Down)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_left",
        label="Nudge Left",
        handler="nudge_left_one",
        shortcut=("Left",),
        icon=None,
        tooltip="Nudge the active face left one source pixel (Left)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_right",
        label="Nudge Right",
        handler="nudge_right_one",
        shortcut=("Right",),
        icon=None,
        tooltip="Nudge the active face right one source pixel (Right)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_up_fast",
        label="Nudge Up (10px)",
        handler="nudge_up_fast",
        shortcut=("Shift+Up",),
        icon=None,
        tooltip="Nudge the active face up ten source pixels (Shift+Up)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_down_fast",
        label="Nudge Down (10px)",
        handler="nudge_down_fast",
        shortcut=("Shift+Down",),
        icon=None,
        tooltip="Nudge the active face down ten source pixels (Shift+Down)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_left_fast",
        label="Nudge Left (10px)",
        handler="nudge_left_fast",
        shortcut=("Shift+Left",),
        icon=None,
        tooltip="Nudge the active face left ten source pixels (Shift+Left)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_right_fast",
        label="Nudge Right (10px)",
        handler="nudge_right_fast",
        shortcut=("Shift+Right",),
        icon=None,
        tooltip="Nudge the active face right ten source pixels (Shift+Right)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="extract_faces",
        label="Extract Faces",
        handler="extract_faces",
        shortcut=("Ctrl+E",),
        icon="task_save_as",
        tooltip="Extract aligned faces to a folder (Ctrl+E)",
        separator_before=True,
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
    """Emitted with ``(stage, message)`` as each named stage starts."""
    progress_percent = Signal(int, str)
    """Emitted with ``(percent, message)`` for live sub-stage progress (eg.
    per-frame thumbnail regeneration).  Each event carries its own immutable
    percent value so queued signals always paint the correct bar position."""
    completed = Signal(bool, str)
    """Emitted with ``(has_thumbnails, summary)`` on success."""
    failed = Signal(str)
    """Emitted with a user-facing error message on failure."""
    _start = Signal()

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        session: ManualSession,
    ) -> None:
        super().__init__()
        self._handle = handle
        self._editable = editable
        self._session = session
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule the worker on its owning thread."""
        self._start.emit()

    STAGE_PERCENT: T.ClassVar[dict[str, int]] = {
        "open": 33,
        "thumbs": 66,
        "complete": 100,
    }
    """Static stage anchors used by :meth:`progress` events.

    Live thumbnail-regeneration percentages are *not* stored here — they're
    delivered through the dedicated :attr:`progress_percent` signal so each
    event carries its own immutable percent and cannot bleed between
    Manual Tool windows (see issue #114).
    """

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
            needs_regen = self._session.thumb_regenerate or not has_thumbnails
            if needs_regen and face_count:
                regenerated = self._handle.regenerate_thumbnails(
                    self._session, progress=self._emit_thumb_progress
                )
                if regenerated:
                    has_thumbnails = True
                    logger.info("Manual Tool generated thumbnails for %d frame(s)", regenerated)

            summary = f"Loaded {face_count} face(s) across {len(alignments.data)} frame(s)"
            self.progress.emit("complete", summary)
            self.completed.emit(has_thumbnails, summary)
        except Exception as err:  # noqa: BLE001 - surface any startup failure
            logger.exception("Manual Tool startup failed")
            self.failed.emit(str(err))

    def _emit_thumb_progress(self, done: int, total: int, message: str) -> None:
        """Bridge :mod:`thumbnail_generation` progress to immutable per-event percent.

        Maps the (done, total) pair into the 66 → 99 % band of the startup bar
        and emits it on the dedicated :attr:`progress_percent` signal.  Also
        fires the named ``progress`` signal so console fanout and status-bar
        message paths continue to receive each step.
        """
        denominator = max(1, total)
        percent = min(99, 66 + round((done / denominator) * 33))
        self.progress_percent.emit(int(percent), message)
        self.progress.emit("thumbs", message)


class ManualStartupWorker(QObject):
    """Owns the QThread that runs :class:`_ManualStartupTask`.

    The window subscribes to :attr:`progress`, :attr:`completed` and
    :attr:`failed` for user-facing feedback.  Calling :meth:`stop` is safe at
    any time — it gracefully terminates the thread even if the user closes
    the window mid-load.
    """

    progress = Signal(str, str)
    progress_percent = Signal(int, str)
    completed = Signal(bool, str)
    failed = Signal(str)

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        session: ManualSession,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualStartupTask(handle, editable, session)
        self._task.moveToThread(self._thread)
        self._task.progress.connect(self.progress)
        self._task.progress_percent.connect(self.progress_percent)
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


class _ManualExtractFacesTask(QObject):
    """Worker that runs Tk-parity Extract Faces off the UI thread.

    Construct with the alignments handle, the live session, and the user-
    picked output folder; connect to :attr:`progress` / :attr:`completed` /
    :attr:`failed`; then call :meth:`kick_off`.  ``cancel`` is thread-safe
    and surfaces to the next ``is_cancelled()`` poll between frames.
    """

    progress = Signal(int, int, str)
    """``(done, total, message)`` — total is the number of source frames."""
    completed = Signal(object)
    """Emits the :class:`ExtractFacesResult` from a finished run."""
    failed = Signal(str)
    """Emits a user-facing error string when extraction raises."""
    _start = Signal()

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        session: ManualSession,
        output_folder: str,
        editable_targets: T.Any = None,
    ) -> None:
        super().__init__()
        self._handle = handle
        self._session = session
        self._output_folder = output_folder
        # Editable snapshot — when supplied, ``extract_faces`` uses it instead
        # of the persisted alignments file so unsaved Manual Tool edits are
        # included in the extracted PNG output (parity with Tk Manual which
        # extracts from its live in-memory face list).
        self._editable_targets = editable_targets
        self._cancelled = False
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule :meth:`run` on this object's owning thread."""
        self._start.emit()

    def cancel(self) -> None:
        """Request cancellation between frames.  Safe to call from any thread."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Return whether :meth:`cancel` has been called."""
        return self._cancelled

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Execute the extraction and emit terminal signals."""
        try:
            from tools.manual.face_extraction import (
                FaceExtractionRequest,
                extract_faces,
            )

            request = FaceExtractionRequest(
                handle=self._handle,
                session=self._session,
                output_folder=self._output_folder,
                editable_targets=self._editable_targets,
            )
            result = extract_faces(
                request,
                progress=lambda done, total, message: self.progress.emit(done, total, message),
                is_cancelled=self.is_cancelled,
            )
        except Exception as err:  # noqa: BLE001 - surface any failure
            logger.exception("Manual Tool extract faces failed")
            self.failed.emit(str(err))
            return
        self.completed.emit(result)


class ManualExtractFacesWorker(QObject):
    """Owns the QThread that drives :class:`_ManualExtractFacesTask`."""

    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        session: ManualSession,
        output_folder: str,
        editable_targets: T.Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualExtractFacesTask(
            handle, session, output_folder, editable_targets=editable_targets
        )
        self._task.moveToThread(self._thread)
        self._task.progress.connect(self.progress)
        self._task.completed.connect(self.completed)
        self._task.failed.connect(self.failed)
        self._task.completed.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background extraction on the worker thread."""
        self._task.kick_off()

    def cancel(self) -> None:
        """Forward a cancel request to the underlying task."""
        self._task.cancel()

    _STOP_WAIT_MS = 3000
    """How long ``stop()`` waits for the QThread to exit before giving up."""

    def stop(self, *, wait_ms: int | None = None) -> bool:
        """Try to shut down the worker thread; report whether it exited.

        Cancellation is polled by ``_ManualExtractFacesTask`` between
        frames, so a long-running per-frame operation (image read, mask
        encode, alignment) may not finish within the wait window.  Returns
        ``True`` iff the QThread has actually exited by the deadline; the
        caller can then drop its reference safely.  ``False`` means the
        worker is still in-flight and must NOT have its reference cleared
        — see :meth:`ManualToolWindow.closeEvent` for the reentrant-safe
        cleanup path (issue #119 task 2).
        """
        if not self._thread.isRunning():
            return True
        self.cancel()
        self._thread.quit()
        timeout = self._STOP_WAIT_MS if wait_ms is None else int(wait_ms)
        # ``QThread.wait`` returns True iff the thread terminated within
        # the timeout — that's our return contract.
        return bool(self._thread.wait(timeout))


class _ManualAlignerLoadTask(QObject):
    """Worker task that preloads one aligner backend off the UI thread."""

    status = Signal(str, str, str)
    """``(kind, aligner, message)`` for visible load-progress updates."""
    completed = Signal(str, str, bool, str)
    """``(aligner, normalization, ok, message)`` once preload finishes."""
    _start = Signal()

    def __init__(self, service: T.Any, aligner: str, normalization: str) -> None:
        super().__init__()
        self._service = service
        self._aligner = str(aligner)
        self._normalization = str(normalization)
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule :meth:`run` on this object's owning thread."""
        self._start.emit()

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Preload the requested aligner backend and emit terminal status."""
        loading = f"Loading aligner '{self._aligner}'…"
        self.status.emit("loading", self._aligner, loading)
        try:
            # Keep GUI updates on Qt signals by avoiding ManualAlignerService.preload(),
            # whose status callback may call directly from this worker thread.
            # The private method is part of the same service boundary this Qt
            # host already depends on for cached backend lifetime.
            get_backend = getattr(self._service, "_get_backend", None)
            if callable(get_backend):
                get_backend(self._aligner, self._normalization)
            else:  # pragma: no cover - compatibility fallback for non-standard stubs
                ok = bool(self._service.preload(self._aligner, self._normalization))
                if not ok:
                    raise RuntimeError(f"Aligner '{self._aligner}' failed to load")
        except Exception as err:  # noqa: BLE001 - surface to the host
            logger.exception("Manual Tool: aligner preload failed (%s)", self._aligner)
            message = f"Aligner '{self._aligner}' failed to load: {err}"
            self.status.emit("failed", self._aligner, message)
            self.completed.emit(self._aligner, self._normalization, False, message)
            return
        message = f"Aligner '{self._aligner}' ready"
        self.status.emit("ready", self._aligner, message)
        self.completed.emit(self._aligner, self._normalization, True, message)


class ManualAlignerLoadWorker(QObject):
    """Owns the QThread that preloads an aligner backend."""

    status = Signal(str, str, str)
    completed = Signal(str, str, bool, str)

    def __init__(
        self,
        service: T.Any,
        aligner: str,
        normalization: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualAlignerLoadTask(service, aligner, normalization)
        self._task.moveToThread(self._thread)
        self._task.status.connect(self.status)
        self._task.completed.connect(self.completed)
        self._task.completed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background aligner preload."""
        self._task.kick_off()

    def stop(self, *, wait_ms: int = 1000) -> bool:
        """Stop the loader thread and report whether it exited."""
        if not self._thread.isRunning():
            return True
        self._thread.quit()
        return bool(self._thread.wait(int(wait_ms)))


class _ManualSaveTask(QObject):
    """Worker that runs alignment persistence off the UI thread.

    Mirrors :class:`_ManualExtractFacesTask`: the host installs progress /
    completion / failure signal handlers and calls :meth:`kick_off`.  The
    task is single-shot — re-entry is prevented by ``_save_in_flight`` on
    the host before this is ever constructed.

    The editable model and frame-name resolver are captured at schedule time
    by the host; mutating actions are gated through ``_busy_operation`` so
    no caller thread can race the persist with concurrent edits.
    """

    completed = Signal(int)
    """Emits the number of frames modified on a successful save."""
    failed = Signal(str)
    """Emits a user-facing error string when persistence raises."""
    _start = Signal()

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        frame_names: T.Any,
    ) -> None:
        super().__init__()
        self._handle = handle
        self._editable = editable
        self._frame_names = frame_names
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule :meth:`run` on this object's owning thread."""
        self._start.emit()

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Execute the persist call and emit the terminal signal."""
        try:
            modified = self._handle.persist(self._editable, frame_names=self._frame_names)
        except Exception as err:  # noqa: BLE001 - surface any persist failure
            logger.exception("Manual Tool save worker failed")
            self.failed.emit(str(err))
            return
        self.completed.emit(int(modified))


class ManualSaveWorker(QObject):
    """Owns the QThread that drives :class:`_ManualSaveTask`.

    Save is short-lived and uncancellable — the worker exists purely to keep
    persistence off the UI thread so the busy progress bar and disabled
    mutating-action state can repaint before ``Alignments.save`` blocks.
    """

    completed = Signal(int)
    failed = Signal(str)

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        frame_names: T.Any,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualSaveTask(handle, editable, frame_names)
        self._task.moveToThread(self._thread)
        self._task.completed.connect(self.completed)
        self._task.failed.connect(self.failed)
        self._task.completed.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background persistence on the worker thread."""
        self._task.kick_off()

    def stop(self) -> None:
        """Stop the worker thread (after any in-flight persist completes)."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)


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
        aligner_service: T.Any = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-tool-window")
        self.setWindowTitle("Faceswap Manual Tool")
        self._session = session
        self._editor_state = session.create_editor_state()
        # Aligner integration (#104) — accept an optional injected service so
        # tests don't need to construct the production plugin pipeline.  The
        # real path lazily imports the default service so importing this
        # module stays cheap.
        if aligner_service is None:
            from tools.manual.aligner_service import ManualAlignerService

            aligner_service = ManualAlignerService(status_callback=self._on_aligner_status)
        self._aligner_service = aligner_service
        self._aligner_load_worker: ManualAlignerLoadWorker | None = None
        self._aligner_load_target: tuple[str, str] | None = None
        self._aligner_loaded_targets: set[tuple[str, str]] = set()
        self._video_metadata: ManualVideoMetadata | None = None
        self._video_provider: VideoFrameProvider | None = None
        self._video_frames: list[ManualFrame] = []
        self._legacy_args = legacy_args or []
        # One cached alignments handle for the lifetime of this window so we
        # are not reopening the alignments file on every face refresh.
        self._alignments_handle = session.alignments_handle()
        self._editable = ManualEditableAlignments()
        self._editable.subscribe(self._on_editable_changed)
        self._current_frame: ManualFrame | None = None
        self._frame_view = ManualFrameView()
        self._thumbnail_panel = ManualThumbnailPanel()
        self._face_panel = FaceThumbnailPanel()
        self._face_grid_renderer = FaceGridThumbnailRenderer(self._editable)
        self._face_grid_panel = CrossFrameFaceGridPanel(self._face_grid_renderer)
        self._face_grid_size_combo = QComboBox()
        self._status_label = QLabel()
        self._metadata_label = QLabel()
        # Frame-navigation widgets — driven from the *filtered* frame list
        # (#107).  The transport position is an index into
        # ``_filtered_frame_indices``; the thumbnail panel still shows every
        # frame but navigation skips frames the active filter rejects.
        self._transport_bar = ManualTransportBar()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000 / 24))
        self._play_timer.setTimerType(Qt.PreciseTimer)
        self._play_timer.timeout.connect(self._advance_during_playback)
        # Filtered model (#107).  Empty until the thumbnail panel is
        # populated; ``_refresh_filter_results`` keeps it in lock-step with
        # ``editor_state.filter_mode`` / ``filter_distance`` and any
        # editable-model change that affects face count or alignment.
        self._filtered_frame_indices: tuple[int, ...] = ()
        self._progress_bar: QProgressBar | None = None
        self._console_logger = console_logger
        self._startup_worker: ManualStartupWorker | None = None
        self._startup_complete = False
        # #121: once a per-event thumb progress signal lands the named
        # ``"thumbs"`` stage must NOT reset the bar back to 66% — the
        # named-stage emit still fires alongside ``progress_percent`` so
        # console + status messaging stay per-frame, but the progress bar
        # only follows the live percent from that point on.
        self._thumb_progress_seen = False
        # Extract Faces worker — held so cancel/close can stop it cleanly.
        self._extract_worker: ManualExtractFacesWorker | None = None
        self._extract_total: int = 0
        # Visible Cancel control next to the extract progress bar (issue #118).
        self._extract_cancel_button: QToolButton | None = None
        self._actions: dict[str, QAction] = {}
        # Save/bulk-operation lock state.  ``_busy_operation`` carries a short
        # user-facing label for the in-flight job ("Saving alignments…", etc.)
        # and is set/unset by :meth:`_with_busy_lock`.  Mutating actions are
        # disabled while this is non-empty.
        self._busy_operation: str | None = None
        self._save_in_flight: bool = False
        # Async save state — when set, the ExitStack holds the busy-lock open
        # until the worker's completed/failed signal fires, and ``_save_worker``
        # is the live :class:`ManualSaveWorker` whose completion drains the
        # stack and finalises the save (clears dirty / shows status / dialog).
        self._save_worker: ManualSaveWorker | None = None
        self._save_busy_stack: contextlib.ExitStack | None = None
        # When extract is triggered against a dirty session, persistence is
        # scheduled first; this holds the output folder until save completes.
        self._pending_extract_folder: str | None = None
        self._overlay = ManualFrameOverlay(
            self._editable,
            frame_index_provider=self._current_frame_index,
        )
        self._overlay.install_mask_render_seam(
            mask_type_provider=self.active_mask_type,
            mask_opacity_provider=lambda: self._editor_state.mask_opacity,
            mask_show_provider=self._should_render_mask,
        )
        self._editor_state.subscribe("unsaved", self._on_unsaved_changed)
        self._editor_state.subscribe("face_index", lambda _v: self._sync_actions())
        self._editor_state.subscribe(
            "face_index", lambda value: self._overlay.set_active(int(value))
        )
        self._editor_state.subscribe("face_index", lambda _v: self._frame_view.update())
        self._editor_state.subscribe("face_index", lambda _v: self._refresh_face_grid_active())
        # When the active face changes, refresh the Mask dropdown so the
        # option list reflects whatever the new face has persisted on disk.
        self._editor_state.subscribe(
            "face_index", lambda _v: self._refresh_mask_controls_visibility()
        )
        self._editor_state.subscribe(
            "face_index", lambda _v: self._refresh_aligner_controls_visibility()
        )
        self._editor_state.subscribe("frame_index", lambda _v: self._sync_actions())
        self._editor_state.subscribe(
            "frame_index", lambda _v: self._refresh_mask_controls_visibility()
        )
        self._editor_state.subscribe("frame_index", lambda _v: self._refresh_face_grid_active())
        self._editor_state.subscribe(
            "frame_index", lambda _v: self._refresh_aligner_controls_visibility()
        )
        self._editor_state.subscribe("editor_mode", self._on_editor_mode_changed)
        # Keep the filter-controls label + threshold-slider visibility in
        # lock-step with editor-state changes.  ``filter_mode`` already
        # gets ``_refresh_filter_results`` from the cycle action, but
        # programmatic flips (tests, future menu items) need this too.
        self._editor_state.subscribe("filter_mode", lambda _v: self._refresh_filter_results())
        self._editor_state.subscribe("filter_distance", lambda _v: self._refresh_filter_results())
        self._editor_state.subscribe("faces_size", self._on_face_grid_size_state_changed)
        self._editor_state.subscribe("annotation_mode", lambda _v: self._refresh_face_grid())
        self._editor_state.subscribe("annotation_mode", lambda _v: self._frame_view.update())
        self._editor_state.subscribe("mask_type", lambda _v: self._refresh_face_grid())
        self._editor_state.subscribe("mask_opacity", lambda _v: self._refresh_face_grid())
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
    def face_grid_panel(self) -> CrossFrameFaceGridPanel:
        """Return the filtered-session cross-frame face grid widget."""
        return self._face_grid_panel

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
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._extract_worker is not None:
            # ``stop()`` returns True iff the worker thread actually
            # exited within the wait window.  If extraction is stuck in
            # a long per-frame op the thread keeps running — we must NOT
            # drop the reference (Python would then GC the QThread mid-
            # flight and SIGABRT in Qt) and we must NOT consume the
            # close event (so the user can try again once it does
            # finish).  See #119 task 2.
            stopped = self._extract_worker.stop()
            if not stopped:
                self.statusBar().showMessage(
                    "Extraction still running — please wait for it to finish, then close again.",
                    7000,
                )
                event.ignore()
                return
            self._extract_worker = None
        if self._save_worker is not None:
            # The save worker is uncancellable but short-lived; stop() blocks
            # only until the in-flight persist returns.  Drain the busy-lock
            # so close doesn't leak the disabled-actions / progress-bar state.
            self._save_worker.stop()
            self._save_worker = None
            self._drain_save_busy_stack()
        if self._aligner_load_worker is not None:
            stopped = self._aligner_load_worker.stop()
            if not stopped:
                self.statusBar().showMessage(
                    "Aligner is still loading — please wait, then close again.",
                    7000,
                )
                event.ignore()
                return
            self._aligner_load_worker = None
            self._aligner_load_target = None
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
        """Schedule a non-blocking save of editable edits.

        Persistence runs on :class:`ManualSaveWorker` so the busy-lock
        progress bar and disabled mutating actions can repaint *before*
        ``Alignments.save`` blocks (issue #115). The host event loop is
        flushed once after the busy state is installed so the user sees
        the busy feedback no matter how brief the persist actually is.

        Returns:

        * ``True`` when a save was scheduled (or the no-op fast path was
          taken because there is nothing to persist).
        * ``False`` when a save is already in flight — duplicate ``Ctrl+S``
          presses cannot race the persist call.

        Completion arrives through :meth:`_on_save_completed` or
        :meth:`_on_save_failed`; those handlers drain the busy-lock,
        update dirty state and post the status message / dialog.
        """
        if self._save_in_flight:
            self.statusBar().showMessage("Save already in progress…", 3000)
            return False
        if self._current_frame_index() < 0 and not any(
            self._editable.face_count(i) for i in range(self._session.frame_count)
        ):
            self.statusBar().showMessage("Nothing to save", 3000)
            return True

        # Install the busy-lock through an ExitStack so the completion handler
        # can drain it asynchronously.  This keeps the visible busy state,
        # disabled-mutating-actions, and progress-bar lifecycle identical to
        # the old synchronous path while moving persistence onto a thread.
        #
        # Importantly we do NOT call ``QCoreApplication.processEvents`` here:
        # the busy lock has already installed the progress bar + disabled
        # mutating actions by the time we return to the caller, and the
        # worker's persist call runs on its own QThread — so the OS event
        # loop naturally paints the busy state on its next tick.  Calling
        # ``processEvents`` mid-schedule would let unrelated workers (e.g.
        # the startup thumbnail-cache scan) fire their completion and tear
        # the progress bar back down from underneath us.
        self._save_busy_stack = contextlib.ExitStack()
        self._save_busy_stack.enter_context(self._with_busy_lock("Saving alignments…", save=True))

        try:
            frame_names = self._frame_names_for_persist()
            worker = ManualSaveWorker(
                self._alignments_handle, self._editable, frame_names, parent=self
            )
        except Exception as err:  # noqa: BLE001 - schedule-time failure
            logger.exception("Manual Tool save: failed to schedule worker")
            self._drain_save_busy_stack()
            self.statusBar().showMessage(f"Manual Tool save failed: {err}", 7000)
            QMessageBox.critical(self, "Manual Tool Save", f"Manual Tool save failed: {err}")
            return False
        worker.completed.connect(self._on_save_completed)
        worker.failed.connect(self._on_save_failed)
        self._save_worker = worker
        worker.start()
        return True

    def _on_save_completed(self, modified: int) -> None:
        """Worker reported success — finalise dirty state and refresh views."""
        self._drain_save_busy_stack()
        self._teardown_save_worker()
        self._editable.clear_history()
        self.mark_dirty(False)
        self._editor_state.set("edited", False)
        self._editor_state.set("face_count_changed", False)
        self.refresh_faces()
        self._frame_view.update()
        self.statusBar().showMessage(f"Saved {modified} frame(s) to alignments file", 5000)
        self._emit_console(f"Manual Tool: saved {modified} frame(s)")
        # A pending extract was waiting for this save — resume now.
        self._resume_extract_after_save()

    def _on_save_failed(self, message: str) -> None:
        """Worker reported failure — leave dirty state intact, surface error."""
        self._drain_save_busy_stack()
        self._teardown_save_worker()
        full = f"Manual Tool save failed: {message}"
        logger.error(full)
        self.statusBar().showMessage(full, 7000)
        self._emit_console(full)
        QMessageBox.critical(self, "Manual Tool Save", full)
        # Abort any pending extract — extracting against a broken save would
        # write stale or partial output.
        self._pending_extract_folder = None

    def _drain_save_busy_stack(self) -> None:
        """Release the busy-lock context held for the in-flight save."""
        stack = self._save_busy_stack
        self._save_busy_stack = None
        if stack is not None:
            stack.close()

    def _teardown_save_worker(self) -> None:
        """Stop + drop the save worker, joining its QThread."""
        if self._save_worker is not None:
            self._save_worker.stop()
            self._save_worker.deleteLater()
            self._save_worker = None

    @contextlib.contextmanager
    def _with_busy_lock(self, label: str, *, save: bool = False) -> T.Iterator[None]:
        """Run a blocking operation with progress + action gating.

        While the block is active:

        * ``_busy_operation`` carries ``label`` so :meth:`_sync_actions` knows
          to disable mutating actions and prevent conflicting edits.
        * The status-bar progress bar is shown in indeterminate mode with
          ``label`` as the format string.
        * ``processEvents`` runs once so the busy state paints before the
          (potentially blocking) work begins.

        Always restores prior state in a ``finally``, including when an
        exception escapes — the caller is free to ``return``/``raise`` inside
        the ``with`` block.
        """
        prior_progress_format = None
        prior_progress_range: tuple[int, int] | None = None
        prior_progress_visible = False
        owns_progress_bar = False
        self._busy_operation = label
        if save:
            self._save_in_flight = True
        # The startup worker tears down ``_progress_bar`` once startup is
        # finished; for save/bulk ops we re-materialize it for the duration
        # of the operation so the user always gets visible busy feedback.
        if self._progress_bar is None:
            self._progress_bar = self._build_progress_bar()
            self.statusBar().addPermanentWidget(self._progress_bar)
            owns_progress_bar = True
        if self._progress_bar is not None:
            prior_progress_format = self._progress_bar.format()
            prior_progress_range = (
                self._progress_bar.minimum(),
                self._progress_bar.maximum(),
            )
            prior_progress_visible = self._progress_bar.isVisible()
            self._progress_bar.setRange(0, 0)  # indeterminate
            self._progress_bar.setFormat(label)
            self._progress_bar.show()
        self._sync_actions()
        self.statusBar().showMessage(label, 3000)
        # Intentionally do NOT call ``processEvents`` here — flushing the event
        # loop inside the lock can let unrelated workers (eg. the startup
        # thumbnail-cache scan) complete and tear down the progress bar we
        # just materialized, which then disappears from underneath the persist
        # callback.  The lock is already entered before any blocking work
        # begins, so the paint will happen on the next natural event-loop tick.
        try:
            yield
        finally:
            self._busy_operation = None
            if save:
                self._save_in_flight = False
            if self._progress_bar is not None:
                if owns_progress_bar:
                    self._hide_progress_bar()
                else:
                    if prior_progress_range is not None:
                        self._progress_bar.setRange(*prior_progress_range)
                    if prior_progress_format is not None:
                        self._progress_bar.setFormat(prior_progress_format)
                    if not prior_progress_visible:
                        self._progress_bar.hide()
            self._sync_actions()

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

    def extract_faces(self) -> bool:
        """Prompt for an output folder and run Tk-parity Extract Faces.

        Returns ``True`` when the extraction was scheduled, ``False`` when
        the user cancelled the folder picker or there is nothing to extract.
        Mirrors the Tk Extract button: aligned 512-px PNGs per face, named
        ``{source_stem}_{face_index}.png``, with alignments + source
        metadata embedded in the PNG header.
        """
        if self._extract_worker is not None:
            self.statusBar().showMessage("Extraction already in progress…", 4000)
            return False
        # Extract the live editable state — unsaved adds/deletes/moves count.
        # Persisting bootstraps the alignments file on demand, so we
        # also persist any in-flight edits first to keep alignments.version
        # and source metadata accurate inside the PNG headers.
        if not self._editable_has_any_face():
            self.statusBar().showMessage(
                "Nothing to extract — no faces in the editable model", 4000
            )
            return False
        initial_dir = self._initial_extract_dir()
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select output folder for extracted faces",
            initial_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not chosen:
            return False
        self._start_extract_worker(chosen)
        return True

    def _editable_has_any_face(self) -> bool:
        """Return whether the live editable model has any face anywhere."""
        return any(
            self._editable.face_count(index) > 0 for index in self._editable_frame_indices()
        )

    def _editable_frame_indices(self) -> tuple[int, ...]:
        """Return every frame index that currently has editable state.

        Covers both ``has_images`` (where indices match the discovered frame
        list) and the video case (where frame_index is sparse) by combining
        the discovered range with whatever indices the editable model knows
        about.  The set is union-deduped and returned sorted.
        """
        known = set(self._editable.known_frame_indices())
        if self._session.has_images:
            known.update(range(len(self._session.frame_list)))
        return tuple(sorted(known))

    def _build_editable_extract_targets(self) -> tuple[tuple[str, tuple[T.Any, ...]], ...]:
        """Materialize the live editable model as ``EditableFaceSpec`` targets.

        Iterates every frame the editable model tracks, resolves each to a
        frame name through the same mapping ``save`` uses, and copies
        bbox + landmarks from the editable face.  ``mask`` / ``identity`` /
        ``metadata`` are sourced from the persisted alignments entry at the
        same ``face_index`` when available (so identity vectors and mask
        payloads survive an extract on a face the user has merely moved);
        if no persisted entry exists, they're empty — matching how Tk
        Manual treats a brand-new face.
        """
        import numpy as np

        from tools.manual.face_extraction import EditableFaceSpec

        resolver = self._frame_names_for_persist()
        try:
            alignments = self._alignments_handle.open()
            persisted_by_name = {
                name: tuple(entry.faces) for name, entry in alignments.data.items()
            }
        except Exception:  # noqa: BLE001 - extract still works against unsaved-only sessions
            persisted_by_name = {}

        targets: list[tuple[str, tuple[EditableFaceSpec, ...]]] = []
        for frame_index in self._editable_frame_indices():
            faces = self._editable.faces(frame_index)
            if not faces:
                continue
            if callable(resolver):
                frame_name = resolver(frame_index)
            elif 0 <= frame_index < len(resolver):
                frame_name = resolver[frame_index]
            else:
                frame_name = None
            if not frame_name:
                continue
            persisted = persisted_by_name.get(frame_name, ())
            specs: list[EditableFaceSpec] = []
            for face in faces:
                landmarks = (
                    np.asarray(face.landmarks, dtype=np.float32)
                    if face.landmarks
                    else np.zeros((0, 2), dtype=np.float32)
                )
                prev = persisted[face.face_index] if face.face_index < len(persisted) else None
                x, y, w, h = (int(round(value)) for value in face.bbox)
                specs.append(
                    EditableFaceSpec(
                        face_index=face.face_index,
                        bbox=(x, y, w, h),
                        landmarks_xy=landmarks,
                        mask=dict(getattr(prev, "mask", {}) or {}),
                        identity=dict(getattr(prev, "identity", {}) or {}),
                        metadata=dict(getattr(prev, "metadata", {}) or {}),
                    )
                )
            targets.append((frame_name, tuple(specs)))
        return tuple(targets)

    def _initial_extract_dir(self) -> str:
        """Default the folder picker to a sibling of the input source."""
        if self._session.is_video_input:
            return os.path.dirname(self._session.frames)
        return self._session.frames

    def _start_extract_worker(self, output_folder: str) -> None:
        """Spin up :class:`ManualExtractFacesWorker` and wire its signals.

        Persists in-flight edits first (which bootstraps the alignments file
        if it doesn't exist yet so ``alignments.version`` is well-defined
        inside the PNG header), then builds an editable snapshot to drive
        the extraction.  ``extract_faces`` iterates the snapshot directly
        rather than re-reading ``alignments.data``, so unsaved adds,
        deletes, moves, and resizes all appear in the extracted output —
        matching Tk Manual which extracts its live in-memory face list.

        Save is now asynchronous (#115), so when the session is dirty we
        chain the actual extract launch onto the save worker's
        ``completed`` signal — the user sees the "Saving alignments…" busy
        state, then "Extracting faces…" without either being clobbered.
        """
        if self._editor_state.unsaved:
            # Persist on the worker thread, then chain the extract launch off
            # the save completion.  ``_pending_extract_folder`` is consumed by
            # :meth:`_resume_extract_after_save` so the same code path runs
            # whether save was sync (legacy) or async.
            self._pending_extract_folder = output_folder
            self.save()
            return
        self._launch_extract_worker(output_folder)

    def _resume_extract_after_save(self) -> None:
        """Continue a pending extract once a chained save completes."""
        folder = self._pending_extract_folder
        self._pending_extract_folder = None
        if folder is None:
            return
        self._launch_extract_worker(folder)

    def _launch_extract_worker(self, output_folder: str) -> None:
        """Materialize the editable snapshot and start the extract worker."""
        editable_targets = self._build_editable_extract_targets()
        worker = ManualExtractFacesWorker(
            self._alignments_handle,
            self._session,
            output_folder,
            editable_targets=editable_targets,
            parent=self,
        )
        worker.progress.connect(self._on_extract_progress)
        worker.completed.connect(self._on_extract_completed)
        worker.failed.connect(self._on_extract_failed)
        self._extract_worker = worker
        self._extract_total = 0
        if self._progress_bar is None:
            self._progress_bar = self._build_progress_bar()
            self.statusBar().addPermanentWidget(self._progress_bar)
        if self._extract_cancel_button is None:
            self._extract_cancel_button = self._build_extract_cancel_button()
            self.statusBar().addPermanentWidget(self._extract_cancel_button)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat("Extracting faces…")
        self._progress_bar.show()
        self._extract_cancel_button.setEnabled(True)
        self._extract_cancel_button.show()
        self.statusBar().showMessage(f"Extracting faces to {output_folder}", 5000)
        self._busy_operation = "Extracting faces…"
        self._sync_actions()
        logger.info("Manual Tool extract started → %s", output_folder)
        self._emit_console(f"Manual Tool: extracting faces to {output_folder}")
        worker.start()

    def _on_extract_progress(self, done: int, total: int, message: str) -> None:
        """Update the determinate progress bar from the extract worker."""
        if total > 0 and self._extract_total != total:
            self._extract_total = total
            if self._progress_bar is not None:
                self._progress_bar.setRange(0, total)
                self._progress_bar.setFormat("Extracting %v / %m frame(s)")
        if self._progress_bar is not None and total > 0:
            self._progress_bar.setValue(min(done, total))
        self.statusBar().showMessage(message, 2000)

    def _on_extract_completed(self, result: object) -> None:
        """Surface a summary message and tear down the extract worker."""
        from tools.manual.face_extraction import ExtractFacesResult

        self._teardown_extract_worker()
        if not isinstance(result, ExtractFacesResult):  # pragma: no cover - defensive
            return
        if result.cancelled:
            self.statusBar().showMessage("Extract cancelled", 5000)
            return
        summary = (
            f"Extracted {result.faces_written} face(s) from {result.frames_processed} frame(s)"
        )
        skipped_bits: list[str] = []
        if result.skipped_frames:
            skipped_bits.append(f"{result.skipped_frames} frame(s)")
        if result.skipped_faces:
            skipped_bits.append(f"{result.skipped_faces} face(s)")
        if skipped_bits:
            summary += f" — skipped {', '.join(skipped_bits)}"
        self.statusBar().showMessage(summary, 7000)
        self._emit_console(f"Manual Tool: {summary}")
        logger.info("Manual Tool extract completed: %s", summary)
        if result.errors:
            joined_errors = chr(10).join(result.errors[:10])
            QMessageBox.warning(
                self,
                "Extract Faces",
                f"{summary}{chr(10)}{chr(10)}{joined_errors}",
            )

    def _on_extract_failed(self, message: str) -> None:
        """Report worker-level failure and tear down the extract worker."""
        self._teardown_extract_worker()
        full = f"Extract faces failed: {message}"
        logger.error(full)
        self._emit_console(full)
        self.statusBar().showMessage(full, 7000)
        QMessageBox.critical(self, "Extract Faces", full)

    def _teardown_extract_worker(self) -> None:
        """Hide progress + Cancel button, clear busy state, join the worker thread."""
        if self._progress_bar is not None:
            self._progress_bar.hide()
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setFormat("Loading %p%")
        if self._extract_cancel_button is not None:
            self._extract_cancel_button.hide()
            self._extract_cancel_button.setEnabled(False)
        self._busy_operation = None
        # ``_teardown_extract_worker`` fires from completed/failed handlers;
        # the underlying QThread has already received ``quit()`` via signal
        # connections, so ``stop()`` is just the join.  Still respect the
        # bool contract (#119 task 2): if the thread refuses to exit, keep
        # the reference alive so the QThread destructor doesn't fire on a
        # live thread.
        if self._extract_worker is not None and self._extract_worker.stop():
            self._extract_worker.deleteLater()
            self._extract_worker = None
        self._extract_total = 0
        self._sync_actions()

    def _build_extract_cancel_button(self) -> QToolButton:
        """Return a status-bar Cancel button bound to :meth:`cancel_extract`."""
        button = QToolButton(self)
        button.setObjectName("qt-manual-extract-cancel")
        button.setText("Cancel")
        button.setToolTip("Cancel Extract Faces (stops at the next frame boundary)")
        button.setAutoRaise(False)
        button.setEnabled(False)
        button.hide()
        button.clicked.connect(self.cancel_extract)
        return button

    def cancel_extract(self) -> bool:
        """Request an in-flight Extract Faces job to stop at the next frame.

        Returns ``True`` when a worker was actively cancelled, ``False`` when
        no extraction was running.  The Cancel button is disabled immediately
        so a double-click cannot fire the worker cancel path twice; the
        teardown later hides + re-disables it.
        """
        worker = self._extract_worker
        if worker is None:
            return False
        worker.cancel()
        if self._extract_cancel_button is not None:
            self._extract_cancel_button.setEnabled(False)
        self.statusBar().showMessage("Cancelling extract — stopping at next frame…", 4000)
        self._emit_console("Manual Tool: extract cancellation requested")
        return True

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

    # ---- #107 filtered navigation primitives ----

    def _all_frame_indices(self) -> tuple[int, ...]:
        """Return every known source-frame index (image folder or video).

        The thumbnail panel's row count is the authoritative count once a
        session is loaded — image-folder rows are ``frame.index`` (0..N-1)
        and video rows are likewise consecutive indices.  Used as the
        universe the active filter draws from.
        """
        return tuple(range(self._thumbnail_panel.count()))

    def _refresh_filter_results(self, *, preserve_current: bool = True) -> None:
        """Recompute ``_filtered_frame_indices`` from the active filter.

        Also keeps the transport bar's total in sync and clamps the current
        thumbnail row to a matching frame when ``preserve_current`` is
        ``True`` (the default).  When the current frame still matches the
        filter, it stays put — otherwise the first matching frame is
        selected.  An empty filter leaves the panel untouched and reports
        an empty filtered list so navigation actions disable cleanly.
        """
        from tools.manual.frame_filter import (
            DEFAULT_FILTER_MODE,
            filtered_frame_indices,
            misaligned_predicate_for_model,
        )

        mode = self._editor_state.filter_mode or DEFAULT_FILTER_MODE
        threshold = int(self._editor_state.filter_distance)
        predicate = misaligned_predicate_for_model(self._editable, threshold)
        self._filtered_frame_indices = filtered_frame_indices(
            self._all_frame_indices(),
            self._editable.face_count,
            mode,
            misaligned_predicate=predicate,
        )
        total = len(self._filtered_frame_indices)
        self._transport_bar.set_total(total)
        if total == 0:
            # Empty filter — no transport position to clamp.  Leave the
            # thumbnail panel as-is so the user still sees the source
            # frames; navigation actions disable via ``_sync_actions``.
            self._sync_actions()
            self._refresh_filter_controls()
            self._refresh_face_grid()
            return
        current_row = self._thumbnail_panel.currentRow()
        if preserve_current and current_row in self._filtered_frame_indices:
            position = self._filtered_frame_indices.index(current_row)
            self._transport_bar.set_position(position)
        else:
            new_row = self._filtered_frame_indices[0]
            self._thumbnail_panel.setCurrentRow(new_row)
        self._sync_actions()
        self._refresh_filter_controls()
        self._refresh_face_grid()

    def filtered_frame_indices(self) -> tuple[int, ...]:
        """Return the current filtered frame index list.

        Public read-only accessor — :class:`ManualFaceGrid` (the future
        cross-frame viewer landing in #108) consumes this so its visible
        set matches the active filter.
        """
        return self._filtered_frame_indices

    def _filtered_position(self) -> int:
        """Return the current frame's index in ``_filtered_frame_indices``.

        Returns ``-1`` when the current frame isn't in the filtered list
        (filter rejected it) or no frame is loaded.
        """
        row = self._thumbnail_panel.currentRow()
        if row < 0:
            return -1
        try:
            return self._filtered_frame_indices.index(row)
        except ValueError:
            return -1

    def goto_first_frame(self) -> None:
        """Select the first frame in the active filter (#107)."""
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[0])

    def goto_last_frame(self) -> None:
        """Select the last frame in the active filter (#107)."""
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[-1])

    def toggle_play(self) -> None:
        """Toggle playback through the *filtered* frames (#107)."""
        if self._editor_state.is_playing:
            self._stop_playback()
            return
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            # Rewind to the first filtered frame so Play does something
            # visible instead of being an immediate no-op.
            self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[0])
        self._editor_state.set("is_playing", True)
        self._play_timer.start()
        self._sync_play_action_icon()

    def _stop_playback(self) -> None:
        """Halt the auto-advance timer and reset the play-action icon."""
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._editor_state.is_playing:
            self._editor_state.set("is_playing", False)
        self._sync_play_action_icon()

    def _advance_during_playback(self) -> None:
        """QTimer slot: step forward through the filtered list (#107)."""
        if not self._filtered_frame_indices:
            self._stop_playback()
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            self._stop_playback()
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position + 1])

    def _no_filter_match_message(self) -> str:
        """Compose the empty-filter user message."""
        mode = self._editor_state.filter_mode or "All Frames"
        return f"No frames match filter: {mode}"

    def _sync_play_action_icon(self) -> None:
        """Switch the Play/Pause toolbar icon to match playback state."""
        action = self._actions.get("play_pause")
        if action is None:
            return
        theme = QtTheme.default()
        icon_key = "stop" if self._editor_state.is_playing else "run"
        icon = icon_for_action(theme, icon_key)
        if not icon.isNull():
            action.setIcon(icon)
        action.setText("Pause" if self._editor_state.is_playing else "Play")

    def _on_thumbnail_row_changed(self, _row: int) -> None:
        """Refresh action availability + keep the transport bar in sync.

        The transport bar tracks the *filtered* position so its counter
        and slider reflect the active filter.  When the current frame
        isn't in the filtered list (e.g. the user is paused on a frame
        the filter would normally skip), the slider stays at the closest
        previously-painted position so it doesn't jitter.
        """
        self._sync_actions()
        if not self._filtered_frame_indices:
            return
        row = self._thumbnail_panel.currentRow()
        try:
            position = self._filtered_frame_indices.index(row)
        except ValueError:
            return
        self._transport_bar.set_position(position)

    def _on_transport_position_changed(self, position: int) -> None:
        """Apply a user-driven slider / jump-entry change (filtered position).

        ``position`` is an index into ``_filtered_frame_indices``; the
        actual thumbnail row is the frame index stored at that filtered
        position.  Out-of-range positions are clamped silently.
        """
        if not self._filtered_frame_indices:
            return
        if not 0 <= position < len(self._filtered_frame_indices):
            return
        target_row = self._filtered_frame_indices[position]
        if target_row == self._thumbnail_panel.currentRow():
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(target_row)

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
        # Aligner integration (#104): initialise landmarks for the new face
        # through the configured aligner when auto-run is on.  Failures are
        # surfaced through ``rerun_aligner_for_face`` and leave the new face
        # in place with empty landmarks — Tk's behaviour when the aligner
        # refuses to produce points.
        self._maybe_run_aligner(int(new_index))
        return new_index

    def nudge_active_face(self, dx: float, dy: float) -> bool:
        """Translate the active face's bbox + landmarks by ``(dx, dy)`` pixels."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return False
        face_index = self._editor_state.face_index
        if face_index < 0:
            self.statusBar().showMessage("No active face to nudge", 3000)
            return False
        if not self._editable.move_face(frame_index, face_index, dx, dy):
            self.statusBar().showMessage("Nudge failed (no active face)", 3000)
            return False
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        self.refresh_faces()
        self._frame_view.update()
        return True

    def nudge_up_one(self) -> bool:
        """Nudge active face up by 1 source pixel."""
        return self.nudge_active_face(0.0, -1.0)

    def nudge_down_one(self) -> bool:
        """Nudge active face down by 1 source pixel."""
        return self.nudge_active_face(0.0, 1.0)

    def nudge_left_one(self) -> bool:
        """Nudge active face left by 1 source pixel."""
        return self.nudge_active_face(-1.0, 0.0)

    def nudge_right_one(self) -> bool:
        """Nudge active face right by 1 source pixel."""
        return self.nudge_active_face(1.0, 0.0)

    def nudge_up_fast(self) -> bool:
        """Nudge active face up by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(0.0, -10.0)

    def nudge_down_fast(self) -> bool:
        """Nudge active face down by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(0.0, 10.0)

    def nudge_left_fast(self) -> bool:
        """Nudge active face left by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(-10.0, 0.0)

    def nudge_right_fast(self) -> bool:
        """Nudge active face right by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(10.0, 0.0)

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
        """Advance to the next filter mode in the legacy rotation order.

        Recomputes the filtered frame list immediately so navigation +
        transport bar update in lock-step (#107).
        """
        from tools.manual.frame_filter import FILTER_MODES

        current = self._editor_state.filter_mode or FILTER_MODES[0]
        try:
            index = FILTER_MODES.index(current)
        except ValueError:
            index = -1
        new_mode = FILTER_MODES[(index + 1) % len(FILTER_MODES)]
        self._editor_state.set("filter_mode", new_mode)
        self._refresh_filter_results()
        filtered_count = len(self._filtered_frame_indices)
        suffix = f" ({filtered_count} match)" if filtered_count else " (no matches)"
        self.statusBar().showMessage(f"Filter: {new_mode}{suffix}", 5000)

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

    def magnify_active_face(self) -> bool:
        """Fit the active face's bbox to the viewport (Landmark editor M).

        Returns ``False`` and surfaces a status message when no face is
        active — Tk's Magnify is a no-op in that case too, so parity holds.
        """
        bbox = self._active_face_bbox()
        if bbox is None:
            self.statusBar().showMessage("No active face to magnify", 3000)
            return False
        if not self._frame_view.magnify_to_source_rect(bbox):
            self.statusBar().showMessage("Could not magnify active face", 3000)
            return False
        return True

    def _build_face_grid_panel(self) -> QWidget:
        """Return the face-strip + filtered-session face grid container."""
        container = QWidget()
        container.setObjectName("qt-manual-face-browser-panel")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        current_box = QWidget()
        current_layout = QVBoxLayout(current_box)
        current_layout.setContentsMargins(0, 0, 0, 0)
        current_layout.setSpacing(4)
        current_label = QLabel("Current frame faces")
        current_label.setObjectName("qt-manual-current-frame-faces-label")
        current_layout.addWidget(current_label)
        current_layout.addWidget(self._face_panel, 1)
        layout.addWidget(current_box, 1)

        grid_box = QWidget()
        grid_layout = QVBoxLayout(grid_box)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        session_label = QLabel("Filtered session faces")
        session_label.setObjectName("qt-manual-filtered-session-faces-label")
        controls.addWidget(session_label)
        controls.addStretch(1)
        controls.addWidget(QLabel("Size:"))
        self._face_grid_size_combo.setObjectName("qt-manual-face-grid-size")
        self._face_grid_size_combo.addItems(tuple(_FACE_GRID_SIZES))
        size_name = self._editor_state.faces_size or "Medium"
        if size_name not in _FACE_GRID_SIZES:
            size_name = "Medium"
        self._face_grid_size_combo.setCurrentText(size_name)
        self._face_grid_size_combo.currentTextChanged.connect(self._on_face_grid_size_changed)
        self._face_grid_panel.set_face_size(size_name)
        controls.addWidget(self._face_grid_size_combo)
        grid_layout.addLayout(controls)
        grid_layout.addWidget(self._face_grid_panel, 1)
        layout.addWidget(grid_box, 2)
        return container

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
        # Mask editor (#101): inline dropdown that lets the user pick which
        # mask type to edit.  Hidden by default; surfaced only when the Mask
        # editor (F5) is active so it doesn't clutter the View mode.
        self._mask_controls = self._build_mask_controls()
        left_layout.addWidget(self._mask_controls)
        # Bounding Box editor (#104): aligner plugin selection, normalization,
        # auto-run, explicit rerun, and visible load progress.  Hidden except
        # while F2 / BoundingBox mode is active.
        self._aligner_controls = self._build_aligner_controls()
        left_layout.addWidget(self._aligner_controls)
        left_layout.addWidget(self._frame_view, 1)
        # Filter awareness (#107): a slim row that surfaces filter state and
        # the Misaligned threshold control.  Always visible (so the user can
        # see the active filter + match count); the threshold slider inside
        # only appears for the Misaligned Faces filter.
        self._filter_controls = self._build_filter_controls()
        left_layout.addWidget(self._filter_controls)
        left_layout.addWidget(self._transport_bar)
        left_layout.addWidget(self._status_label)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("qt-manual-main-splitter")
        splitter.addWidget(left)
        splitter.addWidget(self._build_face_grid_panel())
        splitter.addWidget(self._thumbnail_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([500, 140, 80])
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        """Build the Manual Tool action toolbar from :data:`MANUAL_ACTIONS`."""
        toolbar = QToolBar("Manual Tool")
        toolbar.setObjectName("qt-manual-toolbar")
        self.addToolBar(toolbar)
        theme = QtTheme.default()
        for spec in MANUAL_ACTIONS:
            if spec.separator_before and spec.toolbar_visible:
                toolbar.addSeparator()
            owner: QWidget = self
            shortcut_context = Qt.WindowShortcut
            if spec.focus_scope == "frame_view":
                owner = self._frame_view
                shortcut_context = Qt.WidgetWithChildrenShortcut
                self._frame_view.setFocusPolicy(Qt.StrongFocus)
            action = QAction(spec.label, owner)
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
                action.setShortcutContext(shortcut_context)
            handler = getattr(self, spec.handler)
            action.triggered.connect(self._make_action_dispatch(spec.key, handler))
            # Owner registration wires the shortcut: ``self`` for window-scope,
            # ``self._frame_view`` for frame-view scope so the face panel can
            # claim arrow keys for its own navigation when it has focus.
            owner.addAction(action)
            if spec.toolbar_visible:
                toolbar.addAction(action)
            self._actions[spec.key] = action
        # Initial play/pause icon + text mirrors the editor state's not-playing
        # state — actions are built fresh so the default label/icon land here.
        self._sync_play_action_icon()

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
        self._thumbnail_panel.currentRowChanged.connect(self._on_thumbnail_row_changed)
        self._transport_bar.position_changed.connect(self._on_transport_position_changed)
        self._face_panel.face_selected.connect(self._on_face_selected)
        self._face_grid_panel.face_activated.connect(self._on_face_grid_activated)
        self._face_grid_panel.face_hovered.connect(self._on_face_grid_hovered)
        self._face_grid_panel.face_context_menu_requested.connect(
            self._on_face_grid_context_menu_requested
        )
        self._face_grid_panel.face_delete_requested.connect(self._delete_face_at)
        self._frame_view.clicked_at.connect(self._on_frame_clicked)
        self._frame_view.face_move_requested.connect(self._on_face_move_requested)
        self._frame_view.face_resize_requested.connect(self._on_face_resize_requested)
        self._frame_view.face_add_requested.connect(self._on_face_add_requested)
        self._frame_view.face_context_menu_requested.connect(self._on_frame_context_menu_requested)
        self._frame_view.landmark_move_requested.connect(self._on_landmark_move_requested)
        self._frame_view.landmarks_move_requested.connect(self._on_landmarks_move_requested)
        self._frame_view.landmarks_select_requested.connect(self._on_landmarks_select_requested)
        self._face_panel.face_context_menu_requested.connect(
            self._on_face_panel_context_menu_requested
        )
        self._frame_view.install_editor_seams(
            active_face_provider=self._active_face_index,
            active_bbox_provider=self._active_face_bbox,
            add_mode_provider=self._is_add_mode_active,
            face_hit_provider=self._face_at_source_point,
        )
        self._frame_view.install_landmark_seams(
            landmark_mode_provider=self._is_landmark_mode_active,
            landmark_provider=self._active_face_landmarks,
            landmark_selection_provider=lambda: self._overlay.selected_landmarks,
        )
        self._frame_view.install_extract_seams(
            extract_mode_provider=self._is_extract_mode_active,
        )
        self._frame_view.install_mask_seams(
            mask_mode_provider=self._is_mask_mode_active,
            brush_provider=self._current_brush_spec,
        )
        self._frame_view.face_scale_requested.connect(self._on_face_scale_requested)
        self._frame_view.face_rotate_requested.connect(self._on_face_rotate_requested)
        self._frame_view.mask_paint_requested.connect(self._on_mask_paint_requested)
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
            self._refresh_filter_results(preserve_current=False)
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
        # Reset the per-startup #121 flag so a retry can paint the static
        # ``thumbs`` anchor again before any live percent fires.
        self._thumb_progress_seen = False
        self._progress_bar = self._build_progress_bar()
        if self._progress_bar is not None:
            self.statusBar().addPermanentWidget(self._progress_bar)
            self._progress_bar.show()
        self._startup_worker = ManualStartupWorker(
            self._alignments_handle, self._editable, self._session, parent=self
        )
        self._startup_worker.progress.connect(self._on_startup_progress)
        self._startup_worker.progress_percent.connect(self._on_startup_progress_percent)
        self._startup_worker.completed.connect(self._on_startup_completed)
        self._startup_worker.failed.connect(self._on_startup_failed)
        logger.info("Manual Tool startup worker scheduled")
        self._emit_console("Manual Tool: preparing session…")
        self._startup_worker.start()

    def _build_progress_bar(self) -> QProgressBar:
        """Return a Qt-native determinate progress bar for the status bar.

        Stages reported by :class:`_ManualStartupTask` map to discrete
        percentages (``open=33``, ``thumbs=66``, ``complete=100``) so users
        see real forward motion instead of a spinning indeterminate bar.
        """
        bar = QProgressBar()
        bar.setObjectName("qt-manual-startup-progress")
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setMaximumWidth(180)
        bar.setTextVisible(True)
        bar.setFormat("Loading %p%")
        bar.hide()
        return bar

    def _on_startup_progress(self, stage: str, message: str) -> None:
        """Surface intermediate startup messages to the status bar + console.

        Static stage percents (open / thumbs / complete) paint a fresh value
        on the determinate bar.  The named ``"thumbs"`` stage in particular
        fires alongside every per-event ``progress_percent`` emit so console
        + status messaging stay per-frame — but once live thumbnail progress
        has started, painting the static 66% anchor would visibly *reset*
        the bar (see issues #121 / #119 task 1).  Skip that one repaint.
        """
        logger.debug("Manual Tool startup [%s]: %s", stage, message)
        self.statusBar().showMessage(message)
        self._emit_console(f"Manual Tool [{stage}]: {message}")
        percent = _ManualStartupTask.STAGE_PERCENT.get(stage)
        if (
            percent is not None
            and self._progress_bar is not None
            and not (stage == "thumbs" and self._thumb_progress_seen)
        ):
            self._progress_bar.setValue(percent)

    def _on_startup_progress_percent(self, percent: int, _message: str) -> None:
        """Paint the determinate progress bar from a per-event thumb percent.

        The percent is carried by the signal payload itself, so queued
        deliveries cannot disagree with the value computed at emit time —
        unlike the previous ``STAGE_PERCENT["thumbs_progress"]`` trick.

        Also marks ``_thumb_progress_seen`` so the named-stage handler
        stops resetting the bar to the 66% static ``thumbs`` anchor for
        the rest of this startup.
        """
        self._thumb_progress_seen = True
        if self._progress_bar is None:
            return
        clamped = max(0, min(100, int(percent)))
        self._progress_bar.setValue(clamped)

    def _on_startup_completed(self, has_thumbnails: bool, summary: str) -> None:
        """Finalize UI state once the startup worker reports success.

        The worker's ``completed`` signal fires once; after the handler
        runs there's no further use for the worker.  Drain it explicitly
        (mirrors the aligner preload + extract-worker patterns) so the
        QThread doesn't linger past the test's qtbot assertions.
        """
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
        self._refresh_face_grid()
        self._frame_view.update()
        self._sync_actions()
        self._teardown_startup_worker()

    def _on_startup_failed(self, message: str) -> None:
        """Surface startup failures through status, console, log and dialog."""
        self._startup_complete = False
        self._status_label.setText("Manual Tool startup failed")
        self.statusBar().showMessage(message, 7000)
        self._hide_progress_bar()
        logger.error("Manual Tool startup failed: %s", message)
        self._emit_console(f"Manual Tool startup failed: {message}")
        QMessageBox.critical(self, "Manual Tool Startup", message)
        self._teardown_startup_worker()

    def _teardown_startup_worker(self) -> None:
        """Drain + drop the startup worker after its terminal signal fires.

        Mirrors the aligner-preload and extract-worker teardown patterns:
        after the worker's ``completed`` or ``failed`` signal fires there's
        no more work for it to do, so we tell the QThread to quit, wait
        briefly for it to exit, then mark it for Qt deletion and clear
        ``_startup_worker``.  Without this every Manual Tool test left a
        QThread alive until widget teardown, which is the closest remaining
        cross-test signal-leak candidate.
        """
        worker = self._startup_worker
        if worker is None:
            return
        worker.stop()
        worker.deleteLater()
        self._startup_worker = None

    def _hide_progress_bar(self) -> None:
        """Hide and detach the indeterminate startup progress widget.

        Yields the bar back to a busy-lock when one is currently active —
        the lock's ``finally`` restores its prior state, so we mustn't yank
        the widget out from underneath the in-flight operation.  Without
        this guard the startup worker's completion can tear down the bar
        that an async save just materialised.
        """
        if self._busy_operation:
            # Hide is still safe; only avoid the detach/null-out.
            if self._progress_bar is not None:
                self._progress_bar.hide()
            return
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
        self._refresh_filter_results(preserve_current=False)
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
        self._refresh_filter_results()
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

    def _active_face_index(self) -> int | None:
        """Return the currently selected ``face_index`` or ``None``."""
        index = self._editor_state.face_index
        if not isinstance(index, int) or index < 0:
            return None
        return index

    def _active_face_bbox(self) -> QRectF | None:
        """Return the active face's source-coordinate bbox or ``None``."""
        face_index = self._active_face_index()
        if face_index is None:
            return None
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return None
        faces = self._editable.faces(frame_index)
        if face_index >= len(faces):
            return None
        x, y, w, h = faces[face_index].bbox
        return QRectF(x, y, w, h)

    def _on_face_move_requested(self, face_index: int, dx: float, dy: float) -> None:
        """Apply a pointer-driven bbox translation and refresh views."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.move_face(frame_index, face_index, dx, dy):
            self.statusBar().showMessage("Move failed (no active face)", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()
        # Aligner integration (#104): refresh landmarks from the moved bbox
        # when the Bounding Box editor + auto-run is on.
        self._maybe_run_aligner(face_index)

    def _on_face_resize_requested(self, face_index: int, bbox: QRectF) -> None:
        """Apply a pointer-driven bbox resize and refresh views."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        new_bbox = (bbox.x(), bbox.y(), bbox.width(), bbox.height())
        if not self._editable.resize_face(frame_index, face_index, new_bbox):
            self.statusBar().showMessage("Resize failed (no active face)", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()
        self._maybe_run_aligner(face_index)

    def _is_add_mode_active(self) -> bool:
        """Return whether empty-space clicks should create a new face."""
        return self._editor_state.editor_mode == "BoundingBox"

    def _is_landmark_mode_active(self) -> bool:
        """Return whether the Landmark editor (F4) is active."""
        return self._editor_state.editor_mode == "Landmarks"

    def _is_extract_mode_active(self) -> bool:
        """Return whether the Extract Box editor (F3) is active."""
        return self._editor_state.editor_mode == "ExtractBox"

    def _is_mask_mode_active(self) -> bool:
        """Return whether the Mask editor (F5) is active."""
        return self._editor_state.editor_mode == "Mask"

    def _should_render_mask(self) -> bool:
        """Show mask overlay when the Mask editor is active or annotation toggle on.

        Matches the legacy Tk behaviour where ``F10`` toggles the mask
        overlay layer in any editor mode and F5 implicitly shows it.
        """
        return (
            self._editor_state.editor_mode == "Mask"
            or self._editor_state.annotation_mode == "Mask"
        )

    def _on_mask_paint_requested(
        self, face_index: int, source_x: float, source_y: float, invert: bool
    ) -> None:
        """Apply a Mask editor brush stamp at ``(source_x, source_y)`` (#101)."""
        self.paint_mask_at(int(face_index), float(source_x), float(source_y), invert=bool(invert))

    def _on_face_scale_requested(self, face_index: int, scale: float) -> None:
        """Apply an Extract Box corner-drag scale to the active face (#102)."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.scale_face(frame_index, int(face_index), float(scale)):
            self.statusBar().showMessage("Scale failed (degenerate result)", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()

    def _on_face_rotate_requested(self, face_index: int, angle: float) -> None:
        """Apply an Extract Box rotation-zone drag to the active face (#102)."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.rotate_face(frame_index, int(face_index), float(angle)):
            self.statusBar().showMessage("Rotate failed (no active face)", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()

    # ---- Mask editor (#101) public action handlers ----
    _BRUSH_MIN: T.ClassVar[int] = 2
    _BRUSH_MAX: T.ClassVar[int] = 64
    _BRUSH_STEP: T.ClassVar[int] = 2
    """Brush size bounds + step (source pixels)."""

    DEFAULT_MASK_TYPE: T.ClassVar[str] = "components"
    """Mask type to use when the user has not picked one explicitly."""

    def set_mask_draw_mode(self) -> bool:
        """Switch the Mask editor to Draw mode (action handler, B)."""
        self._editor_state.set("brush_mode", "draw")
        self.statusBar().showMessage("Mask: Draw mode", 3000)
        return True

    def set_mask_erase_mode(self) -> bool:
        """Switch the Mask editor to Erase mode (action handler, D)."""
        self._editor_state.set("brush_mode", "erase")
        self.statusBar().showMessage("Mask: Erase mode", 3000)
        return True

    def increase_brush_size(self) -> int:
        """Step brush size up by ``_BRUSH_STEP``, clamped at ``_BRUSH_MAX``."""
        new_size = min(self._BRUSH_MAX, self._editor_state.brush_size + self._BRUSH_STEP)
        self._editor_state.set("brush_size", new_size)
        self.statusBar().showMessage(f"Brush size: {new_size}", 2000)
        return new_size

    def decrease_brush_size(self) -> int:
        """Step brush size down by ``_BRUSH_STEP``, clamped at ``_BRUSH_MIN``."""
        new_size = max(self._BRUSH_MIN, self._editor_state.brush_size - self._BRUSH_STEP)
        self._editor_state.set("brush_size", new_size)
        self.statusBar().showMessage(f"Brush size: {new_size}", 2000)
        return new_size

    def active_mask_type(self) -> str:
        """Return the selected mask type, falling back to the default."""
        return self._editor_state.mask_type or self.DEFAULT_MASK_TYPE

    def _current_brush_spec(self) -> tuple[float, str] | None:
        """Return ``(radius_source_px, mode)`` for the frame view's brush preview.

        Returns ``None`` when the Mask editor isn't active or no face is
        selected — the frame view uses ``None`` to mean "hide the preview".
        The radius mirrors :meth:`paint_mask_at`'s ``brush_size / 2`` math
        so the visible circle matches the actual stamp footprint.
        """
        if self._editor_state.editor_mode != "Mask":
            return None
        if self._active_face_index() is None:
            return None
        radius = max(1.0, float(self._editor_state.brush_size) / 2.0)
        mode = str(self._editor_state.brush_mode or "draw")
        return (radius, mode)

    def paint_mask_at(
        self,
        face_index: int,
        source_x: float,
        source_y: float,
        *,
        invert: bool = False,
    ) -> bool:
        """Stamp the Mask editor brush at the given source coordinate (#101).

        ``invert`` flips the current draw/erase mode for this single
        stamp — used by Ctrl+click in the legacy Tk editor.  Returns the
        result of :meth:`ManualEditableAlignments.paint_mask_stroke`.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return False
        mode = self._editor_state.brush_mode
        if invert:
            mode = "erase" if mode == "draw" else "draw"
        value = 255 if mode == "draw" else 0
        radius = max(1.0, self._editor_state.brush_size / 2.0)
        ok = self._editable.paint_mask_stroke(
            frame_index,
            int(face_index),
            self.active_mask_type(),
            float(source_x),
            float(source_y),
            radius,
            value,
        )
        if ok:
            self._editor_state.set("edited", True)
            self._refresh_face_grid()
            self._frame_view.update()
        return ok

    # ---- Aligner integration (#104) ----

    def _on_aligner_status(self, status: T.Any) -> None:
        """Forward :class:`AlignerStatus` events to status, console and controls."""
        message = getattr(status, "message", "") or ""
        kind = getattr(status, "kind", "") or ""
        aligner = getattr(status, "aligner", "") or self._active_aligner_name()
        if not message:
            return
        self._set_aligner_load_status(kind, aligner, message)
        timeout = 5000 if kind == "failed" else 3000
        self.statusBar().showMessage(message, timeout)
        self._emit_console(f"Manual Tool aligner: {message}")
        if kind == "failed":
            logger.error("Manual Tool aligner: %s", message)

    def available_aligners(self) -> tuple[str, ...]:
        """Return aligner plugin display names from the cached service."""
        return self._aligner_service.available_aligners

    def available_normalizations(self) -> tuple[str, ...]:
        """Return normalization methods supported by the aligner service."""
        return self._aligner_service.available_normalizations()

    def set_aligner_name(self, name: str) -> None:
        """Select the active aligner plugin (Bounding Box dropdown handler)."""
        value = str(name).strip()
        if not value or value.startswith("No aligners"):
            return
        self._editor_state.set("aligner_name", value)
        self._sync_aligner_controls()
        self._schedule_aligner_preload()

    def set_aligner_normalization(self, method: str) -> None:
        """Apply a normalization method to the aligner service (radio handler)."""
        value = str(method)
        self._editor_state.set("aligner_normalization", value)
        self._aligner_service.set_normalization(value)
        self._sync_aligner_controls()
        self._schedule_aligner_preload()

    def _active_aligner_name(self) -> str:
        """Return the editor-state aligner name, falling back to the service default."""
        return self._editor_state.aligner_name or self._aligner_service.default_aligner()

    def _build_filter_controls(self) -> QWidget:
        """Return the filter-awareness control row (#107).

        Always visible: shows the active filter + match count.  The
        Misaligned threshold slider inside is shown only when
        ``filter_mode == "Misaligned Faces"`` (Tk parity) so the row
        stays slim for the other five filter modes.
        """
        from tools.manual.frame_filter import (
            MISALIGNED_THRESHOLD_MAX,
            MISALIGNED_THRESHOLD_MIN,
        )

        container = QWidget()
        container.setObjectName("qt-manual-filter-controls")
        # Pin to its sizeHint so the always-visible filter row doesn't steal
        # vertical space from the frame view when BoundingBox controls and
        # the transport bar are also visible.
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        self._filter_label = QLabel()
        self._filter_label.setObjectName("qt-manual-filter-label")
        layout.addWidget(self._filter_label)
        layout.addStretch(1)

        # Misaligned threshold sub-group — visible only for that filter mode.
        self._filter_threshold_label = QLabel("Threshold:")
        self._filter_threshold_label.setObjectName("qt-manual-filter-threshold-label")
        layout.addWidget(self._filter_threshold_label)
        self._filter_threshold_slider = QSlider(Qt.Horizontal)
        self._filter_threshold_slider.setObjectName("qt-manual-filter-threshold-slider")
        self._filter_threshold_slider.setRange(MISALIGNED_THRESHOLD_MIN, MISALIGNED_THRESHOLD_MAX)
        self._filter_threshold_slider.setValue(int(self._editor_state.filter_distance))
        self._filter_threshold_slider.setFixedWidth(120)
        self._filter_threshold_slider.valueChanged.connect(self._on_filter_threshold_changed)
        layout.addWidget(self._filter_threshold_slider)
        self._filter_threshold_value = QLabel(str(self._editor_state.filter_distance))
        self._filter_threshold_value.setObjectName("qt-manual-filter-threshold-value")
        layout.addWidget(self._filter_threshold_value)

        self._refresh_filter_controls()
        return container

    def _refresh_filter_controls(self) -> None:
        """Mirror the active filter mode + match count into the control row."""
        label = getattr(self, "_filter_label", None)
        if label is None:
            return
        mode = self._editor_state.filter_mode or "All Frames"
        total = len(self._filtered_frame_indices)
        if mode == "All Frames":
            text = f"Filter: All Frames ({total})"
        else:
            text = f"Filter: {mode} ({total} match)"
        label.setText(text)

        misaligned = mode == "Misaligned Faces"
        if hasattr(self, "_filter_threshold_slider"):
            self._filter_threshold_label.setVisible(misaligned)
            self._filter_threshold_slider.setVisible(misaligned)
            self._filter_threshold_value.setVisible(misaligned)
            current = int(self._editor_state.filter_distance)
            self._filter_threshold_value.setText(str(current))
            self._filter_threshold_slider.blockSignals(True)
            try:
                self._filter_threshold_slider.setValue(current)
            finally:
                self._filter_threshold_slider.blockSignals(False)

    def _on_filter_threshold_changed(self, value: int) -> None:
        """Persist threshold + refresh the filter when the user moves the slider."""
        self._editor_state.set("filter_distance", int(value))
        self._refresh_filter_results()

    def _build_aligner_controls(self) -> QWidget:
        """Return the Bounding Box aligner control panel (#104).

        Hidden by default; surfaced only while ``editor_mode == "BoundingBox"``.
        The selected aligner, normalization and auto-run values are stored on
        :class:`ManualEditorState` so they persist across editor-mode switches.
        """
        container = QWidget()
        container.setObjectName("qt-manual-aligner-controls")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(QLabel("Aligner:"))

        self._aligner_combo = QComboBox()
        self._aligner_combo.setObjectName("qt-manual-aligner-combo")
        self._aligner_combo.setMinimumWidth(150)
        self._aligner_combo.currentTextChanged.connect(self._on_aligner_combo_changed)
        top.addWidget(self._aligner_combo)

        self._aligner_auto_run_checkbox = QCheckBox("Auto-run")
        self._aligner_auto_run_checkbox.setObjectName("qt-manual-aligner-auto-run")
        self._aligner_auto_run_checkbox.setChecked(bool(self._editor_state.aligner_auto_run))
        self._aligner_auto_run_checkbox.toggled.connect(self._on_aligner_auto_run_toggled)
        top.addWidget(self._aligner_auto_run_checkbox)

        self._aligner_rerun_button = QToolButton()
        self._aligner_rerun_button.setObjectName("qt-manual-aligner-rerun")
        self._aligner_rerun_button.setText("Run")
        self._aligner_rerun_button.setToolTip("Run the selected aligner for the active face")
        self._aligner_rerun_button.clicked.connect(self._on_aligner_rerun_clicked)
        top.addWidget(self._aligner_rerun_button)
        top.addStretch(1)
        outer.addLayout(top)

        norm_row = QHBoxLayout()
        norm_row.setContentsMargins(0, 0, 0, 0)
        norm_row.setSpacing(8)
        norm_row.addWidget(QLabel("Normalization:"))
        self._aligner_normalization_group = QButtonGroup(container)
        self._aligner_normalization_group.setExclusive(True)
        self._aligner_normalization_buttons: dict[str, QRadioButton] = {}
        for method in self.available_normalizations():
            button = QRadioButton(str(method))
            button.setObjectName(f"qt-manual-aligner-normalization-{method}")
            button.toggled.connect(
                lambda checked, value=str(method): (
                    self.set_aligner_normalization(value) if checked else None
                )
            )
            self._aligner_normalization_group.addButton(button)
            self._aligner_normalization_buttons[str(method)] = button
            norm_row.addWidget(button)
        norm_row.addStretch(1)
        outer.addLayout(norm_row)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)
        self._aligner_load_progress = QProgressBar()
        self._aligner_load_progress.setObjectName("qt-manual-aligner-load-progress")
        self._aligner_load_progress.setRange(0, 100)
        self._aligner_load_progress.setValue(0)
        self._aligner_load_progress.setMaximumWidth(160)
        self._aligner_load_progress.setTextVisible(True)
        self._aligner_load_progress.hide()
        progress_row.addWidget(self._aligner_load_progress)

        self._aligner_status_label = QLabel("Aligner idle")
        self._aligner_status_label.setObjectName("qt-manual-aligner-status-label")
        self._aligner_status_label.setWordWrap(True)
        progress_row.addWidget(self._aligner_status_label, 1)
        outer.addLayout(progress_row)

        container.setVisible(self._editor_state.editor_mode == "BoundingBox")
        self._sync_aligner_controls()
        return container

    def _on_aligner_combo_changed(self, name: str) -> None:
        """Handle a user-selected aligner plugin from the dropdown."""
        self.set_aligner_name(name)

    def _on_aligner_auto_run_toggled(self, checked: bool) -> None:
        """Persist the auto-run checkbox value on editor state."""
        self._editor_state.set("aligner_auto_run", bool(checked))
        self._sync_aligner_controls()

    def _on_aligner_rerun_clicked(self) -> None:
        """Run the active aligner against the selected face."""
        face_index = self._active_face_index()
        if face_index is None:
            self.statusBar().showMessage("No active face to align", 3000)
            return
        self.rerun_aligner_for_face(int(face_index))

    def _refresh_aligner_controls_visibility(self) -> None:
        """Show or hide Bounding Box aligner controls to match editor mode.

        Merely entering BoundingBox mode must not preload the production
        aligner.  Passive preload makes unrelated GUI tests and ordinary mode
        switches touch plugin/model loading.  Explicit user changes to the
        aligner dropdown or normalization radios still call
        :meth:`_schedule_aligner_preload`, and actual alignment runs continue
        to load on demand through :meth:`rerun_aligner_for_face`.
        """
        controls = getattr(self, "_aligner_controls", None)
        if controls is None:
            return
        active = self._editor_state.editor_mode == "BoundingBox"
        controls.setVisible(active)
        if active:
            self._sync_aligner_controls()

    def _sync_aligner_controls(self) -> None:
        """Mirror editor-state aligner settings into the visible controls."""
        combo = getattr(self, "_aligner_combo", None)
        if combo is None:
            return
        aligners = list(self.available_aligners())
        current = self._editor_state.aligner_name or (
            self._aligner_service.default_aligner() if aligners else ""
        )
        if current and current not in aligners:
            aligners.insert(0, current)

        combo.blockSignals(True)
        combo.clear()
        if aligners:
            combo.addItems(aligners)
            combo.setEnabled(True)
            index = combo.findText(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        else:
            combo.addItem("No aligners available")
            combo.setEnabled(False)
        combo.blockSignals(False)

        normalization = str(self._editor_state.aligner_normalization)
        for method, button in self._aligner_normalization_buttons.items():
            button.blockSignals(True)
            button.setChecked(method == normalization)
            button.blockSignals(False)
        if (
            normalization not in self._aligner_normalization_buttons
            and self._aligner_normalization_buttons
        ):
            first = next(iter(self._aligner_normalization_buttons.values()))
            first.blockSignals(True)
            first.setChecked(True)
            first.blockSignals(False)

        auto_run = getattr(self, "_aligner_auto_run_checkbox", None)
        if auto_run is not None:
            auto_run.blockSignals(True)
            auto_run.setChecked(bool(self._editor_state.aligner_auto_run))
            auto_run.blockSignals(False)

        rerun = getattr(self, "_aligner_rerun_button", None)
        if rerun is not None:
            rerun.setEnabled(bool(aligners) and self._active_face_index() is not None)

    def _schedule_aligner_preload(self) -> None:
        """Preload the selected aligner/backend while BBox controls are visible."""
        controls = getattr(self, "_aligner_controls", None)
        if controls is None or not controls.isVisible():
            return
        aligner = self._active_aligner_name()
        normalization = str(self._editor_state.aligner_normalization)
        if not aligner:
            self._set_aligner_load_status("failed", "", "No aligner plugins available")
            return
        target = (aligner, normalization)
        if target in self._aligner_loaded_targets:
            self._set_aligner_load_status("ready", aligner, f"Aligner '{aligner}' ready")
            return
        if self._aligner_load_worker is not None and self._aligner_load_target == target:
            return
        if self._aligner_load_worker is not None:
            if not self._aligner_load_worker.stop():
                self.statusBar().showMessage(
                    "Aligner is still loading — wait for it to finish before changing selection.",
                    5000,
                )
                return
            self._aligner_load_worker.deleteLater()
            self._aligner_load_worker = None

        worker = ManualAlignerLoadWorker(
            self._aligner_service,
            aligner,
            normalization,
            parent=self,
        )
        worker.status.connect(self._on_aligner_load_status)
        worker.completed.connect(self._on_aligner_load_completed)
        self._aligner_load_worker = worker
        self._aligner_load_target = target
        self._set_aligner_load_status("loading", aligner, f"Loading aligner '{aligner}'…")
        worker.start()

    def _on_aligner_load_status(self, kind: str, aligner: str, message: str) -> None:
        """Handle status emitted by the background aligner preload worker."""
        self._set_aligner_load_status(kind, aligner, message)
        timeout = 5000 if kind == "failed" else 3000
        self.statusBar().showMessage(message, timeout)
        self._emit_console(f"Manual Tool aligner: {message}")

    def _on_aligner_load_completed(
        self, aligner: str, normalization: str, ok: bool, message: str
    ) -> None:
        """Finalize the visible aligner preload state."""
        target = (str(aligner), str(normalization))
        if self._aligner_load_target is not None and target != self._aligner_load_target:
            return
        if ok:
            self._aligner_loaded_targets.add(target)
            self._set_aligner_load_status("ready", aligner, message)
        else:
            self._set_aligner_load_status("failed", aligner, message)

        worker = self._aligner_load_worker
        if worker is not None:
            worker.stop(wait_ms=1000)
            worker.deleteLater()
            self._aligner_load_worker = None

        self._aligner_load_target = None
        self._sync_aligner_controls()

    def _set_aligner_load_status(self, kind: str, aligner: str, message: str) -> None:
        """Paint the inline aligner load-progress widget."""
        progress = getattr(self, "_aligner_load_progress", None)
        label = getattr(self, "_aligner_status_label", None)
        if label is not None and message:
            label.setText(message)
        if progress is None:
            return
        if kind in ("loading", "aligning"):
            progress.setRange(0, 0)
            progress.setFormat("Loading aligner…")
            progress.show()
            rerun = getattr(self, "_aligner_rerun_button", None)
            if rerun is not None:
                rerun.setEnabled(False)
        elif kind in ("ready", "aligned"):
            progress.hide()
            progress.setRange(0, 100)
            progress.setValue(100)
        elif kind == "failed":
            progress.hide()
            progress.setRange(0, 100)
            progress.setValue(0)

    def rerun_aligner_for_face(self, face_index: int) -> bool:
        """Rerun the configured aligner against ``face_index`` on the current frame.

        Returns ``True`` if landmarks were refreshed. Failures (no image,
        empty bbox, aligner unavailable, model error) surface a status
        message + console line and never corrupt the editable model — the
        existing landmark cloud stays put.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to align", 3000)
            return False
        faces = self._editable.faces(frame_index)
        if face_index < 0 or face_index >= len(faces):
            self.statusBar().showMessage("No active face to align", 3000)
            return False
        image = self._frame_view.current_frame_array()
        if image is None:
            self.statusBar().showMessage("Cannot align: frame image not loaded", 4000)
            return False
        face = faces[face_index]
        try:
            landmarks = self._aligner_service.align(
                image,
                face.bbox,
                aligner=self._active_aligner_name() or None,
                normalization=self._editor_state.aligner_normalization,
            )
        except Exception as err:  # noqa: BLE001 - surface to the user; model untouched
            logger.exception("Manual Tool: aligner run failed")
            self.statusBar().showMessage(f"Aligner failed: {err}", 5000)
            self._emit_console(f"Manual Tool aligner failed: {err}")
            return False
        new_points = [(float(point[0]), float(point[1])) for point in landmarks]
        if not self._editable.set_landmarks(frame_index, face_index, new_points):
            return False
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()
        return True

    def _maybe_run_aligner(self, face_index: int) -> None:
        """Run the aligner after a bbox edit when ``aligner_auto_run`` is on."""
        if not self._editor_state.aligner_auto_run:
            return
        if self._editor_state.editor_mode != "BoundingBox":
            return
        self.rerun_aligner_for_face(int(face_index))

    def _face_at_source_point(self, sx: float, sy: float) -> int | None:
        """Hit-test the editable model at a source-coordinate point."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return None
        return self._editable.hit_test(frame_index, sx, sy)

    def _active_face_landmarks(self) -> tuple[tuple[float, float], ...] | None:
        """Return the active face's landmark sequence (Landmark editor seam)."""
        frame_index = self._current_frame_index()
        face_index = self._editor_state.face_index
        if frame_index < 0 or face_index < 0:
            return None
        faces = self._editable.faces(frame_index)
        if face_index >= len(faces):
            return None
        return faces[face_index].landmarks

    def _on_landmark_move_requested(
        self, face_index: int, landmark_index: int, new_x: float, new_y: float
    ) -> None:
        """Apply a single-point landmark edit and refresh the overlay."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.update_landmark(
            frame_index, int(face_index), int(landmark_index), float(new_x), float(new_y)
        ):
            self.statusBar().showMessage("Landmark edit failed", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()

    def _on_landmarks_move_requested(
        self,
        face_index: int,
        indices: T.Any,
        dx: float,
        dy: float,
    ) -> None:
        """Apply a group-move edit across the marquee-selected landmark set."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        idx_tuple = tuple(indices) if not isinstance(indices, tuple) else indices
        if not self._editable.move_landmarks(
            frame_index, int(face_index), idx_tuple, float(dx), float(dy)
        ):
            self.statusBar().showMessage("Landmark group move failed", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()

    def _on_landmarks_select_requested(self, face_index: int, indices: T.Any) -> None:
        """Update the overlay's selection set from a marquee release."""
        if face_index != self._editor_state.face_index:
            return
        idx_tuple = tuple(indices) if not isinstance(indices, tuple) else indices
        self._overlay.set_selected_landmarks(idx_tuple)
        if not idx_tuple:
            self.statusBar().showMessage("Landmark selection cleared", 2000)
        else:
            self.statusBar().showMessage(f"Selected {len(idx_tuple)} landmark(s)", 2000)
        self._frame_view.update()

    def _on_face_add_requested(self, bbox: QRectF) -> None:
        """Create a new face from a pointer-driven add gesture."""
        new_index = self.add_face_at_center((bbox.x(), bbox.y(), bbox.width(), bbox.height()))
        if new_index is None:
            self.statusBar().showMessage("Could not add face", 5000)

    def _on_frame_context_menu_requested(self, face_index: int, global_pos: QPointF) -> None:
        """Open a Delete Face context menu for a right-clicked frame face."""
        self._show_face_context_menu(face_index, global_pos)

    def _on_face_panel_context_menu_requested(self, face_index: int, global_pos: QPointF) -> None:
        """Open a Delete Face context menu for a right-clicked panel item."""
        self._show_face_context_menu(face_index, global_pos)

    def _show_face_context_menu(
        self,
        face_index: int,
        global_pos: QPointF,
        *,
        frame_index: int | None = None,
    ) -> None:
        """Build and show a Delete-Face context menu at ``global_pos``.

        Uses :meth:`QMenu.popup` (non-blocking) rather than ``exec`` so a
        right-click does not freeze the event loop while the menu is open —
        important for tests and for keeping the rest of the window responsive.
        The selected action is routed back through ``triggered``.
        """
        if self._save_in_flight:
            return
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        delete_action = menu.addAction("Delete Face")
        if frame_index is None:
            delete_action.triggered.connect(
                lambda _checked=False, fi=face_index: self._delete_face_by_index(fi)
            )
        else:
            delete_action.triggered.connect(
                lambda _checked=False, fr=frame_index, fi=face_index: self._delete_face_at(fr, fi)
            )
        menu.aboutToHide.connect(menu.deleteLater)
        menu.popup(global_pos.toPoint())

    def _delete_face_by_index(self, face_index: int) -> None:
        """Delete the face with ``face_index`` on the current frame.

        Routes through the editable model so the operation is undoable and the
        same dirty/refresh side-effects as :meth:`delete_active_face` apply.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.delete_face(frame_index, face_index):
            self.statusBar().showMessage("Could not delete face", 3000)
            return
        new_count = self._editable.face_count(frame_index)
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        active = self._editor_state.face_index
        if new_count == 0:
            self._editor_state.set("face_index", -1)
        elif active >= new_count:
            self._editor_state.set("face_index", new_count - 1)
        self.mark_dirty(True)
        self.refresh_faces()
        self._frame_view.update()
        self.statusBar().showMessage(f"Deleted face index {face_index}", 5000)

    def _delete_face_at(self, frame_index: int, face_index: int) -> None:
        """Delete a face on any source frame and refresh filtered-session views."""
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.delete_face(frame_index, face_index):
            self.statusBar().showMessage("Could not delete face", 3000)
            return
        new_count = self._editable.face_count(frame_index)
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        if frame_index == self._current_frame_index():
            active = self._editor_state.face_index
            if new_count == 0:
                self._editor_state.set("face_index", -1)
            elif active >= new_count or active == face_index:
                self._editor_state.set("face_index", min(face_index, new_count - 1))
            self.refresh_faces()
            self._frame_view.update()
        self.mark_dirty(True)
        self._refresh_filter_results()
        self._refresh_face_grid()
        self.statusBar().showMessage(
            f"Deleted face {face_index + 1} on frame {frame_index + 1}", 5000
        )

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
        entries = self._face_thumbnail_entries_for_frame(frame_index)
        if not entries:
            self._face_panel.set_faces(())
            return
        self._face_panel.set_faces(entries)

    def _face_thumbnail_entries_for_frame(self, frame_index: int) -> tuple[FaceThumbnail, ...]:
        """Return face thumbnail entries for one editable frame index."""
        editable_faces = self._editable.faces(frame_index)
        if not editable_faces:
            return ()
        frame_name = self._frame_name_for_index(frame_index)
        if frame_name is None:
            frame_name = (
                self._current_frame.name
                if self._current_frame is not None and self._current_frame.index == frame_index
                else f"frame_{frame_index:06d}"
            )
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
        return tuple(entries)

    def _face_grid_entries(self) -> tuple[FaceGridEntry, ...]:
        """Build one grid entry per visible face in the active filtered session."""
        entries: list[FaceGridEntry] = []
        for frame_index in self.filtered_frame_indices():
            frame_name = self._frame_name_for_index(frame_index) or f"frame_{frame_index:06d}"
            thumbs = {
                thumb.face_index: thumb
                for thumb in self._face_thumbnail_entries_for_frame(frame_index)
            }
            for face in self._editable.faces(frame_index):
                entries.append(
                    FaceGridEntry(
                        frame_index=frame_index,
                        frame_name=frame_name,
                        face_index=int(face.face_index),
                        thumbnail=thumbs.get(int(face.face_index)),
                        bbox=face.bbox,
                        landmarks=face.landmarks,
                    )
                )
        return tuple(entries)

    def _refresh_face_grid(self) -> None:
        """Rebuild the cross-frame face grid from filters, state and annotations."""
        panel = getattr(self, "_face_grid_panel", None)
        if panel is None:
            return
        panel.set_overlay_state(
            show_mesh=self._editor_state.annotation_mode == "Mesh",
            show_mask=self._should_render_mask(),
            mask_type=self.active_mask_type(),
            mask_opacity=int(self._editor_state.mask_opacity),
        )
        panel.set_entries(self._face_grid_entries())
        self._refresh_face_grid_active()

    def _refresh_face_grid_active(self) -> None:
        """Update active frame/face styling without rebuilding thumbnail icons."""
        panel = getattr(self, "_face_grid_panel", None)
        if panel is None:
            return
        panel.set_active(self._current_frame_index(), int(self._editor_state.face_index))

    def _on_face_grid_activated(self, frame_index: int, face_index: int) -> None:
        """Navigate to the clicked grid face and make it the active face."""
        if frame_index < 0:
            return
        self._stop_playback()
        if self._thumbnail_panel.currentRow() != frame_index:
            self._thumbnail_panel.setCurrentRow(frame_index)
        self._editor_state.set("face_index", int(face_index))
        self._face_panel.select_face(int(face_index))
        self._face_grid_panel.set_active(int(frame_index), int(face_index))
        self._frame_view.update()
        self._sync_actions()

    def _on_face_grid_hovered(self, frame_index: int, face_index: int) -> None:
        """Surface simple hover feedback for cross-frame thumbnails."""
        self.statusBar().showMessage(
            f"Frame {frame_index + 1} / Face {face_index + 1}",
            2000,
        )

    def _on_face_grid_context_menu_requested(
        self,
        frame_index: int,
        face_index: int,
        global_pos: QPointF,
    ) -> None:
        """Open a Delete Face context menu for a cross-frame grid item."""
        self._show_face_context_menu(face_index, global_pos, frame_index=frame_index)

    def _on_face_grid_size_changed(self, size_name: str) -> None:
        """Persist a user-selected face-grid size."""
        if size_name not in _FACE_GRID_SIZES:
            return
        self._editor_state.set("faces_size", size_name)

    def _on_face_grid_size_state_changed(self, size_name: object) -> None:
        """Apply persisted face-grid size state and relayout immediately."""
        name = str(size_name or "Medium")
        if name not in _FACE_GRID_SIZES:
            name = "Medium"
        combo = getattr(self, "_face_grid_size_combo", None)
        if combo is not None and combo.currentText() != name:
            combo.blockSignals(True)
            try:
                combo.setCurrentText(name)
            finally:
                combo.blockSignals(False)
        self._face_grid_panel.set_face_size(name)
        self._refresh_face_grid()

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
        """Select previous frame from the active filter (#107)."""
        self._stop_playback()
        if not self._filtered_frame_indices:
            return
        position = self._filtered_position()
        if position <= 0:
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position - 1])

    def _next_frame(self) -> None:
        """Select next frame from the active filter (#107)."""
        self._stop_playback()
        if not self._filtered_frame_indices:
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position + 1])

    _MUTATING_ACTION_KEYS: T.ClassVar[tuple[str, ...]] = (
        "save",
        "revert_frame",
        "copy_prev_face",
        "copy_next_face",
        "delete_face",
        "add_face",
        "undo_edit",
        "redo_edit",
        "extract_faces",
    )

    def _sync_actions(self) -> None:
        """Update action availability from session, selection and dirty state.

        Navigation gates run off the *filtered* frame list (#107): first /
        previous / next / last / play are disabled when no frame matches
        the active filter, and enabled bounds follow the filtered position
        rather than the raw thumbnail row.  ``copy_prev_face`` /
        ``copy_next_face`` use the same bounds so a filter that hides
        adjacent frames also hides the copy-from-neighbor affordance.
        """
        if not self._actions:
            return
        filtered = self._filtered_frame_indices
        filtered_total = len(filtered)
        has_frames = filtered_total > 0
        if has_frames:
            filtered_position = self._filtered_position()
            not_first = filtered_position > 0
            not_last = 0 <= filtered_position < filtered_total - 1
        else:
            not_first = False
            not_last = False
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
            "extract_faces": has_frames and self._alignments_handle.exists,
        }
        # When a blocking operation is in flight, disable every mutating /
        # save-conflicting action regardless of the underlying availability so
        # the user cannot race the save (or future bulk job).  Read-only
        # navigation, mode switches and view zoom remain enabled so they don't
        # feel unresponsive while the worker runs.
        if self._busy_operation is not None:
            for key in self._MUTATING_ACTION_KEYS:
                if key in availability:
                    availability[key] = False
        for key, enabled in availability.items():
            action = self._actions.get(key)
            if action is not None:
                action.setEnabled(enabled)

    def _on_editor_mode_changed(self, _mode: object) -> None:
        """Refresh action availability when the editor mode flips."""
        self._sync_actions()
        self._refresh_mask_controls_visibility()
        self._refresh_aligner_controls_visibility()

    MASK_DEFAULT_TYPES: T.ClassVar[tuple[str, ...]] = (
        "components",
        "extended",
        "bisenet-fp_face",
        "bisenet-fp_head",
        "vgg-clear",
        "vgg-obstructed",
        "unet-dfl",
        "custom",
    )
    """Faceswap-standard mask plugins.  The dropdown always shows these so
    the user can pick a type to paint *before* any blob exists on disk."""

    def _build_mask_controls(self) -> QWidget:
        """Return the Mask-editor control row (mask-type dropdown + opacity).

        Hidden by default; surfaced only while ``editor_mode == "Mask"``.
        The dropdown is populated lazily by
        :meth:`_refresh_mask_controls_visibility` so the option list always
        reflects whatever the active face has persisted on disk + the
        Faceswap-standard set.
        """
        container = QWidget()
        container.setObjectName("qt-manual-mask-controls")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Mask type:"))
        self._mask_type_combo = QComboBox()
        self._mask_type_combo.setObjectName("qt-manual-mask-type-combo")
        self._mask_type_combo.setMinimumWidth(140)
        self._mask_type_combo.currentTextChanged.connect(self._on_mask_type_combo_changed)
        layout.addWidget(self._mask_type_combo)

        layout.addSpacing(12)
        layout.addWidget(QLabel("Opacity:"))
        self._mask_opacity_slider = QSlider(Qt.Horizontal)
        self._mask_opacity_slider.setObjectName("qt-manual-mask-opacity")
        self._mask_opacity_slider.setRange(0, 100)
        self._mask_opacity_slider.setValue(int(self._editor_state.mask_opacity))
        self._mask_opacity_slider.setFixedWidth(140)
        self._mask_opacity_slider.valueChanged.connect(self._on_mask_opacity_changed)
        layout.addWidget(self._mask_opacity_slider)
        layout.addStretch(1)

        container.setVisible(self._editor_state.editor_mode == "Mask")
        return container

    def _refresh_mask_controls_visibility(self) -> None:
        """Show or hide the Mask control row to match the editor mode."""
        controls = getattr(self, "_mask_controls", None)
        if controls is None:
            return
        active = self._editor_state.editor_mode == "Mask"
        controls.setVisible(active)
        if active:
            self._refresh_mask_type_combo()

    def _refresh_mask_type_combo(self) -> None:
        """Populate the mask-type dropdown for the current frame + face.

        Combines:
        * Faceswap-standard mask plugins (always available so the user can
          start painting a type that doesn't yet exist on disk).
        * Any mask type that already exists for the active face in memory
          or on disk via :meth:`ManualEditableAlignments.known_mask_types`.

        Preserves the user's current selection when possible; falls back
        to :attr:`DEFAULT_MASK_TYPE` when no mask_type is set yet.
        """
        combo = getattr(self, "_mask_type_combo", None)
        if combo is None:
            return
        frame_index = self._current_frame_index()
        face_index = int(self._editor_state.face_index)
        existing: tuple[str, ...] = ()
        if frame_index >= 0 and face_index >= 0:
            existing = self._editable.known_mask_types(frame_index, face_index)
        options = sorted({*self.MASK_DEFAULT_TYPES, *existing})
        if not options:
            return
        current = self._editor_state.mask_type or self.DEFAULT_MASK_TYPE
        if current not in options:
            current = self.DEFAULT_MASK_TYPE if self.DEFAULT_MASK_TYPE in options else options[0]
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(options)
            combo.setCurrentText(current)
        finally:
            combo.blockSignals(False)
        if self._editor_state.mask_type != current:
            self._editor_state.set("mask_type", current)

    def _on_mask_type_combo_changed(self, mask_type: str) -> None:
        """Persist the chosen mask type to editor state + refresh the overlay."""
        if not mask_type:
            return
        self._editor_state.set("mask_type", mask_type)
        self._frame_view.update()

    def _on_mask_opacity_changed(self, value: int) -> None:
        """Push the new opacity onto editor state + repaint the overlay."""
        self._editor_state.set("mask_opacity", int(value))
        self._frame_view.update()


__all__ = [
    "MANUAL_ACTIONS",
    "CrossFrameFaceGridPanel",
    "FaceGridEntry",
    "FaceGridRenderRequest",
    "FaceGridThumbnailRenderer",
    "FaceThumbnailPanel",
    "FrameViewport",
    "ManualAction",
    "ManualFrameOverlay",
    "ManualFrameView",
    "ManualThumbnailPanel",
    "ManualToolWindow",
    "VideoFrameProvider",
]
