#!/usr/bin/env python3
"""Generic frame display widget for the Qt Manual Tool."""

from __future__ import annotations

import logging
import typing as T
from collections.abc import Sequence

from PySide6.QtCore import (
    QPoint,
    QPointF,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QSizePolicy,
    QWidget,
)

from tools.manual.session import (
    ManualFrame,
)

from .editor.bounding_box import BoundingBoxFrameEditorMixin
from .editor.drag import FrameEditDragMixin
from .editor.extract_box import ExtractBoxFrameEditorMixin
from .editor.landmarks import LandmarkFrameEditorMixin
from .editor.mask import MaskFrameEditorMixin
from .viewport import FrameViewport, OverlayPainter

logger = logging.getLogger(__name__)


class ManualFrameView(
    MaskFrameEditorMixin,
    LandmarkFrameEditorMixin,
    ExtractBoxFrameEditorMixin,
    BoundingBoxFrameEditorMixin,
    FrameEditDragMixin,
    QWidget,
):
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
    mask_paint_requested = Signal(int, float, float, bool, bool)
    """``(face_index, source_x, source_y, invert, started)`` during a Mask editor stroke.

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
        self._landmark_faces_provider: (
            T.Callable[[], T.Sequence[tuple[int, T.Sequence[tuple[float, float]]]]] | None
        ) = None
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
        # ``(radius_source_px, mode, shape, color)`` tuple while the mask editor is active
        # *and* the pointer is inside the active face's bbox, the view draws
        # a circular cursor preview so the user can see brush size + mode
        # before clicking.  Hidden in any other state.
        self._brush_provider: T.Callable[[], tuple[float, str, str, str] | None] | None = None
        self._mask_roi_provider: T.Callable[[float, float], bool] | None = None
        self._brush_preview_source_point: QPointF | None = None
        self._brush_preview_radius: float = 0.0
        self._brush_preview_mode: str = "draw"
        self._brush_preview_shape: str = "Circle"
        self._brush_preview_color: str = "#ffffff"

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

    def view_state(self) -> dict[str, object]:
        """Return a compact zoom/pan state snapshot."""
        return {
            "zoom": float(self._zoom),
            "offset": (float(self._offset.x()), float(self._offset.y())),
        }

    def restore_view_state(self, state: T.Mapping[str, object]) -> bool:
        """Restore a previously captured zoom/pan state."""
        try:
            zoom = float(state.get("zoom", 1.0))
            offset_raw = state.get("offset", (0.0, 0.0))
            if not isinstance(offset_raw, Sequence) or len(offset_raw) != 2:
                return False
            ox = float(offset_raw[0])
            oy = float(offset_raw[1])
        except (TypeError, ValueError):
            return False
        self._zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, zoom))
        self._offset = QPointF(float(ox), float(oy))
        self._clamp_offset()
        self.update()
        self.view_changed.emit()
        return True

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
        landmark_faces_provider: (
            T.Callable[[], T.Sequence[tuple[int, T.Sequence[tuple[float, float]]]]] | None
        ) = None,
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
        self._landmark_faces_provider = landmark_faces_provider

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
        brush_provider: T.Callable[[], tuple[float, str, str, str] | None] | None = None,
        mask_roi_provider: T.Callable[[float, float], bool] | None = None,
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
        self._mask_roi_provider = mask_roi_provider

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
            "shape": str(self._brush_preview_shape),
            "color": str(self._brush_preview_color),
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
        painter.fillRect(self.rect(), QColor("#000000"))
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
