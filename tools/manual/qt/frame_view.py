#!/usr/bin/env python3
"""Qt Manual Tool implementation module."""

from __future__ import annotations

# ruff: noqa: F401
import contextlib
import json
import logging
import os
import subprocess
import typing as T
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import (
    QByteArray,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
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
    QPolygonF,
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
from .overlays import ManualFrameOverlay
from .types import FrameViewport, OverlayPainter


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
