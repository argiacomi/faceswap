#!/usr/bin/env python3
"""Overlay painters for Qt Manual Tool frame annotations."""

from __future__ import annotations

import logging
import typing as T

from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
)

from tools.manual.session import ManualEditableAlignments

from .viewport import FrameViewport

logger = logging.getLogger(__name__)


class ManualFrameOverlay:
    """Stateful painter for editable bounding boxes + landmarks.

    The overlay is registered with :meth:`ManualFrameView.add_overlay` and
    draws one rectangle per editable face returned by the bound
    :class:`ManualEditableAlignments`.  The active face is rendered with a
    contrasting accent and a halo so editor surfaces can share hit targets.
    """

    _DEFAULT_COLOR = QColor("#3aa0ff")
    _ACTIVE_COLOR = QColor("#ffb000")
    _LANDMARK_COLOR = QColor("#ffffff")
    _LANDMARK_SELECTED_COLOR = QColor("#ffb000")
    _LANDMARK_RADIUS = 2.0
    _LANDMARK_SELECTED_RADIUS = 3.5
    _HANDLE_SIZE = 7.0
    _HANDLE_FILL = QColor("#ffffff")
    _HANDLE_EDGE = QColor("#11191c")
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
        self._selected_landmarks: frozenset[int] = frozenset()
        self._mask_type_provider: T.Callable[[], str] | None = None
        self._mask_opacity_provider: T.Callable[[], int] | None = None
        self._mask_show_provider: T.Callable[[], bool] | None = None
        self._color_provider: T.Callable[[str], QColor] | None = None
        self._editor_mode_provider: T.Callable[[], str] | None = None
        self._annotation_mode_provider: T.Callable[[], str] | None = None

    def set_active(self, face_index: int | None) -> None:
        """Mark a face as the active selection for highlight rendering."""
        self._active_face = face_index
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
        """Hook the Mask editor render seam."""
        self._mask_type_provider = mask_type_provider
        self._mask_opacity_provider = mask_opacity_provider
        self._mask_show_provider = mask_show_provider

    def install_color_provider(self, provider: T.Callable[[str], QColor]) -> None:
        """Hook editor-aware overlay colors."""
        self._color_provider = provider

    def install_visibility_providers(
        self,
        *,
        editor_mode_provider: T.Callable[[], str],
        annotation_mode_provider: T.Callable[[], str],
    ) -> None:
        """Hook editor/annotation state so overlays follow the display matrix."""
        self._editor_mode_provider = editor_mode_provider
        self._annotation_mode_provider = annotation_mode_provider

    def __call__(self, painter: QPainter, viewport: FrameViewport) -> None:
        """Draw the overlay during :meth:`ManualFrameView.paintEvent`."""
        frame_index = self._frame_index_provider()
        faces = self._model.faces(frame_index)
        if not faces:
            return
        visibility = self.annotation_visibility()
        if not any(visibility.values()):
            return
        pen_width = max(1.0, 1.5 * (1.0 / max(viewport.zoom, 0.001)) * viewport.zoom)
        for face in faces:
            is_active = face.face_index == self._active_face
            painter.save()
            rect = viewport.source_rect_to_widget(*face.bbox)
            if visibility["bbox"]:
                color = self._color("active") if is_active else self._color("bbox")
                pen = QPen(color)
                pen.setWidthF(pen_width + (1.0 if is_active else 0.0))
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
            if visibility["landmarks"] and face.landmarks:
                painter.setPen(Qt.NoPen)
                selected = self._selected_landmarks if is_active else frozenset()
                for lm_index, (lx, ly) in enumerate(face.landmarks):
                    point = viewport.source_to_widget(lx, ly)
                    if lm_index in selected:
                        painter.setBrush(QBrush(self._color("landmark_selected")))
                        radius = self._LANDMARK_SELECTED_RADIUS
                    else:
                        painter.setBrush(QBrush(self._color("landmark")))
                        radius = self._LANDMARK_RADIUS
                    painter.drawEllipse(point, radius, radius)
            if is_active:
                if visibility["mask"]:
                    self._paint_mask_overlay(painter, viewport, face)
                if visibility["handles"]:
                    self._draw_handles(painter, rect)
            painter.restore()

    def annotation_visibility(self) -> dict[str, bool]:
        """Return active overlay visibility for bbox, handles, landmarks and mask."""
        editor_mode = self._editor_mode_provider() if self._editor_mode_provider else "View"
        annotation_mode = (
            self._annotation_mode_provider() if self._annotation_mode_provider else "None"
        )
        if not annotation_mode:
            annotation_mode = "None"
        return {
            "bbox": editor_mode in {"BoundingBox", "ExtractBox", "Landmarks", "Mask"}
            or annotation_mode in {"Mesh", "Mask", "Landmarks"},
            "handles": editor_mode in {"BoundingBox", "ExtractBox"},
            "landmarks": editor_mode in {"ExtractBox", "Landmarks"}
            or annotation_mode in {"Mesh", "Landmarks"},
            "mask": editor_mode == "Mask" or annotation_mode == "Mask",
        }

    def _color(self, role: str) -> QColor:
        """Return the configured overlay color for ``role``."""
        if self._color_provider is not None:
            color = self._color_provider(role)
            if color.isValid():
                return color
        defaults = {
            "bbox": self._DEFAULT_COLOR,
            "active": self._ACTIVE_COLOR,
            "landmark": self._LANDMARK_COLOR,
            "landmark_selected": self._LANDMARK_SELECTED_COLOR,
            "mask": QColor(255, 80, 80),
        }
        return defaults.get(role, self._DEFAULT_COLOR)

    def _paint_mask_overlay(self, painter: QPainter, viewport: FrameViewport, face: T.Any) -> None:
        """Render the active face's mask as a colour-tinted alpha layer."""
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
        try:
            mask_array = mask
            if mask_array.size == 0:
                return
            height, width = mask_array.shape[:2]
            from PySide6.QtGui import QImage

            argb = bytearray(width * height * 4)
            tint = self._color("mask")
            alpha_factor = opacity_pct / 100.0
            mv = memoryview(argb)
            data = mask_array.tobytes()
            for i, value in enumerate(data):
                base = i * 4
                if value == 0:
                    mv[base : base + 4] = b"\x00\x00\x00\x00"
                    continue
                a = int(min(255, value * alpha_factor))
                mv[base] = tint.blue()
                mv[base + 1] = tint.green()
                mv[base + 2] = tint.red()
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
    def handle_at(cls, bbox: QRectF, widget_point: QPointF, tolerance: float = 0.0) -> str | None:
        """Return the handle name under ``widget_point``."""
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
        """Return the index of the landmark within ``tolerance`` of ``source_point``."""
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
        """Return indices of every landmark inside ``source_rect`` (inclusive)."""
        if source_rect.width() <= 0.0 or source_rect.height() <= 0.0:
            return ()
        x0 = source_rect.x()
        y0 = source_rect.y()
        x1 = x0 + source_rect.width()
        y1 = y0 + source_rect.height()
        return tuple(
            index for index, (lx, ly) in enumerate(landmarks) if x0 <= lx <= x1 and y0 <= ly <= y1
        )
