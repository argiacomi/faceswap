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
    QPolygonF,
)

from tools.manual.session import ManualEditableAlignments

from .viewport import FrameViewport

logger = logging.getLogger(__name__)


class ManualFrameOverlay:
    """Stateful painter for editable frame annotations."""

    _BBOX_COLOR = QColor("#0000ff")
    _EXTRACT_COLOR = QColor("#00ff00")
    _LANDMARK_COLOR = QColor("#ff00ff")
    _LANDMARK_SELECTED_COLOR = QColor("#ff00ff")
    _MESH_COLOR = QColor("#00ffff")
    _MASK_COLOR = QColor("#ff0000")
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
            extract_polygon = self._extract_polygon_for(face.landmarks, viewport)
            if visibility["bbox"]:
                self._draw_rect(painter, rect, self._color("bbox"), pen_width)
            if visibility["extract"] and extract_polygon is not None:
                self._draw_polygon(painter, extract_polygon, self._color("extract"), pen_width)
            if visibility["mesh"] and face.landmarks:
                self._draw_mesh(painter, viewport, face.landmarks, pen_width)
            if visibility["landmarks"] and face.landmarks:
                self._draw_landmarks(painter, viewport, face.landmarks, is_active)
            if is_active:
                if visibility["mask"]:
                    self._paint_mask_overlay(painter, viewport, face)
                if visibility["handles"]:
                    handle_rect = (
                        extract_polygon.boundingRect() if extract_polygon is not None else rect
                    )
                    self._draw_handles(painter, handle_rect)
            painter.restore()

    def annotation_visibility(self) -> dict[str, bool]:
        """Return active legacy overlay visibility flags."""
        editor_mode = self._editor_mode_provider() if self._editor_mode_provider else "View"
        annotation_mode = (
            self._annotation_mode_provider() if self._annotation_mode_provider else "None"
        )
        if not annotation_mode:
            annotation_mode = "None"
        explicit_mesh = annotation_mode == "Mesh"
        explicit_mask = annotation_mode == "Mask"
        explicit_landmarks = annotation_mode == "Landmarks"
        if editor_mode == "View":
            return {
                "bbox": True,
                "extract": True,
                "handles": False,
                "landmarks": True,
                "mesh": True,
                "mask": explicit_mask,
            }
        if editor_mode == "BoundingBox":
            return {
                "bbox": True,
                "extract": False,
                "handles": True,
                "landmarks": explicit_landmarks,
                "mesh": True,
                "mask": explicit_mask,
            }
        if editor_mode == "ExtractBox":
            return {
                "bbox": False,
                "extract": True,
                "handles": True,
                "landmarks": explicit_landmarks,
                "mesh": True,
                "mask": explicit_mask,
            }
        if editor_mode == "Landmarks":
            return {
                "bbox": False,
                "extract": True,
                "handles": False,
                "landmarks": True,
                "mesh": True,
                "mask": explicit_mask,
            }
        if editor_mode == "Mask":
            return {
                "bbox": False,
                "extract": False,
                "handles": False,
                "landmarks": False,
                "mesh": explicit_mesh,
                "mask": True,
            }
        return {
            "bbox": explicit_mesh or explicit_landmarks,
            "extract": False,
            "handles": False,
            "landmarks": explicit_landmarks,
            "mesh": explicit_mesh,
            "mask": explicit_mask,
        }

    def _color(self, role: str) -> QColor:
        """Return the configured overlay color for ``role``."""
        if self._color_provider is not None:
            color = self._color_provider(role)
            if color.isValid():
                return color
        defaults = {
            "bbox": self._BBOX_COLOR,
            "extract": self._EXTRACT_COLOR,
            "landmark": self._LANDMARK_COLOR,
            "landmark_selected": self._LANDMARK_SELECTED_COLOR,
            "mesh": self._MESH_COLOR,
            "mask": self._MASK_COLOR,
        }
        return defaults.get(role, self._BBOX_COLOR)

    @staticmethod
    def _draw_rect(painter: QPainter, rect: QRectF, color: QColor, pen_width: float) -> None:
        pen = QPen(color)
        pen.setWidthF(pen_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

    @staticmethod
    def _draw_polygon(
        painter: QPainter,
        polygon: QPolygonF,
        color: QColor,
        pen_width: float,
    ) -> None:
        pen = QPen(color)
        pen.setWidthF(pen_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(polygon)

    def _draw_landmarks(
        self,
        painter: QPainter,
        viewport: FrameViewport,
        landmarks: T.Sequence[tuple[float, float]],
        is_active: bool,
    ) -> None:
        painter.setPen(Qt.NoPen)
        selected = self._selected_landmarks if is_active else frozenset()
        for lm_index, (lx, ly) in enumerate(landmarks):
            point = viewport.source_to_widget(lx, ly)
            if lm_index in selected:
                painter.setBrush(QBrush(self._color("landmark_selected")))
                radius = self._LANDMARK_SELECTED_RADIUS
            else:
                painter.setBrush(QBrush(self._color("landmark")))
                radius = self._LANDMARK_RADIUS
            painter.drawEllipse(point, radius, radius)

    def _draw_mesh(
        self,
        painter: QPainter,
        viewport: FrameViewport,
        landmarks: T.Sequence[tuple[float, float]],
        pen_width: float,
    ) -> None:
        groups = self.landmark_part_groups(landmarks)
        if not groups["polygon"] and not groups["line"]:
            return
        pen = QPen(self._color("mesh"))
        pen.setWidthF(max(1.0, pen_width))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for group in groups["polygon"]:
            painter.drawPolygon(
                QPolygonF(tuple(viewport.source_to_widget(x, y) for x, y in group))
            )
        for group in groups["line"]:
            painter.drawPolyline(
                QPolygonF(tuple(viewport.source_to_widget(x, y) for x, y in group))
            )

    @classmethod
    def landmark_part_groups(
        cls,
        landmarks: T.Sequence[tuple[float, float]],
    ) -> dict[T.Literal["polygon", "line"], tuple[tuple[tuple[float, float], ...], ...]]:
        """Return Tk-parity LANDMARK_PARTS groups for ``landmarks``."""
        if not landmarks:
            return {"polygon": (), "line": ()}
        try:
            import numpy as np

            from lib.align import LANDMARK_PARTS, LandmarkType

            landmark_array = np.asarray(landmarks, dtype=np.float32)
            landmark_type = LandmarkType.from_shape(landmark_array.shape)
            groups: dict[T.Literal["polygon", "line"], list[tuple[tuple[float, float], ...]]] = {
                "polygon": [],
                "line": [],
            }
            for start, end, fill in LANDMARK_PARTS[landmark_type].values():
                key: T.Literal["polygon", "line"] = "polygon" if fill else "line"
                groups[key].append(
                    tuple((float(x), float(y)) for x, y in landmark_array[start:end].tolist())
                )
            return {key: tuple(value) for key, value in groups.items()}
        except Exception:
            return {"polygon": (), "line": (tuple(landmarks),)}

    @staticmethod
    def _extract_polygon_for(
        landmarks: T.Sequence[tuple[float, float]],
        viewport: FrameViewport,
    ) -> QPolygonF | None:
        """Return the legacy aligned-face extract-box polygon in widget coordinates."""
        if not landmarks:
            return None
        try:
            import numpy as np

            from lib.align import AlignedFace

            aligned = AlignedFace(np.asarray(landmarks, dtype=np.float32), centering="face")
            roi = getattr(aligned, "original_roi", None)
            if roi is None:
                return None
            points = tuple(viewport.source_to_widget(float(x), float(y)) for x, y in roi.tolist())
            if len(points) < 3:
                return None
            return QPolygonF(points)
        except Exception:
            return None

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
        except Exception:
            return
        rect = viewport.source_rect_to_widget(*face.bbox)
        painter.save()
        painter.setOpacity(1.0)
        painter.drawImage(rect, image)
        painter.restore()

    def _draw_handles(self, painter: QPainter, bbox: QRectF) -> None:
        """Draw eight square resize handles around ``bbox``."""
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
