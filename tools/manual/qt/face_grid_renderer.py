#!/usr/bin/env python3
"""Qt Manual Tool implementation module."""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass

from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)

from tools.manual.session import (
    FaceThumbnail,
    ManualEditableAlignments,
)

logger = logging.getLogger(__name__)
from .thumbnails import _decode_jpeg_to_qimage

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
        groups = self.mesh_groups_for_entry(entry, icon_size)
        if not groups["line"] and not groups["polygon"]:
            return
        painter.save()
        pen = QPen(self._MESH_PEN)
        pen.setWidthF(max(1.0, icon_size / 80.0))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for polygon in groups["polygon"]:
            painter.drawPolygon(QPolygonF(polygon))
        for line in groups["line"]:
            painter.drawPolyline(QPolygonF(line))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._LANDMARK_FILL))
        radius = max(1.0, icon_size / 42.0)
        for group in (*groups["polygon"], *groups["line"]):
            for point in group:
                painter.drawEllipse(point, radius, radius)
        painter.restore()

    @classmethod
    def mesh_groups_for_entry(
        cls,
        entry: FaceGridEntry,
        icon_size: int,
    ) -> dict[T.Literal["polygon", "line"], tuple[tuple[QPointF, ...], ...]]:
        """Return Tk-parity mesh groups for ``entry`` scaled to ``icon_size``.

        Tk's face viewer builds F9 mesh annotations from ``AlignedFace`` and
        ``LANDMARK_PARTS``.  Use the same route for supported landmark shapes
        so thumbnail mesh overlays are not just a raw landmark polyline.
        """
        if not entry.landmarks:
            return {"polygon": (), "line": ()}
        try:
            import numpy as np

            from lib.align import LANDMARK_PARTS, AlignedFace

            landmark_array = np.asarray(entry.landmarks, dtype=np.float32)
            aligned = AlignedFace(landmark_array, centering="face", size=int(icon_size))
            groups: dict[T.Literal["polygon", "line"], list[tuple[QPointF, ...]]] = {
                "polygon": [],
                "line": [],
            }
            for start, end, fill in LANDMARK_PARTS[aligned.landmark_type].values():
                shape: T.Literal["polygon", "line"] = "polygon" if fill else "line"
                groups[shape].append(
                    tuple(
                        QPointF(float(x), float(y))
                        for x, y in aligned.landmarks[start:end].tolist()
                    )
                )
            return {key: tuple(value) for key, value in groups.items()}
        except Exception:  # pragma: no cover - fallback for partial/custom landmarks
            return cls._bbox_scaled_mesh_groups(entry, icon_size)

    @staticmethod
    def _bbox_scaled_mesh_groups(
        entry: FaceGridEntry,
        icon_size: int,
    ) -> dict[T.Literal["polygon", "line"], tuple[tuple[QPointF, ...], ...]]:
        """Fallback mesh grouping for incomplete/custom landmarks."""
        x, y, width, height = entry.bbox
        if width <= 0.0 or height <= 0.0:
            return {"polygon": (), "line": ()}
        points = tuple(
            QPointF(
                max(0.0, min(float(icon_size), ((lx - x) / width) * icon_size)),
                max(0.0, min(float(icon_size), ((ly - y) / height) * icon_size)),
            )
            for lx, ly in entry.landmarks
        )
        return {"polygon": (), "line": (points,) if points else ()}
