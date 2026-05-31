#!/usr/bin/env python3
"""Mask editor controllers for the Qt Manual Tool."""

from __future__ import annotations

import typing as T

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QToolButton, QWidget

from tools.manual.qt.frame_viewer.viewport import FrameViewport

EDITOR_MODE = "Mask"


def is_active(editor_mode: str) -> bool:
    """Return whether ``editor_mode`` selects this editor."""
    return editor_mode == EDITOR_MODE


class MaskFrameEditorMixin:
    """Frame-view gesture and preview behavior for the Mask editor."""

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
        previous = self._brush_preview_source_point is not None  # type: ignore[has-type]
        cleared = self._clear_brush_preview_if_no_active_modes()
        if cleared and previous:
            self.update()  # type: ignore[attr-defined]
            return
        if cleared:
            return
        source_point = self._widget_to_source(position)  # type: ignore[attr-defined]
        if source_point is None:
            if previous:
                self._brush_preview_source_point = None
                self.update()  # type: ignore[attr-defined]
            return
        bbox = self._active_bbox_source_rect()  # type: ignore[attr-defined]
        roi_contains = (
            self._mask_roi_provider(source_point.x(), source_point.y())  # type: ignore[attr-defined]
            if self._mask_roi_provider is not None  # type: ignore[attr-defined]
            else bbox is not None and bbox.contains(source_point)
        )
        if not roi_contains:
            if previous:
                self._brush_preview_source_point = None
                self.update()  # type: ignore[attr-defined]
            return
        assert (
            self._brush_provider is not None  # type: ignore[attr-defined]
        )  # narrowed by _clear_brush_preview_if_no_active_modes
        spec = self._brush_provider()  # type: ignore[attr-defined]
        if spec is None:
            if previous:
                self._brush_preview_source_point = None
                self.update()  # type: ignore[attr-defined]
            return
        radius, mode, shape, color = spec
        self._brush_preview_source_point = QPointF(source_point)
        self._brush_preview_radius = max(0.5, float(radius))
        self._brush_preview_mode = str(mode)
        self._brush_preview_shape = str(shape)
        self._brush_preview_color = str(color)
        self.update()  # type: ignore[attr-defined]

    def _clear_brush_preview_if_no_active_modes(self) -> bool:
        """Wipe brush preview when Mask mode / providers are inactive.

        Returns ``True`` when the preview was cleared (caller may need to
        request a repaint).
        """
        if (
            self._mask_mode_provider is None  # type: ignore[attr-defined]
            or not self._mask_mode_provider()  # type: ignore[attr-defined]
            or self._brush_provider is None  # type: ignore[attr-defined]
            or self._source.isNull()  # type: ignore[attr-defined]
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
            self.update()  # type: ignore[attr-defined]
        super().leaveEvent(event)  # type: ignore[misc]

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
        if self._mask_mode_provider is None or not self._mask_mode_provider():  # type: ignore[attr-defined]
            return False
        if source_point is None:
            return False
        if self._active_face_provider is None:  # type: ignore[attr-defined]
            return False
        face_index = self._active_face_provider()  # type: ignore[attr-defined]
        if face_index is None:
            return False
        if self._mask_roi_provider is not None:  # type: ignore[attr-defined]
            if not self._mask_roi_provider(source_point.x(), source_point.y()):  # type: ignore[attr-defined]
                return False
        else:
            bbox = self._active_bbox_provider() if self._active_bbox_provider else None  # type: ignore[attr-defined]
            if bbox is None or not bbox.contains(source_point):
                return False
        self._mask_drag_active = True
        self._mask_drag_invert = bool(event.modifiers() & Qt.ControlModifier)  # type: ignore[attr-defined]
        self.mask_paint_requested.emit(  # type: ignore[attr-defined]
            int(face_index),
            float(source_point.x()),
            float(source_point.y()),
            bool(self._mask_drag_invert),
            True,
        )
        return True

    def _continue_mask_drag(self, position: QPointF) -> bool:
        """Forward a move event during an in-flight Mask stroke."""
        if not self._mask_drag_active:
            return False
        source_point = self._widget_to_source(position)  # type: ignore[attr-defined]
        if source_point is None:
            return True  # swallow moves that exit the source rect
        face_index = self._active_face_provider() if self._active_face_provider else None  # type: ignore[attr-defined]
        if face_index is None:
            return True
        self.mask_paint_requested.emit(  # type: ignore[attr-defined]
            int(face_index),
            float(source_point.x()),
            float(source_point.y()),
            bool(self._mask_drag_invert),
            False,
        )
        return True

    def _end_mask_drag(self) -> None:
        """Tear down Mask drag state on release."""
        self._mask_drag_active = False
        self._mask_drag_invert = False

    def _paint_brush_preview(self, painter: QPainter, viewport: FrameViewport) -> None:
        """Draw a circular brush preview at the pointer in Mask mode (#101).

        Only paints when ``_brush_preview_source_point`` is set — the hover
        path keeps that state in sync with the pointer position, the
        active face's bbox, the editor mode and the brush size.  The
        cursor uses the configured legacy cursor colour and shape.
        """
        point = self._brush_preview_source_point
        if point is None:
            return
        widget_point = viewport.source_to_widget(point.x(), point.y())
        radius_widget = max(1.0, float(self._brush_preview_radius) * float(viewport.zoom))
        painter.save()
        colour = QColor(self._brush_preview_color)
        if not colour.isValid():
            colour = QColor("#ffffff")
        pen = QPen(colour)
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)  # type: ignore[attr-defined]
        if self._brush_preview_shape == "Rectangle":
            painter.drawRect(  # type: ignore[call-overload]
                widget_point.x() - radius_widget,
                widget_point.y() - radius_widget,
                radius_widget * 2.0,
                radius_widget * 2.0,
            )
        else:
            painter.drawEllipse(widget_point, radius_widget, radius_widget)
        painter.restore()


class MaskWindowEditorMixin:
    """Root-window adapter methods for Mask editor state and model updates."""

    _BRUSH_MIN: T.ClassVar[int] = 1
    _BRUSH_MAX: T.ClassVar[int] = 100
    _BRUSH_STEP: T.ClassVar[int] = 2
    DEFAULT_MASK_TYPE: T.ClassVar[str] = "components"
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

    def _is_mask_mode_active(self) -> bool:
        """Return whether the Mask editor (F5) is active."""
        return self._editor_state.editor_mode == "Mask"  # type: ignore[attr-defined, no-any-return]

    def _active_mask_roi_contains(self, source_x: float, source_y: float) -> bool:
        """Return whether a source point is inside the active mask ROI."""
        frame_index = self._current_frame_index()  # type: ignore[attr-defined]
        face_index = self._active_face_index()  # type: ignore[attr-defined]
        if frame_index < 0 or face_index is None:
            return False
        matrix = self._editable.mask_affine_matrix(  # type: ignore[attr-defined]
            frame_index, face_index, self.active_mask_type()
        )
        if matrix is None:
            return False
        try:
            import cv2
            import numpy as np

            point = np.asarray([[(float(source_x), float(source_y))]], dtype=np.float32)
            stored = cv2.transform(point, np.asarray(matrix, dtype=np.float32)[:2]).reshape(2)
            size = self._editable.MASK_STORED_SIZE  # type: ignore[attr-defined]
            return 0.0 <= stored[0] <= size - 1 and 0.0 <= stored[1] <= size - 1  # type: ignore[no-any-return]
        except Exception:  # pragma: no cover - defensive hover path
            return False

    def _should_render_mask(self) -> bool:
        """Show mask overlay when the Mask editor is active or annotation toggle on.

        Matches the legacy Tk behaviour where ``F10`` toggles the mask
        overlay layer in any editor mode and F5 implicitly shows it.
        """
        annotation_tokens = {
            part.strip()
            for part in str(self._editor_state.annotation_mode or "").split(",")  # type: ignore[attr-defined]
            if part.strip()
        }
        return self._editor_state.editor_mode == "Mask" or "Mask" in annotation_tokens  # type: ignore[attr-defined]

    def _on_mask_paint_requested(
        self, face_index: int, source_x: float, source_y: float, invert: bool, started: bool
    ) -> None:
        """Apply a Mask editor brush stamp at ``(source_x, source_y)`` (#101)."""
        if started:
            self._mask_last_source_point = None
        self.paint_mask_at(int(face_index), float(source_x), float(source_y), invert=bool(invert))

    def set_mask_draw_mode(self) -> bool:
        """Switch the Mask editor to Draw mode (action handler, B)."""
        self._editor_state.set("brush_mode", "draw")  # type: ignore[attr-defined]
        self.statusBar().showMessage("Mask: Draw mode", 3000)  # type: ignore[attr-defined]
        return True

    def set_mask_erase_mode(self) -> bool:
        """Switch the Mask editor to Erase mode (action handler, D)."""
        self._editor_state.set("brush_mode", "erase")  # type: ignore[attr-defined]
        self.statusBar().showMessage("Mask: Erase mode", 3000)  # type: ignore[attr-defined]
        return True

    def increase_brush_size(self) -> int:
        """Step brush size up by ``_BRUSH_STEP``, clamped at ``_BRUSH_MAX``."""
        new_size = min(self._BRUSH_MAX, self._editor_state.brush_size + self._BRUSH_STEP)  # type: ignore[attr-defined]
        self._editor_state.set("brush_size", new_size)  # type: ignore[attr-defined]
        self.statusBar().showMessage(f"Brush size: {new_size}", 2000)  # type: ignore[attr-defined]
        return new_size  # type: ignore[no-any-return]

    def decrease_brush_size(self) -> int:
        """Step brush size down by ``_BRUSH_STEP``, clamped at ``_BRUSH_MIN``."""
        new_size = max(self._BRUSH_MIN, self._editor_state.brush_size - self._BRUSH_STEP)  # type: ignore[attr-defined]
        self._editor_state.set("brush_size", new_size)  # type: ignore[attr-defined]
        self.statusBar().showMessage(f"Brush size: {new_size}", 2000)  # type: ignore[attr-defined]
        return new_size  # type: ignore[no-any-return]

    def active_mask_type(self) -> str:
        """Return the selected mask type, falling back to the default."""
        return self._editor_state.mask_type or self.DEFAULT_MASK_TYPE  # type: ignore[attr-defined]

    def _current_brush_spec(self) -> tuple[float, str, str, str] | None:
        """Return brush preview state for the frame view.

        Returns ``None`` when the Mask editor isn't active or no face is
        selected — the frame view uses ``None`` to mean "hide the preview".
        The radius mirrors :meth:`paint_mask_at`'s ``brush_size / 2`` math
        so the visible circle matches the actual stamp footprint.
        """
        if self._editor_state.editor_mode != "Mask":  # type: ignore[attr-defined]
            return None
        if self._active_face_index() is None:  # type: ignore[attr-defined]
            return None
        radius = max(1.0, float(self._editor_state.brush_size) / 2.0)  # type: ignore[attr-defined]
        mode = str(self._editor_state.brush_mode or "draw")  # type: ignore[attr-defined]
        shape = str(self._editor_state.brush_shape or "Circle")  # type: ignore[attr-defined]
        color = str(self._editor_state.brush_color or "#ffffff")  # type: ignore[attr-defined]
        return (radius, mode, shape, color)

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
        frame_index = self._current_frame_index()  # type: ignore[attr-defined]
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)  # type: ignore[attr-defined]
            return False
        mode = self._editor_state.brush_mode  # type: ignore[attr-defined]
        if invert:
            mode = "erase" if mode == "draw" else "draw"
        value = 255 if mode == "draw" else 0
        radius = max(1.0, self._editor_state.brush_size / 2.0)  # type: ignore[attr-defined]
        previous = (
            getattr(self, "_mask_last_source_point", None)
            if getattr(self._frame_view, "_mask_drag_active", False)  # type: ignore[attr-defined]
            else None
        )
        ok = self._editable.paint_mask_stroke(  # type: ignore[attr-defined]
            frame_index,
            int(face_index),
            self.active_mask_type(),
            float(source_x),
            float(source_y),
            radius,
            value,
            previous_source_x=None if previous is None else float(previous[0]),
            previous_source_y=None if previous is None else float(previous[1]),
            shape=str(self._editor_state.brush_shape or "Circle"),  # type: ignore[attr-defined]
        )
        if ok:
            self._mask_last_source_point = (float(source_x), float(source_y))  # type: ignore[assignment]
            self._editor_state.set("edited", True)  # type: ignore[attr-defined]
            self._refresh_face_grid()  # type: ignore[attr-defined]
            self._frame_view.update()  # type: ignore[attr-defined]
        return ok  # type: ignore[no-any-return]

    def set_mask_cursor_shape(self, shape: str) -> bool:
        """Set the Mask brush cursor shape."""
        value = "Rectangle" if str(shape) == "Rectangle" else "Circle"
        self._editor_state.set("brush_shape", value)  # type: ignore[attr-defined]
        self._refresh_mask_cursor_controls()
        self._frame_view.update()  # type: ignore[attr-defined]
        return True

    def set_mask_cursor_color(self, color: str) -> bool:
        """Set the Mask brush cursor colour."""
        qcolor = QColor(str(color))
        if not qcolor.isValid():
            return False
        self._editor_state.set("brush_color", qcolor.name())  # type: ignore[attr-defined]
        self._refresh_mask_cursor_controls()
        self._frame_view.update()  # type: ignore[attr-defined]
        return True

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
        self._mask_opacity_slider = QSlider(Qt.Horizontal)  # type: ignore[attr-defined]
        self._mask_opacity_slider.setObjectName("qt-manual-mask-opacity")
        self._mask_opacity_slider.setRange(0, 100)
        self._mask_opacity_slider.setValue(int(self._editor_state.mask_opacity))  # type: ignore[attr-defined]
        self._mask_opacity_slider.setFixedWidth(140)
        self._mask_opacity_slider.valueChanged.connect(self._on_mask_opacity_changed)
        layout.addWidget(self._mask_opacity_slider)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Cursor:"))
        self._mask_cursor_shape_combo = QComboBox()
        self._mask_cursor_shape_combo.setObjectName("qt-manual-mask-cursor-shape")
        self._mask_cursor_shape_combo.addItems(("Circle", "Rectangle"))
        self._mask_cursor_shape_combo.setCurrentText(str(self._editor_state.brush_shape))  # type: ignore[attr-defined]
        self._mask_cursor_shape_combo.currentTextChanged.connect(self._on_cursor_shape_changed)
        layout.addWidget(self._mask_cursor_shape_combo)
        self._mask_cursor_color_button = QToolButton()
        self._mask_cursor_color_button.setObjectName("qt-manual-mask-cursor-color")
        self._mask_cursor_color_button.setText("Color")
        self._mask_cursor_color_button.setToolTip("Brush cursor color")
        self._mask_cursor_color_button.clicked.connect(
            lambda _checked=False: self.set_mask_cursor_color(self._editor_state.brush_color)  # type: ignore[attr-defined]
        )
        layout.addWidget(self._mask_cursor_color_button)
        layout.addStretch(1)

        container.setVisible(self._editor_state.editor_mode == "Mask")  # type: ignore[attr-defined]
        return container

    def _refresh_mask_controls_visibility(self) -> None:
        """Show or hide the Mask control row to match the editor mode."""
        controls = getattr(self, "_mask_controls", None)
        if controls is None:
            return
        active = self._editor_state.editor_mode == "Mask"  # type: ignore[attr-defined]
        controls.setVisible(active)
        if active:
            self._refresh_mask_type_combo()
            self._refresh_mask_cursor_controls()

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
        frame_index = self._current_frame_index()  # type: ignore[attr-defined]
        face_index = int(self._editor_state.face_index)  # type: ignore[attr-defined]
        existing: tuple[str, ...] = ()
        if frame_index >= 0 and face_index >= 0:
            existing = self._editable.known_mask_types(frame_index, face_index)  # type: ignore[attr-defined]
        options = sorted({*self.MASK_DEFAULT_TYPES, *existing})
        if not options:
            return
        current = self._editor_state.mask_type or self.DEFAULT_MASK_TYPE  # type: ignore[attr-defined]
        if current not in options:
            current = self.DEFAULT_MASK_TYPE if self.DEFAULT_MASK_TYPE in options else options[0]
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(options)
            combo.setCurrentText(current)
        finally:
            combo.blockSignals(False)
        if self._editor_state.mask_type != current:  # type: ignore[attr-defined]
            self._editor_state.set("mask_type", current)  # type: ignore[attr-defined]

    def _on_mask_type_combo_changed(self, mask_type: str) -> None:
        """Persist the chosen mask type to editor state + refresh the overlay."""
        if not mask_type:
            return
        self._editor_state.set("mask_type", mask_type)  # type: ignore[attr-defined]
        self._frame_view.update()  # type: ignore[attr-defined]

    def _on_mask_opacity_changed(self, value: int) -> None:
        """Push the new opacity onto editor state + repaint the overlay."""
        self._editor_state.set("mask_opacity", int(value))  # type: ignore[attr-defined]
        self._frame_view.update()  # type: ignore[attr-defined]

    def _refresh_mask_cursor_controls(self) -> None:
        """Mirror cursor shape/color state into Mask controls."""
        combo = getattr(self, "_mask_cursor_shape_combo", None)
        if combo is not None and combo.currentText() != self._editor_state.brush_shape:  # type: ignore[attr-defined]
            combo.blockSignals(True)
            try:
                combo.setCurrentText(str(self._editor_state.brush_shape))  # type: ignore[attr-defined]
            finally:
                combo.blockSignals(False)
        button = getattr(self, "_mask_cursor_color_button", None)
        if button is not None:
            color = QColor(str(self._editor_state.brush_color))  # type: ignore[attr-defined]
            if not color.isValid():
                color = QColor("#ffffff")
            button.setStyleSheet(f"background-color: {color.name()};")

    def _on_cursor_shape_changed(self, shape: str) -> None:
        """Persist cursor shape combo changes."""
        self.set_mask_cursor_shape(shape)
