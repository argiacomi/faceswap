#!/usr/bin/env python3
"""Bounding Box editor controllers for the Qt Manual Tool."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt

from tools.manual.qt.frame_viewer.overlays import ManualFrameOverlay

EDITOR_MODE = "BoundingBox"


def is_active(editor_mode: str) -> bool:
    """Return whether ``editor_mode`` selects this editor."""
    return editor_mode == EDITOR_MODE


class BoundingBoxFrameEditorMixin:
    """Frame-view hit-test, cursor, add, move and resize behavior."""

    _ADD_MIN_DRAG = 4.0

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


class BoundingBoxWindowEditorMixin:
    """Root-window adapters for face selection, mutation and context menus."""

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

    def _delete_faces_at(self, selections: object) -> None:
        """Delete multiple grid-selected faces across frames as one edit."""
        pairs: tuple[tuple[int, int], ...] = tuple(
            (int(frame_index), int(face_index))
            for frame_index, face_index in selections  # type:ignore[misc]
        )
        if not pairs:
            return
        deleted = self._editable.delete_faces(pairs)
        if deleted <= 0:
            self.statusBar().showMessage("Could not delete selected faces", 3000)
            return
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        current_frame = self._current_frame_index()
        if any(frame_index == current_frame for frame_index, _face_index in pairs):
            new_count = self._editable.face_count(current_frame)
            if new_count == 0:
                self._editor_state.set("face_index", -1)
            else:
                self._editor_state.set(
                    "face_index",
                    min(max(0, int(self._editor_state.face_index)), new_count - 1),
                )
            self.refresh_faces()
            self._frame_view.update()
        self.mark_dirty(True)
        self._refresh_filter_results()
        self._refresh_face_grid()
        self.statusBar().showMessage(f"Deleted {deleted} selected face(s)", 5000)
