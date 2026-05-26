#!/usr/bin/env python3
"""Bounding Box editor controllers for the Qt Manual Tool."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt

from tools.manual.qt.frame_viewer.overlays import ManualFrameOverlay
from tools.manual.session import EditableFace

EDITOR_MODE = "BoundingBox"


def is_active(editor_mode: str) -> bool:
    """Return whether ``editor_mode`` selects this editor."""
    return editor_mode == EDITOR_MODE


class BoundingBoxFrameEditorMixin:
    """Frame-view hit-test, cursor, add, move and resize behavior."""

    _ADD_MIN_DRAG = 4.0
    _MIN_BBOX_SIZE = 20.0
    _HANDLE_GRAB_TOLERANCE = 5.5

    def _update_hover_cursor(self, position: QPointF) -> None:
        """Cursor priority: resize handle > bbox body > pannable > default."""
        # Always refresh the Mask brush preview before falling through to
        # the cursor-shape ladder.  ``_update_brush_preview`` schedules a
        # repaint when the preview state changes.
        self._update_brush_preview(position)
        if self._source.isNull():
            self.setCursor(Qt.ArrowCursor)
            self._hovered_face_index = None
            self._set_landmark_hover(None)
            return
        source_point = self._widget_to_source(position)
        if self._face_hit_provider is not None and source_point is not None:
            self._hovered_face_index = self._face_hit_provider(source_point.x(), source_point.y())
        else:
            self._hovered_face_index = None
        if self._update_landmark_hover(source_point):
            self.setCursor(Qt.BlankCursor)
            return
        if self._mask_mode_provider is not None and self._mask_mode_provider():
            if self._brush_preview_source_point is not None:
                self.setCursor(Qt.BlankCursor)
                return
            self.setCursor(Qt.ArrowCursor)
            return
        in_bbox_mode = bool(self._add_mode_provider and self._add_mode_provider())
        in_extract_mode = bool(self._extract_mode_provider and self._extract_mode_provider())
        if in_bbox_mode:
            hit = self._bbox_hit_at(position)
            if hit is not None:
                face_index, _source_bbox, _widget_bbox, handle = hit
                self._hovered_face_index = int(face_index)
                if handle is not None:
                    self.setCursor(self._cursor_for_handle(handle))
                else:
                    self.setCursor(Qt.SizeAllCursor)
                return
        # Hover an active-face resize handle?
        widget_bbox = self._active_bbox_widget_rect()
        if widget_bbox is not None:
            handle = ManualFrameOverlay.handle_at(
                widget_bbox,
                position,
                tolerance=self._HANDLE_GRAB_TOLERANCE,
            )
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
        """Try to start a move/resize drag on a visible bbox. Returns True on hit."""
        if self._active_face_provider is None:
            return False
        source_point = self._widget_to_source(position)
        if source_point is None:
            return False
        in_bbox_mode = bool(self._add_mode_provider and self._add_mode_provider())
        if in_bbox_mode:
            hit = self._bbox_hit_at(position)
            if hit is None:
                return False
            face_index, source_bbox, widget_bbox, handle = hit
            if self._face_select_callback is not None:
                self._face_select_callback(int(face_index))
        else:
            face_index = self._active_face_provider()
            if face_index is None:
                return False
            widget_bbox = self._active_bbox_widget_rect()
            source_bbox = self._active_bbox_source_rect()
            if widget_bbox is None or source_bbox is None:
                return False
            handle = ManualFrameOverlay.handle_at(
                widget_bbox,
                position,
                tolerance=self._HANDLE_GRAB_TOLERANCE,
            )
        if handle is not None:
            self._edit_drag_mode = "resize"
            self._edit_drag_handle = handle
        elif widget_bbox.contains(position):
            self._edit_drag_mode = "move"
            self._edit_drag_handle = None
        else:
            return False
        self._edit_drag_face_index = int(face_index)
        self._edit_drag_source_anchor = source_point
        self._edit_drag_original_bbox = QRectF(source_bbox)
        self._edit_drag_current_bbox = QRectF(source_bbox)
        return True

    def _resize_bbox(self, original: QRectF, handle: str, dx: float, dy: float) -> QRectF:
        """Compute the new bbox for a resize drag in source coordinates.

        Legacy drags only one corner, clamps it 20 display pixels away from
        the opposite corner, and never flips the rectangle through itself.
        """
        min_w, min_h = self._display_min_bbox_size_source()
        left = original.left()
        top = original.top()
        right = original.right()
        bottom = original.bottom()
        if "w" in handle:
            left = min(right - min_w, left + dx)
        if "e" in handle:
            right = max(left + min_w, right + dx)
        if "n" in handle:
            top = min(bottom - min_h, top + dy)
        if "s" in handle:
            bottom = max(top + min_h, bottom + dy)
        return QRectF(left, top, right - left, bottom - top)

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
            rect = self._default_add_rect(anchor)
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
        """Create a default bbox immediately, then let the same gesture move it."""
        if self._add_mode_provider is None or not self._add_mode_provider():
            return False
        if source_point is None or not self._target_rect().contains(position):
            return False
        rect = self._default_add_rect(source_point)
        if self._face_add_callback is not None:
            face_index = self._face_add_callback(rect)
            if face_index is None:
                return False
            self._edit_drag_mode = "move"
            self._edit_drag_handle = None
            self._edit_drag_face_index = int(face_index)
            self._edit_drag_source_anchor = QPointF(source_point)
            self._edit_drag_original_bbox = QRectF(rect)
            self._edit_drag_current_bbox = QRectF(rect)
            self.setCursor(Qt.SizeAllCursor)
            return True
        self.face_add_requested.emit(rect)
        if self._begin_edit_drag(position):
            self.setCursor(Qt.SizeAllCursor)
            return True
        self.setCursor(Qt.SizeAllCursor)
        return True

    def _default_add_rect(self, anchor: QPointF) -> QRectF:
        """Return the legacy default add bbox centred on ``anchor``."""
        src_w, src_h = self.source_size
        default_size = max(self._MIN_BBOX_SIZE, min(src_w, src_h) / 4.0)
        half = default_size / 2.0
        return QRectF(anchor.x() - half, anchor.y() - half, default_size, default_size)

    def _display_min_bbox_size_source(self) -> tuple[float, float]:
        """Return the legacy 20-display-pixel resize minimum in source pixels."""
        target = self._target_rect()
        src_w, src_h = self.source_size
        if target.width() <= 0.0 or target.height() <= 0.0 or src_w <= 0 or src_h <= 0:
            return (self._MIN_BBOX_SIZE, self._MIN_BBOX_SIZE)
        return (
            max(1.0, self._MIN_BBOX_SIZE * src_w / target.width()),
            max(1.0, self._MIN_BBOX_SIZE * src_h / target.height()),
        )

    def _visible_bbox_records(self) -> tuple[tuple[int, QRectF, QRectF], ...]:
        """Return visible ``(face_index, source_bbox, widget_bbox)`` records."""
        records: list[tuple[int, QRectF, QRectF]] = []
        if self._face_bboxes_provider is not None:
            for face_index, bbox in self._face_bboxes_provider():
                source = QRectF(bbox)
                records.append((int(face_index), source, self._source_bbox_to_widget_rect(source)))
            return tuple(records)
        face_index = self._active_face_provider() if self._active_face_provider else None
        source = self._active_bbox_source_rect()
        widget = self._active_bbox_widget_rect()
        if face_index is not None and source is not None and widget is not None:
            records.append((int(face_index), QRectF(source), QRectF(widget)))
        return tuple(records)

    def _bbox_hit_at(self, position: QPointF) -> tuple[int, QRectF, QRectF, str | None] | None:
        """Hit-test all visible bbox handles first, then bbox bodies."""
        records = self._visible_bbox_records()
        for face_index, source_bbox, widget_bbox in reversed(records):
            handle = ManualFrameOverlay.handle_at(
                widget_bbox,
                position,
                tolerance=self._HANDLE_GRAB_TOLERANCE,
            )
            if handle is not None:
                return (face_index, source_bbox, widget_bbox, handle)
        for face_index, source_bbox, widget_bbox in reversed(records):
            if widget_bbox.contains(position):
                return (face_index, source_bbox, widget_bbox, None)
        return None

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
        new_bbox = self._rounded_bbox_tuple(bbox)
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

    def _maybe_run_aligner(self, face_index: int, *, aligner_name: str | None = None) -> None:
        """Run the aligner after a bbox edit when ``aligner_auto_run`` is on."""
        if not self._editor_state.aligner_auto_run:
            return
        if self._editor_state.editor_mode != "BoundingBox":
            return
        self.rerun_aligner_for_face(int(face_index), aligner_name=aligner_name)

    def _visible_face_bboxes(self) -> tuple[tuple[int, QRectF], ...]:
        """Return every current-frame face bbox for frame-view hit testing."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return ()
        return tuple(
            (int(face.face_index), QRectF(*face.bbox))
            for face in self._editable.faces(frame_index)
        )

    def _select_frame_face(self, face_index: int) -> None:
        """Select a face from a direct frame-view hit test."""
        self._face_panel.select_face(int(face_index))
        self._editor_state.set("face_index", int(face_index))
        self._refresh_face_grid_active()
        self._sync_actions()

    def _face_at_source_point(self, sx: float, sy: float) -> int | None:
        """Hit-test the editable model at a source-coordinate point."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return None
        return self._editable.hit_test(frame_index, sx, sy)

    def _on_face_add_requested(self, bbox: QRectF) -> None:
        """Create a new face from a pointer-driven add gesture."""
        new_index = self.add_face_at_center(
            (bbox.x(), bbox.y(), bbox.width(), bbox.height()),
            aligner_name="cv2-dnn",
        )
        if new_index is None:
            self.statusBar().showMessage("Could not add face", 5000)

    def _on_live_face_add_requested(self, bbox: QRectF) -> int | None:
        """Create a BBox face immediately on mouse press for live dragging."""
        frame_index = self._current_frame_index()
        if frame_index >= 0:
            self._live_bbox_add_undo_start = self._editable.undo_depth
            self._live_bbox_add_previous_faces = self._editable.faces(frame_index)
        new_index = self.add_face_at_center(
            (bbox.x(), bbox.y(), bbox.width(), bbox.height()),
            aligner_name="cv2-dnn",
        )
        if new_index is not None:
            self._live_bbox_added_face = int(new_index)
            self._live_bbox_original_face = None
        return new_index

    def _on_live_face_bbox_requested(self, face_index: int, bbox: QRectF) -> None:
        """Apply a live BBox move/resize and rerun the aligner before release."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return
        if (
            getattr(self, "_live_bbox_added_face", None) != int(face_index)
            and getattr(self, "_live_bbox_original_face", None) is None
        ):
            faces = self._editable.faces(frame_index)
            if 0 <= int(face_index) < len(faces):
                self._live_bbox_original_face = faces[int(face_index)]
        if not self._editable.update_face_bbox_live(
            frame_index,
            int(face_index),
            self._rounded_bbox_tuple(bbox),
        ):
            return
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        if self._editor_state.aligner_auto_run and self._editor_state.editor_mode == "BoundingBox":
            self.rerun_aligner_for_face(int(face_index), live=True)
        self.refresh_faces()
        self._face_panel.select_face(int(face_index))
        self._editor_state.set("face_index", int(face_index))
        self._refresh_face_grid()
        self._frame_view.update()

    def _on_live_face_bbox_committed(self, face_index: int) -> None:
        """Coalesce a live BBox drag into one undo entry."""
        frame_index = self._current_frame_index()
        added_face = getattr(self, "_live_bbox_added_face", None)
        original = getattr(self, "_live_bbox_original_face", None)
        add_undo_start = getattr(self, "_live_bbox_add_undo_start", None)
        add_previous_faces = getattr(self, "_live_bbox_add_previous_faces", ())
        self._live_bbox_added_face = None
        self._live_bbox_original_face = None
        self._live_bbox_add_undo_start = None
        self._live_bbox_add_previous_faces = ()
        if frame_index < 0:
            return
        if added_face == int(face_index):
            if add_undo_start is not None:
                self._editable.replace_undo_since(
                    int(add_undo_start),
                    frame_index,
                    add_previous_faces,
                )
                self._sync_actions()
            self.refresh_faces()
            self._refresh_face_grid()
            self._frame_view.update()
            return
        if original is None:
            return
        if isinstance(original, EditableFace):
            self._editable.record_face_update(frame_index, int(face_index), original)
            self._sync_actions()
            self.refresh_faces()
            self._refresh_face_grid()
            self._frame_view.update()

    @staticmethod
    def _rounded_bbox_tuple(bbox: QRectF) -> tuple[float, float, float, float]:
        """Return a legacy-style integer source bbox tuple."""
        return (
            float(int(round(bbox.x()))),
            float(int(round(bbox.y()))),
            float(max(1, int(round(bbox.width())))),
            float(max(1, int(round(bbox.height())))),
        )

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
        self._refresh_filter_results(preserve_current=True, navigate_on_filter_miss=False)
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
        self._refresh_filter_results(preserve_current=True, navigate_on_filter_miss=False)
        self._refresh_face_grid()
        self.statusBar().showMessage(f"Deleted {deleted} selected face(s)", 5000)
