#!/usr/bin/env python3
"""Qt Manual Tool face-browser wiring helpers."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QListWidgetItem

from tools.manual.session import EditableFace, FaceThumbnail

from .face_viewer.viewport import _FACE_GRID_SIZES, FaceGridEntry


class FaceBrowserMixin:
    """Own root-window glue for face panels, face-grid entries and selection."""

    def refresh_faces(self) -> None:
        """Rebuild the face panel from the editable model and persisted thumbs."""
        if self._current_frame is None:
            self._face_panel.set_faces(())  # type: ignore[attr-defined]
            return
        frame_index = self._current_frame.index
        entries = self._face_thumbnail_entries_for_frame(frame_index)
        if not entries:
            self._face_panel.set_faces(())  # type: ignore[attr-defined]
            return
        self._face_panel.set_faces(entries)  # type: ignore[attr-defined]

    def _face_thumbnail_entries_for_frame(self, frame_index: int) -> tuple[FaceThumbnail, ...]:
        """Return face thumbnail entries for one editable frame index."""
        editable_faces = self._editable.faces(frame_index)  # type: ignore[attr-defined]
        if not editable_faces:
            return ()
        frame_name = self._frame_name_for_index(frame_index)  # type: ignore[attr-defined]
        if frame_name is None:
            frame_name = (
                self._current_frame.name
                if self._current_frame is not None and self._current_frame.index == frame_index
                else f"frame_{frame_index:06d}"
            )
        persisted = {
            entry.face_index: entry
            for entry in self._alignments_handle.faces_for_frame_name(  # type: ignore[attr-defined]
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
        misaligned_filter = self._editor_state.filter_mode == "Misaligned Faces"  # type: ignore[attr-defined]
        misaligned_threshold = float(self._editor_state.filter_distance) / 100.0  # type: ignore[attr-defined]
        for frame_index in self.filtered_frame_indices():  # type: ignore[attr-defined]
            frame_name = self._frame_name_for_index(frame_index) or f"frame_{frame_index:06d}"  # type: ignore[attr-defined]
            thumbs = {
                thumb.face_index: thumb
                for thumb in self._face_thumbnail_entries_for_frame(frame_index)
            }
            for face in self._editable.faces(frame_index):  # type: ignore[attr-defined]
                is_misaligned = self._face_grid_misaligned_state(
                    face,
                    enabled=misaligned_filter,
                    threshold=misaligned_threshold,
                )
                entries.append(
                    FaceGridEntry(
                        frame_index=frame_index,
                        frame_name=frame_name,
                        face_index=int(face.face_index),
                        thumbnail=thumbs.get(int(face.face_index)),
                        bbox=face.bbox,
                        landmarks=face.landmarks,
                        is_misaligned=is_misaligned,
                    )
                )
        return tuple(entries)

    @staticmethod
    def _face_grid_misaligned_state(
        face: EditableFace,
        *,
        enabled: bool,
        threshold: float,
    ) -> bool | None:
        """Return per-face misalignment state when the Misaligned Faces filter is active."""
        if not enabled:
            return None
        from tools.manual.frame_filter import face_average_distance

        distance = face_average_distance(face)
        return bool(distance is not None and distance > threshold)

    def _refresh_face_grid(self) -> None:
        """Rebuild the cross-frame face grid from filters, state and annotations."""
        panel = getattr(self, "_face_grid_panel", None)
        if panel is None:
            return
        annotation_tokens = {
            part.strip()
            for part in str(self._editor_state.annotation_mode or "").split(",")  # type: ignore[attr-defined]
            if part.strip()
        }
        panel.set_overlay_state(
            show_mesh=bool(self._editor_state.face_grid_mesh_visible)  # type: ignore[attr-defined]
            or "Mesh" in annotation_tokens,
            show_mask=bool(self._editor_state.face_grid_mask_visible)  # type: ignore[attr-defined]
            or self._should_render_mask(),  # type: ignore[attr-defined]
            mask_type=self.active_mask_type(),  # type: ignore[attr-defined]
            mask_opacity=int(self._editor_state.mask_opacity),  # type: ignore[attr-defined]
        )
        panel.set_entries(self._face_grid_entries())
        self._refresh_face_grid_active()

    def _on_face_grid_overlay_toggled(self, _value: object) -> None:
        """Mirror independent mini-rail overlay toggles into buttons and thumbnails."""
        for button, enabled in (
            (
                getattr(self, "_face_grid_mesh_button", None),
                self._editor_state.face_grid_mesh_visible,  # type: ignore[attr-defined]
            ),
            (
                getattr(self, "_face_grid_mask_button", None),
                self._editor_state.face_grid_mask_visible,  # type: ignore[attr-defined]
            ),
        ):
            if button is None or button.isChecked() == bool(enabled):
                continue
            button.blockSignals(True)
            try:
                button.setChecked(bool(enabled))
            finally:
                button.blockSignals(False)
        mesh_action = self._actions.get("cycle_annotation")  # type: ignore[attr-defined]
        if mesh_action is not None:
            mesh_action.setCheckable(True)
            mesh_action.setChecked(bool(self._editor_state.face_grid_mesh_visible))  # type: ignore[attr-defined]
        mask_action = self._actions.get("toggle_mask_annotation")  # type: ignore[attr-defined]
        if mask_action is not None:
            mask_action.setCheckable(True)
            mask_action.setChecked(bool(self._editor_state.face_grid_mask_visible))  # type: ignore[attr-defined]
        self._refresh_face_grid()

    def _refresh_face_grid_active(self) -> None:
        """Update active frame/face styling without rebuilding thumbnail icons."""
        panel = getattr(self, "_face_grid_panel", None)
        if panel is None:
            return
        panel.set_active(self._current_frame_index(), int(self._editor_state.face_index))  # type: ignore[attr-defined]

    def _on_face_grid_activated(self, frame_index: int, face_index: int) -> None:
        """Navigate to the clicked grid face and make it the active face."""
        if frame_index < 0:
            return
        self._stop_playback()  # type: ignore[attr-defined]
        if self._thumbnail_panel.currentRow() != frame_index:  # type: ignore[attr-defined]
            self._thumbnail_panel.setCurrentRow(frame_index)  # type: ignore[attr-defined]
        self._editor_state.set("face_index", int(face_index))  # type: ignore[attr-defined]
        self._face_panel.select_face(int(face_index))  # type: ignore[attr-defined]
        self._face_grid_panel.set_active(int(frame_index), int(face_index))  # type: ignore[attr-defined]
        self._frame_view.update()  # type: ignore[attr-defined]
        self._sync_actions()  # type: ignore[attr-defined]

    def _on_face_grid_hovered(self, frame_index: int, face_index: int) -> None:
        """Surface simple hover feedback for cross-frame thumbnails."""
        self.statusBar().showMessage(  # type: ignore[attr-defined]
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
        self._show_face_context_menu(face_index, global_pos, frame_index=frame_index)  # type: ignore[attr-defined]

    def _on_face_grid_size_changed(self, size_name: str) -> None:
        """Persist a user-selected face-grid size."""
        if size_name not in _FACE_GRID_SIZES:
            return
        self._editor_state.set("faces_size", size_name)  # type: ignore[attr-defined]

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
        self._face_grid_panel.set_face_size(name)  # type: ignore[attr-defined]
        self._refresh_face_grid()

    def _thumbnail_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Display the selected source frame."""
        if current is None:
            return
        index = current.data(Qt.UserRole)  # type: ignore[attr-defined]
        if not isinstance(index, int):
            return
        if index != self._current_frame_index():
            self._magnify_restore_state = None
        if self._session.has_images:  # type: ignore[attr-defined]
            if index >= len(self._session.frame_list):  # type: ignore[attr-defined]
                return
            frame = self._session.frame_list[index]  # type: ignore[attr-defined]
            self._current_frame = frame
            self._editor_state.set("frame_index", frame.index)  # type: ignore[attr-defined]
            self.frame_changed.emit(frame.index)  # type: ignore[attr-defined]
            if self._frame_view.load_frame(frame):  # type: ignore[attr-defined]
                self._status_label.setText(  # type: ignore[attr-defined]
                    f"Frame {frame.index + 1} of {self._session.frame_count}: {frame.name}"  # type: ignore[attr-defined]
                )
            self.refresh_faces()
        elif self._video_frames:  # type: ignore[attr-defined]
            if index >= len(self._video_frames):  # type: ignore[attr-defined]
                return
            frame = self._video_frames[index]  # type: ignore[attr-defined]
            self._current_frame = frame
            self._editor_state.set("frame_index", frame.index)  # type: ignore[attr-defined]
            self.frame_changed.emit(frame.index)  # type: ignore[attr-defined]
            if self._video_provider is not None:  # type: ignore[attr-defined]
                self._video_provider.request_frame(frame.index)  # type: ignore[attr-defined]
                self._status_label.setText(  # type: ignore[attr-defined]
                    f"Loading frame {frame.index + 1} of {len(self._video_frames)}"  # type: ignore[attr-defined]
                )
            self.refresh_faces()

    def _on_face_selected(self, face_index: int) -> None:
        """Propagate active face selection to the shared editor state."""
        self._editor_state.set("face_index", face_index if face_index >= 0 else -1)  # type: ignore[attr-defined]
        self._sync_actions()  # type: ignore[attr-defined]

    def _on_frame_clicked(self, point: QPointF) -> None:
        """Hit-test the editable model at the clicked source-image point."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return
        face_index = self._editable.hit_test(frame_index, point.x(), point.y())  # type: ignore[attr-defined]
        if face_index is None:
            return
        self._editor_state.set("face_index", face_index)  # type: ignore[attr-defined]
        self._face_panel.select_face(face_index)  # type: ignore[attr-defined]
        self._refresh_face_grid_active()

    def _active_face_index(self) -> int | None:
        """Return the currently selected face index or ``None``."""
        index = self._editor_state.face_index  # type: ignore[attr-defined]
        if not isinstance(index, int) or index < 0:
            return None
        return index

    def _current_frame_index(self) -> int:
        """Return the sorted-frame index of the active frame, or -1 if none."""
        return -1 if self._current_frame is None else self._current_frame.index

    def _on_editable_changed(self, frame_index: int) -> None:
        """React to any change in the editable alignment model."""
        self.mark_dirty(self._editable.can_undo)  # type: ignore[attr-defined]
        self._refresh_filter_results(  # type: ignore[attr-defined]
            preserve_current=True,
            navigate_on_filter_miss=False,
        )
        if frame_index != self._current_frame_index():
            return
        self.refresh_faces()
        self._frame_view.update()  # type: ignore[attr-defined]
        face_count = self._editable.face_count(frame_index)  # type: ignore[attr-defined]
        active = self._editor_state.face_index  # type: ignore[attr-defined]
        if face_count > 0 and active < 0:
            self._editor_state.set("face_index", 0)  # type: ignore[attr-defined]
        elif face_count == 0 and active >= 0:
            self._editor_state.set("face_index", -1)  # type: ignore[attr-defined]
        self._sync_actions()  # type: ignore[attr-defined]

    def delete_active_face(self) -> None:
        """Delete the active face from the editable alignment model."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return
        hovered = getattr(self._frame_view, "hovered_face_index", None)  # type: ignore[attr-defined]
        if self._editor_state.editor_mode in {"BoundingBox", "ExtractBox"} and hovered is not None:  # type: ignore[attr-defined]
            face_index = int(hovered)
        else:
            face_index = self._editor_state.face_index  # type: ignore[attr-defined]
        if not self._editable.delete_face(frame_index, face_index):  # type: ignore[attr-defined]
            self.statusBar().showMessage("No face selected to delete", 5000)  # type: ignore[attr-defined]
            return
        self._editor_state.set("face_count_changed", True)  # type: ignore[attr-defined]
        self._editor_state.set("edited", True)  # type: ignore[attr-defined]
        self.mark_dirty(True)  # type: ignore[attr-defined]
        new_count = self._editable.face_count(frame_index)  # type: ignore[attr-defined]
        if new_count == 0:
            self._editor_state.set("face_index", -1)  # type: ignore[attr-defined]
        else:
            self._editor_state.set("face_index", min(face_index, new_count - 1))  # type: ignore[attr-defined]
        self.statusBar().showMessage(f"Deleted face index {face_index}", 5000)  # type: ignore[attr-defined]

    def add_face_at_center(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        aligner_name: str | None = None,
    ) -> int | None:
        """Add a new face. Defaults to a centered fixed-size bbox."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return None
        if bbox is None:
            src_w, src_h = self._frame_view.source_size  # type: ignore[attr-defined]
            if src_w <= 0 or src_h <= 0:
                self.statusBar().showMessage("Cannot add face: frame not loaded", 5000)  # type: ignore[attr-defined]
                return None
            size = float(max(20.0, min(src_w, src_h) / 4.0))
            bbox = (src_w / 2 - size / 2, src_h / 2 - size / 2, size, size)
        try:
            new_index = self._editable.add_face(frame_index, bbox)  # type: ignore[attr-defined]
        except ValueError as err:
            self.statusBar().showMessage(str(err), 5000)  # type: ignore[attr-defined]
            return None
        self._editor_state.set("face_count_changed", True)  # type: ignore[attr-defined]
        self._editor_state.set("edited", True)  # type: ignore[attr-defined]
        self.mark_dirty(True)  # type: ignore[attr-defined]
        self._editor_state.set("face_index", new_index)  # type: ignore[attr-defined]
        self.statusBar().showMessage(f"Added face index {new_index}", 5000)  # type: ignore[attr-defined]
        self._maybe_run_aligner(int(new_index), aligner_name=aligner_name)  # type: ignore[attr-defined]
        return new_index  # type: ignore[no-any-return]

    def nudge_active_face(self, dx: float, dy: float) -> bool:
        """Translate the active face bbox and landmarks by ``(dx, dy)`` pixels."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)  # type: ignore[attr-defined]
            return False
        face_index = self._editor_state.face_index  # type: ignore[attr-defined]
        if face_index < 0:
            self.statusBar().showMessage("No active face to nudge", 3000)  # type: ignore[attr-defined]
            return False
        if not self._editable.move_face(frame_index, face_index, dx, dy):  # type: ignore[attr-defined]
            self.statusBar().showMessage("Nudge failed (no active face)", 3000)  # type: ignore[attr-defined]
            return False
        self._editor_state.set("edited", True)  # type: ignore[attr-defined]
        self.mark_dirty(True)  # type: ignore[attr-defined]
        self.refresh_faces()
        self._frame_view.update()  # type: ignore[attr-defined]
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
        """Nudge active face up by 10 source pixels."""
        return self.nudge_active_face(0.0, -10.0)

    def nudge_down_fast(self) -> bool:
        """Nudge active face down by 10 source pixels."""
        return self.nudge_active_face(0.0, 10.0)

    def nudge_left_fast(self) -> bool:
        """Nudge active face left by 10 source pixels."""
        return self.nudge_active_face(-10.0, 0.0)

    def nudge_right_fast(self) -> bool:
        """Nudge active face right by 10 source pixels."""
        return self.nudge_active_face(10.0, 0.0)


__all__ = ["FaceBrowserMixin"]
