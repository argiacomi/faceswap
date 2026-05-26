#!/usr/bin/env python3
"""Qt Manual Tool face-browser wiring helpers."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QListWidgetItem

from tools.manual.session import FaceThumbnail

from .face_viewer.viewport import _FACE_GRID_SIZES, FaceGridEntry


class FaceBrowserMixin:
    """Own root-window glue for face panels, face-grid entries and selection."""

    def refresh_faces(self) -> None:
        """Rebuild the face panel from the editable model and persisted thumbs."""
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
        if index != self._current_frame_index():
            self._magnify_restore_state = None
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

    def _on_face_selected(self, face_index: int) -> None:
        """Propagate active face selection to the shared editor state."""
        self._editor_state.set("face_index", face_index if face_index >= 0 else -1)
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
        """Return the currently selected face index or ``None``."""
        index = self._editor_state.face_index
        if not isinstance(index, int) or index < 0:
            return None
        return index

    def _current_frame_index(self) -> int:
        """Return the sorted-frame index of the active frame, or -1 if none."""
        return -1 if self._current_frame is None else self._current_frame.index

    def _on_editable_changed(self, frame_index: int) -> None:
        """React to any change in the editable alignment model."""
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


__all__ = ["FaceBrowserMixin"]
