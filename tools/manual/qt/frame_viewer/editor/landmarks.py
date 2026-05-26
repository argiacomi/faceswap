#!/usr/bin/env python3
"""Landmark editor controllers for the Qt Manual Tool."""

from __future__ import annotations

import typing as T

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent

from tools.manual.qt.frame_viewer.overlays import ManualFrameOverlay

EDITOR_MODE = "Landmarks"


def is_active(editor_mode: str) -> bool:
    """Return whether ``editor_mode`` selects this editor."""
    return editor_mode == EDITOR_MODE


class LandmarkFrameEditorMixin:
    """Frame-view point, group and marquee behavior for Landmark mode."""

    _LANDMARK_HIT_TOLERANCE = 4.0
    _LANDMARK_GRAB_TOLERANCE = 8.0

    def _set_landmark_hover(self, landmark_index: int | None) -> None:
        """Persist Landmark editor hover state and notify the overlay."""
        if self._landmark_hover_index == landmark_index:
            return
        self._landmark_hover_index = landmark_index
        face_index = self._active_face_provider() if self._active_face_provider else None
        if self._landmark_hover_callback is not None:
            self._landmark_hover_callback(
                None if landmark_index is None else face_index,
                landmark_index,
            )
        self.update()

    def _update_landmark_hover(self, source_point: QPointF | None) -> bool:
        """Update hover target for Landmark mode and return whether a point is hit."""
        if (
            self._landmark_mode_provider is None
            or self._landmark_provider is None
            or self._active_face_provider is None
            or not self._landmark_mode_provider()
            or source_point is None
        ):
            self._set_landmark_hover(None)
            return False
        face_index = self._active_face_provider()
        landmarks = self._landmark_provider() if face_index is not None else None
        if not landmarks:
            self._set_landmark_hover(None)
            return False
        hit_index = ManualFrameOverlay.landmark_at(
            landmarks,
            (source_point.x(), source_point.y()),
            tolerance=self._LANDMARK_GRAB_TOLERANCE,
        )
        self._set_landmark_hover(hit_index)
        return hit_index is not None

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
        face_matches: list[tuple[int, tuple[int, ...]]] = []
        if self._landmark_faces_provider is not None:
            for candidate_face_index, landmarks in self._landmark_faces_provider():
                indices = ManualFrameOverlay.landmarks_in_rect(landmarks, current)
                if indices:
                    face_matches.append((int(candidate_face_index), indices))
        else:
            landmarks = self._landmark_provider() or ()
            indices = ManualFrameOverlay.landmarks_in_rect(landmarks, current)
            if indices:
                face_matches.append((int(face_index), indices))
        if len(face_matches) != 1:
            self.landmarks_select_requested.emit(int(face_index), ())
            return
        selected_face, indices = face_matches[0]
        self.landmarks_select_requested.emit(selected_face, indices)

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
            tolerance=self._LANDMARK_GRAB_TOLERANCE,
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


class LandmarkWindowEditorMixin:
    """Root-window adapters for Landmark editor model updates."""

    def _is_landmark_mode_active(self) -> bool:
        """Return whether the Landmark editor (F4) is active."""
        return self._editor_state.editor_mode == "Landmarks"

    def _active_face_landmarks(self) -> tuple[tuple[float, float], ...] | None:
        """Return the active face's landmark sequence (Landmark editor seam)."""
        frame_index = self._current_frame_index()
        face_index = self._editor_state.face_index
        if frame_index < 0 or face_index < 0:
            return None
        faces = self._editable.faces(frame_index)
        if face_index >= len(faces):
            return None
        return faces[face_index].landmarks

    def _frame_landmark_faces(self) -> tuple[tuple[int, tuple[tuple[float, float], ...]], ...]:
        """Return landmark sets for all faces on the current frame."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return ()
        return tuple(
            (face.face_index, face.landmarks)
            for face in self._editable.faces(frame_index)
            if face.landmarks
        )

    def _on_landmark_move_requested(
        self, face_index: int, landmark_index: int, new_x: float, new_y: float
    ) -> None:
        """Apply a single-point landmark edit and refresh the overlay."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.update_landmark(
            frame_index, int(face_index), int(landmark_index), float(new_x), float(new_y)
        ):
            self.statusBar().showMessage("Landmark edit failed", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()

    def _on_landmarks_move_requested(
        self,
        face_index: int,
        indices: T.Any,
        dx: float,
        dy: float,
    ) -> None:
        """Apply a group-move edit across the marquee-selected landmark set."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        idx_tuple = tuple(indices) if not isinstance(indices, tuple) else indices
        if not self._editable.move_landmarks(
            frame_index, int(face_index), idx_tuple, float(dx), float(dy)
        ):
            self.statusBar().showMessage("Landmark group move failed", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()

    def _on_landmarks_select_requested(self, face_index: int, indices: T.Any) -> None:
        """Update the overlay's selection set from a marquee release."""
        if face_index != self._editor_state.face_index:
            return
        idx_tuple = tuple(indices) if not isinstance(indices, tuple) else indices
        self._overlay.set_selected_landmarks(idx_tuple)
        if not idx_tuple:
            self.statusBar().showMessage("Landmark selection cleared", 2000)
        else:
            self.statusBar().showMessage(f"Selected {len(idx_tuple)} landmark(s)", 2000)
        self._frame_view.update()
