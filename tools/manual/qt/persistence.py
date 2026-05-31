#!/usr/bin/env python3
"""Qt Manual Tool save, extract and alignment persistence helpers."""

from __future__ import annotations

import contextlib
import logging
import os
import typing as T

from PySide6.QtWidgets import QFileDialog, QMessageBox, QToolButton

from .workers import ManualExtractFacesWorker, ManualSaveWorker

logger = logging.getLogger(__name__)


class PersistenceMixin:
    """Own save, extract, revert and copy orchestration for editable alignments."""

    def revert_current_frame(self) -> None:
        """Restore the current frame from saved alignments."""
        frame_index = self._current_frame_index()  # type: ignore[attr-defined]
        if frame_index < 0:
            self.statusBar().showMessage("Nothing to revert", 3000)  # type: ignore[attr-defined]
            return
        reverted_to_saved = self._editable.restore_frame_from_handle(  # type: ignore[attr-defined]
            self._alignments_handle,  # type: ignore[attr-defined]
            frame_index=frame_index,
            frame_name=self._frame_name_for_index(frame_index),
        )
        self.mark_dirty(self._editable.can_undo)  # type: ignore[attr-defined]
        if not self._editable.can_undo:  # type: ignore[attr-defined]
            self._editor_state.set("edited", False)  # type: ignore[attr-defined]
        self.refresh_faces()  # type: ignore[attr-defined]
        self._frame_view.update()  # type: ignore[attr-defined]
        self._refresh_filter_results(preserve_current=True, navigate_on_filter_miss=False)  # type: ignore[attr-defined]
        self._refresh_face_grid()  # type: ignore[attr-defined]
        face_count = self._editable.face_count(frame_index)  # type: ignore[attr-defined]
        if face_count == 0:
            self._editor_state.set("face_index", -1)  # type: ignore[attr-defined]
        elif self._editor_state.face_index < 0 or self._editor_state.face_index >= face_count:  # type: ignore[attr-defined]
            self._editor_state.set("face_index", 0)  # type: ignore[attr-defined]
        if reverted_to_saved:
            self.statusBar().showMessage("Reverted current frame to saved alignments", 5000)  # type: ignore[attr-defined]
        else:
            self.statusBar().showMessage("Nothing to revert in this frame", 3000)  # type: ignore[attr-defined]

    def copy_prev_face(self) -> bool:
        """Copy faces from the nearest previous frame that has faces."""
        current = self._current_frame_index()  # type: ignore[attr-defined]
        source = next(
            (index for index in range(current - 1, -1, -1) if self._editable.face_count(index)),  # type: ignore[attr-defined]
            None,
        )
        if source is None:
            self.statusBar().showMessage("No faces in previous frames to copy", 5000)  # type: ignore[attr-defined]
            return False
        return self._copy_faces_from(source, "previous")

    def copy_next_face(self) -> bool:
        """Copy faces from the nearest next frame that has faces."""
        current = self._current_frame_index()  # type: ignore[attr-defined]
        max_index = max(self._editable.known_frame_indices() or (self._session.frame_count - 1,))  # type: ignore[attr-defined]
        source = next(
            (
                index
                for index in range(current + 1, max_index + 1)
                if self._editable.face_count(index)  # type: ignore[attr-defined]
            ),
            None,
        )
        if source is None:
            self.statusBar().showMessage("No faces in next frames to copy", 5000)  # type: ignore[attr-defined]
            return False
        return self._copy_faces_from(source, "next")

    def _copy_faces_from(self, source_frame_index: int, direction: str) -> bool:
        """Replace current frame faces with the source frame's editable faces."""
        current_index = self._current_frame_index()  # type: ignore[attr-defined]
        if current_index < 0:
            return False
        source_faces = self._editable.faces(source_frame_index)  # type: ignore[attr-defined]
        if not source_faces:
            self.statusBar().showMessage(f"No faces in the {direction} frame to copy", 5000)  # type: ignore[attr-defined]
            return False
        faces, mask_state, persisted = self._editable.copied_frame_payload(  # type: ignore[attr-defined]
            source_frame_index,
            current_index,
        )
        self._editable.replace_frame_faces(  # type: ignore[attr-defined]
            current_index,
            faces,
            mask_state=mask_state,
            persisted_mask_blobs=persisted,
        )
        self._editor_state.set("face_count_changed", True)  # type: ignore[attr-defined]
        self._editor_state.set("edited", True)  # type: ignore[attr-defined]
        self.mark_dirty(True)  # type: ignore[attr-defined]
        if self._editable.face_count(current_index):  # type: ignore[attr-defined]
            self._editor_state.set("face_index", 0)  # type: ignore[attr-defined]
        self.statusBar().showMessage(  # type: ignore[attr-defined]
            f"Copied {len(source_faces)} face(s) from the {direction} frame", 5000
        )
        return True

    def undo_edit(self) -> bool:
        """Reverse the last edit, if any."""
        if not self._editable.undo():  # type: ignore[attr-defined]
            self.statusBar().showMessage("Nothing to undo", 3000)  # type: ignore[attr-defined]
            return False
        self.statusBar().showMessage("Undid last edit", 3000)  # type: ignore[attr-defined]
        return True

    def redo_edit(self) -> bool:
        """Re-apply the most recently undone edit, if any."""
        if not self._editable.redo():  # type: ignore[attr-defined]
            self.statusBar().showMessage("Nothing to redo", 3000)  # type: ignore[attr-defined]
            return False
        self.statusBar().showMessage("Redid last edit", 3000)  # type: ignore[attr-defined]
        return True

    def save(self) -> bool:
        """Schedule a non-blocking save of editable edits."""
        if self._save_in_flight:  # type: ignore[attr-defined]
            self.statusBar().showMessage("Save already in progress…", 3000)  # type: ignore[attr-defined]
            return False
        if self._current_frame_index() < 0 and not any(  # type: ignore[attr-defined]
            self._editable.face_count(i)  # type: ignore[attr-defined]
            for i in range(self._session.frame_count)  # type: ignore[attr-defined]
        ):
            self.statusBar().showMessage("Nothing to save", 3000)  # type: ignore[attr-defined]
            return True

        self._save_busy_stack = contextlib.ExitStack()
        self._save_busy_stack.enter_context(self._with_busy_lock("Saving alignments…", save=True))  # type: ignore[attr-defined]

        try:
            frame_names = self._frame_names_for_persist()
            worker = ManualSaveWorker(
                self._alignments_handle,  # type: ignore[attr-defined]
                self._editable,  # type: ignore[attr-defined]
                frame_names,
                parent=self,  # type: ignore[arg-type]
            )
        except Exception as err:  # noqa: BLE001 - schedule-time failure
            logger.exception("Manual Tool save: failed to schedule worker")
            self._drain_save_busy_stack()
            self.statusBar().showMessage(f"Manual Tool save failed: {err}", 7000)  # type: ignore[attr-defined]
            QMessageBox.critical(self, "Manual Tool Save", f"Manual Tool save failed: {err}")
            return False
        worker.completed.connect(self._on_save_completed)
        worker.failed.connect(self._on_save_failed)
        self._save_worker = worker
        worker.start()
        return True

    def _on_save_completed(self, modified: int) -> None:
        """Worker reported success. Finalise dirty state and refresh views."""
        self._drain_save_busy_stack()
        self._teardown_save_worker()
        self._editable.clear_history()  # type: ignore[attr-defined]
        self.mark_dirty(False)  # type: ignore[attr-defined]
        self._editor_state.set("edited", False)  # type: ignore[attr-defined]
        self._editor_state.set("face_count_changed", False)  # type: ignore[attr-defined]
        self.refresh_faces()  # type: ignore[attr-defined]
        self._frame_view.update()  # type: ignore[attr-defined]
        self.statusBar().showMessage(f"Saved {modified} frame(s) to alignments file", 5000)  # type: ignore[attr-defined]
        self._emit_console(f"Manual Tool: saved {modified} frame(s)")  # type: ignore[attr-defined]
        self._resume_extract_after_save()

    def _on_save_failed(self, message: str) -> None:
        """Worker reported failure. Leave dirty state intact and surface error."""
        self._drain_save_busy_stack()
        self._teardown_save_worker()
        full = f"Manual Tool save failed: {message}"
        logger.error(full)
        self.statusBar().showMessage(full, 7000)  # type: ignore[attr-defined]
        self._emit_console(full)  # type: ignore[attr-defined]
        QMessageBox.critical(self, "Manual Tool Save", full)
        self._pending_extract_folder = None

    def _drain_save_busy_stack(self) -> None:
        """Release the busy-lock context held for the in-flight save."""
        stack = self._save_busy_stack
        self._save_busy_stack = None  # type: ignore[assignment]
        if stack is not None:
            stack.close()

    def _teardown_save_worker(self) -> None:
        """Stop and drop the save worker, joining its QThread."""
        if self._save_worker is not None:
            self._save_worker.stop()
            self._save_worker.deleteLater()
            self._save_worker = None  # type: ignore[assignment]

    def _frame_name_for_index(self, frame_index: int) -> str | None:
        """Resolve one editable frame index to its on-disk frame name."""
        mapping = self._frame_names_for_persist()
        if callable(mapping):
            return mapping(frame_index)
        if 0 <= frame_index < len(mapping):
            return mapping[frame_index]
        return None

    def _frame_names_for_persist(self) -> list[str] | T.Callable[[int], str | None]:
        """Return a frame-name mapping ordered to match the editable model."""
        if self._session.has_images:  # type: ignore[attr-defined]
            return [frame.name for frame in self._session.frame_list]  # type: ignore[attr-defined]
        if self._session.is_video_input:  # type: ignore[attr-defined]
            return self._session.frame_name_for_index  # type: ignore[attr-defined, no-any-return]
        return list(self._alignments_handle.sorted_frame_names())  # type: ignore[attr-defined]

    def extract_faces(self) -> bool:
        """Prompt for an output folder and run Extract Faces against live edits."""
        if self._extract_worker is not None:
            self.statusBar().showMessage("Extraction already in progress…", 4000)  # type: ignore[attr-defined]
            return False
        if not self._editable_has_any_face():
            self.statusBar().showMessage(
                "Nothing to extract — no faces in the editable model", 4000
            )
            return False
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select output folder for extracted faces",
            self._initial_extract_dir(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not chosen:
            return False
        self._start_extract_worker(chosen)
        return True

    def _editable_has_any_face(self) -> bool:
        """Return whether the live editable model has any face anywhere."""
        return any(
            self._editable.face_count(index) > 0  # type: ignore[attr-defined]
            for index in self._editable_frame_indices()
        )

    def _editable_frame_indices(self) -> tuple[int, ...]:
        """Return every frame index that currently has editable state."""
        known = set(self._editable.known_frame_indices())  # type: ignore[attr-defined]
        if self._session.has_images:  # type: ignore[attr-defined]
            known.update(range(len(self._session.frame_list)))  # type: ignore[attr-defined]
        return tuple(sorted(known))

    def _build_editable_extract_targets(self) -> tuple[tuple[str, tuple[T.Any, ...]], ...]:
        """Materialize the live editable model as extract targets."""
        import numpy as np

        from tools.manual.face_extraction import EditableFaceSpec

        resolver = self._frame_names_for_persist()
        try:
            alignments = self._alignments_handle.open()  # type: ignore[attr-defined]
            persisted_by_name = {
                name: tuple(entry.faces) for name, entry in alignments.data.items()
            }
        except Exception:  # noqa: BLE001 - extract still works against unsaved-only sessions
            persisted_by_name = {}

        targets: list[tuple[str, tuple[EditableFaceSpec, ...]]] = []
        for frame_index in self._editable_frame_indices():
            faces = self._editable.faces(frame_index)  # type: ignore[attr-defined]
            if not faces:
                continue
            if callable(resolver):
                frame_name = resolver(frame_index)
            elif 0 <= frame_index < len(resolver):
                frame_name = resolver[frame_index]
            else:
                frame_name = None
            if not frame_name:
                continue
            persisted = persisted_by_name.get(frame_name, ())
            specs: list[EditableFaceSpec] = []
            for face in faces:
                landmarks = (
                    np.asarray(face.landmarks, dtype=np.float32)
                    if face.landmarks
                    else np.zeros((0, 2), dtype=np.float32)
                )
                prev = persisted[face.face_index] if face.face_index < len(persisted) else None
                x, y, w, h = (int(round(value)) for value in face.bbox)
                specs.append(
                    EditableFaceSpec(
                        face_index=face.face_index,
                        bbox=(x, y, w, h),
                        landmarks_xy=landmarks,
                        mask=dict(getattr(prev, "mask", {}) or {}),
                        identity=dict(getattr(prev, "identity", {}) or {}),
                        metadata=dict(getattr(prev, "metadata", {}) or {}),
                    )
                )
            targets.append((frame_name, tuple(specs)))
        return tuple(targets)

    def _initial_extract_dir(self) -> str:
        """Default the folder picker to a sibling of the input source."""
        if self._session.is_video_input:  # type: ignore[attr-defined]
            return os.path.dirname(self._session.frames)  # type: ignore[attr-defined, no-any-return]
        return self._session.frames  # type: ignore[attr-defined, no-any-return]

    def _start_extract_worker(self, output_folder: str) -> None:
        """Persist dirty edits first, then start the extract worker."""
        if self._editor_state.unsaved:  # type: ignore[attr-defined]
            self._pending_extract_folder = output_folder  # type: ignore[assignment]
            self.save()
            return
        self._launch_extract_worker(output_folder)

    def _resume_extract_after_save(self) -> None:
        """Continue a pending extract once a chained save completes."""
        folder = self._pending_extract_folder
        self._pending_extract_folder = None
        if folder is None:
            return
        self._launch_extract_worker(folder)

    def _launch_extract_worker(self, output_folder: str) -> None:
        """Materialize the editable snapshot and start the extract worker."""
        editable_targets = self._build_editable_extract_targets()
        worker = ManualExtractFacesWorker(
            self._alignments_handle,  # type: ignore[attr-defined]
            self._session,  # type: ignore[attr-defined]
            output_folder,
            editable_targets=editable_targets,
            parent=self,  # type: ignore[arg-type]
        )
        worker.progress.connect(self._on_extract_progress)
        worker.completed.connect(self._on_extract_completed)
        worker.failed.connect(self._on_extract_failed)
        self._extract_worker = worker
        self._extract_total = 0
        if self._progress_bar is None:  # type: ignore[has-type]
            self._progress_bar = self._build_progress_bar()  # type: ignore[attr-defined]
            self.statusBar().addPermanentWidget(self._progress_bar)  # type: ignore[attr-defined]
        if self._extract_cancel_button is None:  # type: ignore[has-type]
            self._extract_cancel_button = self._build_extract_cancel_button()
            self.statusBar().addPermanentWidget(self._extract_cancel_button)  # type: ignore[attr-defined]
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat("Extracting faces…")
        self._progress_bar.show()
        self._extract_cancel_button.setEnabled(True)
        self._extract_cancel_button.show()
        self.statusBar().showMessage(f"Extracting faces to {output_folder}", 5000)  # type: ignore[attr-defined]
        self._busy_operation = "Extracting faces…"
        self._sync_actions()  # type: ignore[attr-defined]
        logger.info("Manual Tool extract started: %s", output_folder)
        self._emit_console(f"Manual Tool: extracting faces to {output_folder}")  # type: ignore[attr-defined]
        worker.start()

    def _on_extract_progress(self, done: int, total: int, message: str) -> None:
        """Update the determinate progress bar from the extract worker."""
        if total > 0 and self._extract_total != total:
            self._extract_total = total
            if self._progress_bar is not None:
                self._progress_bar.setRange(0, total)
                self._progress_bar.setFormat("Extracting %v / %m frame(s)")
        if self._progress_bar is not None and total > 0:
            self._progress_bar.setValue(min(done, total))
        self.statusBar().showMessage(message, 2000)  # type: ignore[attr-defined]

    def _on_extract_completed(self, result: object) -> None:
        """Surface a summary message and tear down the extract worker."""
        from tools.manual.face_extraction import ExtractFacesResult

        self._teardown_extract_worker()
        if not isinstance(result, ExtractFacesResult):  # pragma: no cover - defensive
            return
        if result.cancelled:
            self.statusBar().showMessage("Extract cancelled", 5000)  # type: ignore[attr-defined]
            return
        summary = (
            f"Extracted {result.faces_written} face(s) from {result.frames_processed} frame(s)"
        )
        skipped_bits: list[str] = []
        if result.skipped_frames:
            skipped_bits.append(f"{result.skipped_frames} frame(s)")
        if result.skipped_faces:
            skipped_bits.append(f"{result.skipped_faces} face(s)")
        if skipped_bits:
            summary += f"; skipped {', '.join(skipped_bits)}"
        self.statusBar().showMessage(summary, 7000)  # type: ignore[attr-defined]
        self._emit_console(f"Manual Tool: {summary}")  # type: ignore[attr-defined]
        logger.info("Manual Tool extract completed: %s", summary)
        if result.errors:
            joined_errors = chr(10).join(result.errors[:10])
            QMessageBox.warning(
                self,
                "Extract Faces",
                f"{summary}{chr(10)}{chr(10)}{joined_errors}",
            )

    def _on_extract_failed(self, message: str) -> None:
        """Report worker-level failure and tear down the extract worker."""
        self._teardown_extract_worker()
        full = f"Extract faces failed: {message}"
        logger.error(full)
        self._emit_console(full)  # type: ignore[attr-defined]
        self.statusBar().showMessage(full, 7000)  # type: ignore[attr-defined]
        QMessageBox.critical(self, "Extract Faces", full)

    def _teardown_extract_worker(self) -> None:
        """Hide progress and Cancel button, clear busy state, join the worker."""
        if self._progress_bar is not None:
            self._progress_bar.hide()
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setFormat("Loading %p%")
        if self._extract_cancel_button is not None:
            self._extract_cancel_button.hide()
            self._extract_cancel_button.setEnabled(False)
        self._busy_operation = None  # type: ignore[assignment]
        if self._extract_worker is not None and self._extract_worker.stop():
            self._extract_worker.deleteLater()
            self._extract_worker = None  # type: ignore[assignment]
        self._extract_total = 0
        self._sync_actions()  # type: ignore[attr-defined]

    def _build_extract_cancel_button(self) -> QToolButton:
        """Return a status-bar Cancel button bound to :meth:`cancel_extract`."""
        button = QToolButton(self)
        button.setObjectName("qt-manual-extract-cancel")
        button.setText("Cancel")
        button.setToolTip("Cancel Extract Faces (stops at the next frame boundary)")
        button.setAutoRaise(False)
        button.setEnabled(False)
        button.hide()
        button.clicked.connect(self.cancel_extract)
        return button

    def cancel_extract(self) -> bool:
        """Request an in-flight Extract Faces job to stop at the next frame."""
        worker = self._extract_worker
        if worker is None:
            return False
        worker.cancel()
        if self._extract_cancel_button is not None:
            self._extract_cancel_button.setEnabled(False)
        self.statusBar().showMessage("Cancelling extract; stopping at next frame…", 4000)  # type: ignore[attr-defined]
        self._emit_console("Manual Tool: extract cancellation requested")  # type: ignore[attr-defined]
        return True


__all__ = ["PersistenceMixin"]
