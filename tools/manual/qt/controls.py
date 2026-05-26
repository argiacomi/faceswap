#!/usr/bin/env python3
"""Qt Manual Tool contextual control-panel helpers."""

from __future__ import annotations

import logging
import typing as T

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .workers import ManualAlignerLoadWorker

logger = logging.getLogger(__name__)


class ControlsMixin:
    """Own mode-specific and shared contextual editor controls."""

    _MUTATING_ACTION_KEYS: T.ClassVar[tuple[str, ...]] = (
        "save",
        "revert_frame",
        "copy_prev_face",
        "copy_next_face",
        "delete_face",
        "add_face",
        "undo_edit",
        "redo_edit",
        "extract_faces",
    )

    def set_editor_overlay_color(self, editor_mode: str, role: str, color: str | QColor) -> None:
        """Override one overlay color role for one editor mode."""
        if role not in self._OVERLAY_COLOR_DEFAULTS:
            raise ValueError(f"Unknown overlay color role: {role}")
        qcolor = color if isinstance(color, QColor) else QColor(str(color))
        if not qcolor.isValid():
            raise ValueError(f"Invalid overlay color: {color}")
        self._overlay_color_overrides.setdefault(str(editor_mode), {})[role] = QColor(qcolor)
        self._frame_view.update()

    def editor_overlay_color(self, editor_mode: str, role: str) -> QColor:
        """Return the configured overlay color for ``editor_mode`` and ``role``."""
        override = self._overlay_color_overrides.get(str(editor_mode), {}).get(role)
        if override is not None:
            return QColor(override)
        return QColor(self._OVERLAY_COLOR_DEFAULTS.get(role, QColor("#3aa0ff")))

    def _overlay_color(self, role: str) -> QColor:
        """Return overlay color for the current editor mode."""
        return self.editor_overlay_color(self._editor_state.editor_mode, role)

    def _on_unsaved_changed(self, dirty: bool) -> None:
        """Forward editor-state unsaved changes to UI signals."""
        self.dirty_changed.emit(bool(dirty))
        self._sync_actions()

    def cycle_filter_mode(self) -> None:
        """Advance to the next filter mode in the legacy rotation order."""
        from tools.manual.frame_filter import FILTER_MODES

        current = self._editor_state.filter_mode or FILTER_MODES[0]
        try:
            index = FILTER_MODES.index(current)
        except ValueError:
            index = -1
        new_mode = FILTER_MODES[(index + 1) % len(FILTER_MODES)]
        self._editor_state.set("filter_mode", new_mode)
        self._refresh_filter_results()
        filtered_count = len(self._filtered_frame_indices)
        suffix = f" ({filtered_count} match)" if filtered_count else " (no matches)"
        self.statusBar().showMessage(f"Filter: {new_mode}{suffix}", 5000)

    def cycle_annotation_display(self) -> None:
        """Cycle annotation overlays in the legacy rotation order."""
        order = ("None", "Mesh", "Mask", "Landmarks")
        current = self._editor_state.annotation_mode or order[0]
        try:
            index = order.index(current)
        except ValueError:
            index = -1
        self._editor_state.set("annotation_mode", order[(index + 1) % len(order)])
        self.statusBar().showMessage(f"Annotation: {self._editor_state.annotation_mode}", 5000)

    def set_editor_view(self) -> None:
        """Activate the View editor mode."""
        self._editor_state.set("editor_mode", "View")

    def set_editor_boundingbox(self) -> None:
        """Activate the Bounding Box editor mode."""
        self._editor_state.set("editor_mode", "BoundingBox")

    def set_editor_extractbox(self) -> None:
        """Activate the Extract Box editor mode."""
        self._editor_state.set("editor_mode", "ExtractBox")

    def set_editor_landmarks(self) -> None:
        """Activate the Landmarks editor mode."""
        self._editor_state.set("editor_mode", "Landmarks")

    def set_editor_mask(self) -> None:
        """Activate the Mask editor mode."""
        self._editor_state.set("editor_mode", "Mask")

    def zoom_in(self) -> None:
        """Zoom into the embedded frame view."""
        self._frame_view.zoom_in()

    def zoom_out(self) -> None:
        """Zoom out of the embedded frame view."""
        self._frame_view.zoom_out()

    def reset_view(self) -> None:
        """Reset the embedded frame view."""
        self._magnify_restore_state = None
        self._frame_view.reset_view()

    def magnify_active_face(self) -> bool:
        """Toggle fitting the active face bbox to the viewport."""
        if self._magnify_restore_state is not None:
            return self._restore_magnified_view()
        bbox = self._active_face_bbox()
        if bbox is None:
            self.statusBar().showMessage("No active face to magnify", 3000)
            return False
        self._magnify_restore_state = self._frame_view.view_state()
        if not self._frame_view.magnify_to_source_rect(bbox):
            self._magnify_restore_state = None
            self.statusBar().showMessage("Could not magnify active face", 3000)
            return False
        return True

    def _auto_magnify_active_face(self) -> bool:
        """Fit the active face when entering detail editors without losing view state."""
        if self._magnify_restore_state is not None:
            return True
        bbox = self._active_face_bbox()
        if bbox is None:
            return False
        self._magnify_restore_state = self._frame_view.view_state()
        if self._frame_view.magnify_to_source_rect(bbox):
            return True
        self._magnify_restore_state = None
        return False

    def _restore_magnified_view(self) -> bool:
        """Restore the zoom/pan state captured before magnification."""
        state = self._magnify_restore_state
        if state is None:
            return False
        self._magnify_restore_state = None
        return self._frame_view.restore_view_state(state)

    def _build_filter_controls(self) -> QWidget:
        """Return the filter-awareness control row."""
        from tools.manual.frame_filter import (
            MISALIGNED_THRESHOLD_MAX,
            MISALIGNED_THRESHOLD_MIN,
        )

        container = QWidget()
        container.setObjectName("qt-manual-filter-controls")
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        self._filter_label = QLabel()
        self._filter_label.setObjectName("qt-manual-filter-label")
        layout.addWidget(self._filter_label)
        layout.addStretch(1)

        self._filter_threshold_label = QLabel("Threshold:")
        self._filter_threshold_label.setObjectName("qt-manual-filter-threshold-label")
        layout.addWidget(self._filter_threshold_label)
        self._filter_threshold_slider = QSlider(Qt.Horizontal)
        self._filter_threshold_slider.setObjectName("qt-manual-filter-threshold-slider")
        self._filter_threshold_slider.setRange(MISALIGNED_THRESHOLD_MIN, MISALIGNED_THRESHOLD_MAX)
        self._filter_threshold_slider.setValue(int(self._editor_state.filter_distance))
        self._filter_threshold_slider.setFixedWidth(120)
        self._filter_threshold_slider.valueChanged.connect(self._on_filter_threshold_changed)
        layout.addWidget(self._filter_threshold_slider)
        self._filter_threshold_value = QLabel(str(self._editor_state.filter_distance))
        self._filter_threshold_value.setObjectName("qt-manual-filter-threshold-value")
        layout.addWidget(self._filter_threshold_value)

        self._refresh_filter_controls()
        return container

    def _refresh_filter_controls(self) -> None:
        """Mirror active filter mode and match count into the control row."""
        label = getattr(self, "_filter_label", None)
        if label is None:
            return
        mode = self._editor_state.filter_mode or "All Frames"
        total = len(self._filtered_frame_indices)
        if mode == "All Frames":
            text = f"Filter: All Frames ({total})"
        else:
            text = f"Filter: {mode} ({total} match)"
        label.setText(text)

        misaligned = mode == "Misaligned Faces"
        if hasattr(self, "_filter_threshold_slider"):
            self._filter_threshold_label.setVisible(misaligned)
            self._filter_threshold_slider.setVisible(misaligned)
            self._filter_threshold_value.setVisible(misaligned)
            current = int(self._editor_state.filter_distance)
            self._filter_threshold_value.setText(str(current))
            self._filter_threshold_slider.blockSignals(True)
            try:
                self._filter_threshold_slider.setValue(current)
            finally:
                self._filter_threshold_slider.blockSignals(False)

    def _on_filter_threshold_changed(self, value: int) -> None:
        """Persist threshold and refresh the filter when the slider moves."""
        self._editor_state.set("filter_distance", int(value))
        self._refresh_filter_results()

    def available_aligners(self) -> tuple[str, ...]:
        """Return aligner plugin display names from the cached service."""
        return self._aligner_service.available_aligners

    def available_normalizations(self) -> tuple[str, ...]:
        """Return normalization methods supported by the aligner service."""
        return self._aligner_service.available_normalizations()

    def set_aligner_name(self, name: str) -> None:
        """Select the active aligner plugin."""
        value = str(name).strip()
        if not value or value.startswith("No aligners"):
            return
        self._editor_state.set("aligner_name", value)
        self._sync_aligner_controls()
        self._schedule_aligner_preload()

    def set_aligner_normalization(self, method: str) -> None:
        """Apply a normalization method to the aligner service."""
        value = str(method)
        self._editor_state.set("aligner_normalization", value)
        self._aligner_service.set_normalization(value)
        self._sync_aligner_controls()
        self._schedule_aligner_preload()

    def _active_aligner_name(self) -> str:
        """Return editor-state aligner name, falling back to the service default."""
        return self._editor_state.aligner_name or self._aligner_service.default_aligner()

    def _build_aligner_controls(self) -> QWidget:
        """Return the Bounding Box aligner control panel."""
        container = QWidget()
        container.setObjectName("qt-manual-aligner-controls")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(QLabel("Aligner:"))

        self._aligner_combo = QComboBox()
        self._aligner_combo.setObjectName("qt-manual-aligner-combo")
        self._aligner_combo.setMinimumWidth(150)
        self._aligner_combo.currentTextChanged.connect(self._on_aligner_combo_changed)
        top.addWidget(self._aligner_combo)

        self._aligner_auto_run_checkbox = QCheckBox("Auto-run")
        self._aligner_auto_run_checkbox.setObjectName("qt-manual-aligner-auto-run")
        self._aligner_auto_run_checkbox.setChecked(bool(self._editor_state.aligner_auto_run))
        self._aligner_auto_run_checkbox.toggled.connect(self._on_aligner_auto_run_toggled)
        top.addWidget(self._aligner_auto_run_checkbox)

        self._aligner_rerun_button = QToolButton()
        self._aligner_rerun_button.setObjectName("qt-manual-aligner-rerun")
        self._aligner_rerun_button.setText("Run")
        self._aligner_rerun_button.setToolTip("Run the selected aligner for the active face")
        self._aligner_rerun_button.clicked.connect(self._on_aligner_rerun_clicked)
        top.addWidget(self._aligner_rerun_button)
        top.addStretch(1)
        outer.addLayout(top)

        norm_row = QHBoxLayout()
        norm_row.setContentsMargins(0, 0, 0, 0)
        norm_row.setSpacing(8)
        norm_row.addWidget(QLabel("Normalization:"))
        self._aligner_normalization_group = QButtonGroup(container)
        self._aligner_normalization_group.setExclusive(True)
        self._aligner_normalization_buttons: dict[str, QRadioButton] = {}
        for method in self.available_normalizations():
            button = QRadioButton(str(method))
            button.setObjectName(f"qt-manual-aligner-normalization-{method}")
            button.toggled.connect(
                lambda checked, value=str(method): (
                    self.set_aligner_normalization(value) if checked else None
                )
            )
            self._aligner_normalization_group.addButton(button)
            self._aligner_normalization_buttons[str(method)] = button
            norm_row.addWidget(button)
        norm_row.addStretch(1)
        outer.addLayout(norm_row)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)
        self._aligner_load_progress = QProgressBar()
        self._aligner_load_progress.setObjectName("qt-manual-aligner-load-progress")
        self._aligner_load_progress.setRange(0, 100)
        self._aligner_load_progress.setValue(0)
        self._aligner_load_progress.setMaximumWidth(160)
        self._aligner_load_progress.setTextVisible(True)
        self._aligner_load_progress.hide()
        progress_row.addWidget(self._aligner_load_progress)

        self._aligner_status_label = QLabel("Aligner idle")
        self._aligner_status_label.setObjectName("qt-manual-aligner-status-label")
        self._aligner_status_label.setWordWrap(True)
        progress_row.addWidget(self._aligner_status_label, 1)
        outer.addLayout(progress_row)

        container.setVisible(self._editor_state.editor_mode == "BoundingBox")
        self._sync_aligner_controls()
        return container

    def _on_aligner_combo_changed(self, name: str) -> None:
        """Handle a user-selected aligner plugin from the dropdown."""
        self.set_aligner_name(name)

    def _on_aligner_auto_run_toggled(self, checked: bool) -> None:
        """Persist the auto-run checkbox value on editor state."""
        self._editor_state.set("aligner_auto_run", bool(checked))
        self._sync_aligner_controls()

    def _on_aligner_rerun_clicked(self) -> None:
        """Run the active aligner against the selected face."""
        face_index = self._active_face_index()
        if face_index is None:
            self.statusBar().showMessage("No active face to align", 3000)
            return
        self.rerun_aligner_for_face(int(face_index))

    def _refresh_aligner_controls_visibility(self) -> None:
        """Show or hide Bounding Box aligner controls to match editor mode."""
        controls = getattr(self, "_aligner_controls", None)
        if controls is None:
            return
        active = self._editor_state.editor_mode == "BoundingBox"
        controls.setVisible(active)
        if active:
            self._sync_aligner_controls()

    def _sync_aligner_controls(self) -> None:
        """Mirror editor-state aligner settings into the visible controls."""
        combo = getattr(self, "_aligner_combo", None)
        if combo is None:
            return
        aligners = list(self.available_aligners())
        current = self._editor_state.aligner_name or (
            self._aligner_service.default_aligner() if aligners else ""
        )
        if current and current not in aligners:
            aligners.insert(0, current)

        combo.blockSignals(True)
        combo.clear()
        if aligners:
            combo.addItems(aligners)
            combo.setEnabled(True)
            index = combo.findText(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        else:
            combo.addItem("No aligners available")
            combo.setEnabled(False)
        combo.blockSignals(False)

        normalization = str(self._editor_state.aligner_normalization)
        for method, button in self._aligner_normalization_buttons.items():
            button.blockSignals(True)
            button.setChecked(method == normalization)
            button.blockSignals(False)
        if (
            normalization not in self._aligner_normalization_buttons
            and self._aligner_normalization_buttons
        ):
            first = next(iter(self._aligner_normalization_buttons.values()))
            first.blockSignals(True)
            first.setChecked(True)
            first.blockSignals(False)

        auto_run = getattr(self, "_aligner_auto_run_checkbox", None)
        if auto_run is not None:
            auto_run.blockSignals(True)
            auto_run.setChecked(bool(self._editor_state.aligner_auto_run))
            auto_run.blockSignals(False)

        rerun = getattr(self, "_aligner_rerun_button", None)
        if rerun is not None:
            rerun.setEnabled(bool(aligners) and self._active_face_index() is not None)

    def _schedule_aligner_preload(self) -> None:
        """Preload the selected aligner/backend while BBox controls are visible."""
        controls = getattr(self, "_aligner_controls", None)
        if controls is None or not controls.isVisible():
            return
        aligner = self._active_aligner_name()
        normalization = str(self._editor_state.aligner_normalization)
        if not aligner:
            self._set_aligner_load_status("failed", "", "No aligner plugins available")
            return
        target = (aligner, normalization)
        if target in self._aligner_loaded_targets:
            self._set_aligner_load_status("ready", aligner, f"Aligner '{aligner}' ready")
            return
        if self._aligner_load_worker is not None and self._aligner_load_target == target:
            return
        if self._aligner_load_worker is not None:
            if not self._aligner_load_worker.stop():
                self.statusBar().showMessage(
                    "Aligner is still loading. Wait before changing selection.",
                    5000,
                )
                return
            self._aligner_load_worker.deleteLater()
            self._aligner_load_worker = None

        worker = ManualAlignerLoadWorker(
            self._aligner_service,
            aligner,
            normalization,
            parent=self,
        )
        worker.status.connect(self._on_aligner_load_status)
        worker.completed.connect(self._on_aligner_load_completed)
        self._aligner_load_worker = worker
        self._aligner_load_target = target
        self._set_aligner_load_status("loading", aligner, f"Loading aligner '{aligner}'…")
        worker.start()

    def _on_aligner_load_status(self, kind: str, aligner: str, message: str) -> None:
        """Handle status emitted by the background aligner preload worker."""
        self._set_aligner_load_status(kind, aligner, message)
        timeout = 5000 if kind == "failed" else 3000
        self.statusBar().showMessage(message, timeout)
        self._emit_console(f"Manual Tool aligner: {message}")

    def _on_aligner_load_completed(
        self, aligner: str, normalization: str, ok: bool, message: str
    ) -> None:
        """Finalize the visible aligner preload state."""
        target = (str(aligner), str(normalization))
        if self._aligner_load_target is not None and target != self._aligner_load_target:
            return
        if ok:
            self._aligner_loaded_targets.add(target)
            self._set_aligner_load_status("ready", aligner, message)
        else:
            self._set_aligner_load_status("failed", aligner, message)

        worker = self._aligner_load_worker
        if worker is not None:
            worker.stop(wait_ms=1000)
            worker.deleteLater()
            self._aligner_load_worker = None

        self._aligner_load_target = None
        self._sync_aligner_controls()

    def _set_aligner_load_status(self, kind: str, aligner: str, message: str) -> None:
        """Paint the inline aligner load-progress widget."""
        progress = getattr(self, "_aligner_load_progress", None)
        label = getattr(self, "_aligner_status_label", None)
        if label is not None and message:
            label.setText(message)
        if progress is None:
            return
        if kind in ("loading", "aligning"):
            progress.setRange(0, 0)
            progress.setFormat("Loading aligner…")
            progress.show()
            rerun = getattr(self, "_aligner_rerun_button", None)
            if rerun is not None:
                rerun.setEnabled(False)
        elif kind in ("ready", "aligned"):
            progress.hide()
            progress.setRange(0, 100)
            progress.setValue(100)
        elif kind == "failed":
            progress.hide()
            progress.setRange(0, 100)
            progress.setValue(0)

    def _on_aligner_status(self, status: T.Any) -> None:
        """Forward aligner service events to status, console and controls."""
        message = getattr(status, "message", "") or ""
        kind = getattr(status, "kind", "") or ""
        aligner = getattr(status, "aligner", "") or self._active_aligner_name()
        if not message:
            return
        self._set_aligner_load_status(kind, aligner, message)
        timeout = 5000 if kind == "failed" else 3000
        self.statusBar().showMessage(message, timeout)
        self._emit_console(f"Manual Tool aligner: {message}")
        if kind == "failed":
            logger.error("Manual Tool aligner: %s", message)

    def rerun_aligner_for_face(self, face_index: int) -> bool:
        """Rerun the configured aligner against one face on the current frame."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to align", 3000)
            return False
        faces = self._editable.faces(frame_index)
        if face_index < 0 or face_index >= len(faces):
            self.statusBar().showMessage("No active face to align", 3000)
            return False
        image = self._frame_view.current_frame_array()
        if image is None:
            self.statusBar().showMessage("Cannot align: frame image not loaded", 4000)
            return False
        face = faces[face_index]
        try:
            landmarks = self._aligner_service.align(
                image,
                face.bbox,
                aligner=self._active_aligner_name() or None,
                normalization=self._editor_state.aligner_normalization,
            )
        except Exception as err:  # noqa: BLE001 - surface to user; model untouched
            logger.exception("Manual Tool: aligner run failed")
            self.statusBar().showMessage(f"Aligner failed: {err}", 5000)
            self._emit_console(f"Manual Tool aligner failed: {err}")
            return False
        new_points = [(float(point[0]), float(point[1])) for point in landmarks]
        if not self._editable.set_landmarks(frame_index, face_index, new_points):
            return False
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()
        return True

    def _sync_actions(self) -> None:
        """Update action availability from session, selection and dirty state."""
        if not self._actions:
            return
        filtered = self._filtered_frame_indices
        filtered_total = len(filtered)
        has_frames = filtered_total > 0
        if has_frames:
            filtered_position = self._filtered_position()
            not_first = filtered_position > 0
            not_last = 0 <= filtered_position < filtered_total - 1
        else:
            not_first = False
            not_last = False
        frame_index = self._current_frame_index()
        editable_count = self._editable.face_count(frame_index) if frame_index >= 0 else 0
        face_selected = self._editor_state.face_index >= 0 and (
            editable_count > 0 or bool(self._face_panel.faces)
        )
        has_frame_loaded = self._frame_view.has_frame
        editor_mode = self._editor_state.editor_mode

        availability: dict[str, bool] = {
            "save": self._editor_state.unsaved,
            "revert_frame": self._editor_state.edited
            or self._editor_state.unsaved
            or self._editable.can_undo,
            "first_frame": not_first,
            "previous_frame": not_first,
            "next_frame": not_last,
            "last_frame": not_last,
            "play_pause": has_frames,
            "copy_prev_face": not_first,
            "copy_next_face": not_last,
            "delete_face": face_selected and editable_count > 0,
            "add_face": has_frame_loaded,
            "undo_edit": self._editable.can_undo,
            "redo_edit": self._editable.can_redo,
            "cycle_filter": has_frames,
            "cycle_annotation": has_frames,
            "set_view_mode": editor_mode != "View",
            "set_boundingbox_mode": editor_mode != "BoundingBox",
            "set_extractbox_mode": editor_mode != "ExtractBox",
            "set_landmarks_mode": editor_mode != "Landmarks",
            "set_mask_mode": editor_mode != "Mask",
            "zoom_in": has_frames,
            "zoom_out": has_frames,
            "reset_view": has_frames,
            "legacy_tool": bool(self._legacy_args),
            "extract_faces": has_frames and self._alignments_handle.exists,
        }
        if self._busy_operation is not None:
            for key in self._MUTATING_ACTION_KEYS:
                if key in availability:
                    availability[key] = False
        for key, enabled in availability.items():
            action = self._actions.get(key)
            if action is not None:
                action.setEnabled(enabled)

    def _on_editor_mode_changed(self, _mode: object) -> None:
        """Refresh action availability when the editor mode flips."""
        self._sync_actions()
        self._refresh_mask_controls_visibility()
        self._refresh_aligner_controls_visibility()
        mode = str(self._editor_state.editor_mode)
        if mode in {"Landmarks", "Mask"}:
            self._auto_magnify_active_face()
        else:
            self._restore_magnified_view()
        self._frame_view.update()


__all__ = ["ControlsMixin"]
