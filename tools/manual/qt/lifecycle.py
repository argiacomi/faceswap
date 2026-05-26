#!/usr/bin/env python3
"""Qt Manual Tool lifecycle, shutdown and status helpers."""

from __future__ import annotations

import contextlib
import logging
import subprocess
import typing as T

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """Own close/shutdown, legacy launch, busy state and console/status plumbing."""

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa:N802
        """Prompt before closing when the editor has unsaved changes."""
        if self._editor_state.unsaved:
            answer = QMessageBox.question(
                self,
                "Unsaved Manual Tool Changes",
                "Close the Manual Tool and discard unsaved changes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._extract_worker is not None:
            stopped = self._extract_worker.stop()
            if not stopped:
                self.statusBar().showMessage(
                    "Extraction still running — please wait for it to finish, then close again.",
                    7000,
                )
                event.ignore()
                return
            self._extract_worker = None
        if self._save_worker is not None:
            self._save_worker.stop()
            self._save_worker = None
            self._drain_save_busy_stack()
        if self._aligner_load_worker is not None:
            stopped = self._aligner_load_worker.stop()
            if not stopped:
                self.statusBar().showMessage(
                    "Aligner is still loading — please wait, then close again.",
                    7000,
                )
                event.ignore()
                return
            self._aligner_load_worker = None
            self._aligner_load_target = None
        if self._video_provider is not None:
            self._video_provider.shutdown()
            self._video_provider = None
        if self._startup_worker is not None:
            self._startup_worker.stop()
            self._startup_worker = None
        self._save_manual_window_state()
        event.accept()

    def mark_dirty(self, dirty: bool = True) -> None:
        """Set dirty state and update action availability."""
        self._editor_state.set("unsaved", dirty)

    @contextlib.contextmanager
    def _with_busy_lock(self, label: str, *, save: bool = False) -> T.Iterator[None]:
        """Run a blocking operation with progress + action gating."""
        prior_progress_format = None
        prior_progress_range: tuple[int, int] | None = None
        prior_progress_visible = False
        owns_progress_bar = False
        self._busy_operation = label
        if save:
            self._save_in_flight = True
        if self._progress_bar is None:
            self._progress_bar = self._build_progress_bar()
            self.statusBar().addPermanentWidget(self._progress_bar)
            owns_progress_bar = True
        if self._progress_bar is not None:
            prior_progress_format = self._progress_bar.format()
            prior_progress_range = (
                self._progress_bar.minimum(),
                self._progress_bar.maximum(),
            )
            prior_progress_visible = self._progress_bar.isVisible()
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setFormat(label)
            self._progress_bar.show()
        self._sync_actions()
        self.statusBar().showMessage(label, 3000)
        try:
            yield
        finally:
            self._busy_operation = None
            if save:
                self._save_in_flight = False
            if self._progress_bar is not None:
                if owns_progress_bar:
                    self._hide_progress_bar()
                else:
                    if prior_progress_range is not None:
                        self._progress_bar.setRange(*prior_progress_range)
                    if prior_progress_format is not None:
                        self._progress_bar.setFormat(prior_progress_format)
                    if not prior_progress_visible:
                        self._progress_bar.hide()
            self._sync_actions()

    def _hide_progress_bar(self) -> None:
        """Hide and detach the status-bar progress widget."""
        if self._busy_operation:
            if self._progress_bar is not None:
                self._progress_bar.hide()
            return
        if self._progress_bar is not None:
            self._progress_bar.hide()
            self.statusBar().removeWidget(self._progress_bar)
            self._progress_bar = None

    def _emit_console(self, message: str) -> None:
        """Forward a user-facing message to the host shell console, if any."""
        if self._console_logger is None:
            return
        try:
            self._console_logger(message)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Manual Tool console logger raised")

    def launch_legacy(self) -> bool:
        """Launch the existing Tk Manual Tool as a fallback subprocess."""
        if not self._legacy_args:
            QMessageBox.warning(self, "Legacy Manual Tool", "Legacy command is unavailable.")
            return False
        try:
            subprocess.Popen(self._legacy_args)  # noqa:S603 - args are built internally
        except OSError as err:
            QMessageBox.critical(self, "Legacy Manual Tool", str(err))
            return False
        self.statusBar().showMessage("Launched legacy Manual Tool", 5000)
        return True


__all__ = ["LifecycleMixin"]
