#!/usr/bin/env python3
"""State persistence helpers for the Qt Manual Tool."""

from __future__ import annotations

import json
import typing as T

from PySide6.QtCore import QByteArray, QSettings


class WindowStateMixin:
    """Persist and restore Manual Tool root-window geometry/state."""

    _SETTINGS_ORG = "Faceswap"
    _SETTINGS_APP = "QtManualTool"
    _WINDOW_STATE_KEY = "manual_tool/window_state"

    def _settings(self) -> QSettings:
        """Return the persistent settings store for Manual Tool UI polish."""
        return QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)

    def _capture_manual_window_state(self) -> dict[str, object]:
        """Capture geometry, window state and Manual Tool splitter sizes."""
        geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
        window_state = bytes(self.saveState().toBase64()).decode("ascii")
        return {
            "geometry": geometry,
            "window_state": window_state,
            "maximized": self.isMaximized(),
            "fullscreen": self.isFullScreen(),
            "splitter_sizes": self._manual_splitter.sizes() if self._manual_splitter else [],
        }

    def _restore_manual_window_state_from(self, state: T.Mapping[str, object]) -> bool:
        """Restore a state payload captured by :meth:`_capture_manual_window_state`."""
        restored = False
        geometry = state.get("geometry")
        if isinstance(geometry, str) and geometry:
            restored = bool(self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii"))))
        window_state = state.get("window_state")
        if isinstance(window_state, str) and window_state:
            self.restoreState(QByteArray.fromBase64(window_state.encode("ascii")))
        sizes = state.get("splitter_sizes")
        if self._manual_splitter is not None and isinstance(sizes, list | tuple):
            try:
                int_sizes = [int(value) for value in sizes]
            except (TypeError, ValueError):
                int_sizes = []
            if int_sizes:
                self._manual_splitter.setSizes(int_sizes)
                restored = True
        if bool(state.get("fullscreen")):
            self.showFullScreen()
            restored = True
        elif bool(state.get("maximized")):
            self.showMaximized()
            restored = True
        return restored

    def _save_manual_window_state(self) -> None:
        """Persist Manual Tool window state."""
        settings = self._settings()
        settings.setValue(self._WINDOW_STATE_KEY, json.dumps(self._capture_manual_window_state()))
        settings.sync()

    def _restore_manual_window_state(self) -> bool:
        """Restore persisted Manual Tool window state, if present."""
        raw = self._settings().value(self._WINDOW_STATE_KEY, "")
        if not isinstance(raw, str) or not raw:
            return False
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(state, dict):
            return False
        return self._restore_manual_window_state_from(state)


__all__ = ["WindowStateMixin"]
