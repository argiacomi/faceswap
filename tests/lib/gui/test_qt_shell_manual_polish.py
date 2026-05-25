#!/usr/bin/env python3
"""Qt Manual Tool visual polish regressions (#124)."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QToolBar

from lib.gui.qt_shell.manual_tool import ManualToolWindow
from tools.manual.session import ManualSession


class _FakeSettings:
    """Tiny in-memory ``QSettings`` stand-in for deterministic state tests."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def value(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)

    def setValue(self, key: str, value: str) -> None:  # noqa:N802 - Qt API spelling
        self.values[key] = value

    def sync(self) -> None:
        return


def _session_with_frames(folder: Path, count: int = 1) -> ManualSession:
    """Write small PNG fixtures and return a Manual Tool session."""
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(160, 120)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    """Create a Manual Tool window with one loaded image frame."""
    session = _session_with_frames(folder)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.frame_view.source_size != (0, 0), timeout=2000)
    return window


def test_manual_tool_persists_and_restores_window_state(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    """Manual Tool saves geometry/window state plus the main splitter sizes."""
    settings = _FakeSettings()
    monkeypatch.setattr(ManualToolWindow, "_settings", lambda _self: settings)
    window = _make_window(qtbot, tmp_path / "first")
    assert window._manual_splitter is not None  # noqa:SLF001
    window._manual_splitter.setSizes([420, 160, 90])  # noqa:SLF001
    expected_sizes = window._manual_splitter.sizes()  # noqa:SLF001

    window._save_manual_window_state()  # noqa:SLF001

    raw = settings.values[ManualToolWindow._WINDOW_STATE_KEY]  # noqa:SLF001
    payload = json.loads(raw)
    assert payload["splitter_sizes"] == expected_sizes
    assert payload["geometry"]
    assert payload["window_state"]

    restored = _make_window(qtbot, tmp_path / "second")
    assert restored._manual_splitter is not None  # noqa:SLF001
    assert restored._manual_splitter.sizes() == expected_sizes  # noqa:SLF001


def test_manual_toolbar_uses_theme_icon_size(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The Manual toolbar adopts the shell theme's configured icon size."""
    window = _make_window(qtbot, tmp_path)
    toolbar = window.findChild(QToolBar, "qt-manual-toolbar")
    assert toolbar is not None
    assert toolbar.iconSize().width() == 16
    assert toolbar.iconSize().height() == 16
