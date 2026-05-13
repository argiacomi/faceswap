#!/usr/bin/env python3
"""Tests for the optional Qt GUI launch wrapper."""

from __future__ import annotations

from argparse import Namespace

import pytest

pytest.importorskip("PySide6.QtWidgets")

from scripts import gui_qt  # noqa:E402


class _FakeApp:
    """Small QApplication stand-in."""

    instance_value = None

    def __init__(self, _argv) -> None:  # type:ignore[no-untyped-def]
        self.exec_count = 0
        self.stylesheet = ""

    @classmethod
    def instance(cls):  # type:ignore[no-untyped-def]
        """Return fake singleton instance."""
        return cls.instance_value

    def exec(self) -> None:
        """Capture event-loop execution."""
        self.exec_count += 1

    def setStyleSheet(self, stylesheet: str) -> None:  # noqa:N802
        """Capture applied stylesheet."""
        self.stylesheet = stylesheet


class _FakeWindow:
    """Small MainWindow stand-in."""

    def __init__(self) -> None:
        self.show_count = 0

    def show(self) -> None:
        """Capture show calls."""
        self.show_count += 1


def test_qt_gui_resolves_no_exec_from_argument() -> None:
    """The smoke launch flag should skip the Qt event loop."""
    assert gui_qt.Gui._resolve_no_exec(Namespace(no_gui_exec=True)) is True
    assert gui_qt.Gui._resolve_no_exec(Namespace(no_gui_exec=False)) is False


def test_qt_gui_resolves_no_exec_from_environment(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """The smoke launch environment variable should skip the Qt event loop."""
    monkeypatch.setenv(gui_qt.QT_NO_EXEC_ENV, "1")

    assert gui_qt.Gui._resolve_no_exec(Namespace()) is True


def test_qt_gui_no_exec_builds_window_without_event_loop(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Smoke mode should instantiate QApplication and MainWindow without entering exec."""
    _FakeApp.instance_value = None
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)

    process = gui_qt.Gui(Namespace(no_gui_exec=True))
    process.process()

    assert isinstance(process.app, _FakeApp)
    assert isinstance(process.root, _FakeWindow)
    assert process.root.show_count == 0
    assert process.app.exec_count == 0


def test_qt_gui_applies_default_theme(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Qt launch should apply the default shell stylesheet."""
    _FakeApp.instance_value = None
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)

    process = gui_qt.Gui(Namespace(no_gui_exec=True))

    assert process.theme.name == "Faceswap Dark"
    assert "QMainWindow" in process.app.stylesheet


def test_qt_gui_normal_launch_shows_window_and_runs_owned_event_loop(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Normal launch should show the window and enter exec when this process owns QApplication."""
    _FakeApp.instance_value = None
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)

    process = gui_qt.Gui(Namespace(no_gui_exec=False))
    process.process()

    assert process.root.show_count == 1
    assert process.app.exec_count == 1


def test_qt_gui_reuses_existing_qapplication_without_exec(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """If a QApplication already exists, the wrapper should not own or run its event loop."""
    existing_app = _FakeApp([])
    _FakeApp.instance_value = existing_app
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)

    process = gui_qt.Gui(Namespace(no_gui_exec=False))
    process.process()

    assert process.app is existing_app
    assert process.root.show_count == 1
    assert existing_app.exec_count == 0
    _FakeApp.instance_value = None
