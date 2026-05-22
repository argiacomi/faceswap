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

    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        self.show_count = 0
        self.apply_calls = 0
        self.theme = kwargs.get("theme")

    def show(self) -> None:
        """Capture show calls."""
        self.show_count += 1

    def apply_gui_settings(self) -> None:
        """Capture gui-config apply calls."""
        self.apply_calls += 1

    def resize(self, *_args, **_kwargs) -> None:
        """No-op resize matching MainWindow signature."""


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
    """Normal launch should show the window, install signal handlers, and enter exec."""
    _FakeApp.instance_value = None
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)
    installed: list[tuple[object, object]] = []
    monkeypatch.setattr(
        gui_qt,
        "install_signal_handlers",
        lambda app, window: installed.append((app, window)),
    )

    process = gui_qt.Gui(Namespace(no_gui_exec=False))
    process.process()

    assert process.root.show_count == 1
    assert process.app.exec_count == 1
    assert installed == [(process.app, process.root)]


def test_qt_gui_reuses_existing_qapplication_without_exec(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """If a QApplication already exists, the wrapper should not own or run its event loop."""
    existing_app = _FakeApp([])
    _FakeApp.instance_value = existing_app
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)
    installed: list[tuple[object, object]] = []
    monkeypatch.setattr(
        gui_qt,
        "install_signal_handlers",
        lambda app, window: installed.append((app, window)),
    )

    process = gui_qt.Gui(Namespace(no_gui_exec=False))
    process.process()

    assert process.app is existing_app
    assert process.root.show_count == 1
    assert existing_app.exec_count == 0
    # When the wrapper doesn't own the QApplication, it should not register
    # signal handlers (the outer host owns SIGINT routing).
    assert installed == []
    _FakeApp.instance_value = None


def test_qt_gui_handles_keyboard_interrupt_via_helper(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """KeyboardInterrupt raised during exec should call interrupt_window and sys.exit(130)."""
    _FakeApp.instance_value = None
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)
    monkeypatch.setattr(gui_qt, "install_signal_handlers", lambda app, window: None)
    interrupted: list[object] = []
    monkeypatch.setattr(gui_qt, "interrupt_window", interrupted.append)

    def raise_interrupt(_self) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(_FakeApp, "exec", raise_interrupt)

    process = gui_qt.Gui(Namespace(no_gui_exec=False))
    with pytest.raises(SystemExit) as excinfo:
        process.process()

    assert excinfo.value.code == gui_qt.INTERRUPT_EXIT_CODE
    assert interrupted == [process.root]


def test_qt_gui_no_exec_does_not_touch_streams_or_root_logger(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Smoke-mode process() must leave sys.stdout/stderr and root handlers untouched."""
    import logging
    import sys

    _FakeApp.instance_value = None
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_handlers = list(logging.getLogger().handlers)

    process = gui_qt.Gui(Namespace(no_gui_exec=True, debug=False))
    # After __init__ alone, no redirection or handler registration should have
    # occurred — install happens inside process() under try/finally.
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert logging.getLogger().handlers == original_handlers

    process.process()

    # And after the early-return process() call the same invariants must hold.
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert logging.getLogger().handlers == original_handlers


def test_qt_gui_reused_qapp_does_not_touch_streams_or_root_logger(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Reused-QApplication process() must leave streams/handlers untouched."""
    import logging
    import sys

    existing_app = _FakeApp([])
    _FakeApp.instance_value = existing_app
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)
    monkeypatch.setattr(gui_qt, "install_signal_handlers", lambda app, window: None)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_handlers = list(logging.getLogger().handlers)

    process = gui_qt.Gui(Namespace(no_gui_exec=False, debug=False))
    process.process()

    # The wrapper returns before installing routers when it doesn't own the
    # QApplication, so the host process must see no change to its log surface.
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert logging.getLogger().handlers == original_handlers
    _FakeApp.instance_value = None


def test_qt_gui_restores_streams_when_signal_install_fails(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """An exception after install must be caught by the same try/finally."""
    import logging
    import sys

    _FakeApp.instance_value = None
    monkeypatch.setattr(gui_qt, "QApplication", _FakeApp)
    monkeypatch.setattr(gui_qt, "MainWindow", _FakeWindow)

    def boom(_app, _window) -> None:
        raise RuntimeError("signal install failed")

    monkeypatch.setattr(gui_qt, "install_signal_handlers", boom)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_handlers = list(logging.getLogger().handlers)

    process = gui_qt.Gui(Namespace(no_gui_exec=False, debug=False))
    with pytest.raises(RuntimeError, match="signal install failed"):
        process.process()

    # Even though the failure happened *after* _install_console_logging, the
    # finally clause must have torn down the redirection and the handler.
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert logging.getLogger().handlers == original_handlers
