#!/usr/bin/env python3
"""Validation tests for GUI shell launch hardening."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace

import pytest

from lib.cli.args import GuiArgs
from lib.cli.launcher import GUI_SHELL_ENV, QT_NO_EXEC_ENV, FaceswapError, ScriptExecutor


def _gui_args(**overrides) -> Namespace:
    """Return a minimal GUI argument namespace."""
    values = {
        "gui_shell": None,
        "no_gui_exec": False,
        "redirect_gui": False,
        "loglevel": "INFO",
        "logfile": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _parser() -> argparse.ArgumentParser:
    """Return a parser containing the GUI-specific options."""
    parser = argparse.ArgumentParser()
    for option in GuiArgs.get_argument_list():
        kwargs = {key: value for key, value in option.items() if key not in ("opts", "group")}
        parser.add_argument(*option["opts"], **kwargs)
    return parser


def test_gui_shell_defaults_to_tk_when_unset(monkeypatch) -> None:
    """The GUI shell selector should keep Tk as the default."""
    monkeypatch.delenv(GUI_SHELL_ENV, raising=False)

    assert ScriptExecutor._resolve_gui_shell(_gui_args()) == "tk"


def test_gui_shell_accepts_shell_qt_and_qt_alias() -> None:
    """Both the explicit shell selector and shortcut should resolve to Qt."""
    parser = _parser()

    assert parser.parse_args(["--shell", "qt"]).gui_shell == "qt"
    assert parser.parse_args(["--qt"]).gui_shell == "qt"


def test_gui_shell_reads_environment_when_cli_is_unset(monkeypatch) -> None:
    """Environment selector should be used when CLI selector is omitted."""
    monkeypatch.setenv(GUI_SHELL_ENV, "qt")

    assert ScriptExecutor._resolve_gui_shell(_gui_args()) == "qt"


def test_gui_shell_cli_overrides_environment(monkeypatch) -> None:
    """CLI selector should override the environment shell selector."""
    monkeypatch.setenv(GUI_SHELL_ENV, "tk")

    assert ScriptExecutor._resolve_gui_shell(_gui_args(gui_shell="qt")) == "qt"


def test_gui_shell_rejects_invalid_environment(monkeypatch) -> None:
    """Invalid environment selectors should fail clearly."""
    monkeypatch.setenv(GUI_SHELL_ENV, "bad")

    with pytest.raises(FaceswapError, match="Invalid GUI shell"):
        ScriptExecutor._resolve_gui_shell(_gui_args())


def test_qt_no_exec_resolves_from_cli_and_environment(monkeypatch) -> None:
    """Qt no-exec smoke mode should resolve from CLI or environment."""
    monkeypatch.delenv(QT_NO_EXEC_ENV, raising=False)
    assert ScriptExecutor._resolve_gui_no_exec(_gui_args(no_gui_exec=True)) is True
    assert ScriptExecutor._resolve_gui_no_exec(_gui_args(no_gui_exec=False)) is False

    monkeypatch.setenv(QT_NO_EXEC_ENV, "yes")
    assert ScriptExecutor._resolve_gui_no_exec(_gui_args(no_gui_exec=False)) is True


def test_qt_no_exec_checks_pyside6_but_skips_display(monkeypatch) -> None:
    """Qt no-exec smoke launch should not require a display after PySide6 check."""
    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access
    executor._gui_no_exec = True  # pylint:disable=protected-access
    calls: list[str] = []
    monkeypatch.setattr(executor, "_test_pyside6", lambda: calls.append("qt"))
    monkeypatch.setattr(executor, "_check_display", lambda: calls.append("display"))

    executor._test_for_gui()  # pylint:disable=protected-access

    assert calls == ["qt"]


def test_qt_normal_launch_checks_display(monkeypatch) -> None:
    """Normal display-backed Qt launch should still run display preflight."""
    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access
    executor._gui_no_exec = False  # pylint:disable=protected-access
    calls: list[str] = []
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setattr(executor, "_test_pyside6", lambda: calls.append("qt"))
    monkeypatch.setattr(executor, "_check_display", lambda: calls.append("display"))

    executor._test_for_gui()  # pylint:disable=protected-access

    assert calls == ["qt", "display"]


def test_qt_offscreen_launch_skips_display(monkeypatch) -> None:
    """Qt offscreen launch should remain headless and skip DISPLAY preflight."""
    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access
    executor._gui_no_exec = False  # pylint:disable=protected-access
    calls: list[str] = []
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(executor, "_test_pyside6", lambda: calls.append("qt"))
    monkeypatch.setattr(executor, "_check_display", lambda: calls.append("display"))

    executor._test_for_gui()  # pylint:disable=protected-access

    assert calls == ["qt"]


def test_tk_launch_keeps_tk_and_display_preflight(monkeypatch) -> None:
    """Tk default path should keep Tkinter and display preflight."""
    executor = ScriptExecutor("gui")
    executor._gui_shell = "tk"  # pylint:disable=protected-access
    executor._gui_no_exec = True  # pylint:disable=protected-access
    calls: list[str] = []
    monkeypatch.setattr(executor, "_test_tkinter", lambda: calls.append("tk"))
    monkeypatch.setattr(executor, "_check_display", lambda: calls.append("display"))

    executor._test_for_gui()  # pylint:disable=protected-access

    assert calls == ["tk", "display"]


def test_missing_pyside6_raises_faceswap_error(monkeypatch) -> None:
    """Missing PySide6 should fail with the expected launch error."""
    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access

    def fake_import(name, *args, **kwargs):
        if name == "PySide6.QtWidgets":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(FaceswapError, match="PySide6 not found"):
        executor._test_for_gui()  # pylint:disable=protected-access


def test_no_display_raises_for_normal_qt_launch(monkeypatch) -> None:
    """Normal Qt launch should remain display-gated on headless non-Windows hosts."""
    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access
    executor._gui_no_exec = False  # pylint:disable=protected-access
    monkeypatch.setattr(executor, "_test_pyside6", lambda: None)
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setattr("platform.system", lambda: "Linux")

    with pytest.raises(FaceswapError, match="No display detected"):
        executor._test_for_gui()  # pylint:disable=protected-access


def test_import_script_routes_qt_gui_to_qt_module(monkeypatch) -> None:
    """Qt shell selection should import scripts.gui_qt."""
    imported: list[str] = []

    class _Module:
        Gui = object

    def fake_import_module(module_name: str):
        imported.append(module_name)
        return _Module

    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access
    monkeypatch.setattr(executor, "_set_environment_variables", lambda: None)
    monkeypatch.setattr(executor, "_test_for_torch_version", lambda: None)
    monkeypatch.setattr(executor, "_test_for_gui", lambda: None)
    monkeypatch.setattr("lib.cli.launcher.import_module", fake_import_module)
    monkeypatch.setattr(sys, "argv", ["faceswap.py"])

    assert executor._import_script() is object  # pylint:disable=protected-access
    assert imported == ["scripts.gui_qt"]


def test_import_script_routes_tk_gui_to_tk_module(monkeypatch) -> None:
    """Tk shell selection should keep the existing scripts.gui import path."""
    imported: list[str] = []

    class _Module:
        Gui = object

    def fake_import_module(module_name: str):
        imported.append(module_name)
        return _Module

    executor = ScriptExecutor("gui")
    executor._gui_shell = "tk"  # pylint:disable=protected-access
    monkeypatch.setattr(executor, "_set_environment_variables", lambda: None)
    monkeypatch.setattr(executor, "_test_for_torch_version", lambda: None)
    monkeypatch.setattr(executor, "_test_for_gui", lambda: None)
    monkeypatch.setattr("lib.cli.launcher.import_module", fake_import_module)
    monkeypatch.setattr(sys, "argv", ["faceswap.py"])

    assert executor._import_script() is object  # pylint:disable=protected-access
    assert imported == ["scripts.gui"]
