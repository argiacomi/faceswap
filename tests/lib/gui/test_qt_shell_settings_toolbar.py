#!/usr/bin/env python3
"""Tests for Qt taskbar settings buttons (Extract/Train/Convert)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QMainWindow, QToolBar  # noqa:E402


def _main_window(qtbot, monkeypatch, tmp_path: Path):  # type:ignore[no-untyped-def]
    """Return a MainWindow with a deterministic schema."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec("faceswap", "extract", (OptionSpec("Input", "-i"),)),
            CommandSpec("faceswap", "train", (OptionSpec("Model", "-m"),)),
            CommandSpec("faceswap", "convert", (OptionSpec("Output", "-o"),)),
        )
    )
    window = MainWindow(schema)
    qtbot.addWidget(window)
    return window


def _toolbar(window: QMainWindow) -> QToolBar:
    """Return the main toolbar."""
    toolbar = window.findChild(QToolBar, "qt-shell-toolbar")
    assert toolbar is not None, "Main toolbar not found"
    return toolbar


def test_toolbar_exposes_three_settings_actions(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The taskbar must include Extract, Train, and Convert settings actions."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    toolbar = _toolbar(window)

    object_names = [action.objectName() for action in toolbar.actions()]

    for name in ("extract", "train", "convert"):
        assert f"qt-shell-toolbar-settings-{name}" in object_names


def test_toolbar_settings_actions_use_tk_parity_tooltips(
    qtbot, monkeypatch, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """Tooltips should match Tk's 'Configure {Name} settings...' wording."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    toolbar = _toolbar(window)
    actions = {action.objectName(): action for action in toolbar.actions()}

    assert actions["qt-shell-toolbar-settings-extract"].toolTip() == (
        "Configure Extract settings..."
    )
    assert actions["qt-shell-toolbar-settings-train"].toolTip() == ("Configure Train settings...")
    assert actions["qt-shell-toolbar-settings-convert"].toolTip() == (
        "Configure Convert settings..."
    )


def test_toolbar_settings_actions_appear_after_task_group(
    qtbot, monkeypatch, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """Settings buttons must follow the task button group in toolbar order."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    toolbar = _toolbar(window)

    names = [action.objectName() for action in toolbar.actions() if action.objectName()]
    settings_idx = names.index("qt-shell-toolbar-settings-extract")
    task_last_idx = names.index("qt-shell-toolbar-task-reload")

    assert task_last_idx < settings_idx
    assert names[settings_idx : settings_idx + 3] == [
        "qt-shell-toolbar-settings-extract",
        "qt-shell-toolbar-settings-train",
        "qt-shell-toolbar-settings-convert",
    ]


@pytest.mark.parametrize("name", ("extract", "train", "convert"))
def test_toolbar_settings_actions_route_to_dialog(
    qtbot, monkeypatch, tmp_path: Path, name: str
) -> None:  # type:ignore[no-untyped-def]
    """Triggering a settings button should open the dialog for that section."""
    import lib.gui.qt_shell.main_window as main_window_module

    created: list[object] = []

    class _DialogDouble:
        def __init__(self, section=None, parent=None) -> None:  # type:ignore[no-untyped-def]
            self.section = section
            self.parent = parent
            self.show_count = 0
            self.selected: list[str] = []
            created.append(self)

        def show(self) -> None:
            self.show_count += 1

        def raise_(self) -> None:
            pass

        def activateWindow(self) -> None:  # noqa:N802
            pass

        def _select_initial_section(self, section: str) -> None:
            self.selected.append(section)

    monkeypatch.setattr(main_window_module, "SettingsDialog", _DialogDouble)
    window = _main_window(qtbot, monkeypatch, tmp_path)
    toolbar = _toolbar(window)
    action = next(
        a for a in toolbar.actions() if a.objectName() == f"qt-shell-toolbar-settings-{name}"
    )

    action.trigger()

    assert len(created) == 1
    assert created[0].section == name
    assert created[0].show_count == 1


def test_toolbar_project_group_matches_tk_parity(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The project-group portion of the toolbar must match Tk's button set."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    toolbar = _toolbar(window)
    names = [action.objectName() for action in toolbar.actions() if action.objectName()]

    project_group = [
        "qt-shell-toolbar-new",
        "qt-shell-toolbar-open",
        "qt-shell-toolbar-save",
        "qt-shell-toolbar-save-as",
        "qt-shell-toolbar-reload",
    ]
    for index, expected in enumerate(project_group):
        assert names[index] == expected, (
            f"Toolbar position {index} should be {expected!r}, got {names[index]!r}"
        )


def test_toolbar_actions_carry_icons(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Every toolbar action should expose a non-null icon for visual Tk parity."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    toolbar = _toolbar(window)

    actions_with_object_names = [a for a in toolbar.actions() if a.objectName()]
    assert actions_with_object_names, "Toolbar should expose object-named actions"
    missing = [a.objectName() for a in actions_with_object_names if a.icon().isNull()]
    assert not missing, f"Toolbar actions missing icons: {missing}"
