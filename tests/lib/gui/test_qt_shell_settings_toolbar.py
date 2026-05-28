#!/usr/bin/env python3
"""Tests for Qt taskbar settings buttons (Extract/Train/Convert)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

from tests.lib.gui._qt_helpers import main_window, toolbar  # noqa:E402


_PROJECT_GROUP = [
    "qt-shell-toolbar-new",
    "qt-shell-toolbar-open",
    "qt-shell-toolbar-save",
    "qt-shell-toolbar-save-as",
    "qt-shell-toolbar-reload",
]
_TASK_GROUP = [
    "qt-shell-toolbar-task-open",
    "qt-shell-toolbar-task-save",
    "qt-shell-toolbar-task-save-as",
    "qt-shell-toolbar-task-reset",
    "qt-shell-toolbar-task-reload",
]
_SETTINGS_GROUP = [
    "qt-shell-toolbar-settings-extract",
    "qt-shell-toolbar-settings-train",
    "qt-shell-toolbar-settings-convert",
]
_SETTINGS_TOOLTIPS = {
    "qt-shell-toolbar-settings-extract": "Configure Extract settings...",
    "qt-shell-toolbar-settings-train": "Configure Train settings...",
    "qt-shell-toolbar-settings-convert": "Configure Convert settings...",
}


def test_toolbar_contract(qtbot, monkeypatch, tmp_path: Path) -> None:
    """Toolbar exposes the expected ordered actions, settings metadata and icons."""
    window = main_window(qtbot, monkeypatch, tmp_path)
    actions = [action for action in toolbar(window).actions() if action.objectName()]
    names = [action.objectName() for action in actions]

    assert names[: len(_PROJECT_GROUP)] == _PROJECT_GROUP
    task_start = names.index(_TASK_GROUP[0])
    settings_start = names.index(_SETTINGS_GROUP[0])
    assert names[task_start : task_start + len(_TASK_GROUP)] == _TASK_GROUP
    assert names[settings_start : settings_start + len(_SETTINGS_GROUP)] == _SETTINGS_GROUP
    assert task_start + len(_TASK_GROUP) <= settings_start

    actions_by_name = {action.objectName(): action for action in actions}
    assert {
        name: actions_by_name[name].toolTip() for name in _SETTINGS_GROUP
    } == _SETTINGS_TOOLTIPS

    missing_icons = [action.objectName() for action in actions if action.icon().isNull()]
    assert not missing_icons, f"Toolbar actions missing icons: {missing_icons}"


def test_toolbar_settings_actions_route_to_dialog_sections(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    """Settings buttons open and reuse the settings dialog for the requested section."""
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
    window = main_window(qtbot, monkeypatch, tmp_path)
    actions = {
        action.objectName(): action
        for action in toolbar(window).actions()
        if action.objectName() in _SETTINGS_GROUP
    }

    for name in ("extract", "train", "convert"):
        actions[f"qt-shell-toolbar-settings-{name}"].trigger()

    assert len(created) == 1
    assert created[0].section == "extract"
    assert created[0].selected == ["train", "convert"]
    assert created[0].show_count == 3
