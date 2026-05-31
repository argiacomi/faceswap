#!/usr/bin/env python3
"""Shared fakes and widget lookups for the Qt settings-dialog test suite.

``test_qt_shell_settings_dialog.py`` and ``test_qt_shell_settings_dialog_parity.py``
used to keep their own copies of ``FakeOption``/``FakeConfig`` and the
``_tree``/``_find_item``/``_button`` widget lookups.  The fakes had drifted in
small ways (the main file taught ``FakeOption.set`` to split list-typed
strings; parity did not), so consolidating here both removes the duplication
and locks the union behaviour in one place.

Each test file keeps its own ``_config_provider`` callable because the two
suites exercise different fake config layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QTreeWidget

    from lib.gui.qt_shell.command_panel import OptionsFormRenderer
    from lib.gui.qt_shell.settings_dialog import SettingsDialog


@dataclass
class FakeOption:
    """``ConfigItem``-compatible fake used by the settings-dialog tests."""

    datatype: type
    default: object
    value: object
    group: str = "General"
    choices: list[str] | str | None = None
    gui_radio: bool = False
    min_max: tuple[int | float, int | float] | None = None
    rounding: int = -1
    helptext: str = "Option help"

    def __post_init__(self) -> None:
        if self.choices is None:
            self.choices = []

    def set(self, value: object) -> None:
        """Record a value, splitting whitespace-delimited strings for list options."""
        if self.datatype is list and isinstance(value, str):
            self.value = value.split()
        else:
            self.value = value


class FakeConfig(SimpleNamespace):
    """``FaceswapConfig``-compatible fake that counts saves."""

    def __init__(self, sections):
        super().__init__(sections=sections, saves=0)

    def save_config(self) -> None:
        self.saves += 1


class FakeSerializer:
    """Preset serializer test double used by the main settings dialog suite."""

    def __init__(self) -> None:
        self.saved: tuple[str, dict[str, object]] | None = None
        self.to_load: dict[str, object] = {}

    def save(self, filename: str, data: dict[str, object]) -> None:
        self.saved = (filename, data)

    def load(self, _filename: str) -> dict[str, object]:
        return self.to_load


def section(helptext: str, **options: FakeOption) -> SimpleNamespace:
    """Return a fake config section with the given help text and options."""
    return SimpleNamespace(helptext=helptext, options=options)


# ---------------------------------------------------------------------------
# Widget lookups - each asserts the widget exists to fail fast on UI drift.
# ---------------------------------------------------------------------------


def tree(dialog: SettingsDialog) -> QTreeWidget:
    """Return the settings dialog's navigation tree."""
    from PySide6.QtWidgets import QTreeWidget

    widget = dialog.findChild(QTreeWidget, "qt-shell-settings-tree")
    assert widget is not None, "settings tree widget not found"
    return widget  # type: ignore[no-any-return]


def label(dialog: SettingsDialog, name: str) -> QLabel:
    """Return a named ``qt-shell-settings-<name>`` label."""
    from PySide6.QtWidgets import QLabel

    widget = dialog.findChild(QLabel, f"qt-shell-settings-{name}")
    assert widget is not None, f"label {name!r} not found"
    return widget  # type: ignore[no-any-return]


def button(dialog: SettingsDialog, name: str) -> QPushButton:
    """Return a named ``qt-shell-settings-<name>`` button."""
    from PySide6.QtWidgets import QPushButton

    widget = dialog.findChild(QPushButton, f"qt-shell-settings-{name}")
    assert widget is not None, f"button {name!r} not found"
    return widget  # type: ignore[no-any-return]


def renderer(dialog: SettingsDialog) -> OptionsFormRenderer:
    """Return the visible settings option renderer."""
    from lib.gui.qt_shell.command_panel import OptionsFormRenderer

    widget = dialog.findChild(OptionsFormRenderer, "qt-shell-settings-options")
    assert widget is not None, "settings option renderer not found"
    return widget  # type: ignore[no-any-return]


def filter_edit(dialog: SettingsDialog) -> QLineEdit:
    """Return the settings navigation filter input."""
    from PySide6.QtWidgets import QLineEdit

    widget = dialog.findChild(QLineEdit, "qt-shell-settings-filter")
    assert widget is not None, "settings filter widget not found"
    return widget  # type: ignore[no-any-return]


def find_item(tree_widget: QTreeWidget, identifier: str):
    """Return a tree item by its hidden ``UserRole`` identifier, or ``None``."""
    from PySide6.QtCore import Qt

    def walk(item):
        if item.data(0, Qt.UserRole) == identifier:
            return item
        for index in range(item.childCount()):
            found = walk(item.child(index))
            if found is not None:
                return found
        return None

    for index in range(tree_widget.topLevelItemCount()):
        found = walk(tree_widget.topLevelItem(index))
        if found is not None:
            return found
    return None
