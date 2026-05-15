#!/usr/bin/env python3
"""Focused parity tests for the Qt settings dialog chrome and routing."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa:E402
from PySide6.QtWidgets import QLineEdit, QPushButton, QTreeWidget  # noqa:E402

from lib.gui.qt_shell.settings_dialog import SettingsDialog  # noqa:E402


@dataclass
class FakeOption:
    """Small ConfigItem-compatible fake."""

    datatype: type
    default: object
    value: object
    group: str = "General"
    choices: list[str] | None = None
    gui_radio: bool = False
    min_max: tuple[int | float, int | float] | None = None
    rounding: int = -1
    helptext: str = "Option help"

    def __post_init__(self) -> None:
        if self.choices is None:
            self.choices = []

    def set(self, value: object) -> None:
        """Set the fake option value."""
        self.value = value


class FakeConfig(SimpleNamespace):
    """Small FaceswapConfig-compatible fake."""

    def __init__(self, sections):  # type:ignore[no-untyped-def]
        super().__init__(sections=sections, saves=0)

    def save_config(self) -> None:
        """Record config saves."""
        self.saves += 1


def _section(helptext: str, **options: FakeOption) -> SimpleNamespace:
    """Return fake config section."""
    return SimpleNamespace(helptext=helptext, options=options)


def _config_provider():
    """Return fake Faceswap config metadata and values."""
    return {
        "extract": FakeConfig(
            {
                "global": _section("Extract global help"),
                "detect.face.s3fd": _section(
                    "Nested detector help",
                    confidence=FakeOption(float, 0.6, 0.6, min_max=(0.0, 1.0), rounding=2),
                ),
            }
        ),
        "train": FakeConfig({"global": _section("Train global help")}),
        "convert": FakeConfig(
            {
                "global": _section(
                    "Convert global help",
                    output=FakeOption(str, "png", "png", choices=["png", "jpg"]),
                ),
            }
        ),
    }


def _tree(dialog: SettingsDialog) -> QTreeWidget:
    """Return the dialog tree widget."""
    tree = dialog.findChild(QTreeWidget, "qt-shell-settings-tree")
    assert tree is not None
    return tree


def _find_item(tree: QTreeWidget, identifier: str):  # type:ignore[no-untyped-def]
    """Return a tree item for a settings identifier."""

    def walk(item):  # type:ignore[no-untyped-def]
        if item.data(0, Qt.UserRole) == identifier:
            return item
        for index in range(item.childCount()):
            found = walk(item.child(index))
            if found is not None:
                return found
        return None

    for index in range(tree.topLevelItemCount()):
        found = walk(tree.topLevelItem(index))
        if found is not None:
            return found
    return None


def _button(dialog: SettingsDialog, name: str) -> QPushButton:
    """Return a named settings button."""
    button = dialog.findChild(QPushButton, f"qt-shell-settings-{name}")
    assert button is not None
    return button


def _filter(dialog: SettingsDialog) -> QLineEdit:
    """Return the settings navigation filter."""
    widget = dialog.findChild(QLineEdit, "qt-shell-settings-filter")
    assert widget is not None
    return widget


def test_settings_dialog_preserves_full_nested_section_route(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Deep plugin identifiers should resolve to their full config section."""
    dialog = SettingsDialog(
        section="extract|detect|face|s3fd",
        config_provider=_config_provider,
    )
    qtbot.addWidget(dialog)

    assert dialog.selected_identifier == "extract|detect|face|s3fd"
    assert dialog.displayed_key == "extract|detect|face|s3fd"


def test_settings_dialog_disables_page_actions_for_navigation_nodes(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Parent navigation pages should not expose save/reset/preset actions."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    detect = _find_item(_tree(dialog), "extract|detect")
    assert detect is not None

    _tree(dialog).setCurrentItem(detect)

    assert not _button(dialog, "save").isEnabled()
    assert not _button(dialog, "reset").isEnabled()
    assert not _button(dialog, "save-preset").isEnabled()
    assert not _button(dialog, "load-preset").isEnabled()


def test_settings_dialog_footer_and_preset_buttons_have_tk_parity_icons(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Action buttons should reuse the legacy icon cache where available."""
    dialog = SettingsDialog(section="convert", config_provider=_config_provider)
    qtbot.addWidget(dialog)

    for name in ("save", "reset", "save-all", "reset-all", "save-preset", "load-preset"):
        assert not _button(dialog, name).icon().isNull()


def test_settings_dialog_filter_keeps_matching_parent_path_visible(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Filtering should keep matched leaf nodes and their parent path scannable."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    tree = _tree(dialog)
    extract = _find_item(tree, "extract")
    detect = _find_item(tree, "extract|detect")
    s3fd = _find_item(tree, "extract|detect|face|s3fd")
    convert = _find_item(tree, "convert")
    assert extract is not None
    assert detect is not None
    assert s3fd is not None
    assert convert is not None

    _filter(dialog).setText("s3fd")

    assert not extract.isHidden()
    assert not detect.isHidden()
    assert not s3fd.isHidden()
    assert convert.isHidden()


def test_settings_dialog_filter_reselects_visible_page_when_current_is_hidden(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Filtering away the selected item should move selection to a visible page."""
    dialog = SettingsDialog(section="convert", config_provider=_config_provider)
    qtbot.addWidget(dialog)

    _filter(dialog).setText("s3fd")

    assert dialog.selected_identifier == "extract|detect|face|s3fd"
    assert dialog.displayed_key == "extract|detect|face|s3fd"
