#!/usr/bin/env python3
"""Tests for Qt Settings dialog metadata routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QLabel, QPushButton, QTreeWidget  # noqa:E402

from lib.gui.qt_shell.settings_dialog import SettingsDialog  # noqa:E402


def _config_provider():
    """Return fake Faceswap config metadata."""
    return {
        "convert": SimpleNamespace(
            sections={
                "global": SimpleNamespace(helptext="Convert global help", options={"a": object()}),
                "writer.output": SimpleNamespace(
                    helptext="Writer output help",
                    options={"b": object(), "c": object()},
                ),
            }
        ),
        "extract": SimpleNamespace(
            sections={
                "global": SimpleNamespace(helptext="Extract global help", options={}),
                "detect.face": SimpleNamespace(helptext="Detect face help", options={"d": object()}),
            }
        ),
        "train": SimpleNamespace(
            sections={"global": SimpleNamespace(helptext="Train global help", options={})}
        ),
        "gui": SimpleNamespace(
            sections={"global": SimpleNamespace(helptext="GUI help", options={"theme": object()})}
        ),
    }


def _tree(dialog: SettingsDialog) -> QTreeWidget:
    """Return the dialog tree widget."""
    tree = dialog.findChild(QTreeWidget, "qt-shell-settings-tree")
    assert tree is not None
    return tree


def _label(dialog: SettingsDialog, name: str) -> QLabel:
    """Return a named label."""
    label = dialog.findChild(QLabel, f"qt-shell-settings-{name}")
    assert label is not None
    return label


def test_settings_dialog_orders_categories_like_tk(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Settings tree should prioritize Extract, Train, Convert before other categories."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    tree = _tree(dialog)

    labels = [tree.topLevelItem(index).text(0) for index in range(tree.topLevelItemCount())]

    assert labels == ["Extract", "Train", "Convert", "Gui"]


def test_settings_dialog_exposes_section_metadata(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Settings sections should include identifiers, help text and option counts."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)

    sections = {section.identifier: section for section in dialog.sections}

    assert sections["extract"].helptext == "Extract global help"
    assert sections["extract"].option_count == 0
    assert sections["convert|writer.output"].label == "Output"
    assert sections["convert|writer.output"].option_count == 2


def test_settings_dialog_selects_requested_category(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Opening settings for a command should select that top-level category."""
    dialog = SettingsDialog(section="train", config_provider=_config_provider)
    qtbot.addWidget(dialog)

    assert dialog.selected_identifier == "train"
    assert _label(dialog, "header").text() == "Train Settings"


def test_settings_dialog_selection_updates_summary(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Selecting a child section should update the header and summary text."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    tree = _tree(dialog)
    convert = tree.topLevelItem(2)
    child = convert.child(0)

    tree.setCurrentItem(child)

    assert dialog.selected_identifier == "convert|writer.output"
    assert _label(dialog, "header").text() == "Convert - Writer.Output Settings"
    assert "Writer output help" in _label(dialog, "summary").text()
    assert "Options available: 2" in _label(dialog, "summary").text()


def test_settings_dialog_footer_actions_update_status(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Save/reset footer actions should route to the selected settings section."""
    dialog = SettingsDialog(section="convert", config_provider=_config_provider)
    qtbot.addWidget(dialog)
    save = dialog.findChild(QPushButton, "qt-shell-settings-save")
    reset_all = dialog.findChild(QPushButton, "qt-shell-settings-reset-all")
    assert save is not None
    assert reset_all is not None

    save.click()
    assert _label(dialog, "status").text() == "Save: convert"

    reset_all.click()
    assert _label(dialog, "status").text() == "Reset All: convert"
