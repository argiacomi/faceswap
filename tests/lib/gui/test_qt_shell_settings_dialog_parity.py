#!/usr/bin/env python3
"""Focused parity tests for the Qt settings dialog chrome and routing."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from lib.gui.qt_shell.settings_dialog import SettingsDialog  # noqa:E402
from tests.lib.gui._settings_fakes import (  # noqa:E402
    FakeConfig,
    FakeOption,
    button,
    filter_edit,
    find_item,
    section,
    tree,
)


def _config_provider():
    """Deep-nested fake config used to exercise filter + routing parity."""
    return {
        "extract": FakeConfig(
            {
                "global": section("Extract global help"),
                "detect.face.s3fd": section(
                    "Nested detector help",
                    confidence=FakeOption(float, 0.6, 0.6, min_max=(0.0, 1.0), rounding=2),
                ),
            }
        ),
        "train": FakeConfig({"global": section("Train global help")}),
        "convert": FakeConfig(
            {
                "global": section(
                    "Convert global help",
                    output=FakeOption(str, "png", "png", choices=["png", "jpg"]),
                ),
            }
        ),
    }


def test_settings_dialog_preserves_full_nested_section_route(qtbot) -> None:
    """Deep plugin identifiers should resolve to their full config section."""
    dialog = SettingsDialog(
        section="extract|detect|face|s3fd",
        config_provider=_config_provider,
    )
    qtbot.addWidget(dialog)

    assert dialog.selected_identifier == "extract|detect|face|s3fd"
    assert dialog.displayed_key == "extract|detect|face|s3fd"


def test_settings_dialog_disables_page_actions_for_navigation_nodes(qtbot) -> None:
    """Parent navigation pages should not expose save/reset/preset actions."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    detect = find_item(tree(dialog), "extract|detect")
    assert detect is not None

    tree(dialog).setCurrentItem(detect)

    assert not button(dialog, "save").isEnabled()
    assert not button(dialog, "reset").isEnabled()
    assert not button(dialog, "save-preset").isEnabled()
    assert not button(dialog, "load-preset").isEnabled()


def test_settings_dialog_footer_and_preset_buttons_have_tk_parity_icons(qtbot) -> None:
    """Action buttons should reuse the legacy icon cache where available."""
    dialog = SettingsDialog(section="convert", config_provider=_config_provider)
    qtbot.addWidget(dialog)

    for name in ("save", "reset", "save-all", "reset-all", "save-preset", "load-preset"):
        assert not button(dialog, name).icon().isNull()


def test_settings_dialog_filter_keeps_matching_parent_path_visible(qtbot) -> None:
    """Filtering should keep matched leaf nodes and their parent path scannable."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    nav = tree(dialog)
    extract = find_item(nav, "extract")
    detect = find_item(nav, "extract|detect")
    s3fd = find_item(nav, "extract|detect|face|s3fd")
    convert = find_item(nav, "convert")
    assert extract is not None
    assert detect is not None
    assert s3fd is not None
    assert convert is not None

    filter_edit(dialog).setText("s3fd")

    assert not extract.isHidden()
    assert not detect.isHidden()
    assert not s3fd.isHidden()
    assert convert.isHidden()


def test_settings_dialog_filter_reselects_visible_page_when_current_is_hidden(qtbot) -> None:
    """Filtering away the selected item should move selection to a visible page."""
    dialog = SettingsDialog(section="convert", config_provider=_config_provider)
    qtbot.addWidget(dialog)

    filter_edit(dialog).setText("s3fd")

    assert dialog.selected_identifier == "extract|detect|face|s3fd"
    assert dialog.displayed_key == "extract|detect|face|s3fd"
