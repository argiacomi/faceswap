#!/usr/bin/env python3
"""Tests for Qt Settings dialog config parity."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import (  # noqa:E402
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QSlider,
)

from lib.gui.qt_shell.settings_dialog import SettingsDialog  # noqa:E402
from tests.lib.gui._settings_fakes import (  # noqa:E402
    FakeConfig,
    FakeOption,
    FakeSerializer,
)
from tests.lib.gui._settings_fakes import (
    find_item as _find_item,
)
from tests.lib.gui._settings_fakes import (
    label as _label,
)
from tests.lib.gui._settings_fakes import (
    renderer as _renderer,
)
from tests.lib.gui._settings_fakes import (
    section as _section,
)
from tests.lib.gui._settings_fakes import (
    tree as _tree,
)


def _config_provider():
    """Return fake Faceswap config metadata and values."""
    return {
        "convert": FakeConfig(
            {
                "global": _section(
                    "Convert global help",
                    output=FakeOption(str, "png", "png", choices=["png", "jpg"]),
                ),
                "writer.output": _section(
                    "Writer output help",
                    enabled=FakeOption(bool, True, True),
                    scale=FakeOption(float, 0.5, 0.5, min_max=(0.0, 1.0), rounding=2),
                    modes=FakeOption(list, ["fast"], ["fast"], choices=["fast", "safe"]),
                ),
            }
        ),
        "extract": FakeConfig(
            {
                "global": _section("Extract global help"),
                "detect.face": _section(
                    "Detect face help",
                    detector=FakeOption(
                        str, "cv2", "cv2", choices=["cv2", "s3fd"], gui_radio=True
                    ),
                ),
                "detect.mask": _section(
                    "Detect mask help",
                    threshold=FakeOption(int, 5, 5, min_max=(0, 10), rounding=1),
                ),
            }
        ),
        "train": FakeConfig({"global": _section("Train global help")}),
        "gui": FakeConfig(
            {
                "global": _section(
                    "GUI help",
                    theme=FakeOption(str, "default", "default"),
                )
            }
        ),
    }


def test_settings_dialog_orders_categories_like_tk(qtbot) -> None:
    """Settings tree should prioritize Extract, Train, Convert before other categories."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    tree = _tree(dialog)

    labels = [tree.topLevelItem(index).text(0) for index in range(tree.topLevelItemCount())]  # type: ignore[union-attr]

    assert labels == ["Extract", "Train", "Convert", "Gui"]


def test_settings_dialog_exposes_section_metadata(qtbot) -> None:
    """Settings sections should include identifiers, help text and option counts."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)

    sections = {section.identifier: section for section in dialog.sections}

    assert sections["extract"].helptext == "Extract global help"
    assert sections["extract"].option_count == 0
    assert sections["convert|writer|output"].label == "Output"
    assert sections["convert|writer|output"].option_count == 3


def test_settings_dialog_selects_requested_category(qtbot) -> None:
    """Opening settings for a command should select that top-level category."""
    dialog = SettingsDialog(section="train", config_provider=_config_provider)
    qtbot.addWidget(dialog)

    assert dialog.selected_identifier == "train"
    assert _label(dialog, "header").text() == "Train Settings"
    assert _label(dialog, "page-header").text() == "Global"


def test_settings_dialog_uses_nested_tree_and_links_pages(qtbot) -> None:
    """Selecting a parent section without options should show child links."""
    dialog = SettingsDialog(config_provider=_config_provider)
    qtbot.addWidget(dialog)
    tree = _tree(dialog)
    detect = _find_item(tree, "extract|detect")
    assert detect is not None

    tree.setCurrentItem(detect)
    links = dialog.findChildren(QPushButton, "qt-shell-settings-link")

    assert {button.text() for button in links} == {"Face", "Mask"}

    links_by_text = {button.text(): button for button in links}
    links_by_text["Face"].click()

    assert dialog.selected_identifier == "extract|detect|face"
    assert _label(dialog, "page-header").text() == "Detect - Face"


def test_settings_dialog_renders_real_option_widgets(qtbot) -> None:
    """Config options should render through the reusable Qt option renderer."""
    dialog = SettingsDialog(section="convert|writer|output", config_provider=_config_provider)
    qtbot.addWidget(dialog)
    renderer = _renderer(dialog)

    enabled = renderer.widget_for_switch("enabled")
    scale = renderer.widget_for_switch("scale")
    modes = renderer.widget_for_switch("modes")

    assert isinstance(enabled, QCheckBox)
    assert scale.findChild(QSlider) is not None
    assert {box.text() for box in modes.findChildren(QCheckBox)} == {"fast", "safe"}


def test_settings_dialog_renders_ensemble_paths_with_file_browsers(qtbot) -> None:
    """Ensemble setup/weights paths should open file selectors in the Qt settings UI."""
    configs = {
        "extract": FakeConfig(
            {
                "align.ensemble": _section(
                    "Ensemble help",
                    setup_path=FakeOption(str, "", "", group="landmark_ensemble"),
                    weights_path=FakeOption(str, "", "", group="landmark_ensemble"),
                )
            }
        )
    }
    dialog = SettingsDialog(
        section="extract|align|ensemble",
        config_provider=lambda: configs,
    )
    qtbot.addWidget(dialog)
    renderer = _renderer(dialog)

    assert isinstance(renderer.widget_for_switch("setup_path"), QLineEdit)
    assert isinstance(renderer.widget_for_switch("weights_path"), QLineEdit)

    file_buttons = renderer.findChildren(QPushButton, "qt-shell-browser-file")
    assert len(file_buttons) == 2
    assert all("JSON files" in spec.file_filter for spec in renderer._specs)


def test_settings_dialog_page_save_and_reset_update_config(qtbot) -> None:
    """Page save/reset should act on the selected page's live config state."""
    configs = _config_provider()
    dialog = SettingsDialog(
        section="convert",
        config_provider=lambda: configs,
    )
    qtbot.addWidget(dialog)
    renderer = _renderer(dialog)
    output = renderer.widget_for_switch("output")
    assert isinstance(output, QComboBox)

    output.setCurrentText("jpg")
    save = dialog.findChild(QPushButton, "qt-shell-settings-save")
    assert save is not None
    save.click()

    assert configs["convert"].sections["global"].options["output"].value == "jpg"
    assert configs["convert"].saves == 1

    output.setCurrentText("png")
    reset = dialog.findChild(QPushButton, "qt-shell-settings-reset")
    assert reset is not None
    reset.click()

    assert output.currentText() == "png"
    assert _label(dialog, "status").property("status") == "success"


def test_settings_dialog_save_all_scopes_to_selected_config(qtbot) -> None:
    """Save All should persist all settings for the selected top-level config only."""
    configs = _config_provider()
    dialog = SettingsDialog(section="gui", config_provider=lambda: configs)
    qtbot.addWidget(dialog)
    renderer = _renderer(dialog)
    theme = renderer.widget_for_switch("theme")
    assert isinstance(theme, QLineEdit)

    theme.setText("dark")
    save_all = dialog.findChild(QPushButton, "qt-shell-settings-save-all")
    assert save_all is not None
    save_all.click()

    assert configs["gui"].sections["global"].options["theme"].value == "dark"
    assert configs["gui"].saves == 1
    assert configs["convert"].saves == 0


def test_settings_dialog_presets_save_and_load_current_page(qtbot, tmp_path, monkeypatch) -> None:
    """Preset load/save should include metadata and update the current page only."""
    serializer = FakeSerializer()
    dialog = SettingsDialog(
        section="gui",
        config_provider=_config_provider,
        serializer=serializer,
        preset_base_path=tmp_path,
    )
    qtbot.addWidget(dialog)
    renderer = _renderer(dialog)
    theme = renderer.widget_for_switch("theme")
    assert isinstance(theme, QLineEdit)
    save_path = tmp_path / "theme.json"
    monkeypatch.setattr(
        "lib.gui.qt_shell.settings_dialog.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(save_path), ""),
    )

    theme.setText("dark")
    save = dialog.findChild(QPushButton, "qt-shell-settings-save-preset")
    assert save is not None
    save.click()

    assert serializer.saved == (
        str(save_path),
        {"theme": "dark", "__filetype": "faceswap_preset", "__section": "gui|global"},
    )

    serializer.to_load = {
        "theme": "light",
        "__filetype": "faceswap_preset",
        "__section": "gui|global",
    }
    monkeypatch.setattr(
        "lib.gui.qt_shell.settings_dialog.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(save_path), ""),
    )
    load = dialog.findChild(QPushButton, "qt-shell-settings-load-preset")
    assert load is not None
    load.click()

    assert theme.text() == "light"


def test_gui_config_save_rebuilds_or_defers(qtbot) -> None:
    """GUI config saves should rebuild immediately unless a task is running."""
    configs = _config_provider()
    rebuilt: list[str] = []
    dialog = SettingsDialog(
        section="gui",
        config_provider=lambda: configs,
        running_task_provider=lambda: False,
        gui_rebuild_callback=lambda: rebuilt.append("now"),
    )
    qtbot.addWidget(dialog)
    theme = _renderer(dialog).widget_for_switch("theme")
    assert isinstance(theme, QLineEdit)

    theme.setText("dark")
    dialog.save(page_only=True)

    assert rebuilt == ["now"]

    deferred = SettingsDialog(
        section="gui",
        config_provider=_config_provider,
        running_task_provider=lambda: True,
        gui_rebuild_callback=lambda: rebuilt.append("deferred"),
    )
    qtbot.addWidget(deferred)
    deferred_theme = _renderer(deferred).widget_for_switch("theme")
    assert isinstance(deferred_theme, QLineEdit)

    deferred_theme.setText("dark")
    deferred.save(page_only=True)

    assert rebuilt == ["now"]
    assert _label(deferred, "status").property("status") == "warning"
