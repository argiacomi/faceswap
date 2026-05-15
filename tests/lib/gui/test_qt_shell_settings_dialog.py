#!/usr/bin/env python3
"""Tests for Qt Settings dialog config parity."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa:E402
from PySide6.QtWidgets import (  # noqa:E402
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTreeWidget,
)

from lib.gui.qt_shell.command_panel import OptionsFormRenderer  # noqa:E402
from lib.gui.qt_shell.settings_dialog import SettingsDialog  # noqa:E402


@dataclass
class FakeOption:
    """Small ConfigItem-compatible fake."""

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
        """Set the fake option value."""
        if self.datatype is list and isinstance(value, str):
            self.value = value.split()
        else:
            self.value = value


class FakeConfig(SimpleNamespace):
    """Small FaceswapConfig-compatible fake."""

    def __init__(self, sections):  # type:ignore[no-untyped-def]
        super().__init__(sections=sections, saves=0)

    def save_config(self) -> None:
        """Record config saves."""
        self.saves += 1


class FakeSerializer:
    """Preset serializer test double."""

    def __init__(self) -> None:
        self.saved: tuple[str, dict[str, object]] | None = None
        self.to_load: dict[str, object] = {}

    def save(self, filename: str, data: dict[str, object]) -> None:
        """Record saved preset data."""
        self.saved = (filename, data)

    def load(self, _filename: str) -> dict[str, object]:
        """Return configured preset data."""
        return self.to_load


def _section(helptext: str, **options: FakeOption) -> SimpleNamespace:
    """Return fake config section."""
    return SimpleNamespace(helptext=helptext, options=options)


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


def _renderer(dialog: SettingsDialog) -> OptionsFormRenderer:
    """Return the visible settings option renderer."""
    renderer = dialog.findChild(OptionsFormRenderer, "qt-shell-settings-options")
    assert renderer is not None
    return renderer


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
    assert sections["convert|writer|output"].label == "Output"
    assert sections["convert|writer|output"].option_count == 3


def test_settings_dialog_selects_requested_category(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Opening settings for a command should select that top-level category."""
    dialog = SettingsDialog(section="train", config_provider=_config_provider)
    qtbot.addWidget(dialog)

    assert dialog.selected_identifier == "train"
    assert _label(dialog, "header").text() == "Train Settings"
    assert _label(dialog, "page-header").text() == "Global"


def test_settings_dialog_uses_nested_tree_and_links_pages(qtbot) -> None:  # type:ignore[no-untyped-def]
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


def test_settings_dialog_renders_real_option_widgets(qtbot) -> None:  # type:ignore[no-untyped-def]
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


def test_settings_dialog_page_save_and_reset_update_config(qtbot) -> None:  # type:ignore[no-untyped-def]
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


def test_settings_dialog_save_all_scopes_to_selected_config(qtbot) -> None:  # type:ignore[no-untyped-def]
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


def test_settings_dialog_presets_save_and_load_current_page(qtbot, tmp_path, monkeypatch) -> None:  # type:ignore[no-untyped-def]
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


def test_gui_config_save_rebuilds_or_defers(qtbot) -> None:  # type:ignore[no-untyped-def]
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
