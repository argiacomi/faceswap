#!/usr/bin/env python3
"""Qt command option widget tests."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
)

from lib.gui.services.command_builder import CommandBuilder


def _command_panel(*options):  # type:ignore[no-untyped-def]
    """Return a CommandPanel backed by a single extract command."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec

    return CommandPanel(
        CommandSchema(
            (
                CommandSpec(
                    "faceswap",
                    "extract",
                    tuple(options),
                ),
            )
        )
    )


def _option_spec(*args, **kwargs):  # type:ignore[no-untyped-def]
    """Return an OptionSpec without importing Qt shell schema at module import time."""
    from lib.gui.qt_shell.command_schema import OptionSpec

    return OptionSpec(*args, **kwargs)


def test_radio_option_extracts_and_restores_value(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Radio metadata should render exclusive choices and restore stored values."""
    panel = _command_panel(
        _option_spec(
            "Mode",
            "--mode",
            str,
            "one",
            ("one", "two"),
            is_radio=True,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--mode")
    buttons = {button.text(): button for button in widget.findChildren(QRadioButton)}

    buttons["two"].setChecked(True)

    assert panel.command_spec()[2] == {"--mode": "two"}

    panel.set_command("extract", {"--mode": "one"})
    widget = panel.renderer.widget_for_switch("--mode")
    buttons = {button.text(): button for button in widget.findChildren(QRadioButton)}

    assert buttons["one"].isChecked() is True
    assert panel.command_spec()[2] == {"--mode": "one"}


def test_multi_option_extracts_and_restores_values(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Multi-option metadata should emit selected choices and restore stored choices."""
    panel = _command_panel(
        _option_spec(
            "Features",
            "--features",
            str,
            ("fast",),
            ("fast", "safe", "slow"),
            is_multi_option=True,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--features")
    checkboxes = {checkbox.text(): checkbox for checkbox in widget.findChildren(QCheckBox)}

    checkboxes["safe"].setChecked(True)
    checkboxes["fast"].setChecked(False)

    assert panel.command_spec()[2] == {"--features": ["safe"]}

    panel.set_command("extract", {"--features": ["fast", "slow"]})
    widget = panel.renderer.widget_for_switch("--features")
    checkboxes = {checkbox.text(): checkbox for checkbox in widget.findChildren(QCheckBox)}

    assert checkboxes["fast"].isChecked() is True
    assert checkboxes["safe"].isChecked() is False
    assert checkboxes["slow"].isChecked() is True
    assert panel.command_spec()[2] == {"--features": ["fast", "slow"]}


def test_empty_multi_option_skips_cli_emission(qtbot) -> None:  # type:ignore[no-untyped-def]
    """An empty multi-option selection should be omitted from built CLI args."""
    panel = _command_panel(
        _option_spec(
            "Features",
            "--features",
            str,
            "",
            ("fast", "safe"),
            is_multi_option=True,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--features")

    for checkbox in widget.findChildren(QCheckBox):
        checkbox.setChecked(False)

    assert panel.command_spec()[2] == {"--features": ""}
    assert CommandBuilder.build_options(panel.command_spec()[2]) == []


def test_store_false_checkbox_emits_when_unchecked(qtbot) -> None:  # type:ignore[no-untyped-def]
    """store_false checkboxes should emit their switch when unchecked."""
    panel = _command_panel(
        _option_spec(
            "Enabled Feature",
            "--disable-feature",
            bool,
            True,
            action="store_false",
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--disable-feature")
    assert isinstance(widget, QCheckBox)

    assert widget.isChecked() is True
    assert CommandBuilder.build_options(panel.command_spec()[2]) == []

    widget.setChecked(False)

    assert panel.command_spec()[2] == {"--disable-feature": True}
    assert CommandBuilder.build_options(panel.command_spec()[2]) == ["--disable-feature"]

    panel.set_command("extract", {"--disable-feature": True})
    widget = panel.renderer.widget_for_switch("--disable-feature")
    assert isinstance(widget, QCheckBox)

    assert widget.isChecked() is False
    assert CommandBuilder.build_options(panel.command_spec()[2]) == ["--disable-feature"]


def test_int_slider_extracts_restores_and_clamps(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Integer sliders should sync their line edit, restore values and clamp bounds."""
    panel = _command_panel(
        _option_spec(
            "Batch Size",
            "--batch-size",
            int,
            64,
            action="Slider",
            slider_min=1,
            slider_max=256,
            slider_rounding=1,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--batch-size")
    slider = widget.findChild(QSlider)
    line_edit = widget.findChild(QLineEdit)
    assert slider is not None
    assert line_edit is not None

    assert slider.value() == 64
    assert line_edit.text() == "64"
    assert panel.command_spec()[2] == {"--batch-size": 64}

    panel.set_command("extract", {"--batch-size": 128})
    widget = panel.renderer.widget_for_switch("--batch-size")
    slider = widget.findChild(QSlider)
    line_edit = widget.findChild(QLineEdit)
    assert slider is not None
    assert line_edit is not None
    assert slider.value() == 128
    assert line_edit.text() == "128"

    line_edit.setText("300")
    line_edit.editingFinished.emit()
    assert slider.value() == 256
    assert line_edit.text() == "256"
    assert panel.command_spec()[2] == {"--batch-size": 256}

    line_edit.setText("-5")
    line_edit.editingFinished.emit()
    assert slider.value() == 1
    assert line_edit.text() == "1"

    line_edit.setText("bad")
    line_edit.editingFinished.emit()
    assert slider.value() == 64
    assert line_edit.text() == "64"


def test_float_slider_extracts_rounds_and_clamps(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Float sliders should preserve configured precision and clamp bounds."""
    panel = _command_panel(
        _option_spec(
            "Threshold",
            "--threshold",
            float,
            0.5,
            action="Slider",
            slider_min=0.01,
            slider_max=0.99,
            slider_rounding=2,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--threshold")
    slider = widget.findChild(QSlider)
    line_edit = widget.findChild(QLineEdit)
    assert slider is not None
    assert line_edit is not None

    assert slider.value() == 50
    assert line_edit.text() == "0.5"

    line_edit.setText("0.456")
    line_edit.editingFinished.emit()
    assert slider.value() == 46
    assert line_edit.text() == "0.46"
    assert panel.command_spec()[2] == {"--threshold": 0.46}

    line_edit.setText("2.5")
    line_edit.editingFinished.emit()
    assert slider.value() == 99
    assert line_edit.text() == "0.99"

    line_edit.setText("bad")
    line_edit.editingFinished.emit()
    assert slider.value() == 50
    assert line_edit.text() == "0.5"


def test_browse_buttons_get_stable_object_names(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Path options should wrap line edits with browse buttons named by mode."""
    panel = _command_panel(
        _option_spec(
            "Input",
            "-i",
            helptext="Pick paths",
            browser_modes=("folder", "file", "files", "save"),
        )
    )
    qtbot.addWidget(panel)

    assert {button.objectName() for button in panel.renderer.findChildren(QPushButton)} == {
        "qt-shell-browser-folder",
        "qt-shell-browser-file",
        "qt-shell-browser-files",
        "qt-shell-browser-save",
    }
    assert isinstance(panel.renderer.widget_for_switch("-i"), QLineEdit)


def test_group_sections_render_labels_and_containers(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Grouped options should render titled group sections."""
    panel = _command_panel(
        _option_spec("Input", "-i", group="Data"),
        _option_spec("Output", "-o", group="Data"),
        _option_spec("Debug", "--debug", bool, False, group="_master"),
    )
    qtbot.addWidget(panel)
    group_labels = panel.renderer.findChildren(QLabel, "qt-shell-option-group-label")
    groups = panel.renderer.findChildren(QGroupBox, "qt-shell-option-group")

    assert [label.text() for label in group_labels] == ["Data"]
    assert len(groups) == 1
    assert set(panel.renderer.rendered_switches) == {"-i", "-o", "--debug"}
