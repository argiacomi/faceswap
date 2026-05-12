#!/usr/bin/env python3
"""Qt option renderer tests."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QRadioButton,
    QSlider,
)


def test_option_renderer_uses_group_sections(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Grouped options should render inside visually separated sections."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec("Input", "-i", group="Data"),
                    OptionSpec("Output", "-o", group="Data"),
                    OptionSpec("Detector", "-D", group="Plugins"),
                ),
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    groups = panel.renderer.findChildren(QGroupBox)

    assert len(groups) == 2
    assert panel.renderer.rendered_switches == ("-i", "-o", "-D")


def test_command_panel_renders_command_info(qtbot) -> None:  # type:ignore[no-untyped-def]
    """CommandPanel should show CLI command info in the header."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (OptionSpec("Input", "-i"),),
                "Extract faces from sources.\nConfigure plugins in settings.",
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    info = panel.findChild(QLabel, "qt-shell-command-info")

    assert info is not None
    assert "Extract faces from sources." in info.text()
    assert "Configure plugins in settings." in info.text()


def test_radio_widget_extracts_and_restores_values(  # type:ignore[no-untyped-def]
    qtbot,
) -> None:
    """Small radio options should extract and restore exclusive values."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec(
                        "Mode",
                        "--mode",
                        str,
                        "one",
                        ("one", "two"),
                        is_radio=True,
                    ),
                ),
            ),
        )
    )
    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    buttons = {
        button.text(): button
        for button in panel.renderer.widget_for_switch("--mode").findChildren(QRadioButton)
    }

    buttons["two"].setChecked(True)
    assert panel.command_spec()[2] == {"--mode": "two"}

    panel.set_command("extract", {"--mode": "one"})
    assert buttons["one"].isChecked() is True


def test_multi_option_widget_extracts_and_restores_values(  # type:ignore[no-untyped-def]
    qtbot,
) -> None:
    """Multi-options should extract and restore selected values."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec(
                        "Masks",
                        "--masks",
                        str,
                        ("eyes",),
                        ("eyes", "mouth", "nose"),
                        is_multi_option=True,
                    ),
                ),
            ),
        )
    )
    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    boxes = {
        checkbox.text(): checkbox
        for checkbox in panel.renderer.widget_for_switch("--masks").findChildren(QCheckBox)
    }

    boxes["mouth"].setChecked(True)
    assert panel.command_spec()[2] == {"--masks": ["eyes", "mouth"]}

    panel.set_command("extract", {"--masks": ["nose"]})
    assert boxes["nose"].isChecked() is True


def test_empty_multi_option_is_skipped(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Empty multi-option widgets should not emit CLI args."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.services.command_builder import CommandBuilder

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec(
                        "Masks",
                        "--masks",
                        str,
                        "",
                        ("eyes", "mouth"),
                        is_multi_option=True,
                    ),
                ),
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)

    assert panel.command_spec()[2] == {"--masks": ""}
    assert CommandBuilder.build_options(panel.command_spec()[2]) == []


def test_slider_widget_extracts_and_restores_values(  # type:ignore[no-untyped-def]
    qtbot,
) -> None:
    """Slider widgets should extract and restore typed values."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec(
                        "Min Size",
                        "-M",
                        int,
                        5,
                        action="Slider",
                        slider_min=0,
                        slider_max=10,
                        slider_rounding=1,
                    ),
                ),
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("-M")
    slider = widget.findChild(QSlider)
    line_edit = widget.findChild(QLineEdit)
    assert slider is not None
    assert line_edit is not None

    slider.setValue(7)
    assert panel.command_spec()[2] == {"-M": 7}

    panel.set_command("extract", {"-M": 3})
    assert slider.value() == 3
    assert line_edit.text() == "3"
