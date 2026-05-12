#!/usr/bin/env python3
"""Qt option renderer tests."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QGroupBox, QRadioButton


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
        for button in panel.renderer.widget_for_switch("--mode").findChildren(
            QRadioButton
        )
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
        for checkbox in panel.renderer.widget_for_switch("--masks").findChildren(
            QCheckBox
        )
    }

    boxes["mouth"].setChecked(True)
    assert panel.command_spec()[2] == {"--masks": ["eyes", "mouth"]}

    panel.set_command("extract", {"--masks": ["nose"]})
    assert boxes["nose"].isChecked() is True
