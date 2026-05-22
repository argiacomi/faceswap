#!/usr/bin/env python3
"""Qt shell native Manual Tool launch path tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_command_panel_set_external_errors_shown_inline(qtbot) -> None:  # type:ignore[no-untyped-def]
    """External errors render inline in the command panel validation area."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (CommandSpec("tools", "manual", (OptionSpec("Frames", "-f"),)),),
    )
    panel = CommandPanel(schema)
    qtbot.addWidget(panel)

    panel.set_external_errors(("Frames input does not exist: /missing",))

    assert "Frames input does not exist" in panel._validation_label.text()  # noqa: SLF001
    assert panel._validation_label.isHidden() is False  # noqa: SLF001


def test_command_panel_switching_commands_clears_external_errors(  # type:ignore[no-untyped-def]
    qtbot,
) -> None:
    """Switching to a different command must drop stale launcher errors."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec("tools", "manual", (OptionSpec("Frames", "-f"),)),
            CommandSpec("tools", "alignments", (OptionSpec("Input", "-i"),)),
        ),
    )
    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    # Seed an external error against the manual command.
    panel.set_external_errors(("Frames input does not exist: /missing",))
    assert panel._validation_label.isHidden() is False  # noqa: SLF001

    # Switching to another command should drop the stale error.
    panel.set_command("alignments", {})

    assert panel._external_errors == ()  # noqa: SLF001
    assert panel._validation_label.isHidden() is True  # noqa: SLF001


def test_command_panel_value_change_clears_external_errors(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Editing a field should drop stale external errors."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (CommandSpec("tools", "manual", (OptionSpec("Frames", "-f"),)),),
    )
    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    panel.set_external_errors(("Frames input does not exist: /missing",))
    assert panel._validation_label.isHidden() is False  # noqa: SLF001

    panel._value_changed()  # noqa: SLF001 - simulate renderer signal

    assert panel._external_errors == ()  # noqa: SLF001
    assert panel._validation_label.isHidden() is True  # noqa: SLF001


def test_manual_window_from_command_values_raises_for_missing_frames(
    qtbot,  # type:ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """The native window builder surfaces session validation errors as ValueError."""
    from lib.gui.qt_shell.manual_tool import ManualToolWindow
    from lib.gui.services.command_builder import CommandBuilder

    builder = CommandBuilder(base_path=str(tmp_path))
    with pytest.raises(ValueError, match="does not exist"):
        ManualToolWindow.from_command_values(
            {"-f": str(tmp_path / "missing")},
            builder=builder,
        )


def test_manual_window_from_command_values_succeeds_for_image_folder(
    qtbot,  # type:ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """A valid image folder produces a native window with the shared session."""
    from lib.gui.qt_shell.manual_tool import ManualToolWindow
    from lib.gui.services.command_builder import CommandBuilder

    (tmp_path / "frame.png").write_bytes(b"png")
    builder = CommandBuilder(base_path=str(tmp_path))

    window = ManualToolWindow.from_command_values(
        {"-f": str(tmp_path)},
        builder=builder,
    )
    qtbot.addWidget(window)

    assert window.session.frames == str(tmp_path.resolve())
    assert window.session.frame_count == 1
    # Legacy fallback args are populated through CommandBuilder.
    legacy_args = window._legacy_args  # noqa: SLF001
    assert any("tools.py" in part for part in legacy_args)
