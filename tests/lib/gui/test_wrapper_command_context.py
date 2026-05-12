#!/usr/bin/env python3
"""Tests for ProcessWrapper command execution context application."""

from __future__ import annotations

from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.wrapper import ProcessWrapper


class _PreviewExtract:
    """Preview extract test double."""

    def __init__(self) -> None:
        self.output_path: str | None = None
        self.batch_mode: bool | None = None

    def set_faceswap_output_path(self, output_path: str, *, batch_mode: bool = False) -> None:
        """Capture preview output arguments."""
        self.output_path = output_path
        self.batch_mode = batch_mode


class _Images:
    """Image cache test double."""

    def __init__(self) -> None:
        self.preview_extract = _PreviewExtract()


def test_apply_execution_context_sets_training_session_and_preview(monkeypatch) -> None:
    """ProcessWrapper applies context side effects without parsing command values itself."""
    images = _Images()
    wrapper = ProcessWrapper.__new__(ProcessWrapper)
    wrapper._training_session_location = {}  # pylint:disable=protected-access
    monkeypatch.setattr("lib.gui.wrapper.get_images", lambda: images)

    wrapper._apply_execution_context(  # pylint:disable=protected-access
        CommandExecutionContext(
            model_name="original_model",
            model_folder="/models",
            preview_output_path="/output",
            batch_mode=True,
        )
    )

    assert wrapper._training_session_location == {  # pylint:disable=protected-access
        "model_name": "original_model",
        "model_folder": "/models",
    }
    assert images.preview_extract.output_path == "/output"
    assert images.preview_extract.batch_mode is True


def test_apply_execution_context_skips_training_session_for_generated_commands(
    monkeypatch,
) -> None:
    """Generated commands should not update training session state."""
    images = _Images()
    wrapper = ProcessWrapper.__new__(ProcessWrapper)
    wrapper._training_session_location = {}  # pylint:disable=protected-access
    monkeypatch.setattr("lib.gui.wrapper.get_images", lambda: images)

    wrapper._apply_execution_context(  # pylint:disable=protected-access
        CommandExecutionContext(
            model_name="original_model",
            model_folder="/models",
        ),
        generate=True,
    )

    assert wrapper._training_session_location == {}  # pylint:disable=protected-access


def test_apply_execution_context_skips_preview_for_generated_commands(
    monkeypatch,
) -> None:
    """Generated commands should not update preview image state."""
    images = _Images()
    wrapper = ProcessWrapper.__new__(ProcessWrapper)
    wrapper._training_session_location = {}  # pylint:disable=protected-access
    monkeypatch.setattr("lib.gui.wrapper.get_images", lambda: images)

    wrapper._apply_execution_context(  # pylint:disable=protected-access
        CommandExecutionContext(preview_output_path="/output", batch_mode=True),
        generate=True,
    )

    assert images.preview_extract.output_path is None
    assert images.preview_extract.batch_mode is None
