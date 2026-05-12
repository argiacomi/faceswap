#!/usr/bin/env python3
"""Tests for command execution context derivation."""

from __future__ import annotations

from lib.gui.services.command_context import CommandExecutionContext


def test_command_context_extract_preview_output() -> None:
    """Extract commands expose preview output and batch mode context."""
    context = CommandExecutionContext.from_values(
        "extract",
        {"-o": "/output", "-b": True},
    )

    assert context.preview_output_path == "/output"
    assert context.batch_mode is True
    assert context.model_name is None
    assert context.model_folder is None


def test_command_context_convert_preview_output() -> None:
    """Convert commands expose preview output but never batch mode."""
    context = CommandExecutionContext.from_values(
        "convert",
        {"-o": "/output", "-b": True},
    )

    assert context.preview_output_path == "/output"
    assert context.batch_mode is False


def test_command_context_train_model_info() -> None:
    """Train commands expose normalized model info for analysis sessions."""
    context = CommandExecutionContext.from_values(
        "train",
        {"-t": "Original-Model", "-m": "/models"},
    )

    assert context.model_name == "original_model"
    assert context.model_folder == "/models"
    assert context.preview_output_path is None
    assert context.batch_mode is False
