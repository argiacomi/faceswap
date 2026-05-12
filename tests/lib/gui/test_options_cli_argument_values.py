#!/usr/bin/env python3
"""Tests for extracting switch-keyed CLI values from GUI options."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lib.gui.options import CliOption, CliOptions


@dataclass
class _PanelOption:
    """Minimal panel option test double."""

    value: object

    def get(self) -> object:
        """Return the stored test value."""
        return self.value


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


def _option(value: object, switch: str, nargs: str | None = None) -> CliOption:
    """Build a minimal CliOption for adapter tests."""
    return CliOption(
        panel_option=_PanelOption(value),  # type: ignore[arg-type]
        opts=(switch,),
        nargs=nargs,  # type: ignore[arg-type]
    )


def test_get_cli_argument_values_returns_switch_keyed_values() -> None:
    """User-facing option titles are converted to real CLI switches."""
    cli_options = CliOptions.__new__(CliOptions)
    cli_options._opts = {  # pylint:disable=protected-access
        "extract": {
            "Input Dir": _option("/input", "-i"),
            "Batch Mode": _option(True, "-b"),
            "helptext": "ignored",
        }
    }

    values = cli_options.get_cli_argument_values("extract")

    assert values == {"-i": "/input", "-b": True}


def test_get_cli_argument_values_splits_nargs_strings() -> None:
    """nargs strings are split before being handed to the command builder."""
    cli_options = CliOptions.__new__(CliOptions)
    cli_options._opts = {  # pylint:disable=protected-access
        "extract": {
            "Input Items": _option('"one path" two', "-i", nargs="+"),
        }
    }

    values = cli_options.get_cli_argument_values("extract")

    assert values == {"-i": ["one path", "two"]}


def test_get_cli_argument_values_preserves_nargs_sequences() -> None:
    """Existing sequence values for nargs options are preserved as list values."""
    cli_options = CliOptions.__new__(CliOptions)
    cli_options._opts = {  # pylint:disable=protected-access
        "extract": {
            "Input Items": _option(("one", "two"), "-i", nargs="+"),
        }
    }

    values = cli_options.get_cli_argument_values("extract")

    assert values == {"-i": ["one", "two"]}


def test_gen_cli_arguments_delegates_to_command_builder_groups() -> None:
    """The deprecated generator should share command builder option emission."""
    cli_options = CliOptions.__new__(CliOptions)
    cli_options._opts = {  # pylint:disable=protected-access
        "extract": {
            "Input Dir": _option("/input", "-i"),
            "Batch Mode": _option(True, "-b"),
            "Zero": _option(0, "-z"),
            "Empty": _option("", "-e"),
        }
    }

    with pytest.deprecated_call(match="CliOptions.gen_cli_arguments"):
        args = list(cli_options.gen_cli_arguments("extract"))

    assert args == [("-b",), ("-i", "/input"), ("-z", "0")]


def test_gen_cli_arguments_applies_preview_context(monkeypatch) -> None:
    """The deprecated generator should reuse command execution context side effects."""
    images = _Images()
    monkeypatch.setattr("lib.gui.options.get_images", lambda: images)
    cli_options = CliOptions.__new__(CliOptions)
    cli_options._opts = {  # pylint:disable=protected-access
        "extract": {
            "Output Dir": _option("/output", "-o"),
            "Batch Mode": _option(True, "-b"),
        }
    }

    with pytest.deprecated_call():
        args = list(cli_options.gen_cli_arguments("extract"))

    assert args == [("-b",), ("-o", "/output")]
    assert images.preview_extract.output_path == "/output"
    assert images.preview_extract.batch_mode is True
