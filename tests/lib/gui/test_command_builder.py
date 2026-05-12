"""Tests for GUI command construction."""

from __future__ import annotations

import os

from lib.gui.command_builder import CommandBuilder


def script_path(base_path: str) -> str:
    """Return the platform-native faceswap.py path."""
    return os.path.join(base_path, "faceswap.py")


def test_build_preserves_zero_values() -> None:
    """Numeric zeroes are valid CLI values and should not be treated as False."""
    builder = CommandBuilder(executable="python", base_path="/faceswap")

    args = builder.build(
        "faceswap",
        "extract",
        {"-a": 0, "-b": 0.0, "-c": False, "-d": ""},
    )

    assert args == [
        "python",
        "-u",
        script_path("/faceswap"),
        "extract",
        "-a",
        "0",
        "-b",
        "0.0",
        "-G",
    ]


def test_build_skips_empty_values() -> None:
    """Empty option values should not emit switches."""
    builder = CommandBuilder(executable="python", base_path="/faceswap")

    args = builder.build(
        "faceswap",
        "extract",
        {"-a": None, "-b": False, "-c": "", "-d": [], "-e": ()},
    )

    assert args == ["python", "-u", script_path("/faceswap"), "extract", "-G"]


def test_build_groups_short_boolean_switches() -> None:
    """Short boolean switches are grouped to match the legacy generated CLI shape."""
    builder = CommandBuilder(executable="python", base_path="/faceswap")

    args = builder.build(
        "faceswap",
        "extract",
        {"-a": True, "-b": True, "--long": True},
    )

    assert args == [
        "python",
        "-u",
        script_path("/faceswap"),
        "extract",
        "-ab",
        "--long",
        "-G",
    ]


def test_build_expands_sequence_values() -> None:
    """Sequence values should be expanded for argparse nargs options."""
    builder = CommandBuilder(executable="python", base_path="/faceswap")

    args = builder.build("faceswap", "extract", {"-i": ["one", "two"]})

    assert args == [
        "python",
        "-u",
        script_path("/faceswap"),
        "extract",
        "-i",
        "one",
        "two",
        "-G",
    ]


def test_generate_command_omits_unbuffered_and_gui_flags_and_quotes_spaces() -> None:
    """Generated display commands should match legacy display behavior."""
    builder = CommandBuilder(executable="python", base_path="/face swap")

    args = builder.build(
        "faceswap",
        "extract",
        {"-i": "/input folder", "-b": True},
        generate=True,
    )

    assert args == [
        "python",
        f'"{script_path("/face swap")}"',
        "extract",
        "-b",
        "-i",
        '"/input folder"',
    ]
