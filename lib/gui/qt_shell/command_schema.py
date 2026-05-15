#!/usr/bin/env python3
"""Command schema adapter for the Qt shell prototype."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionSpec:
    """Small Qt-renderable command option descriptor."""

    title: str
    switch: str
    value_type: type = str
    default: object = ""
    choices: tuple[str, ...] = ()
    nargs: bool = False
    action: str | None = None
    group: str | None = None
    helptext: str = ""
    browser_modes: tuple[str, ...] = ()
    is_radio: bool = False
    is_multi_option: bool = False
    slider_min: float | None = None
    slider_max: float | None = None
    slider_rounding: float | None = None
    is_required: bool = False
    is_advanced: bool = False
    file_filter: str = ""


@dataclass(frozen=True)
class CommandSpec:
    """A command and the options rendered for it."""

    category: str
    command: str
    options: tuple[OptionSpec, ...]
    info: str = ""


class CommandSchema:
    """Small command schema adapter for Qt command rendering."""

    def __init__(self, commands: T.Iterable[CommandSpec]) -> None:
        self._commands = tuple(commands)
        self._by_category = self._build_category_index(self._commands)
        self._by_command = {spec.command: spec for spec in self._commands}

    @property
    def categories(self) -> tuple[str, ...]:
        """Return available command categories."""
        return tuple(self._by_category)

    def commands(self, category: str) -> tuple[str, ...]:
        """Return command names for the given category."""
        return tuple(spec.command for spec in self._by_category.get(category, ()))

    def options(self, command: str) -> tuple[OptionSpec, ...]:
        """Return option specs for the given command."""
        spec = self._by_command.get(command)
        return spec.options if spec is not None else self.default_options()

    def command_info(self, command: str) -> str:
        """Return description text for the given command."""
        spec = self._by_command.get(command)
        return "" if spec is None else spec.info

    def category_for_command(self, command: str) -> str | None:
        """Return the category that owns the given command."""
        spec = self._by_command.get(command)
        return None if spec is None else spec.category

    @staticmethod
    def default_options() -> tuple[OptionSpec, ...]:
        """Return fallback placeholder options for unknown commands."""
        return (OptionSpec("Input", "-i"), OptionSpec("Output", "-o"))

    @classmethod
    def prototype(cls) -> CommandSchema:
        """Return the lightweight prototype schema.

        This keeps the Qt shell schema-backed without attempting full CLI parity.
        """
        return cls(
            (
                CommandSpec(
                    "faceswap",
                    "extract",
                    (
                        OptionSpec("Input Dir", "-i"),
                        OptionSpec("Output Dir", "-o"),
                        OptionSpec("Detector", "-D"),
                        OptionSpec("Aligner", "-A"),
                        OptionSpec("Batch Mode", "-b", bool, False),
                    ),
                    "Extract faces from image or video sources.",
                ),
                CommandSpec(
                    "faceswap",
                    "train",
                    (
                        OptionSpec("Input A", "-A"),
                        OptionSpec("Input B", "-B"),
                        OptionSpec("Model Dir", "-m"),
                        OptionSpec("Trainer", "-t"),
                    ),
                    "Train a Faceswap model.",
                ),
                CommandSpec(
                    "faceswap",
                    "convert",
                    (
                        OptionSpec("Input Dir", "-i"),
                        OptionSpec("Output Dir", "-o"),
                        OptionSpec("Model Dir", "-m"),
                        OptionSpec("Trainer", "-t"),
                    ),
                    "Swap faces in image or video sources.",
                ),
                CommandSpec("tools", "alignments", cls.default_options()),
                CommandSpec("tools", "preview", cls.default_options()),
                CommandSpec("tools", "sort", cls.default_options()),
            )
        )

    @staticmethod
    def _build_category_index(
        commands: tuple[CommandSpec, ...],
    ) -> dict[str, tuple[CommandSpec, ...]]:
        """Build a category ordered index from command specs."""
        index: dict[str, list[CommandSpec]] = {}
        for spec in commands:
            index.setdefault(spec.category, []).append(spec)
        return {category: tuple(specs) for category, specs in index.items()}
