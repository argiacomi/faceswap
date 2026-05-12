#!/usr/bin/env python3
"""Derived execution context for GUI command side effects."""

from __future__ import annotations

from dataclasses import dataclass
import typing as T

from lib.utils import get_module_objects


@dataclass(frozen=True)
class CommandExecutionContext:
    """Side-effect context derived from command values."""

    model_name: str | None = None
    model_folder: str | None = None
    preview_output_path: str | None = None
    batch_mode: bool = False

    @classmethod
    def from_values(
        cls, command: str, values: T.Mapping[str, object]
    ) -> CommandExecutionContext:
        """Build execution context from switch-keyed command values."""
        model_name = None
        model_folder = None
        preview_output_path = None
        batch_mode = False

        if command == "train":
            model_name = cls._normalize_model_name(values.get("-t"))
            model_folder = cls._string_value(values.get("-m"))

        if command in ("extract", "convert"):
            preview_output_path = cls._string_value(values.get("-o"))
            batch_mode = command == "extract" and values.get("-b") is True

        return cls(
            model_name=model_name,
            model_folder=model_folder,
            preview_output_path=preview_output_path,
            batch_mode=batch_mode,
        )

    @classmethod
    def _normalize_model_name(cls, value: object) -> str | None:
        """Normalize a model name for the training analysis session."""
        text = cls._string_value(value)
        return None if text is None else text.lower().replace("-", "_")

    @classmethod
    def _string_value(cls, value: object) -> str | None:
        """Return a non-empty string value or ``None``."""
        if value is None or value is False:
            return None
        if isinstance(value, (list, tuple)):
            return None if not value else str(value[0])
        text = str(value)
        return text if text else None


__all__ = get_module_objects(__name__)
