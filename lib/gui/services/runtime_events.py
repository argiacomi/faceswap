#!/usr/bin/env python3
"""Structured GUI runtime events."""

from __future__ import annotations

from dataclasses import dataclass

from lib.utils import get_module_objects


@dataclass(frozen=True)
class RuntimeEvent:
    """A structured event emitted while a Faceswap process is running."""

    kind: str
    message: str = ""
    progress: float | None = None
    payload: dict[str, object] | None = None


@dataclass(frozen=True)
class ParsedRuntimeOutput:
    """Structured events parsed from a single process output line."""

    events: tuple[RuntimeEvent, ...] = ()
    consumed: bool = False


__all__ = get_module_objects(__name__)
