#!/usr/bin/env python3
"""Tests for structured GUI runtime events."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lib.gui.services.runtime_events import ParsedRuntimeOutput, RuntimeEvent


def test_runtime_event_defaults() -> None:
    """RuntimeEvent should support a minimal status/progress contract."""
    event = RuntimeEvent(kind="status")

    assert event.message == ""
    assert event.progress is None
    assert event.payload is None


def test_runtime_event_is_frozen() -> None:
    """RuntimeEvent should be immutable once emitted."""
    event = RuntimeEvent(kind="progress", progress=50.0)

    with pytest.raises(FrozenInstanceError):
        event.progress = 75.0  # type:ignore[misc]


def test_parsed_runtime_output_defaults_to_unconsumed_empty_events() -> None:
    """Unparsed output should pass through to legacy console behavior."""
    parsed = ParsedRuntimeOutput()

    assert parsed.events == ()
    assert parsed.consumed is False
