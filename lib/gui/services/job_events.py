#!/usr/bin/env python3
"""Backward-compatible names for GUI runtime events."""

from __future__ import annotations

from lib.utils import get_module_objects

from .progress_parser import ProgressParser
from .runtime_events import ParsedRuntimeOutput, RuntimeEvent

JobEvent = RuntimeEvent
ParsedJobOutput = ParsedRuntimeOutput
JobOutputParser = ProgressParser

__all__ = get_module_objects(__name__)
