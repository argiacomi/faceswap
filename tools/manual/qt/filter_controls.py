#!/usr/bin/env python3
"""Filter-control helpers for the Qt Manual Tool."""

from __future__ import annotations

MISALIGNED_DISTANCE_RANGE = (5, 20)


def filter_status_text(mode: str, match_count: int) -> str:
    """Return the user-facing filter status message."""
    suffix = f" ({match_count} match)" if match_count else " (no matches)"
    return f"Filter: {mode}{suffix}"
