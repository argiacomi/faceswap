#!/usr/bin/env python3
"""Mask shortcut parity tests for the Qt Manual Tool."""

from __future__ import annotations

from tools.manual.qt.actions import MANUAL_ACTIONS


def _action_shortcuts(key: str) -> tuple[str, ...]:
    return next(action.shortcut for action in MANUAL_ACTIONS if action.key == key)


def test_mask_draw_and_erase_shortcuts_match_legacy_tk() -> None:
    """Mask Draw uses D and Mask Erase uses E."""
    assert _action_shortcuts("mask_draw") == ("D",)
    assert _action_shortcuts("mask_erase") == ("E",)
