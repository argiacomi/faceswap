#!/usr/bin/env python3
"""Prototype Qt shell for validating the Faceswap GUI service layer."""

from __future__ import annotations

import typing as T

__all__ = ["MainWindow", "main"]


def __getattr__(name: str) -> T.Any:
    """Lazily expose Qt shell entrypoints without import-time cycles."""
    if name == "main":
        from lib.gui.qt_shell.main import main

        return main
    if name == "MainWindow":
        from lib.gui.qt_shell.main_window import MainWindow

        return MainWindow
    raise AttributeError(name)
