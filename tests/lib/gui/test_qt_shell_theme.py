#!/usr/bin/env python3
"""Tests for Qt shell theme rendering."""

from __future__ import annotations

from pathlib import Path

from lib.gui.qt_shell.theme import QtTheme, apply_theme, load_theme, render_qss
from lib.serializer import get_serializer


class _AppDouble:
    """Small QApplication stand-in."""

    def __init__(self) -> None:
        self.stylesheet = ""

    def setStyleSheet(self, stylesheet: str) -> None:  # noqa:N802
        """Capture applied stylesheet."""
        self.stylesheet = stylesheet


def test_default_theme_has_required_colors() -> None:
    """Default theme should expose all required color tokens."""
    theme = QtTheme.default()

    assert theme.name == "Faceswap Dark"
    assert theme.color("window") == "#2b2b2b"
    assert theme.color("accent") == "#5aa9e6"


def test_theme_from_mapping_merges_valid_overrides() -> None:
    """Theme mapping should merge valid color and sizing overrides."""
    theme = QtTheme.from_mapping(
        {
            "name": "Custom",
            "font_family": "Inter",
            "font_size": 12,
            "spacing": 10,
            "radius": 6,
            "colors": {"accent": "#123456", "bad": "not-a-color"},
        }
    )

    assert theme.name == "Custom"
    assert theme.font_family == "Inter"
    assert theme.font_size == 12
    assert theme.spacing == 10
    assert theme.radius == 6
    assert theme.color("accent") == "#123456"
    assert "bad" not in theme.colors


def test_theme_from_mapping_uses_defaults_for_invalid_values() -> None:
    """Invalid numeric values should fall back to defaults."""
    theme = QtTheme.from_mapping({"font_size": 0, "spacing": -1, "radius": False})

    assert theme.font_size == QtTheme.default().font_size
    assert theme.spacing == QtTheme.default().spacing
    assert theme.radius == QtTheme.default().radius


def test_render_qss_contains_core_widgets_and_tokens() -> None:
    """Rendered QSS should include key widgets and selected colors."""
    theme = QtTheme.from_mapping({"colors": {"accent": "#123456"}})

    qss = render_qss(theme)

    assert "QMainWindow" in qss
    assert "QTabBar::tab:selected" in qss
    assert "QProgressBar::chunk" in qss
    assert "#123456" in qss


def test_apply_theme_sets_application_stylesheet() -> None:
    """Applying a theme should set the application stylesheet."""
    app = _AppDouble()
    theme = QtTheme.from_mapping({"colors": {"accent": "#123456"}})

    applied = apply_theme(app, theme)  # type:ignore[arg-type]

    assert applied == theme
    assert "#123456" in app.stylesheet


def test_load_theme_reads_json_payload(tmp_path: Path) -> None:
    """Themes should load from JSON serializer payloads."""
    filename = tmp_path / "theme.json"
    get_serializer("json").save(
        str(filename),
        {"name": "Loaded", "colors": {"accent": "#abcdef"}},
    )

    theme = load_theme(filename)

    assert theme.name == "Loaded"
    assert theme.color("accent") == "#abcdef"
