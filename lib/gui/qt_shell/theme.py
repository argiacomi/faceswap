#!/usr/bin/env python3
"""Qt shell theme model and stylesheet helpers."""

from __future__ import annotations

import re
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtWidgets import QApplication

from lib.serializer import get_serializer
from lib.utils import get_module_objects

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class QtTheme:
    """Serializable visual theme settings for the Qt shell."""

    name: str = "Faceswap Dark"
    font_family: str = "Arial"
    font_size: int = 10
    spacing: int = 8
    radius: int = 4
    colors: dict[str, str] = field(default_factory=dict)

    DEFAULT_COLORS: T.ClassVar[dict[str, str]] = {
        "window": "#2b2b2b",
        "panel": "#353535",
        "panel_alt": "#404040",
        "text": "#f0f0f0",
        "muted_text": "#b8b8b8",
        "border": "#555555",
        "accent": "#5aa9e6",
        "accent_text": "#101820",
        "console_background": "#1e1e1e",
        "console_text": "#e8e8e8",
        "input_background": "#262626",
    }

    @classmethod
    def default(cls) -> QtTheme:
        """Return the built-in default Qt shell theme."""
        return cls(colors=dict(cls.DEFAULT_COLORS))

    @classmethod
    def from_mapping(cls, payload: T.Mapping[str, T.Any]) -> QtTheme:
        """Build and validate a theme from serializer data."""
        default = cls()
        colors = dict(cls.DEFAULT_COLORS)
        raw_colors = payload.get("colors", {})
        if isinstance(raw_colors, dict):
            colors.update(_valid_colors(raw_colors))
        return cls(
            name=str(payload.get("name") or default.name),
            font_family=str(payload.get("font_family") or default.font_family),
            font_size=_positive_int(payload.get("font_size"), default.font_size),
            spacing=_positive_int(payload.get("spacing"), default.spacing),
            radius=_positive_int(payload.get("radius"), default.radius),
            colors=colors,
        )

    def color(self, name: str) -> str:
        """Return a named color, falling back to defaults."""
        return self.colors.get(name, self.DEFAULT_COLORS[name])


def load_theme(filename: str | Path | None = None) -> QtTheme:
    """Load a theme from disk, or return the default theme when omitted."""
    if filename is None:
        return QtTheme.default()
    payload = get_serializer("json").load(str(filename))
    if not isinstance(payload, dict):
        raise ValueError("Qt theme file must contain a mapping")
    return QtTheme.from_mapping(payload)


def render_qss(theme: QtTheme) -> str:
    """Render the base Qt shell stylesheet."""
    return "\n".join(
        (
            _rule("QWidget", theme, "window", "text"),
            _rule("QMainWindow, QMenuBar, QMenu", theme, "window", "text"),
            _rule("QToolBar, QStatusBar, QTabWidget::pane, QGroupBox", theme, "panel", "text"),
            _rule("QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QListWidget, QTableWidget", theme, "input_background", "text"),
            _rule("QPushButton", theme, "panel_alt", "text"),
            _rule("QTabBar::tab", theme, "panel_alt", "muted_text"),
            _rule("QTabBar::tab:selected", theme, "accent", "accent_text"),
            "QProgressBar::chunk { background-color: %s; }" % theme.color("accent"),
        )
    )


def apply_theme(app: QApplication, theme: QtTheme | None = None) -> QtTheme:
    """Apply a Qt shell theme to an application and return the applied theme."""
    selected = theme or QtTheme.default()
    app.setStyleSheet(render_qss(selected))
    return selected


def _rule(selector: str, theme: QtTheme, background: str, foreground: str) -> str:
    """Render a compact QSS rule for common widget groups."""
    return (
        f"{selector} {{ "
        f"background-color: {theme.color(background)}; "
        f"color: {theme.color(foreground)}; "
        f"border-color: {theme.color('border')}; "
        f"border-radius: {theme.radius}px; "
        f"font-family: \"{theme.font_family}\"; "
        f"font-size: {theme.font_size}pt; "
        f"}}"
    )


def _valid_colors(raw_colors: T.Mapping[object, object]) -> dict[str, str]:
    """Return valid hex color overrides."""
    return {
        key: value
        for key, value in raw_colors.items()
        if isinstance(key, str) and isinstance(value, str) and _HEX_COLOR_RE.match(value)
    }


def _positive_int(value: object, default: int) -> int:
    """Return a positive int or a default."""
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


__all__ = get_module_objects(__name__)
