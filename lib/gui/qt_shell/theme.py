#!/usr/bin/env python3
"""Qt shell theme model and stylesheet helpers."""

from __future__ import annotations

import re
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtGui import QFont
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
    icon_size: int = 16
    spacing: int = 8
    radius: int = 4
    colors: dict[str, str] = field(default_factory=dict)
    icons: dict[str, str] = field(default_factory=dict)

    DEFAULT_COLORS: T.ClassVar[dict[str, str]] = {
        "window": "#2a2d2e",
        "panel": "#333738",
        "panel_alt": "#3c4142",
        "section": "#2f3334",
        "text": "#f2f2f2",
        "muted_text": "#b8c0c2",
        "border": "#565f61",
        "accent": "#66b2d8",
        "accent_hover": "#7dc4e4",
        "accent_text": "#11191c",
        "console_background": "#171a1b",
        "console_text": "#dfe7e9",
        "input_background": "#222627",
        "input_focus": "#293538",
        "button": "#42484a",
        "button_hover": "#4d5557",
        "button_disabled": "#303435",
        "success": "#83c97d",
        "warning": "#d8b866",
        "error": "#df7b7b",
        "info": "#7fb9e3",
        "verbose": "#9fca8b",
        "critical": "#f06f61",
    }
    DEFAULT_ICONS: T.ClassVar[dict[str, str]] = {
        "new": "new",
        "open": "load",
        "save": "save",
        "save_as": "save_as",
        "generate": "generate",
        "run": "play",
        "stop": "stop",
        "reload": "reload",
        "clear": "clear",
        "analysis": "context",
        "preview": "view",
        "graph": "graph",
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
        icons = dict(cls.DEFAULT_ICONS)
        colors.update(_legacy_theme_colors(payload))
        raw_colors = payload.get("colors", {})
        if isinstance(raw_colors, dict):
            colors.update(_valid_colors(raw_colors))
        raw_icons = payload.get("icons", {})
        if isinstance(raw_icons, dict):
            icons.update(
                {
                    str(key): str(value)
                    for key, value in raw_icons.items()
                    if isinstance(key, str) and isinstance(value, str) and value
                }
            )
        return cls(
            name=str(payload.get("name") or default.name),
            font_family=str(payload.get("font_family") or default.font_family),
            font_size=_positive_int(payload.get("font_size"), default.font_size),
            icon_size=_positive_int(payload.get("icon_size"), default.icon_size),
            spacing=_positive_int(payload.get("spacing"), default.spacing),
            radius=_positive_int(payload.get("radius"), default.radius),
            colors=colors,
            icons=icons,
        )

    def color(self, name: str) -> str:
        """Return a named color, falling back to defaults."""
        return self.colors.get(name, self.DEFAULT_COLORS[name])

    def icon_name(self, action: str) -> str:
        """Return the mapped legacy icon name for an action."""
        return self.icons.get(action, self.DEFAULT_ICONS.get(action, action))


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
    spacing = theme.spacing
    radius = theme.radius
    return "\n".join(
        (
            _rule("QWidget", theme, "window", "text"),
            _rule("QMainWindow, QMenuBar, QMenu", theme, "window", "text"),
            _rule("QToolBar, QStatusBar, QTabWidget::pane, QGroupBox", theme, "panel", "text"),
            _rule(
                "QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QListWidget, QTableWidget",
                theme,
                "input_background",
                "text",
            ),
            _rule("QPlainTextEdit#qt-shell-console", theme, "console_background", "console_text"),
            _rule("QPushButton", theme, "button", "text"),
            _rule("QTabBar::tab", theme, "panel_alt", "muted_text"),
            _rule("QTabBar::tab:selected", theme, "accent", "accent_text"),
            f"QToolBar {{ spacing: {spacing}px; padding: {max(2, spacing // 2)}px; }}",
            f"QToolButton {{ padding: {max(2, spacing // 2)}px; border-radius: {radius}px; }}",
            "QToolButton:hover, QPushButton:hover { "
            f"background-color: {theme.color('button_hover')}; }}",
            "QPushButton:disabled { "
            f"background-color: {theme.color('button_disabled')}; "
            f"color: {theme.color('muted_text')}; }}",
            "QGroupBox#qt-shell-option-group, QWidget#qt-shell-option-group-master { "
            f"background-color: {theme.color('section')}; "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; margin-top: {spacing}px; }}",
            "QLabel#qt-shell-option-group-label { "
            f"color: {theme.color('accent')}; font-weight: 600; "
            f"padding-bottom: {spacing // 2}px; }}",
            "QLabel#qt-shell-command-info { "
            f"background-color: {theme.color('section')}; "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; padding: {spacing}px; "
            f"color: {theme.color('muted_text')}; }}",
            "QScrollArea#qt-shell-command-scroll, QScrollArea#qt-shell-preview-scroll { "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; }}",
            "QLineEdit:focus, QComboBox:focus, QTextEdit:focus, "
            "QPlainTextEdit:focus { "
            f"background-color: {theme.color('input_focus')}; "
            f"border: 1px solid {theme.color('accent')}; }}",
            f"QTabWidget::pane {{ border-top: 1px solid {theme.color('border')}; }}",
            f"QTabBar::tab {{ padding: {spacing}px {spacing * 2}px; margin-right: 2px; }}",
            "QProgressBar { "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; text-align: center; }}",
            "QProgressBar::chunk { "
            f"background-color: {theme.color('accent')}; "
            f"border-radius: {max(0, radius - 1)}px; }}",
            f"QLabel[status='success'] {{ color: {theme.color('success')}; }}",
            f"QLabel[status='warning'] {{ color: {theme.color('warning')}; }}",
            f"QLabel[status='error'] {{ color: {theme.color('error')}; }}",
            f"QLabel[status='info'] {{ color: {theme.color('info')}; }}",
            f"QPlainTextEdit[console='stdout'] {{ color: {theme.color('console_text')}; }}",
            f"QPlainTextEdit[console='stderr'] {{ color: {theme.color('error')}; }}",
            f"QPlainTextEdit[console='warning'] {{ color: {theme.color('warning')}; }}",
            f"QPlainTextEdit[console='critical'] {{ color: {theme.color('critical')}; }}",
        )
    )


def apply_theme(app: QApplication, theme: QtTheme | None = None) -> QtTheme:
    """Apply a Qt shell theme to an application and return the applied theme."""
    selected = theme or QtTheme.default()
    font = QFont(selected.font_family, selected.font_size)
    if hasattr(app, "setFont"):
        app.setFont(font)
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
        f'font-family: "{theme.font_family}"; '
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


def _legacy_theme_colors(payload: T.Mapping[str, T.Any]) -> dict[str, str]:
    """Translate legacy Tk theme sections into Qt theme tokens when present."""
    group = payload.get("group_panel")
    tabs = payload.get("command_tabs")
    console = payload.get("console")
    translated: dict[str, str] = {}
    if isinstance(group, dict):
        mapping = {
            "window": "panel_background",
            "panel": "panel_background",
            "panel_alt": "group_background",
            "section": "group_background",
            "text": "group_font",
            "muted_text": "control_color",
            "border": "group_border",
            "accent": "header_color",
            "accent_hover": "control_active",
            "accent_text": "header_font",
            "input_background": "input_color",
            "input_focus": "group_background",
            "button": "button_background",
            "button_hover": "control_color",
            "button_disabled": "control_disabled",
        }
        translated.update(_section_colors(group, mapping))
    if isinstance(tabs, dict):
        translated.update(
            _section_colors(
                tabs,
                {
                    "border": "frame_border",
                    "panel_alt": "tab_color",
                    "accent": "tab_selected",
                    "accent_hover": "tab_hover",
                },
            )
        )
    if isinstance(console, dict):
        translated.update(
            _section_colors(
                console,
                {
                    "console_background": "background_color",
                    "console_text": "stdout_color",
                    "info": "info_color",
                    "verbose": "verbose_color",
                    "warning": "warning_color",
                    "critical": "critical_color",
                    "error": "error_color",
                },
            )
        )
    return translated


def _section_colors(section: T.Mapping[object, object], mapping: dict[str, str]) -> dict[str, str]:
    """Return valid theme token colors from one legacy section mapping."""
    colors: dict[str, str] = {}
    for token, legacy_key in mapping.items():
        value = section.get(legacy_key)
        if isinstance(value, str) and _HEX_COLOR_RE.match(value):
            colors[token] = value
    return colors


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
