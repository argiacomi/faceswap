#!/usr/bin/env python3
"""Tests for Qt shell theme rendering."""

from __future__ import annotations

from pathlib import Path

from lib.gui.qt_shell.theme import (
    QtTheme,
    apply_theme,
    icon_for_action,
    icon_path,
    load_theme,
    render_qss,
)
from lib.serializer import get_serializer


class _AppDouble:
    """Small QApplication stand-in."""

    def __init__(self) -> None:
        self.stylesheet = ""
        self.font = None
        self.window_icon = None

    def setFont(self, font) -> None:  # noqa:N802 ANN001
        """Capture applied font."""
        self.font = font

    def setStyleSheet(self, stylesheet: str) -> None:  # noqa:N802
        """Capture applied stylesheet."""
        self.stylesheet = stylesheet

    def setWindowIcon(self, icon) -> None:  # noqa:N802 ANN001
        """Capture applied application icon."""
        self.window_icon = icon


def test_default_theme_has_required_colors() -> None:
    """Default theme should expose all required color tokens."""
    theme = QtTheme.default()

    assert theme.name == "Faceswap Dark"
    assert theme.color("window") == "#2a2d2e"
    assert theme.color("accent") == "#66b2d8"
    assert theme.color("menu_text") == "#000000"
    assert theme.color("dialog_background") == "#333738"
    assert theme.icon_name("run") == "play"
    assert theme.icon_name("settings") == "settings"


def test_theme_from_mapping_merges_valid_overrides() -> None:
    """Theme mapping should merge valid color and sizing overrides."""
    theme = QtTheme.from_mapping(
        {
            "name": "Custom",
            "font_family": "Inter",
            "font_size": 12,
            "icon_size": 18,
            "spacing": 10,
            "radius": 6,
            "colors": {"accent": "#123456", "bad": "not-a-color"},
            "icons": {"run": "start"},
        }
    )

    assert theme.name == "Custom"
    assert theme.font_family == "Inter"
    assert theme.font_size == 12
    assert theme.icon_size == 18
    assert theme.spacing == 10
    assert theme.radius == 6
    assert theme.color("accent") == "#123456"
    assert theme.icon_name("run") == "start"
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
    assert "qt-shell-option-group" in qss
    assert "qt-shell-console" in qss
    assert "QProgressBar::chunk" in qss
    assert "#123456" in qss


def test_render_qss_contains_chrome_and_dialog_parity_selectors() -> None:
    """Chrome QSS should cover menus, toolbar, tabs, console, status and dialogs."""
    theme = QtTheme.from_mapping(
        {
            "font_family": "Inter",
            "font_size": 11,
            "icon_size": 22,
            "colors": {
                "menu_background": "#111111",
                "menu_hover": "#222222",
                "dialog_background": "#333333",
                "console_background": "#444444",
            },
        }
    )

    qss = render_qss(theme)

    for selector in (
        "QMenuBar::item:selected",
        "QMenu::separator",
        "QToolBar#qt-shell-toolbar QToolButton",
        "QToolBar::separator",
        "QTabWidget#qt-shell-display-tabs::pane",
        "QStatusBar::item",
        "QProgressBar#qt-shell-progress",
        "QPlainTextEdit#qt-shell-console",
        "QDialog QPushButton",
        "QTreeWidget#qt-shell-settings-tree::item:selected",
    ):
        assert selector in qss
    assert "min-width: 30px" in qss
    assert "font-family: \"Inter\"" in qss
    assert "#111111" in qss
    assert "#222222" in qss
    assert "#333333" in qss
    assert "#444444" in qss


def test_theme_from_mapping_translates_legacy_faceswap_theme() -> None:
    """Legacy Tk theme sections should translate to Qt color tokens."""
    theme = QtTheme.from_mapping(
        {
            "group_panel": {
                "panel_background": "#cdd3d5",
                "group_background": "#ffffff",
                "group_border": "#176087",
                "group_font": "#000000",
                "header_color": "#176087",
                "header_font": "#ffffff",
                "input_color": "#eeeeee",
                "button_background": "#dddddd",
            },
            "console": {
                "background_color": "#101010",
                "stdout_color": "#202020",
                "warning_color": "#303030",
            },
        }
    )

    assert theme.color("window") == "#cdd3d5"
    assert theme.color("section") == "#ffffff"
    assert theme.color("border") == "#176087"
    assert theme.color("dialog_background") == "#cdd3d5"
    assert theme.color("console_background") == "#101010"
    assert theme.color("console_text") == "#202020"
    assert theme.color("warning") == "#303030"


def test_legacy_icon_cache_paths_are_resolved() -> None:
    """Qt should reuse the same icon cache as the Tk taskbar and favicon."""
    theme = QtTheme.default()

    assert icon_path(theme, "favicon").name == "favicon.png"
    assert icon_path(theme, "open").name == "load.png"
    assert icon_path(theme, "settings_extract").name == "settings_extract.png"
    assert icon_path(theme, "settings_train").name == "settings_train.png"
    assert icon_path(theme, "settings_convert").name == "settings_convert.png"
    assert icon_path(theme, "favicon").is_file()
    assert icon_path(theme, "settings_extract").is_file()
    assert not icon_for_action(theme, "favicon").isNull()


def test_apply_theme_sets_application_stylesheet_font_and_icon() -> None:
    """Applying a theme should set app font, favicon and stylesheet."""
    app = _AppDouble()
    theme = QtTheme.from_mapping(
        {
            "font_family": "Inter",
            "font_size": 12,
            "colors": {"accent": "#123456"},
        }
    )

    applied = apply_theme(app, theme)  # type:ignore[arg-type]

    assert applied == theme
    assert "#123456" in app.stylesheet
    assert app.font is not None
    assert app.font.family() == "Inter"
    assert app.font.pointSize() == 12
    assert app.window_icon is not None
    assert not app.window_icon.isNull()


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
