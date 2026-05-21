#!/usr/bin/env python3
"""Qt shell theme model and stylesheet helpers."""

from __future__ import annotations

import hashlib
import logging
import re
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from lib.serializer import get_serializer
from lib.utils import PROJECT_ROOT, get_module_objects

logger = logging.getLogger(__name__)

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_ICON_CACHE = Path(PROJECT_ROOT) / "lib" / "gui" / ".cache" / "icons"
_COMBO_ARROW_BUTTON_SIZE = 25
_COMBO_ARROW_CHEVRON_WIDTH = 6
_COMBO_ARROW_CHEVRON_DROP = 3
_COMBO_ARROW_STROKE_WIDTH = 1.6


@dataclass(frozen=True, slots=True)
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
        "input_text": "#f2f2f2",
        "button": "#42484a",
        "button_hover": "#4d5557",
        "button_disabled": "#303435",
        "menu_background": "#f0f0f0",
        "menu_text": "#000000",
        "menu_hover": "#d9e8f0",
        "dialog_background": "#333738",
        "info_background": "#2f3334",
        "info_text": "#f2f2f2",
        "info_border": "#565f61",
        "scrollbar_border": "#565f61",
        "scrollbar_trough": "#2a2d2e",
        "settings_panel": "#333738",
        "settings_accent": "#66b2d8",
        "settings_border": "#565f61",
        "settings_control": "#b8c0c2",
        "settings_control_active": "#7dc4e4",
        "settings_control_disabled": "#303435",
        "settings_scrollbar_border": "#565f61",
        "settings_scrollbar_trough": "#2a2d2e",
        "settings_tree_select": "#66b2d8",
        "settings_tree_unselected": "#222627",
        "settings_tree_subheader": "#3c4142",
        "settings_link": "#f2f2f2",
        "tab_border": "#565f61",
        "tab_background": "#3c4142",
        "tab_selected": "#66b2d8",
        "tab_hover": "#7dc4e4",
        "console_border": "#565f61",
        "stdout": "#dfe7e9",
        "stderr": "#df7b7b",
        "console_scrollbar_border": "#565f61",
        "console_scrollbar_trough": "#2a2d2e",
        "console_scrollbar_background_normal": "#42484a",
        "console_scrollbar_background_disabled": "#303435",
        "console_scrollbar_background_active": "#4d5557",
        "console_scrollbar_foreground_normal": "#dfe7e9",
        "console_scrollbar_foreground_disabled": "#b8c0c2",
        "console_scrollbar_foreground_active": "#dfe7e9",
        "console_scrollbar_border_normal": "#565f61",
        "console_scrollbar_border_disabled": "#303435",
        "console_scrollbar_border_active": "#66b2d8",
        "tooltip_background": "#ffffea",
        "tooltip_border": "#ffffea",
        "tooltip_text": "#000000",
        "success": "#83c97d",
        "warning": "#d8b866",
        "error": "#df7b7b",
        "info": "#7fb9e3",
        "verbose": "#9fca8b",
        "critical": "#f06f61",
        "group_panel_panel_background": "#333738",
        "group_panel_info_color": "#2f3334",
        "group_panel_info_font": "#f2f2f2",
        "group_panel_info_border": "#565f61",
        "group_panel_header_color": "#66b2d8",
        "group_panel_header_font": "#11191c",
        "group_panel_group_background": "#2f3334",
        "group_panel_group_border": "#565f61",
        "group_panel_group_font": "#f2f2f2",
        "group_panel_control_color": "#b8c0c2",
        "group_panel_control_active": "#7dc4e4",
        "group_panel_control_disabled": "#303435",
        "group_panel_input_color": "#222627",
        "group_panel_input_font": "#f2f2f2",
        "group_panel_button_background": "#42484a",
        "group_panel_scrollbar_border": "#565f61",
        "group_panel_scrollbar_trough": "#2a2d2e",
        "group_settings_panel_background": "#333738",
        "group_settings_header_color": "#66b2d8",
        "group_settings_group_border": "#565f61",
        "group_settings_control_color": "#b8c0c2",
        "group_settings_control_active": "#7dc4e4",
        "group_settings_control_disabled": "#303435",
        "group_settings_scrollbar_border": "#565f61",
        "group_settings_scrollbar_trough": "#2a2d2e",
        "group_settings_tree_select": "#66b2d8",
        "group_settings_tree_unselected": "#222627",
        "group_settings_tree_subheader": "#3c4142",
        "group_settings_link_color": "#f2f2f2",
        "command_tabs_frame_border": "#565f61",
        "command_tabs_tab_color": "#3c4142",
        "command_tabs_tab_selected": "#66b2d8",
        "command_tabs_tab_hover": "#7dc4e4",
        "console_background_color": "#171a1b",
        "console_border_color": "#565f61",
        "console_stdout_color": "#dfe7e9",
        "console_stderr_color": "#df7b7b",
        "console_info_color": "#7fb9e3",
        "console_verbose_color": "#9fca8b",
        "console_warning_color": "#d8b866",
        "console_critical_color": "#f06f61",
        "console_error_color": "#df7b7b",
        "console_scrollbar_border_key": "#565f61",
        "console_scrollbar_trough_key": "#2a2d2e",
        "console_scrollbar_background_normal_key": "#42484a",
        "console_scrollbar_background_disabled_key": "#303435",
        "console_scrollbar_background_active_key": "#4d5557",
        "console_scrollbar_foreground_normal_key": "#dfe7e9",
        "console_scrollbar_foreground_disabled_key": "#b8c0c2",
        "console_scrollbar_foreground_active_key": "#dfe7e9",
        "console_scrollbar_border_normal_key": "#565f61",
        "console_scrollbar_border_disabled_key": "#303435",
        "console_scrollbar_border_active_key": "#66b2d8",
        "tooltip_background_color": "#ffffea",
        "tooltip_border_color": "#ffffea",
        "tooltip_font_color": "#000000",
    }
    DEFAULT_ICONS: T.ClassVar[dict[str, str]] = {
        "favicon": "favicon",
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
        "settings": "settings",
        "settings_extract": "settings_extract",
        "settings_train": "settings_train",
        "settings_convert": "settings_convert",
        "browser_folder": "folder",
        "browser_file": "load2",
        "browser_files": "multi_load",
        "browser_save": "save_as",
        "browser_video": "video",
        "browser_picture": "picture",
        "browser_model": "model",
        "browser_context": "context",
        "task_open": "load2",
        "task_save": "save2",
        "task_save_as": "save_as2",
        "task_reset": "clear2",
        "task_reload": "reload2",
    }

    @classmethod
    def default(cls) -> QtTheme:
        """Return the built-in default Qt shell theme."""
        return cls(colors=dict(cls.DEFAULT_COLORS), icons=dict(cls.DEFAULT_ICONS))

    @classmethod
    def from_mapping(cls, payload: T.Mapping[str, T.Any]) -> QtTheme:
        """Build and validate a theme from serializer data."""
        default = cls.default()
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


def icon_path(theme: QtTheme, action: str) -> Path:
    """Return the legacy icon cache path for a themed action."""
    return _ICON_CACHE / f"{theme.icon_name(action)}.png"


def icon_for_action(theme: QtTheme, action: str) -> QIcon:
    """Return a QIcon loaded from the legacy icon cache for a themed action."""
    path = icon_path(theme, action)
    return QIcon(str(path)) if path.is_file() else QIcon()


def load_theme(filename: str | Path | None = None) -> QtTheme:
    """Load a theme from disk, or return the default theme when omitted."""
    if filename is None:
        return QtTheme.default()
    payload = get_serializer("json").load(str(Path(filename)))
    if not isinstance(payload, dict):
        raise ValueError("Qt theme file must contain a mapping")
    return QtTheme.from_mapping(payload)


def render_qss(theme: QtTheme) -> str:
    """Render the base Qt shell stylesheet."""
    spacing = theme.spacing
    radius = theme.radius
    half_spacing = max(2, spacing // 2)
    tab_vpad = max(2, spacing // 3)
    combo_arrow_path, combo_arrow_active_path = _combo_arrow_icon_paths(theme)
    return "\n".join(
        (
            _rule("QWidget", theme, "window", "text"),
            _rule("QMainWindow", theme, "window", "text"),
            _rule("QToolBar, QStatusBar, QTabWidget::pane, QGroupBox", theme, "panel", "text"),
            _rule(
                "QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QListWidget, QTableWidget",
                theme,
                "input_background",
                "text",
            ),
            _rule("QPlainTextEdit#qt-shell-console", theme, "console_background", "console_text"),
            _rule("QPushButton", theme, "button", "text"),
            _rule("QDialog, QMessageBox, QFileDialog", theme, "dialog_background", "text"),
            _rule("QTabBar::tab", theme, "panel_alt", "muted_text"),
            _rule("QTabBar::tab:selected", theme, "accent", "accent_text"),
            _rule("QMenuBar, QMenu", theme, "menu_background", "menu_text"),
            "QMenuBar { "
            f"padding: 0 {half_spacing}px; "
            f"border-bottom: 1px solid {theme.color('border')}; }}",
            "QMenuBar::item { "
            f"padding: {tab_vpad}px {spacing}px; "
            f"background-color: {theme.color('menu_background')}; "
            f"color: {theme.color('menu_text')}; }}",
            "QMenuBar::item:selected, QMenuBar::item:pressed { "
            f"background-color: {theme.color('menu_hover')}; "
            f"color: {theme.color('menu_text')}; }}",
            f"QMenu {{ padding: {half_spacing}px 0; border: 1px solid {theme.color('border')}; }}",
            "QMenu::item { "
            f"padding: {half_spacing}px {spacing * 3}px {half_spacing}px {spacing * 2}px; "
            f"background-color: transparent; }}",
            "QMenu::item:selected { "
            f"background-color: {theme.color('menu_hover')}; "
            f"color: {theme.color('menu_text')}; }}",
            "QMenu::separator { "
            f"height: 1px; margin: {half_spacing}px {spacing}px; "
            f"background-color: {theme.color('border')}; }}",
            "QToolBar#qt-shell-toolbar { "
            f"spacing: {half_spacing}px; padding: {half_spacing}px; "
            f"border-bottom: 1px solid {theme.color('border')}; }}",
            "QToolBar#qt-shell-toolbar QToolButton { "
            f"min-width: {theme.icon_size + (spacing * 5)}px; "
            f"min-height: {theme.icon_size + (spacing // 2)}px; "
            "margin: 4px 0px;"
            f"border-radius: {radius}px; }}",
            "QToolBar::separator { "
            f"width: 1px; margin: {half_spacing}px; "
            f"background-color: {theme.color('border')}; }}",
            "QToolButton:hover, QPushButton:hover { "
            f"background-color: {theme.color('button_hover')}; }}",
            "QPushButton { "
            f"padding: {half_spacing}px {spacing * 2}px; "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; }}",
            "QPushButton:disabled { "
            f"background-color: {theme.color('button_disabled')}; "
            f"color: {theme.color('muted_text')}; }}",
            "QDialog QPushButton, QMessageBox QPushButton, QFileDialog QPushButton { "
            f"min-width: {max(72, theme.icon_size * 4)}px; "
            f"padding: {half_spacing}px {spacing * 2}px; }}",
            "QDialog QLabel#qt-shell-settings-header, QLabel#qt-shell-settings-header { "
            f"font-size: {theme.font_size + 4}pt; font-weight: 700; }}",
            "QLabel#qt-shell-settings-page-header { "
            f"font-size: {theme.font_size + 2}pt; font-weight: 700; "
            f"color: {theme.color('accent')}; }}",
            "QTreeWidget#qt-shell-settings-tree { "
            f"background-color: {theme.color('input_background')}; "
            f"alternate-background-color: {theme.color('section')}; "
            f"border: 1px solid {theme.color('border')}; }}",
            "QTreeWidget#qt-shell-settings-tree::item { "
            f"padding: {tab_vpad}px {half_spacing}px; }}",
            "QTreeWidget#qt-shell-settings-tree::item:selected { "
            f"background-color: {theme.color('accent')}; "
            f"color: {theme.color('accent_text')}; }}",
            "QWidget#qt-shell-option-group-master { "
            f"background-color: {theme.color('section')}; "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; margin-top: {spacing}px; }}",
            "QWidget#qt-shell-option-group-content { "
            f"background-color: {theme.color('section')}; "
            f"border: 1px solid {theme.color('accent')}; "
            f"border-radius: {radius}px; "
            f"margin: 0 {half_spacing}px {half_spacing}px {half_spacing}px; "
            f"padding: {half_spacing}px; }}",
            "QGroupBox#qt-shell-option-cluster { "
            f"background-color: transparent; "
            f"border: 0; "
            f"margin-top: {spacing}px; padding-top: {spacing}px; }}",
            "QGroupBox#qt-shell-option-cluster::title { "
            f"subcontrol-origin: margin; subcontrol-position: top left; "
            f"left: 0; padding: 0 {half_spacing}px 0 0; "
            f"color: {theme.color('text')}; font-weight: 600; "
            f"background-color: transparent; }}",
            "QLabel#qt-shell-option-group-label { "
            f"color: {theme.color('accent')}; font-weight: 600; "
            f"padding-bottom: {half_spacing}px; }}",
            "QLabel#qt-shell-command-info, QLabel#qt-shell-settings-summary { "
            f"background-color: {theme.color('section')}; "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; padding: {spacing}px; "
            f"color: {theme.color('muted_text')}; }}",
            "QScrollArea#qt-shell-command-scroll, QScrollArea#qt-shell-preview-scroll, "
            "QScrollArea#qt-shell-settings-scroll { "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; }}",
            "QLineEdit, QComboBox, QTextEdit, QPlainTextEdit { "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; padding: {tab_vpad}px {half_spacing}px; }}",
            "QLineEdit:focus, QComboBox:focus, QTextEdit:focus, "
            "QPlainTextEdit:focus { "
            f"background-color: {theme.color('input_focus')}; "
            f"border: 1px solid {theme.color('accent')}; }}",
            "QComboBox { "
            f"background-color: {theme.color('input_background')}; "
            f"color: {theme.color('input_text')}; "
            f"selection-background-color: {theme.color('accent')}; "
            f"selection-color: {theme.color('accent_text')}; "
            f"padding-right: {_COMBO_ARROW_BUTTON_SIZE}px; }}",
            "QComboBox::drop-down { "
            "subcontrol-origin: padding; "
            "subcontrol-position: top right; "
            f"width: {_COMBO_ARROW_BUTTON_SIZE}px; "
            f"background-color: {theme.color('button')}; "
            f"border-left: 1px solid {theme.color('border')}; "
            f"border-top-right-radius: {radius}px; "
            f"border-bottom-right-radius: {radius}px; }}",
            "QComboBox::drop-down:hover, "
            "QComboBox::drop-down:pressed { "
            f"background-color: {theme.color('button_hover')}; }}",
            "QComboBox::down-arrow { "
            f"image: {_qss_url(combo_arrow_path)}; "
            f"width: {_COMBO_ARROW_BUTTON_SIZE}px; "
            f"height: {_COMBO_ARROW_BUTTON_SIZE}px; "
            "margin: 0; "
            "}",
            "QComboBox::down-arrow:hover, "
            "QComboBox::down-arrow:pressed, "
            "QComboBox::down-arrow:on { "
            f"image: {_qss_url(combo_arrow_active_path)}; "
            f"width: {_COMBO_ARROW_BUTTON_SIZE}px; "
            f"height: {_COMBO_ARROW_BUTTON_SIZE}px; "
            "margin: 0; "
            "}",
            "QComboBox QAbstractItemView { "
            f"background-color: {theme.color('input_background')}; "
            f"color: {theme.color('input_text')}; "
            f"border: 1px solid {theme.color('border')}; "
            f"selection-background-color: {theme.color('accent')}; "
            f"selection-color: {theme.color('accent_text')}; "
            "outline: 0; }",
            "QComboBox QAbstractItemView::item { "
            f"background-color: {theme.color('input_background')}; "
            f"color: {theme.color('input_text')}; "
            f"padding: {tab_vpad}px {spacing}px; "
            f"min-height: {theme.icon_size + tab_vpad}px; }}",
            "QComboBox QAbstractItemView::item:selected, "
            "QComboBox QAbstractItemView::item:hover, "
            "QComboBox QAbstractItemView::item:selected:!active { "
            f"background-color: {theme.color('accent')}; "
            f"color: {theme.color('accent_text')}; }}",
            "QTabWidget#qt-shell-display-tabs::pane, QTabWidget::pane { "
            f"border: 1px solid {theme.color('border')}; top: -1px; }}",
            "QTabBar::tab { "
            f"padding: {tab_vpad}px {spacing * 2}px; "
            f"margin-right: 2px; border: 1px solid {theme.color('border')}; "
            f"border-bottom: 0; }}",
            "QTabBar::tab:hover:!selected { "
            f"background-color: {theme.color('accent_hover')}; "
            f"color: {theme.color('accent_text')}; }}",
            "QStatusBar { "
            f"padding: {tab_vpad}px {half_spacing}px; "
            f"border-top: 1px solid {theme.color('border')}; }}",
            "QStatusBar::item { border: 0; }",
            "QProgressBar, QProgressBar#qt-shell-progress { "
            f"background-color: {theme.color('input_background')}; "
            f"border: 1px solid {theme.color('border')}; "
            f"border-radius: {radius}px; text-align: center; "
            f"min-width: {max(120, theme.icon_size * 6)}px; "
            f"min-height: {theme.icon_size}px; }}",
            "QProgressBar::chunk { "
            f"background-color: {theme.color('accent')}; "
            f"border-radius: {max(0, radius - 1)}px; }}",
            "QPlainTextEdit#qt-shell-console { "
            f"padding: {half_spacing}px; "
            f"border: 1px solid {theme.color('border')}; "
            f'font-family: "{theme.font_family}"; '
            f"font-size: {theme.font_size}pt; }}",
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
    if hasattr(app, "setWindowIcon"):
        icon = icon_for_action(selected, "favicon")
        if not icon.isNull():
            app.setWindowIcon(icon)
    app.setStyleSheet(render_qss(selected))
    return selected


def theme_from_gui_config(base: QtTheme | None = None) -> QtTheme:
    """Return a QtTheme overridden with ``lib.gui.gui_config`` font/icon settings.

    Falls back to ``base`` (or the default theme) when gui_config has not been
    registered yet or when a value is unusable.
    """
    template = base or QtTheme.default()
    try:
        from lib.gui import gui_config as cfg
    except ImportError:  # pragma: no cover - module always ships
        return template

    def call(attribute: str) -> object:
        func = getattr(cfg, attribute, None)
        try:
            return func() if callable(func) else None
        except (AttributeError, KeyError, ValueError):
            return None

    font_family = call("font")
    if not isinstance(font_family, str) or font_family in ("", "default"):
        font_family = template.font_family
    font_size = call("font_size")
    if not isinstance(font_size, int) or isinstance(font_size, bool) or font_size <= 0:
        font_size = template.font_size
    icon_size = call("icon_size")
    if not isinstance(icon_size, int) or isinstance(icon_size, bool) or icon_size <= 0:
        icon_size = template.icon_size
    return QtTheme(
        name=template.name,
        font_family=str(font_family),
        font_size=int(font_size),
        icon_size=int(icon_size),
        spacing=template.spacing,
        radius=template.radius,
        colors=dict(template.colors),
        icons=dict(template.icons),
    )


def _combo_arrow_icon_paths(theme: QtTheme) -> tuple[Path, Path]:
    """Return cached SVG paths for the QComboBox down-arrow button."""
    size = _COMBO_ARROW_BUTTON_SIZE
    colors = (
        theme.color("button"),
        theme.color("button_hover"),
        theme.color("input_text"),
        theme.color("border"),
        str(theme.radius),
        str(_COMBO_ARROW_CHEVRON_WIDTH),
        str(_COMBO_ARROW_CHEVRON_DROP),
        str(_COMBO_ARROW_STROKE_WIDTH),
    )
    digest = hashlib.sha1("|".join(colors).encode("utf-8")).hexdigest()[:12]
    normal_path = _ICON_CACHE / f"qt_combo_arrow_{digest}_normal.svg"
    active_path = _ICON_CACHE / f"qt_combo_arrow_{digest}_active.svg"

    _write_combo_arrow_svg(
        normal_path,
        size=size,
        background=theme.color("button"),
        foreground=theme.color("input_text"),
        border=theme.color("border"),
        radius=theme.radius,
        chevron_width=_COMBO_ARROW_CHEVRON_WIDTH,
        chevron_drop=_COMBO_ARROW_CHEVRON_DROP,
        stroke_width=_COMBO_ARROW_STROKE_WIDTH,
    )
    _write_combo_arrow_svg(
        active_path,
        size=size,
        background=theme.color("button_hover"),
        foreground=theme.color("input_text"),
        border=theme.color("border"),
        radius=theme.radius,
        chevron_width=_COMBO_ARROW_CHEVRON_WIDTH,
        chevron_drop=_COMBO_ARROW_CHEVRON_DROP,
        stroke_width=_COMBO_ARROW_STROKE_WIDTH,
    )
    return normal_path, active_path


def _write_combo_arrow_svg(
    path: Path,
    *,
    size: int,
    background: str,
    foreground: str,
    border: str,
    radius: int,
    chevron_width: int,
    chevron_drop: int,
    stroke_width: float,
) -> None:
    """Write a small cached SVG button matching the legacy Tk combobox arrow."""
    if path.is_file():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    safe_radius = max(0, min(radius, size // 2))
    right = size - 0.5
    bottom = size - 0.5
    left = 0.5
    top = 0.5
    center_x = size / 2
    arrow_y = size / 2 - chevron_drop / 2
    half_width = chevron_width / 2
    arrow = (
        f"M {center_x - half_width:.2f} {arrow_y:.2f} "
        f"L {center_x:.2f} {arrow_y + chevron_drop:.2f} "
        f"L {center_x + half_width:.2f} {arrow_y:.2f}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <path d="M {left} {top} H {right - safe_radius} Q {right} {top} {right} {top + safe_radius} V {bottom - safe_radius} Q {right} {bottom} {right - safe_radius} {bottom} H {left} Z" fill="{background}" stroke="{border}" stroke-width="1"/>
  <path d="{arrow}" fill="none" stroke="{foreground}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _qss_url(path: Path) -> str:
    """Return a quoted Qt stylesheet URL for a local path."""
    return f'url("{path.resolve().as_posix()}")'


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


_LEGACY_COLOR_TOKENS: dict[str, dict[str, tuple[str, ...]]] = {
    "group_panel": {
        "panel_background": (
            "group_panel_panel_background",
            "window",
            "panel",
            "dialog_background",
        ),
        "info_color": ("group_panel_info_color", "info_background"),
        "info_font": ("group_panel_info_font", "info_text"),
        "info_border": ("group_panel_info_border", "info_border"),
        "header_color": ("group_panel_header_color", "accent"),
        "header_font": ("group_panel_header_font", "accent_text"),
        "group_background": (
            "group_panel_group_background",
            "panel_alt",
            "section",
            "input_focus",
        ),
        "group_border": ("group_panel_group_border", "border"),
        "group_font": ("group_panel_group_font", "text"),
        "control_color": ("group_panel_control_color", "muted_text", "button_hover"),
        "control_active": ("group_panel_control_active", "accent_hover"),
        "control_disabled": ("group_panel_control_disabled", "button_disabled"),
        "input_color": ("group_panel_input_color", "input_background"),
        "input_font": ("group_panel_input_font", "input_text"),
        "button_background": ("group_panel_button_background", "button"),
        "scrollbar_border": ("group_panel_scrollbar_border", "scrollbar_border"),
        "scrollbar_trough": ("group_panel_scrollbar_trough", "scrollbar_trough"),
    },
    "group_settings": {
        "panel_background": ("group_settings_panel_background", "settings_panel"),
        "header_color": ("group_settings_header_color", "settings_accent"),
        "group_border": ("group_settings_group_border", "settings_border"),
        "control_color": ("group_settings_control_color", "settings_control"),
        "control_active": ("group_settings_control_active", "settings_control_active"),
        "control_disabled": ("group_settings_control_disabled", "settings_control_disabled"),
        "scrollbar_border": ("group_settings_scrollbar_border", "settings_scrollbar_border"),
        "scrollbar_trough": ("group_settings_scrollbar_trough", "settings_scrollbar_trough"),
        "tree_select": ("group_settings_tree_select", "settings_tree_select"),
        "tree_unselected": ("group_settings_tree_unselected", "settings_tree_unselected"),
        "tree_subheader": ("group_settings_tree_subheader", "settings_tree_subheader"),
        "link_color": ("group_settings_link_color", "settings_link"),
    },
    "command_tabs": {
        "frame_border": ("command_tabs_frame_border", "tab_border", "border"),
        "tab_color": ("command_tabs_tab_color", "tab_background", "panel_alt"),
        "tab_selected": ("command_tabs_tab_selected", "tab_selected", "accent"),
        "tab_hover": ("command_tabs_tab_hover", "tab_hover", "accent_hover"),
    },
    "console": {
        "background_color": ("console_background_color", "console_background"),
        "border_color": ("console_border_color", "console_border"),
        "stdout_color": ("console_stdout_color", "console_text", "stdout"),
        "stderr_color": ("console_stderr_color", "stderr"),
        "info_color": ("console_info_color", "info"),
        "verbose_color": ("console_verbose_color", "verbose"),
        "warning_color": ("console_warning_color", "warning"),
        "critical_color": ("console_critical_color", "critical"),
        "error_color": ("console_error_color", "error"),
        "scrollbar_border": ("console_scrollbar_border_key", "console_scrollbar_border"),
        "scrollbar_trough": ("console_scrollbar_trough_key", "console_scrollbar_trough"),
        "scrollbar_background_normal": (
            "console_scrollbar_background_normal_key",
            "console_scrollbar_background_normal",
        ),
        "scrollbar_background_disabled": (
            "console_scrollbar_background_disabled_key",
            "console_scrollbar_background_disabled",
        ),
        "scrollbar_background_active": (
            "console_scrollbar_background_active_key",
            "console_scrollbar_background_active",
        ),
        "scrollbar_foreground_normal": (
            "console_scrollbar_foreground_normal_key",
            "console_scrollbar_foreground_normal",
        ),
        "scrollbar_foreground_disabled": (
            "console_scrollbar_foreground_disabled_key",
            "console_scrollbar_foreground_disabled",
        ),
        "scrollbar_foreground_active": (
            "console_scrollbar_foreground_active_key",
            "console_scrollbar_foreground_active",
        ),
        "scrollbar_border_normal": (
            "console_scrollbar_border_normal_key",
            "console_scrollbar_border_normal",
        ),
        "scrollbar_border_disabled": (
            "console_scrollbar_border_disabled_key",
            "console_scrollbar_border_disabled",
        ),
        "scrollbar_border_active": (
            "console_scrollbar_border_active_key",
            "console_scrollbar_border_active",
        ),
    },
    "tooltip": {
        "background_color": ("tooltip_background_color", "tooltip_background"),
        "border_color": ("tooltip_border_color", "tooltip_border"),
        "font_color": ("tooltip_font_color", "tooltip_text"),
    },
}

_METADATA_KEYS = {"info"}


def _legacy_theme_colors(payload: T.Mapping[str, T.Any]) -> dict[str, str]:
    """Translate legacy Tk theme sections into Qt theme tokens when present."""
    translated: dict[str, str] = {}
    for section_name, mapping in _LEGACY_COLOR_TOKENS.items():
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        translated.update(_section_colors(section, mapping))
        _log_unknown_legacy_keys(section_name, section, mapping)
    return translated


def _section_colors(
    section: T.Mapping[object, object],
    mapping: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """Return valid theme token colors from one legacy section mapping."""
    colors: dict[str, str] = {}
    for legacy_key, tokens in mapping.items():
        value = section.get(legacy_key)
        if isinstance(value, str) and _HEX_COLOR_RE.match(value):
            colors.update({token: value for token in tokens})
    return colors


def _log_unknown_legacy_keys(
    section_name: str,
    section: T.Mapping[object, object],
    mapping: dict[str, tuple[str, ...]],
) -> None:
    """Log unsupported legacy JSON keys without rejecting permissive theme files."""
    for key in section:
        if not isinstance(key, str) or key in mapping or key in _METADATA_KEYS:
            continue
        logger.debug("Ignoring unknown legacy theme key: %s.%s", section_name, key)


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
