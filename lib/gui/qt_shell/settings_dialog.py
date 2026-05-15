#!/usr/bin/env python3
"""Qt settings dialog for Faceswap configuration parity."""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lib.config import get_configs
from lib.gui.qt_shell.command_panel import OptionsFormRenderer
from lib.gui.qt_shell.command_schema import OptionSpec
from lib.gui.qt_shell.theme import QtTheme, icon_for_action
from lib.gui.utils import PATH_CACHE, get_config
from lib.serializer import get_serializer
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SettingsSection:
    """One settings navigation section."""

    category: str
    section: str
    label: str
    helptext: str
    option_count: int

    @property
    def identifier(self) -> str:
        """Return the Tk-compatible tree identifier."""
        return (
            self.category
            if self.section == "global"
            else "|".join((self.category, *self.section.split(".")))
        )


@dataclass(slots=True)
class _OptionState:
    """Current editor state for one config option."""

    name: str
    option: T.Any
    spec: OptionSpec
    value: object

    @property
    def default(self) -> object:
        """Return the config default."""
        return self.option.default


class _SettingsPage(QWidget):
    """Config page rendered by the shared Qt option renderer."""

    def __init__(self, key: str, helptext: str, options: dict[str, _OptionState]) -> None:
        super().__init__()
        self.setObjectName(f"qt-shell-settings-page-{key.replace('|', '-')}")
        self._options = options
        self._renderer = OptionsFormRenderer()
        self._renderer.setObjectName("qt-shell-settings-options")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        if helptext:
            summary = QLabel(helptext)
            summary.setObjectName("qt-shell-settings-summary")
            summary.setWordWrap(True)
            summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout.addWidget(summary)
        self._renderer.set_options(tuple(option.spec for option in options.values()))
        self.sync_from_state()
        self._renderer.value_changed.connect(self.sync_to_state)
        layout.addWidget(self._renderer, 1)

    def sync_from_state(self) -> None:
        """Apply stored state to widgets."""
        self._renderer.apply_values({key: opt.value for key, opt in self._options.items()})

    def sync_to_state(self) -> None:
        """Store widget values in state."""
        for key, value in self._renderer.values().items():
            self._options[key].value = value


class _LinksPage(QWidget):
    """Navigation page for nodes that contain child sections only."""

    def __init__(
        self, key: str, links: T.Iterable[str], callback: T.Callable[[str], None]
    ) -> None:
        super().__init__()
        self.setObjectName(f"qt-shell-settings-links-{key.replace('|', '-')}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        header = QLabel("Select a plugin to configure:")
        header.setObjectName("qt-shell-settings-links-header")
        layout.addWidget(header)
        for link in sorted(set(links)):
            identifier = f"{key}|{link}"
            button = QPushButton(link.replace("_", " ").title())
            button.setObjectName("qt-shell-settings-link")
            button.setFlat(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, ident=identifier: callback(ident))
            layout.addWidget(button)
        layout.addStretch(1)


class SettingsDialog(QDialog):
    """Qt settings dialog matching Tk Configure Settings behavior."""

    gui_rebuild_requested = Signal()
    CATEGORY_ORDER = ("extract", "train", "convert")

    def __init__(
        self,
        section: str | None = None,
        parent: QWidget | None = None,
        config_provider: T.Callable[[], T.Mapping[str, T.Any]] = get_configs,
        preset_base_path: str | Path | None = None,
        serializer: T.Any | None = None,
        running_task_provider: T.Callable[[], bool] | None = None,
        gui_rebuild_callback: T.Callable[[], None] | None = None,
        theme: QtTheme | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt-shell-settings-dialog")
        self.setWindowTitle("Configure Settings")
        self.resize(600, 536)
        self._config_provider = config_provider
        self._configs = self._config_provider()
        self._preset_base_path = Path(preset_base_path or Path(PATH_CACHE) / "presets")
        self._serializer = serializer or get_serializer("json")
        self._running_task_provider = running_task_provider
        self._gui_rebuild_callback = gui_rebuild_callback
        self._theme = theme or QtTheme.default()
        self._sections = self._load_sections()
        self._options = self._load_options()
        self._cache: dict[str, QWidget] = {}
        self._page_action_buttons: dict[str, QPushButton] = {}
        self._footer_buttons: dict[str, QPushButton] = {}
        self._displayed_page: QWidget | None = None
        self._displayed_key: str | None = None
        self._header = QLabel("Settings")
        self._page_header = QLabel("Global")
        self._tree = QTreeWidget()
        self._page_host = QWidget()
        self._page_layout = QVBoxLayout(self._page_host)
        self._status = QLabel("Settings Ready")
        self._build_ui()
        self._select_initial_section(section)

    @property
    def sections(self) -> tuple[SettingsSection, ...]:
        """Return discovered settings sections."""
        return self._sections

    @property
    def selected_identifier(self) -> str | None:
        """Return the selected tree identifier."""
        item = self._tree.currentItem()
        data = None if item is None else item.data(0, Qt.UserRole)
        return data if isinstance(data, str) else None

    @property
    def displayed_key(self) -> str | None:
        """Return the visible page key."""
        return self._displayed_key

    def _build_ui(self) -> None:
        """Build the popup layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self._top_header())
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("qt-shell-settings-splitter")
        self._tree.setObjectName("qt-shell-settings-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(160)
        self._populate_tree()
        self._tree.currentItemChanged.connect(self._selection_changed)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._detail_area())
        splitter.setSizes([190, 410])
        layout.addWidget(splitter, 1)
        layout.addLayout(self._footer())
        self._status.setObjectName("qt-shell-settings-status")
        self._status.setProperty("status", "info")
        layout.addWidget(self._status)

    def _top_header(self) -> QWidget:
        """Return the Tk-like dialog header."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._header.setObjectName("qt-shell-settings-header")
        layout.addWidget(self._header)
        line = QFrame()
        line.setObjectName("qt-shell-settings-header-separator")
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        return widget

    def _detail_area(self) -> QWidget:
        """Return the right-side page area."""
        detail = QWidget()
        detail.setObjectName("qt-shell-settings-detail")
        layout = QVBoxLayout(detail)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._page_actions())
        scroll = QScrollArea()
        scroll.setObjectName("qt-shell-settings-scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        self._page_layout.setSpacing(0)
        scroll.setWidget(self._page_host)
        layout.addWidget(scroll, 1)
        return detail

    def _page_actions(self) -> QWidget:
        """Return the page header and preset buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._page_header.setObjectName("qt-shell-settings-page-header")
        layout.addWidget(self._page_header, 1)
        for text, icon_name, callback in (
            ("Load Preset", "open", self._load_preset),
            ("Save Preset", "save", self._save_preset),
        ):
            button = QPushButton(text)
            button.setObjectName(f"qt-shell-settings-{text.lower().replace(' ', '-')}")
            button.setIcon(icon_for_action(self._theme, icon_name))
            button.setToolTip(f"{text} for the selected settings page")
            button.clicked.connect(lambda _checked=False, func=callback: func())
            self._page_action_buttons[text] = button
            layout.addWidget(button)
        return widget

    def _footer(self) -> QHBoxLayout:
        """Return footer action buttons."""
        layout = QHBoxLayout()
        buttons = {
            "Save All": (
                "save",
                "Save all settings in the selected top-level category",
                lambda: self.save(page_only=False),
            ),
            "Reset All": (
                "clear",
                "Reset all settings in the selected top-level category",
                lambda: self.reset(page_only=False),
            ),
            "Cancel": ("clear", "Close without saving additional changes", self.close),
            "Save": ("save", "Save the selected settings page", lambda: self.save(page_only=True)),
            "Reset": (
                "reload",
                "Reset the selected settings page",
                lambda: self.reset(page_only=True),
            ),
        }
        for text, (icon_name, tooltip, callback) in buttons.items():
            button = QPushButton(text)
            button.setObjectName(f"qt-shell-settings-{text.lower().replace(' ', '-')}")
            button.setIcon(icon_for_action(self._theme, icon_name))
            button.setToolTip(tooltip)
            button.clicked.connect(lambda _checked=False, func=callback: func())
            self._footer_buttons[text] = button
        layout.addWidget(self._footer_buttons["Save All"])
        layout.addWidget(self._footer_buttons["Reset All"])
        layout.addStretch(1)
        layout.addWidget(self._footer_buttons["Cancel"])
        layout.addWidget(self._footer_buttons["Save"])
        layout.addWidget(self._footer_buttons["Reset"])
        return layout

    def _populate_tree(self) -> None:
        """Populate nested tree navigation."""
        self._tree.clear()
        roots: dict[str, QTreeWidgetItem] = {}
        for category in self._ordered_categories():
            item = QTreeWidgetItem([category.replace("_", " ").title()])
            item.setData(0, Qt.UserRole, category)
            item.setToolTip(0, self._tree_tooltip(category))
            self._tree.addTopLevelItem(item)
            roots[category] = item
        for category in self._ordered_categories():
            for section in sorted(self._configs[category].sections):
                parts = section.split(".")
                if parts == ["global"]:
                    continue
                self._insert_path(roots[category], category, parts)
        for index in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(index).setExpanded(True)

    def _insert_path(self, root: QTreeWidgetItem, category: str, parts: list[str]) -> None:
        """Insert one nested section path."""
        current = root
        prefix = [category]
        for part in parts:
            prefix.append(part)
            identifier = "|".join(prefix)
            child = self._child(current, identifier)
            if child is None:
                child = QTreeWidgetItem([part.replace("_", " ").title()])
                child.setData(0, Qt.UserRole, identifier)
                child.setToolTip(0, self._tree_tooltip(identifier))
                current.addChild(child)
            current = child

    def _tree_tooltip(self, identifier: str) -> str:
        """Return a compact tooltip for a settings tree item."""
        label = identifier.replace("|", " / ").replace("_", " ").title()
        options = self._options.get(identifier)
        if options is None:
            return label
        count = len(options)
        suffix = "option" if count == 1 else "options"
        return f"{label} ({count} {suffix})"

    @staticmethod
    def _child(parent: QTreeWidgetItem, identifier: str) -> QTreeWidgetItem | None:
        """Return child matching identifier."""
        for index in range(parent.childCount()):
            child = parent.child(index)
            if child.data(0, Qt.UserRole) == identifier:
                return child
        return None

    def _selection_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        """Update the visible page."""
        if current is None:
            return
        identifier = current.data(0, Qt.UserRole)
        if not isinstance(identifier, str):
            return
        category = identifier.split("|")[0]
        self._header.setText(f"{category.replace('_', ' ').title()} Settings")
        labels = ["global"] if "|" not in identifier else identifier.split("|")[1:]
        self._page_header.setText(" - ".join(label.replace("_", " ").title() for label in labels))
        self._set_display(identifier)

    def _set_display(self, key: str) -> None:
        """Display a cached settings or links page."""
        self._sync_displayed_page()
        if self._displayed_page is not None:
            self._page_layout.removeWidget(self._displayed_page)
            self._displayed_page.hide()
        self._cache.setdefault(key, self._create_page(key))
        self._displayed_page = self._cache[key]
        self._displayed_key = key
        if isinstance(self._displayed_page, _SettingsPage):
            self._displayed_page.sync_from_state()
        self._page_layout.addWidget(self._displayed_page)
        self._displayed_page.show()
        self._refresh_action_state()

    def _create_page(self, key: str) -> QWidget:
        """Create a direct option page or a links page."""
        if key in self._options:
            category, section = self._section_for_key(key)
            helptext = str(getattr(self._configs[category].sections[section], "helptext", ""))
            return _SettingsPage(key, helptext, self._options[key])
        prefix = f"{key}|"
        links = [
            item.removeprefix(prefix).split("|")[0]
            for item in self._options
            if item.startswith(prefix)
        ]
        return _LinksPage(key, links, self._link_callback)

    def _refresh_action_state(self) -> None:
        """Enable actions only when they apply to the visible selection."""
        selected = self.selected_identifier
        displayed = self._displayed_key
        page_has_options = bool(displayed and self._options.get(displayed))
        category_has_options = False
        if selected is not None:
            category_has_options = any(
                bool(self._options.get(key)) for key in self._target_keys(selected, False)
            )
        for button in self._page_action_buttons.values():
            button.setEnabled(page_has_options)
        for text in ("Save", "Reset"):
            if text in self._footer_buttons:
                self._footer_buttons[text].setEnabled(page_has_options)
        for text in ("Save All", "Reset All"):
            if text in self._footer_buttons:
                self._footer_buttons[text].setEnabled(category_has_options)

    def _link_callback(self, identifier: str) -> None:
        """Select a tree node from a links page."""
        item = self._tree_item(identifier)
        if item is not None:
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            self._tree.setCurrentItem(item)

    def _tree_item(self, identifier: str) -> QTreeWidgetItem | None:
        """Return a tree item by identifier."""

        def walk(item: QTreeWidgetItem) -> QTreeWidgetItem | None:
            if item.data(0, Qt.UserRole) == identifier:
                return item
            for index in range(item.childCount()):
                found = walk(item.child(index))
                if found is not None:
                    return found
            return None

        for index in range(self._tree.topLevelItemCount()):
            found = walk(self._tree.topLevelItem(index))
            if found is not None:
                return found
        return None

    def _select_initial_section(self, section: str | None) -> None:
        """Select the requested initial tree node."""
        wanted = section or (self._ordered_categories()[0] if self._sections else None)
        if wanted is None:
            return
        wanted = wanted.replace(".", "|")
        if "|" not in wanted and wanted not in self._ordered_categories():
            match = next(
                (item.identifier for item in self._sections if item.section == wanted), None
            )
            wanted = match or wanted
        item = self._tree_item(wanted)
        if item is None and self._tree.topLevelItemCount():
            item = self._tree.topLevelItem(0)
        if item is not None:
            self._tree.setCurrentItem(item)

    def reset(self, page_only: bool = False) -> None:
        """Reset current page or all options for the selected config."""
        self._sync_displayed_page()
        selected = self.selected_identifier
        if selected is None:
            self._set_status("No settings selected", "warning")
            return
        keys = self._target_keys(selected, page_only)
        if not keys:
            self._set_status("No configuration options to reset for current page", "info")
            return
        for key in keys:
            for state in self._options[key].values():
                state.value = state.default
            page = self._cache.get(key)
            if isinstance(page, _SettingsPage):
                page.sync_from_state()
        self._set_status(
            "Reset page settings" if page_only else f"Reset {selected.split('|')[0]} settings",
            "success",
        )

    def save(self, page_only: bool = False) -> None:
        """Save current page or all options for the selected config."""
        self._sync_displayed_page()
        selected = self.selected_identifier
        if selected is None:
            self._set_status("No settings selected", "warning")
            return
        category, lookup = self._section_for_key(selected)
        config = self._configs[category]
        if page_only and lookup not in config.sections:
            logger.info("No settings to save for the current page")
            self._set_status("No settings to save for the current page", "info")
            return
        if not self._update_config(config, category, lookup, page_only):
            logger.info("No config changes to save")
            self._set_status("No config changes to save", "info")
            return
        config.save_config()
        self._set_status("Saved settings", "success")
        if category == "gui":
            self._handle_gui_rebuild()

    def _update_config(self, config: T.Any, category: str, lookup: str, page_only: bool) -> bool:
        """Push edited state into the config object."""
        updated = False
        for section_name, section in config.sections.items():
            if page_only and section_name != lookup:
                continue
            key = self._key(category, section_name)
            states = self._options.get(key, {})
            for option_name, option in section.options.items():
                new_value = self._coerce(option, states[option_name].value)
                if self._equal(new_value, option.value):
                    continue
                logger.debug(
                    "Updating '%s' from %s to %s", option_name, repr(option.value), repr(new_value)
                )
                option.set(new_value)
                states[option_name].value = option.value
                updated = True
        return updated

    def _target_keys(self, selected: str, page_only: bool) -> tuple[str, ...]:
        """Return keys affected by a reset/save action."""
        if page_only:
            return (selected,) if selected in self._options else ()
        category = selected.split("|")[0]
        return tuple(
            key for key in self._options if key == category or key.startswith(f"{category}|")
        )

    def _load_preset(self) -> None:
        """Load preset data into the current page."""
        filename = self._preset_filename("load")
        if filename is None:
            return
        try:
            preset = self._serializer.load(str(filename))
        except (OSError, TypeError, ValueError) as err:
            logger.warning("Could not load preset file '%s': %s", filename, err)
            self._set_status("Could not load the selected preset file", "error")
            return
        if not isinstance(preset, dict) or preset.get("__filetype") != "faceswap_preset":
            logger.warning("'%s' is not a valid plugin preset file", filename)
            self._set_status("Selected file is not a valid Faceswap preset", "error")
            return
        if preset.get("__section") != self._full_preset_key:
            logger.warning(
                "Preset section '%s' does not match '%s'",
                preset.get("__section"),
                self._full_preset_key,
            )
            self._set_status("Preset section does not match the current page", "error")
            return
        page_options = self._options[self._displayed_key or ""]
        for key, value in preset.items():
            if not key.startswith("__") and key in page_options:
                page_options[key].value = value
        if isinstance(self._displayed_page, _SettingsPage):
            self._displayed_page.sync_from_state()
        self._set_status(f"Preset loaded from: '{filename.name}'", "success")

    def _save_preset(self) -> None:
        """Save the current page to a preset file."""
        self._sync_displayed_page()
        filename = self._preset_filename("save")
        if filename is None:
            return
        page_options = self._options[self._displayed_key or ""]
        preset = {key: state.value for key, state in page_options.items()}
        preset.update({"__filetype": "faceswap_preset", "__section": self._full_preset_key})
        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
            self._serializer.save(str(filename), preset)
        except (OSError, TypeError, ValueError) as err:
            logger.warning("Could not save preset file '%s': %s", filename, err)
            self._set_status("Could not save preset", "error")
            return
        self._set_status(f"Preset '{self._full_preset_key}' saved", "success")

    def _preset_filename(self, action: T.Literal["load", "save"]) -> Path | None:
        """Prompt for a preset filename."""
        displayed = self._displayed_key
        if displayed is None or displayed not in self._options:
            self._set_status(f"No settings to {action} for the current page", "info")
            return None
        if not self._options[displayed]:
            self._set_status(f"No settings to {action} for the current page", "info")
            return None
        preset_path = self._preset_base_path / displayed.split("|")[0]
        if action == "save":
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Preset...",
                str(preset_path / self._initial_preset_filename()),
                "JSON files (*.json);;All files (*)",
            )
        else:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Preset...", str(preset_path), "JSON files (*.json);;All files (*)"
            )
        return Path(filename) if filename else None

    @property
    def _full_preset_key(self) -> str:
        """Return the Tk-compatible preset section key."""
        assert self._displayed_key is not None
        return (
            self._displayed_key if "|" in self._displayed_key else f"{self._displayed_key}|global"
        )

    def _initial_preset_filename(self) -> str:
        """Return a non-conflicting initial preset filename."""
        _, key = self._full_preset_key.split("|", 1)
        preset_path = self._preset_base_path / self._full_preset_key.split("|")[0]
        base = f"{key.replace('|', '_')}_preset"
        index = 0
        filename = f"{base}.json"
        while (preset_path / filename).exists():
            filename = f"{base}_{index}.json"
            index += 1
        return filename

    def _handle_gui_rebuild(self) -> None:
        """Rebuild now or defer when a task is active."""
        if self._is_task_running():
            logger.info(
                "Can't redraw GUI whilst a task is running. "
                "GUI Settings will be applied at the next restart."
            )
            self._set_status(
                "Saved GUI settings. Rebuild deferred until restart because a task is running.",
                "warning",
            )
            return
        self.gui_rebuild_requested.emit()
        if self._gui_rebuild_callback is not None:
            self._gui_rebuild_callback()
            return
        rebuild = getattr(self.parent(), "rebuild", None)
        if callable(rebuild):
            rebuild()
            return
        try:
            get_config().root.rebuild()  # type:ignore[attr-defined]
        except (AssertionError, AttributeError):
            logger.debug("No GUI rebuild callback is available")

    def _is_task_running(self) -> bool:
        """Return whether a task is currently running."""
        if self._running_task_provider is not None:
            return self._running_task_provider()
        if self.parent() is not None and hasattr(self.parent(), "_running"):
            return bool(self.parent()._running)
        try:
            return bool(get_config().tk_vars.running_task.get())
        except (AssertionError, AttributeError):
            return False

    def _sync_displayed_page(self) -> None:
        """Store visible widget values."""
        if isinstance(self._displayed_page, _SettingsPage):
            self._displayed_page.sync_to_state()

    def _set_status(self, text: str, status: str) -> None:
        """Set the status label and refresh style properties."""
        self._status.setText(text)
        self._status.setProperty("status", status)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _load_sections(self) -> tuple[SettingsSection, ...]:
        """Load section metadata."""
        return tuple(
            SettingsSection(
                category=category,
                section=name,
                label=name.split(".")[-1].replace("_", " ").title(),
                helptext=str(getattr(section, "helptext", "")),
                option_count=len(getattr(section, "options", {})),
            )
            for category, config in self._configs.items()
            for name, section in config.sections.items()
        )

    def _load_options(self) -> dict[str, dict[str, _OptionState]]:
        """Load editable option state."""
        retval: dict[str, dict[str, _OptionState]] = {}
        for category, config in self._configs.items():
            for section_name, section in config.sections.items():
                key = self._key(category, section_name)
                retval[key] = {
                    name: _OptionState(name, option, self._spec(name, option), option.value)
                    for name, option in section.options.items()
                }
        return retval

    @staticmethod
    def _spec(name: str, option: T.Any) -> OptionSpec:
        """Adapt a Faceswap ConfigItem to a Qt OptionSpec."""
        datatype = getattr(option, "datatype", str)
        choices = getattr(option, "choices", ())
        choice_tuple = tuple(choices) if isinstance(choices, (list, tuple)) else ()
        min_max = getattr(option, "min_max", None)
        rounding = getattr(option, "rounding", -1)
        is_slider = datatype in (int, float) and min_max is not None and rounding > 0
        return OptionSpec(
            title=name.replace("_", " ").title(),
            switch=name,
            value_type=str if datatype is list else datatype,
            default=option.default,
            choices=choice_tuple,
            action="Slider" if is_slider else None,
            group=getattr(option, "group", None),
            helptext=str(getattr(option, "helptext", "")),
            is_radio=bool(getattr(option, "gui_radio", False)),
            is_multi_option=datatype is list,
            slider_min=min_max[0] if is_slider else None,
            slider_max=min_max[1] if is_slider else None,
            slider_rounding=rounding if is_slider else None,
        )

    @staticmethod
    def _key(category: str, section_name: str) -> str:
        """Return display key for a config section."""
        parts = section_name.split(".")
        return category if parts == ["global"] else "|".join((category, *parts))

    @staticmethod
    def _section_for_key(key: str) -> tuple[str, str]:
        """Return config category and section for a display key."""
        parts = key.split("|")
        return (parts[0], "global") if len(parts) == 1 else (parts[0], ".".join(parts[1:]))

    @staticmethod
    def _coerce(option: T.Any, value: object) -> object:
        """Coerce a UI value for ConfigItem.set."""
        datatype = option.datatype
        if datatype is list:
            return value if isinstance(value, list) else str(value)
        if datatype is bool:
            return bool(value)
        if datatype is int:
            return int(value)
        if datatype is float:
            return float(value)
        return "" if value is None else str(value)

    @staticmethod
    def _equal(new_value: object, current_value: object) -> bool:
        """Return whether UI and config values are equivalent."""
        if isinstance(current_value, list):
            new_items = new_value if isinstance(new_value, list) else str(new_value).split()
            return set(str(item).lower() for item in new_items) == set(current_value)
        return new_value == current_value

    def _ordered_categories(self) -> list[str]:
        """Return Tk-compatible category ordering."""
        categories = {section.category for section in self._sections}
        ordered = [category for category in self.CATEGORY_ORDER if category in categories]
        ordered.extend(sorted(category for category in categories if category not in ordered))
        return ordered


class _State:
    """Hold the singleton settings dialog."""

    def __init__(self) -> None:
        self._dialog: SettingsDialog | None = None

    def open_popup(self, name: str | None = None, parent: QWidget | None = None) -> SettingsDialog:
        """Open or raise the singleton dialog."""
        if self._dialog is None:
            self._dialog = SettingsDialog(section=name, parent=parent)
            self._dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            self._dialog.destroyed.connect(self._clear_dialog)
        elif name is not None:
            self._dialog._select_initial_section(name)  # pylint:disable=protected-access
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()
        return self._dialog

    def close_popup(self) -> None:
        """Close the singleton dialog."""
        if self._dialog is not None:
            dialog = self._dialog
            self._dialog = None
            dialog.close()

    def _clear_dialog(self, _obj: object | None = None) -> None:
        """Forget a destroyed dialog."""
        self._dialog = None


_STATE = _State()
open_popup = _STATE.open_popup


__all__ = get_module_objects(__name__)
