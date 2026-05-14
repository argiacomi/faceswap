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
from lib.gui.utils import PATH_CACHE, get_config
from lib.serializer import get_serializer
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

ConfigMapping = T.Mapping[str, T.Any]
Serializer = T.Any


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
        """Return the Tk-compatible section identifier used by the settings tree."""
        if self.section == "global":
            return self.category
        return "|".join((self.category, *self.section.split(".")))


@dataclass(slots=True)
class _OptionState:
    """Mutable UI state for one persisted config option."""

    name: str
    config_option: T.Any
    spec: OptionSpec
    value: object

    @property
    def default(self) -> object:
        """Return the default value for this option."""
        return self.config_option.default

    def reset(self) -> None:
        """Reset this state to the option default."""
        self.value = self.default


class _SettingsPage(QWidget):
    """A cached settings page backed by ``OptionsFormRenderer``."""

    def __init__(
        self,
        key: str,
        helptext: str,
        options: dict[str, _OptionState],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(f"qt-shell-settings-page-{key.replace('|', '-')}")
        self._key = key
        self._options = options
        self._renderer = OptionsFormRenderer()
        self._renderer.setObjectName("qt-shell-settings-options")
        self._build(helptext)
        self._renderer.value_changed.connect(self.sync_to_state)

    @property
    def renderer(self) -> OptionsFormRenderer:
        """Return the page's option renderer."""
        return self._renderer

    def sync_from_state(self) -> None:
        """Apply stored option state to visible widgets."""
        self._renderer.apply_values({name: option.value for name, option in self._options.items()})

    def sync_to_state(self) -> None:
        """Store visible widget values into option state."""
        values = self._renderer.values()
        for name, value in values.items():
            self._options[name].value = value

    def _build(self, helptext: str) -> None:
        """Build page widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        if helptext:
            summary = QLabel(helptext)
            summary.setObjectName("qt-shell-settings-summary")
            summary.setWordWrap(True)
            summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout.addWidget(summary)
        self._renderer.set_options(tuple(option.spec for option in self._options.values()))
        self.sync_from_state()
        layout.addWidget(self._renderer, 1)


class _LinksPage(QWidget):
    """A cached links page for tree nodes without direct settings."""

    def __init__(
        self,
        key: str,
        links: T.Iterable[str],
        callback: T.Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(f"qt-shell-settings-links-{key.replace('|', '-')}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
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
    """Qt settings dialog matching the Tk Configure Settings popup."""

    gui_rebuild_requested = Signal()

    CATEGORY_ORDER = ("extract", "train", "convert")

    def __init__(
        self,
        section: str | None = None,
        parent: QWidget | None = None,
        config_provider: T.Callable[[], ConfigMapping] = get_configs,
        preset_base_path: str | Path | None = None,
        serializer: Serializer | None = None,
        running_task_provider: T.Callable[[], bool] | None = None,
        gui_rebuild_callback: T.Callable[[], None] | None = None,
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
        self._sections = self._load_sections()
        self._options = self._load_options()
        self._header = QLabel("Settings")
        self._page_header = QLabel("Global")
        self._tree = QTreeWidget()
        self._page_host = QWidget()
        self._page_layout = QVBoxLayout(self._page_host)
        self._status = QLabel("Settings Ready")
        self._cache: dict[str, QWidget] = {}
        self._displayed_page: QWidget | None = None
        self._displayed_key: str | None = None
        self._build_ui()
        self._select_initial_section(section)

    @property
    def sections(self) -> tuple[SettingsSection, ...]:
        """Return discovered settings sections."""
        return self._sections

    @property
    def selected_identifier(self) -> str | None:
        """Return the selected settings tree identifier."""
        item = self._tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        return data if isinstance(data, str) else None

    @property
    def displayed_key(self) -> str | None:
        """Return the displayed page lookup key."""
        return self._displayed_key

    def _build_ui(self) -> None:
        """Build the settings dialog layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_top_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("qt-shell-settings-splitter")
        self._tree.setObjectName("qt-shell-settings-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(160)
        self._populate_tree()
        self._tree.currentItemChanged.connect(self._selection_changed)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._build_detail_area())
        splitter.setSizes([190, 410])
        layout.addWidget(splitter, 1)

        layout.addLayout(self._build_footer())
        self._status.setObjectName("qt-shell-settings-status")
        self._status.setProperty("status", "info")
        layout.addWidget(self._status)

    def _build_top_header(self) -> QWidget:
        """Build the top-level header and separator."""
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

    def _build_detail_area(self) -> QWidget:
        """Build the right-side detail area."""
        detail = QWidget()
        detail.setObjectName("qt-shell-settings-detail")
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.setSpacing(8)
        detail_layout.addWidget(self._build_page_header())

        scroll = QScrollArea()
        scroll.setObjectName("qt-shell-settings-scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        self._page_layout.setSpacing(0)
        scroll.setWidget(self._page_host)
        detail_layout.addWidget(scroll, 1)
        return detail

    def _build_page_header(self) -> QWidget:
        """Build the displayed page header with preset actions."""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        self._page_header.setObjectName("qt-shell-settings-page-header")
        header_layout.addWidget(self._page_header, 1)
        load = QPushButton("Load Preset")
        save = QPushButton("Save Preset")
        load.setObjectName("qt-shell-settings-load-preset")
        save.setObjectName("qt-shell-settings-save-preset")
        load.setToolTip("Load preset for this plugin")
        save.setToolTip("Save preset for this plugin")
        load.clicked.connect(lambda _checked=False: self._load_preset())
        save.clicked.connect(lambda _checked=False: self._save_preset())
        header_layout.addWidget(load)
        header_layout.addWidget(save)
        return header

    def _build_footer(self) -> QHBoxLayout:
        """Build footer buttons matching the Tk popup action layout."""
        footer = QHBoxLayout()
        save_all = QPushButton("Save All")
        reset_all = QPushButton("Reset All")
        cancel = QPushButton("Cancel")
        save = QPushButton("Save")
        reset = QPushButton("Reset")
        for button in (save_all, reset_all, cancel, save, reset):
            button.setObjectName(f"qt-shell-settings-{button.text().lower().replace(' ', '-')}")
        save_all.setToolTip("Save all settings for the currently selected config")
        reset_all.setToolTip("Reset all settings for the currently selected config to default values")
        cancel.setToolTip("Close without saving")
        save.setToolTip("Save this page's config")
        reset.setToolTip("Reset this page's config to default values")
        save_all.clicked.connect(lambda _checked=False: self.save(page_only=False))
        reset_all.clicked.connect(lambda _checked=False: self.reset(page_only=False))
        cancel.clicked.connect(self.close)
        save.clicked.connect(lambda _checked=False: self.save(page_only=True))
        reset.clicked.connect(lambda _checked=False: self.reset(page_only=True))
        footer.addWidget(save_all)
        footer.addWidget(reset_all)
        footer.addStretch(1)
        footer.addWidget(cancel)
        footer.addWidget(save)
        footer.addWidget(reset)
        return footer

    def _populate_tree(self) -> None:
        """Populate settings navigation from Faceswap config metadata."""
        self._tree.clear()
        category_items: dict[str, QTreeWidgetItem] = {}
        for category in self._ordered_categories():
            parent = QTreeWidgetItem([category.replace("_", " ").title()])
            parent.setData(0, Qt.UserRole, category)
            self._tree.addTopLevelItem(parent)
            category_items[category] = parent

        for category in self._ordered_categories():
            sections = sorted(self._configs[category].sections)
            for section_name in sections:
                parts = section_name.split(".")
                if parts[-1] == "global" and len(parts) == 1:
                    continue
                self._insert_section_parts(category_items[category], category, parts)

        for index in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(index).setExpanded(True)

    def _insert_section_parts(
        self,
        parent: QTreeWidgetItem,
        category: str,
        parts: list[str],
    ) -> None:
        """Insert nested tree items for a dot-delimited config section."""
        current = parent
        prefix = [category]
        for part in parts:
            prefix.append(part)
            identifier = "|".join(prefix)
            child = self._child_by_identifier(current, identifier)
            if child is None:
                child = QTreeWidgetItem([part.replace("_", " ").title()])
                child.setData(0, Qt.UserRole, identifier)
                current.addChild(child)
            current = child

    @staticmethod
    def _child_by_identifier(
        parent: QTreeWidgetItem,
        identifier: str,
    ) -> QTreeWidgetItem | None:
        """Return a child item for ``identifier`` if it exists."""
        for index in range(parent.childCount()):
            child = parent.child(index)
            if child.data(0, Qt.UserRole) == identifier:
                return child
        return None

    def _selection_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        """Update the displayed settings page when navigation changes."""
        if current is None:
            return
        identifier = current.data(0, Qt.UserRole)
        if not isinstance(identifier, str):
            return
        category = identifier.split("|")[0]
        self._header.setText(f"{category.replace('_', ' ').title()} Settings")
        labels = ["global"] if "|" not in identifier else identifier.split("|")[1:]
        self._page_header.setText(
            " - ".join(label.replace("_", " ").title() for label in labels)
        )
        self._set_display(identifier)

    def _set_display(self, key: str) -> None:
        """Set the visible cached page for ``key``."""
        self._sync_displayed_page()
        if self._displayed_page is not None:
            self._page_layout.removeWidget(self._displayed_page)
            self._displayed_page.hide()

        if key not in self._cache:
            self._cache[key] = self._create_page(key)

        page = self._cache[key]
        self._displayed_page = page
        self._displayed_key = key
        if isinstance(page, _SettingsPage):
            page.sync_from_state()
        self._page_layout.addWidget(page)
        page.show()

    def _create_page(self, key: str) -> QWidget:
        """Create a settings or links page for ``key``."""
        page_options = self._options.get(key)
        if page_options is not None:
            helptext = self._helptext_for_key(key)
            return _SettingsPage(key, helptext, page_options, self._page_host)
        return _LinksPage(key, self._links_for_key(key), self._link_callback, self._page_host)

    def _link_callback(self, identifier: str) -> None:
        """Select a tree item from a links page click."""
        item = self._item_for_identifier(identifier)
        if item is None:
            return
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self._tree.setCurrentItem(item)

    def _item_for_identifier(self, identifier: str) -> QTreeWidgetItem | None:
        """Return the tree item for ``identifier``."""
        for index in range(self._tree.topLevelItemCount()):
            found = self._find_item(self._tree.topLevelItem(index), identifier)
            if found is not None:
                return found
        return None

    def _find_item(
        self,
        item: QTreeWidgetItem,
        identifier: str,
    ) -> QTreeWidgetItem | None:
        """Depth-first search for a tree item identifier."""
        if item.data(0, Qt.UserRole) == identifier:
            return item
        for index in range(item.childCount()):
            found = self._find_item(item.child(index), identifier)
            if found is not None:
                return found
        return None

    def _links_for_key(self, key: str) -> tuple[str, ...]:
        """Return child link labels for a non-option tree node."""
        prefix = f"{key}|"
        links = {
            option_key.removeprefix(prefix).split("|")[0]
            for option_key in self._options
            if option_key.startswith(prefix)
        }
        return tuple(links)

    def _helptext_for_key(self, key: str) -> str:
        """Return help text for a direct settings page."""
        category, section_name = self._section_lookup_for_key(key)
        return str(getattr(self._configs[category].sections[section_name], "helptext", ""))

    def _section_lookup_for_key(self, key: str) -> tuple[str, str]:
        """Return ``(category, section_name)`` for a tree/config key."""
        parts = key.split("|")
        if len(parts) == 1:
            return parts[0], "global"
        return parts[0], ".".join(parts[1:])

    def _select_initial_section(self, section: str | None) -> None:
        """Select an initial category/section, matching Tk ``open_popup(name=...)``."""
        wanted = section or (self._ordered_categories()[0] if self._sections else None)
        if wanted is None:
            return
        wanted = wanted.replace(".", "|")
        if "|" not in wanted and wanted not in self._ordered_categories():
            matches = [sec.identifier for sec in self._sections if sec.section == wanted]
            wanted = matches[0] if matches else wanted
        item = self._item_for_identifier(wanted)
        if item is not None:
            self._tree.setCurrentItem(item)
            return
        if self._tree.topLevelItemCount():
            self._tree.setCurrentItem(self._tree.topLevelItem(0))

    def reset(self, page_only: bool = False) -> None:
        """Reset the current page or selected config to default values."""
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
            for option in self._options[key].values():
                logger.debug(
                    "Resetting item '%s' from '%s' to default '%s'",
                    option.name,
                    option.value,
                    option.default,
                )
                option.reset()
            page = self._cache.get(key)
            if isinstance(page, _SettingsPage):
                page.sync_from_state()
        label = "page" if page_only else selected.split("|")[0]
        self._set_status(f"Reset {label} settings", "success")

    def save(self, page_only: bool = False) -> None:
        """Save the current page or selected config to disk."""
        self._sync_displayed_page()
        selected = self.selected_identifier
        if selected is None:
            self._set_status("No settings selected", "warning")
            return
        category = selected.split("|")[0]
        config = self._configs[category]
        lookup = self._section_lookup_for_key(selected)[1]
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

    def _target_keys(self, selected: str, page_only: bool) -> tuple[str, ...]:
        """Return option keys targeted by page/all actions."""
        if page_only:
            return (selected,) if selected in self._options else ()
        category = selected.split("|")[0]
        return tuple(key for key in self._options if key == category or key.startswith(f"{category}|"))

    def _update_config(
        self,
        config: T.Any,
        category: str,
        current_section: str,
        page_only: bool,
    ) -> bool:
        """Update a FaceswapConfig from currently stored UI option state."""
        updated = False
        for section_name, section in config.sections.items():
            if page_only and section_name != current_section:
                logger.debug("Skipping section '%s' for page_only save", section_name)
                continue
            key = self._key_from_section_name(category, section_name)
            gui_options = self._options.get(key)
            if gui_options is None:
                continue
            for option_name, option in section.options.items():
                state = gui_options[option_name]
                new_value = self._coerce_for_config(option, state.value)
                if self._values_equal(new_value, option.value):
                    logger.debug("Skipping unchanged option '%s'", option_name)
                    continue
                logger.debug(
                    "Updating '%s' from %s to %s",
                    option_name,
                    repr(option.value),
                    repr(new_value),
                )
                option.set(new_value)
                state.value = option.value
                updated = True
        return updated

    def _handle_gui_rebuild(self) -> None:
        """Rebuild the GUI now or defer when a task is running."""
        if self._is_task_running():
            logger.info(
                "Can't redraw GUI whilst a task is running. GUI Settings will be applied at the "
                "next restart."
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
        parent = self.parent()
        rebuild = getattr(parent, "rebuild", None)
        if callable(rebuild):
            rebuild()
            return
        try:
            get_config().root.rebuild()  # type:ignore[attr-defined]
        except (AssertionError, AttributeError):
            logger.debug("No GUI rebuild callback is available")

    def _is_task_running(self) -> bool:
        """Return whether a Faceswap task is currently running."""
        if self._running_task_provider is not None:
            return self._running_task_provider()
        parent = self.parent()
        if parent is not None and hasattr(parent, "_running"):
            return bool(getattr(parent, "_running"))
        try:
            return bool(get_config().tk_vars.running_task.get())
        except (AssertionError, AttributeError):
            return False

    def _load_preset(self) -> None:
        """Load a preset into the displayed settings page."""
        filename = self._preset_filename("load")
        if filename is None:
            return
        try:
            preset = self._serializer.load(str(filename))
        except OSError as err:
            logger.warning("Unable to load preset '%s': %s", filename, err)
            self._set_status(f"Unable to load preset: {err}", "error")
            return
        if preset.get("__filetype") != "faceswap_preset":
            logger.warning("'%s' is not a valid plugin preset file", filename)
            self._set_status("Selected file is not a valid Faceswap preset", "error")
            return
        if preset.get("__section") != self._full_preset_key:
            logger.warning(
                "You are attempting to load a preset for '%s' into '%s'. Aborted.",
                preset.get("__section", "no section"),
                self._full_preset_key,
            )
            self._set_status("Preset section does not match the current page", "error")
            return
        page_options = self._options[self._displayed_key or ""]
        for key, value in preset.items():
            if key.startswith("__") or key not in page_options:
                logger.debug("Skipping non-existent preset item: '%s'", key)
                continue
            page_options[key].value = value
        page = self._cache.get(self._displayed_key or "")
        if isinstance(page, _SettingsPage):
            page.sync_from_state()
        self._set_status(f"Preset loaded from: '{filename.name}'", "success")

    def _save_preset(self) -> None:
        """Save the displayed settings page to a preset file."""
        self._sync_displayed_page()
        filename = self._preset_filename("save")
        if filename is None:
            return
        page_options = self._options[self._displayed_key or ""]
        preset = {name: option.value for name, option in page_options.items()}
        preset["__filetype"] = "faceswap_preset"
        preset["__section"] = self._full_preset_key
        filename.parent.mkdir(parents=True, exist_ok=True)
        self._serializer.save(str(filename), preset)
        self._set_status(f"Preset '{self._full_preset_key}' saved", "success")

    def _preset_filename(self, action: T.Literal["load", "save"]) -> Path | None:
        """Return a preset filename from the user for load/save."""
        displayed = self._displayed_key
        if displayed is None or displayed not in self._options:
            logger.info("No settings to %s for the current page.", action)
            self._set_status(f"No settings to {action} for the current page", "info")
            return None
        preset_path = self._preset_path
        if action == "save":
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Preset...",
                str(preset_path / self._initial_preset_filename()),
                "JSON files (*.json);;All files (*)",
            )
        else:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load Preset...",
                str(preset_path),
                "JSON files (*.json);;All files (*)",
            )
        return Path(filename) if filename else None

    @property
    def _preset_path(self) -> Path:
        """Return the default preset path for the displayed plugin."""
        assert self._displayed_key is not None
        return self._preset_base_path / self._displayed_key.split("|")[0]

    @property
    def _full_preset_key(self) -> str:
        """Return the Tk-compatible full preset section key."""
        assert self._displayed_key is not None
        return self._displayed_key if "|" in self._displayed_key else f"{self._displayed_key}|global"

    def _initial_preset_filename(self) -> str:
        """Return a non-conflicting preset filename for the displayed page."""
        _, key = self._full_preset_key.split("|", 1)
        base_filename = f"{key.replace('|', '_')}_preset"
        index = 0
        filename = f"{base_filename}.json"
        while (self._preset_path / filename).exists():
            logger.debug("File pre-exists: %s", filename)
            filename = f"{base_filename}_{index}.json"
            index += 1
        return filename

    def _sync_displayed_page(self) -> None:
        """Persist visible page values into option state."""
        if isinstance(self._displayed_page, _SettingsPage):
            self._displayed_page.sync_to_state()

    def _set_status(self, text: str, status: str) -> None:
        """Set a styled status message."""
        self._status.setText(text)
        self._status.setProperty("status", status)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _load_sections(self) -> tuple[SettingsSection, ...]:
        """Load settings sections from Faceswap config metadata."""
        sections: list[SettingsSection] = []
        for category, config in self._configs.items():
            for section_name, section in config.sections.items():
                parts = section_name.split(".")
                label = parts[-1].replace("_", " ").title()
                option_count = len(getattr(section, "options", {}))
                sections.append(
                    SettingsSection(
                        category=category,
                        section=section_name,
                        label=label,
                        helptext=str(getattr(section, "helptext", "")),
                        option_count=option_count,
                    )
                )
        return tuple(sections)

    def _load_options(self) -> dict[str, dict[str, _OptionState]]:
        """Load mutable option state from Faceswap config metadata."""
        options: dict[str, dict[str, _OptionState]] = {}
        for category, config in self._configs.items():
            for section_name, section in config.sections.items():
                key = self._key_from_section_name(category, section_name)
                page_options: dict[str, _OptionState] = {}
                for option_name, option in section.options.items():
                    spec = self._spec_from_config_option(option_name, option)
                    page_options[option_name] = _OptionState(
                        name=option_name,
                        config_option=option,
                        spec=spec,
                        value=option.value,
                    )
                options[key] = page_options
        return options

    @classmethod
    def _spec_from_config_option(cls, name: str, option: T.Any) -> OptionSpec:
        """Return an ``OptionSpec`` adapted from a Faceswap config item."""
        datatype = getattr(option, "datatype", str)
        choices = getattr(option, "choices", ())
        choice_tuple = tuple(choices) if isinstance(choices, (list, tuple)) else ()
        min_max = getattr(option, "min_max", None)
        rounding = getattr(option, "rounding", -1)
        is_slider = (
            datatype in (int, float)
            and min_max is not None
            and isinstance(rounding, (int, float))
            and rounding > 0
        )
        value_type = str if datatype is list else datatype
        return OptionSpec(
            title=name.replace("_", " ").title(),
            switch=name,
            value_type=value_type,
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
    def _key_from_section_name(category: str, section_name: str) -> str:
        """Return the Tk-compatible display key for a config section name."""
        section_parts = section_name.split(".")
        section_category = section_parts[0]
        section = section_parts[-1]
        if section == "global":
            return category
        return "|".join((category, section_category, section))

    @staticmethod
    def _coerce_for_config(option: T.Any, value: object) -> object:
        """Coerce a renderer value to the option datatype before persistence."""
        datatype = option.datatype
        if datatype is list:
            return value if isinstance(value, list) else str(value)
        if datatype is bool:
            return bool(value)
        if datatype is int:
            return int(value)
        if datatype is float:
            return float(value)
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _values_equal(new_value: object, current_value: object) -> bool:
        """Return whether a UI value matches the persisted config value."""
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
    """Hold the singleton Qt settings dialog instance."""

    def __init__(self) -> None:
        self._dialog: SettingsDialog | None = None

    def open_popup(self, name: str | None = None, parent: QWidget | None = None) -> SettingsDialog:
        """Launch the settings dialog, ensuring only one instance is open."""
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
        """Close and forget the tracked settings dialog."""
        if self._dialog is None:
            return
        dialog = self._dialog
        self._dialog = None
        dialog.close()

    def _clear_dialog(self, _obj: object | None = None) -> None:
        """Forget the tracked dialog after Qt destroys it."""
        self._dialog = None


_STATE = _State()
open_popup = _STATE.open_popup


__all__ = get_module_objects(__name__)
