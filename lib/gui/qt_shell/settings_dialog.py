#!/usr/bin/env python3
"""Qt settings dialog for Faceswap configuration parity."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lib.config import get_configs
from lib.utils import get_module_objects


@dataclass(frozen=True)
class SettingsSection:
    """One settings navigation section."""

    category: str
    section: str
    label: str
    helptext: str
    option_count: int

    @property
    def identifier(self) -> str:
        """Return the section identifier used by the settings tree."""
        return self.category if self.section == "global" else f"{self.category}|{self.section}"


class SettingsDialog(QDialog):
    """Qt settings dialog matching the Tk Configure Settings entry point."""

    CATEGORY_ORDER = ("extract", "train", "convert")

    def __init__(
        self,
        section: str | None = None,
        parent: QWidget | None = None,
        config_provider: T.Callable[[], T.Mapping[str, T.Any]] = get_configs,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt-shell-settings-dialog")
        self.setWindowTitle("Configure Settings")
        self.resize(600, 536)
        self._config_provider = config_provider
        self._sections = self._load_sections()
        self._header = QLabel("Settings")
        self._summary = QLabel("Select a settings section to configure.")
        self._tree = QTreeWidget()
        self._status = QLabel("Settings Ready")
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

    def _build_ui(self) -> None:
        """Build the settings dialog layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)

        self._header.setObjectName("qt-shell-settings-header")
        layout.addWidget(self._header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("qt-shell-settings-splitter")
        self._tree.setObjectName("qt-shell-settings-tree")
        self._tree.setHeaderHidden(True)
        self._populate_tree()
        self._tree.currentItemChanged.connect(self._selection_changed)
        splitter.addWidget(self._tree)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        self._summary.setObjectName("qt-shell-settings-summary")
        self._summary.setWordWrap(True)
        detail_layout.addWidget(self._summary, 1)
        splitter.addWidget(detail)
        splitter.setSizes([190, 410])
        layout.addWidget(splitter, 1)

        footer = QHBoxLayout()
        save_all = QPushButton("Save All")
        reset_all = QPushButton("Reset All")
        cancel = QPushButton("Cancel")
        save = QPushButton("Save")
        reset = QPushButton("Reset")
        for button in (save_all, reset_all, cancel, save, reset):
            button.setObjectName(f"qt-shell-settings-{button.text().lower().replace(' ', '-')}")
        save_all.clicked.connect(lambda _checked=False: self._mark_action("Save All"))
        reset_all.clicked.connect(lambda _checked=False: self._mark_action("Reset All"))
        cancel.clicked.connect(self.close)
        save.clicked.connect(lambda _checked=False: self._mark_action("Save"))
        reset.clicked.connect(lambda _checked=False: self._mark_action("Reset"))
        footer.addWidget(save_all)
        footer.addWidget(reset_all)
        footer.addStretch(1)
        footer.addWidget(cancel)
        footer.addWidget(save)
        footer.addWidget(reset)
        layout.addLayout(footer)
        self._status.setObjectName("qt-shell-settings-status")
        layout.addWidget(self._status)

    def _populate_tree(self) -> None:
        """Populate settings navigation from Faceswap config metadata."""
        self._tree.clear()
        category_items: dict[str, QTreeWidgetItem] = {}
        for category in self._ordered_categories():
            parent = QTreeWidgetItem([category.replace("_", " ").title()])
            parent.setData(0, Qt.UserRole, category)
            self._tree.addTopLevelItem(parent)
            category_items[category] = parent
        for section in self._sections:
            parent = category_items[section.category]
            if section.section == "global":
                continue
            child = QTreeWidgetItem([section.label])
            child.setData(0, Qt.UserRole, section.identifier)
            parent.addChild(child)
            parent.setExpanded(True)

    def _selection_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        """Update summary text when a settings section is selected."""
        if current is None:
            return
        identifier = current.data(0, Qt.UserRole)
        if not isinstance(identifier, str):
            return
        section = self._section_for_identifier(identifier)
        title = identifier.replace("|", " - ").replace("_", " ").title()
        self._header.setText(f"{title} Settings")
        if section is None:
            self._summary.setText("Select a plugin section to configure.")
            return
        self._summary.setText(f"{section.helptext}\n\nOptions available: {section.option_count}")

    def _select_initial_section(self, section: str | None) -> None:
        """Select an initial category/section, matching Tk open_popup(name=...)."""
        wanted = section or (self._ordered_categories()[0] if self._sections else None)
        if wanted is None:
            return
        matches = self._tree.findItems(
            wanted.replace("_", " ").title(),
            Qt.MatchRecursive | Qt.MatchExactly,
            0,
        )
        if matches:
            self._tree.setCurrentItem(matches[0])
            return
        for index in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(index)
            if item.data(0, Qt.UserRole) == wanted:
                self._tree.setCurrentItem(item)
                return

    def _mark_action(self, action: str) -> None:
        """Record settings action routing in the status label."""
        target = self.selected_identifier or "settings"
        self._status.setText(f"{action}: {target}")

    def _load_sections(self) -> tuple[SettingsSection, ...]:
        """Load settings sections from Faceswap config metadata."""
        sections: list[SettingsSection] = []
        for category, config in self._config_provider().items():
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

    def _ordered_categories(self) -> list[str]:
        """Return Tk-compatible category ordering."""
        categories = {section.category for section in self._sections}
        ordered = [category for category in self.CATEGORY_ORDER if category in categories]
        ordered.extend(sorted(category for category in categories if category not in ordered))
        return ordered

    def _section_for_identifier(self, identifier: str) -> SettingsSection | None:
        """Return a section for a tree identifier."""
        for section in self._sections:
            if section.identifier == identifier:
                return section
        return None


__all__ = get_module_objects(__name__)
