#!/usr/bin/env python3
"""Command panel and schema-backed option renderer for the Qt shell."""

from __future__ import annotations

import os
import shlex
import typing as T

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.command_schema import CommandSchema, OptionSpec


class OptionsFormRenderer(QWidget):
    """Dynamic QWidget/QFormLayout option renderer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._specs: tuple[OptionSpec, ...] = ()
        self._widgets: dict[str, QWidget] = {}

    @property
    def rendered_switches(self) -> tuple[str, ...]:
        """Return the rendered CLI switches."""
        return tuple(self._widgets)

    def set_options(self, specs: tuple[OptionSpec, ...]) -> None:
        """Render a fresh set of command option controls."""
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._widgets.clear()
        self._specs = specs
        current_group: str | None = None
        for spec in specs:
            current_group = self._add_group_header(spec, current_group)
            widget = self._build_widget(spec)
            self._widgets[spec.switch] = widget
            label = self._label_for(spec)
            self._layout.addRow(label, self._row_widget(spec, widget))

    def values(self) -> dict[str, object]:
        """Return command values keyed by CLI switch."""
        return {spec.switch: self._value_for(spec) for spec in self._specs}

    def apply_values(self, values: T.Mapping[str, object]) -> None:
        """Apply stored switch-keyed values to rendered controls."""
        for spec in self._specs:
            widget = self._widgets[spec.switch]
            if isinstance(widget, QCheckBox):
                widget.setChecked(
                    self._checked_for_value(
                        spec,
                        values.get(spec.switch, spec.default),
                        spec.switch in values,
                    )
                )
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(
                    self._string_value(values.get(spec.switch, spec.default))
                )
            elif isinstance(widget, QLineEdit):
                widget.setText(
                    self._string_value(values.get(spec.switch, spec.default))
                )

    def widget_for_switch(self, switch: str) -> QWidget:
        """Return the rendered input widget for a CLI switch."""
        return self._widgets[switch]

    def _add_group_header(
        self, spec: OptionSpec, current_group: str | None
    ) -> str | None:
        """Add a section label when the option group changes."""
        if not spec.group or spec.group == current_group:
            return current_group
        label = QLabel(f"<b>{spec.group}</b>")
        label.setObjectName("qt-shell-option-group")
        self._layout.addRow(label)
        return spec.group

    def _label_for(self, spec: OptionSpec) -> QLabel:
        """Return a label with optional help tooltip."""
        label = QLabel(spec.title)
        if spec.helptext:
            label.setToolTip(spec.helptext)
        return label

    def _build_widget(self, spec: OptionSpec) -> QWidget:
        """Create a Qt widget for an option."""
        if spec.value_type is bool:
            widget = QCheckBox()
            widget.setChecked(self._checked_for_value(spec, spec.default, False))
        elif spec.nargs:
            widget = QLineEdit()
            widget.setText(self._string_value(spec.default))
        elif spec.choices:
            widget = QComboBox()
            default = self._string_value(spec.default)
            if default == "" and default not in spec.choices:
                widget.addItem("")
            widget.addItems(spec.choices)
            widget.setCurrentText(default)
        else:
            widget = QLineEdit()
            widget.setText(self._string_value(spec.default))

        if spec.helptext:
            widget.setToolTip(spec.helptext)
        return widget

    def _row_widget(self, spec: OptionSpec, widget: QWidget) -> QWidget:
        """Return the row widget, wrapping path options with browse buttons."""
        if not spec.browser_modes or not isinstance(widget, QLineEdit):
            return widget

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        for mode in spec.browser_modes:
            button = QPushButton(self._browser_label(mode))
            button.setObjectName(f"qt-shell-browser-{mode}")
            button.setToolTip(spec.helptext or f"Browse for {spec.title}")
            button.clicked.connect(
                lambda _checked=False, m=mode, w=widget: self._browse(m, w)
            )
            layout.addWidget(button)
        return row

    def _value_for(self, spec: OptionSpec) -> object:
        """Read and normalize an option widget value."""
        widget = self._widgets[spec.switch]
        if isinstance(widget, QCheckBox):
            checked = widget.isChecked()
            return not checked if spec.action == "store_false" else checked
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if not isinstance(widget, QLineEdit):
            return ""
        text = widget.text().strip()
        return self._split_nargs(text) if spec.nargs and text else text

    def _browse(self, mode: str, widget: QLineEdit) -> None:
        """Populate a line edit from a simple QFileDialog browser mode."""
        if mode == "folder":
            value = QFileDialog.getExistingDirectory(
                self, "Select Folder", widget.text()
            )
        elif mode == "file":
            value, _ = QFileDialog.getOpenFileName(self, "Select File", widget.text())
        elif mode == "files":
            values, _ = QFileDialog.getOpenFileNames(
                self, "Select Files", widget.text()
            )
            value = self._join_paths(values)
        elif mode == "save":
            value, _ = QFileDialog.getSaveFileName(self, "Save File", widget.text())
        else:
            value = ""

        if value:
            widget.setText(value)

    @staticmethod
    def _browser_label(mode: str) -> str:
        """Return a compact label for a browser mode."""
        labels = {
            "folder": "Folder",
            "file": "File",
            "files": "Files",
            "save": "Save",
        }
        return labels.get(mode, "Browse")

    @staticmethod
    def _checked_for_value(
        spec: OptionSpec, value: object, value_is_command_value: bool
    ) -> bool:
        """Return human-facing checkbox state for defaults or stored command values."""
        checked = bool(value)
        if spec.action == "store_false" and value_is_command_value:
            return not checked
        return checked

    @staticmethod
    def _join_paths(paths: T.Iterable[str]) -> str:
        """Join browser-selected paths for a nargs line edit."""
        return " ".join(
            f'"{path}"' if any(char.isspace() for char in path) else path
            for path in paths
        )

    @staticmethod
    def _split_nargs(value: str) -> list[str]:
        """Split a multi-value CLI option preserving platform-specific path syntax."""
        parts = shlex.split(value, posix=os.name != "nt")
        if os.name == "nt":
            parts = [
                part[1:-1]
                if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'"
                else part
                for part in parts
            ]
        return parts

    @staticmethod
    def _string_value(value: object) -> str:
        """Render a stored value as editable text."""
        if value is None or value is False:
            return ""
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return str(value)


class CommandPanel(QWidget):
    """Left command/options panel backed by CommandSchema."""

    generate_requested = Signal()
    run_requested = Signal()

    def __init__(self, schema: CommandSchema, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._schema = schema
        self._category = QComboBox()
        self._command = QComboBox()
        self._renderer = OptionsFormRenderer()
        self._build()
        self._category.currentTextChanged.connect(self._set_category)
        self._command.currentTextChanged.connect(self._set_command_options)
        self._set_category(self._category.currentText())

    @property
    def renderer(self) -> OptionsFormRenderer:
        """Return the option renderer."""
        return self._renderer

    def command_spec(self) -> tuple[str, str, dict[str, object]]:
        """Return selected category, command and switch-keyed values."""
        return (
            self._category.currentText(),
            self._command.currentText(),
            self._renderer.values(),
        )

    def set_command(self, command: str, values: T.Mapping[str, object]) -> None:
        """Apply a command and values from a loaded project."""
        category = self._schema.category_for_command(command)
        if category is not None:
            self._category.setCurrentText(category)
        if self._command.findText(command) == -1:
            self._command.addItem(command)
        self._command.setCurrentText(command)
        self._renderer.apply_values(values)

    def clear_values(self) -> None:
        """Reset rendered fields to empty/default values."""
        self._set_command_options(self._command.currentText())

    def _build(self) -> None:
        """Build command panel widgets."""
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Command panel placeholder"))
        form = QFormLayout()
        self._category.addItems(self._schema.categories)
        form.addRow("Category", self._category)
        form.addRow("Command", self._command)
        layout.addLayout(form)
        layout.addWidget(self._renderer)
        buttons = QHBoxLayout()
        generate = QPushButton("Generate")
        run = QPushButton("Run")
        generate.clicked.connect(self.generate_requested)
        run.clicked.connect(self.run_requested)
        buttons.addWidget(generate)
        buttons.addWidget(run)
        layout.addLayout(buttons)
        layout.addStretch(1)

    def _set_category(self, category: str) -> None:
        """Update commands for the selected category."""
        self._command.blockSignals(True)
        self._command.clear()
        self._command.addItems(self._schema.commands(category))
        self._command.blockSignals(False)
        self._set_command_options(self._command.currentText())

    def _set_command_options(self, command: str) -> None:
        """Render fields for the selected command."""
        self._renderer.set_options(self._schema.options(command))
