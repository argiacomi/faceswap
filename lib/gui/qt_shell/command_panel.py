#!/usr/bin/env python3
"""Command panel and schema-backed option renderer for the Qt shell."""

from __future__ import annotations

import os
import shlex
import typing as T

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.command_schema import CommandSchema, OptionSpec


class OptionsFormRenderer(QWidget):
    """Dynamic schema-backed Qt option renderer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(5, 5, 5, 5)
        self._layout.setSpacing(8)
        self._specs: tuple[OptionSpec, ...] = ()
        self._widgets: dict[str, QWidget] = {}
        self._radio_groups: dict[str, QButtonGroup] = {}

    @property
    def rendered_switches(self) -> tuple[str, ...]:
        """Return the rendered CLI switches."""
        return tuple(self._widgets)

    def set_options(self, specs: tuple[OptionSpec, ...]) -> None:
        """Render a fresh set of command option controls."""
        self._clear_layout(self._layout)
        self._widgets.clear()
        self._radio_groups.clear()
        self._specs = specs
        for group, group_specs in self._grouped_specs(specs):
            form = self._add_group_section(group)
            for spec in group_specs:
                widget = self._build_widget(spec)
                self._widgets[spec.switch] = widget
                form.addRow(self._label_for(spec), self._row_widget(spec, widget))
        self._layout.addStretch(1)

    def values(self) -> dict[str, object]:
        """Return command values keyed by CLI switch."""
        return {spec.switch: self._value_for(spec) for spec in self._specs}

    def apply_values(self, values: T.Mapping[str, object]) -> None:
        """Apply stored switch-keyed values to rendered controls."""
        for spec in self._specs:
            value = values.get(spec.switch, spec.default)
            widget = self._widgets[spec.switch]
            if spec.switch in self._radio_groups:
                self._set_radio_value(spec, value)
            elif self._is_multi_select(spec):
                self._set_multi_values(widget, value)
            elif self._is_slider(spec):
                self._set_slider_value(spec, widget, value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(self._checked_for_value(spec, value, spec.switch in values))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(self._string_value(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(self._string_value(value))

    def widget_for_switch(self, switch: str) -> QWidget:
        """Return the rendered input widget for a CLI switch."""
        return self._widgets[switch]

    @classmethod
    def _clear_layout(cls, layout: QVBoxLayout | QFormLayout | QHBoxLayout) -> None:
        """Remove and delete child widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            widget = item.widget()
            if child_layout is not None:
                cls._clear_layout(child_layout)
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _grouped_specs(
        specs: tuple[OptionSpec, ...],
    ) -> tuple[tuple[str | None, tuple[OptionSpec, ...]], ...]:
        """Group option specs in first-seen order."""
        group_order: list[str | None] = []
        grouped: dict[str | None, list[OptionSpec]] = {}
        for spec in specs:
            if spec.group not in grouped:
                group_order.append(spec.group)
                grouped[spec.group] = []
            grouped[spec.group].append(spec)
        return tuple((group, tuple(grouped[group])) for group in group_order)

    def _add_group_section(self, group: str | None) -> QFormLayout:
        """Add a visually separated section for one CLI option group."""
        has_title = bool(group and group != "_master")
        section: QWidget = QGroupBox() if has_title else QWidget()
        section.setObjectName(
            "qt-shell-option-group" if has_title else "qt-shell-option-group-master"
        )
        form = QFormLayout(section)
        form.setContentsMargins(10, 8, 10, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        if has_title:
            label = QLabel(str(group).title())
            label.setObjectName("qt-shell-option-group-label")
            form.addRow(label)
        self._layout.addWidget(section)
        return form

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
        elif self._is_multi_select(spec):
            widget = self._build_multi_select(spec)
        elif self._is_radio_group(spec):
            widget = self._build_radio_group(spec)
        elif self._is_slider(spec):
            widget = self._build_slider(spec)
        elif spec.nargs:
            widget = QLineEdit(self._string_value(spec.default))
        elif spec.choices:
            widget = QComboBox()
            default = self._string_value(spec.default)
            if default == "" and default not in spec.choices:
                widget.addItem("")
            widget.addItems(spec.choices)
            widget.setCurrentText(default)
        else:
            widget = QLineEdit(self._string_value(spec.default))
        if spec.helptext:
            widget.setToolTip(spec.helptext)
        return widget

    def _build_radio_group(self, spec: OptionSpec) -> QWidget:
        """Build a compact exclusive choice widget for radio metadata."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        group = QButtonGroup(widget)
        group.setExclusive(True)
        default = self._string_value(spec.default)
        for choice in spec.choices:
            button = QRadioButton(choice)
            button.setChecked(choice == default)
            if spec.helptext:
                button.setToolTip(spec.helptext)
            group.addButton(button)
            layout.addWidget(button)
        layout.addStretch(1)
        self._radio_groups[spec.switch] = group
        return widget

    def _build_multi_select(self, spec: OptionSpec) -> QWidget:
        """Build a compact multi-select widget only for multi-option metadata."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        selected = self._value_set(spec.default)
        for choice in spec.choices:
            checkbox = QCheckBox(choice)
            checkbox.setChecked(choice in selected)
            if spec.helptext:
                checkbox.setToolTip(spec.helptext)
            layout.addWidget(checkbox)
        layout.addStretch(1)
        return widget

    def _build_slider(self, spec: OptionSpec) -> QWidget:
        """Build a slider linked to a compact numeric field."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        slider = QSlider(Qt.Horizontal)
        line_edit = QLineEdit()
        line_edit.setFixedWidth(58)
        slider.setObjectName("qt-shell-option-slider")
        line_edit.setObjectName("qt-shell-option-slider-value")
        slider.setRange(
            self._value_to_slider(spec, spec.slider_min),
            self._value_to_slider(spec, spec.slider_max),
        )
        if spec.value_type is int:
            slider.setSingleStep(max(1, int(spec.slider_rounding or 1)))

        def sync_line(slider_value: int) -> None:
            line_edit.setText(
                self._format_slider_value(spec, self._slider_to_value(spec, slider_value))
            )

        def sync_slider() -> None:
            slider.setValue(self._value_to_slider(spec, line_edit.text()))

        slider.valueChanged.connect(sync_line)
        line_edit.editingFinished.connect(sync_slider)
        layout.addWidget(slider)
        layout.addWidget(line_edit)
        self._set_slider_value(spec, widget, spec.default)
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
            button.clicked.connect(lambda _checked=False, m=mode, w=widget: self._browse(m, w))
            layout.addWidget(button)
        return row

    def _value_for(self, spec: OptionSpec) -> object:
        """Read and normalize an option widget value."""
        widget = self._widgets[spec.switch]
        if spec.switch in self._radio_groups:
            button = self._radio_groups[spec.switch].checkedButton()
            return "" if button is None else button.text()
        if self._is_multi_select(spec):
            selected = [
                checkbox.text()
                for checkbox in widget.findChildren(QCheckBox)
                if checkbox.isChecked()
            ]
            return selected if selected else ""
        if self._is_slider(spec):
            return self._slider_widget_value(spec, widget)
        if isinstance(widget, QCheckBox):
            checked = widget.isChecked()
            return not checked if spec.action == "store_false" else checked
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            return self._split_nargs(text) if spec.nargs and text else text
        return ""

    def _set_radio_value(self, spec: OptionSpec, value: object) -> None:
        """Restore one radio value."""
        text = self._string_value(value)
        for button in self._radio_groups[spec.switch].buttons():
            if isinstance(button, QRadioButton):
                button.setChecked(button.text() == text)

    def _set_multi_values(self, widget: QWidget, value: object) -> None:
        """Restore multi-select values."""
        selected = self._value_set(value)
        for checkbox in widget.findChildren(QCheckBox):
            checkbox.setChecked(checkbox.text() in selected)

    def _set_slider_value(self, spec: OptionSpec, widget: QWidget, value: object) -> None:
        """Restore one slider value."""
        slider = widget.findChild(QSlider)
        line_edit = widget.findChild(QLineEdit)
        if slider is None or line_edit is None:
            return
        slider_value = self._value_to_slider(spec, value)
        slider.setValue(slider_value)
        line_edit.setText(
            self._format_slider_value(spec, self._slider_to_value(spec, slider_value))
        )

    def _browse(self, mode: str, widget: QLineEdit) -> None:
        """Populate a line edit from a simple QFileDialog browser mode."""
        if mode == "folder":
            value = QFileDialog.getExistingDirectory(self, "Select Folder", widget.text())
        elif mode == "file":
            value, _ = QFileDialog.getOpenFileName(self, "Select File", widget.text())
        elif mode == "files":
            values, _ = QFileDialog.getOpenFileNames(self, "Select Files", widget.text())
            value = self._join_paths(values)
        elif mode == "save":
            value, _ = QFileDialog.getSaveFileName(self, "Save File", widget.text())
        else:
            value = ""
        if value:
            widget.setText(value)

    @staticmethod
    def _is_radio_group(spec: OptionSpec) -> bool:
        """Return true when metadata clearly asks for a compact radio group."""
        return spec.is_radio and bool(spec.choices) and len(spec.choices) <= 4

    @staticmethod
    def _is_multi_select(spec: OptionSpec) -> bool:
        """Return true when metadata clearly asks for a multi-select widget."""
        return spec.is_multi_option and bool(spec.choices)

    @staticmethod
    def _is_slider(spec: OptionSpec) -> bool:
        """Return true when metadata clearly asks for a slider."""
        return (
            spec.action == "Slider" and spec.slider_min is not None and spec.slider_max is not None
        )

    @staticmethod
    def _browser_label(mode: str) -> str:
        """Return a compact label for a browser mode."""
        return {
            "folder": "Folder",
            "file": "File",
            "files": "Files",
            "save": "Save",
        }.get(mode, "Browse")

    @staticmethod
    def _checked_for_value(spec: OptionSpec, value: object, value_is_command_value: bool) -> bool:
        """Return human-facing checkbox state for defaults or stored command values."""
        checked = bool(value)
        if spec.action == "store_false" and value_is_command_value:
            return not checked
        return checked

    @classmethod
    def _value_set(cls, value: object) -> set[str]:
        """Return a comparable string set for multi-select values."""
        if value is None or value is False or value == "":
            return set()
        if isinstance(value, str):
            try:
                return set(cls._split_nargs(value))
            except ValueError:
                return {part.strip() for part in value.split(",") if part.strip()}
        if isinstance(value, list | tuple | set):
            return {str(item) for item in value}
        return {str(value)}

    @classmethod
    def _slider_widget_value(cls, spec: OptionSpec, widget: QWidget) -> int | float:
        """Return a typed slider widget value."""
        line_edit = widget.findChild(QLineEdit)
        if line_edit is None:
            slider = widget.findChild(QSlider)
            return cls._slider_to_value(spec, slider.value())  # type:ignore[union-attr]
        return cls._coerce_slider_value(spec, line_edit.text())

    @classmethod
    def _value_to_slider(cls, spec: OptionSpec, value: object) -> int:
        """Return a concrete QSlider integer value from a command value."""
        coerced = cls._coerce_slider_value(spec, value)
        if spec.value_type is float:
            return int(round(float(coerced) * cls._slider_scale(spec)))
        return int(coerced)

    @classmethod
    def _slider_to_value(cls, spec: OptionSpec, value: int) -> int | float:
        """Return a command value from a concrete QSlider integer value."""
        if spec.value_type is float:
            return round(value / cls._slider_scale(spec), cls._slider_decimals(spec))
        return int(value)

    @classmethod
    def _coerce_slider_value(cls, spec: OptionSpec, value: object) -> int | float:
        """Coerce an arbitrary value into the slider's command type and bounds."""
        try:
            if spec.value_type is float:
                coerced: int | float = float(value)
            else:
                coerced = int(float(str(value)))
        except (TypeError, ValueError):
            coerced = 0 if spec.default in (None, "") else spec.default
            coerced = float(coerced) if spec.value_type is float else int(coerced)
        if spec.slider_min is not None:
            coerced = max(coerced, spec.slider_min)
        if spec.slider_max is not None:
            coerced = min(coerced, spec.slider_max)
        if spec.value_type is float:
            return round(float(coerced), cls._slider_decimals(spec))
        return int(coerced)

    @classmethod
    def _format_slider_value(cls, spec: OptionSpec, value: int | float) -> str:
        """Format a slider value for the numeric line edit."""
        if spec.value_type is float:
            text = f"{float(value):.{cls._slider_decimals(spec)}f}"
            return text.rstrip("0").rstrip(".") if "." in text else text
        return str(int(value))

    @staticmethod
    def _slider_decimals(spec: OptionSpec) -> int:
        """Return float slider precision."""
        return max(0, int(spec.slider_rounding or 0))

    @classmethod
    def _slider_scale(cls, spec: OptionSpec) -> int:
        """Return integer scale factor for float sliders."""
        return 10 ** cls._slider_decimals(spec)

    @staticmethod
    def _join_paths(paths: T.Iterable[str]) -> str:
        """Join browser-selected paths for a nargs line edit."""
        return " ".join(
            f'"{path}"' if any(char.isspace() for char in path) else path for path in paths
        )

    @staticmethod
    def _split_nargs(value: str) -> list[str]:
        """Split a multi-value CLI option preserving platform-specific path syntax."""
        parts = shlex.split(value, posix=os.name != "nt")
        if os.name == "nt":
            parts = [
                part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'" else part
                for part in parts
            ]
        return parts

    @staticmethod
    def _string_value(value: object) -> str:
        """Render a stored value as editable text."""
        if value is None or value is False:
            return ""
        if isinstance(value, list | tuple):
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
        self._command_info = QLabel()
        self._renderer = OptionsFormRenderer()
        self._run_button = QPushButton("Run")
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._command_info.setWordWrap(True)
        self._command_info.setObjectName("qt-shell-command-info")
        layout.addWidget(self._command_info)

        form = QFormLayout()
        self._category.addItems(self._schema.categories)
        form.addRow("Category", self._category)
        form.addRow("Command", self._command)
        layout.addLayout(form)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("qt-shell-command-scroll")
        scroll.setWidget(self._renderer)
        layout.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        generate = QPushButton("Generate")
        generate.clicked.connect(self.generate_requested)
        self._run_button.clicked.connect(self.run_requested)
        buttons.addWidget(generate)
        buttons.addWidget(self._run_button)
        layout.addLayout(buttons)

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
        self._set_command_info(command)
        self._run_button.setText(command.title() if command else "Run")

    def _set_command_info(self, command: str) -> None:
        """Render command summary text from CLI metadata."""
        info = self._schema.command_info(command)
        if not info:
            self._command_info.setText(command.title() if command else "")
            return
        lines = [line.strip() for line in info.splitlines() if line.strip()]
        heading = lines[0] if lines else command.title()
        detail = "\n".join(lines[1:])
        if detail:
            self._command_info.setText(f"<b>{heading}</b><br>{detail}")
        else:
            self._command_info.setText(f"<b>{heading}</b>")
