#!/usr/bin/env python3
"""Command panel and schema-backed option renderer for the Qt shell."""

from __future__ import annotations

import os
import shlex
import typing as T
from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.command_schema import CommandSchema, OptionSpec

IS_WINDOWS = os.name == "nt"


class OptionsFormRenderer(QWidget):
    """Dynamic schema-backed Qt option renderer."""

    value_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
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
        section.setMinimumWidth(0)
        section.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        section.setObjectName(
            "qt-shell-option-group" if has_title else "qt-shell-option-group-master"
        )
        form = QFormLayout(section)
        form.setContentsMargins(10, 8, 10, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        if has_title:
            label = QLabel(str(group).title())
            label.setObjectName("qt-shell-option-group-label")
            form.addRow(label)
        self._layout.addWidget(section)
        return form

    def _label_for(self, spec: OptionSpec) -> QLabel:
        """Return a label with optional help tooltip."""
        label = QLabel(spec.title)
        label.setMinimumWidth(0)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
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
        self._apply_widget_policy(widget)
        if spec.helptext:
            widget.setToolTip(spec.helptext)
        self._connect_value_signal(widget, spec)
        return widget

    def _build_radio_group(self, spec: OptionSpec) -> QWidget:
        """Build an exclusive choice widget for radio metadata."""
        widget = QWidget()
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(6)
        group = QButtonGroup(widget)
        group.setExclusive(True)
        default = self._string_value(spec.default)
        columns = self._choice_columns(spec.choices)
        for index, choice in enumerate(spec.choices):
            button = QRadioButton(choice)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setChecked(choice == default)
            if spec.helptext:
                button.setToolTip(spec.helptext)
            group.addButton(button)
            layout.addWidget(button, index // columns, index % columns)
        for column in range(columns):
            layout.setColumnStretch(column, 1)
        self._radio_groups[spec.switch] = group
        return widget

    def _build_multi_select(self, spec: OptionSpec) -> QWidget:
        """Build a multi-select widget only for multi-option metadata."""
        widget = QWidget()
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(6)
        selected = self._value_set(spec.default)
        columns = self._choice_columns(spec.choices)
        for index, choice in enumerate(spec.choices):
            checkbox = QCheckBox(choice)
            checkbox.setMinimumWidth(0)
            checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            checkbox.setChecked(choice in selected)
            if spec.helptext:
                checkbox.setToolTip(spec.helptext)
            layout.addWidget(checkbox, index // columns, index % columns)
        for column in range(columns):
            layout.setColumnStretch(column, 1)
        return widget

    def _build_slider(self, spec: OptionSpec) -> QWidget:
        """Build a slider linked to a compact numeric field."""
        widget = QWidget()
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        slider = QSlider(Qt.Horizontal)
        line_edit = QLineEdit()
        line_edit.setFixedWidth(70)
        slider.setObjectName("qt-shell-option-slider")
        line_edit.setObjectName("qt-shell-option-slider-value")
        slider.setMinimumWidth(0)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        layout.addWidget(slider, 1)
        layout.addWidget(line_edit)
        self._set_slider_value(spec, widget, spec.default)
        return widget

    def _row_widget(self, spec: OptionSpec, widget: QWidget) -> QWidget:
        """Return the row widget, wrapping path options with browse buttons."""
        if not spec.browser_modes or not isinstance(widget, QLineEdit):
            return widget
        row = QWidget()
        row.setMinimumWidth(0)
        row.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(widget, 1)
        for mode in spec.browser_modes:
            button = QPushButton(self._browser_label(mode))
            button.setObjectName(f"qt-shell-browser-{mode}")
            button.setToolTip(spec.helptext or f"Browse for {spec.title}")
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
            self.value_changed.emit()

    def validation_errors(self) -> tuple[str, ...]:
        """Return inline validation errors for rendered option values."""
        errors: list[str] = []
        values = self.values()
        for spec in self._specs:
            if not self._is_required(spec):
                continue
            if self._is_empty_value(values.get(spec.switch)):
                errors.append(f"{spec.title} is required")
        return tuple(errors)

    def _connect_value_signal(self, widget: QWidget, spec: OptionSpec) -> None:
        """Connect widget value changes to the renderer-level signal."""
        if isinstance(widget, QLineEdit):
            widget.textEdited.connect(lambda _text: self.value_changed.emit())
        elif isinstance(widget, QCheckBox):
            widget.clicked.connect(lambda _checked=False: self.value_changed.emit())
        elif isinstance(widget, QComboBox):
            widget.activated.connect(lambda _index=0: self.value_changed.emit())
        elif self._is_radio_group(spec) or self._is_multi_select(spec):
            for checkbox in widget.findChildren(QCheckBox):
                checkbox.clicked.connect(lambda _checked=False: self.value_changed.emit())
            for radio in widget.findChildren(QRadioButton):
                radio.clicked.connect(lambda _checked=False: self.value_changed.emit())
        elif self._is_slider(spec):
            slider = widget.findChild(QSlider)
            line_edit = widget.findChild(QLineEdit)
            if slider is not None:
                slider.sliderReleased.connect(self.value_changed.emit)
            if line_edit is not None:
                line_edit.textEdited.connect(lambda _text: self.value_changed.emit())

    @staticmethod
    def _is_required(spec: OptionSpec) -> bool:
        """Return whether option metadata indicates a required argument."""
        return "[required]" in spec.helptext.lower() or "required" in spec.title.lower()

    @staticmethod
    def _is_empty_value(value: object) -> bool:
        """Return whether a value is empty for validation purposes."""
        if value is None or value is False:
            return True
        if isinstance(value, str):
            return value == ""
        return bool(isinstance(value, list | tuple) and not value)

    @staticmethod
    def _is_radio_group(spec: OptionSpec) -> bool:
        """Return true when metadata clearly asks for a radio group."""
        return spec.is_radio and bool(spec.choices)

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
    def _choice_columns(choices: tuple[str, ...]) -> int:
        """Return a column count that avoids forcing horizontal scroll."""
        if len(choices) <= 1:
            return 1
        if len(choices) == 2:
            return 2
        longest = max(len(choice) for choice in choices)
        return 1 if longest > 12 else 2

    @staticmethod
    def _apply_widget_policy(widget: QWidget) -> None:
        """Apply common growth policy for option widgets."""
        widget.setMinimumWidth(0)
        if isinstance(widget, (QLineEdit, QComboBox)):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

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
        if isinstance(value, (list, tuple, set)):
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
        parts = shlex.split(value, posix=not IS_WINDOWS)
        if IS_WINDOWS:
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
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return str(value)


class CommandPanel(QWidget):
    """Left command/options panel backed by CommandSchema."""

    generate_requested = Signal()
    run_requested = Signal()
    value_changed = Signal()

    def __init__(self, schema: CommandSchema, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self._schema = schema
        self._category = QComboBox()
        self._command = QComboBox()
        self._primary_tabs = QTabBar()
        self._tool_tabs = QTabBar()
        self._command_info = QLabel()
        self._validation_label = QLabel()
        self._renderer = OptionsFormRenderer()
        self._generate_button = QPushButton("Generate")
        self._run_button = QPushButton("Run")
        self._syncing_tabs = False
        self._build()
        self._primary_tabs.currentChanged.connect(self._set_primary_tab)
        self._tool_tabs.currentChanged.connect(self._set_tool_tab)
        self._command.currentTextChanged.connect(self._set_command_options)
        self._renderer.value_changed.connect(self._value_changed)
        self._set_primary_tab(self._primary_tabs.currentIndex())

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
            self._set_category(category)
        if self._command.findText(command) == -1:
            self._command.addItem(command)
        self._command.setCurrentText(command)
        self._select_tabs_for_command(category, command)
        self._renderer.apply_values(values)

    def clear_values(self) -> None:
        """Reset rendered fields to empty/default values."""
        self._set_command_options(self._command.currentText())

    def set_running(self, running: bool) -> None:
        """Disable command execution controls while a job is running."""
        self._generate_button.setEnabled(not running)
        self._run_button.setEnabled(not running)

    def validation_errors(self) -> tuple[str, ...]:
        """Return current command option validation errors."""
        return self._renderer.validation_errors()

    def _build(self) -> None:
        """Build command panel widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._build_tabs()
        layout.addWidget(self._primary_tabs)
        layout.addWidget(self._tool_tabs)

        self._command_info.setWordWrap(True)
        self._command_info.setObjectName("qt-shell-command-info")
        layout.addWidget(self._command_info)

        self._validation_label.setWordWrap(True)
        self._validation_label.setObjectName("qt-shell-command-errors")
        self._validation_label.setProperty("status", "error")
        self._validation_label.setVisible(False)
        layout.addWidget(self._validation_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setObjectName("qt-shell-command-scroll")
        scroll.setWidget(self._renderer)
        layout.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        self._generate_button.clicked.connect(self.generate_requested)
        self._run_button.clicked.connect(self.run_requested)
        buttons.addWidget(self._generate_button)
        buttons.addWidget(self._run_button)
        layout.addLayout(buttons)

    def _build_tabs(self) -> None:
        """Build command and tool tab selectors."""
        self._primary_tabs.setObjectName("qt-shell-command-tabs")
        self._tool_tabs.setObjectName("qt-shell-tool-tabs")
        self._primary_tabs.setExpanding(False)
        self._tool_tabs.setExpanding(False)
        self._primary_tabs.setUsesScrollButtons(False)
        self._tool_tabs.setUsesScrollButtons(True)

        for command in self._schema.commands("faceswap"):
            self._primary_tabs.addTab(command.title())
        if "tools" in self._schema.categories:
            self._primary_tabs.addTab("Tools")
            for command in self._schema.commands("tools"):
                self._tool_tabs.addTab(command.title())

        self._category.addItems(self._schema.categories)
        self._tool_tabs.setVisible(False)

    def _set_primary_tab(self, index: int) -> None:
        """Update category/command for the selected primary tab."""
        if index < 0 or self._syncing_tabs:
            return
        tab = self._primary_tabs.tabText(index).lower()
        if tab == "tools":
            self._tool_tabs.setVisible(True)
            self._set_category("tools")
            tool_index = max(0, self._tool_tabs.currentIndex())
            self._set_tool_tab(tool_index)
            return
        self._tool_tabs.setVisible(False)
        self._set_category("faceswap")
        self._command.setCurrentText(tab)

    def _set_tool_tab(self, index: int) -> None:
        """Update command for the selected tool tab."""
        if index < 0 or self._syncing_tabs:
            return
        if self._primary_tabs.tabText(self._primary_tabs.currentIndex()) != "Tools":
            return
        self._set_category("tools")
        self._command.setCurrentText(self._tool_tabs.tabText(index).lower())

    def _set_category(self, category: str) -> None:
        """Update commands for a category without exposing combo widgets."""
        if self._category.currentText() != category:
            self._category.setCurrentText(category)
        self._command.blockSignals(True)
        self._command.clear()
        self._command.addItems(self._schema.commands(category))
        self._command.blockSignals(False)
        if self._command.count() and not self._command.currentText():
            self._command.setCurrentIndex(0)
        self._set_command_options(self._command.currentText())

    def _set_command_options(self, command: str) -> None:
        """Render fields for the selected command."""
        self._renderer.set_options(self._schema.options(command))
        self._set_command_info(command)
        self._update_validation()
        self._run_button.setText(command.title() if command else "Run")

    def _set_command_info(self, command: str) -> None:
        """Render command summary text from CLI metadata."""
        info = self._schema.command_info(command)
        if not info:
            self._command_info.setText(escape(command.title()) if command else "")
            return
        lines = [line.strip() for line in info.splitlines() if line.strip()]
        heading = lines[0] if lines else command.title()
        detail = "\n".join(lines[1:])
        if detail:
            self._command_info.setText(f"<b>{escape(heading)}</b><br>{escape(detail)}")
        else:
            self._command_info.setText(f"<b>{escape(heading)}</b>")

    def _value_changed(self) -> None:
        """Forward renderer changes after refreshing inline validation."""
        self._update_validation()
        self.value_changed.emit()

    def _update_validation(self) -> None:
        """Display current validation errors inline."""
        errors = self.validation_errors()
        self._validation_label.setText("\n".join(errors))
        self._validation_label.setVisible(bool(errors))

    def _select_tabs_for_command(self, category: str | None, command: str) -> None:
        """Synchronize visible tabs after a project load or programmatic command set."""
        self._syncing_tabs = True
        try:
            if category == "tools":
                tools_index = self._find_tab(self._primary_tabs, "Tools")
                if tools_index >= 0:
                    self._primary_tabs.setCurrentIndex(tools_index)
                tool_index = self._find_tab(self._tool_tabs, command.title())
                if tool_index >= 0:
                    self._tool_tabs.setCurrentIndex(tool_index)
                self._tool_tabs.setVisible(True)
            else:
                command_index = self._find_tab(self._primary_tabs, command.title())
                if command_index >= 0:
                    self._primary_tabs.setCurrentIndex(command_index)
                self._tool_tabs.setVisible(False)
        finally:
            self._syncing_tabs = False

    @staticmethod
    def _find_tab(tabs: QTabBar, text: str) -> int:
        """Return the index for a tab text or -1."""
        for index in range(tabs.count()):
            if tabs.tabText(index) == text:
                return index
        return -1
