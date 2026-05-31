#!/usr/bin/env python3
"""Command panel and schema-backed option renderer for the Qt shell."""

from __future__ import annotations

import os
import shlex
import textwrap
import typing as T
from html import escape

from PySide6.QtCore import QSize, Qt, Signal
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
    QStyle,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.command_schema import CommandSchema, OptionSpec
from lib.gui.qt_shell.theme import QtTheme, icon_for_action

IS_WINDOWS = os.name == "nt"


def _wrap_tooltip(text: str, width: int = 88) -> str:
    """Return tooltip text wrapped to a readable line length."""
    return "\n".join(
        textwrap.fill(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if line.strip()
        else ""
        for line in str(text).splitlines()
    )


class OptionGroupDrawer(QWidget):
    """Collapsible drawer for an option group (Tk ``ToggledFrame`` parity).

    Uses a disclosure-arrow QToolButton header rather than a checkbox so the
    affordance reads as expand/collapse rather than enable/disable.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-shell-option-group")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # type: ignore[attr-defined]
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setObjectName("qt-shell-option-group-toggle")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setArrowType(Qt.DownArrow)  # type: ignore[attr-defined]
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)  # type: ignore[attr-defined]
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[attr-defined]
        self._toggle.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle)

        self._content = QWidget()
        self._content.setObjectName("qt-shell-option-group-content")
        self._content.setMinimumWidth(0)
        self._content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # type: ignore[attr-defined]
        self._form = QFormLayout(self._content)
        self._form.setContentsMargins(10, 4, 10, 10)
        self._form.setHorizontalSpacing(12)
        self._form.setVerticalSpacing(6)
        self._form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self._form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # type: ignore[attr-defined]
        self._form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore[attr-defined]
        outer.addWidget(self._content)

    @property
    def form(self) -> QFormLayout:
        """Return the inner QFormLayout where option rows live."""
        return self._form

    def title(self) -> str:
        """Return the drawer title (mirrors QGroupBox.title())."""
        return self._toggle.text()  # type: ignore[no-any-return]

    def isCheckable(self) -> bool:  # noqa: N802 - Qt-style camelCase for test parity
        """Drawers are always collapsible; kept for test-shape compatibility."""
        return True

    def isChecked(self) -> bool:  # noqa: N802
        """Return whether the drawer is currently expanded."""
        return self._toggle.isChecked()  # type: ignore[no-any-return]

    def setChecked(self, expanded: bool) -> None:  # noqa: N802
        """Expand (``True``) or collapse (``False``) the drawer."""
        self._toggle.setChecked(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        """Update arrow direction and hide content when collapsed."""
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)  # type: ignore[attr-defined]
        self._content.setVisible(expanded)


class OptionsFormRenderer(QWidget):
    """Dynamic schema-backed Qt option renderer."""

    value_changed = Signal()

    def __init__(self, parent: QWidget | None = None, option_columns: int = 4) -> None:
        super().__init__(parent)
        self._option_columns = max(1, int(option_columns))
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # type: ignore[attr-defined]
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
            bool_pairs: list[tuple[OptionSpec, QCheckBox]] = []
            for spec in group_specs:
                widget = self._build_widget(spec)
                self._widgets[spec.switch] = widget
                if self._is_plain_bool(spec, widget):
                    bool_pairs.append((spec, T.cast(QCheckBox, widget)))
                elif self._is_choice_cluster(spec):
                    form.addRow(self._titled_cluster(spec, widget))
                else:
                    form.addRow(self._label_for(spec), self._row_widget(spec, widget))
            if bool_pairs:
                form.addRow(self._build_bool_cluster(bool_pairs))
        self._layout.addStretch(1)

    @staticmethod
    def _is_plain_bool(spec: OptionSpec, widget: QWidget) -> bool:
        """Return true when an option renders as a simple labeled checkbox row."""
        return spec.value_type is bool and isinstance(widget, QCheckBox)

    def _build_bool_cluster(self, bool_pairs: list[tuple[OptionSpec, QCheckBox]]) -> QWidget:
        """Pack plain bool options into a horizontal grid (Tk ``checkbuttons_frame`` parity)."""
        container = QWidget()
        container.setObjectName("qt-shell-option-bool-cluster")
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # type: ignore[attr-defined]
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 4, 0, 4)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        columns = self._choice_columns(len(bool_pairs))
        for index, (spec, checkbox) in enumerate(bool_pairs):
            checkbox.setText(spec.title)
            checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[attr-defined]
            grid.addWidget(checkbox, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        return container

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
            child_layout = item.layout()  # type: ignore[union-attr]
            widget = item.widget()  # type: ignore[union-attr]
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
        """Add a visually separated section for one CLI option group.

        Titled groups render as an :class:`OptionGroupDrawer` (disclosure-arrow
        toggle, Tk ``ToggledFrame`` parity). Untitled / ``_master`` groups remain
        a plain widget so they bleed into the surrounding layout.
        """
        has_title = bool(group and group != "_master")
        if has_title:
            drawer = OptionGroupDrawer(str(group).title())
            self._layout.addWidget(drawer)
            return drawer.form
        section = QWidget()
        section.setMinimumWidth(0)
        section.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # type: ignore[attr-defined]
        section.setObjectName("qt-shell-option-group-master")
        form = QFormLayout(section)
        form.setContentsMargins(10, 8, 10, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # type: ignore[attr-defined]
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore[attr-defined]
        self._layout.addWidget(section)
        return form

    LABEL_COLUMN_WIDTH = 75

    def _label_for(self, spec: OptionSpec) -> QLabel:
        """Return a label with optional help tooltip and required marker.

        The label is fixed-width so all inputs align in the right column. It wraps
        across multiple lines when the title is wider than ``LABEL_COLUMN_WIDTH``.
        """
        if self._is_required(spec):
            label = QLabel(f'{escape(spec.title)} <span style="color:#c0392b;">*</span>')
            label.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
            label.setObjectName("qt-shell-option-label-required")
            label.setProperty("required", True)
        else:
            label = QLabel(spec.title)
            label.setObjectName("qt-shell-option-label")
        label.setMinimumWidth(self.LABEL_COLUMN_WIDTH)
        label.setMaximumWidth(self.LABEL_COLUMN_WIDTH)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore[attr-defined]
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)  # type: ignore[attr-defined]
        if spec.helptext:
            label.setToolTip(self._tooltip_text(self._title_tooltip(spec)))
        return label

    @staticmethod
    def _title_tooltip(spec: OptionSpec) -> str:
        """Return the per-option tooltip that pairs the title with its helptext."""
        if not spec.helptext:
            return spec.title
        return f"{spec.title} - {spec.helptext}"

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
            widget.setToolTip(self._tooltip_text(spec.helptext))
        self._connect_value_signal(widget, spec)
        return widget

    def _build_radio_group(self, spec: OptionSpec) -> QWidget:
        """Build an exclusive choice widget for radio metadata."""
        widget = QWidget()
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # type: ignore[attr-defined]
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(6)
        group = QButtonGroup(widget)
        group.setExclusive(True)
        default = self._string_value(spec.default)
        columns = self._choice_columns(len(spec.choices))
        for index, choice in enumerate(spec.choices):
            button = QRadioButton(choice)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[attr-defined]
            button.setChecked(choice == default)
            if spec.helptext:
                button.setToolTip(self._tooltip_text(spec.helptext))
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
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # type: ignore[attr-defined]
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(6)
        selected = self._value_set(spec.default)
        columns = self._choice_columns(len(spec.choices))
        for index, choice in enumerate(spec.choices):
            checkbox = QCheckBox(choice)
            checkbox.setMinimumWidth(0)
            checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[attr-defined]
            checkbox.setChecked(choice in selected)
            if spec.helptext:
                checkbox.setToolTip(self._tooltip_text(spec.helptext))
            layout.addWidget(checkbox, index // columns, index % columns)
        for column in range(columns):
            layout.setColumnStretch(column, 1)
        return widget

    def _build_slider(self, spec: OptionSpec) -> QWidget:
        """Build a slider linked to a compact numeric field."""
        widget = QWidget()
        widget.setMinimumWidth(0)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)

        slider = QSlider(Qt.Horizontal)  # type: ignore[attr-defined]
        line_edit = QLineEdit()
        line_edit.setFixedWidth(54)
        line_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        slider.setObjectName("qt-shell-option-slider")
        line_edit.setObjectName("qt-shell-option-slider-value")
        slider.setMinimumWidth(0)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[attr-defined]
        slider_height = max(
            slider.sizeHint().height(),
            slider.minimumSizeHint().height(),
            slider.style().pixelMetric(QStyle.PM_SliderThickness, None, slider),
            slider.style().pixelMetric(QStyle.PM_SliderLength, None, slider),
        )
        slider.setMinimumHeight(slider_height + 8)
        widget.setMinimumHeight(max(slider.minimumHeight(), line_edit.sizeHint().height()) + 8)
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
        layout.addWidget(slider, 1, Qt.AlignVCenter)  # type: ignore[attr-defined]
        layout.addWidget(line_edit, 0, Qt.AlignVCenter)  # type: ignore[attr-defined]
        self._set_slider_value(spec, widget, spec.default)
        return widget

    def _row_widget(self, spec: OptionSpec, widget: QWidget) -> QWidget:
        """Return the row widget, wrapping path options with browse buttons.

        Helptext lives on the tooltip only; we deliberately do not duplicate it as
        an inline label because option helptext can run several sentences long and
        bloats the panel vertically when stacked under every field.
        """
        if spec.browser_modes and isinstance(widget, QLineEdit):
            return self._wrap_with_browsers(spec, widget)
        return widget

    def _wrap_with_browsers(self, spec: OptionSpec, widget: QLineEdit) -> QWidget:
        """Wrap a line edit with browse buttons for the configured browser modes."""
        row = QWidget()
        row.setMinimumWidth(0)
        row.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)  # type: ignore[attr-defined]
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(widget, 1)
        theme = QtTheme.default()
        for mode in spec.browser_modes:
            icon_key = self._browser_icon_key(mode, spec)
            button = QPushButton()
            button.setObjectName(f"qt-shell-browser-{mode}")
            button.setProperty("qt-shell-role", "browser")
            icon = icon_for_action(theme, icon_key)
            if not icon.isNull():
                button.setIcon(icon)
                button.setIconSize(QSize(16, 16))
            else:
                button.setText(self._browser_label(mode))
            button.setToolTip(self._tooltip_text(self._browser_tooltip(mode, spec)))
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # type: ignore[attr-defined]
            button.setFixedSize(QSize(24, 24))
            button.setFlat(True)
            button.clicked.connect(
                lambda _checked=False, m=mode, w=widget, f=spec.file_filter: self._browse(m, w, f)
            )
            layout.addWidget(button)
        return row

    @staticmethod
    def _browser_icon_key(mode: str, spec: OptionSpec) -> str:
        """Return the icon key to use for a browser button (Tk parity).

        Mirrors :class:`lib.gui.control_helper.FileBrowser.add_browser_buttons` so
        each filetype/option gets a recognisable icon (film reel for video, mountain
        for folders of images, server for model dirs, file page for generic files).

        Tk looks at the ``dest`` of the argparse argument; the schema may store
        the short opt (``-i``) so we combine ``dest``, the CLI ``switch`` and the
        human-facing ``title`` before scanning for option-name tokens.
        """
        filetypes = (spec.filetypes or "").lower()
        opt_name = " ".join(
            (
                (spec.dest or "").lower(),
                spec.switch.lstrip("-").lower().replace("-", "_"),
                spec.title.lower().replace(" ", "_"),
            )
        )
        if mode in ("file", "files") and filetypes == "video":
            return "browser_video"
        if mode == "folder":
            if filetypes == "image" or any(
                token in opt_name for token in ("frames", "faces", "input")
            ):
                return "browser_picture"
            if "model" in opt_name or filetypes == "model":
                return "browser_model"
        return f"browser_{mode}"

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

    def _browse(self, mode: str, widget: QLineEdit, file_filter: str = "") -> None:
        """Populate a line edit from a simple QFileDialog browser mode.

        ``file_filter`` is the standard Qt filter string (e.g.
        ``"Images (*.png *.jpg);;All files (*)"``); ignored for folder mode.
        """
        if mode == "folder":
            value = QFileDialog.getExistingDirectory(self, "Select Folder", widget.text())
        elif mode == "file":
            value, _ = QFileDialog.getOpenFileName(self, "Select File", widget.text(), file_filter)
        elif mode == "files":
            values, _ = QFileDialog.getOpenFileNames(
                self, "Select Files", widget.text(), file_filter
            )
            value = self._join_paths(values)
        elif mode == "save":
            value, _ = QFileDialog.getSaveFileName(self, "Save File", widget.text(), file_filter)
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
        """Return whether option metadata indicates a required argument.

        Prefers the explicit ``is_required`` flag; falls back to helptext/title
        sniffing for legacy specs that have not been migrated yet.
        """
        if spec.is_required:
            return True
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

    @classmethod
    def _is_choice_cluster(cls, spec: OptionSpec) -> bool:
        """Return true for radio/multi-select clusters that render as titled groups."""
        return cls._is_radio_group(spec) or cls._is_multi_select(spec)

    def _titled_cluster(self, spec: OptionSpec, widget: QWidget) -> QGroupBox:
        """Wrap a choice-cluster widget in a titled QGroupBox for Tk parity."""
        box = QGroupBox(spec.title)
        box.setObjectName("qt-shell-option-cluster")
        box.setProperty("qt-shell-role", "option-cluster")
        if spec.helptext:
            box.setToolTip(self._tooltip_text(spec.helptext))
        inner = QVBoxLayout(box)
        inner.setContentsMargins(10, 8, 10, 8)
        inner.setSpacing(4)
        inner.addWidget(widget)
        return box

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
    def _browser_tooltip(mode: str, spec: OptionSpec) -> str:
        """Return the Tk-parity tooltip for a browser mode/filetypes combo."""
        filetypes = (spec.filetypes or "").lower()
        opt_name = " ".join(
            (
                (spec.dest or "").lower(),
                spec.switch.lstrip("-").lower().replace("-", "_"),
                spec.title.lower().replace(" ", "_"),
            )
        )
        if mode in ("file", "files") and filetypes == "video":
            return "Select a video..."
        if mode == "folder":
            if filetypes == "image" or any(
                token in opt_name for token in ("frames", "faces", "input")
            ):
                return "Select a folder of images..."
            if "model" in opt_name or filetypes == "model":
                return "Select a model folder..."
        return {
            "folder": "Select a folder...",
            "file": "Select a file...",
            "files": "Select one or more files...",
            "save": "Select a save location...",
        }.get(mode, f"Browse for {spec.title}")

    _tooltip_text = staticmethod(_wrap_tooltip)

    def _choice_columns(self, choice_count: int) -> int:
        """Return the row-major column count for choice option groups."""
        return max(1, min(self._option_columns, choice_count))

    @staticmethod
    def _apply_widget_policy(widget: QWidget) -> None:
        """Apply common growth policy for option widgets."""
        widget.setMinimumWidth(0)
        if isinstance(widget, (QLineEdit, QComboBox)):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[attr-defined]
        else:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # type: ignore[attr-defined]

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
            return cls._slider_to_value(spec, slider.value())
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
                coerced: int | float = float(value)  # type: ignore[arg-type]
            else:
                coerced = int(float(str(value)))
        except (TypeError, ValueError):
            coerced = 0 if spec.default in (None, "") else spec.default  # type: ignore[assignment]
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
        return 10 ** cls._slider_decimals(spec)  # type: ignore[no-any-return]

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
        self._renderer = OptionsFormRenderer(option_columns=3)
        self._generate_button = QPushButton("Generate")
        self._run_button = QPushButton("Run")
        self._generate_button.setObjectName("qt-shell-command-generate")
        self._run_button.setObjectName("qt-shell-command-run")
        _command_theme = QtTheme.default()
        _gen_icon = icon_for_action(_command_theme, "generate")
        if not _gen_icon.isNull():
            self._generate_button.setIcon(_gen_icon)
            self._generate_button.setIconSize(QSize(16, 16))
        _run_icon = icon_for_action(_command_theme, "run")
        if not _run_icon.isNull():
            self._run_button.setIcon(_run_icon)
            self._run_button.setIconSize(QSize(16, 16))
        self._advanced_toggle = QCheckBox("Show advanced")
        self._show_advanced = False
        self._syncing_tabs = False
        self._command_value_cache: dict[str, dict[str, object]] = {}
        self._active_cached_command: str | None = None
        self._scroll: QScrollArea | None = None
        self._external_errors: tuple[str, ...] = ()
        self._build()
        self._primary_tabs.currentChanged.connect(self._set_primary_tab)
        self._tool_tabs.currentChanged.connect(self._set_tool_tab)
        self._command.currentTextChanged.connect(self._set_command_options)
        self._renderer.value_changed.connect(self._value_changed)
        self._advanced_toggle.toggled.connect(self._toggle_advanced)
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
        if command:
            self._command_value_cache[command] = dict(values)
        category = self._schema.category_for_command(command)
        if category is not None:
            self._set_category(category)
        if self._command.findText(command) == -1:
            self._command.addItem(command)
        self._command.setCurrentText(command)
        self._select_tabs_for_command(category, command)
        self._set_command_options(command)

    def select_command(self, command: str) -> bool:
        """Select a command tab by name without seeding any values.

        Returns ``True`` if the named command is available; ``False`` otherwise.
        """
        if not command:
            return False
        category = self._schema.category_for_command(command)
        if category is None:
            return False
        self._select_tabs_for_command(category, command)
        return True

    def clear_values(self) -> None:
        """Reset rendered fields to empty/default values."""
        self._command_value_cache.pop(self._command.currentText(), None)
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

        # The CLI ``description`` is surfaced through tab tooltips on Tk and is
        # intentionally suppressed in-panel here to keep the form density Tk-parity.
        self._command_info.setWordWrap(True)
        self._command_info.setObjectName("qt-shell-command-info")
        self._command_info.setVisible(False)

        self._advanced_toggle.setObjectName("qt-shell-command-advanced-toggle")
        self._advanced_toggle.setChecked(False)
        self._advanced_toggle.setToolTip("Show options flagged as advanced for this command")
        advanced_row = QHBoxLayout()
        advanced_row.setContentsMargins(0, 0, 0, 0)
        advanced_row.addWidget(self._advanced_toggle)
        advanced_row.addStretch(1)
        layout.addLayout(advanced_row)

        self._validation_label.setWordWrap(True)
        self._validation_label.setObjectName("qt-shell-command-errors")
        self._validation_label.setProperty("status", "error")
        self._validation_label.setVisible(False)
        layout.addWidget(self._validation_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        scroll.setObjectName("qt-shell-command-scroll")
        scroll.setWidget(self._renderer)
        self._scroll = scroll
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
            index = self._primary_tabs.addTab(command.title())
            tooltip = _wrap_tooltip(self._schema.command_info(command))
            if tooltip:
                self._primary_tabs.setTabToolTip(index, tooltip)
        if "tools" in self._schema.categories:
            self._primary_tabs.addTab("Tools")
            for command in self._schema.commands("tools"):
                tool_index = self._tool_tabs.addTab(command.title())
                tooltip = _wrap_tooltip(self._schema.command_info(command))
                if tooltip:
                    self._tool_tabs.setTabToolTip(tool_index, tooltip)

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
        previous = self._active_cached_command
        if previous and previous != command:
            # Stale external errors (e.g. launcher validation from another
            # command's Run path) should not bleed into a freshly selected
            # command's validation surface.
            self._external_errors = ()
        if previous and previous != command and self._renderer.rendered_switches:
            self._command_value_cache[previous] = self._renderer.values()
        options = self._schema.options(command)
        has_advanced = any(spec.is_advanced for spec in options)
        self._advanced_toggle.setVisible(has_advanced)
        if not self._show_advanced:
            options = tuple(spec for spec in options if not spec.is_advanced)
        self._renderer.set_options(options)
        if command in self._command_value_cache:
            self._renderer.apply_values(self._command_value_cache[command])
        self._set_command_info(command)
        self._update_validation()
        self._run_button.setText(command.title() if command else "Run")
        self._active_cached_command = command
        if self._scroll is not None:
            self._scroll.verticalScrollBar().setValue(0)
            self._scroll.horizontalScrollBar().setValue(0)

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

    def _toggle_advanced(self, checked: bool) -> None:
        """Persist toggle state and re-render the current command's options."""
        if self._show_advanced == checked:
            return
        self._show_advanced = checked
        self._set_command_options(self._command.currentText())

    def _value_changed(self) -> None:
        """Forward renderer changes after refreshing inline validation."""
        self._external_errors = ()
        self._update_validation()
        self.value_changed.emit()

    def _update_validation(self) -> None:
        """Display current validation errors inline."""
        errors = self.validation_errors()
        combined: tuple[str, ...] = errors + self._external_errors
        self._validation_label.setText("\n".join(combined))
        self._validation_label.setVisible(bool(combined))

    def set_external_errors(self, errors: T.Iterable[str]) -> None:
        """Display caller-supplied validation errors inline beneath the form.

        Native command launchers (e.g. the Qt Manual Tool) can call this to
        surface user-facing failures in the same panel-level validation area
        instead of only logging to the process console.
        """
        self._external_errors = tuple(message for message in errors if message)
        self._update_validation()

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
