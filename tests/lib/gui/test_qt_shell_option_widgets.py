#!/usr/bin/env python3
"""Qt command option widget tests."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QWidget,
)

from lib.gui.services.command_builder import CommandBuilder
from tests.lib.gui._qt_helpers import option_spec as _option_spec


def _command_panel(*options):  # type:ignore[no-untyped-def]
    """Return a CommandPanel backed by a single extract command.

    Wraps the shared ``tests.lib.gui._qt_helpers.command_panel`` helper but
    omits the ``qtbot.addWidget`` call, because some legacy tests in this file
    add the widget themselves.  New tests should call the shared helper
    directly.
    """
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec

    return CommandPanel(CommandSchema((CommandSpec("faceswap", "extract", tuple(options)),)))


def test_radio_option_extracts_and_restores_value(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Radio metadata should render exclusive choices and restore stored values."""
    panel = _command_panel(
        _option_spec(
            "Mode",
            "--mode",
            str,
            "one",
            ("one", "two"),
            is_radio=True,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--mode")
    buttons = {button.text(): button for button in widget.findChildren(QRadioButton)}

    buttons["two"].setChecked(True)

    assert panel.command_spec()[2] == {"--mode": "two"}

    panel.set_command("extract", {"--mode": "one"})
    widget = panel.renderer.widget_for_switch("--mode")
    buttons = {button.text(): button for button in widget.findChildren(QRadioButton)}

    assert buttons["one"].isChecked() is True
    assert panel.command_spec()[2] == {"--mode": "one"}


def test_multi_option_extracts_and_restores_values(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Multi-option metadata should emit selected choices and restore stored choices."""
    panel = _command_panel(
        _option_spec(
            "Features",
            "--features",
            str,
            ("fast",),
            ("fast", "safe", "slow"),
            is_multi_option=True,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--features")
    checkboxes = {checkbox.text(): checkbox for checkbox in widget.findChildren(QCheckBox)}

    checkboxes["safe"].setChecked(True)
    checkboxes["fast"].setChecked(False)

    assert panel.command_spec()[2] == {"--features": ["safe"]}

    panel.set_command("extract", {"--features": ["fast", "slow"]})
    widget = panel.renderer.widget_for_switch("--features")
    checkboxes = {checkbox.text(): checkbox for checkbox in widget.findChildren(QCheckBox)}

    assert checkboxes["fast"].isChecked() is True
    assert checkboxes["safe"].isChecked() is False
    assert checkboxes["slow"].isChecked() is True
    assert panel.command_spec()[2] == {"--features": ["fast", "slow"]}


def test_empty_multi_option_skips_cli_emission(qtbot) -> None:  # type:ignore[no-untyped-def]
    """An empty multi-option selection should be omitted from built CLI args."""
    panel = _command_panel(
        _option_spec(
            "Features",
            "--features",
            str,
            "",
            ("fast", "safe"),
            is_multi_option=True,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--features")

    for checkbox in widget.findChildren(QCheckBox):
        checkbox.setChecked(False)

    assert panel.command_spec()[2] == {"--features": ""}
    assert CommandBuilder.build_options(panel.command_spec()[2]) == []


def test_store_false_checkbox_emits_when_unchecked(qtbot) -> None:  # type:ignore[no-untyped-def]
    """store_false checkboxes should emit their switch when unchecked."""
    panel = _command_panel(
        _option_spec(
            "Enabled Feature",
            "--disable-feature",
            bool,
            True,
            action="store_false",
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--disable-feature")
    assert isinstance(widget, QCheckBox)

    assert widget.isChecked() is True
    assert CommandBuilder.build_options(panel.command_spec()[2]) == []

    widget.setChecked(False)

    assert panel.command_spec()[2] == {"--disable-feature": True}
    assert CommandBuilder.build_options(panel.command_spec()[2]) == ["--disable-feature"]

    panel.set_command("extract", {"--disable-feature": True})
    widget = panel.renderer.widget_for_switch("--disable-feature")
    assert isinstance(widget, QCheckBox)

    assert widget.isChecked() is False
    assert CommandBuilder.build_options(panel.command_spec()[2]) == ["--disable-feature"]


def test_int_slider_extracts_restores_and_clamps(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Integer sliders should sync their line edit, restore values and clamp bounds."""
    panel = _command_panel(
        _option_spec(
            "Batch Size",
            "--batch-size",
            int,
            64,
            action="Slider",
            slider_min=1,
            slider_max=256,
            slider_rounding=1,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--batch-size")
    slider = widget.findChild(QSlider)
    line_edit = widget.findChild(QLineEdit)
    assert slider is not None
    assert line_edit is not None

    assert slider.value() == 64
    assert line_edit.text() == "64"
    assert panel.command_spec()[2] == {"--batch-size": 64}

    panel.set_command("extract", {"--batch-size": 128})
    widget = panel.renderer.widget_for_switch("--batch-size")
    slider = widget.findChild(QSlider)
    line_edit = widget.findChild(QLineEdit)
    assert slider is not None
    assert line_edit is not None
    assert slider.value() == 128
    assert line_edit.text() == "128"

    line_edit.setText("300")
    line_edit.editingFinished.emit()
    assert slider.value() == 256
    assert line_edit.text() == "256"
    assert panel.command_spec()[2] == {"--batch-size": 256}

    line_edit.setText("-5")
    line_edit.editingFinished.emit()
    assert slider.value() == 1
    assert line_edit.text() == "1"

    line_edit.setText("bad")
    line_edit.editingFinished.emit()
    assert slider.value() == 64
    assert line_edit.text() == "64"


def test_float_slider_extracts_rounds_and_clamps(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Float sliders should preserve configured precision and clamp bounds."""
    panel = _command_panel(
        _option_spec(
            "Threshold",
            "--threshold",
            float,
            0.5,
            action="Slider",
            slider_min=0.01,
            slider_max=0.99,
            slider_rounding=2,
        )
    )
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--threshold")
    slider = widget.findChild(QSlider)
    line_edit = widget.findChild(QLineEdit)
    assert slider is not None
    assert line_edit is not None

    assert slider.value() == 50
    assert line_edit.text() == "0.5"

    line_edit.setText("0.456")
    line_edit.editingFinished.emit()
    assert slider.value() == 46
    assert line_edit.text() == "0.46"
    assert panel.command_spec()[2] == {"--threshold": 0.46}

    line_edit.setText("2.5")
    line_edit.editingFinished.emit()
    assert slider.value() == 99
    assert line_edit.text() == "0.99"

    line_edit.setText("bad")
    line_edit.editingFinished.emit()
    assert slider.value() == 50
    assert line_edit.text() == "0.5"


def test_browse_buttons_get_stable_object_names(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Path options should wrap line edits with browse buttons named by mode."""
    panel = _command_panel(
        _option_spec(
            "Input",
            "-i",
            helptext="Pick paths",
            browser_modes=("folder", "file", "files", "save"),
        )
    )
    qtbot.addWidget(panel)

    assert {button.objectName() for button in panel.renderer.findChildren(QPushButton)} == {
        "qt-shell-browser-folder",
        "qt-shell-browser-file",
        "qt-shell-browser-files",
        "qt-shell-browser-save",
    }
    assert isinstance(panel.renderer.widget_for_switch("-i"), QLineEdit)


def test_group_sections_render_labels_and_containers(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Grouped options should render titled group sections."""
    panel = _command_panel(
        _option_spec("Input", "-i", group="Data"),
        _option_spec("Output", "-o", group="Data"),
        _option_spec("Debug", "--debug", bool, False, group="_master"),
    )
    qtbot.addWidget(panel)
    # Titled groups render as OptionGroupDrawer (disclosure-arrow toggle)
    groups = panel.renderer.findChildren(QWidget, "qt-shell-option-group")

    assert [group.title() for group in groups] == ["Data"]
    assert all(group.isCheckable() for group in groups)
    assert len(groups) == 1
    assert set(panel.renderer.rendered_switches) == {"-i", "-o", "--debug"}


def test_choice_clusters_render_as_titled_groupboxes(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Radio and multi-select clusters should render inside titled QGroupBoxes.

    Tk renders each detector/aligner/mask plugin cluster inside a bordered
    LabelFrame with the option name as the frame title. The Qt panel mirrors
    that with a ``qt-shell-option-cluster`` QGroupBox spanning the form row,
    so a wide choice grid no longer clips its left-hand label column.
    """
    from PySide6.QtWidgets import QGroupBox

    panel = _command_panel(
        _option_spec(
            "Detector",
            "--detector",
            str,
            "cv2-dnn",
            ("cv2-dnn", "mtcnn", "retinaface"),
            is_radio=True,
        ),
        _option_spec(
            "Color Adjust",
            "--color",
            str,
            ("avg-color",),
            ("avg-color", "manual-balance", "match-hist"),
            is_multi_option=True,
        ),
        _option_spec("Input", "-i"),
    )
    qtbot.addWidget(panel)

    clusters = panel.renderer.findChildren(QGroupBox, "qt-shell-option-cluster")
    titles = {cluster.title() for cluster in clusters}

    assert titles == {"Detector", "Color Adjust"}
    assert len(clusters) == 2


def test_plain_bool_options_pack_into_horizontal_cluster(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Plain boolean options should be collected into a horizontal grid (Tk parity).

    Tk renders boolean controls in a shared ``checkbuttons_frame`` so options like
    ``Compile`` / ``Skip Existing`` / ``Skip Existing Faces`` lay out left-to-right
    on one row rather than stacking vertically as separate labeled rows.
    """
    from PySide6.QtWidgets import QGridLayout

    panel = _command_panel(
        _option_spec("Compile", "--compile", bool, False),
        _option_spec("Skip Existing", "--skip-existing", bool, False),
        _option_spec("Skip Existing Faces", "--skip-existing-faces", bool, False),
    )
    qtbot.addWidget(panel)

    bool_cluster = panel.renderer.findChild(QWidget, "qt-shell-option-bool-cluster")
    assert bool_cluster is not None, "Plain bools should be grouped in a bool cluster"
    assert isinstance(bool_cluster.layout(), QGridLayout)

    checkboxes = bool_cluster.findChildren(QCheckBox)
    assert [box.text() for box in checkboxes] == [
        "Compile",
        "Skip Existing",
        "Skip Existing Faces",
    ]
    # All three should share the same row in the grid for horizontal layout
    grid = bool_cluster.layout()
    rows = {grid.getItemPosition(grid.indexOf(box))[0] for box in checkboxes}
    assert rows == {0}, f"Expected all checkboxes on row 0, got rows: {rows}"


def test_required_option_label_renders_asterisk(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Required options should render a red-asterisk marker in their label."""
    from PySide6.QtCore import Qt

    panel = _command_panel(
        _option_spec("Input Dir", "-i", is_required=True),
        _option_spec("Optional", "--opt"),
    )
    qtbot.addWidget(panel)

    labels = panel.renderer.findChildren(QLabel, "qt-shell-option-label-required")
    plain_labels = panel.renderer.findChildren(QLabel, "qt-shell-option-label")

    assert len(labels) == 1
    assert "Input Dir" in labels[0].text()
    assert "*" in labels[0].text()
    assert labels[0].textFormat() == Qt.RichText
    assert labels[0].property("required") is True
    assert [lbl.text() for lbl in plain_labels] == ["Optional"]


def test_required_flag_drives_validation(qtbot) -> None:  # type:ignore[no-untyped-def]
    """The explicit is_required flag should trigger inline validation."""
    panel = _command_panel(
        _option_spec("Model Dir", "-m", is_required=True),
        _option_spec("Trainer", "-t", default="original"),
    )
    qtbot.addWidget(panel)

    assert panel.validation_errors() == ("Model Dir is required",)

    panel.set_command("extract", {"-m": "/tmp/x", "-t": "original"})
    assert panel.validation_errors() == ()


def test_file_filter_threaded_to_qfile_dialog(monkeypatch, qtbot) -> None:  # type:ignore[no-untyped-def]
    """OptionSpec.file_filter must be passed to QFileDialog browsers."""
    from PySide6.QtWidgets import QFileDialog

    captured: dict[str, str] = {}

    def fake_open(parent, title, dir_, file_filter):
        captured["file"] = file_filter
        return ("/tmp/picked.png", "")

    def fake_save(parent, title, dir_, file_filter):
        captured["save"] = file_filter
        return ("/tmp/save.png", "")

    def fake_files(parent, title, dir_, file_filter):
        captured["files"] = file_filter
        return (["/tmp/a.png", "/tmp/b.png"], "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(fake_open))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(fake_files))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_save))

    panel = _command_panel(
        _option_spec(
            "Input",
            "-i",
            helptext="Pick image",
            browser_modes=("file", "files", "save"),
            file_filter="Images (*.png *.jpg);;All files (*)",
        ),
    )
    qtbot.addWidget(panel)

    buttons = {b.objectName(): b for b in panel.renderer.findChildren(QPushButton)}
    buttons["qt-shell-browser-file"].click()
    buttons["qt-shell-browser-files"].click()
    buttons["qt-shell-browser-save"].click()

    assert captured["file"] == "Images (*.png *.jpg);;All files (*)"
    assert captured["files"] == "Images (*.png *.jpg);;All files (*)"
    assert captured["save"] == "Images (*.png *.jpg);;All files (*)"


def test_group_section_collapses_when_unchecked(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Toggling the drawer should hide all fields in that group."""
    panel = _command_panel(
        _option_spec("Input", "-i", group="Data"),
        _option_spec("Output", "-o", group="Data"),
    )
    qtbot.addWidget(panel)
    group = panel.renderer.findChildren(QWidget, "qt-shell-option-group")[0]
    fields = group.findChildren(QLineEdit)
    assert fields and all(field.isVisibleTo(group) for field in fields)

    group.setChecked(False)
    assert all(not field.isVisibleTo(group) for field in fields)

    group.setChecked(True)
    assert all(field.isVisibleTo(group) for field in fields)


def test_advanced_toggle_filters_advanced_options(qtbot) -> None:  # type:ignore[no-untyped-def]
    """The 'Show advanced' toggle should hide is_advanced options when off."""
    panel = _command_panel(
        _option_spec("Basic", "-b"),
        _option_spec("Verbose", "-v", is_advanced=True),
    )
    qtbot.addWidget(panel)

    # Toggle is visible because an advanced option exists, but it starts off.
    toggle = panel.findChild(QCheckBox, "qt-shell-command-advanced-toggle")
    assert toggle is not None and not toggle.isHidden()
    assert toggle.isChecked() is False
    assert set(panel.renderer.rendered_switches) == {"-b"}

    toggle.setChecked(True)
    assert set(panel.renderer.rendered_switches) == {"-b", "-v"}

    toggle.setChecked(False)
    assert set(panel.renderer.rendered_switches) == {"-b"}


def test_advanced_toggle_hidden_when_no_advanced_options(qtbot) -> None:  # type:ignore[no-untyped-def]
    """The advanced toggle hides itself when the current command has no advanced options."""
    panel = _command_panel(
        _option_spec("Basic", "-b"),
        _option_spec("Other", "-o"),
    )
    qtbot.addWidget(panel)
    toggle = panel.findChild(QCheckBox, "qt-shell-command-advanced-toggle")
    assert toggle is not None
    assert toggle.isHidden() is True


def test_inline_error_clears_when_required_field_filled(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Inline validation error label should hide once a required field is provided."""
    panel = _command_panel(_option_spec("Model", "-m", is_required=True))
    qtbot.addWidget(panel)

    error_label = panel.findChild(QLabel, "qt-shell-command-errors")
    assert error_label is not None
    assert not error_label.isHidden()
    assert "Model is required" in error_label.text()

    widget = panel.renderer.widget_for_switch("-m")
    assert isinstance(widget, QLineEdit)
    widget.setText("/tmp/model")
    widget.textEdited.emit("/tmp/model")

    assert error_label.isHidden() is True
    assert panel.validation_errors() == ()


def test_group_drawer_uses_disclosure_arrow_not_checkbox(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Group drawers must use a QToolButton with arrow indicator, not a QCheckBox.

    Regression for an earlier iteration where QGroupBox.setCheckable rendered a
    native checkbox next to (and on macOS overlapping) the group title.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QToolButton

    from lib.gui.qt_shell.command_panel import OptionGroupDrawer

    panel = _command_panel(
        _option_spec("Input", "-i", group="Data"),
    )
    qtbot.addWidget(panel)

    drawers = panel.renderer.findChildren(OptionGroupDrawer, "qt-shell-option-group")
    assert len(drawers) == 1
    drawer = drawers[0]

    toggle = drawer.findChild(QToolButton, "qt-shell-option-group-toggle")
    assert toggle is not None, "drawer header must be a QToolButton"
    assert toggle.text() == "Data"
    assert toggle.isCheckable() is True
    assert toggle.isChecked() is True
    assert toggle.arrowType() == Qt.DownArrow

    drawer.setChecked(False)
    assert toggle.arrowType() == Qt.RightArrow

    drawer.setChecked(True)
    assert toggle.arrowType() == Qt.DownArrow


def test_helptext_remains_tooltip_only_no_inline_hint(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Helptext is exposed via tooltips; no inline hint label is rendered.

    Regression for an earlier iteration where every option grew a multi-line
    helptext label under the control, bloating the panel vertically.
    """
    panel = _command_panel(
        _option_spec(
            "Detector",
            "-D",
            helptext="Choose a face detector. Some have configurable settings.",
        ),
    )
    qtbot.addWidget(panel)

    hint_labels = panel.renderer.findChildren(QLabel, "qt-shell-option-hint")
    assert hint_labels == []

    widget = panel.renderer.widget_for_switch("-D")
    assert widget.toolTip() == "Choose a face detector. Some have configurable settings."
