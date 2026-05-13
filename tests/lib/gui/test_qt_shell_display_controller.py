#!/usr/bin/env python3
"""Qt display controller tests."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QTabWidget, QWidget


@dataclass(frozen=True)
class _RuntimeEvent:
    """RuntimeEvent-shaped test double."""

    kind: str
    message: str = ""
    progress: float | None = None
    payload: dict[str, object] | None = None


def _controller(qtbot):  # type:ignore[no-untyped-def]
    """Return a DisplayController with an empty tab widget."""
    from lib.gui.qt_shell.display_controller import DisplayController

    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    return DisplayController(tabs), tabs


def test_controller_starts_with_analysis_only(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Analysis should be present before runtime state creates dynamic tabs."""
    controller, _tabs = _controller(qtbot)

    assert controller.tab_names() == ("Analysis",)
    assert controller.visible_tab_names() == ("Analysis",)


def test_extract_runtime_event_creates_preview_tab(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Extract runtime events should create Preview without Graph."""
    controller, _tabs = _controller(qtbot)

    consumed = controller.consume_event(
        _RuntimeEvent("process", payload={"command": "extract", "state": "running"})
    )

    assert consumed is True
    assert controller.tab_names() == ("Analysis", "Preview")
    assert controller.visible_tab_names() == ("Analysis", "Preview")
    assert controller.command == "extract"


def test_convert_progress_event_creates_preview_tab(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Convert progress events should also be enough to make Preview relevant."""
    controller, _tabs = _controller(qtbot)

    consumed = controller.consume_event(
        _RuntimeEvent("progress", progress=42.0, payload={"command": "convert"})
    )

    assert consumed is True
    assert controller.tab_names() == ("Analysis", "Preview")
    assert controller.visible_tab_names() == ("Analysis", "Preview")


def test_train_runtime_event_creates_graph_tab_without_preview(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Training should expose Graph but not extract/convert Preview."""
    controller, _tabs = _controller(qtbot)

    controller.consume_event(
        _RuntimeEvent("runtime", payload={"command": "train", "state": "started"})
    )

    assert controller.tab_names() == ("Analysis", "Graph")
    assert controller.visible_tab_names() == ("Analysis", "Graph")
    assert controller.command == "train"


def test_runtime_display_state_payload_controls_tabs(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Runtime display-state payloads should drive the same dynamic tab rules."""
    controller, _tabs = _controller(qtbot)

    consumed = controller.consume_event(
        _RuntimeEvent("state", payload={"display": "train", "running_task": True})
    )

    assert consumed is True
    assert controller.visible_tab_names() == ("Analysis", "Graph")


def test_terminal_runtime_event_removes_dynamic_tabs(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Finished process events should return the display area to Analysis only."""
    controller, _tabs = _controller(qtbot)

    controller.set_runtime_state("train", running=True)
    consumed = controller.consume_event(
        _RuntimeEvent("process", payload={"command": "train", "state": "finished"})
    )

    assert consumed is True
    assert controller.tab_names() == ("Analysis",)
    assert controller.visible_tab_names() == ("Analysis",)
    assert controller.command is None


def test_unknown_events_are_not_consumed(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Events without display-relevant state should not mutate tabs."""
    controller, _tabs = _controller(qtbot)

    consumed = controller.consume_event(_RuntimeEvent("log", message="plain output"))

    assert consumed is False
    assert controller.tab_names() == ("Analysis",)


def test_preserve_existing_tabs_hides_instead_of_removing(qtbot) -> None:  # type:ignore[no-untyped-def]
    """MainWindow can attach to existing placeholders without changing tab ownership."""
    from lib.gui.qt_shell.display_controller import DisplayController

    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.addTab(QWidget(), "Analysis")
    tabs.addTab(QWidget(), "Preview")
    tabs.addTab(QWidget(), "Graph")

    controller = DisplayController(tabs, preserve_existing_tabs=True)

    assert controller.tab_names() == ("Analysis", "Preview", "Graph")
    assert controller.visible_tab_names() == ("Analysis",)

    controller.set_runtime_state("train", running=True)

    assert controller.tab_names() == ("Analysis", "Preview", "Graph")
    assert controller.visible_tab_names() == ("Analysis", "Graph")
