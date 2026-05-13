#!/usr/bin/env python3
"""Qt shell state restore and live display refresh tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QTabWidget


class _Event:
    """Small runtime event stand-in."""

    def __init__(self, payload: dict[str, object], kind: str = "status") -> None:
        self.payload = payload
        self.kind = kind
        self.message = ""
        self.progress = None


def _main_window(qtbot, monkeypatch, tmp_path: Path):  # type:ignore[no-untyped-def]
    """Return a MainWindow with a deterministic schema."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec("faceswap", "extract", (OptionSpec("Input", "-i"),)),
            CommandSpec("faceswap", "train", (OptionSpec("Model", "-m"),)),
        )
    )
    window = MainWindow(schema)
    qtbot.addWidget(window)
    return window


def test_capture_ui_state_includes_display_geometry_and_sources(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """MainWindow should capture restorable display/session UI state."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    preview_source = tmp_path / "preview-output"
    window._preview_panel_widget.configure_output(preview_source)  # pylint:disable=protected-access
    window._main_splitter.setSizes([321, 654])  # pylint:disable=protected-access
    window._vertical_splitter.setSizes([700, 200])  # pylint:disable=protected-access
    tabs = window.findChild(QTabWidget, "qt-shell-display-tabs")
    assert tabs is not None
    tabs.setCurrentIndex(1)
    window.resize(1111, 777)

    state = window._capture_ui_state()  # pylint:disable=protected-access

    assert state["display_tab"] == "Preview"
    assert state["preview_source"] == str(preview_source)
    assert state["window_size"] == [1111, 777]
    assert state["main_splitter"] == window._main_splitter.sizes()  # pylint:disable=protected-access
    assert state["vertical_splitter"] == window._vertical_splitter.sizes()  # pylint:disable=protected-access


def test_restore_ui_state_restores_geometry_and_panel_sources(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """MainWindow should restore display tab, geometry and panel source paths."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    restored: dict[str, object] = {}
    monkeypatch.setattr(
        window._analysis_panel_widget,  # pylint:disable=protected-access
        "restore_source",
        lambda source: restored.setdefault("analysis", source) or True,
    )
    monkeypatch.setattr(
        window._preview_panel_widget,  # pylint:disable=protected-access
        "restore_source",
        lambda source: restored.setdefault("preview", source) or True,
    )
    monkeypatch.setattr(
        window._graph_panel_widget,  # pylint:disable=protected-access
        "restore_source",
        lambda source: restored.setdefault("graph", source) or True,
    )

    window._restore_ui_state(  # pylint:disable=protected-access
        {
            "display_tab": "Preview",
            "window_size": [900, 600],
            "main_splitter": [300, 600],
            "vertical_splitter": [500, 100],
            "analysis_source": "analysis_state.json",
            "preview_source": "preview-output",
            "graph_source": "graph_state.json",
        }
    )
    tabs = window.findChild(QTabWidget, "qt-shell-display-tabs")
    assert tabs is not None

    assert restored == {
        "analysis": "analysis_state.json",
        "preview": "preview-output",
        "graph": "graph_state.json",
    }
    assert tabs.tabText(tabs.currentIndex()) == "Preview"
    assert window._main_splitter.sizes() == [300, 600]  # pylint:disable=protected-access
    assert window._vertical_splitter.sizes() == [500, 100]  # pylint:disable=protected-access
    assert window.width() == 900
    assert window.height() == 600


def test_restore_last_session_applies_saved_ui_state(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Restoring the last session should also apply cached UI state."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "last.fsw")
    window._project_store.save(  # pylint:disable=protected-access
        filename,
        window._session_service.snapshot_project(  # pylint:disable=protected-access
            window._project,  # pylint:disable=protected-access
            "extract",
            {"-i": "input"},
        ),
    )
    restored = {}
    monkeypatch.setattr(
        window,
        "_restore_ui_state",
        lambda state: restored.update(state),
    )
    window._last_session.save(  # pylint:disable=protected-access
        filename,
        "project",
        {"display_tab": "Graph", "preview_source": "preview-output"},
    )

    assert window._restore_last_session() is True  # pylint:disable=protected-access

    assert restored == {"display_tab": "Graph", "preview_source": "preview-output"}


def test_runtime_events_refresh_requested_display_panels(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Runtime refresh payloads should route to the matching display panels."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        window._analysis_panel_widget,  # pylint:disable=protected-access
        "refresh_session",
        lambda: calls.append("analysis") or True,
    )
    monkeypatch.setattr(
        window._preview_panel_widget,  # pylint:disable=protected-access
        "refresh_preview",
        lambda: calls.append("preview") or True,
    )
    monkeypatch.setattr(
        window._graph_panel_widget,  # pylint:disable=protected-access
        "refresh_graph",
        lambda: calls.append("graph") or True,
    )

    window._refresh_displays_from_event(  # pylint:disable=protected-access
        _Event({"preview_refresh": True, "graph_refresh": True})
    )

    assert calls == ["preview", "graph"]


def test_saved_model_event_refreshes_all_display_panels(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Saved-model/session events should refresh analysis, preview and graph panels."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        window._analysis_panel_widget,  # pylint:disable=protected-access
        "refresh_session",
        lambda: calls.append("analysis") or True,
    )
    monkeypatch.setattr(
        window._preview_panel_widget,  # pylint:disable=protected-access
        "refresh_preview",
        lambda: calls.append("preview") or True,
    )
    monkeypatch.setattr(
        window._graph_panel_widget,  # pylint:disable=protected-access
        "refresh_graph",
        lambda: calls.append("graph") or True,
    )

    window._refresh_displays_from_event(  # pylint:disable=protected-access
        _Event({"saved_model": True}, kind="saved_model")
    )

    assert calls == ["analysis", "preview", "graph"]
