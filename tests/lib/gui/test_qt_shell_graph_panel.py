#!/usr/bin/env python3
"""Qt Graph runtime panel tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QLabel, QPlainTextEdit, QPushButton, QTabWidget

from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.training_graph_service import TrainingGraphService


class _SessionDouble:
    """Small Analysis Session test double."""

    def __init__(self) -> None:
        self.is_loaded = False
        self.session_ids = [1, 2]
        self.initialized: list[tuple[str, str, bool]] = []
        self.loss_by_session: dict[int | None, dict[str, object]] = {
            None: {"loss_a": [3.0, 2.0, 1.0], "loss_b": [1.0, 2.0, 3.0]},
            1: {"loss_a": [3.0, 2.0, 1.0]},
            2: {"loss_b": [1.0, 2.0, 3.0]},
        }

    def initialize_session(
        self,
        model_folder: str,
        model_name: str,
        is_training: bool = False,
    ) -> None:
        """Capture initialization calls."""
        self.is_loaded = True
        self.initialized.append((model_folder, model_name, is_training))

    def get_loss(self, session_id: int | None) -> dict[str, object]:
        """Return loss values."""
        return self.loss_by_session.get(session_id, {})

    def get_loss_keys(self, session_id: int | None) -> list[str]:
        """Return loss keys."""
        return sorted(self.get_loss(session_id))


def _state_file(tmp_path: Path, name: str = "model") -> Path:
    """Create a model state file."""
    state_file = tmp_path / f"{name}_state.json"
    state_file.write_text("{}", encoding="utf-8")
    return state_file


def _panel(session: _SessionDouble | None = None):  # type:ignore[no-untyped-def]
    """Return a GraphPanel with injected graph service."""
    from lib.gui.qt_shell.graph_panel import GraphPanel

    session = _SessionDouble() if session is None else session
    return GraphPanel(TrainingGraphService(session))


def _button(panel, name: str) -> QPushButton:  # type:ignore[no-untyped-def]
    """Return a GraphPanel button by object name suffix."""
    button = panel.findChild(QPushButton, f"qt-shell-graph-{name}")
    assert button is not None
    return button


def _label(panel, name: str) -> QLabel:  # type:ignore[no-untyped-def]
    """Return a GraphPanel label by object name suffix."""
    label = panel.findChild(QLabel, f"qt-shell-graph-{name}")
    assert label is not None
    return label


def _graph_text(panel) -> QPlainTextEdit:  # type:ignore[no-untyped-def]
    """Return the GraphPanel text display."""
    widget = panel.findChild(QPlainTextEdit, "qt-shell-graph-text")
    assert widget is not None
    return widget


def _session_combo(panel) -> QComboBox:  # type:ignore[no-untyped-def]
    """Return the GraphPanel session selector."""
    combo = panel.findChild(QComboBox, "qt-shell-graph-session")
    assert combo is not None
    return combo


def test_graph_panel_initial_state(qtbot) -> None:  # type:ignore[no-untyped-def]
    """GraphPanel should start empty with only Open enabled."""
    panel = _panel()
    qtbot.addWidget(panel)

    assert _label(panel, "source").text() == "No graph source configured"
    assert _label(panel, "status").text() == "No graph data loaded"
    assert _graph_text(panel).toPlainText() == "No graph data loaded"
    assert _button(panel, "open").isEnabled() is True
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False
    assert _session_combo(panel).isEnabled() is False


def test_graph_panel_apply_context_configures_source(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Command context should configure model source for a future graph refresh."""
    panel = _panel()
    qtbot.addWidget(panel)
    context = CommandExecutionContext(model_folder=str(tmp_path), model_name="model")

    applied = panel.apply_context(context)

    assert applied is True
    assert panel.service.source is not None
    assert panel.service.source.model_dir == tmp_path
    assert panel.service.source.model_name == "model"
    assert _label(panel, "source").text() == f"Graph source: model  |  {tmp_path}"
    assert _button(panel, "refresh").isEnabled() is True
    assert _button(panel, "clear").isEnabled() is True


def test_graph_panel_apply_context_ignores_missing_model_info(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Command contexts without model details should be ignored."""
    panel = _panel()
    qtbot.addWidget(panel)

    applied = panel.apply_context(CommandExecutionContext())

    assert applied is False
    assert panel.service.source is None
    assert _label(panel, "source").text() == "No graph source configured"


def test_graph_panel_loads_state_file_and_renders_series(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Loading a state source should render loss sparklines and summary text."""
    session = _SessionDouble()
    panel = _panel(session)
    qtbot.addWidget(panel)
    state_file = _state_file(tmp_path, "my_model")

    loaded = panel.load_source(state_file)
    graph_text = _graph_text(panel).toPlainText()
    combo = _session_combo(panel)

    assert loaded is True
    assert session.initialized == [(str(tmp_path), "my_model", False)]
    assert "loss_a:" in graph_text
    assert "loss_b:" in graph_text
    assert "points=3" in graph_text
    assert _label(panel, "status").text().startswith("Loaded 2 series, 3 points")
    assert combo.isEnabled() is True
    assert combo.count() == 3
    assert combo.itemText(0) == "All sessions"
    assert combo.itemText(1) == "1"
    assert combo.itemText(2) == "2"


def test_graph_panel_refresh_loads_configured_training_source(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Refresh should initialize a configured training source once state exists."""
    session = _SessionDouble()
    panel = _panel(session)
    qtbot.addWidget(panel)
    panel.apply_context(CommandExecutionContext(model_folder=str(tmp_path), model_name="model"))
    _state_file(tmp_path, "model")

    refreshed = panel.refresh_graph()

    assert refreshed is True
    assert session.initialized == [(str(tmp_path), "model", True)]
    assert "loss_a:" in _graph_text(panel).toPlainText()


def test_graph_panel_session_selector_filters_series(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Selecting a session should refresh graph text for that session."""
    panel = _panel(_SessionDouble())
    qtbot.addWidget(panel)
    assert panel.load_source(_state_file(tmp_path, "model")) is True
    combo = _session_combo(panel)

    combo.setCurrentIndex(1)

    assert panel.service.session_id == 1
    graph_text = _graph_text(panel).toPlainText()
    assert "loss_a:" in graph_text
    assert "loss_b:" not in graph_text


def test_graph_panel_load_failure_displays_error(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Load failures should stay inside the panel."""
    panel = _panel()
    qtbot.addWidget(panel)

    loaded = panel.load_source(tmp_path / "missing_state.json")

    assert loaded is False
    assert "does not exist" in _label(panel, "status").text()
    assert _graph_text(panel).toPlainText() == "No graph data loaded"


def test_graph_panel_clear_resets_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Clear should reset source, data, combo and buttons."""
    panel = _panel(_SessionDouble())
    qtbot.addWidget(panel)
    assert panel.load_source(_state_file(tmp_path, "model")) is True

    panel.clear_graph()

    assert panel.service.source is None
    assert _session_combo(panel).count() == 0
    assert _label(panel, "source").text() == "No graph source configured"
    assert _label(panel, "status").text() == "No graph data loaded"
    assert _graph_text(panel).toPlainText() == "No graph data loaded"
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False


def test_main_window_uses_real_graph_panel(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """MainWindow should install a real GraphPanel in the Graph tab."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.graph_panel import GraphPanel
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema((CommandSpec("faceswap", "train", (OptionSpec("Model Dir", "-m"),)),))
    window = MainWindow(schema)
    qtbot.addWidget(window)
    tabs = window.findChild(QTabWidget, "qt-shell-display-tabs")
    graph = window.findChild(GraphPanel, "qt-shell-graph-panel")
    assert tabs is not None
    assert graph is not None

    window._apply_graph_context(  # pylint:disable=protected-access
        CommandExecutionContext(model_folder=str(tmp_path), model_name="model")
    )

    assert graph.service.source is not None
    assert graph.service.source.model_dir == tmp_path
    assert tabs.tabText(2) == "Graph"
