#!/usr/bin/env python3
"""Tests for graph data service."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.gui.services.training_graph_service import TrainingGraphError, TrainingGraphService


class _SessionDouble:
    """Small Analysis Session test double."""

    def __init__(self) -> None:
        self.is_loaded = False
        self.session_ids = [1, 2]
        self.initialized: list[tuple[str, str, bool]] = []
        self.loss_by_session = {
            None: {"loss_a": [1.0, 0.5, 0.25], "loss_b": [2.0, 1.0, 0.5]},
            1: {"loss_a": [1.0, 0.5], "loss_b": [2.0, 1.0]},
            2: {"loss_a": [0.25], "loss_b": [0.5]},
        }

    def initialize_session(self, model_folder: str, model_name: str, is_training: bool = False) -> None:
        """Capture initialization calls."""
        self.is_loaded = True
        self.initialized.append((model_folder, model_name, is_training))

    def get_loss(self, session_id: int | None) -> dict[str, list[float]]:
        """Return loss data."""
        return self.loss_by_session[session_id]

    def get_loss_keys(self, session_id: int | None) -> list[str]:
        """Return available loss keys."""
        return list(self.loss_by_session[session_id])


def _state_file(tmp_path: Path, name: str = "model") -> Path:
    """Create a model state file."""
    state_file = tmp_path / f"{name}_state.json"
    state_file.write_text("{}", encoding="utf-8")
    return state_file


def test_graph_service_configures_pending_source(tmp_path: Path) -> None:
    """GraphService should accept command-derived pending model context."""
    service = TrainingGraphService(_SessionDouble())

    source = service.configure(model_folder=tmp_path, model_name="model")

    assert source is not None
    assert source.model_dir == tmp_path
    assert source.model_name == "model"
    assert source.state_file == tmp_path / "model_state.json"
    assert service.snapshot.is_empty is True


def test_graph_service_loads_source_and_refreshes_loss(tmp_path: Path) -> None:
    """Loading a source should initialize the session and return loss series."""
    session = _SessionDouble()
    service = TrainingGraphService(session)

    snapshot = service.load_source(_state_file(tmp_path, "model"))

    assert session.initialized == [(str(tmp_path), "model", False)]
    assert service.is_loaded is True
    assert service.session_ids == (1, 2)
    assert snapshot.point_count == 3
    assert [series.name for series in snapshot.series] == ["loss_a", "loss_b"]
    assert snapshot.series[0].values == (1.0, 0.5, 0.25)


def test_graph_service_loads_configured_source_when_state_exists(tmp_path: Path) -> None:
    """Configured context should load when the model state file appears."""
    session = _SessionDouble()
    service = TrainingGraphService(session)
    service.configure(model_folder=tmp_path, model_name="model")
    _state_file(tmp_path, "model")

    snapshot = service.load_configured_source(is_training=True)

    assert session.initialized == [(str(tmp_path), "model", True)]
    assert snapshot.is_empty is False


def test_graph_service_rejects_missing_configured_state(tmp_path: Path) -> None:
    """Configured context should report pending state files."""
    service = TrainingGraphService(_SessionDouble())
    service.configure(model_folder=tmp_path, model_name="missing")

    with pytest.raises(TrainingGraphError, match="does not exist"):
        service.load_configured_source(is_training=True)
