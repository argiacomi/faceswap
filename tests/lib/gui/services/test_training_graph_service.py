#!/usr/bin/env python3
"""Tests for training graph data service."""

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
        self.loss_by_session: dict[int | None, dict[str, object]] = {
            None: {"loss_a": [3.0, 2.0, 1.0], "loss_b": [4.0, 3.0]},
            1: {"loss_a": [3.0, 2.0, 1.0]},
            2: {"loss_b": [4.0, 3.0]},
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
        """Return loss values for session id."""
        return self.loss_by_session.get(session_id, {})

    def get_loss_keys(self, session_id: int | None) -> list[str]:
        """Return loss keys for session id."""
        return sorted(self.get_loss(session_id))


def _state_file(tmp_path: Path, name: str = "model") -> Path:
    """Create a model state file."""
    state_file = tmp_path / f"{name}_state.json"
    state_file.write_text("{}", encoding="utf-8")
    return state_file


def test_training_graph_service_configures_pending_source(tmp_path: Path) -> None:
    """Graph source can be configured before a state file exists."""
    service = TrainingGraphService(_SessionDouble())

    source = service.configure(model_folder=tmp_path, model_name="model")

    assert source is not None
    assert source.model_dir == tmp_path
    assert source.model_name == "model"
    assert service.snapshot.source == source
    assert service.snapshot.is_empty is True


def test_training_graph_service_loads_state_file_and_refreshes(tmp_path: Path) -> None:
    """Loading a state file should initialize Analysis and create graph series."""
    session = _SessionDouble()
    service = TrainingGraphService(session)
    state_file = _state_file(tmp_path, "my_model")

    snapshot = service.load_source(state_file)

    assert session.initialized == [(str(tmp_path), "my_model", False)]
    assert service.source is not None
    assert service.source.model_name == "my_model"
    assert service.loss_keys == ("loss_a", "loss_b")
    assert [series.name for series in snapshot.series] == ["loss_a", "loss_b"]
    assert snapshot.point_count == 3
    assert snapshot.series[0].values == (3.0, 2.0, 1.0)


def test_training_graph_service_loads_from_folder(tmp_path: Path) -> None:
    """Loading a folder should resolve the first state file."""
    session = _SessionDouble()
    service = TrainingGraphService(session)
    _state_file(tmp_path, "z_model")
    _state_file(tmp_path, "a_model")

    service.load_source(tmp_path)

    assert session.initialized[0] == (str(tmp_path), "a_model", False)


def test_training_graph_service_loads_configured_source_when_file_exists(tmp_path: Path) -> None:
    """Configured train context should load later when state file exists."""
    session = _SessionDouble()
    service = TrainingGraphService(session)
    service.configure(model_folder=tmp_path, model_name="model")
    _state_file(tmp_path, "model")

    snapshot = service.load_configured_source(is_training=True)

    assert session.initialized == [(str(tmp_path), "model", True)]
    assert snapshot.is_empty is False


def test_training_graph_service_reports_missing_configured_source(tmp_path: Path) -> None:
    """Configured source load should fail clearly when state file is not present."""
    service = TrainingGraphService(_SessionDouble())
    service.configure(model_folder=tmp_path, model_name="missing")

    with pytest.raises(TrainingGraphError, match="state file does not exist"):
        service.load_configured_source()


def test_training_graph_service_selects_session_id() -> None:
    """Changing selected session should refresh the snapshot for that session."""
    session = _SessionDouble()
    session.is_loaded = True
    service = TrainingGraphService(session)

    snapshot = service.set_session_id(1)

    assert service.session_id == 1
    assert service.session_ids == (1, 2)
    assert [series.name for series in snapshot.series] == ["loss_a"]
    assert snapshot.point_count == 3


def test_training_graph_service_filters_invalid_values() -> None:
    """Graph values should skip bools, non-numeric values and NaNs."""
    session = _SessionDouble()
    session.is_loaded = True
    session.loss_by_session[None] = {"loss_a": [1, True, "bad", float("nan"), 2]}
    service = TrainingGraphService(session)

    snapshot = service.refresh()

    assert snapshot.series[0].values == (1.0, 2.0)


def test_training_graph_service_clear_resets_state(tmp_path: Path) -> None:
    """Clear should reset source, selection, keys and snapshot."""
    service = TrainingGraphService(_SessionDouble())
    service.configure(model_folder=tmp_path, model_name="model")

    service.clear()

    assert service.source is None
    assert service.session_id is None
    assert service.loss_keys == ()
    assert service.snapshot.is_empty is True
