#!/usr/bin/env python3
"""Preview diagnostics GUI service tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from lib.gui.services.preview_diagnostics_service import (
    PreviewDiagnosticsError,
    PreviewDiagnosticsService,
)


def _state_file(tmp_path: Path, name: str = "model") -> Path:
    """Create a model state file."""
    state_file = tmp_path / f"{name}_state.json"
    state_file.write_text("{}", encoding="utf-8")
    return state_file


def _write_diagnostics(tmp_path: Path, model_name: str = "model", session_id: int = 1) -> None:
    """Write preview diagnostics TensorBoard scalar events."""
    writer = SummaryWriter(tmp_path / f"{model_name}_logs" / f"session_{session_id}" / "train")
    writer.add_scalar("batch_preview_diagnostics/reconstruction_mae_A", 0.1, 10)
    writer.add_scalar("batch_preview_diagnostics/reconstruction_mae_A_count", 14.0, 10)
    writer.add_scalar("batch_preview_diagnostics/reconstruction_mae_B", 0.2, 10)
    writer.add_scalar("batch_preview_diagnostics/reconstruction_imbalance_mae", 0.1, 10)
    writer.add_scalar("batch_preview_diagnostics/boundary_mae_A", 0.05, 10)
    writer.add_scalar("batch_preview_diagnostics/reconstruction_mae_A", 0.08, 20)
    writer.add_scalar("batch_preview_diagnostics/reconstruction_mae_A_count", 28.0, 20)
    writer.add_scalar("batch_preview_diagnostics/reconstruction_mae_B", 0.16, 20)
    writer.flush()
    writer.close()


def test_preview_diagnostics_service_reads_latest_tensorboard_metrics(tmp_path: Path) -> None:
    """Service should return latest preview diagnostics scalars from TensorBoard."""
    _state_file(tmp_path)
    _write_diagnostics(tmp_path)
    service = PreviewDiagnosticsService()

    snapshot = service.load_source(tmp_path)

    assert snapshot.session_id == 1
    assert snapshot.iteration == 20
    assert snapshot.metric("reconstruction_mae_A").value == pytest.approx(0.08)  # type: ignore[union-attr]
    assert snapshot.metric("reconstruction_mae_A").count == pytest.approx(28.0)  # type: ignore[union-attr]
    assert snapshot.metric("reconstruction_mae_B").value == pytest.approx(0.16)  # type: ignore[union-attr]
    assert snapshot.metric("boundary_mae_A") is None
    assert "reconstruction mae A: 0.0800" in service.compact_text(snapshot)
    assert "n=28" in service.compact_text(snapshot)

    series = {item.name: item for item in service.get_series(session_id=1)}
    assert series["reconstruction_mae_A"].iterations == (10, 20)
    assert series["reconstruction_mae_A"].values == pytest.approx((0.1, 0.08))
    assert series["boundary_mae_A"].iterations == (10,)
    assert series["boundary_mae_A"].values == pytest.approx((0.05,))


def test_preview_diagnostics_service_empty_when_no_metrics(tmp_path: Path) -> None:
    """Missing diagnostics logs should produce an empty snapshot, not an error."""
    _state_file(tmp_path)
    service = PreviewDiagnosticsService()

    snapshot = service.load_source(tmp_path)

    assert snapshot.is_empty is True
    assert service.compact_text(snapshot) == "Preview diagnostics: no metrics logged"


def test_preview_diagnostics_service_configured_source_can_refresh_later(
    tmp_path: Path,
) -> None:
    """Configured model contexts should refresh once logs appear."""
    service = PreviewDiagnosticsService()
    service.configure(model_folder=tmp_path, model_name="model")

    assert service.refresh().is_empty is True

    _write_diagnostics(tmp_path)
    snapshot = service.refresh()

    assert snapshot.is_empty is False
    assert snapshot.metric("reconstruction_mae_A").value == pytest.approx(0.08)  # type: ignore[union-attr]


def test_preview_diagnostics_service_rejects_invalid_source(tmp_path: Path) -> None:
    """Invalid diagnostics sources should raise a clear service error."""
    service = PreviewDiagnosticsService()

    with pytest.raises(PreviewDiagnosticsError, match="does not exist"):
        service.load_source(tmp_path / "missing_state.json")
