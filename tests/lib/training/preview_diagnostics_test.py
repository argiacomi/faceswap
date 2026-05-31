#!/usr/bin/env python3
"""Unit tests for :mod:`lib.training.preview_diagnostics`."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lib.training.preview_diagnostics import PreviewDiagnostics
from plugins.train.trainer import trainer_config as cfg


def _get_preview_arrays(
    errors: tuple[float, float] = (0.1, 0.2),
    include_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic preview predictions and targets."""
    samples = 3
    size = 8
    channels = 4 if include_mask else 3
    targets: np.ndarray = np.zeros((2, samples, size, size, channels), dtype=np.float32)
    targets[0, ..., :3] = 0.2
    targets[1, ..., :3] = 0.4
    if include_mask:
        targets[..., 3] = 0.0
        targets[:, :, 2:6, 2:6, 3] = 1.0

    predictions: np.ndarray = np.zeros((2, 2, samples, size, size, 3), dtype=np.float32)
    predictions[0, 0, ..., :3] = targets[0, ..., :3] + errors[0]
    predictions[1, 1, ..., :3] = targets[1, ..., :3] + errors[1]
    return predictions, targets


def test_preview_diagnostics_updates_expected_metric_payload() -> None:
    """Preview diagnostics returns rolling reconstruction and mask metrics."""
    predictions, targets = _get_preview_arrays()
    diagnostics = PreviewDiagnostics(ema_alpha=0.5)

    logs = diagnostics.update(predictions, targets, iteration=10)

    assert logs["reconstruction_mae_A"] == pytest.approx(0.1)
    assert logs["reconstruction_mae_B"] == pytest.approx(0.2)
    assert logs["reconstruction_mae_A_count"] == 3.0
    assert logs["reconstruction_mae_count"] == 6.0
    assert logs["reconstruction_mse_A"] == pytest.approx(0.01)
    assert logs["reconstruction_mse_B"] == pytest.approx(0.04)
    assert logs["reconstruction_imbalance_mae"] == pytest.approx(0.1)
    assert logs["masked_reconstruction_mae_A"] == pytest.approx(0.1)
    assert logs["masked_reconstruction_mae_B"] == pytest.approx(0.2)
    assert logs["boundary_mae_A"] == pytest.approx(0.1)
    assert logs["boundary_mae_B"] == pytest.approx(0.2)
    assert "detail_mae_A" in logs
    assert "prediction_detail_A" in logs
    assert "target_detail_A" in logs


def test_preview_diagnostics_rolls_ema_and_variance() -> None:
    """Preview diagnostics rolls metrics across refreshes."""
    diagnostics = PreviewDiagnostics(ema_alpha=0.5)
    predictions, targets = _get_preview_arrays(errors=(0.1, 0.2))
    diagnostics.update(predictions, targets, iteration=1)

    predictions, targets = _get_preview_arrays(errors=(0.3, 0.4))
    logs = diagnostics.update(predictions, targets, iteration=2)

    assert logs["reconstruction_mae_A"] == pytest.approx(0.2)
    assert logs["reconstruction_mae_B"] == pytest.approx(0.3)
    assert logs["reconstruction_mae_A_mean"] == pytest.approx(0.2)
    assert logs["reconstruction_mae_A_std"] > 0.0
    assert logs["reconstruction_mae_A_count"] == 6.0


def test_preview_diagnostics_handles_missing_masks() -> None:
    """Mask-specific diagnostics are skipped when preview targets have no mask channel."""
    predictions, targets = _get_preview_arrays(include_mask=False)
    diagnostics = PreviewDiagnostics()

    logs = diagnostics.update(predictions, targets, iteration=1)

    assert logs["reconstruction_mae_A"] == pytest.approx(0.1)
    assert not any(key.startswith("masked_") for key in logs)
    assert not any(key.startswith("boundary_") for key in logs)


def test_preview_diagnostics_writes_jsonl(tmp_path) -> None:
    """Preview diagnostics can write structured JSONL artifacts."""
    predictions, targets = _get_preview_arrays()
    jsonl_path = tmp_path / "preview_diagnostics.jsonl"
    diagnostics = PreviewDiagnostics(ema_alpha=0.5, jsonl_path=str(jsonl_path))

    diagnostics.update(predictions, targets, iteration=42)

    payload = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert payload["iteration"] == 42
    assert payload["metrics"]["reconstruction_mae_A"]["ema"] == pytest.approx(0.1)
    assert payload["metrics"]["reconstruction_mae_A"]["count"] == 3


def test_preview_diagnostics_rejects_invalid_alpha() -> None:
    """Preview diagnostics validates EMA alpha."""
    with pytest.raises(ValueError, match="EMA alpha"):
        PreviewDiagnostics(ema_alpha=0.0)


def test_preview_diagnostics_config_defaults_disabled() -> None:
    """Preview diagnostics is opt-in by default."""
    assert cfg.Augmentation.preview_diagnostics.default is False
    assert cfg.Augmentation.preview_diagnostics_jsonl.default is False
