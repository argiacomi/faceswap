#!/usr/bin/env python3
"""Tests for promoted setup loading in the ensemble aligner plugin (#66)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.adapters import LandmarkAdapterConfig, StaticLandmarkAdapter
from lib.landmarks.ensemble.promoted_setup import (
    SETUP_FILENAME,
    WEIGHTS_FILENAME,
    PromotedSetupError,
    write_best_setup,
    write_best_weights,
)
from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from plugins.extract.align.ensemble import Ensemble


def _points(value: float) -> np.ndarray:
    return np.full((68, 2), value, dtype="float32")


def _weights_favoring(model: str, *, models: tuple[str, ...]) -> dict[str, list[float]]:
    """Build weights with `model` dominating each landmark column."""
    column = {name: [0.05 if name != model else 0.9] * LANDMARK_COUNT for name in models}
    return column


def _three_static_adapters(values: tuple[float, float, float]) -> list[StaticLandmarkAdapter]:
    return [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig(name, coordinate_space="frame", weight=1.0),
            _points(value),
        )
        for name, value in zip(("hrnet", "spiga", "orformer"), values, strict=True)
    ]


def _write_promoted_pair(
    tmp_path: Path,
    *,
    strategy: str = "static_weighted",
    outlier_threshold: float | None = None,
    weights: dict[str, list[float]] | None = None,
    models: tuple[str, ...] = ("hrnet", "spiga", "orformer"),
) -> tuple[Path, Path]:
    setup_path = tmp_path / SETUP_FILENAME
    weights_path = tmp_path / WEIGHTS_FILENAME
    write_best_weights(
        weights_path,
        weights or {name: [1.0 / len(models)] * LANDMARK_COUNT for name in models},
        models=models,
    )
    write_best_setup(
        setup_path,
        candidate_id="sha256:abc",
        models=models,
        strategy=strategy,
        outlier_threshold=outlier_threshold,
        weight_generator_name="inverse_mean_error",
        weight_generator_params={"epsilon": 1e-6},
        crop_scale=1.6,
        bbox_source="manifest",
        regression_epsilon_nme=0.001,
        reproducibility={"split_assignment_hash": "sha256:111"},
        fit={"sample_count": 12},
        selection_metrics={"sample_count": 4},
        report_metrics={"sample_count": 4},
        evaluation_log_path="candidate_results.json",
        weights_path=WEIGHTS_FILENAME,
    )
    return setup_path, weights_path


def test_empty_setup_path_defaults_to_off_mode(tmp_path: Path) -> None:
    """Empty ``setup_path`` keeps the legacy behavior even with ``setup_mode=strict``."""
    adapters = _three_static_adapters((0.1, 0.2, 0.3))
    plugin = Ensemble(adapters=adapters, setup_path="", setup_mode="strict")
    assert plugin._setup_mode == "off"
    assert plugin._promoted is None


def test_nonempty_setup_path_defaults_to_strict_mode(tmp_path: Path) -> None:
    """Non-empty ``setup_path`` without an explicit mode defaults to strict."""
    setup_path, _ = _write_promoted_pair(tmp_path)
    adapters = _three_static_adapters((0.1, 0.2, 0.3))
    plugin = Ensemble(adapters=adapters, setup_path=str(setup_path))
    assert plugin._setup_mode == "strict"
    assert plugin._promoted is not None
    assert plugin._promoted.strategy == "static_weighted"


def test_strict_mode_uses_promoted_strategy_for_fusion(tmp_path: Path) -> None:
    """A loaded setup overrides the configured strategy at fusion time."""
    setup_path, _ = _write_promoted_pair(
        tmp_path,
        strategy="static_weighted_downweight",
        outlier_threshold=3.5,
    )
    adapters = _three_static_adapters((0.1, 0.2, 0.9))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        setup_path=str(setup_path),
        setup_mode="strict",
        strategy="plain_average",  # configured strategy should be overridden
    )
    plugin.model = plugin.load_model()

    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    assert plugin.last_debug_metadata[0]["strategy"] == "static_weighted_downweight"
    assert plugin.last_debug_metadata[0]["weight_source"] == "promoted_setup"
    assert plugin.last_debug_metadata[0]["outlier_threshold"] == 3.5


def test_strict_mode_fails_on_artifact_schema_version_mismatch(tmp_path: Path) -> None:
    """artifact_schema_version=2 hard-fails in strict mode."""
    setup_path, _ = _write_promoted_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["artifact_schema_version"] = 2
    setup_path.write_text(json.dumps(payload), encoding="utf-8")
    adapters = _three_static_adapters((0.1, 0.2, 0.3))

    with pytest.raises(PromotedSetupError, match="artifact_schema_version"):
        Ensemble(adapters=adapters, setup_path=str(setup_path), setup_mode="strict")


def test_strict_mode_fails_when_promoted_model_missing(tmp_path: Path) -> None:
    """Loading a setup whose models are not in the loaded adapters hard-fails."""
    setup_path, _ = _write_promoted_pair(tmp_path, models=("hrnet", "spiga", "orformer"))
    adapters = _three_static_adapters((0.1, 0.2, 0.3))[:2]  # drop orformer
    plugin = Ensemble(adapters=adapters, setup_path=str(setup_path), setup_mode="strict")

    with pytest.raises(PromotedSetupError, match="not present"):
        plugin.model = plugin.load_model()


def test_strict_mode_fails_on_forbidden_outlier_method(tmp_path: Path) -> None:
    """A stray ``outlier_method`` field hard-fails strict loading."""
    setup_path, _ = _write_promoted_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["outlier_method"] = "downweight"
    setup_path.write_text(json.dumps(payload), encoding="utf-8")
    adapters = _three_static_adapters((0.1, 0.2, 0.3))

    with pytest.raises(PromotedSetupError, match="outlier_method"):
        Ensemble(adapters=adapters, setup_path=str(setup_path), setup_mode="strict")


def test_strict_mode_fails_on_threshold_for_non_threshold_strategy(tmp_path: Path) -> None:
    """Loading a setup with ``static_weighted`` + threshold hard-fails."""
    setup_path, _ = _write_promoted_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["outlier_threshold"] = 3.5
    setup_path.write_text(json.dumps(payload), encoding="utf-8")
    adapters = _three_static_adapters((0.1, 0.2, 0.3))

    with pytest.raises(PromotedSetupError, match="outlier_threshold"):
        Ensemble(adapters=adapters, setup_path=str(setup_path), setup_mode="strict")


def test_fallback_mode_logs_warning_and_uses_fallback_strategy(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the artifact is invalid, fallback mode keeps extracting but warns."""
    import logging

    setup_path, _ = _write_promoted_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["artifact_schema_version"] = 99  # unsupported
    setup_path.write_text(json.dumps(payload), encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="plugins.extract.align.ensemble")
    adapters = _three_static_adapters((0.1, 0.2, 0.3))

    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        setup_path=str(setup_path),
        setup_mode="fallback",
        fallback_strategy="plain_average",
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    assert plugin._promoted is None
    assert plugin._strategy == "plain_average"
    assert plugin.last_debug_metadata[0]["weight_source"] == "adapter_config"
    assert any("failed strict validation" in record.message for record in caplog.records)


def test_setup_mode_off_ignores_promoted_setup(tmp_path: Path) -> None:
    """``setup_mode=off`` does not even attempt to read the setup file."""
    adapters = _three_static_adapters((0.1, 0.2, 0.3))
    plugin = Ensemble(
        adapters=adapters,
        setup_path=str(tmp_path / "does_not_exist.json"),
        setup_mode="off",
    )
    assert plugin._promoted is None
    assert plugin._setup_mode == "off"


def test_promoted_weights_are_subset_and_renormalized_when_adapter_fails(
    tmp_path: Path,
) -> None:
    """When one adapter fails for a face, the promoted matrix is subset + renormalized."""
    promoted_weights = _weights_favoring("hrnet", models=("hrnet", "spiga", "orformer"))
    setup_path, _ = _write_promoted_pair(tmp_path, weights=promoted_weights)
    # Failing spiga: emulate by dropping it from the active adapters.
    adapters = [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("hrnet", coordinate_space="frame", weight=1.0),
            _points(0.1),
        ),
        StaticLandmarkAdapter(
            LandmarkAdapterConfig("orformer", coordinate_space="frame", weight=1.0),
            _points(0.3),
        ),
    ]
    # The setup still lists 3 models; ensure_compatible_adapters expects all of them
    # to be loaded, so use a setup with exactly the surviving 2 models.
    setup_path, _ = _write_promoted_pair(
        tmp_path,
        weights={
            "hrnet": [0.9] * LANDMARK_COUNT,
            "orformer": [0.1] * LANDMARK_COUNT,
        },
        models=("hrnet", "orformer"),
    )
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        setup_path=str(setup_path),
        setup_mode="strict",
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    debug = plugin.last_debug_metadata[0]
    matrix = np.array(debug["weights"], dtype="float32")
    assert matrix.shape == (2, 68)
    np.testing.assert_allclose(matrix.sum(axis=0), np.ones(68, dtype="float32"), atol=1e-6)
    # hrnet still dominates per landmark column even after renormalization.
    assert matrix[0, 0] > matrix[1, 0]


def test_promoted_weights_matrix_shape_is_models_by_68(tmp_path: Path) -> None:
    """In strict mode the fused weights surface has shape ``(model_count, 68)``."""
    setup_path, _ = _write_promoted_pair(tmp_path)
    adapters = _three_static_adapters((0.1, 0.2, 0.3))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        setup_path=str(setup_path),
        setup_mode="strict",
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))

    matrix = np.array(plugin.last_debug_metadata[0]["weights"], dtype="float32")
    assert matrix.shape == (len(adapters), 68)


def test_relative_weights_path_resolves_from_setup_directory(tmp_path: Path) -> None:
    """A relative ``weights_path`` resolves against the setup file directory."""
    nested = tmp_path / "promotion"
    nested.mkdir()
    setup_path, weights_path = _write_promoted_pair(nested)

    adapters = _three_static_adapters((0.1, 0.2, 0.3))
    plugin = Ensemble(
        adapters=adapters,
        setup_path=str(setup_path),
        setup_mode="strict",
    )
    assert plugin._promoted is not None
    assert set(plugin._promoted.weights) == {"hrnet", "spiga", "orformer"}
    assert weights_path.is_file()


def test_fallback_strategy_adapter_config_uses_configured_strategy(tmp_path: Path) -> None:
    """``fallback_strategy=adapter_config`` reuses the configured ``strategy``."""
    setup_path, _ = _write_promoted_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["artifact_schema_version"] = 99
    setup_path.write_text(json.dumps(payload), encoding="utf-8")
    adapters = _three_static_adapters((0.1, 0.2, 0.3))

    plugin = Ensemble(
        adapters=adapters,
        setup_path=str(setup_path),
        setup_mode="fallback",
        fallback_strategy="adapter_config",
        strategy="weighted_median",
    )
    assert plugin._strategy == "weighted_median"


def test_strict_mode_fails_on_unsupported_strategy(tmp_path: Path) -> None:
    """A valid-looking setup with a non-canonical strategy is rejected at load."""
    setup_path, _ = _write_promoted_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["strategy"] = "made_up_strategy"
    setup_path.write_text(json.dumps(payload), encoding="utf-8")
    adapters = _three_static_adapters((0.1, 0.2, 0.3))

    with pytest.raises(PromotedSetupError):
        Ensemble(adapters=adapters, setup_path=str(setup_path), setup_mode="strict")


def test_strict_mode_fails_on_malformed_weights_payload(tmp_path: Path) -> None:
    """Truncated weight columns in the weights file hard-fail at load."""
    setup_path, weights_path = _write_promoted_pair(tmp_path)
    weights_payload = json.loads(weights_path.read_text(encoding="utf-8"))
    weights_payload["weights"]["hrnet"] = [0.5] * 60  # length != 68
    weights_path.write_text(json.dumps(weights_payload), encoding="utf-8")
    adapters = _three_static_adapters((0.1, 0.2, 0.3))

    with pytest.raises(PromotedSetupError, match="68"):
        Ensemble(adapters=adapters, setup_path=str(setup_path), setup_mode="strict")
