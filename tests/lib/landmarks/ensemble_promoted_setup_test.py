#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.ensemble.promoted_setup` (issue #71)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.landmarks.ensemble.promoted_setup import (
    ARTIFACT_SCHEMA_VERSION,
    SETUP_FILENAME,
    WEIGHTS_FILENAME,
    PromotedSetupError,
    ensure_compatible_adapters,
    load_promoted_setup,
    strategy_supported_by_runtime,
    validate_setup_payload,
    validate_weights_payload,
    write_best_setup,
    write_best_weights,
)
from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from lib.landmarks.schema import CANONICAL_SCHEMA


def _equal_weights() -> dict[str, list[float]]:
    return {
        "hrnet": [1.0 / 3] * LANDMARK_COUNT,
        "spiga": [1.0 / 3] * LANDMARK_COUNT,
        "orformer": [1.0 / 3] * LANDMARK_COUNT,
    }


def _write_valid_pair(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    """Write a valid best_setup.json + best_weights.json pair for the tests."""
    weights_path = tmp_path / WEIGHTS_FILENAME
    setup_path = tmp_path / SETUP_FILENAME
    write_best_weights(weights_path, _equal_weights(), models=("hrnet", "spiga", "orformer"))
    write_best_setup(
        setup_path,
        candidate_id="sha256:0123abc",
        models=overrides.get("models", ("hrnet", "spiga", "orformer")),
        strategy=overrides.get("strategy", "static_weighted"),
        outlier_threshold=overrides.get("outlier_threshold"),
        weight_generator_name=overrides.get("weight_generator_name", "inverse_mean_error"),
        weight_generator_params=overrides.get("weight_generator_params", {"epsilon": 1e-6}),
        crop_scale=overrides.get("crop_scale", 1.6),
        bbox_source=overrides.get("bbox_source", "manifest"),
        regression_epsilon_nme=overrides.get("regression_epsilon_nme", 0.001),
        reproducibility=overrides.get(
            "reproducibility",
            {
                "split_assignment_hash": "sha256:abc",
                "candidate_search_seed": 1337,
                "objective": "extract_alignment_v1",
            },
        ),
        fit=overrides.get("fit", {"sample_count": 12, "scenario_buckets": ["fixture:clean"]}),
        selection_metrics=overrides.get("selection_metrics", {"sample_count": 4}),
        report_metrics=overrides.get("report_metrics", {"sample_count": 4}),
        evaluation_log_path=overrides.get("evaluation_log_path", "candidate_results.json"),
        weights_path=WEIGHTS_FILENAME,
    )
    return setup_path, weights_path


def test_load_valid_promoted_setup_attaches_weights(tmp_path: Path) -> None:
    """A well-formed setup round-trips through the loader with weights attached."""
    setup_path, weights_path = _write_valid_pair(tmp_path)

    setup = load_promoted_setup(setup_path)

    assert setup.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    assert setup.schema == CANONICAL_SCHEMA
    assert setup.strategy == "static_weighted"
    assert setup.outlier_threshold is None
    assert setup.models == ("hrnet", "spiga", "orformer")
    assert set(setup.weights) == set(setup.models)
    assert all(len(setup.weights[model]) == LANDMARK_COUNT for model in setup.models)
    assert setup.weights_path == WEIGHTS_FILENAME
    assert setup.weights_path
    assert weights_path.is_file()


def test_load_resolves_relative_weights_path(tmp_path: Path) -> None:
    """Relative weights paths resolve against the setup file's directory."""
    nested_dir = tmp_path / "promotion"
    nested_dir.mkdir()
    setup_path, _ = _write_valid_pair(nested_dir)

    setup = load_promoted_setup(setup_path)

    assert setup.weights  # weights dict populated
    assert set(setup.weights) == {"hrnet", "spiga", "orformer"}


def test_load_skipping_weights_returns_empty_weights_dict(tmp_path: Path) -> None:
    """``load_weights=False`` validates but does not read the weights file."""
    setup_path, weights_path = _write_valid_pair(tmp_path)
    weights_path.unlink()

    setup = load_promoted_setup(setup_path, load_weights=False)

    assert not setup.has_weights()


def test_load_rejects_unsupported_artifact_schema_version_2(tmp_path: Path) -> None:
    """artifact_schema_version=2 in a v1 loader hard-fails clearly."""
    setup_path, _ = _write_valid_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["artifact_schema_version"] = 2
    setup_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="artifact_schema_version"):
        load_promoted_setup(setup_path)


def test_load_rejects_unsupported_schema(tmp_path: Path) -> None:
    """Non-2d_68 schemas are rejected at load time."""
    setup_path, _ = _write_valid_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["schema"] = "2d_98"
    setup_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="schema"):
        load_promoted_setup(setup_path)


def test_load_rejects_forbidden_outlier_method_field(tmp_path: Path) -> None:
    """A setup carrying a stray ``outlier_method`` field hard-fails."""
    setup_path, _ = _write_valid_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["outlier_method"] = "downweight"
    setup_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="outlier_method"):
        load_promoted_setup(setup_path)


def test_load_rejects_non_canonical_strategy(tmp_path: Path) -> None:
    """Promoted artifacts must serialize canonical strategy names, not aliases."""
    setup_path, _ = _write_valid_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["strategy"] = "static_weighted_outliers"  # legacy alias
    setup_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="canonical"):
        load_promoted_setup(setup_path)


def test_load_rejects_missing_required_keys(tmp_path: Path) -> None:
    """Setup files missing required keys must fail with the missing list."""
    setup_path, _ = _write_valid_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    del payload["weights_path"]
    setup_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="weights_path"):
        load_promoted_setup(setup_path)


def test_load_rejects_threshold_for_non_threshold_strategy(tmp_path: Path) -> None:
    """``static_weighted`` with a numeric threshold violates strategy scoping."""
    setup_path, _ = _write_valid_pair(tmp_path)
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["outlier_threshold"] = 3.5
    setup_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="outlier_threshold"):
        load_promoted_setup(setup_path)


def test_load_rejects_missing_threshold_for_threshold_strategy(tmp_path: Path) -> None:
    """``static_weighted_hard_drop`` requires a positive threshold."""
    setup_path, _ = _write_valid_pair(
        tmp_path, strategy="static_weighted_hard_drop", outlier_threshold=3.5
    )
    payload = json.loads(setup_path.read_text(encoding="utf-8"))
    payload["outlier_threshold"] = None
    setup_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="outlier_threshold"):
        load_promoted_setup(setup_path)


def test_write_best_setup_rejects_forbidden_threshold(tmp_path: Path) -> None:
    """The writer hard-fails on a strategy + threshold combination it cannot serialize."""
    with pytest.raises(ValueError):
        write_best_setup(
            tmp_path / SETUP_FILENAME,
            candidate_id="sha256:abc",
            models=("hrnet", "spiga", "orformer"),
            strategy="static_weighted",
            outlier_threshold=3.5,  # invalid for static_weighted
            weight_generator_name="inverse_mean_error",
            weight_generator_params={},
            crop_scale=1.6,
            bbox_source="manifest",
            regression_epsilon_nme=0.001,
            reproducibility={},
            fit={},
            selection_metrics={},
        )


def test_validate_weights_payload_rejects_short_column() -> None:
    """A weight column with the wrong length triggers a clear error."""
    payload = {
        "artifact_schema_version": 1,
        "schema": "2d_68",
        "models": ["hrnet"],
        "weights": {"hrnet": [0.5] * 67},
    }
    with pytest.raises(PromotedSetupError, match="68"):
        validate_weights_payload(payload)


def test_validate_weights_payload_rejects_negative_values() -> None:
    """Non-negative constraint applies per cell."""
    column = [0.0] * LANDMARK_COUNT
    column[0] = -0.1
    payload = {
        "artifact_schema_version": 1,
        "schema": "2d_68",
        "models": ["hrnet"],
        "weights": {"hrnet": column},
    }
    with pytest.raises(PromotedSetupError, match="non-negative"):
        validate_weights_payload(payload)


def test_validate_weights_payload_rejects_model_mismatch(tmp_path: Path) -> None:
    """Setup and weights disagree on model order → loader hard-fails."""
    setup_path, weights_path = _write_valid_pair(tmp_path)
    weights_payload = json.loads(weights_path.read_text(encoding="utf-8"))
    weights_payload["models"] = ["spiga", "hrnet", "orformer"]
    weights_payload["weights"] = {
        model: weights_payload["weights"][model] for model in weights_payload["models"]
    }
    weights_path.write_text(json.dumps(weights_payload), encoding="utf-8")

    with pytest.raises(PromotedSetupError, match="do not match"):
        load_promoted_setup(setup_path)


def test_validate_setup_payload_rejects_duplicate_models() -> None:
    """Models list must contain unique entries."""
    payload = {
        "artifact_schema_version": 1,
        "schema": "2d_68",
        "candidate_id": "sha256:abc",
        "models": ["hrnet", "hrnet"],
        "strategy": "static_weighted",
        "outlier_threshold": None,
        "weights_path": "best_weights.json",
        "weight_generator": {"name": "inverse_mean_error"},
        "reproducibility": {},
        "fit": {},
        "selection_metrics": {},
        "report_metrics": {},
    }
    with pytest.raises(PromotedSetupError, match="duplicates"):
        validate_setup_payload(payload)


def test_ensure_compatible_adapters_passes_for_loaded_models(tmp_path: Path) -> None:
    """Adapter compatibility check succeeds when every model is present."""
    setup_path, _ = _write_valid_pair(tmp_path)
    setup = load_promoted_setup(setup_path)

    ensure_compatible_adapters(setup, ["hrnet", "spiga", "orformer"])


def test_ensure_compatible_adapters_fails_for_missing_model(tmp_path: Path) -> None:
    """Adapter compatibility check fails when a configured model is absent."""
    setup_path, _ = _write_valid_pair(tmp_path)
    setup = load_promoted_setup(setup_path)

    with pytest.raises(PromotedSetupError, match="not present"):
        ensure_compatible_adapters(setup, ["hrnet", "spiga"])


def test_strategy_supported_by_runtime_raises_on_unsupported() -> None:
    """The runtime helper rejects strategies it cannot dispatch."""
    with pytest.raises(PromotedSetupError, match="strategy"):
        strategy_supported_by_runtime("mythical_strategy", ["plain_average", "static_weighted"])
    # Inverse case: present in the list → no raise.
    strategy_supported_by_runtime("plain_average", ["plain_average", "static_weighted"])


def test_load_rejects_missing_weights_file(tmp_path: Path) -> None:
    """Missing weights file produces a clear PromotedSetupError."""
    setup_path, weights_path = _write_valid_pair(tmp_path)
    weights_path.unlink()

    with pytest.raises(PromotedSetupError, match="weights file"):
        load_promoted_setup(setup_path)


def test_load_rejects_invalid_json_payload(tmp_path: Path) -> None:
    """Corrupt setup JSON surfaces a clear error mentioning the file."""
    setup_path = tmp_path / SETUP_FILENAME
    setup_path.write_text("{not: valid json", encoding="utf-8")
    with pytest.raises(PromotedSetupError, match="valid JSON"):
        load_promoted_setup(setup_path)
