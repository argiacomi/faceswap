#!/usr/bin/env python3
"""Tests for Phase 5 #9 region-level landmark fusion.

Covers the ``region_weighted`` strategy registration, the region weight
representation + converter, the backward-compatible ``region_weights`` block in
the v2 weights artifact, and the runtime region-fusion candidate appension.
"""

from __future__ import annotations

import json
import typing as T
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.promoted_setup import (
    SETUP_FILENAME,
    WEIGHTS_FILENAME,
    WEIGHTS_SCHEMA_VERSION_WITH_BUCKETS,
    PromotedSetupError,
    load_promoted_setup,
    validate_weights_payload,
    write_best_setup,
    write_best_weights,
)
from lib.landmarks.ensemble.runtime_resolver import (
    CandidateMetrics,
    CandidateRecord,
    RuntimeResolverConfig,
    append_region_weight_candidate,
)
from lib.landmarks.ensemble.strategies import (
    CANONICAL_STRATEGIES,
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
    validate_threshold,
)
from lib.landmarks.ensemble.weights import (
    FUSION_REGION_INDICES,
    LANDMARK_COUNT,
    normalize_region_weights,
    region_weights_to_matrix,
)
from lib.landmarks.evaluation.profile_metrics import REGION_INDICES

MODELS = ("hrnet", "spiga", "orformer")


# --------------------------------------------------------------------------- #
# Strategy registration
# --------------------------------------------------------------------------- #
def test_region_weighted_is_a_canonical_non_threshold_strategy() -> None:
    assert "region_weighted" in CANONICAL_STRATEGIES
    assert canonical_strategy("region_weighted") == "region_weighted"
    assert strategy_outlier_method("region_weighted") == "none"
    assert strategy_requires_weights("region_weighted") is True
    assert strategy_uses_threshold("region_weighted") is False


def test_region_weighted_rejects_threshold() -> None:
    validate_threshold("region_weighted", None)
    with pytest.raises(ValueError, match="does not accept outlier_threshold"):
        validate_threshold("region_weighted", 3.5)


# --------------------------------------------------------------------------- #
# Region partition + converter
# --------------------------------------------------------------------------- #
def test_fusion_regions_match_profile_metrics_and_partition_68() -> None:
    """The fusion partition mirrors profile_metrics and covers all 68 once."""
    assert {key: tuple(value) for key, value in REGION_INDICES.items()} == FUSION_REGION_INDICES
    covered = sorted(index for indices in FUSION_REGION_INDICES.values() for index in indices)
    assert covered == list(range(LANDMARK_COUNT))


def test_normalize_region_weights_normalizes_and_rejects_unknown_region() -> None:
    table = {"nose": {"hrnet": 3.0, "spiga": 1.0}}
    normalized = normalize_region_weights(table)
    assert normalized["nose"]["hrnet"] == pytest.approx(0.75)
    assert normalized["nose"]["spiga"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="unknown fusion region"):
        normalize_region_weights({"not_a_region": {"hrnet": 1.0}})


def test_region_weights_to_matrix_broadcasts_per_region() -> None:
    table = {region: {"hrnet": 3.0, "spiga": 1.0} for region in FUSION_REGION_INDICES}
    matrix = region_weights_to_matrix(table, ("hrnet", "spiga"))
    assert matrix.shape == (2, LANDMARK_COUNT)
    np.testing.assert_allclose(matrix[0], 0.75, atol=1e-6)
    np.testing.assert_allclose(matrix[1], 0.25, atol=1e-6)


# --------------------------------------------------------------------------- #
# Fit tooling
# --------------------------------------------------------------------------- #
def test_compute_region_weights_emits_normalized_table(tmp_path: Path) -> None:
    from lib.landmarks.cache.prediction_cache import DiskPredictionCache
    from lib.landmarks.core.schema import LandmarkPrediction
    from lib.landmarks.ensemble.static_weight_fit import compute_region_weights

    def _points(offset: float = 0.0) -> np.ndarray:
        base: np.ndarray = np.stack(
            (
                np.linspace(0, 67, LANDMARK_COUNT, dtype="float32"),
                np.linspace(10, 77, LANDMARK_COUNT, dtype="float32"),
            ),
            axis=1,
        )
        return T.cast(np.ndarray, (base + offset).astype("float32"))

    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    samples = []
    for sample_id in ("a", "b"):
        np.save(str(truth_dir / f"{sample_id}.npy"), _points())
        samples.append(
            {
                "sample_id": sample_id,
                "image": f"{sample_id}.png",
                "landmarks": f"truth/{sample_id}.npy",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}) + "\n", encoding="utf-8")

    cache = DiskPredictionCache(tmp_path / "cache")
    for sample_id in ("a", "b"):
        cache.write(sample_id, LandmarkPrediction(_points(1.0), model_name="hrnet"))
        cache.write(sample_id, LandmarkPrediction(_points(2.0), model_name="spiga"))
        cache.write(sample_id, LandmarkPrediction(_points(4.0), model_name="orformer"))

    table = compute_region_weights(manifest, tmp_path / "cache")

    assert set(table) == set(FUSION_REGION_INDICES)
    for columns in table.values():
        assert tuple(columns) == MODELS
        assert sum(columns.values()) == pytest.approx(1.0)
        # Lower-error hrnet (offset 1) outweighs higher-error orformer (offset 4).
        assert columns["hrnet"] > columns["orformer"]


# --------------------------------------------------------------------------- #
# Schema v2 region_weights
# --------------------------------------------------------------------------- #
def _equal_weights() -> dict[str, list[float]]:
    return {model: [1.0] * LANDMARK_COUNT for model in MODELS}


def _region_weights() -> dict[str, dict[str, float]]:
    return {
        region: {"hrnet": 2.0, "spiga": 1.0, "orformer": 1.0} for region in FUSION_REGION_INDICES
    }


def test_write_best_weights_region_block_is_v2_and_normalized(tmp_path: Path) -> None:
    path = tmp_path / WEIGHTS_FILENAME
    write_best_weights(path, _equal_weights(), models=MODELS, region_weights=_region_weights())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_schema_version"] == WEIGHTS_SCHEMA_VERSION_WITH_BUCKETS
    assert set(payload["region_weights"]) == set(FUSION_REGION_INDICES)
    assert payload["region_weights"]["nose"]["hrnet"] == pytest.approx(0.5)
    validate_weights_payload(payload, expected_models=MODELS)


def test_write_best_weights_rejects_region_missing_declared_model(tmp_path: Path) -> None:
    path = tmp_path / WEIGHTS_FILENAME
    region_weights = _region_weights()
    region_weights["nose"] = {"hrnet": 2.0, "spiga": 1.0}

    with pytest.raises(ValueError, match="missing weights for models"):
        write_best_weights(path, _equal_weights(), models=MODELS, region_weights=region_weights)


def test_validate_weights_rejects_region_block_on_v1() -> None:
    payload = {
        "artifact_schema_version": 1,
        "schema": "2d_68",
        "models": list(MODELS),
        "weights": {model: [1.0 / 3] * LANDMARK_COUNT for model in MODELS},
        "region_weights": normalize_region_weights(_region_weights()),
    }
    with pytest.raises(PromotedSetupError, match="require version"):
        validate_weights_payload(payload)


def test_validate_weights_rejects_unknown_region() -> None:
    payload = {
        "artifact_schema_version": WEIGHTS_SCHEMA_VERSION_WITH_BUCKETS,
        "schema": "2d_68",
        "models": list(MODELS),
        "weights": {model: [1.0 / 3] * LANDMARK_COUNT for model in MODELS},
        "region_weights": {"not_a_region": {model: 1.0 for model in MODELS}},
    }
    with pytest.raises(PromotedSetupError, match="unknown region"):
        validate_weights_payload(payload)


def test_load_promoted_setup_round_trips_region_weights(tmp_path: Path) -> None:
    weights_path = tmp_path / WEIGHTS_FILENAME
    setup_path = tmp_path / SETUP_FILENAME
    write_best_weights(
        weights_path, _equal_weights(), models=MODELS, region_weights=_region_weights()
    )
    write_best_setup(
        setup_path,
        candidate_id="sha256:0123abc",
        models=MODELS,
        strategy="static_weighted",
        outlier_threshold=None,
        weight_generator_name="region_inverse_error",
        weight_generator_params={"epsilon": 1e-6},
        crop_scale=1.6,
        bbox_source="manifest",
        regression_epsilon_nme=0.001,
        reproducibility={"split_assignment_hash": "sha256:abc"},
        fit={"sample_count": 12},
        selection_metrics={"sample_count": 4},
        report_metrics={"sample_count": 4},
        weights_path=WEIGHTS_FILENAME,
    )
    setup = load_promoted_setup(setup_path)
    assert setup.has_region_weights()
    assert set(setup.region_weights) == set(FUSION_REGION_INDICES)
    assert setup.region_weights["nose"]["hrnet"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Runtime region-fusion candidate
# --------------------------------------------------------------------------- #
def _single(name: str, points: np.ndarray) -> CandidateRecord:
    return CandidateRecord(
        name=name, landmarks=points.astype("float32"), is_fusion=False, contributing_models=(name,)
    )


def _grid_face(offset: float = 0.0) -> np.ndarray:
    base = np.linspace(40.0, 160.0, LANDMARK_COUNT, dtype="float32")
    return T.cast(np.ndarray, (np.stack([base, base[::-1]], axis=1) + offset).astype("float32"))


def test_append_region_candidate_noop_without_region_weights() -> None:
    candidates = [_single("hrnet", _grid_face(0.0)), _single("spiga", _grid_face(2.0))]
    metrics: dict[str, CandidateMetrics] = {}
    append_region_weight_candidate(
        candidates,
        RuntimeResolverConfig(),
        metrics=metrics,
        reference_bbox=(0.0, 0.0, 200.0, 200.0),
    )
    assert [candidate.name for candidate in candidates] == ["hrnet", "spiga"]


def test_append_region_candidate_assembles_region_by_region() -> None:
    """Each region's output follows the model that region weights favor."""
    hrnet = _single("hrnet", _grid_face(0.0))
    spiga = _single("spiga", _grid_face(10.0))
    candidates = [hrnet, spiga]
    metrics: dict[str, CandidateMetrics] = {}
    # jaw trusts hrnet fully, mouth_outer trusts spiga fully, others split.
    region_weights = {region: {"hrnet": 1.0, "spiga": 1.0} for region in FUSION_REGION_INDICES}
    region_weights["visible_jaw"] = {"hrnet": 1.0, "spiga": 0.0}
    region_weights["mouth_outer"] = {"hrnet": 0.0, "spiga": 1.0}
    config = RuntimeResolverConfig(region_weights=region_weights)
    append_region_weight_candidate(
        candidates, config, metrics=metrics, reference_bbox=(0.0, 0.0, 200.0, 200.0)
    )
    fused = next(c for c in candidates if c.name == "region_weighted")
    assert fused.is_fusion is True
    jaw = list(FUSION_REGION_INDICES["visible_jaw"])
    mouth = list(FUSION_REGION_INDICES["mouth_outer"])
    np.testing.assert_allclose(fused.landmarks[jaw], hrnet.landmarks[jaw], atol=1e-4)
    np.testing.assert_allclose(fused.landmarks[mouth], spiga.landmarks[mouth], atol=1e-4)
    assert metrics["region_weighted"].landmark_consensus_distance is not None
