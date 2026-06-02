#!/usr/bin/env python3
"""Tests for Phase 5 #8 per-bucket landmark fusion weights.

Covers the runtime-bucket -> weight-bucket mapping, the backward-compatible
``best_weights.json`` v2 schema (optional ``bucket_weights`` block), and the
second-pass per-bucket fusion candidate appension in the runtime resolver.
"""

from __future__ import annotations

import json
import typing as T
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.hard_condition_taxonomy import (
    GLOBAL_WEIGHT_BUCKET,
    WEIGHT_BUCKETS,
    applicable_weight_buckets,
    weight_bucket_from_pose,
    weight_bucket_from_runtime,
)
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
    ModelPrediction,
    RuntimeResolverConfig,
    append_bucket_weight_candidates,
    bucket_candidate_name,
    build_candidates,
)
from lib.landmarks.ensemble.weights import LANDMARK_COUNT, normalize_bucket_weights

MODELS = ("hrnet", "spiga", "orformer")


# --------------------------------------------------------------------------- #
# Bucket mapping helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("runtime_bucket", "occluded", "expected"),
    [
        ("frontal", False, "frontal"),
        ("intermediate", False, "frontal"),
        ("large_roll", False, "frontal"),
        ("extreme_roll", False, "frontal"),
        ("frontal", True, "occlusion"),
        ("large_yaw_left", False, "large_yaw"),
        ("rolled_large_yaw_right", False, "large_yaw"),
        ("large_yaw_left", True, "occlusion"),
        ("profile_left", False, "profile"),
        ("profile_right", True, "profile_occlusion"),
        ("rolled_profile_left", False, "rolled_profile"),
        ("rolled_profile_right", True, "profile_occlusion"),
    ],
)
def test_weight_bucket_from_runtime(runtime_bucket: str, occluded: bool, expected: str) -> None:
    assert weight_bucket_from_runtime(runtime_bucket, occluded=occluded) == expected
    assert expected in WEIGHT_BUCKETS


@pytest.mark.parametrize(
    ("yaw", "roll", "occluded", "expected"),
    [
        (0.0, 0.0, False, "frontal"),
        (40.0, 0.0, False, "large_yaw"),
        (60.0, 0.0, False, "profile"),
        (60.0, 40.0, False, "rolled_profile"),
        (60.0, 40.0, True, "profile_occlusion"),
        (10.0, 5.0, True, "occlusion"),
        (None, None, False, "frontal"),
    ],
)
def test_weight_bucket_from_pose(yaw, roll, occluded: bool, expected: str) -> None:
    assert weight_bucket_from_pose(yaw, roll, occluded=occluded) == expected


def test_applicable_weight_buckets_pairs_pose_with_occlusion_sibling() -> None:
    assert applicable_weight_buckets("profile_left") == ("profile", "profile_occlusion")
    assert applicable_weight_buckets("rolled_profile_right") == (
        "rolled_profile",
        "profile_occlusion",
    )
    assert applicable_weight_buckets("large_yaw_left") == ("large_yaw", "occlusion")
    assert applicable_weight_buckets("frontal") == ("frontal", "occlusion")


def test_global_weight_bucket_is_not_a_pose_bucket() -> None:
    assert GLOBAL_WEIGHT_BUCKET not in WEIGHT_BUCKETS


# --------------------------------------------------------------------------- #
# Schema v2 round-trip + validation
# --------------------------------------------------------------------------- #
def _equal_weights() -> dict[str, list[float]]:
    return {model: [1.0] * LANDMARK_COUNT for model in MODELS}


def _bucket_weights() -> dict[str, dict[str, list[float]]]:
    return {
        "profile": {
            "hrnet": [2.0] * LANDMARK_COUNT,
            "spiga": [1.0] * LANDMARK_COUNT,
            "orformer": [1.0] * LANDMARK_COUNT,
        },
        "frontal": _equal_weights(),
    }


def test_write_best_weights_without_buckets_stays_v1(tmp_path: Path) -> None:
    path = tmp_path / WEIGHTS_FILENAME
    write_best_weights(path, _equal_weights(), models=MODELS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_schema_version"] == 1
    assert "bucket_weights" not in payload


def test_write_best_weights_with_buckets_is_v2_and_normalized(tmp_path: Path) -> None:
    path = tmp_path / WEIGHTS_FILENAME
    write_best_weights(path, _equal_weights(), models=MODELS, bucket_weights=_bucket_weights())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_schema_version"] == WEIGHTS_SCHEMA_VERSION_WITH_BUCKETS
    assert set(payload["bucket_weights"]) == {"profile", "frontal"}
    # hrnet weighted 2:1:1 in the profile bucket -> normalized 0.5 per landmark.
    assert payload["bucket_weights"]["profile"]["hrnet"][0] == pytest.approx(0.5)
    validate_weights_payload(payload, expected_models=MODELS)


def test_validate_weights_rejects_buckets_on_v1() -> None:
    payload = {
        "artifact_schema_version": 1,
        "schema": "2d_68",
        "models": list(MODELS),
        "weights": normalize_bucket_weights({"g": _equal_weights()})["g"],
        "bucket_weights": _bucket_weights(),
    }
    with pytest.raises(PromotedSetupError, match="require version"):
        validate_weights_payload(payload)


def test_validate_weights_rejects_malformed_bucket_column() -> None:
    bad = _bucket_weights()
    bad["profile"]["hrnet"] = [1.0] * (LANDMARK_COUNT - 1)
    payload = {
        "artifact_schema_version": WEIGHTS_SCHEMA_VERSION_WITH_BUCKETS,
        "schema": "2d_68",
        "models": list(MODELS),
        "weights": normalize_bucket_weights({"g": _equal_weights()})["g"],
        "bucket_weights": bad,
    }
    with pytest.raises(PromotedSetupError, match="bucket_weights"):
        validate_weights_payload(payload)


def _write_setup_pair(tmp_path: Path, *, bucket_weights=None) -> Path:
    weights_path: Path = tmp_path / WEIGHTS_FILENAME
    setup_path: Path = tmp_path / SETUP_FILENAME
    write_best_weights(
        weights_path, _equal_weights(), models=MODELS, bucket_weights=bucket_weights
    )
    write_best_setup(
        setup_path,
        candidate_id="sha256:0123abc",
        models=MODELS,
        strategy="static_weighted",
        outlier_threshold=None,
        weight_generator_name="inverse_mean_error",
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
    return setup_path


def test_load_promoted_setup_v1_has_empty_bucket_weights(tmp_path: Path) -> None:
    setup = load_promoted_setup(_write_setup_pair(tmp_path))
    assert setup.bucket_weights == {}
    assert not setup.has_bucket_weights()


def test_load_promoted_setup_v2_round_trips_bucket_weights(tmp_path: Path) -> None:
    setup = load_promoted_setup(_write_setup_pair(tmp_path, bucket_weights=_bucket_weights()))
    assert setup.has_bucket_weights()
    assert set(setup.bucket_weights) == {"profile", "frontal"}
    assert setup.bucket_weights["profile"]["hrnet"][0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Runtime second-pass candidate appension
# --------------------------------------------------------------------------- #
def _single(name: str, points: np.ndarray) -> CandidateRecord:
    return CandidateRecord(
        name=name,
        landmarks=points.astype("float32"),
        is_fusion=False,
        contributing_models=(name,),
    )


def _grid_face(offset: float = 0.0) -> np.ndarray:
    base = np.linspace(40.0, 160.0, LANDMARK_COUNT, dtype="float32")
    return T.cast(np.ndarray, (np.stack([base, base[::-1]], axis=1) + offset).astype("float32"))


def _two_model_candidates() -> list[CandidateRecord]:
    return [_single("hrnet", _grid_face(0.0)), _single("spiga", _grid_face(1.0))]


def test_append_bucket_candidates_noop_without_bucket_weights() -> None:
    candidates = _two_model_candidates()
    metrics: dict[str, CandidateMetrics] = {}
    append_bucket_weight_candidates(
        candidates,
        RuntimeResolverConfig(weights={m: [0.5] * LANDMARK_COUNT for m in ("hrnet", "spiga")}),
        bucket="profile_left",
        metrics=metrics,
        reference_bbox=(0.0, 0.0, 200.0, 200.0),
    )
    assert [candidate.name for candidate in candidates] == ["hrnet", "spiga"]


def test_append_bucket_candidates_adds_named_fusion_candidate() -> None:
    candidates = _two_model_candidates()
    metrics: dict[str, CandidateMetrics] = {}
    config = RuntimeResolverConfig(
        bucket_weights={
            "profile": {"hrnet": [3.0] * LANDMARK_COUNT, "spiga": [1.0] * LANDMARK_COUNT}
        }
    )
    append_bucket_weight_candidates(
        candidates,
        config,
        bucket="profile_left",
        metrics=metrics,
        reference_bbox=(0.0, 0.0, 200.0, 200.0),
    )
    name = bucket_candidate_name("static_weighted", "profile")
    by_name = {candidate.name: candidate for candidate in candidates}
    assert name in by_name
    appended = by_name[name]
    assert appended.is_fusion is True
    assert appended.contributing_models == ("hrnet", "spiga")
    # 3:1 hrnet weighting pulls the fusion toward hrnet's grid (offset 0).
    np.testing.assert_allclose(appended.landmarks, _grid_face(0.25), rtol=0, atol=1e-4)
    # Metrics + consensus geometry are populated for the new candidate.
    assert name in metrics
    assert metrics[name].landmark_consensus_distance is not None


def test_append_bucket_candidates_falls_back_when_bucket_missing() -> None:
    candidates = _two_model_candidates()
    metrics: dict[str, CandidateMetrics] = {}
    # Only a frontal weight set is fit; a profile face has no applicable bucket.
    config = RuntimeResolverConfig(
        bucket_weights={
            "frontal": {"hrnet": [1.0] * LANDMARK_COUNT, "spiga": [1.0] * LANDMARK_COUNT}
        }
    )
    append_bucket_weight_candidates(
        candidates,
        config,
        bucket="profile_left",
        metrics=metrics,
        reference_bbox=(0.0, 0.0, 200.0, 200.0),
    )
    assert [candidate.name for candidate in candidates] == ["hrnet", "spiga"]


def test_append_bucket_candidates_skips_single_model_faces() -> None:
    candidates = [_single("hrnet", _grid_face(0.0))]
    metrics: dict[str, CandidateMetrics] = {}
    config = RuntimeResolverConfig(bucket_weights={"profile": {"hrnet": [1.0] * LANDMARK_COUNT}})
    append_bucket_weight_candidates(
        candidates,
        config,
        bucket="profile_left",
        metrics=metrics,
        reference_bbox=(0.0, 0.0, 200.0, 200.0),
    )
    assert [candidate.name for candidate in candidates] == ["hrnet"]


def test_build_candidates_unchanged_without_bucket_weights() -> None:
    """The base candidate set is untouched; bucket appension is a separate pass."""
    predictions = [
        ModelPrediction("hrnet", _grid_face(0.0)),
        ModelPrediction("spiga", _grid_face(1.0)),
    ]
    config = RuntimeResolverConfig(weights={m: [0.5] * LANDMARK_COUNT for m in ("hrnet", "spiga")})
    names = {candidate.name for candidate in build_candidates(predictions, config)}
    assert "hrnet" in names and "spiga" in names
    assert not any("@" in name for name in names)
