#!/usr/bin/env python3
"""Smoke tests for the lightweight landmark scorer pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.scorer_contexts import load_scorer_contexts
from lib.landmarks.ensemble.scorer_eval import evaluate_runtime_resolver_scorer
from lib.landmarks.ensemble.scorer_training import SCORER_ARTIFACT, train_runtime_resolver_scorer
from lib.landmarks.ensemble.weights import save_weights
from lib.landmarks.pipeline_conventions import (
    SCORER_POLICY_REPORT_JSON,
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
    metadata_key,
)

MODELS = ("hrnet", "spiga", "orformer")
CANDIDATES = ("hrnet", "spiga", "orformer", "static_weighted_downweight")


@dataclass(frozen=True)
class SmokeFixture:
    root: Path
    gt_manifest: Path
    production_manifest: Path
    cache_dir: Path
    weights: Path
    gt_sidecar: Path


def _ellipse_points(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    count: int,
    start: float = 0.0,
    stop: float = 2.0 * np.pi,
) -> list[list[float]]:
    angles = np.linspace(start, stop, count, endpoint=False, dtype="float32")
    return [
        [center_x + radius_x * float(np.cos(a)), center_y + radius_y * float(np.sin(a))]
        for a in angles
    ]


def _canonical_face(offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    """Return a tiny deterministic 68-point face-like landmark cloud."""
    points: list[list[float]] = []
    points.extend(_ellipse_points(50, 64, 34, 42, 17, start=np.pi * 0.03, stop=np.pi * 0.97))
    points.extend(_ellipse_points(36, 35, 10, 3, 5, start=np.pi, stop=2 * np.pi))
    points.extend(_ellipse_points(64, 35, 10, 3, 5, start=np.pi, stop=2 * np.pi))
    points.extend([[50, 40], [50, 47], [50, 54], [50, 61]])
    points.extend(_ellipse_points(50, 64, 11, 5, 5, start=np.pi, stop=2 * np.pi))
    points.extend(_ellipse_points(36, 48, 8, 4, 6))
    points.extend(_ellipse_points(64, 48, 8, 4, 6))
    points.extend(_ellipse_points(50, 78, 17, 8, 12))
    points.extend(_ellipse_points(50, 78, 9, 4, 8))
    array = np.asarray(points, dtype="float32")
    assert array.shape == (68, 2)
    array[:, 0] += offset_x
    array[:, 1] += offset_y
    return array


def _write_manifest(
    path: Path, sample_id: str, landmarks: Path, *, metadata: dict[str, object]
) -> None:
    payload = {
        "schema": "2d_68",
        "samples": [
            {
                "sample_id": sample_id,
                "image": f"{sample_id}.png",
                "landmarks": landmarks.name,
                "dataset": "smoke",
                "condition": "profile_left" if sample_id.startswith("gt") else "production",
                "face_bbox": [10, 10, 90, 105],
                "metadata": metadata,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_cache(cache_dir: Path, sample_id: str, truth: np.ndarray, *, offset: float) -> None:
    cache = DiskPredictionCache(cache_dir)
    for idx, model in enumerate(MODELS):
        # Keep predictions close to truth but not identical so NME, target rows,
        # and policy reports have useful numeric content.
        prediction = truth + np.float32(offset + idx * 0.25)
        cache.write(
            sample_id,
            LandmarkPrediction(
                landmarks=prediction,
                schema="2d_68",
                model_name=model,
                source_landmark_count=68,
                coordinate_space="frame",
            ),
            checkpoint="smoke",
            config={"model": model, "fixture": "smoke"},
            refresh=True,
        )


def _write_sidecar(path: Path, sample_id: str, *, face_index: int = 0) -> None:
    row = {
        "sample_id": sample_id,
        "face_index": face_index,
        "landmark_ensemble": {
            "runtime_bucket": "profile_left",
            "runtime_bucket_source": "smoke_resolver_sidecar",
            "selected_candidate": "hrnet",
            "runtime_bucket_features": {
                "landmark_pose_yaw": 45.0,
                "landmark_pose_roll": 0.0,
                "max_disagreement_px": 2.0,
                "candidate_yaw_disagreement": 0.0,
            },
        },
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _make_fixture(tmp_path: Path) -> SmokeFixture:
    root = tmp_path / "landmark_smoke"
    root.mkdir()
    cache_dir = root / "cache"
    weights_path = root / "weights.json"
    gt_dir = root / "gt_hard"
    production_dir = root / "production_validated"
    gt_dir.mkdir()
    production_dir.mkdir()

    gt_truth = _canonical_face()
    production_truth = _canonical_face(offset_x=2.0, offset_y=1.0)
    gt_landmarks = gt_dir / "gt_smoke.npy"
    production_landmarks = production_dir / "production_smoke.npy"
    np.save(gt_landmarks, gt_truth)
    np.save(production_landmarks, production_truth)

    gt_manifest = gt_dir / "manifest.json"
    production_manifest = production_dir / "manifest.json"
    _write_manifest(gt_manifest, "gt_smoke", gt_landmarks, metadata={"face_index": 0})
    _write_manifest(
        production_manifest,
        "production_smoke",
        production_landmarks,
        metadata={
            "face_index": 0,
            "landmark_ensemble": {
                "runtime_bucket": "frontal",
                "runtime_bucket_source": "stored_manifest_landmark_ensemble",
                "selected_candidate": "hrnet",
                "runtime_bucket_features": {
                    "landmark_pose_yaw": 0.0,
                    "landmark_pose_roll": 0.0,
                    "max_disagreement_px": 1.0,
                    "candidate_yaw_disagreement": 0.0,
                },
            },
        },
    )

    _write_cache(cache_dir, "gt_smoke", gt_truth, offset=0.0)
    _write_cache(cache_dir, "production_smoke", production_truth, offset=0.1)
    save_weights(weights_path, {model: [1.0] * 68 for model in MODELS})
    sidecar_path = root / "resolver_metadata.jsonl"
    _write_sidecar(sidecar_path, "gt_smoke", face_index=0)
    return SmokeFixture(
        root=root,
        gt_manifest=gt_manifest,
        production_manifest=production_manifest,
        cache_dir=cache_dir,
        weights=weights_path,
        gt_sidecar=sidecar_path,
    )


def test_scorer_pipeline_smoke_build_train_evaluate_and_report(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    contexts = load_scorer_contexts(
        gt_manifest=fixture.gt_manifest,
        gt_cache_dir=fixture.cache_dir,
        production_manifest=fixture.production_manifest,
        production_cache_dir=fixture.cache_dir,
        weights_path=fixture.weights,
        candidates=CANDIDATES,
        gt_hard_resolver_metadata=fixture.gt_sidecar,
    )
    assert {context.source for context in contexts} == {
        SOURCE_GT_HARD,
        SOURCE_PRODUCTION_VALIDATED,
    }
    assert any(context.runtime_bucket_source == "smoke_resolver_sidecar" for context in contexts)
    assert any(
        context.runtime_bucket_source == "stored_manifest_landmark_ensemble"
        for context in contexts
    )

    train_dir = fixture.root / "train"
    metrics = train_runtime_resolver_scorer(
        gt_manifest=fixture.gt_manifest,
        gt_cache_dir=fixture.cache_dir,
        production_manifest=fixture.production_manifest,
        production_cache_dir=fixture.cache_dir,
        weights_path=fixture.weights,
        candidates=CANDIDATES,
        output_dir=train_dir,
        gt_hard_resolver_metadata=fixture.gt_sidecar,
        iterations=5,
        eval_fraction=0.0,
    )
    scorer_path = train_dir / SCORER_ARTIFACT
    assert scorer_path.is_file()
    assert metrics["row_count"] > 0
    assert metrics["candidate_count"] == len(CANDIDATES)

    eval_dir = fixture.root / "eval"
    report = evaluate_runtime_resolver_scorer(
        gt_manifest=fixture.gt_manifest,
        gt_cache_dir=fixture.cache_dir,
        production_manifest=fixture.production_manifest,
        production_cache_dir=fixture.cache_dir,
        weights_path=fixture.weights,
        scorer_path=scorer_path,
        candidates=CANDIDATES,
        output_dir=eval_dir,
        promotion_scope="production",
        gt_hard_resolver_metadata=fixture.gt_sidecar,
    )
    report_path = eval_dir / SCORER_POLICY_REPORT_JSON
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path.is_file()
    assert report["sample_count"] == 2
    assert payload["sample_count"] == 2
    assert payload["promotion_scope"] == "production"
    assert payload["promotion_status"] in {"pass", "fail"}
    assert isinstance(payload["failed_gates"], list)
    assert payload["primary_scorer_policy"] in payload
    assert payload["runtime_policy"] == "learned_quality_v1"
    assert "production_only_policy_metrics" in payload
    assert payload["production_only_policy_metrics"]["sample_count"] == 1
    assert "gt_hard_all_policy_metrics" in payload
    assert payload["gt_hard_all_policy_metrics"]["sample_count"] == 1


def test_scorer_pipeline_smoke_requires_gt_hard_sidecar(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    with pytest.raises(ValueError, match="resolver metadata missing"):
        load_scorer_contexts(
            gt_manifest=fixture.gt_manifest,
            gt_cache_dir=fixture.cache_dir,
            production_manifest=None,
            production_cache_dir=None,
            weights_path=fixture.weights,
            candidates=CANDIDATES,
            gt_hard_resolver_metadata=None,
        )


def test_scorer_pipeline_smoke_catches_broken_cache_path(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    with pytest.raises(ValueError, match="no scorer contexts were loaded"):
        load_scorer_contexts(
            gt_manifest=fixture.gt_manifest,
            gt_cache_dir=fixture.root / "missing_cache",
            production_manifest=None,
            production_cache_dir=None,
            weights_path=fixture.weights,
            candidates=CANDIDATES,
            gt_hard_resolver_metadata=fixture.gt_sidecar,
        )


def test_scorer_pipeline_smoke_catches_schema_drift_in_sidecar_key(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    bad_sidecar = fixture.root / "bad_resolver_metadata.jsonl"
    _write_sidecar(bad_sidecar, "gt_smoke", face_index=1)

    with pytest.raises(ValueError, match="not present in manifest"):
        load_scorer_contexts(
            gt_manifest=fixture.gt_manifest,
            gt_cache_dir=fixture.cache_dir,
            production_manifest=None,
            production_cache_dir=None,
            weights_path=fixture.weights,
            candidates=CANDIDATES,
            gt_hard_resolver_metadata=bad_sidecar,
        )


def test_smoke_fixture_contains_expected_gt_sidecar_key(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    rows = [
        json.loads(line) for line in fixture.gt_sidecar.read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    assert metadata_key(rows[0]["sample_id"], rows[0]["face_index"]) == ("gt_smoke", 0)
