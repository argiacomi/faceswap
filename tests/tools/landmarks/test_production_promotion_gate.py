#!/usr/bin/env python3
"""Tests for production promotion validation gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.weights import save_weights
from tools.landmarks.production_promotion_gate import (
    ProductionGateConfig,
    _sidecar_metadata_by_key,
    run_production_promotion_gate,
)


def _points(offset: float = 0.0) -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[:, 0] = np.linspace(10.0, 90.0, 68) + offset
    points[:, 1] = np.linspace(20.0, 110.0, 68)
    return points


def _write_fixture(
    tmp_path: Path,
    samples: list[dict[str, object]],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[Path, Path, Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "gate"
    weights_path = tmp_path / "weights.json"
    model_weights = weights or {"hrnet": 1.0, "spiga": 0.0, "orformer": 0.0}
    save_weights(weights_path, {model: [value] * 68 for model, value in model_weights.items()})
    cache = DiskPredictionCache(cache_dir)
    manifest_samples = []
    for spec in samples:
        sample_id = str(spec["sample_id"])
        truth_path = tmp_path / f"{sample_id}.npy"
        np.save(str(truth_path), _points())
        predictions = spec["predictions"]
        assert isinstance(predictions, dict)
        for model, offset in predictions.items():
            cache.write(
                sample_id,
                LandmarkPrediction(_points(float(offset)), model_name=str(model)),
                checkpoint="test",
            )
        metadata = spec.get(
            "metadata",
            {
                "landmark_ensemble": {
                    "runtime_bucket": spec.get("condition", "frontal"),
                    "bucket": spec.get("condition", "frontal"),
                    "selected_candidate": spec.get("selected_candidate", "hrnet"),
                }
            },
        )
        manifest_samples.append(
            {
                "sample_id": sample_id,
                "image": f"{sample_id}.jpg",
                "landmarks": truth_path.name,
                "dataset": "production_validated",
                "condition": spec.get("condition", "frontal"),
                "normalizer": 100.0,
                "face_bbox": [0.0, 0.0, 100.0, 120.0],
                "metadata": metadata,
            }
        )
    manifest_path.write_text(
        json.dumps({"dataset": "production_validated", "samples": manifest_samples}),
        encoding="utf-8",
    )
    return manifest_path, cache_dir, weights_path, output_dir


def _run(
    tmp_path: Path,
    samples: list[dict[str, object]],
    *,
    policy: str,
    weights: dict[str, float] | None = None,
    min_hard_bucket_gate_count: int = 20,
) -> dict[str, object]:
    manifest_path, cache_dir, weights_path, output_dir = _write_fixture(
        tmp_path,
        samples,
        weights=weights,
    )
    return run_production_promotion_gate(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        output_dir=output_dir,
        config=ProductionGateConfig(
            policy=policy,
            min_hard_bucket_gate_count=min_hard_bucket_gate_count,
        ),
    )


def test_production_gate_passes_competitive_policy(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "sample_id": "s1",
                "condition": "profile_left",
                "predictions": {"hrnet": 1.0, "spiga": 4.0, "orformer": 5.0},
            },
            {
                "sample_id": "s2",
                "condition": "large_yaw_left",
                "predictions": {"hrnet": 1.0, "spiga": 3.0, "orformer": 4.0},
            },
        ],
        policy="candidate:hrnet",
    )

    assert report["status"] == "pass"
    assert report["best_static_ensemble"]["candidate"] in {  # type: ignore[index]
        "static_weighted",
        "static_weighted_downweight",
    }
    assert report["current_promoted_setup"]["candidate"] == "static_weighted_downweight"  # type: ignore[index]
    assert Path(report["artifacts"]["json"]).is_file()  # type: ignore[index]
    assert Path(report["artifacts"]["per_bucket_csv"]).is_file()  # type: ignore[index]


def test_production_gate_fails_against_best_single(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "sample_id": "s1",
                "condition": "profile_left",
                "predictions": {"hrnet": 1.0, "spiga": 5.0, "orformer": 6.0},
            }
        ],
        policy="candidate:spiga",
    )

    assert report["status"] == "fail"
    assert "chosen_policy_mean_nme_regresses_vs_best_single" in report["failed_gates"]  # type: ignore[operator]


def test_production_gate_fails_against_static_ensemble(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "sample_id": "s1",
                "condition": "frontal",
                "predictions": {"hrnet": 1.0, "spiga": -1.0},
            }
        ],
        policy="candidate:hrnet",
        weights={"hrnet": 0.5, "spiga": 0.5},
    )

    assert report["status"] == "fail"
    assert "chosen_policy_mean_nme_regresses_vs_static_downweight" in report["failed_gates"]  # type: ignore[operator]


def test_production_gate_fails_missing_runtime_metadata(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "sample_id": "s1",
                "condition": "frontal",
                "predictions": {"hrnet": 1.0, "spiga": 3.0, "orformer": 4.0},
                "metadata": {},
            }
        ],
        policy="candidate:hrnet",
    )

    assert report["status"] == "fail"
    assert "missing_production_runtime_metadata" in report["failed_gates"]  # type: ignore[operator]


def test_production_gate_fails_derived_no_image_runtime_metadata(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "sample_id": "s1",
                "condition": "frontal",
                "predictions": {"hrnet": 1.0, "spiga": 3.0, "orformer": 4.0},
                "metadata": {
                    "landmark_ensemble": {
                        "runtime_bucket": "frontal",
                        "bucket": "frontal",
                        "selected_candidate": "hrnet",
                        "runtime_bucket_source": "derived_no_image_evidence",
                    }
                },
            }
        ],
        policy="candidate:hrnet",
    )

    assert report["status"] == "fail"
    assert report["derived_no_image_runtime_metadata_count"] == 1
    assert "production_runtime_bucket_source_derived_no_image_evidence" in report["failed_gates"]  # type: ignore[operator]


def test_production_gate_uses_image_aware_sidecar_metadata(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path, output_dir = _write_fixture(
        tmp_path,
        [
            {
                "sample_id": "s1",
                "condition": "frontal",
                "predictions": {"hrnet": 1.0, "spiga": 3.0, "orformer": 4.0},
                "metadata": {},
            }
        ],
    )
    sidecar = tmp_path / "resolver_metadata.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "face_index": 0,
                "landmark_ensemble": {
                    "runtime_bucket": "profile_left",
                    "bucket": "profile_left",
                    "selected_candidate": "spiga",
                    "runtime_bucket_source": "image_aware_backfill",
                    "runtime_bucket_features": {"landmark_pose_yaw": -45.0},
                    "resolver": {
                        "runtime_bucket": "profile_left",
                        "bucket": "profile_left",
                        "selected_candidate": "spiga",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_production_promotion_gate(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        output_dir=output_dir,
        resolver_metadata_path=sidecar,
        config=ProductionGateConfig(policy="candidate:hrnet"),
    )

    assert report["missing_runtime_metadata_count"] == 0
    assert report["derived_no_image_runtime_metadata_count"] == 0
    assert report["production_condition_counts"] == {"profile_left": 1}


def test_production_sidecar_metadata_keeps_face_index_key(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "production_validated",
                "samples": [
                    {
                        "sample_id": "frame1",
                        "image": "frame1.jpg",
                        "landmarks": "face0.npy",
                        "metadata": {"face_index": 0},
                    },
                    {
                        "sample_id": "frame1",
                        "image": "frame1.jpg",
                        "landmarks": "face1.npy",
                        "metadata": {"face_index": 1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    sidecar = tmp_path / "resolver_metadata.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "frame1",
                        "face_index": 0,
                        "landmark_ensemble": {"runtime_bucket": "frontal"},
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "frame1",
                        "face_index": 1,
                        "landmark_ensemble": {"runtime_bucket": "profile_left"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = _sidecar_metadata_by_key(manifest_path, sidecar)

    assert metadata[("frame1", 0)]["landmark_ensemble"]["runtime_bucket"] == "frontal"
    assert metadata[("frame1", 1)]["landmark_ensemble"]["runtime_bucket"] == "profile_left"


def test_production_gate_fails_per_bucket_regression(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "sample_id": "frontal_1",
                "condition": "frontal",
                "predictions": {"hrnet": 5.0, "spiga": 1.0, "orformer": 5.0},
            },
            {
                "sample_id": "profile_1",
                "condition": "profile_left",
                "predictions": {"hrnet": 1.0, "spiga": 5.0, "orformer": 5.0},
            },
        ],
        policy="candidate:spiga",
        min_hard_bucket_gate_count=1,
    )

    assert report["status"] == "fail"
    assert "bucket_profile_left_mean_regresses_vs_best_single" in report["failed_gates"]  # type: ignore[operator]


def test_production_gate_warns_for_tiny_hard_bucket_regression(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "sample_id": "frontal_1",
                "condition": "frontal",
                "predictions": {"hrnet": 5.0, "spiga": 1.0, "orformer": 5.0},
            },
            {
                "sample_id": "profile_1",
                "condition": "profile_left",
                "predictions": {"hrnet": 1.0, "spiga": 5.0, "orformer": 5.0},
            },
        ],
        policy="candidate:spiga",
    )

    assert report["status"] == "pass"
    assert not report["failed_gates"]
    assert report["warnings"] == ["bucket_profile_left_sample_count_1_below_gate_min_20"]
    assert report["per_bucket"]["profile_left"]["sample_count"] == 1.0  # type: ignore[index]
