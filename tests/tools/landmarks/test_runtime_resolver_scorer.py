#!/usr/bin/env python3
"""Tests for learned runtime resolver scorer training and evaluation tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.landmarks.runtime_resolver_scorer_data as scorer_data
from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverScorer,
    write_runtime_resolver_scorer,
)
from lib.landmarks.ensemble.weights import save_weights
from tools.landmarks.evaluate_runtime_resolver_scorer import (
    evaluate_runtime_resolver_scorer,
)
from tools.landmarks.export_resolver_candidate_table import (
    export_resolver_candidate_table,
)
from tools.landmarks.export_resolver_candidate_table import (
    main as export_resolver_candidate_table_main,
)
from tools.landmarks.production_promotion_gate import (
    ProductionGateConfig,
    run_production_promotion_gate,
)
from tools.landmarks.train_runtime_resolver_scorer import train_runtime_resolver_scorer


def _face(offset: float = 0.0) -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")
    points[0:17, 0] = np.linspace(10, 90, 17)
    points[0:17, 1] = 80 + 20 * np.sin(np.linspace(0, np.pi, 17))
    points[17:22, 0] = np.linspace(20, 40, 5)
    points[17:22, 1] = 35
    points[22:27, 0] = np.linspace(60, 80, 5)
    points[22:27, 1] = 35
    points[27:36, 0] = 50
    points[27:36, 1] = np.linspace(40, 70, 9)
    points[36:42, 0] = np.linspace(25, 38, 6)
    points[36:42, 1] = 48
    points[42:48, 0] = np.linspace(62, 75, 6)
    points[42:48, 1] = 48
    points[48:68, 0] = np.linspace(35, 65, 20)
    points[48:68, 1] = 82
    points[:, 0] += offset
    return points


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    cache_dir = tmp_path / "cache"
    weights_path = tmp_path / "weights.json"
    save_weights(
        weights_path,
        {"hrnet": [1.0] * 68, "spiga": [0.0] * 68, "orformer": [0.0] * 68},
    )
    cache = DiskPredictionCache(cache_dir)
    samples = []
    for idx in range(2):
        sample_id = f"s{idx}"
        truth_path = tmp_path / f"{sample_id}.npy"
        np.save(str(truth_path), _face())
        for model, offset in {"hrnet": 1.0, "spiga": 16.0, "orformer": 6.0}.items():
            cache.write(
                sample_id,
                LandmarkPrediction(_face(offset), model_name=model),
                checkpoint="test",
            )
        samples.append(
            {
                "sample_id": sample_id,
                "image": f"{sample_id}.jpg",
                "landmarks": truth_path.name,
                "dataset": "production_validated",
                "condition": "profile_left",
                "normalizer": 100.0,
                "face_bbox": [0.0, 20.0, 100.0, 110.0],
                "metadata": {
                    "landmark_ensemble": {
                        "runtime_bucket": "stored_profile_left",
                        "bucket": "stored_profile_left",
                        "selected_candidate": "hrnet",
                        "runtime_bucket_features": {
                            "candidate_yaw_disagreement": 12.5,
                            "max_disagreement_px": 42.0,
                            "landmark_pose_yaw": -36.0,
                            "landmark_pose_roll": 4.0,
                        },
                    }
                },
            }
        )
    manifest_path.write_text(
        json.dumps({"dataset": "production_validated", "samples": samples}),
        encoding="utf-8",
    )
    return manifest_path, cache_dir, weights_path


def test_train_runtime_resolver_scorer_writes_artifact_and_rows(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "train"

    metrics = train_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=output_dir,
        iterations=20,
    )

    artifact = json.loads((output_dir / "runtime_resolver_scorer.json").read_text())
    assert metrics["row_count"] == 6
    assert metrics["train_metrics"]["row_count"] == 3
    assert metrics["eval_metrics"]["row_count"] == 3
    assert metrics["production_only_eval_metrics"]["row_count"] == 3
    assert metrics["gt_hard_only_eval_metrics"]["row_count"] == 0
    assert artifact["model_type"] == "logistic_regression"
    assert "candidate_name=spiga" in artifact["features"]
    assert "candidate_distance_to_hrnet" in artifact["features"]
    assert "single_model_disagreement_px" in artifact["features"]
    assert "hrnet_geometry_valid" in artifact["features"]
    assert "runtime_bucket_source=stored_manifest_landmark_ensemble" in artifact["features"]
    assert (output_dir / "runtime_resolver_scorer_training_rows.csv").is_file()
    assert (output_dir / "runtime_resolver_scorer_eval_rows.csv").is_file()
    assert (output_dir / "candidate_table.csv").is_file()


def test_evaluate_runtime_resolver_scorer_reports_policy(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=hrnet", "candidate_name=spiga"),
            coefficients=(-5.0, 5.0),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        scorer_path=scorer_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=tmp_path / "eval",
    )

    assert report["status"] == "pass"
    assert report["learned_quality_v1"]["pick_counts"] == {"hrnet": 2}
    assert report["production_only_policy_metrics"]["sample_count"] == 2
    assert report["production_only_policy_metrics"]["learned_quality_v1"]["pick_counts"] == {
        "hrnet": 2
    }
    assert report["gt_hard_only_policy_metrics"]["sample_count"] == 0
    assert report["best_single"]["candidate"] == "hrnet"
    assert (tmp_path / "eval" / "scorer_policy_report.csv").is_file()
    assert (tmp_path / "eval" / "scorer_feature_importance.csv").is_file()


def test_evaluate_runtime_resolver_scorer_uses_safe_fallback_for_high_risk(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=static_weighted",),
            coefficients=(-0.25,),
            intercept=1.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        scorer_path=scorer_path,
        candidates=(
            "hrnet",
            "spiga",
            "orformer",
            "static_weighted",
            "static_weighted_downweight",
        ),
        output_dir=tmp_path / "eval_safe",
    )

    assert report["learned_quality_v1"]["pick_counts"] == {"hrnet": 2}
    assert report["safe_fallback_count"] == 2


def test_evaluate_runtime_resolver_scorer_uses_hrnet_for_hard_slice_contradiction(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample["condition"] = "rolled_large_yaw_right"
        ensemble = sample["metadata"]["landmark_ensemble"]
        ensemble["runtime_bucket"] = "rolled_large_yaw_right"
        ensemble["bucket"] = "rolled_large_yaw_right"
        ensemble["runtime_bucket_features"]["candidate_yaw_disagreement"] = 95.0
        ensemble["runtime_bucket_features"]["max_disagreement_px"] = 95.0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=static_weighted",),
            coefficients=(-5.0,),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        scorer_path=scorer_path,
        candidates=(
            "hrnet",
            "spiga",
            "orformer",
            "static_weighted",
            "static_weighted_downweight",
        ),
        output_dir=tmp_path / "eval_hard_slice",
    )

    assert report["learned_quality_v1"]["pick_counts"] == {"hrnet": 2}
    assert report["hard_slice_fallback_count"] == 2


def test_export_resolver_candidate_table_row_count_and_gate_metrics(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    output_csv = tmp_path / "candidate_table.csv"

    report = export_resolver_candidate_table(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_csv=output_csv,
    )
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    gate = run_production_promotion_gate(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        output_dir=tmp_path / "gate",
        config=ProductionGateConfig(policy="candidate:hrnet"),
    )
    hrnet_mean = sum(float(row["nme"]) for row in rows if row["candidate"] == "hrnet") / 2

    assert report["row_count"] == 6
    assert len(rows) == 6
    assert {row["runtime_bucket"] for row in rows} == {"stored_profile_left"}
    assert {row["runtime_bucket_source"] for row in rows} == {"stored_manifest_landmark_ensemble"}
    assert set(rows[0]) >= {
        "sample_id",
        "candidate",
        "nme",
        "failure",
        "runtime_bucket",
        "runtime_bucket_source",
        "geometry_veto_reasons",
    }
    assert hrnet_mean == gate["best_single_mean_nme"]


def test_export_resolver_candidate_table_cli_default_includes_plain_average(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    output_csv = tmp_path / "candidate_table_default.csv"

    exit_code = export_resolver_candidate_table_main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(cache_dir),
            "--weights",
            str(weights_path),
            "--output-csv",
            str(output_csv),
        ]
    )

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert exit_code == 0
    assert len(rows) == 16
    assert sum(1 for _line in output_csv.open(encoding="utf-8")) == 17
    assert {row["candidate"] for row in rows} == set(scorer_data.DEFAULT_SCORER_CANDIDATES)
    assert "plain_average" in {row["candidate"] for row in rows}


def test_scorer_evaluator_fails_when_current_policy_candidate_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=hrnet", "candidate_name=spiga"),
            coefficients=(-5.0, 5.0),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )
    monkeypatch.setattr(
        scorer_data,
        "resolve_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(selected_candidate="plain_average"),
    )

    with pytest.raises(ValueError, match="missing from evaluation set"):
        evaluate_runtime_resolver_scorer(
            gt_manifest=None,
            gt_cache_dir=None,
            production_manifest=manifest_path,
            production_cache_dir=cache_dir,
            weights_path=weights_path,
            scorer_path=scorer_path,
            candidates=("hrnet", "spiga", "static_weighted_downweight"),
            output_dir=tmp_path / "eval",
        )


def test_export_resolver_candidate_table_skips_missing_candidate_prediction(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    cache_dir = tmp_path / "cache"
    weights_path = tmp_path / "weights.json"
    save_weights(weights_path, {"hrnet": [1.0] * 68, "spiga": [0.0] * 68})
    truth_path = tmp_path / "s1.npy"
    np.save(str(truth_path), _face())
    DiskPredictionCache(cache_dir).write(
        "s1",
        LandmarkPrediction(_face(1.0), model_name="hrnet"),
        checkpoint="test",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "s1",
                        "image": "s1.jpg",
                        "landmarks": truth_path.name,
                        "dataset": "production_validated",
                        "condition": "frontal",
                        "normalizer": 100.0,
                        "face_bbox": [0.0, 20.0, 100.0, 110.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = export_resolver_candidate_table(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga"),
        output_csv=tmp_path / "missing_candidate_table.csv",
    )

    assert report["row_count"] == 1
