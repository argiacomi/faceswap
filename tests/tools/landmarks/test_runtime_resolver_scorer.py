#!/usr/bin/env python3
"""Tests for learned runtime resolver scorer training and evaluation tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

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
from tools.landmarks.export_resolver_candidate_table import export_resolver_candidate_table
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
    assert artifact["model_type"] == "logistic_regression"
    assert "candidate_name=spiga" in artifact["features"]
    assert (output_dir / "runtime_resolver_scorer_training_rows.csv").is_file()
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
    assert report["best_single"]["candidate"] == "hrnet"
    assert (tmp_path / "eval" / "scorer_policy_report.csv").is_file()
    assert (tmp_path / "eval" / "scorer_feature_importance.csv").is_file()


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
    assert set(rows[0]) >= {
        "sample_id",
        "candidate",
        "nme",
        "failure",
        "runtime_bucket",
        "geometry_veto_reasons",
    }
    assert hrnet_mean == gate["best_single_mean_nme"]


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
