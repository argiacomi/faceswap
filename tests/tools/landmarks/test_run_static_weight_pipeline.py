#!/usr/bin/env python3
"""Tests for the landmark static weight pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tools.landmarks.run_static_weight_pipeline import main as pipeline_main


def _points(offset: float = 0.0) -> np.ndarray:
    """Return deterministic canonical landmarks."""
    return np.stack(
        (
            np.linspace(20, 70, 68, dtype="float32"),
            np.linspace(25, 75, 68, dtype="float32"),
        ),
        axis=1,
    ) + offset


def _write_fixture_run(root: Path, *, models: tuple[str, ...] = ("hrnet", "spiga")) -> dict[str, Path]:
    """Create a tiny manifest, image, truth, and raw prediction roots."""
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    image = np.zeros((96, 96, 3), dtype="uint8")
    assert cv2.imwrite(str(dataset_dir / "sample.png"), image)
    np.save(str(dataset_dir / "truth.npy"), _points())
    manifest = dataset_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "samples": [
                    {
                        "sample_id": "sample",
                        "dataset": "fixture",
                        "condition": "clean",
                        "image": "sample.png",
                        "landmarks": "truth.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    prediction_roots: dict[str, Path] = {}
    for index, model in enumerate(models):
        pred_dir = root / "raw_predictions" / model
        pred_dir.mkdir(parents=True)
        np.save(str(pred_dir / "sample.npy"), _points(float(index)))
        prediction_roots[model] = pred_dir
    return {"manifest": manifest, **prediction_roots}


def _prediction_root_args(paths: dict[str, Path], models: tuple[str, ...]) -> list[str]:
    """Return repeated --prediction-root args."""
    args: list[str] = []
    for model in models:
        args.extend(["--prediction-root", f"{model}={paths[model]}"])
    return args


def _summary(root: Path) -> dict[str, object]:
    """Load run_summary.json."""
    return json.loads((root / "run_summary.json").read_text(encoding="utf-8"))


def test_static_weight_pipeline_dry_run(tmp_path: Path) -> None:
    """Dry run writes planned stages without executing the pipeline."""
    root = tmp_path / "run"

    result = pipeline_main(
        [
            "--output-root",
            str(root),
            "--datasets",
            "wflw,cofw",
            "--build-datasets",
            "--run-predictions",
            "--prediction-mode",
            "import",
            "--prediction-root",
            f"hrnet={tmp_path / 'preds'}",
            "--dry-run",
        ]
    )

    payload = _summary(root)
    assert result == 0
    assert payload["dry_run"] is True
    assert "dataset:wflw" in payload["planned_stages"]
    assert "predictions:import" in payload["planned_stages"]


def test_static_weight_pipeline_import_predictions_happy_path(tmp_path: Path) -> None:
    """Pipeline runs from an existing manifest and imported fake predictions."""
    root = tmp_path / "run"
    models = ("hrnet", "spiga")
    paths = _write_fixture_run(root, models=models)

    result = pipeline_main(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--run-predictions",
            "--prediction-mode",
            "import",
            *_prediction_root_args(paths, models),
            "--baseline-variants",
            "plain_average",
            "--weighted-variants",
            "static_weighted",
            "--failure-viewer-limit",
            "2",
        ]
    )

    payload = _summary(root)
    assert result == 0
    assert (root / "cache" / "sample" / "hrnet.npy").is_file()
    assert (root / "baseline_metrics" / "metrics.json").is_file()
    assert (root / "weights" / "static_landmark_weights.json").is_file()
    assert (root / "weights" / "static_weight_report.json").is_file()
    assert (root / "weighted_metrics" / "metrics.json").is_file()
    assert (root / "debug" / "worst_cases.json").is_file()
    assert payload["dataset_counts"] == {"fixture": 1, "total": 1}
    assert payload["cache_counts"]["predictions"] == 2
    assert payload["generated_weight_path"].endswith("static_landmark_weights.json")
    assert payload["baseline_best_model"] == "hrnet"
    assert payload["weighted_best_variant"] == "static_weighted"
    assert "static_weighted" in payload["ensemble_deltas_vs_best_single"]
    assert payload["ensemble_improved_over_best_single"] in (True, False)


def test_static_weight_pipeline_skip_flags_use_existing_cache(tmp_path: Path) -> None:
    """Skip flags allow reuse of an existing manifest and prediction cache."""
    root = tmp_path / "run"
    models = ("hrnet", "spiga")
    paths = _write_fixture_run(root, models=models)
    first = pipeline_main(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--run-predictions",
            "--prediction-mode",
            "import",
            *_prediction_root_args(paths, models),
            "--baseline-variants",
            "plain_average",
            "--weighted-variants",
            "static_weighted",
            "--skip-failure-viewer",
        ]
    )

    second = pipeline_main(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--skip-dataset-build",
            "--skip-predictions",
            "--skip-baseline",
            "--skip-failure-viewer",
            "--weighted-variants",
            "static_weighted",
        ]
    )

    payload = _summary(root)
    assert first == 0
    assert second == 0
    stage_names = [stage["name"] for stage in payload["stages"]]
    assert "cache_predictions" not in stage_names
    assert "baseline_harness" not in stage_names
    assert "compute_static_weights" in stage_names
    assert "weighted_harness" in stage_names


def test_static_weight_pipeline_reports_failed_stage(tmp_path: Path) -> None:
    """Failures are recorded with the failed stage and actionable error."""
    root = tmp_path / "run"

    result = pipeline_main(
        [
            "--output-root",
            str(root),
            "--models",
            "hrnet",
            "--skip-dataset-build",
            "--skip-predictions",
            "--skip-failure-viewer",
        ]
    )

    payload = _summary(root)
    assert result == 1
    assert "manifest not found" in payload["failed_stage_error"]
    assert payload["dataset_counts"] == {}


def test_static_weight_pipeline_continue_on_dataset_error(tmp_path: Path) -> None:
    """Dataset-specific failures can be skipped when later datasets succeed."""
    root = tmp_path / "run"
    source = tmp_path / "source"
    source.mkdir()
    assert cv2.imwrite(str(source / "sample.png"), np.zeros((16, 16, 3), dtype="uint8"))
    np.save(str(source / "sample.npy"), _points())

    result = pipeline_main(
        [
            "--output-root",
            str(root),
            "--datasets",
            "wflw,directory",
            "--build-datasets",
            "--continue-on-error",
            "--directory-source-dir",
            str(source),
            "--skip-predictions",
            "--skip-baseline",
            "--skip-failure-viewer",
            "--models",
            "hrnet",
        ]
    )

    payload = _summary(root)
    assert result == 1
    assert any(stage["name"] == "dataset:wflw" and stage["status"] == "failed" for stage in payload["stages"])
    assert any(stage["name"] == "dataset:directory" and stage["status"] == "ok" for stage in payload["stages"])
    assert payload["dataset_counts"]["directory"] == 1
