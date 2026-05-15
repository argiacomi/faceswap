#!/usr/bin/env python3
"""CLI integration tests for ``search_ensemble_setup`` (issue #69)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.eval.splits import (
    SplitRatios,
    save_split_file,
    split_manifest_samples,
)
from lib.landmarks.schema import LandmarkPrediction
from tools.landmarks.search_ensemble_setup import main as search_main


def _truth_points() -> np.ndarray:
    return np.stack(
        (
            np.linspace(20.0, 70.0, LANDMARK_COUNT, dtype="float32"),
            np.linspace(25.0, 75.0, LANDMARK_COUNT, dtype="float32"),
        ),
        axis=1,
    )


def _prediction(noise: float) -> np.ndarray:
    return _truth_points() + noise


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create a manifest, ground truth, prediction cache, and splits.json."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    truth = dataset_dir / "truth.npy"
    np.save(str(truth), _truth_points())
    cache_dir = tmp_path / "cache"
    cache = DiskPredictionCache(cache_dir)

    samples: list[dict[str, str]] = []
    for bucket, count in {"fixture:clean": 9, "fixture:occluded": 9}.items():
        dataset, condition = bucket.split(":", 1)
        for idx in range(count):
            sid = f"{bucket}-{idx:03d}"
            samples.append(
                {
                    "sample_id": sid,
                    "dataset": dataset,
                    "condition": condition,
                    "image": "image.png",
                    "landmarks": "truth.npy",
                }
            )
            for model, noise in (("hrnet", 0.5), ("spiga", 1.0), ("orformer", 2.5)):
                cache.write(sid, LandmarkPrediction(_prediction(noise), model_name=model))
    manifest = dataset_dir / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    ratios = SplitRatios(fit=0.5, select=0.25, report=0.25)
    assignment, diagnostics = split_manifest_samples(
        samples, mode="scenario-stratified", ratios=ratios, seed=42
    )
    splits_path = tmp_path / "splits.json"
    save_split_file(
        splits_path,
        assignment,
        mode="scenario-stratified",
        ratios=ratios,
        seed=42,
        diagnostics=diagnostics,
    )
    return {
        "manifest": manifest,
        "cache": cache_dir,
        "splits": splits_path,
    }


def _run_search(fixture: dict[str, Path], output_dir: Path, *extra: str) -> int:
    return search_main(
        [
            "--manifest",
            str(fixture["manifest"]),
            "--cache-dir",
            str(fixture["cache"]),
            "--splits",
            str(fixture["splits"]),
            "--models",
            "hrnet,spiga,orformer",
            "--model-subsets",
            "all",
            "--weight-generators",
            "equal,inverse_mean_error",
            "--strategies",
            "static_weighted,static_weighted_downweight",
            "--outlier-thresholds",
            "3.5",
            "--output-dir",
            str(output_dir),
            *extra,
        ]
    )


def test_search_writes_all_required_output_files(tmp_path: Path) -> None:
    """Search emits candidate logs, best setup/weights, and a promotion report."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    exit_code = _run_search(fixture, output_dir)

    assert exit_code == 0
    for filename in (
        "candidate_results.csv",
        "candidate_results.json",
        "best_setup.json",
        "best_weights.json",
        "promotion_report.md",
    ):
        path = output_dir / filename
        assert path.is_file(), f"missing required output {filename}"


def test_search_best_setup_payload_includes_required_provenance(tmp_path: Path) -> None:
    """best_setup.json includes candidate id, strategy, weights path, and split hash."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    assert _run_search(fixture, output_dir) == 0

    setup = json.loads((output_dir / "best_setup.json").read_text(encoding="utf-8"))
    assert setup["artifact_schema_version"] == 1
    assert setup["schema"] == "2d_68"
    assert setup["candidate_id"].startswith("sha256:")
    assert setup["weights_path"] == "best_weights.json"
    assert setup["strategy"] in {
        "static_weighted",
        "static_weighted_downweight",
    }
    assert setup["weight_generator"]["name"] in {"equal", "inverse_mean_error"}
    assert "split_assignment_hash" in setup["reproducibility"]
    assert "candidate_search_seed" in setup["reproducibility"]
    assert setup["regression_epsilon_nme"] == pytest.approx(0.001)
    assert setup["selection_metrics"]["sample_count"] >= 1
    assert setup["report_metrics"]["sample_count"] >= 1
    assert "evaluation_log_path" in setup


def test_search_best_weights_schema_v1(tmp_path: Path) -> None:
    """best_weights.json conforms to the v1 schema: 68 non-negative columns per model."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    assert _run_search(fixture, output_dir) == 0

    payload = json.loads((output_dir / "best_weights.json").read_text(encoding="utf-8"))
    assert payload["artifact_schema_version"] == 1
    assert payload["schema"] == "2d_68"
    assert set(payload["weights"]) == set(payload["models"])
    for model in payload["models"]:
        column = payload["weights"][model]
        assert len(column) == LANDMARK_COUNT
        assert all(value >= 0 for value in column)
    column_sums = [
        sum(payload["weights"][model][landmark] for model in payload["models"])
        for landmark in range(LANDMARK_COUNT)
    ]
    for total in column_sums:
        assert total == pytest.approx(1.0)


def test_search_candidate_results_csv_and_json_match(tmp_path: Path) -> None:
    """Every CSV row corresponds to an entry in the JSON candidate log."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    assert _run_search(fixture, output_dir) == 0

    csv_lines = (output_dir / "candidate_results.csv").read_text(encoding="utf-8").splitlines()
    json_payload = json.loads((output_dir / "candidate_results.json").read_text(encoding="utf-8"))

    # Header + at least one data row.
    assert len(csv_lines) >= 2
    assert len(json_payload["candidates"]) == len(csv_lines) - 1


def test_search_candidate_ids_are_deterministic(tmp_path: Path) -> None:
    """Two runs against the same fixture produce identical candidate IDs."""
    fixture = _build_fixture(tmp_path)
    output_a = tmp_path / "search-a"
    output_b = tmp_path / "search-b"

    assert _run_search(fixture, output_a) == 0
    assert _run_search(fixture, output_b) == 0

    a_ids = sorted(
        candidate["candidate_id"]
        for candidate in json.loads(
            (output_a / "candidate_results.json").read_text(encoding="utf-8")
        )["candidates"]
    )
    b_ids = sorted(
        candidate["candidate_id"]
        for candidate in json.loads(
            (output_b / "candidate_results.json").read_text(encoding="utf-8")
        )["candidates"]
    )
    assert a_ids == b_ids


def test_search_promotion_report_mentions_winner_and_report_metrics(tmp_path: Path) -> None:
    """The Markdown report names the winning strategy and held-out metrics."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    assert _run_search(fixture, output_dir) == 0

    report_text = (output_dir / "promotion_report.md").read_text(encoding="utf-8")
    assert "Winner" in report_text
    assert "report_nme" in report_text
    setup = json.loads((output_dir / "best_setup.json").read_text(encoding="utf-8"))
    assert setup["strategy"] in report_text


def test_search_rejects_invalid_model_subset_preset(tmp_path: Path) -> None:
    """Unknown subset names cause an exit before any artifact is written."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    with pytest.raises(ValueError, match="unknown"):
        search_main(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--splits",
                str(fixture["splits"]),
                "--models",
                "hrnet,spiga,orformer",
                "--model-subsets",
                "quadruples",
                "--weight-generators",
                "inverse_mean_error",
                "--strategies",
                "static_weighted",
                "--output-dir",
                str(output_dir),
            ]
        )
    assert not (output_dir / "best_setup.json").exists()


def test_search_does_not_modify_prediction_cache(tmp_path: Path) -> None:
    """Cache-only constraint: search must not write new files into the cache dir."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"
    before = sorted(p.name for p in fixture["cache"].rglob("*") if p.is_file())

    assert _run_search(fixture, output_dir) == 0

    after = sorted(p.name for p in fixture["cache"].rglob("*") if p.is_file())
    assert before == after


def test_search_selects_lowest_score_winner(tmp_path: Path) -> None:
    """The winning candidate in best_setup.json must have the lowest score in the log."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    assert _run_search(fixture, output_dir) == 0

    payload = json.loads((output_dir / "candidate_results.json").read_text(encoding="utf-8"))
    scores = [candidate["score"] for candidate in payload["candidates"]]
    assert payload["candidates"][0]["score"] == min(scores)
    setup = json.loads((output_dir / "best_setup.json").read_text(encoding="utf-8"))
    assert setup["candidate_id"] == payload["candidates"][0]["candidate_id"]
