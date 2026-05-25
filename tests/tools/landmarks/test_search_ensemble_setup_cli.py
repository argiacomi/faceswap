#!/usr/bin/env python3
"""CLI integration tests for ``search_ensemble_setup`` (issue #69)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tools.landmarks.search_ensemble_setup as search_cli
from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from lib.landmarks.evaluation.splits import (
    SplitRatios,
    save_split_file,
    split_manifest_samples,
)
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


def test_search_with_geometry_gate_writes_no_promotion(tmp_path: Path) -> None:
    """A geometry-improvement gate that no candidate meets writes ``no_promotion.json``."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
            "--allow-single-model-baselines",
            "--include-geometry-metrics",
            "--require-geometry-improvement",
            # Force an impossible bar so no ensemble beats the baseline.
            "--max-catastrophic-geometry-failure-rate",
            "-1.0",
        ]
    )

    assert exit_code == 1
    no_promo = json.loads((output_dir / "no_promotion.json").read_text(encoding="utf-8"))
    assert no_promo["status"] == "no_promotion"
    assert no_promo["top_failing_candidates"]
    assert not (output_dir / "best_setup.json").exists()


def test_search_fans_out_candidates_across_crop_scales(tmp_path: Path) -> None:
    """``--crop-scale 1.4,1.8`` enumerates one candidate per scale.

    The fanout proves crop_scale is now a real search dimension: each scale
    produces distinct candidate_ids (the hash includes ``crop_scale``) and
    the recorded ``crop_scale`` column reflects the chosen value rather than
    a single CLI-default constant.
    """
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"
    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--crop-scale",
            "1.4,1.8",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    payload = json.loads((output_dir / "candidate_results.json").read_text(encoding="utf-8"))
    crop_scales = sorted({float(c["crop_scale"]) for c in payload["candidates"]})
    assert crop_scales == [1.4, 1.8]

    csv_lines = (output_dir / "candidate_results.csv").read_text(encoding="utf-8").splitlines()
    header = csv_lines[0].split(",")
    assert "crop_scale" in header
    crop_scale_col = header.index("crop_scale")
    csv_scales = sorted({float(line.split(",")[crop_scale_col]) for line in csv_lines[1:]})
    assert csv_scales == [1.4, 1.8]


def test_search_persists_per_candidate_geometry_payload(tmp_path: Path) -> None:
    """``candidate_results.{csv,json}`` carry per-bucket geometry diagnostics.

    Once any geometry gate fires, the search builds per-candidate geometry
    aggregates and the worst-bucket regression summary; persisting both so
    downstream consumers don't have to recompute them is the contract this
    test locks in.
    """
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"
    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
            "--allow-single-model-baselines",
            "--max-catastrophic-geometry-failure-rate",
            "1.0",
            "--min-hull-iou",
            "0.0",
        ]
    )
    assert exit_code == 0

    payload = json.loads((output_dir / "candidate_results.json").read_text(encoding="utf-8"))
    assert "geometry_per_candidate" in payload
    geom = payload["geometry_per_candidate"]
    assert "__baseline__" in geom
    # Drop the baseline metadata key; every remaining entry is a candidate.
    candidate_ids = [key for key in geom if key != "__baseline__"]
    assert candidate_ids
    sample = geom[candidate_ids[0]]
    required_fields = {
        "overall_score",
        "catastrophic_failure_rate",
        "p95_translation_normalized",
        "p95_roi_center_normalized",
        "mean_hull_iou",
        "p05_hull_iou",
        "max_bucket_regression_score",
        "worst_bucket",
        "worst_bucket_score",
        "worst_bucket_baseline_score",
        "per_bucket",
    }
    assert required_fields.issubset(sample), sorted(required_fields - set(sample))
    assert sample["per_bucket"]
    bucket_payload = next(iter(sample["per_bucket"].values()))
    required_bucket_fields = {
        "overall_score",
        "catastrophic_failure_rate",
        "p95_translation_normalized",
        "p95_roi_center_normalized",
        "mean_hull_iou",
        "p05_hull_iou",
    }
    assert required_bucket_fields.issubset(bucket_payload), sorted(
        required_bucket_fields - set(bucket_payload)
    )

    csv_text = (output_dir / "candidate_results.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0].split(",")
    for column in (
        "geometry_overall_score",
        "geometry_max_bucket_regression_score",
        "geometry_worst_bucket",
    ):
        assert column in header, header


def test_search_with_geometry_gate_promotes_when_thresholds_met(tmp_path: Path) -> None:
    """A relaxed geometry gate still finds a passing candidate and writes the setup."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
            "--allow-single-model-baselines",
            "--max-catastrophic-geometry-failure-rate",
            "1.0",
            "--min-hull-iou",
            "0.0",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "best_setup.json").is_file()
    setup = json.loads((output_dir / "best_setup.json").read_text(encoding="utf-8"))
    assert setup["candidate_id"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Follow-up fixes (alignment_geometry_v1 ranking, bbox plumbing, no_promotion
# gate config)
# ---------------------------------------------------------------------------


def test_search_geometry_objective_reranks_by_geometry_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``alignment_geometry_v1`` reorders multiple candidates by geometry score."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"
    pre_rank_order: list[str] = []

    def fake_geometry_aggregate(
        result: object,
        *,
        context: object,
        aligned_size: int,
        region_failure_threshold: float,
    ) -> search_cli.GeometryAggregate:
        del context, aligned_size, region_failure_threshold
        candidate_id = result.candidate_id  # type: ignore[attr-defined]
        pre_rank_order.append(candidate_id)
        # Force the pre-rank last candidate to become best by geometry. This
        # proves the geometry objective is not simply preserving NME order.
        score = float(100 - len(pre_rank_order))
        return search_cli.GeometryAggregate(
            label=candidate_id,
            sample_count=1,
            overall_score=score,
            catastrophic_failure_rate=0.0,
            mean_scale_delta=0.0,
            mean_relative_scale_delta=0.0,
            mean_rotation_degrees_delta=0.0,
            mean_translation_normalized=score,
            p95_translation_normalized=score,
            p95_rotation_degrees_delta=0.0,
            p95_relative_scale_delta=0.0,
            p95_roll_degrees_delta=0.0,
            mean_roi_iou=1.0,
            p05_roi_iou=1.0,
            mean_roi_center_normalized=0.0,
            p95_roi_center_normalized=0.0,
            mean_hull_iou=1.0,
            p05_hull_iou=1.0,
            mean_pitch_delta_degrees=0.0,
            mean_yaw_delta_degrees=0.0,
            mean_roll_delta_degrees=0.0,
            mean_average_distance_delta=0.0,
        )

    monkeypatch.setattr(search_cli, "_evaluate_candidate_geometry", fake_geometry_aggregate)

    rc = search_main(
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
            "--objective",
            "alignment_geometry_v1",
            "--include-geometry-metrics",
            "--no-progress",
        ]
    )
    assert rc == 0
    assert len(pre_rank_order) >= 2
    payload = json.loads((output_dir / "candidate_results.json").read_text(encoding="utf-8"))
    ranked_ids = [candidate["candidate_id"] for candidate in payload["candidates"]]
    assert ranked_ids != pre_rank_order
    assert ranked_ids[0] == pre_rank_order[-1]
    assert ranked_ids[-1] == pre_rank_order[0]
    setup = json.loads((output_dir / "best_setup.json").read_text(encoding="utf-8"))
    assert setup["candidate_id"] == ranked_ids[0]


def test_search_geometry_objective_auto_enables_geometry_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Selecting the geometry objective auto-enables geometry evaluation."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    rc = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--objective",
            "alignment_geometry_v1",
        ]
    )

    assert rc == 0
    assert (output_dir / "best_setup.json").is_file()
    assert (output_dir / "candidate_results.json").is_file()
    captured = capsys.readouterr()
    assert "START geometry_evaluate_candidates" in captured.err
    assert "Re-ranking" in captured.err
    assert "alignment_geometry_v1" in captured.err


def test_search_geometry_objective_allows_nme_fallback_when_opted_in(
    tmp_path: Path,
) -> None:
    """``--allow-nme-only-promotion`` lets the geometry objective fall back to NME."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    rc = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--objective",
            "alignment_geometry_v1",
            "--allow-nme-only-promotion",
        ]
    )
    assert rc == 0
    assert (output_dir / "best_setup.json").is_file()


def test_no_promotion_payload_includes_geometry_gate_fields(tmp_path: Path) -> None:
    """``no_promotion.json`` reproduces every configured geometry gate."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    rc = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
            "--allow-single-model-baselines",
            "--include-geometry-metrics",
            "--require-geometry-improvement",
            "--max-catastrophic-geometry-failure-rate",
            "-1.0",
            "--max-p95-transform-error",
            "0.0",
            "--max-p95-crop-center-error",
            "0.0",
            "--max-p95-roll-error",
            "0.0",
            "--min-hull-iou",
            "2.0",
            "--max-hard-slice-regression-rate",
            "0.0",
        ]
    )
    assert rc == 1
    no_promo = json.loads((output_dir / "no_promotion.json").read_text(encoding="utf-8"))
    gate_config = no_promo["gate_config"]
    for field in (
        "require_geometry_improvement",
        "max_catastrophic_geometry_failure_rate",
        "max_p95_transform_error",
        "max_p95_crop_center_error",
        "max_p95_roll_error",
        "min_hull_iou",
        "max_hard_slice_regression_rate",
        "allow_nme_only_promotion",
    ):
        assert field in gate_config, f"missing geometry gate field {field!r}"
    assert gate_config["require_geometry_improvement"] is True
    assert gate_config["max_catastrophic_geometry_failure_rate"] == -1.0
    assert gate_config["min_hull_iou"] == 2.0


def test_search_profile_gate_runs_with_cached_context(tmp_path: Path) -> None:
    """Profile gate path completes end-to-end after the context-cache refactor.

    Locks the new ``_build_profile_context`` shape: the per-candidate
    aggregate now reads truth/cache through the pre-built context, so any
    bug in the context wiring would surface as a crash or wrong-shape
    aggregate when the profile-eval stage runs.
    """
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"
    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
            "--allow-single-model-baselines",
            # Activate the profile gate so _profile_eval (and the cached
            # context underneath it) actually runs in this test.
            "--require-profile-improvement",
        ]
    )
    # The fixture predictions may or may not beat the baseline profile
    # score; what we lock here is that the cached-context path completes
    # without error and emits a candidate_results.json. Either gate
    # outcome (promote / no_promotion) proves the wiring is intact.
    assert exit_code in (0, 1)
    assert (output_dir / "candidate_results.json").is_file()


# ---------------------------------------------------------------------------
# Single-model promotion loophole regression tests
# ---------------------------------------------------------------------------


def _force_single_model_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force a single-model baseline to top the rank so the no-gates branch
    has to consider it for promotion."""
    real_run = search_cli.run_candidate_search

    def _ranked(*args: object, **kwargs: object) -> list[search_cli.CandidateResult]:
        results = list(real_run(*args, **kwargs))  # type: ignore[arg-type]
        # Push every single-model baseline ahead of the fusion ensembles
        # so ``results[0]`` would be a baseline without the guard.
        results.sort(key=lambda result: (0 if result.is_single_model_baseline else 1,))
        return results

    monkeypatch.setattr(search_cli, "run_candidate_search", _ranked)


def test_search_no_gates_blocks_single_model_promotion_without_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When baselines get enumerated for comparison and no promotion gates
    are configured, the no-gates fallback used to silently promote whichever
    candidate sat first in the ranked list — including single-model
    baselines. The guarded fallback must refuse single-model promotion
    unless ``--allow-single-model-promotion`` was passed."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"
    _force_single_model_winner(monkeypatch)

    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
        ]
    )

    assert exit_code == 0
    setup = json.loads((output_dir / "best_setup.json").read_text(encoding="utf-8"))
    assert len(setup["models"]) > 1, (
        "no-gates fallback promoted a single-model baseline despite "
        "--allow-single-model-promotion not being set"
    )


def test_search_no_gates_promotes_single_model_when_flag_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--allow-single-model-promotion`` explicitly opts into single-model
    promotion, so the no-gates fallback may write a single-model best_setup
    when a baseline tops the ranking."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"
    _force_single_model_winner(monkeypatch)

    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
            "--allow-single-model-promotion",
        ]
    )

    assert exit_code == 0
    setup = json.loads((output_dir / "best_setup.json").read_text(encoding="utf-8"))
    assert len(setup["models"]) == 1
    assert not (output_dir / "no_promotion.json").exists()


def test_search_no_gates_writes_no_promotion_when_only_singles_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every evaluated candidate is a single-model baseline and the
    promotion flag is not set, the no-gates fallback writes a
    ``no_promotion.json`` explaining that single-model promotion is
    disabled, instead of silently promoting one."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    real_run = search_cli.run_candidate_search

    def _singles_only(*args: object, **kwargs: object) -> list[search_cli.CandidateResult]:
        results = list(real_run(*args, **kwargs))  # type: ignore[arg-type]
        return [result for result in results if result.is_single_model_baseline]

    monkeypatch.setattr(search_cli, "run_candidate_search", _singles_only)

    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--include-single-model-baselines",
        ]
    )

    assert exit_code == 1
    assert not (output_dir / "best_setup.json").exists()
    payload = json.loads((output_dir / "no_promotion.json").read_text(encoding="utf-8"))
    assert payload["status"] == "no_promotion"
    assert "single-model" in payload["reason"]


def _candidate_result_stub(
    *,
    strategy: str,
    models: tuple[str, ...],
    is_single_model_baseline: bool,
) -> search_cli.CandidateResult:
    """Minimal CandidateResult for testing the production-policy resolver.

    The resolver only inspects ``candidate.strategy`` /
    ``candidate.models`` and ``is_single_model_baseline``; the rest of the
    record is unused, so we provide trivial values for the other fields.
    Strategies that require an outlier threshold (e.g.
    ``static_weighted_downweight``) get a default 3.5 so ``Candidate``
    construction validates.
    """
    from lib.landmarks.ensemble.strategies import strategy_uses_threshold
    from lib.landmarks.search.candidate_search import CandidateMetrics
    from lib.landmarks.search.candidates import Candidate

    candidate = Candidate(
        models=models,
        weight_generator="equal",
        strategy=strategy,
        outlier_threshold=3.5 if strategy_uses_threshold(strategy) else None,
    )
    metrics = CandidateMetrics(
        sample_count=1,
        overall_nme=0.0,
        failure_rate=0.0,
        auc=0.0,
        regression_rate_vs_best_single=0.0,
        bucket_regression_rate_vs_best_single=0.0,
    )
    return search_cli.CandidateResult(
        candidate=candidate,
        candidate_id="sha256:test",
        weights={model: [1.0] * LANDMARK_COUNT for model in models},
        weights_hash="sha256:weights",
        score=0.0,
        objective="default",
        regression_epsilon_nme=0.0,
        metrics=metrics,
        is_single_model_baseline=is_single_model_baseline,
    )


def test_resolve_production_policy_auto_picks_ensemble_strategy() -> None:
    """``--production-policy auto`` (the new default) anchors the production
    gate to whichever candidate the search promoted. For an ensemble winner
    that is the fusion strategy name, so the gate's per-bucket regressions
    match the artifact that's about to land on disk instead of reporting
    the unrelated ``bucket_aware_veto`` runtime policy."""
    winner = _candidate_result_stub(
        strategy="static_weighted_downweight",
        models=("hrnet", "spiga", "orformer"),
        is_single_model_baseline=False,
    )

    assert (
        search_cli._resolve_production_policy("auto", winner)
        == "candidate:static_weighted_downweight"
    )


def test_resolve_production_policy_auto_picks_single_model_for_baseline() -> None:
    """Single-model baselines fuse via ``plain_average``, but the production
    gate evaluates per-candidate so the resolver picks the model name
    instead — ``candidate:plain_average`` would point at the multi-model
    averaging strategy, not the baseline."""
    winner = _candidate_result_stub(
        strategy="plain_average",
        models=("hrnet",),
        is_single_model_baseline=True,
    )

    assert search_cli._resolve_production_policy("auto", winner) == "candidate:hrnet"


def test_resolve_production_policy_passes_explicit_policy_through() -> None:
    """An explicit ``--production-policy`` always wins so users who really
    want runtime-policy validation keep the old behavior."""
    winner = _candidate_result_stub(
        strategy="static_weighted_downweight",
        models=("hrnet", "spiga"),
        is_single_model_baseline=False,
    )

    assert (
        search_cli._resolve_production_policy("bucket_aware_veto", winner) == "bucket_aware_veto"
    )
    assert (
        search_cli._resolve_production_policy("candidate:weighted_median", winner)
        == "candidate:weighted_median"
    )


def test_production_gate_failure_does_not_block_artifacts_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A production-gate failure used to make the search return 1 and skip
    artifact writes, which broke the resolver pipeline at candidate_search
    even though the downstream ``production_promotion_check`` stage is the
    authoritative gate. The new default writes ``best_setup.json`` and the
    production report side-by-side and exits zero; ``--production-gate-blocks``
    restores the old fail-fast behavior."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    def _fake_gate(**_kwargs: object) -> dict[str, object]:
        report_path = _kwargs["output_dir"] / "production_promotion_report.json"
        payload = {
            "status": "fail",
            "failed_gates": ["bucket_profile_left_mean_regresses_vs_best_single"],
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(search_cli, "run_production_promotion_gate", _fake_gate)

    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--production-manifest",
            str(fixture["manifest"]),
            "--production-cache-dir",
            str(fixture["cache"]),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "best_setup.json").is_file()
    assert (output_dir / "best_weights.json").is_file()
    assert (output_dir / "production_promotion_report.json").is_file()


def test_production_gate_failure_blocks_when_explicitly_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--production-gate-blocks`` keeps the old fail-fast contract so
    standalone users who treat this script's production gate as
    authoritative can still rely on a non-zero exit."""
    fixture = _build_fixture(tmp_path)
    output_dir = tmp_path / "search"

    def _fake_gate(**_kwargs: object) -> dict[str, object]:
        report_path = _kwargs["output_dir"] / "production_promotion_report.json"
        payload = {"status": "fail", "failed_gates": ["bucket_x"]}
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(search_cli, "run_production_promotion_gate", _fake_gate)

    exit_code = search_main(
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
            "inverse_mean_error",
            "--strategies",
            "static_weighted",
            "--output-dir",
            str(output_dir),
            "--production-manifest",
            str(fixture["manifest"]),
            "--production-cache-dir",
            str(fixture["cache"]),
            "--production-gate-blocks",
        ]
    )

    assert exit_code == 1
    assert not (output_dir / "best_setup.json").exists()
    assert (output_dir / "production_promotion_report.json").is_file()
