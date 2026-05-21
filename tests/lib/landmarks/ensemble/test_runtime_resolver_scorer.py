#!/usr/bin/env python3
"""Tests for learned runtime resolver scorer training and evaluation tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lib.landmarks.ensemble.runtime_resolver_scorer_data as scorer_data
from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.promoted_setup import write_best_setup
from lib.landmarks.ensemble.runtime_resolver import CandidateMetrics, CandidateRecord
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverScorer,
    load_runtime_resolver_scorer,
    write_runtime_resolver_scorer,
)
from lib.landmarks.ensemble.scorer_target_config import (
    DEFAULT_COLLAPSE_COST_PENALTY,
    DEFAULT_FAILURE_COST_PENALTY,
    DEFAULT_REGRET_NORMALIZER,
    MODEL_TYPE_LINEAR_REGRESSION,
    SCORE_SEMANTICS_PREDICTED_COST,
    SCORE_SEMANTICS_PREDICTED_RISK,
    TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
    TARGET_SELECTION_COST,
)
from lib.landmarks.ensemble.weights import save_weights
from tools.landmarks.backfill_runtime_resolver_metadata import (
    backfill_runtime_resolver_metadata,
)
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


def _write_fixture_images(manifest_path: Path) -> None:
    import cv2  # type: ignore[import-not-found]

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        image = np.full((128, 128, 3), 128, dtype="uint8")
        cv2.imwrite(str(manifest_path.parent / sample["image"]), image)


def _candidate_context(
    *,
    nme_by_candidate: dict[str, float],
    failure_by_candidate: dict[str, bool] | None = None,
    geometry_veto_reasons: dict[str, tuple[str, ...]] | None = None,
    oracle: str = "oracle",
) -> scorer_data.SampleCandidateContext:
    candidates = tuple(
        CandidateRecord(
            name=name,
            landmarks=_face(),
            is_fusion=False,
            contributing_models=(name,),
        )
        for name in ("oracle", "zero", "small", "large", "failure")
    )
    veto_reasons = geometry_veto_reasons or {}
    return scorer_data.SampleCandidateContext(
        sample_id="sample",
        dataset="test",
        source="production_validated",
        condition="frontal",
        candidates=candidates,
        metrics={
            candidate.name: CandidateMetrics(
                roll_degrees=0.0,
                yaw_degrees=0.0,
                pitch_degrees=0.0,
                geometry_veto_reasons=veto_reasons.get(candidate.name, ()),
            )
            for candidate in candidates
        },
        nme_by_candidate=nme_by_candidate,
        failure_by_candidate={
            candidate.name: bool((failure_by_candidate or {}).get(candidate.name, False))
            for candidate in candidates
        },
        runtime_bucket="frontal",
        risk_route="low_risk",
        current_policy_choice="oracle",
        oracle=oracle,
        model_predictions_available={candidate.name: True for candidate in candidates},
        roll_estimate=0.0,
        yaw_estimate=0.0,
        candidate_yaw_disagreement=0.0,
        max_disagreement_px=0.0,
        runtime_bucket_source="test",
        selected_candidate_missing_from_eval=False,
        candidate_extra_features={},
    )


def test_rows_for_context_adds_continuous_regret_targets() -> None:
    rows = scorer_data.rows_for_context(
        _candidate_context(
            nme_by_candidate={
                "oracle": 0.01,
                "zero": 0.01,
                "small": 0.015,
                "large": 0.05,
                "failure": 0.02,
            },
            failure_by_candidate={"failure": True},
            geometry_veto_reasons={"failure": ("cloud_area_too_small",)},
        )
    )
    by_name = {row.candidate_name: row for row in rows}

    assert by_name["oracle"].is_oracle is True
    assert by_name["oracle"].regret_vs_oracle == pytest.approx(0.0)
    assert by_name["oracle"].normalized_regret == pytest.approx(0.0)
    assert by_name["oracle"].selection_cost == pytest.approx(0.0)

    assert by_name["zero"].regret_vs_oracle == pytest.approx(0.0)
    assert by_name["zero"].normalized_regret == pytest.approx(0.0)
    assert by_name["zero"].candidate_failure_or_high_gap is False

    assert by_name["small"].regret_vs_oracle == pytest.approx(0.005)
    assert by_name["small"].normalized_regret == pytest.approx(0.005 / DEFAULT_REGRET_NORMALIZER)
    assert by_name["small"].large_regret_label is False

    assert by_name["large"].regret_vs_oracle == pytest.approx(0.04)
    assert by_name["large"].normalized_regret == pytest.approx(1.0)
    assert by_name["large"].large_regret_label is True
    assert by_name["large"].candidate_failure_or_high_gap is True

    assert by_name["failure"].failure_label is True
    assert by_name["failure"].candidate_failure_or_high_gap is True
    assert by_name["failure"].selection_cost == pytest.approx(
        (0.01 / DEFAULT_REGRET_NORMALIZER)
        + DEFAULT_FAILURE_COST_PENALTY
        + DEFAULT_COLLAPSE_COST_PENALTY
    )


def test_rows_for_context_rejects_missing_nme() -> None:
    context = _candidate_context(
        nme_by_candidate={
            "oracle": 0.01,
            "zero": 0.01,
            "small": 0.015,
            "large": 0.05,
        }
    )

    with pytest.raises(ValueError, match="missing NME for candidate 'failure'"):
        scorer_data.rows_for_context(context)


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
    assert metrics["target"] == TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP
    assert metrics["model_type"] == "logistic_regression"
    assert metrics["score_semantics"] == SCORE_SEMANTICS_PREDICTED_RISK
    assert metrics["higher_is_better"] is False
    assert metrics["target_stats"]["target"] == TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP
    assert metrics["target_stats"]["target_p90"] >= metrics["target_stats"]["target_p50"]
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
    training_rows = output_dir / "runtime_resolver_scorer_training_rows.csv"
    with training_rows.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.DictReader(handle))
    assert "oracle_nme" in header
    assert "regret_vs_oracle" in header
    assert "normalized_regret" in header
    assert "large_regret_label" in header
    assert "candidate_failure_or_high_gap" in header
    assert "selection_cost" in header


def test_train_runtime_resolver_scorer_supports_selection_cost_regressor(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "train_regressor"

    metrics = train_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=output_dir,
        iterations=20,
        target=TARGET_SELECTION_COST,
    )

    artifact = json.loads((output_dir / "runtime_resolver_scorer.json").read_text())
    assert metrics["target"] == TARGET_SELECTION_COST
    assert metrics["model_type"] == MODEL_TYPE_LINEAR_REGRESSION
    assert metrics["train_metrics"]["mse"] >= 0.0
    assert artifact["target"] == TARGET_SELECTION_COST
    assert artifact["model_type"] == MODEL_TYPE_LINEAR_REGRESSION
    assert artifact["score_semantics"] == SCORE_SEMANTICS_PREDICTED_COST
    assert artifact["higher_is_better"] is False
    assert artifact["version"] == "continuous_regret_v1_1"
    assert artifact["scorer_version"] == "continuous_regret_v1_1"
    assert artifact["selection_target"] == "continuous_regret"
    assert artifact["objective"] == "minimize_candidate_selection_regret"
    assert artifact["training_mode"] == "continuous_selection_cost"
    assert artifact["runtime_policy"] == "learned_quality_v1"
    assert metrics["score_semantics"] == SCORE_SEMANTICS_PREDICTED_COST
    assert metrics["higher_is_better"] is False
    assert metrics["target_stats"]["target"] == TARGET_SELECTION_COST
    assert metrics["target_stats"]["target_mean"] >= 0.0
    assert metrics["target_stats"]["target_p50"] >= 0.0
    assert metrics["target_stats"]["target_p90"] >= metrics["target_stats"]["target_p50"]
    assert metrics["target_stats"]["target_p99"] >= metrics["target_stats"]["target_p90"]
    assert 0.0 <= metrics["target_stats"]["zero_cost_rate"] <= 1.0
    assert 0.0 <= metrics["target_stats"]["large_cost_rate"] <= 1.0


def test_selection_cost_regressor_artifact_ranks_lower_cost_features_first(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "train_rank_smoke"

    train_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=output_dir,
        iterations=20,
        target=TARGET_SELECTION_COST,
    )

    scorer = load_runtime_resolver_scorer(output_dir / "runtime_resolver_scorer.json")
    low_cost_score = scorer.score_feature_map({"candidate_name=hrnet": 1.0})
    high_cost_score = scorer.score_feature_map({"candidate_name=spiga": 1.0})

    assert scorer.model_type == MODEL_TYPE_LINEAR_REGRESSION
    assert scorer.score_semantics == SCORE_SEMANTICS_PREDICTED_COST
    assert scorer.higher_is_better is False
    assert low_cost_score < high_cost_score


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

    assert report["status"] == "fail"
    assert "scorer_mean_nme_not_better_than_static_downweight" in report["failed_gates"]
    assert report["heldout_eval"] is False
    assert report["runtime_policy"] == "learned_quality_v1"
    assert report["promoted_scorer_label"] == "current_binary_logistic_scorer"
    assert "learned_quality_v1" not in report
    assert report["current_binary_logistic_scorer"]["pick_counts"] == {"hrnet": 2}
    assert report["production_only_policy_metrics"]["sample_count"] == 2
    assert "learned_quality_v1" not in report["production_only_policy_metrics"]
    assert report["production_only_policy_metrics"]["current_binary_logistic_scorer"][
        "pick_counts"
    ] == {"hrnet": 2}
    assert report["gt_hard_only_policy_metrics"]["sample_count"] == 0
    assert report["best_single"]["candidate"] == "hrnet"
    assert (tmp_path / "eval" / "scorer_policy_report.csv").is_file()
    assert (tmp_path / "eval" / "scorer_feature_importance.csv").is_file()


def test_evaluate_runtime_resolver_scorer_compares_binary_and_continuous(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    binary_scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=hrnet", "candidate_name=spiga"),
            coefficients=(-5.0, 5.0),
            intercept=0.0,
        ),
        tmp_path / "binary_runtime_resolver_scorer.json",
    )
    continuous_scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=hrnet", "candidate_name=spiga"),
            coefficients=(-1.0, 1.0),
            intercept=0.0,
            model_type=MODEL_TYPE_LINEAR_REGRESSION,
            target=TARGET_SELECTION_COST,
            score_semantics=SCORE_SEMANTICS_PREDICTED_COST,
            higher_is_better=False,
            version="continuous_regret_v1_1",
            selection_target="continuous_regret",
        ),
        tmp_path / "continuous_runtime_resolver_scorer.json",
    )

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        scorer_path=continuous_scorer_path,
        binary_scorer_path=binary_scorer_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=tmp_path / "eval_compare",
    )

    assert report["primary_scorer_policy"] == "scorer_version"
    assert report["runtime_policy"] == "learned_quality_v1"
    assert report["promoted_scorer_version"] == "continuous_regret_v1_1"
    assert report["promoted_scorer_target"] == TARGET_SELECTION_COST
    assert report["promoted_scorer_label"] == "scorer_version"
    assert report["scorer_target"] == TARGET_SELECTION_COST
    assert report["scorer_model_type"] == MODEL_TYPE_LINEAR_REGRESSION
    assert report["scorer_comparison"]["uses_same_contexts"] is True
    assert report["scorer_comparison"]["uses_same_candidates"] is True
    assert report["scorer_comparison"]["context_count"] == report["sample_count"]
    assert report["scorer_version"]["pick_counts"] == {"hrnet": 2}
    assert "learned_quality_v1" not in report
    assert report["current_binary_logistic_scorer"]["pick_counts"] == {"hrnet": 2}
    assert "static_weighted_downweight" in report
    assert "oracle" in report
    assert report["production_only_policy_metrics"]["scorer_version"]["pick_counts"] == {
        "hrnet": 2
    }
    assert report["production_only_policy_metrics"]["current_binary_logistic_scorer"][
        "pick_counts"
    ] == {"hrnet": 2}


def test_evaluate_runtime_resolver_scorer_filters_to_eval_split(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=hrnet",),
            coefficients=(-5.0,),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )
    eval_split = tmp_path / "runtime_resolver_scorer_eval_rows.csv"
    eval_split.write_text(
        "sample_id,dataset,condition,candidate_name\n"
        "s1,production_validated,profile_left,hrnet\n"
        "s1,production_validated,profile_left,spiga\n",
        encoding="utf-8",
    )

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        scorer_path=scorer_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=tmp_path / "heldout_eval",
        eval_split=eval_split,
    )

    assert report["heldout_eval"] is True
    assert report["eval_split"] == str(eval_split)
    assert report["sample_count"] == 1
    assert report["production_only_policy_metrics"]["sample_count"] == 1
    assert (tmp_path / "heldout_eval" / "scorer_policy_eval_report.json").is_file()


def test_evaluate_runtime_resolver_scorer_blocks_safe_fallback_without_score_delta(
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

    assert report["current_binary_logistic_scorer"]["pick_counts"] == {"static_weighted": 2}
    assert report["safe_fallback_count"] == 0
    assert report["safe_fallback_min_delta"] == 0.05
    assert report["fallback_impact"]["count_with_rejected_candidate"] == 0


def test_evaluate_runtime_resolver_scorer_refuses_gt_hard_without_sidecar(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    save_weights(
        weights_path,
        {"hrnet": [1.0 / 3.0] * 68, "spiga": [1.0 / 3.0] * 68, "orformer": [1.0 / 3.0] * 68},
    )
    cache = DiskPredictionCache(cache_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample["dataset"] = "aflw2000_3d"
        sample["condition"] = "rolled_large_yaw_right"
        sample.pop("metadata", None)
        sample_id = sample["sample_id"]
        for model, offset in {"hrnet": 0.0, "spiga": 40.0, "orformer": 20.0}.items():
            cache.write(
                sample_id,
                LandmarkPrediction(_face(offset), model_name=model),
                checkpoint="test",
            )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=static_weighted", "candidate_name=orformer"),
            coefficients=(-5.0, -4.0),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )

    with pytest.raises(RuntimeError, match="GT-hard sample missing stored resolver metadata"):
        evaluate_runtime_resolver_scorer(
            gt_manifest=manifest_path,
            gt_cache_dir=cache_dir,
            production_manifest=None,
            production_cache_dir=None,
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


def test_backfill_runtime_resolver_metadata_writes_image_aware_source(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    _write_fixture_images(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample.pop("metadata", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    setup_path = tmp_path / "best_setup.json"
    write_best_setup(
        setup_path,
        candidate_id="test",
        models=("hrnet", "spiga", "orformer"),
        strategy="static_weighted",
        outlier_threshold=None,
        weight_generator_name="test",
        weight_generator_params={},
        crop_scale=1.6,
        bbox_source="manifest",
        regression_epsilon_nme=0.001,
        reproducibility={},
        fit={},
        selection_metrics={},
        weights_path=weights_path.name,
    )
    output_path = tmp_path / "manifest.runtime_metadata.json"

    report = backfill_runtime_resolver_metadata(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        setup_path=setup_path,
        output_path=output_path,
        models=("hrnet", "spiga", "orformer"),
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["updated_count"] == 2
    for sample in output["samples"]:
        metadata = sample["metadata"]["landmark_ensemble"]
        assert metadata["runtime_bucket_source"] == "image_aware_backfill"
        assert metadata["runtime_bucket"]
        assert metadata["bucket"] == metadata["runtime_bucket"]
        assert "runtime_bucket_features" in metadata


def test_load_contexts_can_backfill_image_aware_runtime_metadata(
    tmp_path: Path,
) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    _write_fixture_images(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample.pop("metadata", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    contexts = scorer_data.load_contexts(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "orformer", "static_weighted_downweight"),
        allow_image_backfill=True,
    )

    assert len(contexts) == 2
    assert {context.runtime_bucket_source for context in contexts} == {"image_aware_backfill"}


def test_gt_hard_uses_stored_resolver_metadata(tmp_path: Path) -> None:
    sidecar = tmp_path / "resolver_metadata.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "sample_id": "AFLW2000/image03123",
                "face_index": 0,
                "condition": "profile_right",
                "landmark_ensemble": {"runtime_bucket": "profile_right"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = scorer_data.load_resolver_metadata(sidecar)
    row = metadata[("AFLW2000/image03123", 0)]

    assert scorer_data.runtime_bucket_from_resolver_metadata(row) == "profile_right"


def test_gt_hard_missing_metadata_refuses_image_backfill(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample["dataset"] = "aflw2000_3d"
        sample["condition"] = "profile_right"
        sample.pop("metadata", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="GT-hard sample missing stored resolver metadata"):
        scorer_data.load_contexts(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            weights_path=weights_path,
            candidates=("hrnet", "spiga", "static_weighted_downweight"),
            source="gt_hard",
            resolver_metadata={},
            allow_image_backfill=True,
        )


def test_gt_hard_load_contexts_reads_frozen_resolver_metadata(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample["dataset"] = "aflw2000_3d"
        sample["condition"] = "profile_right"
        sample.pop("metadata", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    sidecar = tmp_path / "resolver_metadata.jsonl"
    with sidecar.open("w", encoding="utf-8") as handle:
        for sample in payload["samples"]:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "face_index": 0,
                        "condition": "profile_right",
                        "landmark_ensemble": {
                            "runtime_bucket": "profile_right",
                            "selected_candidate": "hrnet",
                            "runtime_bucket_source": "stored_manifest_landmark_ensemble",
                        },
                    }
                )
                + "\n"
            )

    contexts = scorer_data.load_contexts(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        source="gt_hard",
        resolver_metadata=scorer_data.load_resolver_metadata(sidecar),
        allow_image_backfill=True,
    )

    assert {context.runtime_bucket for context in contexts} == {"profile_right"}
    assert {context.runtime_bucket_source for context in contexts} == {
        "stored_manifest_landmark_ensemble"
    }
    assert {context.current_policy_choice for context in contexts} == {"hrnet"}


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
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample.pop("metadata", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
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
