#!/usr/bin/env python3
"""Tests for learned runtime resolver scorer training and evaluation tools."""

from __future__ import annotations

import csv
import json
import sys
import typing as T
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lib.landmarks.ensemble.runtime_features as runtime_features
import lib.landmarks.ensemble.runtime_resolver_scorer as runtime_resolver_scorer_impl
import lib.landmarks.ensemble.runtime_resolver_scorer_data as scorer_data
import lib.landmarks.ensemble.scorer_eval as scorer_eval_impl
import lib.landmarks.ensemble.scorer_training as scorer_training
from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.promoted_setup import write_best_setup, write_best_weights
from lib.landmarks.ensemble.runtime_features import (
    RUNTIME_FEATURE_CONTRACT_VERSION,
    RUNTIME_PREFERRED_FEATURE_ORDER,
    candidate_feature_map,
    runtime_candidate_feature_maps,
    runtime_feature_order,
)
from lib.landmarks.ensemble.runtime_resolver import (
    CandidateMetrics,
    CandidateRecord,
    bucket_candidate_name,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import load_runtime_resolver_scorer
from lib.landmarks.ensemble.scorer_target_config import (
    DEFAULT_COLLAPSE_COST_PENALTY,
    DEFAULT_FAILURE_COST_PENALTY,
    DEFAULT_REGRET_NORMALIZER,
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    SCORE_SEMANTICS_PREDICTED_COST,
    TARGET_TRANSFORM_REGRET_V3,
)
from lib.landmarks.ensemble.weights import save_weights
from tests.lib.landmarks.ensemble.scorer_test_utils import LinearTestScorer
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
from tools.landmarks.train_runtime_resolver_scorer import (
    SCORER_V3_ARTIFACT,
    train_runtime_resolver_scorer,
    train_runtime_resolver_scorer_v3,
)


def _face(offset: float = 0.0) -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
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


def _runtime_feature_metric(**overrides: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "cloud_area_ratio": 0.9,
        "hull_area_ratio": 0.8,
        "points_outside_expanded_bbox_fraction": 0.05,
        "eye_mouth_order_valid_after_deroll": True,
        "roi_center_consensus_distance": 0.12,
        "landmark_consensus_distance": 0.10,
        "shape_plausibility_score": 0.3,
        "max_edge_length_ratio": 0.4,
        "mean_shape_fit_error": 0.02,
        "topology_violation_count": 0,
        "roll_degrees": 12.0,
        "yaw_degrees": -30.0,
        "geometry_veto_reasons": ("jaw_crop_warning",),
        "shape_veto_reasons": ("borderline_hull",),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_runtime_feature_extractor_is_single_source_of_truth_for_train_and_runtime() -> None:
    assert scorer_data.candidate_feature_map is runtime_features.candidate_feature_map
    assert (
        runtime_resolver_scorer_impl.candidate_feature_map
        is runtime_features.candidate_feature_map
    )

    candidate = SimpleNamespace(name="hrnet", is_fusion=False)
    metric = _runtime_feature_metric()
    context: dict[str, T.Any] = {
        "runtime_bucket": "profile_left",
        "risk_route": "high_risk",
        "model_predictions_available": {"hrnet": True, "spiga": True},
        "roll_estimate": 4.0,
        "yaw_estimate": -20.0,
        "candidate_yaw_disagreement": 15.0,
        "max_disagreement_px": 42.0,
        "runtime_bucket_source": "runtime_image_evidence",
        "hard_case_tags": ("profile",),
        "candidate_extra_features": {
            "hrnet": {
                "candidate_is_consensus_like": 0.0,
                "single_model_disagreement_px": 18.0,
                "candidate_distance_to_hrnet": 0.0,
            }
        },
    }

    direct = candidate_feature_map(candidate, metric, **context)
    batched = runtime_candidate_feature_maps([candidate], {"hrnet": metric}, **context)[0]

    assert batched == direct
    assert "candidate_name=hrnet" in direct
    assert "runtime_bucket=profile_left" in direct
    assert "geometry_veto_reason=jaw_crop_warning" in direct
    assert "shape_veto_reason=borderline_hull" in direct
    assert "candidate_is_consensus_like" in direct
    assert {
        "candidate_nme",
        "oracle_nme",
        "selection_cost",
        "transform_cost_v3",
        "transform_regret_v3",
        "transform_oracle_candidate_v3",
    }.isdisjoint(direct)

    ordered = runtime_feature_order([direct])
    assert ordered[:2] == ("candidate_is_single_model", "candidate_is_fusion")
    assert [name for name in RUNTIME_PREFERRED_FEATURE_ORDER if name in direct] == list(
        ordered[: len([name for name in RUNTIME_PREFERRED_FEATURE_ORDER if name in direct])]
    )


def test_training_feature_order_uses_runtime_feature_order() -> None:
    row = SimpleNamespace(
        feature_values={
            "candidate_name=hrnet": 1.0,
            "yaw_degrees": -30.0,
            "candidate_is_fusion": 0.0,
            "candidate_is_single_model": 1.0,
            "has_geometry_veto": 1.0,
        }
    )

    typed_row = T.cast(scorer_data.CandidateQualityRow, row)
    assert scorer_training.feature_order([typed_row]) == runtime_feature_order(
        [row.feature_values]
    )


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


def _write_fixture_bucket_weights(weights_path: Path) -> None:
    models = ("hrnet", "spiga", "orformer")
    write_best_weights(
        weights_path,
        {"hrnet": [1.0] * 68, "spiga": [0.0] * 68, "orformer": [0.0] * 68},
        models=models,
        bucket_weights={"frontal": {model: [1.0] * 68 for model in models}},
    )


def _write_fixture_images(manifest_path: Path) -> None:
    import cv2

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        image = np.full((128, 128, 3), 128, dtype="uint8")  # type: ignore[var-annotated]
        cv2.imwrite(str(manifest_path.parent / sample["image"]), image)


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a deterministic fake LightGBM module for scorer tests."""

    class FakeBooster:
        def __init__(
            self,
            model_str: str = "fake-model",
            feature_count: int = 1,
            **_kwargs: object,
        ) -> None:
            self.model_str = model_str
            self.feature_count = feature_count

        def feature_importance(self, *, importance_type: str = "gain") -> np.ndarray:
            assert importance_type == "gain"
            return np.arange(self.feature_count, 0, -1, dtype="float64")

        def model_to_string(self) -> str:
            return self.model_str

        def predict(self, matrix: np.ndarray, **_kwargs: object) -> np.ndarray:
            # Lower score is better. Make hrnet-like one-hot rows score below
            # spiga-like one-hot rows for deterministic ranking smoke tests.
            if matrix.shape[1] == 0:
                return T.cast(np.ndarray, np.zeros(matrix.shape[0], dtype="float64"))
            return T.cast(
                np.ndarray, np.asarray([float(row[0]) for row in matrix], dtype="float64")
            )

    class FakeRanker:
        def __init__(self, **params: object) -> None:
            self.params = params
            self.booster_ = FakeBooster()

        def fit(
            self,
            matrix: np.ndarray,
            labels: np.ndarray,
            *,
            group: list[int],
            sample_weight: np.ndarray,
        ) -> FakeRanker:
            assert matrix.shape[0] == labels.shape[0] == sample_weight.shape[0]
            assert sum(group) == matrix.shape[0]
            self.booster_ = FakeBooster(feature_count=matrix.shape[1])
            return self

    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(LGBMRanker=FakeRanker, Booster=FakeBooster),
    )


def _write_v3_test_scorer(
    path: Path,
    *,
    features: tuple[str, ...],
    runtime_policy: str = "learned_quality_v3",
    version: str = "learned_quality_v3",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "version": version,
                "scorer_version": version,
                "model_type": MODEL_TYPE_LIGHTGBM_LAMBDARANK,
                "target": TARGET_TRANSFORM_REGRET_V3,
                "objective": "lambdarank_transform_regret_v3",
                "training_mode": "grouped_lambdarank_v3",
                "selection_target": TARGET_TRANSFORM_REGRET_V3,
                "runtime_policy": runtime_policy,
                "score_semantics": SCORE_SEMANTICS_PREDICTED_COST,
                "higher_is_better": False,
                "failure_threshold": 0.08,
                "features": list(features),
                "runtime_feature_contract_version": RUNTIME_FEATURE_CONTRACT_VERSION,
                "model_data": "fake-model",
                "feature_importances": {feature: 1.0 for feature in features},
                "calibration": {"type": "none", "params": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _v3_training_row(
    *,
    sample_id: str,
    candidate_name: str,
    transform_regret_v3: float,
) -> scorer_data.CandidateQualityRow:
    return scorer_data.CandidateQualityRow(
        sample_id=sample_id,
        face_index=0,
        dataset="test",
        condition="profile",
        candidate_name=candidate_name,
        candidate_nme=0.0,
        oracle_nme=0.0,
        regret_vs_oracle=0.0,
        normalized_regret=0.0,
        failure_label=False,
        large_regret_label=False,
        candidate_failure_or_high_gap=False,
        selection_cost=0.0,
        is_oracle=candidate_name == "hrnet",
        was_selected_by_current_policy=candidate_name == "hrnet",
        gap_vs_oracle=0.0,
        runtime_bucket="profile_left",
        hard_case_tags=("profile",),
        risk_route="high_risk",
        feature_values={
            f"candidate_name={candidate_name}": 1.0,
            "candidate_is_single_model": 1.0,
            "candidate_is_fusion": 0.0,
            "candidate_distance_to_hrnet": 0.0 if candidate_name == "hrnet" else 1.0,
            "single_model_disagreement_px": 1.0,
            "hrnet_geometry_valid": 1.0,
            "runtime_bucket_source=stored_manifest_landmark_ensemble": 1.0,
        },
        selected_by_current_policy="hrnet",
        selected_candidate_missing_from_eval=False,
        oracle="hrnet",
        runtime_bucket_source="stored_manifest_landmark_ensemble",
        geometry_veto_reasons=(),
        transform_cost_v3=transform_regret_v3,
        corner_delta_v3=0.0,
        center_delta_v3=0.0,
        scale_delta_v3=0.0,
        roll_delta_degrees_v3=0.0,
        fit_delta_v3=0.0,
        transform_oracle_cost_v3=0.0,
        transform_regret_v3=transform_regret_v3,
        transform_oracle_candidate_v3="hrnet",
        transform_oracle_gap_v3=0.25,
        rankable_v3=True,
        hard_invalid_v3=False,
        hard_invalid_reasons_v3=(),
        soft_structural_penalty_v3=0.0,
    )


def _patch_v3_training_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    regrets = {
        "hrnet": 0.0,
        "spiga": 0.25,
        "static_weighted_downweight": 0.10,
    }
    rows = [
        (
            _v3_training_row(
                sample_id=sample_id,
                candidate_name=candidate_name,
                transform_regret_v3=regret,
            ),
            "gt_hard",
        )
        for sample_id in ("s1", "s2")
        for candidate_name, regret in regrets.items()
    ]
    monkeypatch.setattr(scorer_training, "load_scorer_contexts", lambda **_kwargs: [])
    monkeypatch.setattr(scorer_training, "tagged_quality_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        scorer_training,
        "scorer_candidate_table_rows",
        lambda *_args, **_kwargs: [],
    )


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
        face_index=0,
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
        truth_landmarks=_face(),
        normalizer=100.0,
        visibility=None,
        nme_by_candidate=nme_by_candidate,
        failure_by_candidate={
            candidate.name: bool((failure_by_candidate or {}).get(candidate.name, False))
            for candidate in candidates
        },
        runtime_bucket="frontal",
        hard_case_tags=(),
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


def test_stored_condition_seeds_hard_case_tags_when_tags_missing() -> None:
    tags = scorer_data._hard_case_tags_with_stored_condition(
        stored_condition="profile_occlusion",
        taxonomy_tags=("profile_pose",),
    )

    assert tags == ("profile_occlusion", "profile_pose")


def test_stored_non_hard_condition_does_not_seed_hard_case_tags() -> None:
    tags = scorer_data._hard_case_tags_with_stored_condition(
        stored_condition="frontal",
        taxonomy_tags=("occlusion",),
    )

    assert tags == ("occlusion",)


def test_split_labels_do_not_count_occlusion_as_normal() -> None:
    context = SimpleNamespace(
        sample_id="sample",
        source="gt_hard",
        dataset="test",
        condition="occlusion",
        runtime_bucket="frontal",
        hard_case_tags=("occlusion",),
        failure_by_candidate={"hrnet": False},
    )

    labels = scorer_eval_impl.split_labels_for_context(
        context,
        choices={"sample": "hrnet"},
        source_by_sample_id={},
    )

    assert "occlusion" in labels
    assert "normal" not in labels


def test_occluded_side_requires_visibility_or_yaw_evidence() -> None:
    context = SimpleNamespace(visibility=None, yaw_estimate=0.0)

    assert scorer_eval_impl._indices_for_region(context, "occluded_side") == ()
    assert scorer_eval_impl._indices_for_region(context, "visible_side") == ()


def test_scorer_training_weights_hard_condition_rows() -> None:
    context = _candidate_context(
        nme_by_candidate={
            "oracle": 0.01,
            "zero": 0.01,
            "small": 0.015,
            "large": 0.05,
            "failure": 0.02,
        },
        failure_by_candidate={"failure": True},
    )
    context = scorer_data.SampleCandidateContext(
        **{
            **context.__dict__,
            "condition": "profile_occlusion",
            "runtime_bucket": "profile_left",
            "hard_case_tags": ("profile_occlusion", "occlusion", "profile_pose"),
        }
    )
    row = scorer_data.rows_for_context(context)[0]

    assert scorer_eval_impl is not None
    assert scorer_training.scorer_sample_weight(row, "gt_hard") >= 4.0


def test_hard_bucket_gates_fail_on_profile_failures() -> None:
    failed = scorer_eval_impl.hard_bucket_promotion_gates(
        {
            "profile": {
                "sample_count": 2,
                "failure_rate": 0.5,
                "catastrophic_failure_count": 1,
            }
        }
    )

    assert "profile_failure_rate_above_hard_bucket_gate" in failed
    assert "profile_catastrophic_failures_above_hard_bucket_gate" in failed


def test_rows_for_context_adds_downstream_weighted_alignment_cost() -> None:
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
        + scorer_data.JAW_CROP_MASK_PENALTY
        + DEFAULT_FAILURE_COST_PENALTY
        + DEFAULT_COLLAPSE_COST_PENALTY
        + scorer_data.PRODUCTION_FAILURE_PENALTY
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


def test_candidate_feature_map_includes_shape_plausibility_fields() -> None:
    candidate = CandidateRecord(
        name="spiga",
        landmarks=_face(),
        is_fusion=False,
        contributing_models=("spiga",),
    )
    metric = CandidateMetrics(
        roll_degrees=0.0,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        shape_plausibility_score=1.25,
        shape_veto_reasons=("edge_length_extreme",),
        max_edge_length_ratio=1.6,
        mean_shape_fit_error=0.14,
        topology_violation_count=3,
    )

    features = candidate_feature_map(candidate, metric)

    assert features["shape_plausibility_score"] == pytest.approx(1.25)
    assert features["max_edge_length_ratio"] == pytest.approx(1.6)
    assert features["mean_shape_fit_error"] == pytest.approx(0.14)
    assert features["topology_violation_count"] == pytest.approx(3.0)
    assert features["shape_veto_reason=edge_length_extreme"] == 1.0


def test_candidate_table_rows_include_shape_plausibility_diagnostics() -> None:
    context = _candidate_context(
        nme_by_candidate={
            "oracle": 0.01,
            "zero": 0.01,
            "small": 0.015,
            "large": 0.05,
            "failure": 0.02,
        }
    )
    metric = context.metrics["oracle"]
    metric.shape_plausibility_score = 1.25
    metric.shape_veto_reasons = ("edge_length_extreme",)
    metric.max_edge_length_ratio = 1.6
    metric.mean_shape_fit_error = 0.14
    metric.topology_violation_count = 3

    rows = scorer_data.candidate_table_rows_for_context(context)
    oracle = next(row for row in rows if row["candidate"] == "oracle")

    assert oracle["shape_plausibility_score"] == pytest.approx(1.25)
    assert oracle["shape_veto_reasons"] == "edge_length_extreme"
    assert oracle["max_edge_length_ratio"] == pytest.approx(1.6)
    assert oracle["mean_shape_fit_error"] == pytest.approx(0.14)
    assert oracle["topology_violation_count"] == 3


def test_rows_and_candidate_table_include_hard_case_tags() -> None:
    context = _candidate_context(
        nme_by_candidate={
            "oracle": 0.01,
            "zero": 0.01,
            "small": 0.015,
            "large": 0.05,
            "failure": 0.02,
        }
    )
    context = scorer_data.SampleCandidateContext(
        **{
            **context.__dict__,
            "condition": "profile_occlusion",
            "runtime_bucket": "profile_left",
            "hard_case_tags": ("profile_occlusion", "occlusion", "profile_pose"),
        }
    )

    row = scorer_data.rows_for_context(context)[0]
    candidate_row = scorer_data.candidate_table_rows_for_context(context)[0]

    assert row.to_csv_row()["hard_case_tags"] == "profile_occlusion|occlusion|profile_pose"
    assert candidate_row["hard_case_tags"] == "profile_occlusion|occlusion|profile_pose"
    assert row.feature_values["hard_case_tag=profile_occlusion"] == pytest.approx(1.0)


def test_scorer_suite_trains_only_active_v3_target_and_persists_feature_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    weights_path = tmp_path / "weights.json"
    save_weights(weights_path, {"hrnet": [1.0] * 68, "spiga": [0.0] * 68})

    def v3_row(
        *,
        sample_id: str,
        candidate_name: str,
        transform_regret_v3: float,
    ) -> scorer_data.CandidateQualityRow:
        return scorer_data.CandidateQualityRow(
            sample_id=sample_id,
            face_index=0,
            dataset="test",
            condition="profile",
            candidate_name=candidate_name,
            candidate_nme=0.0,
            oracle_nme=0.0,
            regret_vs_oracle=0.0,
            normalized_regret=0.0,
            failure_label=False,
            large_regret_label=False,
            candidate_failure_or_high_gap=False,
            selection_cost=0.0,
            is_oracle=candidate_name == "hrnet",
            was_selected_by_current_policy=candidate_name == "hrnet",
            gap_vs_oracle=0.0,
            runtime_bucket="profile_left",
            hard_case_tags=("profile",),
            risk_route="high_risk",
            feature_values={
                f"candidate_name={candidate_name}": 1.0,
                "candidate_is_single_model": 1.0,
                "candidate_is_fusion": 0.0,
            },
            selected_by_current_policy="hrnet",
            selected_candidate_missing_from_eval=False,
            oracle="hrnet",
            runtime_bucket_source="test",
            geometry_veto_reasons=(),
            transform_cost_v3=transform_regret_v3,
            corner_delta_v3=0.0,
            center_delta_v3=0.0,
            scale_delta_v3=0.0,
            roll_delta_degrees_v3=0.0,
            fit_delta_v3=0.0,
            transform_oracle_cost_v3=0.0,
            transform_regret_v3=transform_regret_v3,
            transform_oracle_candidate_v3="hrnet",
            transform_oracle_gap_v3=0.25,
            rankable_v3=True,
            hard_invalid_v3=False,
            hard_invalid_reasons_v3=(),
            soft_structural_penalty_v3=0.0,
        )

    fake_rows = [
        (v3_row(sample_id="s1", candidate_name="hrnet", transform_regret_v3=0.0), "gt_hard"),
        (v3_row(sample_id="s1", candidate_name="spiga", transform_regret_v3=0.25), "gt_hard"),
        (v3_row(sample_id="s2", candidate_name="hrnet", transform_regret_v3=0.0), "gt_hard"),
        (v3_row(sample_id="s2", candidate_name="spiga", transform_regret_v3=0.30), "gt_hard"),
    ]

    monkeypatch.setattr(scorer_training, "load_scorer_contexts", lambda **_kwargs: [])
    monkeypatch.setattr(
        scorer_training, "tagged_quality_rows", lambda *_args, **_kwargs: fake_rows
    )
    monkeypatch.setattr(
        scorer_training, "scorer_candidate_table_rows", lambda *_args, **_kwargs: []
    )

    output_dir = tmp_path / "train_suite_v3_only"
    metrics = scorer_training.train_runtime_resolver_scorer_suite(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=None,
        production_cache_dir=None,
        weights_path=weights_path,
        candidates=("hrnet", "spiga"),
        output_dir=output_dir,
        iterations=4,
        eval_fraction=0.0,
    )

    canonical_v3 = output_dir / "scorers" / "learned_quality_v3.json"
    artifact = json.loads(canonical_v3.read_text(encoding="utf-8"))

    assert set(metrics["scorers"]) == {"learned_quality_v3"}
    assert metrics["active_target"] == TARGET_TRANSFORM_REGRET_V3
    assert metrics["artifact"] == str(canonical_v3)
    assert canonical_v3.is_file()
    assert artifact["target"] == TARGET_TRANSFORM_REGRET_V3
    assert artifact["runtime_feature_contract_version"] == RUNTIME_FEATURE_CONTRACT_VERSION
    assert metrics["scorers"]["learned_quality_v3"]["runtime_feature_contract_version"] == (
        RUNTIME_FEATURE_CONTRACT_VERSION
    )


def test_train_runtime_resolver_scorer_wrapper_writes_v3_artifact_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    _patch_v3_training_rows(monkeypatch)
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
        eval_fraction=0.0,
    )

    artifact_path = output_dir / SCORER_V3_ARTIFACT
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact_path.is_file()
    assert metrics["artifact"] == str(artifact_path)
    assert metrics["target"] == TARGET_TRANSFORM_REGRET_V3
    assert metrics["model_type"] == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert metrics["score_semantics"] == SCORE_SEMANTICS_PREDICTED_COST
    assert metrics["higher_is_better"] is False
    assert metrics["training_data_counts"]["row_count"] == 6
    assert metrics["training_data_counts"]["sample_group_count"] == 2

    assert artifact["version"] == "learned_quality_v3"
    assert artifact["scorer_version"] == "learned_quality_v3"
    assert artifact["runtime_policy"] == "learned_quality_v3"
    assert artifact["runtime_feature_contract_version"] == RUNTIME_FEATURE_CONTRACT_VERSION
    assert artifact["target"] == TARGET_TRANSFORM_REGRET_V3
    assert artifact["model_type"] == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert artifact["score_semantics"] == SCORE_SEMANTICS_PREDICTED_COST
    assert artifact["higher_is_better"] is False
    assert "candidate_name=spiga" in artifact["features"]
    assert "candidate_distance_to_hrnet" in artifact["features"]
    assert "single_model_disagreement_px" in artifact["features"]
    assert "hrnet_geometry_valid" in artifact["features"]
    assert "runtime_bucket_source=stored_manifest_landmark_ensemble" in artifact["features"]

    assert (output_dir / "runtime_resolver_scorer_training_rows.csv").is_file()
    assert (output_dir / "runtime_resolver_scorer_eval_rows.csv").is_file()
    assert (output_dir / "runtime_resolver_scorer_v3_feature_importances.csv").is_file()

    training_rows = output_dir / "runtime_resolver_scorer_training_rows.csv"
    with training_rows.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.DictReader(handle))
    assert "oracle_nme" in header
    assert "regret_vs_oracle" in header
    assert "normalized_regret" in header
    assert "large_regret_label" in header
    assert "candidate_failure_or_high_gap" in header
    assert "selection_cost" in header


def test_train_runtime_resolver_scorer_supports_v3_ranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    _patch_v3_training_rows(monkeypatch)
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
        eval_fraction=0.0,
        target=TARGET_TRANSFORM_REGRET_V3,
    )

    artifact = json.loads((output_dir / SCORER_V3_ARTIFACT).read_text(encoding="utf-8"))
    assert metrics["target"] == TARGET_TRANSFORM_REGRET_V3
    assert metrics["model_type"] == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert metrics["score_semantics"] == SCORE_SEMANTICS_PREDICTED_COST
    assert metrics["higher_is_better"] is False
    assert metrics["training_data_counts"]["row_count"] == 6
    assert metrics["training_data_counts"]["sample_group_count"] == 2
    assert metrics["sample_weighting"]["strategy"] == "v3_condition_query_weighting"

    assert artifact["target"] == TARGET_TRANSFORM_REGRET_V3
    assert artifact["model_type"] == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert artifact["score_semantics"] == SCORE_SEMANTICS_PREDICTED_COST
    assert artifact["higher_is_better"] is False
    assert artifact["version"] == "learned_quality_v3"
    assert artifact["scorer_version"] == "learned_quality_v3"
    assert artifact["selection_target"] == "inverse_transform_regret_v3_rank"
    assert artifact["objective"] == "lambdarank_visible_transform_regret"
    assert artifact["training_mode"] == "grouped_lambdarank_rankable_v3_only"
    assert artifact["runtime_policy"] == "learned_quality_v3"


def test_v3_artifact_ranks_lower_cost_features_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    _patch_v3_training_rows(monkeypatch)
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
        eval_fraction=0.0,
        target=TARGET_TRANSFORM_REGRET_V3,
    )

    scorer = load_runtime_resolver_scorer(output_dir / "runtime_resolver_scorer_v3.json")
    low_cost_score = scorer.score_feature_map({"candidate_name=hrnet": 1.0})
    high_cost_score = scorer.score_feature_map({"candidate_name=spiga": 1.0})

    assert scorer.model_type == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert scorer.score_semantics == SCORE_SEMANTICS_PREDICTED_COST
    assert scorer.higher_is_better is False
    assert np.isfinite(low_cost_score)
    assert np.isfinite(high_cost_score)


def test_train_runtime_resolver_scorer_v3_writes_lightgbm_ranker_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    _patch_v3_training_rows(monkeypatch)
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "train_v3"

    metrics = train_runtime_resolver_scorer_v3(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=output_dir,
        iterations=4,
        eval_fraction=0.0,
        split_seed=7,
    )

    artifact = json.loads((output_dir / SCORER_V3_ARTIFACT).read_text(encoding="utf-8"))
    scorer = load_runtime_resolver_scorer(output_dir / SCORER_V3_ARTIFACT)
    feature_map = {"candidate_name=hrnet": 1.0}

    assert metrics["model_type"] == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert metrics["score_semantics"] == SCORE_SEMANTICS_PREDICTED_COST
    assert artifact["model_type"] == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert artifact["version"] == "learned_quality_v3"
    assert artifact["runtime_policy"] == "learned_quality_v3"
    assert artifact["higher_is_better"] is False
    assert artifact["training_data_counts"]["sample_group_count"] == 2
    assert artifact["split_ids"]["seed"] == 7
    assert artifact["feature_importances"]
    assert (output_dir / "runtime_resolver_scorer_v3_feature_importances.csv").is_file()
    assert scorer.model_type == MODEL_TYPE_LIGHTGBM_LAMBDARANK
    assert scorer.score_semantics == SCORE_SEMANTICS_PREDICTED_COST
    assert scorer.higher_is_better is False
    assert scorer.score_feature_map(feature_map) == pytest.approx(
        scorer.score_feature_map(feature_map)
    )


def test_evaluate_runtime_resolver_scorer_reports_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    _patch_v3_training_rows(monkeypatch)
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    scorer_path = _write_v3_test_scorer(
        tmp_path / "runtime_resolver_scorer_v3.json",
        features=("candidate_name=hrnet", "candidate_name=spiga"),
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
    assert "transform_error_missing_rankable_pair_eval" in report["failed_gates"]
    assert report["heldout_eval"] is False
    assert report["runtime_policy"] == "learned_quality_v3"
    assert report["promoted_scorer_label"] == "learned_quality_v3"
    assert "learned_quality_v3" in report
    assert report["learned_quality_v3"]["pick_counts"] == {"hrnet": 2}
    assert report["production_only_policy_metrics"]["sample_count"] == 2
    assert "learned_quality_v3" in report["production_only_policy_metrics"]
    assert report["production_only_policy_metrics"]["learned_quality_v3"]["pick_counts"] == {
        "hrnet": 2
    }
    assert report["gt_hard_only_policy_metrics"]["sample_count"] == 0
    assert report["best_single"]["candidate"] == "hrnet"
    assert set(report["metrics_by_split"]) == {
        "normal",
        "profile",
        "occlusion",
        "profile_occlusion",
        "production_failures",
    }
    assert report["metrics_by_split"]["profile"]["sample_count"] == 2
    assert report["metrics_by_split"]["profile"]["full_face_mean_nme"] >= 0.0
    assert report["region_reports"]["region_metrics_available"] is True
    assert (tmp_path / "eval" / "scorer_policy_report.csv").is_file()
    assert (tmp_path / "eval" / "scorer_feature_importance.csv").is_file()
    assert (tmp_path / "eval" / "per_region_nme.csv").is_file()
    assert (tmp_path / "eval" / "per_region_geometry.csv").is_file()
    assert (tmp_path / "eval" / "per_region_worst_samples.json").is_file()
    with (tmp_path / "eval" / "per_region_nme.csv").open(newline="", encoding="utf-8") as handle:
        region_rows = list(csv.DictReader(handle))
    assert {row["region"] for row in region_rows} >= {
        "jaw",
        "eyes",
        "mouth",
        "occluded_side",
        "visible_side",
    }


def test_evaluate_runtime_resolver_scorer_reports_v3_primary_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    _patch_v3_training_rows(monkeypatch)
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    train_dir = tmp_path / "train_v3_primary"
    train_runtime_resolver_scorer_v3(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=train_dir,
        iterations=4,
        eval_fraction=0.0,
    )
    scorer_path = train_dir / SCORER_V3_ARTIFACT

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        scorer_path=scorer_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=tmp_path / "eval_v3",
    )

    assert report["primary_scorer_policy"] == "learned_quality_v3"
    assert report["runtime_policy"] == "learned_quality_v3"
    assert report["promoted_scorer_label"] == "learned_quality_v3"
    assert report["promoted_scorer_target"] == TARGET_TRANSFORM_REGRET_V3
    assert report["scorer_target"] == TARGET_TRANSFORM_REGRET_V3
    assert "learned_quality_v3" in report
    assert report["primary_scorer"]["label"] == "learned_quality_v3"
    assert report["primary_scorer"]["metrics"] == report["learned_quality_v3"]


def test_evaluate_runtime_resolver_scorer_emits_stable_v3_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    _patch_v3_training_rows(monkeypatch)
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    train_dir = tmp_path / "train_v3_only_keys"
    train_runtime_resolver_scorer_v3(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=train_dir,
        iterations=4,
        eval_fraction=0.0,
    )
    scorer_path = train_dir / SCORER_V3_ARTIFACT

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=manifest_path,
        production_cache_dir=cache_dir,
        weights_path=weights_path,
        scorer_path=scorer_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=tmp_path / "eval_v3_only",
    )

    assert report["primary_scorer_policy"] == "learned_quality_v3"
    assert report["runtime_policy"] == "learned_quality_v3"
    assert report["primary_scorer"]["label"] == "learned_quality_v3"
    assert report["primary_scorer"]["metrics"] == report["learned_quality_v3"]

    production = report["production_only_policy_metrics"]
    assert "learned_quality_v3" in production
    assert "learned_quality_v3" in report


def test_evaluate_runtime_resolver_scorer_filters_to_eval_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    scorer_path = _write_v3_test_scorer(
        tmp_path / "runtime_resolver_scorer_v3.json",
        features=("candidate_name=hrnet",),
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

    assert report["eval_split"] == str(eval_split)
    assert report["sample_count"] == 1
    assert report["heldout_eval"] is True
    assert report["row_backed_eval"] is False
    assert report["scorer_rows"] == ""
    assert report["production_only_policy_metrics"]["sample_count"] == 1
    assert (tmp_path / "heldout_eval" / "scorer_policy_eval_report.json").is_file()


def test_evaluate_runtime_resolver_scorer_blocks_safe_fallback_without_score_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    scorer_path = _write_v3_test_scorer(
        tmp_path / "runtime_resolver_scorer_v3.json",
        features=("candidate_name=static_weighted",),
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

    assert report["learned_quality_v3"]["pick_counts"] == {"static_weighted": 2}
    assert report["safe_fallback_count"] == 0
    assert report["safe_fallback_min_delta"] == 0.05
    assert report["fallback_impact"]["count_with_rejected_candidate"] == 0


def test_evaluate_runtime_resolver_scorer_refuses_gt_hard_without_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
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
    scorer_path = _write_v3_test_scorer(
        tmp_path / "runtime_resolver_scorer_v3.json",
        features=("candidate_name=static_weighted", "candidate_name=orformer"),
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


def test_load_contexts_explicit_candidates_do_not_gain_adaptive_rows(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    _write_fixture_bucket_weights(weights_path)

    contexts = scorer_data.load_contexts(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga"),
    )

    assert len(contexts) == 2
    for context in contexts:
        assert tuple(candidate.name for candidate in context.candidates) == ("hrnet", "spiga")


def test_load_contexts_can_request_adaptive_candidate_explicitly(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    _write_fixture_bucket_weights(weights_path)
    candidate_name = bucket_candidate_name("static_weighted", "frontal")

    contexts = scorer_data.load_contexts(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=(candidate_name,),
    )

    assert len(contexts) == 2
    for context in contexts:
        assert tuple(candidate.name for candidate in context.candidates) == (candidate_name,)


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
    # ``fan`` is a default scorer candidate but this fixture only caches
    # hrnet/spiga/orformer predictions (and weights), so the table builder
    # legitimately skips fan -- leaving 8 candidates across 2 samples.
    expected_candidates = set(scorer_data.DEFAULT_SCORER_CANDIDATES) - {"fan"}
    assert len(rows) == len(expected_candidates) * 2 == 16
    assert sum(1 for _line in output_csv.open(encoding="utf-8")) == 17
    assert {row["candidate"] for row in rows} == expected_candidates
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
    scorer_path = _write_v3_test_scorer(
        tmp_path / "runtime_resolver_scorer_v3.json",
        features=("candidate_name=hrnet", "candidate_name=spiga"),
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


def _write_canonical_scorer_rows(path: Path) -> Path:
    fieldnames = [
        "split",
        "source",
        "sample_id",
        "face_index",
        "dataset",
        "condition",
        "candidate_name",
        "candidate_nme",
        "oracle_nme",
        "regret_vs_oracle",
        "normalized_regret",
        "failure_label",
        "large_regret_label",
        "candidate_failure_or_high_gap",
        "selection_cost",
        "is_oracle",
        "was_selected_by_current_policy",
        "gap_vs_oracle",
        "runtime_bucket",
        "runtime_bucket_source",
        "risk_route",
        "geometry_veto_reasons",
        "selected_by_current_policy",
        "selected_candidate_missing_from_eval",
        "oracle",
        "transform_cost_v3",
        "transform_oracle_cost_v3",
        "transform_regret_v3",
        "transform_oracle_candidate_v3",
        "transform_oracle_gap_v3",
        "rankable_v3",
        "hard_invalid_v3",
        "hard_invalid_reasons_v3",
        "soft_structural_penalty_v3",
        "features_json",
    ]

    def row(split: str, sample_id: str, candidate: str, nme: float) -> dict[str, object]:
        oracle_nme = 0.01
        gap = nme - oracle_nme
        return {
            "split": split,
            "source": "production_validated",
            "sample_id": sample_id,
            "face_index": 0,
            "dataset": "production_validated",
            "condition": "profile_left",
            "candidate_name": candidate,
            "candidate_nme": nme,
            "oracle_nme": oracle_nme,
            "regret_vs_oracle": gap,
            "normalized_regret": max(gap, 0.0),
            "failure_label": 0,
            "large_regret_label": 0,
            "candidate_failure_or_high_gap": 0,
            "selection_cost": max(gap, 0.0),
            "is_oracle": int(candidate == "hrnet"),
            "was_selected_by_current_policy": int(candidate == "hrnet"),
            "gap_vs_oracle": gap,
            "runtime_bucket": "stored_profile_left",
            "runtime_bucket_source": "stored_manifest_landmark_ensemble",
            "risk_route": "low_risk",
            "geometry_veto_reasons": "",
            "selected_by_current_policy": "hrnet",
            "selected_candidate_missing_from_eval": 0,
            "oracle": "hrnet",
            "transform_cost_v3": max(gap, 0.0),
            "transform_oracle_cost_v3": 0.0,
            "transform_regret_v3": max(gap, 0.0),
            "transform_oracle_candidate_v3": "hrnet",
            "transform_oracle_gap_v3": 0.01,
            "rankable_v3": 1,
            "hard_invalid_v3": 0,
            "hard_invalid_reasons_v3": "",
            "soft_structural_penalty_v3": 0.0,
            "features_json": json.dumps({f"candidate_name={candidate}": 1.0}, sort_keys=True),
        }

    rows = [
        row("train", "train_sample", "hrnet", 0.01),
        row("train", "train_sample", "spiga", 0.04),
        row("train", "train_sample", "static_weighted_downweight", 0.02),
        row("eval", "eval_sample", "hrnet", 0.01),
        row("eval", "eval_sample", "spiga", 0.04),
        row("eval", "eval_sample", "static_weighted_downweight", 0.02),
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def test_row_contexts_from_scorer_rows_uses_only_eval_split(tmp_path: Path) -> None:
    rows_path = _write_canonical_scorer_rows(tmp_path / "rows.csv")

    contexts, source_by_sample_id = scorer_eval_impl.row_contexts_from_scorer_rows(rows_path)

    assert [context.sample_id for context in contexts] == ["eval_sample"]
    assert source_by_sample_id == {"eval_sample": "production_validated"}

    context = contexts[0]
    assert context.dataset == "production_validated"
    assert context.source == "production_validated"
    assert context.oracle == "hrnet"
    assert context.current_policy_choice == "hrnet"
    assert context.runtime_bucket == "stored_profile_left"
    assert set(context.nme_by_candidate) == {
        "hrnet",
        "spiga",
        "static_weighted_downweight",
    }
    by_candidate = {candidate.name: candidate for candidate in context.candidates}
    assert by_candidate["hrnet"].is_fusion is False
    assert by_candidate["spiga"].is_fusion is False
    assert by_candidate["static_weighted_downweight"].is_fusion is True
    assert len(context.scorer_rows) == 3
    hrnet_row = next(row for row in context.scorer_rows if row.candidate_name == "hrnet")
    assert hrnet_row.transform_regret_v3 == pytest.approx(0.0)
    assert "transform_regret_v3" not in hrnet_row.feature_values


def test_policy_summary_reports_v3_transform_metrics_and_skips_production() -> None:
    scorer_rows = (
        SimpleNamespace(
            candidate_name="hrnet",
            feature_values={},
            transform_regret_v3=0.0,
            transform_oracle_candidate_v3="hrnet",
            transform_oracle_gap_v3=0.25,
            rankable_v3=True,
            hard_invalid_v3=False,
        ),
        SimpleNamespace(
            candidate_name="spiga",
            feature_values={},
            transform_regret_v3=0.25,
            transform_oracle_candidate_v3="hrnet",
            transform_oracle_gap_v3=0.25,
            rankable_v3=True,
            hard_invalid_v3=False,
        ),
        SimpleNamespace(
            candidate_name="bad",
            feature_values={},
            transform_regret_v3=3.0,
            transform_oracle_candidate_v3="hrnet",
            transform_oracle_gap_v3=0.25,
            rankable_v3=False,
            hard_invalid_v3=True,
        ),
    )
    context = SimpleNamespace(
        sample_id="gt_sample",
        dataset="gt_hard",
        source="gt_hard",
        condition="profile",
        runtime_bucket="profile",
        runtime_bucket_source="",
        nme_by_candidate={"hrnet": 0.01, "spiga": 0.04, "bad": 0.20},
        failure_by_candidate={"hrnet": False, "spiga": False, "bad": True},
        oracle="hrnet",
        scorer_rows=scorer_rows,
    )
    contexts = T.cast("list[scorer_data.SampleCandidateContext]", [context])
    summary = scorer_eval_impl.policy_summary(
        contexts,
        {"gt_sample": "spiga"},
        source_by_sample_id={},
    )
    assert summary["mean_transform_regret_v3"] == pytest.approx(0.25)
    assert summary["p95_transform_regret_v3"] == pytest.approx(0.25)
    assert summary["oracle_match_rate_v3"] == pytest.approx(0.0)
    assert summary["invalid_selection_count_v3"] == 0
    assert summary["transform_eval_count_v3"] == 1

    invalid_summary = scorer_eval_impl.policy_summary(
        contexts,
        {"gt_sample": "bad"},
        source_by_sample_id={},
    )
    assert invalid_summary["invalid_selection_count_v3"] == 1
    assert invalid_summary["mean_transform_regret_v3"] == pytest.approx(0.0)

    production_context = SimpleNamespace(**{**context.__dict__, "source": "production_validated"})
    production_contexts = T.cast("list[scorer_data.SampleCandidateContext]", [production_context])
    production_summary = scorer_eval_impl.policy_summary(
        production_contexts,
        {"gt_sample": "spiga"},
        source_by_sample_id={},
    )
    assert production_summary["transform_eval_count_v3"] == 0
    assert production_summary["mean_transform_regret_v3"] == pytest.approx(0.0)


def test_scorer_policy_key_detects_v3_transform_artifact() -> None:
    scorer = LinearTestScorer(
        features=("candidate_name=hrnet",),
        coefficients=(1.0,),
    )

    assert scorer_eval_impl.scorer_policy_key(T.cast(T.Any, scorer)) == "learned_quality_v3"


def test_evaluate_runtime_resolver_scorer_uses_scorer_rows_without_context_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    rows_path = _write_canonical_scorer_rows(tmp_path / "rows.csv")
    scorer_path = _write_v3_test_scorer(
        tmp_path / "runtime_resolver_scorer_v3.json",
        features=(
            "candidate_name=hrnet",
            "candidate_name=spiga",
            "candidate_name=static_weighted_downweight",
        ),
    )

    def fail_context_rebuild(**_kwargs: object) -> object:
        raise AssertionError("row-backed evaluation must not rebuild scorer contexts")

    monkeypatch.setattr(scorer_eval_impl, "load_scorer_contexts", fail_context_rebuild)

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=None,
        production_cache_dir=None,
        weights_path=tmp_path / "unused_weights.json",
        scorer_path=scorer_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=tmp_path / "row_backed_eval",
        scorer_rows=rows_path,
        installed_scorer_dir=None,
    )

    assert report["sample_count"] == 1
    assert report["heldout_eval"] is True
    assert report["row_backed_eval"] is True
    assert report["scorer_rows"] == str(rows_path)
    assert report["eval_split"] == ""
    assert report["production_only_policy_metrics"]["sample_count"] == 1
    assert report["learned_quality_v3"]["pick_counts"] == {"hrnet": 1}
    assert (tmp_path / "row_backed_eval" / "scorer_policy_eval_report.json").is_file()


def test_row_backed_eval_high_risk_safe_fallback_handles_fusion_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    rows_path = _write_canonical_scorer_rows(tmp_path / "rows.csv")
    scorer_path = _write_v3_test_scorer(
        tmp_path / "high_risk_scorer.json",
        features=("candidate_name=hrnet",),
    )

    def fail_context_rebuild(**_kwargs: object) -> object:
        raise AssertionError("row-backed evaluation must not rebuild scorer contexts")

    monkeypatch.setattr(scorer_eval_impl, "load_scorer_contexts", fail_context_rebuild)

    report = evaluate_runtime_resolver_scorer(
        gt_manifest=None,
        gt_cache_dir=None,
        production_manifest=None,
        production_cache_dir=None,
        weights_path=tmp_path / "unused_weights.json",
        scorer_path=scorer_path,
        candidates=("hrnet", "spiga", "static_weighted_downweight"),
        output_dir=tmp_path / "row_backed_high_risk_eval",
        scorer_rows=rows_path,
        installed_scorer_dir=None,
        risk_floor_for_safe_fallback=0.50,
    )

    assert report["row_backed_eval"] is True
    assert report["sample_count"] == 1
    assert report["learned_quality_v3"]["pick_counts"]


def _v3_eval_row_for_summary(
    *,
    candidate_name: str,
    transform_regret_v3: float = 0.0,
    oracle: str = "",
    oracle_gap: float = 0.2,
    rankable: bool = True,
    hard_invalid: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_name=candidate_name,
        feature_values={},
        transform_cost_v3=transform_regret_v3,
        transform_oracle_cost_v3=0.0,
        transform_regret_v3=transform_regret_v3,
        transform_oracle_candidate_v3=oracle,
        transform_oracle_gap_v3=oracle_gap,
        rankable_v3=rankable,
        hard_invalid_v3=hard_invalid,
        hard_invalid_reasons_v3=("hard_invalid",) if hard_invalid else (),
        soft_structural_penalty_v3=0.0,
    )


def _v3_eval_context_for_summary(
    sample_id: str,
    rows: tuple[SimpleNamespace, ...],
    *,
    source: str = "gt_hard",
) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        source=source,
        dataset="gt_hard",
        runtime_bucket_source="",
        scorer_rows=rows,
    )


def test_transform_policy_summary_v3_excludes_single_valid_no_choice_group() -> None:
    context = _v3_eval_context_for_summary(
        "single_valid",
        (
            _v3_eval_row_for_summary(
                candidate_name="candidate_a",
                transform_regret_v3=0.0,
                oracle="candidate_a",
                oracle_gap=0.2,
                rankable=True,
                hard_invalid=False,
            ),
        ),
    )

    summary = scorer_eval_impl.transform_policy_summary_v3(
        [context],
        {"single_valid": "candidate_a"},
        source_by_sample_id={},
    )

    assert summary["transform_group_count_v3"] == 1
    assert summary["transform_eval_count_v3"] == 0
    assert summary["single_valid_group_count_v3"] == 1
    assert summary["zero_valid_group_count_v3"] == 0
    assert summary["near_tie_excluded_count_v3"] == 0
    assert summary["mean_transform_regret_v3"] == pytest.approx(0.0)
    assert summary["oracle_match_rate_v3"] == pytest.approx(0.0)


def test_transform_policy_summary_v3_counts_only_rankable_pair_groups_as_eval() -> None:
    context = _v3_eval_context_for_summary(
        "rankable_pair",
        (
            _v3_eval_row_for_summary(
                candidate_name="candidate_a",
                transform_regret_v3=0.0,
                oracle="candidate_a",
                oracle_gap=0.2,
                rankable=True,
                hard_invalid=False,
            ),
            _v3_eval_row_for_summary(
                candidate_name="candidate_b",
                transform_regret_v3=0.25,
                oracle="candidate_a",
                oracle_gap=0.2,
                rankable=True,
                hard_invalid=False,
            ),
            _v3_eval_row_for_summary(
                candidate_name="candidate_c",
                transform_regret_v3=0.0,
                oracle="candidate_a",
                oracle_gap=0.2,
                rankable=False,
                hard_invalid=True,
            ),
        ),
    )

    summary = scorer_eval_impl.transform_policy_summary_v3(
        [context],
        {"rankable_pair": "candidate_b"},
        source_by_sample_id={},
    )

    assert summary["transform_group_count_v3"] == 1
    assert summary["transform_eval_count_v3"] == 1
    assert summary["single_valid_group_count_v3"] == 0
    assert summary["invalid_selection_count_v3"] == 0
    assert summary["mean_transform_regret_v3"] == pytest.approx(0.25)
    assert summary["oracle_match_rate_v3"] == pytest.approx(0.0)


def test_transform_policy_summary_v3_keeps_zero_valid_separate_from_single_valid() -> None:
    context = _v3_eval_context_for_summary(
        "zero_valid",
        (
            _v3_eval_row_for_summary(
                candidate_name="candidate_a",
                rankable=False,
                hard_invalid=True,
            ),
            _v3_eval_row_for_summary(
                candidate_name="candidate_b",
                rankable=False,
                hard_invalid=True,
            ),
        ),
    )

    summary = scorer_eval_impl.transform_policy_summary_v3(
        [context],
        {"zero_valid": "candidate_a"},
        source_by_sample_id={},
    )

    assert summary["transform_group_count_v3"] == 1
    assert summary["transform_eval_count_v3"] == 0
    assert summary["zero_valid_group_count_v3"] == 1
    assert summary["single_valid_group_count_v3"] == 0
    assert summary["invalid_selection_count_v3"] == 1


def _v3_gate_metrics(
    *,
    mean: float,
    p95: float | None = None,
    invalid_rate: float = 0.0,
    eval_count: int = 8,
) -> dict[str, float | int]:
    return {
        "mean_transform_regret_v3": mean,
        "p95_transform_regret_v3": mean if p95 is None else p95,
        "invalid_selection_rate_v3": invalid_rate,
        "invalid_selection_count_v3": int(round(invalid_rate * eval_count)),
        "transform_eval_count_v3": eval_count,
    }


def test_v3_learnability_gate_attributes_high_transform_regret() -> None:
    result = scorer_eval_impl.v3_learnability_promotion_gates(
        scorer_policy_name="learned_quality_v3",
        comparison_metrics={
            "learned_quality_v3": _v3_gate_metrics(mean=0.20, p95=0.25),
            "best_single": _v3_gate_metrics(mean=0.10, p95=0.20),
            "static_weighted_downweight": _v3_gate_metrics(mean=0.11, p95=0.21),
        },
        normal_policy_metrics={},
        hard_bucket_failed_gates=(),
    )

    assert "transform_error_mean_not_learnable_vs_best_single" in result["failed_gates"]
    failure = next(
        item
        for item in result["failures"]
        if item["gate"] == "transform_error_mean_not_learnable_vs_best_single"
    )
    assert failure["attribution"] == "ranker_or_runtime_feature_predictability_problem"
    assert failure["geometry_gate"] == "transform_error"
    assert "crop_center_error" in failure["geometry_gate_vocabulary"]
    assert "roll_error" in failure["geometry_gate_vocabulary"]
    assert "hull_iou" in failure["geometry_gate_vocabulary"]


def test_v3_learnability_gate_attributes_invalid_selection() -> None:
    result = scorer_eval_impl.v3_learnability_promotion_gates(
        scorer_policy_name="learned_quality_v3",
        comparison_metrics={
            "learned_quality_v3": _v3_gate_metrics(
                mean=0.05,
                p95=0.07,
                invalid_rate=0.10,
            ),
            "best_single": _v3_gate_metrics(mean=0.10, p95=0.12),
            "static_weighted_downweight": _v3_gate_metrics(mean=0.11, p95=0.13),
        },
        normal_policy_metrics={},
        hard_bucket_failed_gates=(),
    )

    assert "invalid_selection_rate_above_validity_detector_gate" in result["failed_gates"]
    failure = next(
        item
        for item in result["failures"]
        if item["gate"] == "invalid_selection_rate_above_validity_detector_gate"
    )
    assert failure["attribution"] == "validity_detector_or_runtime_feature_coverage_problem"


def test_v3_learnability_gate_attributes_normal_bucket_regression() -> None:
    result = scorer_eval_impl.v3_learnability_promotion_gates(
        scorer_policy_name="learned_quality_v3",
        comparison_metrics={
            "learned_quality_v3": _v3_gate_metrics(mean=0.05, p95=0.07),
            "best_single": _v3_gate_metrics(mean=0.10, p95=0.12),
            "static_weighted_downweight": _v3_gate_metrics(mean=0.11, p95=0.13),
        },
        normal_policy_metrics={
            "learned_quality_v3": _v3_gate_metrics(mean=0.20, p95=0.25),
            "best_single": _v3_gate_metrics(mean=0.10, p95=0.20),
            "static_weighted_downweight": _v3_gate_metrics(mean=0.11, p95=0.21),
        },
        hard_bucket_failed_gates=(),
    )

    assert "normal_bucket_no_regression" in result["failed_gates"]
    failure = next(
        item for item in result["failures"] if item["gate"] == "normal_bucket_no_regression"
    )
    assert failure["attribution"] == "hard_case_query_weighting_or_contested_subset_overfit"


def test_v3_learnability_gate_requires_hard_buckets_to_pass() -> None:
    result = scorer_eval_impl.v3_learnability_promotion_gates(
        scorer_policy_name="learned_quality_v3",
        comparison_metrics={
            "learned_quality_v3": _v3_gate_metrics(mean=0.05, p95=0.07),
            "best_single": _v3_gate_metrics(mean=0.10, p95=0.12),
            "static_weighted_downweight": _v3_gate_metrics(mean=0.11, p95=0.13),
        },
        normal_policy_metrics={},
        hard_bucket_failed_gates=("profile_catastrophic_failures_above_hard_bucket_gate",),
    )

    assert "hard_bucket_catastrophic_failure_gate_failed" in result["failed_gates"]
    failure = next(
        item
        for item in result["failures"]
        if item["gate"] == "hard_bucket_catastrophic_failure_gate_failed"
    )
    assert failure["geometry_gate"] == "catastrophic_failure"
    assert "catastrophic_failure" in failure["geometry_gate_vocabulary"]
