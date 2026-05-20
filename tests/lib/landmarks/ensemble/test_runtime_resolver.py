#!/usr/bin/env python3
"""Tests for the production landmark runtime resolver."""

from __future__ import annotations

import numpy as np

from lib.landmarks.coordinates import roi_to_matrix
from lib.landmarks.ensemble import runtime_resolver
from lib.landmarks.ensemble.runtime_resolver import (
    CandidateMetrics,
    CandidateRecord,
    ModelPrediction,
    RuntimeBucketResult,
    RuntimeResolverConfig,
    resolve_runtime,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverScorer,
    write_runtime_resolver_scorer,
)

_LEGACY_LANDMARK_POSE_BUCKET_KEY = "_".join(("landmark", "pose", "bucket"))


def test_runtime_bucket_family_is_canonical() -> None:
    """Runtime routing only advertises the supported production bucket family."""
    assert set(runtime_resolver.BUCKET_PRIORITIES) <= runtime_resolver.RUNTIME_BUCKETS
    assert {
        "frontal",
        "intermediate",
        "large_yaw_left",
        "large_yaw_right",
        "profile_left",
        "profile_right",
        "large_roll",
        "extreme_roll",
        "rolled_large_yaw_left",
        "rolled_large_yaw_right",
        "rolled_profile_left",
        "rolled_profile_right",
    } == runtime_resolver.RUNTIME_BUCKETS


def _face() -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")
    points[0:17, 0] = np.linspace(40, 160, 17)
    points[0:17, 1] = 120 + 30 * np.sin(np.linspace(0, np.pi, 17))
    points[17:22, 0] = np.linspace(50, 90, 5)
    points[17:22, 1] = 70
    points[22:27, 0] = np.linspace(110, 150, 5)
    points[22:27, 1] = 70
    points[27:36, 0] = 100
    points[27:36, 1] = np.linspace(75, 110, 9)
    points[36:42, 0] = np.linspace(60, 80, 6)
    points[36:42, 1] = 85
    points[42:48, 0] = np.linspace(120, 140, 6)
    points[42:48, 1] = 85
    points[48:60, 0] = np.linspace(70, 130, 12)
    points[48:60, 1] = 130
    points[60:68, 0] = np.linspace(80, 120, 8)
    points[60:68, 1] = 130
    return points


def _candidate_metrics(
    yaws: dict[str, float],
    *,
    rolls: dict[str, float] | None = None,
) -> dict[str, CandidateMetrics]:
    return {
        name: CandidateMetrics(
            roll_degrees=0.0 if rolls is None else rolls.get(name, 0.0),
            yaw_degrees=yaw,
            pitch_degrees=None,
        )
        for name, yaw in yaws.items()
    }


def _runtime_bucket_for_production_signals(
    monkeypatch,
    *,
    image_geometry_yaw_signal: float,
    nose_offset_from_face_center: float,
    mouth_nose_jaw_asymmetry: float,
    landmark_pose_yaw: float,
    candidate_yaws: dict[str, float],
    candidate_rolls: dict[str, float] | None = None,
    max_disagreement_px: float = 190.0,
    roll_estimate: float = 0.0,
) -> RuntimeBucketResult:
    monkeypatch.setattr(
        runtime_resolver,
        "_image_geometry_yaw_signal",
        lambda *args, **kwargs: (
            image_geometry_yaw_signal,
            nose_offset_from_face_center,
            mouth_nose_jaw_asymmetry,
        ),
    )
    candidates = [
        CandidateRecord(name=name, landmarks=_face(), is_fusion=False, contributing_models=(name,))
        for name in candidate_yaws
    ]
    return runtime_resolver.infer_runtime_bucket(
        image_crop=None,
        crop_to_frame_matrix=None,
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
        candidates=candidates,
        metrics=_candidate_metrics(candidate_yaws, rolls=candidate_rolls),
        yaw_estimate=landmark_pose_yaw,
        roll_estimate=roll_estimate,
        max_disagreement_px=max_disagreement_px,
        hard_roll_degrees=30.0,
    )


def test_runtime_resolver_applies_v7_bucket_priority(monkeypatch) -> None:
    """Large-yaw-left bucket priority prefers SPIGA when it survives vetoes."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="large_yaw_left", features={}),
    )
    base = _face()
    predictions = [
        ModelPrediction("hrnet", base + 0.2),
        ModelPrediction("spiga", base),
        ModelPrediction("orformer", base + 0.4),
    ]

    result = resolve_runtime(
        predictions,
        RuntimeResolverConfig(
            weights={name: [1.0 / 3.0] * 68 for name in ("hrnet", "spiga", "orformer")}
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )

    assert result.selected_candidate == "spiga"
    assert result.metadata["runtime_bucket"] == "large_yaw_left"
    assert result.metadata["bucket"] == "large_yaw_left"
    assert _LEGACY_LANDMARK_POSE_BUCKET_KEY not in result.metadata
    assert result.metadata["candidate_priority"][0] == "spiga"
    assert result.metadata["model_predictions_available"] == {
        "hrnet": True,
        "spiga": True,
        "orformer": True,
    }
    assert set(result.metadata["cloud_area_ratio"]) >= {
        "hrnet",
        "spiga",
        "orformer",
        "static_weighted",
    }


def test_runtime_resolver_metadata_contains_requested_debug_fields(monkeypatch) -> None:
    """Resolver metadata is immediately useful for profile extraction debugging."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="profile_right", features={}),
    )
    base = _face()

    result = resolve_runtime(
        [
            ModelPrediction("spiga", base),
            ModelPrediction("orformer", base + 0.25),
        ],
        RuntimeResolverConfig(weights={"spiga": [0.5] * 68, "orformer": [0.5] * 68}),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )

    for key in (
        "selected_candidate",
        "bucket",
        "runtime_bucket",
        "runtime_bucket_features",
        "candidate_priority",
        "vetoed",
        "veto_reasons",
        "roll_estimate",
        "yaw_estimate",
        "cloud_area_ratio",
        "landmark_consensus_distance",
        "model_predictions_available",
    ):
        assert key in result.metadata


def test_learned_quality_policy_scores_geometry_valid_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    """learned_quality_v1 chooses the lowest predicted risk instead of bucket priority."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="large_yaw_left", features={}),
    )
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=hrnet", "candidate_name=spiga"),
            coefficients=(-5.0, 5.0),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )
    base = _face()

    result = resolve_runtime(
        [
            ModelPrediction("hrnet", base + 0.1),
            ModelPrediction("spiga", base),
            ModelPrediction("orformer", base + 0.2),
        ],
        RuntimeResolverConfig(
            policy="learned_quality_v1",
            scorer_path=str(scorer_path),
            weights={name: [1.0 / 3.0] * 68 for name in ("hrnet", "spiga", "orformer")},
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )

    assert result.selected_candidate == "hrnet"
    assert result.metadata["policy"] == "learned_quality_v1"
    assert (
        result.metadata["selected_candidate_score"] < result.metadata["candidate_scores"]["spiga"]
    )
    assert result.metadata["candidate_risk_rank"][0] == "hrnet"
    assert result.metadata["scorer_path"] == str(scorer_path)
    assert result.metadata["fallback_used"] is False


def test_learned_quality_policy_falls_back_to_hrnet_when_all_risks_high(
    monkeypatch,
    tmp_path,
) -> None:
    """High-risk scorer outputs prefer HRNet over a less-bad fusion candidate."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="frontal", features={}),
    )
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=static_weighted",),
            coefficients=(-0.25,),
            intercept=1.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )
    base = _face()

    result = resolve_runtime(
        [
            ModelPrediction("hrnet", base + 0.1),
            ModelPrediction("spiga", base),
            ModelPrediction("orformer", base + 0.2),
        ],
        RuntimeResolverConfig(
            policy="learned_quality_v1",
            scorer_path=str(scorer_path),
            weights={name: [1.0 / 3.0] * 68 for name in ("hrnet", "spiga", "orformer")},
            safe_fallback_min_delta=-1.0,
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )

    assert result.selected_candidate == "hrnet"
    assert (
        result.metadata["candidate_scores"]["static_weighted"]
        < result.metadata["candidate_scores"]["hrnet"]
    )
    assert result.metadata["candidate_scores"]["static_weighted"] > 0.50
    assert result.metadata["scorer_safe_fallback_used"] is True
    assert result.metadata["fallback_reason"] == "scorer_high_risk_safe_fallback"


def test_all_candidates_vetoed_uses_deterministic_hard_case_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    """All-vetoed fallback ignores scorer ranking and prefers static downweight."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="large_yaw_left", features={}),
    )
    monkeypatch.setattr(
        runtime_resolver,
        "_shape_reasons",
        lambda *_args, **_kwargs: ("forced_test_veto",),
    )
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=spiga", "candidate_name=static_weighted_downweight"),
            coefficients=(-8.0, 8.0),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )
    base = _face()

    result = resolve_runtime(
        [
            ModelPrediction("hrnet", base + 0.1),
            ModelPrediction("spiga", base),
            ModelPrediction("orformer", base + 0.2),
        ],
        RuntimeResolverConfig(
            policy="learned_quality_v1",
            scorer_path=str(scorer_path),
            weights={name: [1.0 / 3.0] * 68 for name in ("hrnet", "spiga", "orformer")},
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )

    assert result.selected_candidate == "static_weighted_downweight"
    assert (
        result.metadata["candidate_scores"]["spiga"]
        < result.metadata["candidate_scores"]["static_weighted_downweight"]
    )
    assert result.metadata["fallback_reason"] == "all_candidates_vetoed"
    assert result.metadata["scorer_safe_fallback_used"] is False
    assert result.metadata["replacement_candidate"] == "static_weighted_downweight"
    assert result.metadata["all_candidates_vetoed_count"] == 8
    assert set(result.metadata["vetoed_candidates"]) >= {
        "hrnet",
        "spiga",
        "orformer",
        "plain_average",
        "static_weighted_downweight",
    }
    assert result.metadata["candidate_veto_reasons"]["spiga"] == ["forced_test_veto"]


def test_all_candidates_vetoed_fallback_skips_nonfinite_candidates() -> None:
    """The deterministic all-vetoed fallback only returns finite candidates."""
    finite = _face()
    nonfinite = finite.copy()
    nonfinite[0, 0] = np.nan

    selected = runtime_resolver._all_candidates_vetoed_fallback(  # noqa: SLF001
        [
            CandidateRecord(
                name="static_weighted_downweight",
                landmarks=nonfinite,
                is_fusion=True,
                contributing_models=("hrnet", "spiga"),
            ),
            CandidateRecord(
                name="static_weighted_hard_drop",
                landmarks=finite,
                is_fusion=True,
                contributing_models=("hrnet", "spiga"),
            ),
        ],
        RuntimeResolverConfig(),
    )

    assert selected == "static_weighted_hard_drop"


def test_learned_quality_policy_rejects_consensus_collapse_fusion_for_best_single(
    monkeypatch,
    tmp_path,
) -> None:
    """Rolled hard slices reject consensus-collapse fusion and choose the best single."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(
            bucket="rolled_large_yaw_right",
            features={"candidate_yaw_disagreement": 95.0},
        ),
    )
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=static_weighted", "candidate_name=orformer"),
            coefficients=(-5.0, -4.0),
            intercept=0.0,
        ),
        tmp_path / "runtime_resolver_scorer.json",
    )
    base = _face()

    result = resolve_runtime(
        [
            ModelPrediction("hrnet", base + 0.1),
            ModelPrediction("spiga", base + 40.0),
            ModelPrediction("orformer", base + 20.0),
        ],
        RuntimeResolverConfig(
            policy="learned_quality_v1",
            scorer_path=str(scorer_path),
            weights={name: [1.0 / 3.0] * 68 for name in ("hrnet", "spiga", "orformer")},
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )

    assert result.selected_candidate == "orformer"
    assert (
        result.metadata["candidate_scores"]["static_weighted"]
        < result.metadata["candidate_scores"]["orformer"]
    )
    assert result.metadata["hard_slice_safe_fallback_used"] is True
    assert result.metadata["fallback_reason"] == "consensus_collapse_fusion_rejected"
    assert result.metadata["rejected_candidate"] == "static_weighted"
    assert result.metadata["replacement_candidate"] == "orformer"


def test_runtime_resolver_records_eye_visual_evidence_for_runtime_bucket() -> None:
    """Eye visual evidence is diagnostic and stays within the canonical bucket family."""
    base = _face()
    crop = np.full((200, 200, 3), 0.55, dtype="float32")
    for y in range(75, 96):
        for x in range(52, 90):
            crop[y, x] = 0.05 if (x + y) % 2 else 0.95

    result = resolve_runtime(
        [
            ModelPrediction("hrnet", base + 0.1),
            ModelPrediction("spiga", base),
            ModelPrediction("orformer", base + 0.2),
        ],
        RuntimeResolverConfig(
            weights={name: [1.0 / 3.0] * 68 for name in ("hrnet", "spiga", "orformer")}
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
        image_crop=crop,
        crop_to_frame_matrix=roi_to_matrix(np.array([0, 0, 200, 200], dtype="float32")),
    )

    assert result.metadata["runtime_bucket"] == "frontal"
    assert result.metadata["bucket"] == "frontal"
    assert _LEGACY_LANDMARK_POSE_BUCKET_KEY not in result.metadata
    assert result.metadata["left_eye_visual_score"] > result.metadata["right_eye_visual_score"]
    assert result.metadata["eye_visibility_asymmetry"] > 0.12
    assert result.metadata["runtime_bucket_eye_side"] == "right"


def test_runtime_bucket_demotes_single_model_profile_like_00012_to_large_yaw(
    monkeypatch,
) -> None:
    """Single-model yaw plus moderate shape support is large-yaw, not profile."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.03647917256402465,
        nose_offset_from_face_center=0.12693713183071523,
        mouth_nose_jaw_asymmetry=0.2492649984348901,
        landmark_pose_yaw=29.204193115234364,
        candidate_yaws={
            "hrnet": 71.3886489868164,
            "orformer": -56.88693618774414,
            "spiga": 25.79043197631836,
            "static_weighted": 27.650964736938477,
            "static_weighted_downweight": 30.757421493530273,
        },
    )

    assert result.bucket == "large_yaw_left"
    assert result.features["runtime_bucket_side"] == "left"
    assert result.features["runtime_bucket_side_source"] == "hrnet_yaw"
    assert result.features["runtime_bucket_severity"] == "large_yaw"
    assert result.features["runtime_bucket_severity_source"] == "yaw_evidence"
    assert result.features["candidate_yaw_disagreement"] > 120.0


def test_runtime_bucket_caps_weak_visual_shape_single_yaw_00114(monkeypatch) -> None:
    """Weak image/shape support caps a single-model yaw spike at frontal/intermediate."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=0.03766283153632713,
        nose_offset_from_face_center=0.1263110927574864,
        mouth_nose_jaw_asymmetry=0.12288589636609427,
        landmark_pose_yaw=6.2777396440506035,
        candidate_yaws={
            "hrnet": 53.39934158325195,
            "orformer": -28.21310806274414,
            "spiga": 11.22113037109375,
            "static_weighted": -4.392822265625,
            "static_weighted_downweight": -1.6221264600753784,
        },
    )

    assert result.bucket == "frontal"
    assert result.features["runtime_bucket_side"] == "left"
    assert result.features["runtime_bucket_side_source"] == "hrnet_yaw"
    assert result.features["runtime_bucket_severity_source"] == "low_yaw"
    assert result.features["candidate_yaw_disagreement"] > 80.0


def test_runtime_bucket_keeps_obvious_large_yaw_out_of_profile(monkeypatch) -> None:
    """Large-yaw keeps both-side support when profile instability is not present."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.16,
        nose_offset_from_face_center=0.18,
        mouth_nose_jaw_asymmetry=0.18,
        landmark_pose_yaw=31.0,
        candidate_yaws={"hrnet": 44.0, "spiga": 20.0, "orformer": -22.0},
        max_disagreement_px=35.0,
    )

    assert result.bucket == "large_yaw_left"
    assert result.features["runtime_bucket_severity"] == "large_yaw"
    assert result.features["runtime_bucket_severity_source"] == "yaw_evidence"


def test_runtime_bucket_routes_obvious_profile_by_image_geometry(monkeypatch) -> None:
    """Strong image yaw plus nose and jaw geometry is sufficient profile evidence."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.55,
        nose_offset_from_face_center=-0.72,
        mouth_nose_jaw_asymmetry=0.45,
        landmark_pose_yaw=-18.0,
        candidate_yaws={"hrnet": -28.0, "spiga": -24.0, "orformer": -15.0},
        max_disagreement_px=45.0,
    )

    assert result.bucket == "profile_right"
    assert result.features["runtime_bucket_side"] == "right"
    assert result.features["runtime_bucket_severity_source"] == "image_geometry"


def test_runtime_bucket_routes_profile_by_multi_model_yaw_agreement(monkeypatch) -> None:
    """Two non-fusion high-yaw models plus visual/shape support can create profile."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.13,
        nose_offset_from_face_center=0.10,
        mouth_nose_jaw_asymmetry=0.16,
        landmark_pose_yaw=28.0,
        candidate_yaws={"hrnet": -65.0, "spiga": -63.0, "orformer": -12.0},
        max_disagreement_px=35.0,
    )

    assert result.bucket == "profile_right"
    assert result.features["runtime_bucket_severity_source"] == "multi_model_yaw_agreement"
    assert result.features["profile_yaw_agreement"] is True
    assert result.features["profile_yaw_agreement_count"] == 2


def test_runtime_bucket_keeps_borderline_profile_evidence_as_large_yaw(monkeypatch) -> None:
    """Borderline side-on cues stay large-yaw until instability crosses profile evidence."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.11,
        nose_offset_from_face_center=0.11,
        mouth_nose_jaw_asymmetry=0.21,
        landmark_pose_yaw=19.5,
        candidate_yaws={"hrnet": 55.0, "spiga": 15.0, "orformer": -58.0},
        max_disagreement_px=40.0,
    )

    assert result.bucket == "large_yaw_left"
    assert result.features["runtime_bucket_severity"] == "large_yaw"


def test_runtime_bucket_demotes_candidate_instability_without_strong_yaw(
    monkeypatch,
) -> None:
    """Candidate disagreement alone cannot create profile or large-yaw buckets."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=0.07,
        nose_offset_from_face_center=0.10,
        mouth_nose_jaw_asymmetry=0.12,
        landmark_pose_yaw=18.0,
        candidate_yaws={
            "hrnet": 24.0,
            "spiga": 28.0,
            "orformer": 20.0,
            "static_weighted": -125.0,
        },
        max_disagreement_px=315.0,
    )

    assert result.bucket == "intermediate"
    assert result.features["runtime_bucket_severity"] == "intermediate"
    assert result.features["candidate_yaw_disagreement"] > 120.0


def test_runtime_bucket_allows_candidate_instability_profile_with_trusted_yaw(
    monkeypatch,
) -> None:
    """Candidate instability can support profile only with model side agreement and shape."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=0.50,
        nose_offset_from_face_center=0.31,
        mouth_nose_jaw_asymmetry=0.10,
        landmark_pose_yaw=22.0,
        candidate_yaws={
            "hrnet": 44.0,
            "spiga": 40.0,
            "orformer": 42.0,
            "static_weighted": -86.0,
        },
        max_disagreement_px=190.0,
    )

    assert result.bucket == "profile_left"
    assert result.features["runtime_bucket_severity_source"] == "candidate_instability"
    assert result.features["candidate_profile_yaw_agreement"] is True


def test_runtime_bucket_caps_listed_single_orformer_yaw_spike(monkeypatch) -> None:
    """00770-style single Orformer yaw spike stays intermediate with weak visual support."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=0.07436019444074712,
        nose_offset_from_face_center=0.10958707838149795,
        mouth_nose_jaw_asymmetry=-0.09854961423473703,
        landmark_pose_yaw=-24.902402400970463,
        candidate_yaws={
            "hrnet": -11.392191886901855,
            "spiga": -11.261571884155273,
            "orformer": -74.40572357177734,
        },
        max_disagreement_px=315.4366903184972,
    )

    assert result.bucket == "intermediate"
    assert result.features["runtime_bucket_severity_source"] == "weak_visual_shape_cap"
    assert result.features["profile_yaw_agreement"] is False


def test_runtime_bucket_demotes_02181_nose_only_profile_signal(monkeypatch) -> None:
    """02181-style high nose offset without jaw corroboration is large-yaw, not profile."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=0.3422617854807459,
        nose_offset_from_face_center=0.46056017070386623,
        mouth_nose_jaw_asymmetry=-0.10832770523689941,
        landmark_pose_yaw=-31.592052459716793,
        candidate_yaws={
            "hrnet": -9.942437171936035,
            "spiga": -37.887,
            "orformer": 8.0,
        },
        max_disagreement_px=203.0580158386968,
    )

    assert result.bucket == "large_yaw_right"
    assert result.features["runtime_bucket_severity_source"] == "yaw_evidence"


def test_runtime_bucket_uses_rolled_large_yaw_family(monkeypatch) -> None:
    """Hard roll plus large-yaw uses the canonical rolled yaw bucket family."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.16,
        nose_offset_from_face_center=0.18,
        mouth_nose_jaw_asymmetry=0.18,
        landmark_pose_yaw=31.0,
        candidate_yaws={"hrnet": 44.0, "spiga": 20.0, "orformer": -22.0},
        candidate_rolls={"hrnet": 34.0, "spiga": 35.0, "orformer": 3.0},
        max_disagreement_px=35.0,
        roll_estimate=36.0,
    )

    assert result.bucket == "rolled_large_yaw_left"


def test_runtime_bucket_uses_rolled_profile_family(monkeypatch) -> None:
    """Hard roll plus profile uses the canonical rolled profile bucket family."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.55,
        nose_offset_from_face_center=-0.72,
        mouth_nose_jaw_asymmetry=0.45,
        landmark_pose_yaw=-18.0,
        candidate_yaws={"hrnet": -28.0, "spiga": -24.0, "orformer": -15.0},
        candidate_rolls={"hrnet": 34.0, "spiga": 35.0, "orformer": 3.0},
        max_disagreement_px=45.0,
        roll_estimate=34.0,
    )

    assert result.bucket == "rolled_profile_right"


def test_runtime_bucket_demotes_unsupported_roll_estimate(monkeypatch) -> None:
    """A high consensus roll without two agreeing candidates cannot create roll buckets."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=0.08,
        nose_offset_from_face_center=0.02,
        mouth_nose_jaw_asymmetry=0.05,
        landmark_pose_yaw=2.0,
        candidate_yaws={"hrnet": 2.0, "spiga": 3.0, "orformer": -1.0},
        candidate_rolls={"hrnet": 116.0, "spiga": 2.0, "orformer": -1.0},
        max_disagreement_px=20.0,
        roll_estimate=116.0,
    )

    assert result.bucket == "intermediate"
    assert result.features["runtime_bucket_extreme_roll_supported"] is False
    assert result.features["runtime_bucket_extreme_roll_support_count"] == 1


def test_runtime_bucket_routes_known_visual_left_profiles_to_profile_left(monkeypatch) -> None:
    """Known strong-image-geometry profiles should resolve profile-left."""
    cases = (
        ("strong_profile_left_a", 0.550, 0.491, -0.450, -37.395, 65.761, 124.710),
        ("strong_profile_left_b", 0.560, 1.101, -1.000, -49.134, 65.159, 124.414),
    )
    for sample_id, image_yaw, nose, jaw, pose_yaw, dominant_yaw, yaw_spread in cases:
        other_yaw = dominant_yaw - yaw_spread if dominant_yaw > 0 else dominant_yaw + yaw_spread
        result = _runtime_bucket_for_production_signals(
            monkeypatch,
            image_geometry_yaw_signal=image_yaw,
            nose_offset_from_face_center=nose,
            mouth_nose_jaw_asymmetry=jaw,
            landmark_pose_yaw=pose_yaw,
            candidate_yaws={"hrnet": dominant_yaw, "spiga": other_yaw},
            max_disagreement_px=190.0,
        )

        assert result.bucket == "profile_left", sample_id
        assert result.features["runtime_bucket_side"] == "left", sample_id
