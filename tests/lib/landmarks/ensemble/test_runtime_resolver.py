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


def _candidate_metrics(yaws: dict[str, float]) -> dict[str, CandidateMetrics]:
    return {
        name: CandidateMetrics(roll_degrees=0.0, yaw_degrees=yaw, pitch_degrees=None)
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
        metrics=_candidate_metrics(candidate_yaws),
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


def test_runtime_resolver_uses_eye_visual_evidence_for_runtime_bucket() -> None:
    """A one-eye visual crop can promote a side-profile runtime bucket."""
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

    assert result.metadata["runtime_bucket"] == "profile_left"
    assert result.metadata["bucket"] == "profile_left"
    assert _LEGACY_LANDMARK_POSE_BUCKET_KEY not in result.metadata
    assert result.metadata["left_eye_visual_score"] > result.metadata["right_eye_visual_score"]
    assert result.metadata["eye_visibility_asymmetry"] > 0.12
    assert result.metadata["runtime_bucket_severity_source"] == "eye_visibility_asymmetry"


def test_runtime_bucket_routes_profile_like_00012_to_profile_left(monkeypatch) -> None:
    """00012-style positive pose yaw and far-side instability is profile-left."""
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

    assert result.bucket == "profile_left"
    assert result.features["runtime_bucket_side"] == "left"
    assert result.features["runtime_bucket_side_source"] == "nose_offset"
    assert result.features["runtime_bucket_severity"] == "profile"
    assert result.features["runtime_bucket_severity_source"] == "candidate_instability"
    assert result.features["candidate_yaw_disagreement"] > 120.0


def test_runtime_bucket_uses_image_facing_side_convention_for_00114(monkeypatch) -> None:
    """00114-style weak image geometry still resolves positive pose yaw to image-left."""
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

    assert result.bucket == "large_yaw_left"
    assert result.features["runtime_bucket_side"] == "left"
    assert result.features["runtime_bucket_side_source"] == "nose_offset"
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
    """Strong silhouette/nose geometry is sufficient profile evidence."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.38,
        nose_offset_from_face_center=-0.72,
        mouth_nose_jaw_asymmetry=0.45,
        landmark_pose_yaw=-18.0,
        candidate_yaws={"hrnet": -28.0, "spiga": -24.0, "orformer": -15.0},
        max_disagreement_px=45.0,
    )

    assert result.bucket == "profile_right"
    assert result.features["runtime_bucket_side"] == "right"
    assert result.features["runtime_bucket_severity_source"] == "image_geometry"


def test_runtime_bucket_keeps_borderline_profile_evidence_as_large_yaw(monkeypatch) -> None:
    """Borderline side-on cues stay large-yaw until instability crosses profile evidence."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.11,
        nose_offset_from_face_center=0.11,
        mouth_nose_jaw_asymmetry=0.21,
        landmark_pose_yaw=19.5,
        candidate_yaws={"hrnet": 55.0, "spiga": 15.0, "orformer": -62.0},
        max_disagreement_px=40.0,
    )

    assert result.bucket == "large_yaw_left"
    assert result.features["runtime_bucket_severity"] == "large_yaw"


def test_runtime_bucket_uses_rolled_large_yaw_family(monkeypatch) -> None:
    """Hard roll plus large-yaw uses the canonical rolled yaw bucket family."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.16,
        nose_offset_from_face_center=0.18,
        mouth_nose_jaw_asymmetry=0.18,
        landmark_pose_yaw=31.0,
        candidate_yaws={"hrnet": 44.0, "spiga": 20.0, "orformer": -22.0},
        max_disagreement_px=35.0,
        roll_estimate=36.0,
    )

    assert result.bucket == "rolled_large_yaw_left"


def test_runtime_bucket_uses_rolled_profile_family(monkeypatch) -> None:
    """Hard roll plus profile uses the canonical rolled profile bucket family."""
    result = _runtime_bucket_for_production_signals(
        monkeypatch,
        image_geometry_yaw_signal=-0.38,
        nose_offset_from_face_center=-0.72,
        mouth_nose_jaw_asymmetry=0.45,
        landmark_pose_yaw=-18.0,
        candidate_yaws={"hrnet": -28.0, "spiga": -24.0, "orformer": -15.0},
        max_disagreement_px=45.0,
        roll_estimate=34.0,
    )

    assert result.bucket == "rolled_profile_right"


def test_runtime_bucket_routes_known_visual_left_profiles_to_profile_left(monkeypatch) -> None:
    """Known profile-right exports face image-left and should resolve profile-left."""
    cases = (
        ("00015", 0.380, 0.491, -0.047, -37.395, 65.761, 124.710),
        ("00025", 0.382, 1.101, -1.000, -49.134, 65.159, 124.414),
        ("00036", 0.161, 0.617, -0.990, -59.495, -79.521, 146.152),
        ("00040", -0.042, 0.098, -0.210, -31.619, -77.617, 138.380),
        ("00106", 0.272, 0.810, -0.757, -38.291, -73.171, 132.394),
        ("00109", 0.130, 0.551, -0.533, 15.570, -76.759, 131.690),
        ("00113", 0.092, 0.395, -0.317, -1.516, -78.386, 132.493),
    )
    for sample_id, image_yaw, nose, jaw, pose_yaw, dominant_yaw, yaw_spread in cases:
        other_yaw = dominant_yaw - yaw_spread if dominant_yaw > 0 else dominant_yaw + yaw_spread
        result = _runtime_bucket_for_production_signals(
            monkeypatch,
            image_geometry_yaw_signal=image_yaw,
            nose_offset_from_face_center=nose,
            mouth_nose_jaw_asymmetry=jaw,
            landmark_pose_yaw=pose_yaw,
            candidate_yaws={"dominant": dominant_yaw, "other": other_yaw},
            max_disagreement_px=190.0,
        )

        assert result.bucket == "profile_left", sample_id
        assert result.features["runtime_bucket_side"] == "left", sample_id
