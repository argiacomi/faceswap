#!/usr/bin/env python3
"""Tests for the production landmark runtime resolver."""

from __future__ import annotations

import numpy as np

from lib.landmarks.ensemble import runtime_resolver
from lib.landmarks.ensemble.runtime_resolver import (
    ModelPrediction,
    RuntimeResolverConfig,
    resolve_runtime,
)


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


def test_runtime_resolver_applies_v7_bucket_priority(monkeypatch) -> None:
    """Large-yaw-left bucket priority prefers SPIGA when it survives vetoes."""
    monkeypatch.setattr(
        runtime_resolver, "hard_slice_label", lambda *args, **kwargs: "large_yaw_left"
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
    assert result.metadata["bucket"] == "large_yaw_left"
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
        runtime_resolver, "hard_slice_label", lambda *args, **kwargs: "profile_right"
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
