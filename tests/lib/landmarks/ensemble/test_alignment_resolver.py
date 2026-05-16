#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.ensemble.alignment_resolver` (#78)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.ensemble.alignment_resolver import (
    AlignmentResolverConfig,
    AlignmentResolverError,
    CandidateInput,
    resolve_alignment_geometry,
)


def _truth_face() -> np.ndarray:
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


def _bbox() -> tuple[float, float, float, float]:
    return (40.0, 60.0, 160.0, 150.0)


def test_low_risk_route_uses_general_strategy() -> None:
    """Closely-agreeing candidates trigger the low-risk path."""
    truth = _truth_face()
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("spiga", truth + 0.5),
        CandidateInput("orformer", truth - 0.5),
    ]
    config = AlignmentResolverConfig(
        general_strategy="static_weighted",
        hard_case_strategy="static_weighted_downweight",
    )
    result = resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())
    assert result.risk_route == "low_risk"
    assert result.chosen_strategy == "static_weighted"
    assert result.rejected_models == ()
    assert result.geometry_confidence > 0.5
    assert result.alignment_landmarks.shape == (68, 2)


def test_high_risk_route_when_disagreement_exceeds_threshold() -> None:
    """Large disagreement routes to the hard-case strategy."""
    truth = _truth_face()
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("spiga", truth + np.array([20.0, 0.0], dtype="float32")),
        CandidateInput("orformer", truth + np.array([-20.0, 0.0], dtype="float32")),
    ]
    config = AlignmentResolverConfig(
        general_strategy="static_weighted",
        hard_case_strategy="static_weighted_downweight",
        high_disagreement_px=8.0,
    )
    result = resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())
    assert result.risk_route == "high_risk"
    assert result.chosen_strategy == "static_weighted_downweight"
    assert "high_disagreement" in result.geometry_flags


def test_invalid_candidates_are_rejected() -> None:
    """A cloud-collapsed candidate is dropped and remaining candidates fuse."""
    truth = _truth_face()
    collapsed = np.full_like(truth, fill_value=100.0)
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("spiga", truth + 0.5),
        CandidateInput("collapsed_model", collapsed),
    ]
    config = AlignmentResolverConfig()
    result = resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())
    assert "collapsed_model" in result.rejected_models
    assert "cloud_collapse" in result.geometry_flags
    # Rejection happened but other candidates were still valid → fusion succeeds.
    assert result.alignment_landmarks.shape == (68, 2)


def test_all_candidates_invalid_raises_resolver_error() -> None:
    """When every candidate is degenerate, the resolver hard-fails."""
    truth = _truth_face()
    collapsed = np.full_like(truth, fill_value=100.0)
    candidates = [
        CandidateInput("a", collapsed),
        CandidateInput("b", collapsed + 0.01),
    ]
    config = AlignmentResolverConfig()
    with pytest.raises(AlignmentResolverError):
        resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())


def test_unusual_bbox_aspect_triggers_hard_case_route() -> None:
    """Skewed detector bbox routes to the hard-case strategy."""
    truth = _truth_face()
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("spiga", truth + 0.5),
    ]
    config = AlignmentResolverConfig(
        general_strategy="static_weighted",
        hard_case_strategy="static_weighted_downweight",
        unusual_bbox_aspect_delta=0.3,
    )
    # Wide bbox: 240×80 covers the face (which spans y∈[60,150]) but has
    # aspect 3.0 → delta from 1.0 = 2.0, well above the 0.3 trigger.
    result = resolve_alignment_geometry(
        candidates,
        config=config,
        detector_bbox=(20.0, 70.0, 260.0, 150.0),
    )
    assert result.risk_route == "high_risk"
    assert "unusual_bbox_aspect" in result.geometry_flags


def test_partial_candidate_rejection_flags_risk_signal() -> None:
    """When some candidates are rejected, the resolver still reports a risk signal."""
    truth = _truth_face()
    flipped = truth.copy()
    # Crude vertical flip about y=100 → eyes-below-mouth in image coords, but
    # AlignedFace can still recover. Force a cloud collapse instead via shrink.
    flipped = truth.mean(axis=0) + (truth - truth.mean(axis=0)) * 0.05
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("collapsed", flipped),
        CandidateInput("spiga", truth + 0.4),
    ]
    config = AlignmentResolverConfig()
    result = resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())
    assert "partial_candidate_rejection" in result.geometry_flags
    assert "collapsed" in result.rejected_models
    assert result.risk_route in {"high_risk", "low_risk"}


def test_resolver_falls_back_to_plain_average_when_weights_missing() -> None:
    """A weighted strategy without static weights still produces a result via fallback."""
    truth = _truth_face()
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("spiga", truth + 0.5),
    ]
    # weighted_median requires weights, but we deliberately omit them so the
    # resolver's fusion-fallback path activates.
    config = AlignmentResolverConfig(
        general_strategy="plain_average",
        hard_case_strategy="weighted_median",
        fallback_strategy="plain_average",
        weights=None,
        high_disagreement_px=0.1,  # force high_risk
    )
    result = resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())
    # We forced the high-risk route but weighted_median has no weights, so the
    # resolver should still emit a usable result (via the equal-weights default).
    assert result.alignment_landmarks.shape == (68, 2)


def test_resolver_emits_debug_metadata() -> None:
    """The resolver surfaces enough metadata for downstream inspection."""
    truth = _truth_face()
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("spiga", truth + 5.0),
    ]
    config = AlignmentResolverConfig()
    result = resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())
    assert "max_disagreement_px" in result.debug_metadata
    assert "per_model_disagreement_px" in result.debug_metadata
    assert "risk_signals" in result.debug_metadata
    assert result.active_models == ("hrnet", "spiga")


def test_resolver_returns_distinct_landmark_outputs() -> None:
    """The fused landmarks differ from any single candidate when fusion happens."""
    truth = _truth_face()
    shifted = truth + np.array([2.0, 1.0], dtype="float32")
    candidates = [
        CandidateInput("hrnet", truth),
        CandidateInput("spiga", shifted),
    ]
    config = AlignmentResolverConfig(general_strategy="plain_average")
    result = resolve_alignment_geometry(candidates, config=config, detector_bbox=_bbox())
    # Plain average of truth + shifted = midpoint, not equal to either input.
    assert not np.allclose(result.alignment_landmarks, truth)
    assert not np.allclose(result.alignment_landmarks, shifted)
