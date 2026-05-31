#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.evaluation.signal_validation` (#80)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.evaluation.geometry_signals import AlignmentSummary
from lib.landmarks.evaluation.signal_validation import (
    SELECTORS,
    TRUTH_FREE_SELECTORS,
    CandidateRecord,
    attach_consensus_signals,
    consensus_alignment_summary,
    evaluate_selector,
    evaluate_selectors,
    evaluate_signals,
    label_bad_candidates,
    selector_pick,
    tag_oracle,
    validate_signal,
)


def _record(
    sample: str,
    label: str,
    *,
    geometry: float,
    nme: float = 0.05,
    transform: float = 0.01,
    hull_iou: float = 0.9,
    hard_slice: str = "frontal",
    is_baseline: bool = False,
) -> CandidateRecord:
    return CandidateRecord(
        sample_id=sample,
        dataset="ds",
        condition=hard_slice,
        hard_slice=hard_slice,
        candidate_label=label,
        is_baseline=is_baseline,
        geometry_score=geometry,
        nme=nme,
        transform_normalized=transform,
        crop_center_normalized=transform,
        roll_degrees_delta=transform * 100.0,
        hull_iou=hull_iou,
        catastrophic=False,
    )


def test_tag_oracle_marks_lowest_geometry_score_per_sample() -> None:
    """Each sample's best candidate is flagged ``is_oracle=True``."""
    records = [
        _record("a", "hrnet", geometry=0.05),
        _record("a", "ensemble", geometry=0.10),
        _record("b", "hrnet", geometry=0.20),
        _record("b", "ensemble", geometry=0.05),
    ]
    tagged = tag_oracle(records)
    oracle_map = {r.sample_id: r.candidate_label for r in tagged if r.is_oracle}
    assert oracle_map == {"a": "hrnet", "b": "ensemble"}


def test_label_bad_candidates_uses_margin_vs_per_sample_oracle() -> None:
    """A candidate beyond the per-sample oracle + margin is labeled bad."""
    records = [
        _record("a", "hrnet", geometry=0.05),
        _record("a", "ensemble", geometry=0.20),  # +0.15 from oracle, exceeds margin
        _record("b", "hrnet", geometry=0.20),
        _record("b", "ensemble", geometry=0.05),
    ]
    labels = label_bad_candidates(records, margin=0.05)
    assert labels.tolist() == [False, True, True, False]


def test_validate_signal_reports_precision_recall_and_auc() -> None:
    """Signals that strongly correlate with bad candidates score high precision/recall/AUC."""
    records = []
    for idx in range(10):
        records.append(_record(f"s{idx}", "good", geometry=0.05, transform=0.02))
        records.append(_record(f"s{idx}", "bad", geometry=0.30, transform=0.20))
    report = validate_signal(records, signal="transform_normalized", margin=0.05)
    assert report.direction == "higher_is_worse"
    assert report.precision > 0.5
    assert report.recall > 0.5
    assert report.auc >= 0.5


def test_validate_signal_supports_lower_is_worse_signals() -> None:
    """``hull_iou`` is treated as lower-is-worse and still produces a useful AUC."""
    records = []
    for idx in range(10):
        records.append(_record(f"s{idx}", "good", geometry=0.05, hull_iou=0.95))
        records.append(_record(f"s{idx}", "bad", geometry=0.30, hull_iou=0.5))
    report = validate_signal(records, signal="hull_iou", margin=0.05)
    assert report.direction == "lower_is_worse"
    assert report.auc >= 0.5


def test_evaluate_signals_returns_one_report_per_signal() -> None:
    """The convenience runner produces a SignalReport per named signal."""
    records = [
        _record("a", "good", geometry=0.05),
        _record("a", "bad", geometry=0.30),
        _record("b", "good", geometry=0.05),
        _record("b", "bad", geometry=0.30),
    ]
    reports = evaluate_signals(records)
    names = [r.name for r in reports]
    assert "nme" in names
    assert "transform_normalized" in names
    assert "hull_iou" in names
    assert "geometry_score" in names


def test_selector_pick_uses_signal_direction_correctly() -> None:
    """Lower-is-worse selectors pick the candidate with the highest signal value."""
    records = [
        _record("a", "low_iou", geometry=0.10, hull_iou=0.4),
        _record("a", "high_iou", geometry=0.05, hull_iou=0.95),
    ]
    chosen = selector_pick(records, signal="hull_iou", direction="lower_is_worse")
    assert chosen.candidate_label == "high_iou"


def test_evaluate_selector_reports_oracle_match_rate_and_gap() -> None:
    """A selector matching the oracle on every sample scores 1.0."""
    records = [
        _record("a", "hrnet", geometry=0.05, nme=0.04),
        _record("a", "ensemble", geometry=0.10, nme=0.06),
        _record("b", "hrnet", geometry=0.05, nme=0.04),
        _record("b", "ensemble", geometry=0.10, nme=0.06),
    ]
    report = evaluate_selector(
        records, name="lowest_nme", signal="nme", direction="higher_is_worse"
    )
    assert report.sample_count == 2
    assert report.oracle_match_rate == pytest.approx(1.0)
    assert report.mean_score_gap_vs_oracle == pytest.approx(0.0)


def test_evaluate_selectors_yields_one_report_per_named_selector() -> None:
    """Default SELECTORS tuple drives a full ablation sweep."""
    records = [_record(f"s{i}", "a", geometry=0.05) for i in range(5)] + [
        _record(f"s{i}", "b", geometry=0.10) for i in range(5)
    ]
    reports = evaluate_selectors(records)
    assert {r.name for r in reports} == {name for name, _, _ in SELECTORS}
    for report in reports:
        assert 0.0 <= report.oracle_match_rate <= 1.0


def test_per_bucket_breakdown_groups_by_hard_slice() -> None:
    """Selector reports separate oracle-match rate per hard-slice bucket."""
    records = [
        _record("a", "hrnet", geometry=0.05, hard_slice="profile_left", nme=0.04),
        _record("a", "ensemble", geometry=0.10, hard_slice="profile_left", nme=0.06),
        _record("b", "hrnet", geometry=0.20, hard_slice="profile_right", nme=0.04),
        _record("b", "ensemble", geometry=0.05, hard_slice="profile_right", nme=0.06),
    ]
    report = evaluate_selector(
        records, name="lowest_nme", signal="nme", direction="higher_is_worse"
    )
    assert "profile_left" in report.per_bucket
    assert "profile_right" in report.per_bucket
    # Lowest-NME picks "hrnet" on both samples; it matches oracle on sample a
    # but not on sample b, so per-bucket match rate diverges.
    assert report.per_bucket["profile_left"]["oracle_match_rate"] == pytest.approx(1.0)
    assert report.per_bucket["profile_right"]["oracle_match_rate"] == pytest.approx(0.0)


def test_tag_oracle_is_idempotent() -> None:
    """Re-tagging an already-oracle-tagged set keeps the same labels."""
    records = [
        _record("a", "hrnet", geometry=0.05),
        _record("a", "ensemble", geometry=0.10),
    ]
    first = tag_oracle(records)
    second = tag_oracle(first)
    assert [r.is_oracle for r in first] == [r.is_oracle for r in second]


def test_validate_signal_handles_empty_records() -> None:
    """Empty input produces a zero-precision/recall/AUC report rather than crashing."""
    report = validate_signal([], signal="nme")
    assert report.precision == pytest.approx(0.0)
    assert report.recall == pytest.approx(0.0)
    assert report.auc == pytest.approx(0.0)


def _make_summary(
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
    roi_center: tuple[float, float] = (100.0, 100.0),
    roll: float = 0.0,
) -> AlignmentSummary:
    """Build a minimal AlignmentSummary for consensus-math tests."""
    half = 50.0
    cx, cy = roi_center
    roi = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype="float64",
    )
    matrix = np.array(
        [[scale, 0.0, translation[0]], [0.0, scale, translation[1]]], dtype="float64"
    )
    landmarks = np.zeros((68, 2), dtype="float64")  # type: ignore[var-annotated]
    return AlignmentSummary(
        matrix=matrix,
        roi=roi,
        aligned_landmarks=landmarks,
        normalized_landmarks=landmarks,
        average_distance=0.0,
        relative_eye_mouth_position=0.5,
        pitch=0.0,
        yaw=0.0,
        roll=roll,
        scale=scale,
        rotation_degrees=rotation,
        translation=translation,
    )


def test_consensus_alignment_summary_returns_per_component_median() -> None:
    """Consensus picks the median of each scalar field and array column."""
    summaries = [
        _make_summary(scale=1.0, translation=(0.0, 0.0), roll=0.0),
        _make_summary(scale=2.0, translation=(10.0, 10.0), roll=5.0),
        _make_summary(scale=3.0, translation=(20.0, 20.0), roll=10.0),
    ]
    consensus = consensus_alignment_summary(summaries)
    assert consensus.scale == pytest.approx(2.0)
    assert consensus.translation == pytest.approx((10.0, 10.0))
    assert consensus.roll == pytest.approx(5.0)
    np.testing.assert_allclose(consensus.matrix, summaries[1].matrix)


def test_consensus_alignment_summary_is_robust_to_one_outlier() -> None:
    """Median consensus does not collapse to a single catastrophic candidate."""
    good_a = _make_summary(scale=1.0, translation=(0.0, 0.0))
    good_b = _make_summary(scale=1.0, translation=(0.0, 0.0))
    catastrophic = _make_summary(scale=10.0, translation=(500.0, 500.0))
    consensus = consensus_alignment_summary([good_a, good_b, catastrophic])
    assert consensus.scale == pytest.approx(1.0)
    assert consensus.translation == pytest.approx((0.0, 0.0))


def test_consensus_alignment_summary_requires_at_least_one_summary() -> None:
    """Empty input is a programmer error, not a silent 0-vector."""
    with pytest.raises(ValueError):
        consensus_alignment_summary([])


def test_attach_consensus_signals_fills_truth_free_fields() -> None:
    """Each candidate gets a delta-to-consensus filled into the truth-free fields."""
    records = [
        _record("a", "good", geometry=0.05),
        _record("a", "bad", geometry=0.20),
    ]
    summaries_by_sample = {
        "a": {
            "good": _make_summary(translation=(0.0, 0.0), roi_center=(100.0, 100.0), roll=0.0),
            "bad": _make_summary(translation=(40.0, 40.0), roi_center=(140.0, 140.0), roll=20.0),
        }
    }
    normalizer_by_sample = {"a": 100.0}

    attached = attach_consensus_signals(
        records,
        summaries_by_sample=summaries_by_sample,
        normalizer_by_sample=normalizer_by_sample,
    )

    by_label = {r.candidate_label: r for r in attached}
    # With two candidates, the median equals each end of the pair, so both
    # deltas to consensus are equal in magnitude. What matters is that the
    # numbers are populated and non-degenerate.
    assert by_label["good"].transform_consensus_distance > 0.0
    assert by_label["bad"].transform_consensus_distance > 0.0
    assert by_label["good"].crop_center_consensus_distance > 0.0
    assert by_label["bad"].roll_consensus_delta == pytest.approx(10.0)


def test_attach_consensus_signals_picks_closer_candidate_to_median_with_three() -> None:
    """The candidate closest to the cohort median scores the lowest consensus distance."""
    records = [
        _record("a", "median", geometry=0.05),
        _record("a", "outlier_high", geometry=0.10),
        _record("a", "outlier_low", geometry=0.10),
    ]
    summaries_by_sample = {
        "a": {
            "median": _make_summary(translation=(10.0, 10.0)),
            "outlier_high": _make_summary(translation=(100.0, 100.0)),
            "outlier_low": _make_summary(translation=(-80.0, -80.0)),
        }
    }
    attached = attach_consensus_signals(
        records,
        summaries_by_sample=summaries_by_sample,
        normalizer_by_sample={"a": 100.0},
    )
    by_label = {r.candidate_label: r for r in attached}
    # The median candidate should be the closest to the consensus.
    assert (
        by_label["median"].transform_consensus_distance
        < by_label["outlier_high"].transform_consensus_distance
    )
    assert (
        by_label["median"].transform_consensus_distance
        < by_label["outlier_low"].transform_consensus_distance
    )


def test_attach_consensus_signals_passes_records_through_when_summary_missing() -> None:
    """Records without a matching summary remain unchanged (defaults stay at 0.0)."""
    record = _record("missing", "x", geometry=0.05)
    attached = attach_consensus_signals(
        [record],
        summaries_by_sample={},
        normalizer_by_sample={},
    )
    assert attached[0].transform_consensus_distance == pytest.approx(0.0)
    assert attached[0].crop_center_consensus_distance == pytest.approx(0.0)
    assert attached[0].roll_consensus_delta == pytest.approx(0.0)


def test_truth_free_selectors_pick_median_candidate() -> None:
    """The truth-free transform selector picks the candidate closest to the consensus."""
    records = [
        _record("a", "median", geometry=0.05),
        _record("a", "outlier_high", geometry=0.10),
        _record("a", "outlier_low", geometry=0.10),
    ]
    summaries_by_sample = {
        "a": {
            "median": _make_summary(translation=(10.0, 10.0)),
            "outlier_high": _make_summary(translation=(100.0, 100.0)),
            "outlier_low": _make_summary(translation=(-80.0, -80.0)),
        }
    }
    attached = attach_consensus_signals(
        records,
        summaries_by_sample=summaries_by_sample,
        normalizer_by_sample={"a": 100.0},
    )
    chosen = selector_pick(
        attached,
        signal="transform_consensus_distance",
        direction="higher_is_worse",
    )
    assert chosen.candidate_label == "median"


def test_evaluate_selectors_runs_truth_free_set() -> None:
    """The TRUTH_FREE_SELECTORS tuple is wired through evaluate_selectors."""
    records = [_record(f"s{i}", "a", geometry=0.05) for i in range(3)] + [
        _record(f"s{i}", "b", geometry=0.10) for i in range(3)
    ]
    summaries_by_sample = {
        f"s{i}": {
            "a": _make_summary(translation=(0.0, 0.0)),
            "b": _make_summary(translation=(50.0, 50.0)),
        }
        for i in range(3)
    }
    attached = attach_consensus_signals(
        records,
        summaries_by_sample=summaries_by_sample,
        normalizer_by_sample={f"s{i}": 100.0 for i in range(3)},
    )
    reports = evaluate_selectors(attached, selectors=TRUTH_FREE_SELECTORS)
    names = {r.name for r in reports}
    assert names == {name for name, _, _ in TRUTH_FREE_SELECTORS}
