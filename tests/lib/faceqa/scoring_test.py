#!/usr/bin/env python3
"""Tests for FaceQA readiness scoring."""

from __future__ import annotations

import json

from lib.align.faceset_qa import FaceQARecord
from lib.faceqa.coverage import compute_coverage
from lib.faceqa.readiness import generate_readiness_report
from lib.faceqa.scoring import (
    COMPONENT_WEIGHTS,
    MIN_SAMPLES_PER_BIN,
    RISK_SCORE_THRESHOLD,
    STRENGTH_SCORE_THRESHOLD,
    compute_readiness_scores,
)


def _balanced_records(per_bin: int = 6) -> list[FaceQARecord]:
    """Return records spread across a broad range of pose/expression/lighting bins."""
    yaw_values = [-75.0, -45.0, -20.0, 0.0, 20.0, 45.0, 75.0]
    pitch_values = [-40.0, -20.0, 0.0, 20.0, 40.0]
    expression_buckets = (
        "neutral",
        "slight_open",
        "talking_open",
        "smile",
        "eyes_closed",
        "expressive",
    )
    lighting_buckets = (
        ("dark", 30.0),
        ("overexposed", 240.0),
        ("flat_frontal", 128.0),
        ("high_contrast", 128.0),
    )
    records: list[FaceQARecord] = []
    counter = 0
    for yaw in yaw_values:
        for pitch in pitch_values:
            for _ in range(per_bin):
                exp = expression_buckets[counter % len(expression_buckets)]
                light_bucket, luminance = lighting_buckets[counter % len(lighting_buckets)]
                contrast = 80.0 if light_bucket == "high_contrast" else 10.0
                records.append(
                    FaceQARecord(
                        frame=f"f{counter}.png",
                        face_index=0,
                        yaw=yaw,
                        pitch=pitch,
                        expression_bucket=exp,
                        mean_luminance=luminance,
                        contrast=contrast,
                        left_right_ratio=1.0,
                        top_bottom_ratio=1.0,
                        color_warmth=0.0,
                    )
                )
                counter += 1
    return records


def test_compute_readiness_scores_is_deterministic() -> None:
    """Calling compute_readiness_scores twice should yield identical results."""
    records = _balanced_records(per_bin=6)
    coverage = compute_coverage(records)

    first = compute_readiness_scores(coverage).to_dict()
    second = compute_readiness_scores(coverage).to_dict()

    assert first == second


def test_compute_readiness_scores_components_present_and_explainable() -> None:
    """Each component score must decompose into entropy + occupied + min-sample."""
    records = _balanced_records(per_bin=6)
    coverage = compute_coverage(records)

    scores = compute_readiness_scores(coverage)

    assert set(scores.components.keys()) == {"pose", "expression", "lighting", "quality"}
    for name, component in scores.components.items():
        assert 0.0 <= component.score <= 100.0
        assert 0.0 <= component.weight <= 1.0
        if name != "quality":
            assert 0.0 <= component.entropy_coverage <= 1.0
            assert 0.0 <= component.occupied_coverage <= 1.0
            assert 0.0 <= component.min_sample_coverage <= 1.0


def test_compute_readiness_scores_overall_is_weighted_average() -> None:
    """The overall readiness score should be the component-weighted average."""
    records = _balanced_records(per_bin=6)
    coverage = compute_coverage(records)

    scores = compute_readiness_scores(coverage)
    expected = sum(comp.score * comp.weight for comp in scores.components.values())

    assert abs(scores.overall_readiness_score - round(expected, 2)) < 1e-6


def test_compute_readiness_scores_handles_empty_coverage() -> None:
    """Empty coverage should produce a zero-score report without errors."""
    coverage = compute_coverage([])

    scores = compute_readiness_scores(coverage)

    assert scores.overall_readiness_score == 0.0
    assert scores.confidence == 0.0
    for component in scores.components.values():
        assert component.score == 0.0


def test_compute_readiness_scores_flags_strong_and_weak_components() -> None:
    """Balanced records should produce strong pose; sparse records should produce risks."""
    rich_coverage = compute_coverage(_balanced_records(per_bin=6))
    rich = compute_readiness_scores(rich_coverage)
    assert any("pose" in entry for entry in rich.strengths)

    sparse_records = [
        FaceQARecord(
            frame=f"f{i}.png",
            face_index=0,
            yaw=0.0,
            pitch=0.0,
            expression_bucket="neutral",
            mean_luminance=128.0,
            contrast=10.0,
            left_right_ratio=1.0,
            top_bottom_ratio=1.0,
            color_warmth=0.0,
        )
        for i in range(20)
    ]
    sparse = compute_readiness_scores(compute_coverage(sparse_records))
    sparse_components = {
        c.name for c in sparse.components.values() if c.score < RISK_SCORE_THRESHOLD
    }
    assert {"pose", "expression", "lighting"}.issubset(sparse_components)
    assert sparse.expected_training_risks


def test_compute_readiness_scores_quality_penalizes_blur_and_dupes() -> None:
    """Per-frame quality issues should drive the quality score down."""
    records = [
        FaceQARecord(
            frame=f"f{i}.png",
            face_index=0,
            blur_score=0.5,
            resolution=[60, 60],
            duplicate_keep_recommendation="prune_candidate",
            identity_quality_flag="outlier",
            yaw=0.0,
            pitch=0.0,
            expression_bucket="neutral",
            mean_luminance=128.0,
            contrast=10.0,
            left_right_ratio=1.0,
            top_bottom_ratio=1.0,
            color_warmth=0.0,
        )
        for i in range(20)
    ]
    coverage = compute_coverage(records)

    scores = compute_readiness_scores(coverage)

    assert scores.components["quality"].score < 60.0
    assert any(
        "overfit" in risk.lower() or "identity" in risk.lower()
        for risk in scores.expected_training_risks
    )


def test_min_samples_per_bin_constant_is_used() -> None:
    """A bin must reach MIN_SAMPLES_PER_BIN to count toward the min-samples score."""
    just_below = MIN_SAMPLES_PER_BIN - 1
    sparse = [
        FaceQARecord(frame=f"f{i}.png", face_index=0, expression_bucket="neutral")
        for i in range(just_below)
    ]
    enough = [
        FaceQARecord(frame=f"f{i}.png", face_index=0, expression_bucket="neutral")
        for i in range(MIN_SAMPLES_PER_BIN)
    ]

    sparse_score = compute_readiness_scores(compute_coverage(sparse)).components["expression"]
    enough_score = compute_readiness_scores(compute_coverage(enough)).components["expression"]

    assert sparse_score.min_sample_coverage == 0.0
    assert enough_score.min_sample_coverage > 0.0


def test_readiness_report_exposes_scores_in_json_and_markdown() -> None:
    """Readiness scores should round-trip through JSON and appear in markdown."""
    records = _balanced_records(per_bin=4)

    report = generate_readiness_report(compute_coverage(records))
    payload = json.loads(report.to_json())

    assert "readiness_scores" in payload
    scoring = payload["readiness_scores"]
    assert "overall_readiness_score" in scoring
    assert "confidence" in scoring
    assert set(scoring["components"]).issuperset({"pose", "expression", "lighting", "quality"})
    markdown = report.to_markdown()
    assert "## Readiness Score" in markdown
    assert "Overall readiness" in markdown
    assert "| Component |" in markdown


def test_component_weights_sum_to_one() -> None:
    """Component weight constants should sum to 1.0 so the overall score is calibrated."""
    assert abs(sum(COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9


def test_strength_threshold_higher_than_risk_threshold() -> None:
    """Sanity check: strength threshold must exceed risk threshold."""
    assert STRENGTH_SCORE_THRESHOLD > RISK_SCORE_THRESHOLD
