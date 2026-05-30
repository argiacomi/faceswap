#!/usr/bin/env python3
"""Tests for FaceQA source-target compatibility scoring."""

from __future__ import annotations

import json

from lib.faceqa.compatibility import (
    COMPATIBILITY_WEIGHTS,
    compute_compatibility,
)
from lib.faceqa.coverage import compute_coverage
from lib.faceqa.record import FaceQARecord


def _records(
    *,
    count: int,
    yaw: float = 0.0,
    pitch: float = 0.0,
    expression: str = "neutral",
    luminance: float = 128.0,
    contrast: float = 10.0,
    left_right_ratio: float = 1.0,
    blur: float = 5.0,
    resolution: tuple[int, int] = (256, 256),
) -> list[FaceQARecord]:
    return [
        FaceQARecord(
            frame=f"{expression}_{yaw}_{pitch}_{i}.png",
            face_index=0,
            yaw=yaw,
            pitch=pitch,
            expression_bucket=expression,
            mean_luminance=luminance,
            contrast=contrast,
            left_right_ratio=left_right_ratio,
            top_bottom_ratio=1.0,
            color_warmth=0.0,
            blur_score=blur,
            resolution=list(resolution),
            average_distance=0.05,
            occlusion_score=0.0,
            duplicate_keep_recommendation="keep",
            identity_quality_flag="inlier",
        )
        for i in range(count)
    ]


def test_compute_compatibility_full_match_is_high_scoring() -> None:
    """When source and target distributions match, every score should be ~100."""
    records = (
        _records(count=20, yaw=0.0, pitch=0.0, expression="neutral", luminance=128.0)
        + _records(count=20, yaw=-45.0, pitch=0.0, expression="smile", luminance=128.0)
        + _records(count=20, yaw=45.0, pitch=20.0, expression="talking_open", luminance=128.0)
    )
    source = compute_coverage(records)
    target = compute_coverage(records)

    report = compute_compatibility(source, target)

    assert report.pose_compatibility_score >= 99.0
    assert report.expression_compatibility_score >= 99.0
    assert report.lighting_compatibility_score >= 99.0
    assert report.quality_compatibility_score >= 99.0
    assert report.source_target_compatibility_score >= 99.0
    assert report.coverage_gaps == []


def test_compute_compatibility_flags_mismatched_dimensions() -> None:
    """Source missing target pose/expression/lighting demand should produce gaps."""
    source = compute_coverage(
        _records(count=60, yaw=0.0, pitch=0.0, expression="neutral", luminance=128.0)
    )
    target = compute_coverage(
        _records(count=20, yaw=-45.0, pitch=0.0, expression="talking_open", luminance=30.0)
        + _records(count=20, yaw=45.0, pitch=20.0, expression="smile", luminance=30.0)
        + _records(
            count=20,
            yaw=0.0,
            pitch=0.0,
            expression="expressive",
            luminance=128.0,
            left_right_ratio=2.0,
        )
    )

    report = compute_compatibility(source, target)

    assert report.pose_compatibility_score < 60.0
    assert report.expression_compatibility_score < 60.0
    assert report.lighting_compatibility_score < 80.0
    gaps_text = " ".join(report.coverage_gaps).lower()
    assert "talking" in gaps_text or "smiling" in gaps_text
    assert "low-light" in gaps_text or "side-lighting" in gaps_text
    assert any("profile" in gap.lower() or "yaw" in gap.lower() for gap in report.coverage_gaps)


def test_compute_compatibility_is_deterministic() -> None:
    """Repeated calls with the same coverage produce identical reports."""
    source = compute_coverage(_records(count=10, expression="smile"))
    target = compute_coverage(_records(count=10, expression="smile"))

    first = compute_compatibility(source, target).to_dict()
    second = compute_compatibility(source, target).to_dict()

    assert first == second


def test_compute_compatibility_handles_empty_sides() -> None:
    """Empty source or empty target should not raise and produce a 0 overall score."""
    empty = compute_coverage([])
    populated = compute_coverage(_records(count=10, expression="smile"))

    empty_to_empty = compute_compatibility(empty, empty)
    empty_to_populated = compute_compatibility(empty, populated)

    assert empty_to_empty.source_target_compatibility_score == 0.0
    assert empty_to_populated.source_target_compatibility_score < 50.0
    assert empty_to_populated.confidence == 0.0


def test_compute_compatibility_overall_is_weighted() -> None:
    """The overall score is the weighted average of the four dimension scores."""
    source = compute_coverage(
        _records(count=10, expression="smile")
        + _records(count=10, expression="talking_open")
        + _records(count=10, expression="neutral")
    )
    target = compute_coverage(
        _records(count=10, expression="smile")
        + _records(count=10, expression="talking_open")
        + _records(count=10, expression="neutral")
    )

    report = compute_compatibility(source, target)
    expected = (
        report.pose_compatibility_score * COMPATIBILITY_WEIGHTS["pose"]
        + report.expression_compatibility_score * COMPATIBILITY_WEIGHTS["expression"]
        + report.lighting_compatibility_score * COMPATIBILITY_WEIGHTS["lighting"]
        + report.quality_compatibility_score * COMPATIBILITY_WEIGHTS["quality"]
    )

    assert abs(report.source_target_compatibility_score - round(expected, 2)) < 1e-6


def test_quality_compatibility_penalizes_worse_source() -> None:
    """A blurrier/lower-res source than the target should reduce quality compatibility."""
    high_quality = _records(count=30, blur=8.0, resolution=(512, 512))
    low_quality = _records(count=30, blur=0.2, resolution=(48, 48))

    source = compute_coverage(low_quality)
    target = compute_coverage(high_quality)

    report = compute_compatibility(source, target)

    assert report.quality_compatibility_score < 80.0


def test_compatibility_round_trips_through_json_and_markdown() -> None:
    """Compatibility report should serialise consistently to JSON and Markdown."""
    source = compute_coverage(
        _records(count=20, yaw=0.0, pitch=0.0, expression="neutral", luminance=128.0)
    )
    target = compute_coverage(
        _records(count=20, yaw=45.0, pitch=20.0, expression="smile", luminance=30.0)
    )

    report = compute_compatibility(source, target, source_path="src.fsa", target_path="tgt.fsa")
    payload = json.loads(report.to_json())

    assert set(payload).issuperset(
        {
            "source_target_compatibility_score",
            "pose_compatibility_score",
            "expression_compatibility_score",
            "lighting_compatibility_score",
            "quality_compatibility_score",
            "confidence",
            "dimensions",
            "coverage_gaps",
        }
    )
    markdown = report.to_markdown()
    assert "# FaceQA Source-Target Compatibility" in markdown
    assert "## Compatibility Summary" in markdown
    assert "Overall score" in markdown


def test_compatibility_weights_sum_to_one() -> None:
    assert abs(sum(COMPATIBILITY_WEIGHTS.values()) - 1.0) < 1e-9


def test_broad_source_is_not_penalized_for_extra_coverage() -> None:
    """A source that covers more buckets than the target needs should still score high.

    This is the regression that motivated switching from proportion-matching to
    absolute source-sample adequacy: a source with plenty of frontal samples
    plus equal profile samples used to score ~50 against an all-frontal target.
    """
    frontal_source = compute_coverage(_records(count=40, yaw=0.0, pitch=0.0))
    broad_source = compute_coverage(
        _records(count=40, yaw=0.0, pitch=0.0)
        + _records(count=40, yaw=-45.0, pitch=0.0)
        + _records(count=40, yaw=45.0, pitch=20.0)
    )
    target = compute_coverage(_records(count=40, yaw=0.0, pitch=0.0))

    narrow = compute_compatibility(frontal_source, target)
    broad = compute_compatibility(broad_source, target)

    # Both sources have plenty of frontal samples → both should score near 100
    # on pose. The proportion-matching design previously regressed the broad
    # source to ~50; the absolute-sample design keeps it at full credit.
    assert narrow.pose_compatibility_score >= 99.0
    assert broad.pose_compatibility_score >= 99.0


def test_source_needs_enough_samples_per_target_demanded_bucket() -> None:
    """Source must reach the per-bucket sample minimum for full credit."""
    # Source has only 2 frontal samples (below the 5-sample threshold).
    sparse_source = compute_coverage(_records(count=2, yaw=0.0, pitch=0.0))
    # Target has 40 frontal frames.
    target = compute_coverage(_records(count=40, yaw=0.0, pitch=0.0))

    report = compute_compatibility(sparse_source, target)

    # Coverage factor should be 2/5 = 0.4 → pose score ~40, well below full.
    assert report.pose_compatibility_score < 60.0
    pose_dim = report.dimensions["pose"]
    assert any(
        gap.source_count == 2 and gap.required_source_count >= 5 for gap in pose_dim.deficiencies
    )


def test_gap_phrasing_reports_sample_counts() -> None:
    """Gap phrases should communicate raw source/required counts."""
    source = compute_coverage(_records(count=60, yaw=0.0, pitch=0.0, expression="neutral"))
    target = compute_coverage(_records(count=40, yaw=-45.0, pitch=0.0, expression="smile"))

    report = compute_compatibility(source, target)

    assert any("samples required" in gap for gap in report.coverage_gaps)
    assert any(
        "not present in source" in gap or "barely present" in gap for gap in report.coverage_gaps
    )
