#!/usr/bin/env python3
"""Regression tests for FaceQA coverage-aware redundancy pruning.

These tests cover the scenarios listed under "Regression tests to add" in
issue #176, plus exercise the public ``RedundancyReport`` interface.
"""

from __future__ import annotations

import json

from lib.align.faceset_qa import FaceQARecord
from lib.faceqa.coverage import compute_coverage
from lib.faceqa.redundancy import (
    AGGRESSIVENESS_LEVELS,
    KEEP,
    PRUNE,
    REVIEW,
    compute_redundancy,
)


def _record(
    frame: str,
    face_index: int = 0,
    *,
    yaw: float | None = 0.0,
    pitch: float | None = 0.0,
    roll: float | None = 0.0,
    mouth_openness: float | None = 0.05,
    smile_proxy: float | None = 0.05,
    eye_closure: float | None = 0.30,
    expression_bucket: str | None = "neutral",
    mean_luminance: float | None = 128.0,
    contrast: float | None = 30.0,
    left_right_ratio: float | None = 1.0,
    top_bottom_ratio: float | None = 1.0,
    color_warmth: float | None = 0.0,
    saturation: float | None = 60.0,
    average_distance: float | None = 0.06,
    blur_score: float | None = 6.0,
    resolution: list[int] | None = None,
    identity_model: str | None = "insightface",
    identity_quality_flag: str | None = "inlier",
) -> FaceQARecord:
    return FaceQARecord(
        frame=frame,
        face_index=face_index,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        mouth_openness=mouth_openness,
        smile_proxy=smile_proxy,
        eye_closure=eye_closure,
        expression_bucket=expression_bucket,
        mean_luminance=mean_luminance,
        contrast=contrast,
        left_right_ratio=left_right_ratio,
        top_bottom_ratio=top_bottom_ratio,
        color_warmth=color_warmth,
        saturation=saturation,
        average_distance=average_distance,
        blur_score=blur_score,
        resolution=resolution if resolution is not None else [256, 256],
        identity_model=identity_model,
        identity_quality_flag=identity_quality_flag,
    )


def _near_identical_run(
    count: int,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    expression_bucket: str = "neutral",
    frame_start: int = 1,
    blur_score: float = 6.0,
) -> list[FaceQARecord]:
    return [
        _record(
            frame=f"frame_{(frame_start + idx):06d}.png",
            yaw=yaw,
            pitch=pitch,
            expression_bucket=expression_bucket,
            blur_score=blur_score,
        )
        for idx in range(count)
    ]


# ---------------------------------------------------------------------------
# Issue acceptance: regression scenarios
# ---------------------------------------------------------------------------


def test_same_identity_different_pose_is_not_redundant() -> None:
    """Identity similarity must not imply redundancy when pose differs."""
    records = [
        _record("frame_000001.png", yaw=0.0, pitch=0.0),
        _record("frame_000002.png", yaw=-60.0, pitch=10.0),
        _record("frame_000003.png", yaw=60.0, pitch=-10.0),
    ]

    report = compute_redundancy(records)

    assert report.cluster_count == 3
    assert all(r.recommendation == KEEP for r in report.records)


def test_hundred_near_identical_in_rare_bucket_not_all_kept() -> None:
    """100 near-identical faces in a single bucket should mostly prune.

    The protected-bucket guardrail keeps a small handful around to preserve
    fragile effective coverage; everything else should land in prune_candidate.
    """
    records = _near_identical_run(100, yaw=70.0, pitch=20.0, expression_bucket="smile")

    report = compute_redundancy(records, aggressiveness="balanced")

    # Effective coverage should compress the 100 raw faces into ~1 distinct
    # redundancy cluster for the rare extreme-profile slot.
    eff = report.effective_coverage["pose"]
    raw_total = sum(eff.raw_counts.values())
    effective_total = sum(eff.effective_counts.values())
    assert raw_total == 100
    assert effective_total <= 5
    assert report.keep_count <= 5
    assert report.prune_candidate_count >= 80


def test_effective_coverage_lower_than_raw_for_repeated_faces() -> None:
    """Repeated near-identical faces should not satisfy a raw coverage threshold."""
    records = _near_identical_run(50, yaw=-45.0, expression_bucket="neutral")

    report = compute_redundancy(records)

    eff = report.effective_coverage["pose"]
    assert sum(eff.raw_counts.values()) == 50
    assert sum(eff.effective_counts.values()) < 10
    assert any(ratio > 5.0 for ratio in eff.redundancy_ratios.values())


def test_temporal_close_higher_confidence_than_temporal_far() -> None:
    """Close-in-time near-identical faces should be more confident duplicates."""
    close = _near_identical_run(8, yaw=0.0, frame_start=1)
    far = [
        _record("frame_000001.png"),
        _record("frame_005000.png"),
        _record("frame_010000.png"),
        _record("frame_015000.png"),
        _record("frame_020000.png"),
        _record("frame_025000.png"),
        _record("frame_030000.png"),
        _record("frame_035000.png"),
    ]

    close_report = compute_redundancy(close, aggressiveness="balanced")
    far_report = compute_redundancy(far, aggressiveness="balanced")

    # Close: heavily pruned. Far: mostly review.
    assert close_report.prune_candidate_count > far_report.prune_candidate_count
    assert far_report.review_count >= close_report.review_count


def test_underrepresented_buckets_are_protected() -> None:
    """A rare bucket should receive limited keeps even when faces are similar."""
    records = _near_identical_run(
        40, yaw=0.0, expression_bucket="neutral", frame_start=1
    ) + _near_identical_run(3, yaw=70.0, expression_bucket="expressive", frame_start=2000)

    report = compute_redundancy(records, aggressiveness="balanced")

    assert any("pose:right_extreme" in label for label in report.protected_buckets)
    rare = [r for r in report.records if r.pose_bucket == "right_extreme"]
    assert any(r.recommendation == KEEP for r in rare)


def test_near_identical_high_quality_mostly_prunes_not_reviews() -> None:
    """100 near-identical high-quality faces should be prune-heavy, not review-heavy."""
    records = _near_identical_run(
        100, yaw=0.0, pitch=0.0, expression_bucket="neutral", blur_score=10.0
    )

    report = compute_redundancy(records, aggressiveness="balanced")

    assert report.prune_candidate_count >= report.review_count * 3


def test_missing_metrics_route_to_review() -> None:
    """Faces with no representation metrics should route to review, not prune."""
    records = [
        _record("frame_000001.png"),  # full metrics
        _record(
            "frame_000002.png",
            yaw=None,
            pitch=None,
            mouth_openness=None,
            smile_proxy=None,
            eye_closure=None,
            mean_luminance=None,
            left_right_ratio=None,
            expression_bucket=None,
        ),
    ]

    report = compute_redundancy(records)

    incomplete = next(r for r in report.records if r.frame == "frame_000002.png")
    # The incomplete face is either kept (no comparable signal so not clustered)
    # or routed to review — never silently pruned.
    assert incomplete.recommendation != PRUNE


def test_identity_outlier_routes_to_review_not_prune() -> None:
    """An identity-outlier face should surface as review, not get pruned silently."""
    records = _near_identical_run(6, yaw=0.0, expression_bucket="neutral") + [
        _record(
            "frame_000007.png",
            yaw=0.0,
            expression_bucket="neutral",
            identity_quality_flag="outlier",
        )
    ]

    report = compute_redundancy(records, aggressiveness="balanced")

    outlier = next(r for r in report.records if r.frame == "frame_000007.png")
    assert outlier.identity_outlier is True
    assert outlier.recommendation != PRUNE


def test_singleton_clusters_are_kept() -> None:
    """A cluster of size 1 must always be kept (no duplicate to drop into)."""
    records = [
        _record("frame_000001.png", yaw=-70.0, pitch=-30.0, expression_bucket="smile"),
        _record("frame_000002.png", yaw=70.0, pitch=30.0, expression_bucket="talking_open"),
    ]

    report = compute_redundancy(records)

    assert all(r.cluster_size == 1 for r in report.records)
    assert all(r.recommendation == KEEP for r in report.records)


# ---------------------------------------------------------------------------
# Public surface checks
# ---------------------------------------------------------------------------


def test_aggressiveness_levels_are_stable() -> None:
    """All advertised aggressiveness levels must be selectable."""
    records = _near_identical_run(20)
    for level in AGGRESSIVENESS_LEVELS:
        report = compute_redundancy(records, aggressiveness=level)
        assert report.aggressiveness == level


def test_report_round_trips_through_json() -> None:
    """The report's JSON form should be parseable and self-consistent."""
    records = _near_identical_run(15) + [_record("frame_001000.png", yaw=60.0)]

    report = compute_redundancy(records)
    payload = json.loads(report.to_json())

    assert payload["total_faces"] == len(records)
    assert payload["keep_count"] + payload["review_count"] + payload[
        "prune_candidate_count"
    ] == len(records)
    assert {"pose", "pitch", "expression", "lighting"}.issubset(payload["effective_coverage"])


def test_redundancy_accepts_optional_coverage_report() -> None:
    """When a coverage report is supplied, raw counts should agree with it."""
    records = _near_identical_run(10) + [_record("frame_001000.png", yaw=70.0)]
    coverage = compute_coverage(records)

    report = compute_redundancy(records, coverage=coverage)

    raw_pose_counts = report.effective_coverage["pose"].raw_counts
    for bucket, count in coverage.bucket_counts.get("pose", {}).items():
        assert raw_pose_counts.get(bucket, 0) >= count or count == 0


def test_unknown_aggressiveness_raises() -> None:
    """Invalid aggressiveness should raise ValueError early."""
    import pytest

    with pytest.raises(ValueError):
        compute_redundancy([_record("frame_000001.png")], aggressiveness="moderate")


def test_empty_records_returns_empty_report() -> None:
    """An empty input must produce an empty (but valid) report."""
    report = compute_redundancy([])
    assert report.total_faces == 0
    assert report.records == []
    assert report.effective_coverage == {}


# ---------------------------------------------------------------------------
# Reviewer feedback regressions (see PR review of #176)
# ---------------------------------------------------------------------------


def test_singleton_identity_outlier_routes_to_review_not_keep() -> None:
    """A singleton identity outlier must not default to keep just because it is a singleton."""
    records = [
        _record(
            "frame_000001.png",
            yaw=-65.0,
            expression_bucket="expressive",
            identity_quality_flag="outlier",
        ),
    ]

    report = compute_redundancy(records)

    outlier = report.records[0]
    assert outlier.identity_outlier is True
    assert outlier.recommendation == REVIEW


def test_representative_identity_outlier_routes_to_review() -> None:
    """An identity outlier promoted to representative still routes to review."""
    records = (
        # Two non-outlier members with lower quality (smaller resolution).
        [
            _record(
                "frame_000001.png",
                blur_score=3.0,
                resolution=[96, 96],
                yaw=0.0,
            ),
            _record(
                "frame_000002.png",
                blur_score=3.0,
                resolution=[96, 96],
                yaw=0.0,
            ),
        ]
        # High-quality outlier; will rank highest by quality_score and become rep.
        + [
            _record(
                "frame_000003.png",
                blur_score=12.0,
                resolution=[512, 512],
                yaw=0.0,
                identity_quality_flag="outlier",
            ),
        ]
    )

    report = compute_redundancy(records)

    rep = next(r for r in report.records if r.representative)
    assert rep.frame == "frame_000003.png"
    assert rep.identity_outlier is True
    assert rep.recommendation == REVIEW


def test_obvious_duplicate_prunes_even_in_protected_bucket() -> None:
    """Protection budget must not rescue exact duplicates from pruning."""
    # Only fragile-bucket faces present — every cluster's bucket is protected.
    records = _near_identical_run(8, yaw=70.0, expression_bucket="smile")

    report = compute_redundancy(records, aggressiveness="balanced")

    pruned = [r for r in report.records if r.recommendation == PRUNE]
    assert pruned, "expected at least one prune candidate even in protected bucket"
    assert all("obvious duplicate" in r.reason for r in pruned), (
        "obvious-duplicate prunes should fire before protection budget"
    )


def test_lighting_bucket_is_populated_on_records() -> None:
    """Records must expose lighting_bucket rather than leaving the slot empty."""
    records = [
        _record("frame_000001.png", mean_luminance=30.0),  # dark
        _record("frame_000002.png", mean_luminance=240.0),  # overexposed
        _record("frame_000003.png", mean_luminance=128.0),  # flat_frontal
    ]

    report = compute_redundancy(records)
    by_frame = {r.frame: r.lighting_bucket for r in report.records}

    assert by_frame["frame_000001.png"] == "dark"
    assert by_frame["frame_000002.png"] == "overexposed"
    assert by_frame["frame_000003.png"] == "flat_frontal"


def test_missing_identity_routes_non_obvious_redundancy_to_review() -> None:
    """Faces without an identity embedding must route non-obvious redundancy to review."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        _decide_single_member,
        _features_for,
    )

    config = _AGGRESSIVENESS_PRESETS["balanced"]
    rep_record = _record("frame_000001.png")
    member_record = _record(
        "frame_000010.png",
        yaw=10.0,
        identity_model=None,
        identity_quality_flag=None,
    )
    member_features = _features_for(member_record)

    # Distance just above the obvious threshold so the hard-floor prune rule
    # does not fire and we exercise the non-obvious redundancy path.
    decision, reason, _budget = _decide_single_member(
        member_features=member_features,
        distance=config.obvious_duplicate_threshold + 0.05,
        compared=10,
        temporal_confidence=1.0,
        config=config,
        member_record=member_record,
        rep_record=rep_record,
        protection_budget_remaining=0,
        member_in_surplus_bucket=False,
    )

    assert decision == REVIEW
    assert "identity" in reason.lower()


def test_protected_bucket_keeps_budget_even_when_other_dim_is_surplus() -> None:
    """A protected fragile bucket must keep its budget even if another dim is surplus."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        _protection_budget_for_cluster,
    )

    rep = _record("frame_000001.png", yaw=70.0)
    cluster = [0, 1, 2]
    protected = {("pose", "right_extreme")}
    surplus = {("lighting", "flat_frontal")}
    config = _AGGRESSIVENESS_PRESETS["balanced"]

    budget = _protection_budget_for_cluster(
        members=cluster,
        representative_index=0,
        record_list=[rep, rep, rep],
        protected=protected,
        surplus=surplus,
        config=config,
    )

    assert budget == config.min_effective_bucket_count - 1


def test_write_manifests_does_not_mix_review_into_keep(tmp_path) -> None:
    """``keep.csv`` must contain only KEEP records; REVIEW gets its own manifest."""
    import json

    from lib.faceqa.redundancy_outputs import write_manifests

    records = _near_identical_run(6, yaw=0.0, expression_bucket="neutral") + [
        _record(
            "frame_999999.png",
            yaw=0.0,
            expression_bucket="neutral",
            identity_quality_flag="outlier",
        )
    ]
    report = compute_redundancy(records, aggressiveness="balanced")

    artefacts = write_manifests(report, tmp_path / "manifests")

    keep_payload = [json.loads(line) for line in artefacts["keep_jsonl"].read_text().splitlines()]
    review_payload = [
        json.loads(line) for line in artefacts["review_jsonl"].read_text().splitlines()
    ]
    assert all(rec["recommendation"] == KEEP for rec in keep_payload)
    assert review_payload, "expected at least one REVIEW record (identity outlier)"
    assert all(rec["recommendation"] == REVIEW for rec in review_payload)
    assert all(rec["frame"] != "frame_999999.png" for rec in keep_payload)


def test_classify_buckets_is_coverage_aware() -> None:
    """The classifier should use min_bucket_pct and surplus_margin, not just a fixed floor."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        EffectiveCoverageDimension,
        _classify_buckets,
    )

    effective = {
        "pose": EffectiveCoverageDimension(
            dimension="pose",
            raw_counts={"frontal": 100, "right_extreme": 4},
            effective_counts={"frontal": 50, "right_extreme": 4},
            redundancy_ratios={"frontal": 2.0, "right_extreme": 1.0},
        )
    }
    config = _AGGRESSIVENESS_PRESETS["balanced"]
    # 1000 total faces × 5% = 50 → surplus threshold = 50 × 1.5 = 75.
    # frontal at 50 sits at floor → protected (count <= floor).
    # right_extreme at 4 is also below floor → protected.
    protected, surplus = _classify_buckets(effective, config, total_faces=1000, min_bucket_pct=5.0)
    assert ("pose", "frontal") in protected
    assert ("pose", "right_extreme") in protected
    assert surplus == set()

    # Lower the floor so frontal becomes surplus while right_extreme stays fragile.
    # 1000 × 0.5% = 5 → surplus threshold = 5 × 1.5 = 7.5.
    # frontal=50 >> 7.5 → surplus. right_extreme=4 < 5 → protected.
    protected, surplus = _classify_buckets(effective, config, total_faces=1000, min_bucket_pct=0.5)
    assert ("pose", "right_extreme") in protected
    assert ("pose", "frontal") in surplus
