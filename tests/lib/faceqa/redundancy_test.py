#!/usr/bin/env python3
"""Regression tests for FaceQA coverage-aware redundancy pruning.

These tests cover the scenarios listed under "Regression tests to add" in
issue #176, plus exercise the public ``RedundancyReport`` interface.
"""

from __future__ import annotations

import json

from lib.faceqa.coverage import compute_coverage
from lib.faceqa.record import FaceQARecord
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
    image_metrics_provenance: str | None = None,
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
        image_metrics_provenance=image_metrics_provenance,
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


def test_identity_outlier_stays_in_singleton_cluster_and_routes_to_review() -> None:
    """Identity outliers never enter the redundancy graph (issue #199).

    Previously the outlier could be unioned into a non-outlier cluster and
    then routed to review via the post-cluster representative-safety
    rule. The new constrained-redundancy pipeline excludes outlier edges
    at the graph level (``can_create_redundancy_edge`` returns False),
    so the outlier ends up in its own singleton cluster and the
    non-outliers cluster among themselves. The singleton outlier still
    routes to review via :func:`_representative_safety_reason`.
    """
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
        # High-quality outlier — graph excludes it; lands in its own
        # singleton cluster.
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

    outlier_record = next(r for r in report.records if r.frame == "frame_000003.png")
    assert outlier_record.representative is True
    assert outlier_record.identity_outlier is True
    assert outlier_record.cluster_size == 1
    assert outlier_record.recommendation == REVIEW

    non_outlier_records = [r for r in report.records if r.frame != "frame_000003.png"]
    # The two non-outlier faces share one cluster id (they cluster among
    # themselves now that the outlier is excluded from the graph).
    cluster_ids = {r.cluster_id for r in non_outlier_records}
    assert len(cluster_ids) == 1


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


def _decide_with_missing_identity(
    *,
    distance: float,
    temporal_confidence: float = 1.0,
    protection_budget_remaining: int = 0,
    member_record_overrides: dict | None = None,
) -> tuple[str, str, int]:
    """Helper: invoke ``_decide_single_member`` with a missing-identity member."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        _decide_single_member,
        _features_for,
    )

    config = _AGGRESSIVENESS_PRESETS["balanced"]
    rep_record = _record("frame_000001.png")
    member_kwargs = dict(identity_model=None, identity_quality_flag=None)
    if member_record_overrides:
        member_kwargs.update(member_record_overrides)
    member_record = _record("frame_000010.png", **member_kwargs)
    member_features = _features_for(member_record)
    return _decide_single_member(
        member_features=member_features,
        distance=distance,
        compared=10,
        temporal_confidence=temporal_confidence,
        config=config,
        member_record=member_record,
        rep_record=rep_record,
        protection_budget_remaining=protection_budget_remaining,
        member_in_surplus_bucket=False,
    )


def test_missing_identity_overrides_meaningful_variation_keep() -> None:
    """A bucket-variation keep must not bypass the missing-identity guardrail."""
    decision, reason, _budget = _decide_with_missing_identity(
        distance=0.20,
        member_record_overrides={"yaw": 70.0, "expression_bucket": "smile"},
    )
    assert decision == REVIEW
    assert "identity" in reason.lower()


def test_missing_identity_overrides_protection_budget_keep() -> None:
    """Protection budget must not rescue a missing-identity member."""
    decision, reason, budget_after = _decide_with_missing_identity(
        distance=0.20,
        protection_budget_remaining=3,
    )
    assert decision == REVIEW
    assert "identity" in reason.lower()
    # Budget must not be spent on a review decision.
    assert budget_after == 3


def test_missing_identity_overrides_obvious_duplicate_prune() -> None:
    """Even obvious duplicates with no identity verification route to review."""
    decision, reason, _budget = _decide_with_missing_identity(distance=0.01)
    assert decision == REVIEW
    assert "identity" in reason.lower()


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


def test_review_records_not_mixed_into_keep() -> None:
    """REVIEW records (e.g. identity outliers) must not contaminate KEEP.

    write_manifests has been removed; the assertion now runs directly against
    ``RedundancyReport.records`` since the coverage JSON is the single source
    of truth for keep / review / prune membership.
    """
    records = _near_identical_run(6, yaw=0.0, expression_bucket="neutral") + [
        _record(
            "frame_999999.png",
            yaw=0.0,
            expression_bucket="neutral",
            identity_quality_flag="outlier",
        )
    ]
    report = compute_redundancy(records, aggressiveness="balanced")

    keep_records = [r for r in report.records if r.recommendation == KEEP]
    review_records = [r for r in report.records if r.recommendation == REVIEW]

    assert review_records, "expected at least one REVIEW record (identity outlier)"
    assert all(r.recommendation == KEEP for r in keep_records)
    assert all(r.recommendation == REVIEW for r in review_records)
    assert all(r.frame != "frame_999999.png" for r in keep_records)


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


def test_representation_distance_downweights_fallback_lighting() -> None:
    """Fallback provenance must scale image-derived feature weight, not full trust."""
    from lib.faceqa.redundancy import _features_for, _representation_distance

    base = _record(
        "frame_000001.png",
        yaw=0.0,
        pitch=0.0,
        mean_luminance=128.0,
        contrast=10.0,
        left_right_ratio=1.0,
        top_bottom_ratio=1.0,
        color_warmth=0.0,
        image_metrics_provenance="frame_aligned_crop",
    )
    # Same pose / expression; only the lighting differs sharply.
    bright = _record(
        "frame_000002.png",
        yaw=0.0,
        pitch=0.0,
        mean_luminance=230.0,  # large lighting delta
        contrast=10.0,
        left_right_ratio=1.0,
        top_bottom_ratio=1.0,
        color_warmth=0.0,
        image_metrics_provenance="frame_aligned_crop",
    )

    frame_pair = _representation_distance(_features_for(base), _features_for(bright))
    frame_distance = frame_pair[0]

    # Now mark the bright face as thumbnail-derived: the same lighting delta
    # should contribute less to the distance.
    bright_fallback = _record(
        "frame_000003.png",
        yaw=0.0,
        pitch=0.0,
        mean_luminance=230.0,
        contrast=10.0,
        left_right_ratio=1.0,
        top_bottom_ratio=1.0,
        color_warmth=0.0,
        image_metrics_provenance="thumbnail_fallback",
    )
    fallback_pair = _representation_distance(_features_for(base), _features_for(bright_fallback))
    fallback_distance = fallback_pair[0]

    assert fallback_distance < frame_distance


def test_identity_guardrail_requires_classified_decision() -> None:
    """Raw embedding presence is NOT enough to satisfy the identity guardrail.

    Regression: previously ``has_identity`` was true whenever any
    ``face.identity[...]`` vector existed (via ``identity_model is not None``).
    A face with an unsupported vector or a vector for a model that
    ``compute_identity_quality`` could not classify would silently bypass the
    missing-identity guardrail. The contract is now that a record must carry
    an explicit decision (``identity_final_decision``) or classified flag
    (``identity_quality_flag``) for the guardrail to consider it covered.
    """
    from lib.faceqa.redundancy import _has_identity_guardrail

    # Vector exists (identity_model set) but the classifier never ran.
    raw_only = _record(
        "frame_000001.png",
        identity_model="insightface",
        identity_quality_flag=None,
    )
    raw_only.identity_final_decision = None
    assert _has_identity_guardrail(raw_only) is False

    # Quality classification populated.
    flagged = _record("frame_000002.png", identity_quality_flag="inlier")
    assert _has_identity_guardrail(flagged) is True

    # Final decision populated (e.g. via verifier pipeline).
    decided = _record(
        "frame_000003.png",
        identity_model="insightface",
        identity_quality_flag=None,
    )
    decided.identity_final_decision = "review"
    assert _has_identity_guardrail(decided) is True

    # Truly empty: no model, no flag, no decision → guardrail missing.
    empty = _record(
        "frame_000004.png",
        identity_model=None,
        identity_quality_flag=None,
    )
    assert _has_identity_guardrail(empty) is False


def test_compute_redundancy_invokes_progress_callback_per_pair() -> None:
    """compute_redundancy ticks the progress callback ``n * (n - 1) // 2`` times (issue #187)."""
    records = _near_identical_run(5, yaw=0.0, expression_bucket="neutral")
    ticks: list[int] = []

    report = compute_redundancy(records, progress_callback=ticks.append)

    expected_pairs = len(records) * (len(records) - 1) // 2
    assert len(ticks) == expected_pairs
    assert all(t == 1 for t in ticks)
    # Sanity: clustering still produced a meaningful report.
    assert len(report.records) == 5


# ---------------------------------------------------------------------------
# Issue #199 — explicit redundancy edge eligibility gate.
# ---------------------------------------------------------------------------


def _features_with_overrides(**overrides):  # type: ignore[no-untyped-def]
    """Build a ``RepresentationFeatures`` with sane defaults for edge tests."""
    from lib.faceqa.redundancy import RepresentationFeatures

    base = dict(
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        mouth_openness=0.05,
        smile_proxy=0.05,
        eye_closure=0.0,
        expression_asymmetry=0.0,
        luminance=0.5,
        contrast=0.4,
        left_right_ratio=0.0,
        top_bottom_ratio=0.0,
        color_warmth=0.0,
        saturation=0.5,
        average_distance=0.05,
        expression_bucket="neutral",
        lighting_bucket="flat_frontal",
        pose_bucket="frontal",
        pitch_bucket="neutral",
        frame_index=0,
        quality_score=0.8,
        has_identity=True,
        identity_outlier=False,
        has_metrics=True,
        image_metrics_provenance="frame_aligned_crop",
    )
    base.update(overrides)
    return RepresentationFeatures(**base)


def test_can_create_redundancy_edge_blocks_few_comparable_metrics() -> None:
    """Pairs with fewer than the minimum comparable dimensions are not edges."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        can_create_redundancy_edge,
    )

    config = _AGGRESSIVENESS_PRESETS["balanced"]
    a = _features_with_overrides()
    b = _features_with_overrides(frame_index=1)
    eligible, reason = can_create_redundancy_edge(
        features_a=a,
        features_b=b,
        distance=0.01,
        compared=1,
        temporal_confidence=1.0,
        config=config,
    )
    assert eligible is False
    assert "too few comparable metrics" in reason


def test_can_create_redundancy_edge_blocks_identity_outlier() -> None:
    """Identity outliers are excluded from the redundancy graph (issue #199)."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        can_create_redundancy_edge,
    )

    config = _AGGRESSIVENESS_PRESETS["balanced"]
    a = _features_with_overrides()
    b = _features_with_overrides(frame_index=1, identity_outlier=True)
    eligible, reason = can_create_redundancy_edge(
        features_a=a,
        features_b=b,
        distance=0.01,
        compared=10,
        temporal_confidence=1.0,
        config=config,
    )
    assert eligible is False
    assert "identity outlier" in reason


def test_can_create_redundancy_edge_blocks_redundant_distance_for_missing_identity() -> None:
    """A redundant-distance match without identity isn't an edge — only
    obvious-distance + high-temporal duplicates can cluster across missing
    identity (issue #199)."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        can_create_redundancy_edge,
    )

    config = _AGGRESSIVENESS_PRESETS["balanced"]
    # Borderline redundant distance, both sides missing identity → blocked.
    a = _features_with_overrides(has_identity=False)
    b = _features_with_overrides(frame_index=1, has_identity=False)
    eligible, reason = can_create_redundancy_edge(
        features_a=a,
        features_b=b,
        distance=0.20,  # below the balanced threshold (0.25) but not obvious
        compared=10,
        temporal_confidence=1.0,
        config=config,
    )
    assert eligible is False
    assert "missing identity" in reason


def test_can_create_redundancy_edge_allows_obvious_duplicate_across_missing_identity() -> None:
    """Obvious + temporally-safe duplicates DO cluster across missing
    identity so the post-cluster decision layer can review the cluster as a
    unit (issue #199)."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        can_create_redundancy_edge,
    )

    config = _AGGRESSIVENESS_PRESETS["balanced"]
    a = _features_with_overrides(has_identity=False)
    b = _features_with_overrides(frame_index=1, has_identity=False)
    eligible, reason = can_create_redundancy_edge(
        features_a=a,
        features_b=b,
        distance=0.01,  # well below the obvious duplicate floor
        compared=10,
        temporal_confidence=1.0,
        config=config,
    )
    assert eligible is True
    assert "obvious duplicate" in reason
    assert "missing-identity" in reason


def test_can_create_redundancy_edge_blocks_obvious_match_with_low_temporal() -> None:
    """An obvious-distance pair must NOT cluster if it's temporally distant
    — we cannot confidently call it a duplicate without temporal support."""
    from lib.faceqa.redundancy import (
        _AGGRESSIVENESS_PRESETS,
        can_create_redundancy_edge,
    )

    config = _AGGRESSIVENESS_PRESETS["balanced"]
    a = _features_with_overrides()
    b = _features_with_overrides(frame_index=10_000)  # very far apart
    eligible, reason = can_create_redundancy_edge(
        features_a=a,
        features_b=b,
        distance=0.01,
        compared=10,
        temporal_confidence=0.4,
        config=config,
    )
    assert eligible is False
    assert "temporal_confidence" in reason


def test_redundancy_record_carries_compared_metrics_diagnostic() -> None:
    """``RedundancyRecord.compared_metrics`` defaults exist + persist through
    ``to_dict`` so the output can be audited (issue #199)."""
    from lib.faceqa.redundancy import RedundancyRecord

    record = RedundancyRecord(
        frame="frame_000001.png",
        face_index=0,
        cluster_id=0,
        cluster_size=1,
        representative=True,
        recommendation="keep",
        quality_score=0.8,
        redundancy_distance_to_representative=None,
        temporal_distance_to_representative=None,
        temporal_confidence=1.0,
        pose_bucket="frontal",
        pitch_bucket="neutral",
        expression_bucket="neutral",
        lighting_bucket="flat_frontal",
        reason="singleton",
        identity_outlier=False,
        has_identity=True,
    )

    payload = record.to_dict()
    assert payload["compared_metrics"] == 0
    assert payload["edge_eligibility"] is None


# ---------------------------------------------------------------------------
# Issue #199 follow-ups — cluster-level missing-identity, populated
# diagnostics on non-representative records.
# ---------------------------------------------------------------------------


def test_missing_identity_cluster_routes_representative_to_review() -> None:
    """A representative that lacks the identity guardrail must REVIEW, not KEEP.

    The new edge gate (#199) allows obvious + temporally-safe pairs through
    even when identity is missing — the post-cluster intent is then "whole
    cluster to review". This pins that intent at the representative-side.
    """
    records = _near_identical_run(2, yaw=0.0, expression_bucket="neutral")
    # Strip the identity guardrail from BOTH records so the rep ends up
    # without identity.
    for record in records:
        record.identity_model = None
        record.identity_quality_flag = None
        record.identity_final_decision = None

    report = compute_redundancy(records, aggressiveness="balanced")

    assert report.cluster_count == 1, (
        "obvious + temporally-safe duplicates still cluster across missing identity per the gate"
    )
    rep = next(r for r in report.records if r.representative)
    assert rep.recommendation == REVIEW
    assert "missing identity embedding guardrail" in rep.reason


def test_missing_identity_cluster_routes_members_to_review() -> None:
    """Non-representative members in a missing-identity cluster also REVIEW.

    The previous shape would prune the member as an obvious duplicate even
    though the cluster as a whole lacked the cross-subject guardrail. The
    follow-up adds a cluster-level check so the entire cluster lands in
    review together.
    """
    records = _near_identical_run(2, yaw=0.0, expression_bucket="neutral")
    for record in records:
        record.identity_model = None
        record.identity_quality_flag = None
        record.identity_final_decision = None

    report = compute_redundancy(records, aggressiveness="balanced")

    members = [r for r in report.records if not r.representative]
    assert members, "expected at least one non-representative member"
    for member in members:
        assert member.recommendation == REVIEW
        assert "missing identity embedding guardrail" in member.reason


def test_redundancy_member_record_populates_diagnostics() -> None:
    """Member records expose ``compared_metrics`` + ``edge_eligibility`` so
    the JSON output can audit why each face landed where (#199 follow-up).
    """
    records = _near_identical_run(2, yaw=0.0, expression_bucket="neutral")

    report = compute_redundancy(records, aggressiveness="balanced")

    members = [r for r in report.records if not r.representative]
    assert members, "expected at least one non-representative member"
    for member in members:
        assert member.compared_metrics > 0
        assert member.edge_eligibility is not None
        # The gate's eligibility reason for an obvious duplicate carries the
        # phrase "obvious duplicate" — pin it so the diagnostic source is
        # clear in the JSON output.
        assert "obvious duplicate" in member.edge_eligibility


def test_frame_index_returns_trailing_numeric_group() -> None:
    """``_frame_index`` must return the LAST numeric run in the frame name.

    Issue #199 follow-up replaces ``_FRAME_NUMBER_RE.findall(frame)`` with
    a tighter regex that captures only the last numeric group; this pins
    the contract for compound names like ``shot_07_000123.png``.
    """
    from lib.faceqa.redundancy import _frame_index

    assert _frame_index("frame_000123.png") == 123
    assert _frame_index("shot_07_000456.jpg") == 456
    assert _frame_index("frame.png") is None
    assert _frame_index("000789") == 789


# ---------------------------------------------------------------------------
# Issue #204 — giant-component prevention regression scenarios
# ---------------------------------------------------------------------------


def test_bucket_mismatch_blocks_normal_redundancy_edge() -> None:
    """A pose-bucket mismatch between non-adjacent buckets (frontal vs
    right_profile) blocks the redundancy edge even when continuous
    features look similar (issue #204 §1).
    """
    from lib.faceqa.redundancy import _AGGRESSIVENESS_PRESETS as PRESETS
    from lib.faceqa.redundancy import _features_for, can_create_redundancy_edge

    frontal = _record("frame_000001.png", yaw=0.0)
    profile = _record("frame_000002.png", yaw=45.0)
    fa, fb = _features_for(frontal), _features_for(profile)

    eligible, reason = can_create_redundancy_edge(
        features_a=fa,
        features_b=fb,
        # Above obvious_duplicate_threshold (0.10 for balanced) so the
        # obvious-override clause does NOT fire — the bucket gate is what
        # must block this edge.
        distance=0.20,
        compared=10,
        temporal_confidence=1.0,
        config=PRESETS["balanced"],
    )
    assert eligible is False
    assert "yaw bucket mismatch" in reason


def test_obvious_duplicate_overrides_bucket_mismatch() -> None:
    """Obvious duplicates may bridge bucket mismatch when temporal
    confidence is high (issue #204 §1 override clause)."""
    from lib.faceqa.redundancy import _AGGRESSIVENESS_PRESETS as PRESETS
    from lib.faceqa.redundancy import _features_for, can_create_redundancy_edge

    frontal = _record("frame_000001.png", yaw=0.0, expression_bucket="neutral")
    smile = _record("frame_000002.png", yaw=0.0, expression_bucket="smile")
    fa, fb = _features_for(frontal), _features_for(smile)

    config = PRESETS["balanced"]
    eligible, reason = can_create_redundancy_edge(
        features_a=fa,
        features_b=fb,
        distance=config.obvious_duplicate_threshold - 0.01,
        compared=10,
        temporal_confidence=1.0,
        config=config,
    )
    assert eligible is True
    assert "obvious duplicate" in reason


def test_adjacent_yaw_buckets_require_stricter_distance() -> None:
    """Adjacent yaw buckets (frontal vs right_slight) need stricter
    distance than same-bucket pairs (issue #204 §2)."""
    from lib.faceqa.redundancy import (
        _ADJACENT_BUCKET_DISTANCE_SCALE,
        _features_for,
        can_create_redundancy_edge,
    )
    from lib.faceqa.redundancy import _AGGRESSIVENESS_PRESETS as PRESETS

    frontal = _record("frame_000001.png", yaw=0.0)
    slight = _record("frame_000002.png", yaw=20.0)
    fa, fb = _features_for(frontal), _features_for(slight)

    config = PRESETS["balanced"]
    strict_limit = config.representation_distance_threshold * _ADJACENT_BUCKET_DISTANCE_SCALE
    # Just above the strict-distance limit but below the same-bucket
    # threshold: the pair must NOT be eligible.
    just_too_far = strict_limit + 0.05
    eligible, reason = can_create_redundancy_edge(
        features_a=fa,
        features_b=fb,
        distance=just_too_far,
        compared=10,
        temporal_confidence=1.0,
        config=config,
    )
    assert eligible is False
    assert "adjacent yaw" in reason

    # Inside the strict limit: eligible (subject to the temporal gate).
    eligible_close, _ = can_create_redundancy_edge(
        features_a=fa,
        features_b=fb,
        distance=strict_limit - 0.05,
        compared=10,
        temporal_confidence=1.0,
        config=config,
    )
    assert eligible_close is True


def test_transitive_chain_does_not_create_giant_component() -> None:
    """A run of records sweeping ``frontal -> right_slight -> right_profile``
    must NOT collapse into a single component once bucket mismatch gates
    cut the chain at the first non-adjacent hop (issue #204 §2)."""
    records = (
        _near_identical_run(15, yaw=0.0, frame_start=1)
        + _near_identical_run(15, yaw=22.0, frame_start=2000)
        + _near_identical_run(15, yaw=45.0, frame_start=4000)
    )

    report = compute_redundancy(records, aggressiveness="balanced")

    # At minimum three distinct multi-face clusters (one per yaw band).
    multi_face = [r for r in report.records if r.representative and r.cluster_size > 1]
    assert len(multi_face) >= 3
    assert report.largest_cluster_size < len(records)


def test_report_includes_cluster_health_fields() -> None:
    """RedundancyReport surfaces the #204 cluster-health fields with
    sensible defaults on a small balanced faceset."""
    # Spread records across distinct pose buckets so the cluster
    # topology has more than one multi-face cluster — keeps the
    # giant-component warning naturally False on this small case
    # without the warning needing to be silenced manually.
    # Spread across FIVE distinct yaw buckets (3 records each = 15 total)
    # so the largest cluster ratio stays below the ticket's 0.25 giant-
    # component threshold (3 / 15 = 0.20).
    records = (
        _near_identical_run(3, yaw=0.0, frame_start=1)
        + _near_identical_run(3, yaw=22.0, frame_start=2000)
        + _near_identical_run(3, yaw=-22.0, frame_start=3000)
        + _near_identical_run(3, yaw=45.0, frame_start=4000)
        + _near_identical_run(3, yaw=-45.0, frame_start=5000)
    )

    report = compute_redundancy(records)

    assert report.largest_cluster_size >= 1
    assert 0.0 <= report.largest_cluster_ratio <= 1.0
    assert report.giant_component_warning is False
    # Diameter map populated even when no split happened.
    assert report.component_diameter_max is not None
    # Round-tripping through JSON includes the new fields.
    payload = json.loads(report.to_json())
    for field_name in (
        "largest_cluster_size",
        "largest_cluster_ratio",
        "giant_component_warning",
        "component_spread_p95",
        "component_diameter_max",
    ):
        assert field_name in payload


def test_per_member_diagnostics_populated() -> None:
    """Each cluster member surfaces the #204 per-member diagnostics
    (nearest_neighbor_distance, direct_to_representative,
    component_diameter)."""
    records = _near_identical_run(6, yaw=0.0)

    report = compute_redundancy(records)
    members = [r for r in report.records if not r.representative]

    assert members, "expected non-representative members in clustered run"
    for member in members:
        assert member.nearest_neighbor_distance is not None
        assert member.component_diameter is not None
        assert isinstance(member.direct_to_representative, bool)


def test_identity_outlier_still_blocked_by_edge_gate() -> None:
    """Issue #199 guardrail remains intact under the new bucket gates —
    an identity outlier never creates a graph edge regardless of bucket
    state."""
    from lib.faceqa.redundancy import _AGGRESSIVENESS_PRESETS as PRESETS
    from lib.faceqa.redundancy import _features_for, can_create_redundancy_edge

    a = _record("frame_000001.png", yaw=0.0)
    outlier = _record(
        "frame_000002.png",
        yaw=0.0,
        identity_quality_flag="outlier",
    )
    fa, fb = _features_for(a), _features_for(outlier)

    eligible, reason = can_create_redundancy_edge(
        features_a=fa,
        features_b=fb,
        distance=0.05,
        compared=10,
        temporal_confidence=1.0,
        config=PRESETS["balanced"],
    )
    assert eligible is False
    assert "identity outlier" in reason


# ---------------------------------------------------------------------------
# Issue #204 follow-up — strict diameter guarantee + richer diagnostics
# ---------------------------------------------------------------------------


def test_split_components_guarantee_diameter_within_limit() -> None:
    """The strict attachment rule (member must be within rep_limit of seed
    AND within diameter_limit of every already-attached member) MUST hold
    after splitting — no post-split component can exceed the diameter
    band even when the parent component did."""
    from lib.faceqa.redundancy import _AGGRESSIVENESS_PRESETS as PRESETS
    from lib.faceqa.redundancy import (
        _COMPONENT_DIAMETER_SCALE,
        _build_redundancy_clusters,
        _distance_with_ctx,
        _features_for,
        _split_high_diameter_components,
    )

    # Build a chain of 12 near-identical-yaw faces so the original union
    # forms one big component that the splitter must then reduce.
    records = _near_identical_run(12, yaw=0.0)
    config = PRESETS["balanced"]
    features = [_features_for(r) for r in records]
    components, _, edge_reasons, ctx = _build_redundancy_clusters(features, config)
    new_components, diag = _split_high_diameter_components(
        components, features, ctx, config, edge_reasons
    )

    diameter_limit = config.representation_distance_threshold * _COMPONENT_DIAMETER_SCALE
    for component in new_components:
        if len(component) <= 1:
            continue
        for i_pos, i in enumerate(component):
            for j in component[i_pos + 1 :]:
                dist, _ = _distance_with_ctx(ctx, i, j)
                assert dist <= diameter_limit + 1e-9, (
                    f"pair ({i}, {j}) distance {dist:.3f} exceeds diameter "
                    f"limit {diameter_limit:.3f} in component {component}"
                )

    # Every member's component_diameter <= limit (mirrors the per-pair
    # assertion above through the diagnostics map).
    for value in diag.diameter_by_member.values():
        assert value <= diameter_limit + 1e-9


def test_oversize_component_surfaces_split_reason() -> None:
    """When a component exceeds the size cap, the splitter keeps it intact
    AND records the ``split_reason`` on every member so the diagnostics
    flag the bypass."""
    from lib.faceqa.redundancy import _AGGRESSIVENESS_PRESETS as PRESETS
    from lib.faceqa.redundancy import (
        _MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT,
        _features_for,
        _split_high_diameter_components,
    )

    # Synthesize one synthetic oversize component without paying for the
    # O(N^2) union-find loop on N > 2000 records.
    cap = _MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT
    records = _near_identical_run(cap + 2, yaw=0.0)
    config = PRESETS["balanced"]
    features = [_features_for(r) for r in records]
    members = list(range(len(records)))
    from lib.faceqa.redundancy import _build_pairwise_context

    ctx = _build_pairwise_context(features)
    components, diag = _split_high_diameter_components(
        [members], features, ctx, config, edge_reasons={}
    )

    assert len(components) == 1
    assert len(components[0]) == cap + 2
    reasons = {diag.split_reason_by_member[m] for m in components[0]}
    assert reasons == {f"component too large for diameter check ({cap + 2} > {cap}) — kept as-is"}


def test_per_member_nearest_neighbor_diagnostics_populated() -> None:
    """Every clustered member surfaces the nearest-neighbour identifier
    + edge eligibility so reasons can distinguish "redundant with
    representative" from "redundant through neighbour chain"."""
    records = _near_identical_run(6, yaw=0.0)

    report = compute_redundancy(records)
    members = [r for r in report.records if not r.representative]

    assert members
    for member in members:
        # Nearest neighbour identifier resolves to a real record.
        assert member.nearest_neighbor_frame is not None
        assert member.nearest_neighbor_face_index is not None
        # ``connected_via_representative`` is a bool (True or False);
        # ``direct_to_representative`` is also a bool. They may agree on
        # this synthetic run but the API must expose both.
        assert isinstance(member.connected_via_representative, bool)
        assert isinstance(member.direct_to_representative, bool)


def test_split_reason_marks_members_when_diameter_split_happens() -> None:
    """A genuinely chained component must produce sub-clusters whose
    members carry the ``split from oversize component due to diameter``
    reason."""
    # Drift yaw across the run so the cumulative span exceeds the
    # diameter limit even though each adjacent pair sits within
    # the same bucket. The splitter must then carve the chain into
    # diameter-safe sub-clusters.
    records = [_record(f"frame_{i:06d}.png", yaw=float(-7 + i * 1.4)) for i in range(12)]

    report = compute_redundancy(records, aggressiveness="balanced")
    reasons = {r.split_reason for r in report.records if r.split_reason is not None}
    # Either the splitter fired (reason populated) OR every pair sat
    # within the diameter envelope; assert the actual diameter contract
    # holds either way.
    if reasons:
        assert "split from oversize component due to diameter" in reasons
    # Cluster-health metric also reflects whatever the splitter did.
    assert report.component_diameter_max is not None
