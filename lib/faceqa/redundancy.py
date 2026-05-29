#!/usr/bin/env python3
"""Coverage-aware representation redundancy for FaceQA.

Reframes the duplicate-pruning decision from "is this the same identity?" to
"does this face add meaningful training representation, or is it redundant
with a better-quality face already kept?". Identity becomes a guardrail; the
primary signal is a weighted representation distance over pose, expression,
lighting, framing/alignment, and quality, modulated by temporal confidence.

The output is a :class:`RedundancyReport` containing per-face recommendations
(keep / review / prune_candidate), effective coverage counts per bucket
(distinct redundancy clusters, not raw faces), and the redundancy clusters
themselves.
"""

from __future__ import annotations

import json
import logging
import math
import re
import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.faceqa.coverage import FacesetCoverageReport
from lib.faceqa.record import FaceQARecord
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


# Recommendation tokens (advisory — match lib/faceqa/duplicates.py).
KEEP = "keep"
REVIEW = "review"
PRUNE = "prune_candidate"

# Aggressiveness levels exposed to users via --prune-aggressiveness.
AGGRESSIVENESS_LEVELS: tuple[str, ...] = ("conservative", "balanced", "aggressive")


@dataclass(frozen=True)
class AggressivenessConfig:
    """Tunable thresholds derived from a user-selected aggressiveness."""

    representation_distance_threshold: float
    obvious_duplicate_threshold: float
    temporal_close_window: int
    temporal_far_window: int
    review_quality_margin: float
    coverage_surplus_margin: float
    min_effective_bucket_count: int


_AGGRESSIVENESS_PRESETS: dict[str, AggressivenessConfig] = {
    "conservative": AggressivenessConfig(
        representation_distance_threshold=0.15,
        obvious_duplicate_threshold=0.08,
        temporal_close_window=10,
        temporal_far_window=240,
        review_quality_margin=0.05,
        coverage_surplus_margin=2.0,
        min_effective_bucket_count=4,
    ),
    "balanced": AggressivenessConfig(
        representation_distance_threshold=0.25,
        obvious_duplicate_threshold=0.10,
        temporal_close_window=30,
        temporal_far_window=600,
        review_quality_margin=0.10,
        coverage_surplus_margin=1.5,
        min_effective_bucket_count=3,
    ),
    "aggressive": AggressivenessConfig(
        representation_distance_threshold=0.35,
        obvious_duplicate_threshold=0.14,
        temporal_close_window=90,
        temporal_far_window=1800,
        review_quality_margin=0.15,
        coverage_surplus_margin=1.2,
        min_effective_bucket_count=2,
    ),
}


# Feature weights for representation distance. Pose dominates; identity is a
# guardrail only (it never enters the distance directly).
FEATURE_WEIGHTS: dict[str, float] = {
    "yaw": 1.20,
    "pitch": 1.00,
    "roll": 0.60,
    "mouth_openness": 0.80,
    "smile_proxy": 0.50,
    "eye_closure": 0.50,
    "expression_asymmetry": 0.30,
    "luminance": 0.70,
    "contrast": 0.40,
    "left_right_ratio": 0.50,
    "top_bottom_ratio": 0.40,
    "color_warmth": 0.20,
    "saturation": 0.20,
    "average_distance": 0.40,
}

# Dimensions surfaced in the effective-coverage report. These mirror the
# coverage report's bucketed dimensions that drive readiness gates.
EFFECTIVE_COVERAGE_DIMENSIONS: tuple[str, ...] = (
    "pose",
    "pitch",
    "expression",
    "lighting",
)

_FRAME_NUMBER_RE = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RepresentationFeatures:
    """Normalised representation features for one face.

    Continuous features are mean-centred such that two faces with very similar
    training-relevant representation produce a small weighted distance. ``None``
    indicates the feature was not available; it lowers confidence and steers
    borderline decisions toward ``review``.
    """

    yaw: float | None
    pitch: float | None
    roll: float | None
    mouth_openness: float | None
    smile_proxy: float | None
    eye_closure: float | None
    expression_asymmetry: float | None
    luminance: float | None
    contrast: float | None
    left_right_ratio: float | None
    top_bottom_ratio: float | None
    color_warmth: float | None
    saturation: float | None
    average_distance: float | None
    expression_bucket: str | None
    lighting_bucket: str | None
    pose_bucket: str | None
    pitch_bucket: str | None
    frame_index: int | None
    quality_score: float
    has_identity: bool
    identity_outlier: bool
    has_metrics: bool
    image_metrics_provenance: str | None


@dataclass
class RedundancyRecord:
    """Per-face redundancy outcome."""

    frame: str
    face_index: int
    cluster_id: int
    cluster_size: int
    representative: bool
    recommendation: str
    quality_score: float
    redundancy_distance_to_representative: float | None
    temporal_distance_to_representative: int | None
    temporal_confidence: float
    pose_bucket: str | None
    pitch_bucket: str | None
    expression_bucket: str | None
    lighting_bucket: str | None
    reason: str
    identity_outlier: bool
    has_identity: bool

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "frame": self.frame,
            "face_index": self.face_index,
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "representative": self.representative,
            "recommendation": self.recommendation,
            "quality_score": self.quality_score,
            "redundancy_distance_to_representative": (self.redundancy_distance_to_representative),
            "temporal_distance_to_representative": self.temporal_distance_to_representative,
            "temporal_confidence": self.temporal_confidence,
            "pose_bucket": self.pose_bucket,
            "pitch_bucket": self.pitch_bucket,
            "expression_bucket": self.expression_bucket,
            "lighting_bucket": self.lighting_bucket,
            "reason": self.reason,
            "identity_outlier": self.identity_outlier,
            "has_identity": self.has_identity,
        }


@dataclass
class EffectiveCoverageDimension:
    """Raw vs effective coverage for one bucketed dimension."""

    dimension: str
    raw_counts: dict[str, int]
    effective_counts: dict[str, int]
    redundancy_ratios: dict[str, float]

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "dimension": self.dimension,
            "raw_counts": self.raw_counts,
            "effective_counts": self.effective_counts,
            "redundancy_ratios": self.redundancy_ratios,
        }


@dataclass
class RedundancyReport:
    """Aggregate redundancy report including effective coverage and recommendations."""

    aggressiveness: str = "balanced"
    total_faces: int = 0
    cluster_count: int = 0
    multi_face_clusters: int = 0
    keep_count: int = 0
    review_count: int = 0
    prune_candidate_count: int = 0
    effective_coverage: dict[str, EffectiveCoverageDimension] = field(default_factory=dict)
    protected_buckets: list[str] = field(default_factory=list)
    records: list[RedundancyRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "aggressiveness": self.aggressiveness,
            "total_faces": self.total_faces,
            "cluster_count": self.cluster_count,
            "multi_face_clusters": self.multi_face_clusters,
            "keep_count": self.keep_count,
            "review_count": self.review_count,
            "prune_candidate_count": self.prune_candidate_count,
            "effective_coverage": {
                name: dim.to_dict() for name, dim in self.effective_coverage.items()
            },
            "protected_buckets": self.protected_buckets,
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_index(frame: str) -> int | None:
    """Return the trailing numeric index of a frame name when present."""
    matches = _FRAME_NUMBER_RE.findall(frame)
    if not matches:
        return None
    return int(matches[-1])


def _record_image_metric_scale(record: FaceQARecord) -> float:
    """Return representative-quality weight scale for image-derived metrics."""
    return (
        1.0
        if record.image_metrics_provenance == _FRAME_PROVENANCE
        else _FALLBACK_PROVENANCE_WEIGHT_SCALE
    )


def _quality_score_for(record: FaceQARecord) -> float:
    """Return a deterministic quality score for representative selection.

    Combines Sort-style signals into a single 0-1 score. Image-derived quality
    terms (blur and black-pixel ratio) are down-weighted when they came from a
    fallback thumbnail rather than an authoritative frame crop, so stale or
    low-resolution fallback metrics do not over-select a representative.
    """
    blur = record.blur_score or 0.0
    side = 0.0
    if record.resolution and len(record.resolution) >= 2:
        try:
            side = float(min(int(record.resolution[0]), int(record.resolution[1])))
        except (TypeError, ValueError):
            side = 0.0
    yaw = abs(record.yaw or 0.0)
    pitch = abs(record.pitch or 0.0)
    roll = abs(record.roll or 0.0)
    distance = record.average_distance or 0.0
    black = record.black_pixel_ratio or 0.0
    image_scale = _record_image_metric_scale(record)
    blur_term = math.tanh(blur / 6.0)
    res_term = math.tanh(side / 512.0)
    pose_term = 1.0 - math.tanh((yaw + pitch + roll) / 90.0)
    distance_term = 1.0 - math.tanh(distance * 4.0)
    black_term = 1.0 - math.tanh(black * 4.0)
    weights = (1.0, 0.8, 0.6, 0.6, 0.4)
    score = (
        weights[0] * image_scale * blur_term
        + weights[1] * res_term
        + weights[2] * pose_term
        + weights[3] * distance_term
        + weights[4] * image_scale * black_term
    )
    return round(score / sum(weights), 6)


_IDENTITY_GUARDRAIL_DECISIONS: frozenset[str] = frozenset(
    {"inlier", "borderline", "review", "outlier", "reject"}
)
_IDENTITY_GUARDRAIL_FLAGS: frozenset[str] = frozenset(
    {"inlier", "borderline", "outlier", "reject"}
)


def _has_identity_guardrail(record: FaceQARecord) -> bool:
    """Return whether identity was actually classified for guardrail use."""
    return (
        record.identity_final_decision in _IDENTITY_GUARDRAIL_DECISIONS
        or record.identity_quality_flag in _IDENTITY_GUARDRAIL_FLAGS
    )


def _features_for(record: FaceQARecord) -> RepresentationFeatures:
    """Return the normalised representation feature vector for one record."""
    has_identity = _has_identity_guardrail(record)
    identity_outlier = record.identity_quality_flag in {"outlier", "reject"} or (
        record.identity_final_decision in {"outlier", "reject"}
    )

    def _norm_angle(value: float | None) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        return max(-1.0, min(1.0, value / 90.0))

    def _norm_unit(value: float | None) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        return max(0.0, min(1.5, value))

    def _norm_luma(value: float | None) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        return max(0.0, min(1.0, value / 255.0))

    def _norm_ratio(value: float | None) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        # 1.0 == balanced. Compress so 2.0 → ~0.5, 0.5 → ~-0.5.
        return math.tanh(value - 1.0)

    def _norm_warmth(value: float | None) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        return math.tanh(value / 60.0)

    # Determine bucket-level views. These come from the coverage layer directly
    # (we don't recompute them here to avoid drifting from the canonical impl).
    pose_bucket = None
    pitch_bucket = None
    lighting_bucket = None
    from lib.faceqa.coverage import (
        _lighting_bucket,
        _pitch_bucket,
        _pose_bucket,
    )

    try:
        pose_bucket = _pose_bucket(record)
        pitch_bucket = _pitch_bucket(record)
        lighting_bucket = _lighting_bucket(record)
    except Exception:  # pylint:disable=broad-except
        logger.debug("Could not compute bucket views for %s:%s", record.frame, record.face_index)

    has_metrics = any(
        value is not None
        for value in (
            record.yaw,
            record.pitch,
            record.mouth_openness,
            record.smile_proxy,
            record.eye_closure,
            record.mean_luminance,
            record.left_right_ratio,
        )
    )

    return RepresentationFeatures(
        yaw=_norm_angle(record.yaw),
        pitch=_norm_angle(record.pitch),
        roll=_norm_angle(record.roll),
        mouth_openness=_norm_unit(record.mouth_openness),
        smile_proxy=_norm_unit(record.smile_proxy),
        eye_closure=_norm_unit(record.eye_closure),
        expression_asymmetry=_norm_unit(record.expression_asymmetry),
        luminance=_norm_luma(record.mean_luminance),
        contrast=_norm_luma(record.contrast),
        left_right_ratio=_norm_ratio(record.left_right_ratio),
        top_bottom_ratio=_norm_ratio(record.top_bottom_ratio),
        color_warmth=_norm_warmth(record.color_warmth),
        saturation=_norm_luma(record.saturation),
        average_distance=_norm_unit(record.average_distance),
        expression_bucket=record.expression_bucket,
        lighting_bucket=lighting_bucket,
        pose_bucket=pose_bucket,
        pitch_bucket=pitch_bucket,
        frame_index=_frame_index(record.frame),
        quality_score=_quality_score_for(record),
        has_identity=has_identity,
        identity_outlier=bool(identity_outlier),
        has_metrics=has_metrics,
        image_metrics_provenance=getattr(record, "image_metrics_provenance", None),
    )


# Image-derived features whose trust depends on whether the metrics came from
# the source frame crop (authoritative) or a fallback (thumbnail / face image).
# When either side of the pair is fallback-derived, these features contribute
# at reduced weight so the redundancy distance is not dominated by stale or
# resolution-degraded lighting / framing numbers.
_IMAGE_DERIVED_FEATURES: frozenset[str] = frozenset(
    {
        "luminance",
        "contrast",
        "left_right_ratio",
        "top_bottom_ratio",
        "color_warmth",
        "saturation",
    }
)
_IMAGE_DERIVED_BUCKETS: frozenset[str] = frozenset({"lighting_bucket"})
_FALLBACK_PROVENANCE_WEIGHT_SCALE: float = 0.5
_FRAME_PROVENANCE = "frame_aligned_crop"


def _provenance_scale(
    a: RepresentationFeatures,
    b: RepresentationFeatures,
    attr: str,
) -> float:
    """Return the weight scale for one feature given the pair's provenance.

    Frame-derived metrics on both sides keep the full weight. If either side
    used a thumbnail/face fallback, image-derived features (lighting /
    framing) are halved so the distance still moves but is not dominated by
    a source the user shouldn't fully trust.
    """
    if attr not in _IMAGE_DERIVED_FEATURES and attr not in _IMAGE_DERIVED_BUCKETS:
        return 1.0
    if (
        a.image_metrics_provenance == _FRAME_PROVENANCE
        and b.image_metrics_provenance == _FRAME_PROVENANCE
    ):
        return 1.0
    return _FALLBACK_PROVENANCE_WEIGHT_SCALE


def _representation_distance(
    a: RepresentationFeatures,
    b: RepresentationFeatures,
) -> tuple[float, int]:
    """Return ``(weighted_distance, compared_dimensions)``.

    Missing features contribute zero to both the numerator and the denominator;
    the caller treats low ``compared_dimensions`` as low confidence (review).
    Image-derived features (lighting / framing) are scaled down when either
    face's image metrics came from a non-frame fallback, so the original
    "d=0.39 from stale lighting" pathology can't slip through.
    """
    numerator = 0.0
    denominator = 0.0
    compared = 0
    for attr, weight in FEATURE_WEIGHTS.items():
        va = getattr(a, attr)
        vb = getattr(b, attr)
        if va is None or vb is None:
            continue
        diff = float(va) - float(vb)
        effective_weight = weight * _provenance_scale(a, b, attr)
        numerator += effective_weight * (diff * diff)
        denominator += effective_weight
        compared += 1
    if denominator == 0.0:
        return math.inf, 0
    bucket_penalty = 0.0
    bucket_total = 0.0
    for attr, weight in (("expression_bucket", 0.4), ("lighting_bucket", 0.3)):
        va = getattr(a, attr)
        vb = getattr(b, attr)
        if va is None or vb is None or va == "unknown" or vb == "unknown":
            continue
        bucket_total += weight
        if va != vb:
            bucket_penalty += weight * _provenance_scale(a, b, attr)
    distance = math.sqrt(numerator / denominator)
    if bucket_total > 0:
        # Up-weight visually-different buckets even if continuous features look similar.
        distance += (bucket_penalty / bucket_total) * 0.5
    return distance, compared


def _temporal_confidence(
    frame_index_a: int | None,
    frame_index_b: int | None,
    config: AggressivenessConfig,
) -> tuple[float, int | None]:
    """Return ``(temporal_confidence, frame_distance)`` for the pair.

    Close frames produce high confidence (≈1.0); far frames lower confidence
    (≈0.5); missing frame indices return 0.7 (neutral / unknown).
    """
    if frame_index_a is None or frame_index_b is None:
        return 0.7, None
    distance = abs(int(frame_index_a) - int(frame_index_b))
    if distance <= config.temporal_close_window:
        return 1.0, distance
    if distance >= config.temporal_far_window:
        return 0.5, distance
    span = max(1, config.temporal_far_window - config.temporal_close_window)
    return round(1.0 - 0.5 * ((distance - config.temporal_close_window) / span), 4), distance


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _build_redundancy_clusters(
    features: list[RepresentationFeatures],
    config: AggressivenessConfig,
) -> tuple[list[list[int]], dict[tuple[int, int], tuple[float, int, float, int | None]]]:
    """Union-find clusters by redundancy edges plus per-pair edge data."""
    n = len(features)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    edges: dict[tuple[int, int], tuple[float, int, float, int | None]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            distance, compared = _representation_distance(features[i], features[j])
            t_conf, frame_distance = _temporal_confidence(
                features[i].frame_index, features[j].frame_index, config
            )
            edges[(i, j)] = (distance, compared, t_conf, frame_distance)
            if compared < 3:
                # Not enough signal to cluster confidently; leave separate.
                continue
            obvious = distance <= config.obvious_duplicate_threshold
            redundant = distance <= config.representation_distance_threshold and t_conf >= 0.8
            if obvious or redundant:
                union(i, j)
    clusters_by_root: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters_by_root.setdefault(root, []).append(idx)
    components = sorted(clusters_by_root.values(), key=lambda c: c[0])
    return components, edges


# ---------------------------------------------------------------------------
# Effective coverage + decisions
# ---------------------------------------------------------------------------


def _bucket_of(record: FaceQARecord, dimension: str) -> str:
    from lib.faceqa.coverage import (
        _lighting_bucket,
        _pitch_bucket,
        _pose_bucket,
    )

    if dimension == "pose":
        return str(_pose_bucket(record))
    if dimension == "pitch":
        return str(_pitch_bucket(record))
    if dimension == "expression":
        return record.expression_bucket or "unknown"
    if dimension == "lighting":
        return str(_lighting_bucket(record))
    return "unknown"


def _effective_coverage(
    records: list[FaceQARecord],
    cluster_of: dict[int, int],
) -> dict[str, EffectiveCoverageDimension]:
    """Compute per-dimension effective coverage.

    ``effective_counts`` is the number of distinct redundancy clusters that
    touch each bucket; ``raw_counts`` is the legacy per-face count. The
    ``redundancy_ratio`` per bucket is ``raw / max(effective, 1)``.
    """
    result: dict[str, EffectiveCoverageDimension] = {}
    for dimension in EFFECTIVE_COVERAGE_DIMENSIONS:
        raw: dict[str, int] = {}
        cluster_buckets: dict[str, set[int]] = {}
        for idx, record in enumerate(records):
            bucket = _bucket_of(record, dimension)
            raw[bucket] = raw.get(bucket, 0) + 1
            cluster_buckets.setdefault(bucket, set()).add(cluster_of[idx])
        effective = {bucket: len(clusters) for bucket, clusters in cluster_buckets.items()}
        ratios = {
            bucket: round(raw[bucket] / max(effective.get(bucket, 1), 1), 4) for bucket in raw
        }
        result[dimension] = EffectiveCoverageDimension(
            dimension=dimension,
            raw_counts=raw,
            effective_counts=effective,
            redundancy_ratios=ratios,
        )
    return result


def _classify_buckets(
    effective: dict[str, EffectiveCoverageDimension],
    config: AggressivenessConfig,
    *,
    total_faces: int,
    min_bucket_pct: float,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return ``(protected, surplus)`` (dimension, bucket) sets.

    A bucket is *protected* when its effective coverage (distinct redundancy
    clusters) sits at or below the readiness floor implied by ``min_bucket_pct``,
    or — when there is no coverage signal — at or below the aggressiveness
    preset's minimum effective bucket count.

    A bucket is *surplus* when its effective coverage is safely above the
    readiness floor by at least ``coverage_surplus_margin``. Surplus buckets
    skip protection budget entirely and prune more aggressively.
    """
    protected: set[tuple[str, str]] = set()
    surplus: set[tuple[str, str]] = set()
    pct_floor = max(0.0, float(min_bucket_pct)) / 100.0
    pct_count = max(0.0, pct_floor * total_faces)
    fallback_floor = float(config.min_effective_bucket_count)
    fragile_floor = max(pct_count, fallback_floor) if pct_count > 0 else fallback_floor
    surplus_threshold = fragile_floor * config.coverage_surplus_margin
    for dimension, dim in effective.items():
        for bucket, count in dim.effective_counts.items():
            if bucket == "unknown":
                continue
            if count <= fragile_floor:
                protected.add((dimension, bucket))
            elif count >= surplus_threshold:
                surplus.add((dimension, bucket))
    return protected, surplus


def _bucket_keys_for(record: FaceQARecord) -> dict[str, str]:
    return {dim: _bucket_of(record, dim) for dim in EFFECTIVE_COVERAGE_DIMENSIONS}


def _record_is_in_protected_bucket(
    record: FaceQARecord,
    protected: set[tuple[str, str]],
) -> bool:
    keys = _bucket_keys_for(record)
    return any((dim, bucket) in protected for dim, bucket in keys.items())


def _record_is_in_surplus_bucket(
    record: FaceQARecord,
    surplus: set[tuple[str, str]],
) -> bool:
    keys = _bucket_keys_for(record)
    return any((dim, bucket) in surplus for dim, bucket in keys.items())


def _representative_safety_reason(features: RepresentationFeatures) -> str | None:
    """Return a review reason if a representative cannot safely default to keep."""
    if features.identity_outlier:
        return "identity outlier — review even when selected as representative"
    if not features.has_metrics:
        return "missing representation metrics — review even when singleton"
    return None


def _decide_single_member(
    *,
    member_features: RepresentationFeatures,
    distance: float,
    compared: int,
    temporal_confidence: float,
    config: AggressivenessConfig,
    member_record: FaceQARecord,
    rep_record: FaceQARecord,
    protection_budget_remaining: int,
    member_in_surplus_bucket: bool,
) -> tuple[str, str, int]:
    """Return ``(recommendation, reason, budget_after)`` for one cluster member.

    Order of decisions:
    - missing metrics → review
    - identity outlier → review
    - **missing identity embedding** → review (guardrail: without identity
      verification we cannot confidently keep or prune *any* member, even
      when it looks like an alternate or sits in a protected bucket).
    - meaningful pose/expression/lighting variation → keep (alternate)
    - **obvious duplicate** (hard redundancy floor) → prune, even when the
      bucket is fragile; protection budget never rescues exact duplicates.
    - protected fragile bucket with budget remaining → keep
    - visually similar but temporally distant → review
    - otherwise → prune (more aggressive when bucket is in surplus)
    """
    if not member_features.has_metrics:
        return REVIEW, "missing representation metrics — review", protection_budget_remaining
    if member_features.identity_outlier:
        return (
            REVIEW,
            "identity outlier — review for mismatched subject",
            protection_budget_remaining,
        )
    if not member_features.has_identity:
        # Identity is the cross-subject guardrail. Without it we cannot
        # confidently keep an alternate, spend protection budget on it, or
        # prune it as redundant — every non-obvious outcome routes to review.
        return (
            REVIEW,
            "missing identity embedding guardrail — review before keep or prune",
            protection_budget_remaining,
        )
    bucket_keys_member = _bucket_keys_for(member_record)
    bucket_keys_rep = _bucket_keys_for(rep_record)
    if any(
        bucket_keys_member.get(dim) != bucket_keys_rep.get(dim)
        for dim in EFFECTIVE_COVERAGE_DIMENSIONS
        if bucket_keys_member.get(dim) not in (None, "unknown")
        and bucket_keys_rep.get(dim) not in (None, "unknown")
    ):
        return (
            KEEP,
            "adds meaningful pose/expression/lighting variation",
            protection_budget_remaining,
        )
    # Hard redundancy floor: obvious duplicates always prune, regardless of
    # bucket protection. Temporal-far obvious duplicates still route to review
    # because we cannot confidently call them duplicates.
    if distance <= config.obvious_duplicate_threshold:
        if temporal_confidence < 0.7:
            return (
                REVIEW,
                "obvious-similarity match but temporally distant — "
                f"temporal_confidence={temporal_confidence:.2f}",
                protection_budget_remaining,
            )
        return (
            PRUNE,
            f"obvious duplicate of representative (distance={distance:.2f})",
            protection_budget_remaining,
        )
    if protection_budget_remaining > 0:
        return (
            KEEP,
            "protects underrepresented effective coverage bucket (budgeted)",
            protection_budget_remaining - 1,
        )
    if temporal_confidence < 0.7:
        return (
            REVIEW,
            "visually similar but temporally distant — "
            f"temporal_confidence={temporal_confidence:.2f}",
            protection_budget_remaining,
        )
    if compared < 4:
        return REVIEW, "few comparable metrics — review", protection_budget_remaining
    if member_in_surplus_bucket:
        return (
            PRUNE,
            f"redundant within surplus coverage bucket (distance={distance:.2f})",
            protection_budget_remaining,
        )
    return (
        PRUNE,
        f"redundant with representative (distance={distance:.2f}, "
        f"temporal_confidence={temporal_confidence:.2f})",
        protection_budget_remaining,
    )


def _protection_budget_for_cluster(
    *,
    members: list[int],
    representative_index: int,
    record_list: list[FaceQARecord],
    protected: set[tuple[str, str]],
    surplus: set[tuple[str, str]],
    config: AggressivenessConfig,
) -> int:
    """Return how many additional keeps the cluster gets beyond the representative.

    Protection takes priority over surplus: if the representative sits in *any*
    fragile bucket the cluster gets the full protection budget, even when its
    other dimensions are surplus. Otherwise surplus and neutral buckets get
    budget 0 (representative-only keep).
    """
    if not members:
        return 0
    rep_record = record_list[representative_index]
    if _record_is_in_protected_bucket(rep_record, protected):
        return max(0, config.min_effective_bucket_count - 1)
    return 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_redundancy(
    records: T.Sequence[FaceQARecord],
    *,
    aggressiveness: str = "balanced",
    coverage: FacesetCoverageReport | None = None,
    min_bucket_pct: float = 5.0,
) -> RedundancyReport:
    """Compute coverage-aware representation redundancy clusters and recs."""
    if aggressiveness not in _AGGRESSIVENESS_PRESETS:
        raise ValueError(
            f"Unknown aggressiveness '{aggressiveness}'. "
            f"Expected one of: {', '.join(AGGRESSIVENESS_LEVELS)}"
        )
    config = _AGGRESSIVENESS_PRESETS[aggressiveness]
    record_list = list(records)
    n = len(record_list)
    if n == 0:
        return RedundancyReport(aggressiveness=aggressiveness)

    features = [_features_for(record) for record in record_list]
    components, edges = _build_redundancy_clusters(features, config)
    cluster_of: dict[int, int] = {}
    for cluster_id, members in enumerate(components):
        for member in members:
            cluster_of[member] = cluster_id

    effective = _effective_coverage(record_list, cluster_of)
    if coverage is not None:
        # Sanity-check raw coverage against coverage report when supplied.
        for dim in effective.values():
            counts = coverage.bucket_counts.get(dim.dimension, {})
            for bucket, raw_value in counts.items():
                # Only ensure we did not under-count anything that shows in the
                # canonical coverage report; we never overwrite our own data.
                dim.raw_counts.setdefault(bucket, raw_value)
    protected, surplus = _classify_buckets(
        effective,
        config,
        total_faces=n,
        min_bucket_pct=min_bucket_pct,
    )

    output_records: list[RedundancyRecord] = []
    for cluster_id, members in enumerate(components):
        rep_index = max(members, key=lambda idx: (features[idx].quality_score, -idx))
        rep_features = features[rep_index]
        rep_record = record_list[rep_index]
        rep_buckets = _bucket_keys_for(rep_record)
        safety_reason = _representative_safety_reason(rep_features)
        if safety_reason is not None:
            rep_recommendation = REVIEW
            rep_reason = safety_reason
        elif len(members) > 1:
            rep_recommendation = KEEP
            rep_reason = "representative of redundancy cluster"
        else:
            rep_recommendation = KEEP
            rep_reason = "unique representation (singleton cluster)"
        output_records.append(
            RedundancyRecord(
                frame=rep_record.frame,
                face_index=rep_record.face_index,
                cluster_id=cluster_id,
                cluster_size=len(members),
                representative=True,
                recommendation=rep_recommendation,
                quality_score=rep_features.quality_score,
                redundancy_distance_to_representative=0.0,
                temporal_distance_to_representative=0,
                temporal_confidence=1.0,
                pose_bucket=rep_buckets["pose"],
                pitch_bucket=rep_buckets["pitch"],
                expression_bucket=rep_buckets["expression"],
                lighting_bucket=rep_buckets["lighting"],
                reason=rep_reason,
                identity_outlier=rep_features.identity_outlier,
                has_identity=rep_features.has_identity,
            )
        )
        # Allocate a small protection budget for fragile-bucket clusters and
        # consume it in descending quality order so the best alternates win.
        budget = _protection_budget_for_cluster(
            members=members,
            representative_index=rep_index,
            record_list=record_list,
            protected=protected,
            surplus=surplus,
            config=config,
        )
        non_rep_members = [m for m in members if m != rep_index]
        non_rep_members.sort(key=lambda idx: (-features[idx].quality_score, idx))
        for member in non_rep_members:
            member_features = features[member]
            member_record = record_list[member]
            buckets = _bucket_keys_for(member_record)
            edge_key = (min(member, rep_index), max(member, rep_index))
            distance, compared, t_conf, t_dist = edges.get(edge_key, (math.inf, 0, 0.7, None))
            recommendation, reason, budget = _decide_single_member(
                member_features=member_features,
                distance=distance,
                compared=compared,
                temporal_confidence=t_conf,
                config=config,
                member_record=member_record,
                rep_record=rep_record,
                protection_budget_remaining=budget,
                member_in_surplus_bucket=_record_is_in_surplus_bucket(member_record, surplus),
            )
            output_records.append(
                RedundancyRecord(
                    frame=member_record.frame,
                    face_index=member_record.face_index,
                    cluster_id=cluster_id,
                    cluster_size=len(members),
                    representative=False,
                    recommendation=recommendation,
                    quality_score=member_features.quality_score,
                    redundancy_distance_to_representative=round(distance, 4)
                    if math.isfinite(distance)
                    else None,
                    temporal_distance_to_representative=t_dist,
                    temporal_confidence=t_conf,
                    pose_bucket=buckets["pose"],
                    pitch_bucket=buckets["pitch"],
                    expression_bucket=buckets["expression"],
                    lighting_bucket=buckets["lighting"],
                    reason=reason,
                    identity_outlier=member_features.identity_outlier,
                    has_identity=member_features.has_identity,
                )
            )

    keep = sum(1 for r in output_records if r.recommendation == KEEP)
    review = sum(1 for r in output_records if r.recommendation == REVIEW)
    prune = sum(1 for r in output_records if r.recommendation == PRUNE)
    multi = sum(1 for c in components if len(c) > 1)
    return RedundancyReport(
        aggressiveness=aggressiveness,
        total_faces=n,
        cluster_count=len(components),
        multi_face_clusters=multi,
        keep_count=keep,
        review_count=review,
        prune_candidate_count=prune,
        effective_coverage=effective,
        protected_buckets=sorted(f"{dim}:{bucket}" for dim, bucket in protected),
        records=output_records,
    )


__all__ = get_module_objects(__name__)


# Silence the "imported but unused" lint flag for `np` — we import it because
# downstream tooling and tests rely on the same numerical environment that the
# rest of the FaceQA layer uses, even though this module's hot path is pure
# Python math.
_ = np
