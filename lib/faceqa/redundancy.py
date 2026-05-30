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

from lib.faceqa.buckets import (
    lighting_bucket as _lighting_bucket,
)
from lib.faceqa.buckets import (
    pitch_bucket as _pitch_bucket,
)
from lib.faceqa.buckets import (
    pose_bucket as _pose_bucket,
)
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
    # Issue #208 — fine pruning compatibility signals. Bands are integer
    # codes (lower idx = lower magnitude); ``None`` means the underlying
    # feature was missing and the band cannot gate.
    yaw_micro_band: int | None = None
    """Fine yaw band (7.5 deg) for pruning compatibility — preserves
    micro-yaw / slight-head-turn variation that the coarse readiness
    bucket would otherwise compress into one ``frontal`` blob."""
    pitch_micro_band: int | None = None
    """Fine pitch band (7.5 deg)."""
    mouth_openness_band: int | None = None
    """Mouth-open intensity band for distinguishing closed / slight /
    open / wide-open mouth poses."""
    smile_proxy_band: int | None = None
    """Smile intensity band for distinguishing neutral / slight / broad
    smiles."""
    eye_closure_band: int | None = None
    """Eye-closure intensity band for distinguishing open / squint /
    closed eyes."""
    brow_raise_band: int | None = None
    """Brow-raise intensity band for distinguishing neutral / raised
    brows."""
    expression_asymmetry_band: int | None = None
    """Expression-asymmetry band for distinguishing symmetric /
    asymmetric faces."""
    appearance_mode: tuple[int, int, int, int] | None = None
    """Discrete (luminance, color_warmth, saturation, contrast) signature
    used as a lightweight visual-look-mode fingerprint. Different
    appearance modes block normal redundancy edges so visually distinct
    looks (hairstyle / makeup / event / lighting) are preserved by
    default."""


@dataclass
class RedundancyRecord:
    """Per-face redundancy outcome.

    The new ``compared_metrics`` and ``edge_eligibility`` fields are part
    of the explicit constrained-redundancy pipeline from issue #199 — they
    expose enough state on each output row to audit why a face landed in a
    cluster (or stayed alone). Defaults are provided so older test
    constructions that pre-date #199 still build.
    """

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
    # Issue #199 — per-record edge diagnostics.
    compared_metrics: int = 0
    """Number of comparable continuous-feature dimensions between this face
    and its cluster representative. Treat low values (<3) as low confidence."""
    edge_eligibility: str | None = None
    """If a redundancy edge was created between this face and its
    representative, the reason the edge passed eligibility (e.g. ``obvious
    duplicate``, ``redundant + safe temporal``). ``None`` for singleton
    clusters and representatives. See :func:`can_create_redundancy_edge`."""
    # Issue #204 — per-member component diagnostics that distinguish
    # direct redundancy from transitive-chain membership.
    nearest_neighbor_distance: float | None = None
    """Distance to the nearest member within the same (post-split)
    component. ``0.0`` for singletons. Helps audit whether a member is
    truly close to ANY other cluster member or only reachable through
    a chain."""
    direct_to_representative: bool = True
    """``True`` when the member's pairwise distance to the cluster
    representative is within ``representation_distance_threshold *
    _MEMBER_TO_REP_SCALE``. ``False`` indicates the member sits in the
    same component only via transitive redundancy chaining and should be
    treated more conservatively."""
    component_diameter: float | None = None
    """Maximum pairwise distance among all members of the (post-split)
    component. Surfaced so the user can spot sprawling clusters even
    when the splitter chose not to break them up."""
    # Issue #204 follow-up — richer per-member diagnostics so reasons
    # can distinguish "redundant with representative" from "redundant
    # through nearest neighbour" from "split from oversize component".
    nearest_neighbor_frame: str | None = None
    """Frame name of the nearest cluster neighbour, when one exists."""
    nearest_neighbor_face_index: int | None = None
    """Face index of the nearest cluster neighbour, when one exists."""
    nearest_neighbor_edge_eligibility: str | None = None
    """Eligibility reason for the graph edge between this face and its
    nearest cluster neighbour. ``None`` when no edge was scored
    (singletons or post-split components where the original union-find
    never saw the pair)."""
    connected_via_representative: bool = True
    """``True`` when the member had an ELIGIBLE redundancy edge to the
    cluster representative in the original graph; ``False`` flags
    transitive-chain membership where the member's redundancy is only
    reachable through another neighbour. Distinct from
    ``direct_to_representative`` (which is a distance check)."""
    split_reason: str | None = None
    """Populated when the diameter splitter touched the parent
    component:

    * ``"split from oversize component due to diameter"`` — the member
      ended up in a sub-cluster because the parent component's pairwise
      diameter exceeded the limit.
    * ``"component too large for diameter check (N > cap) — kept as-is"``
      — the parent component exceeded
      ``_MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT``, so the splitter
      bypassed it to avoid O(N^2) memory / CPU on a worst-case shape.

    ``None`` indicates the cluster was within the diameter limit and
    untouched by the splitter."""
    # Issue #208 follow-up 2 — surface the fine visual signals on each
    # output record so the contact-sheet renderer can sort tiles by
    # visual mode and downstream consumers can audit the cluster's
    # visual coherence without re-running ``_features_for``.
    appearance_mode: tuple[int, int, int, int] | None = None
    """Discrete (luminance, warmth, saturation, contrast) appearance
    signature. ``None`` when one or more channels were missing on the
    underlying record."""
    yaw_micro_band: int | None = None
    """Signed 7.5-degree yaw micro-band index. ``None`` for missing yaw."""
    pitch_micro_band: int | None = None
    """Signed 7.5-degree pitch micro-band index. ``None`` for missing pitch."""
    mouth_openness_band: int | None = None
    """Discrete mouth-openness intensity band. ``None`` for missing
    mouth_openness."""
    smile_proxy_band: int | None = None
    """Discrete smile-proxy intensity band. ``None`` for missing
    smile_proxy."""

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
            "compared_metrics": self.compared_metrics,
            "edge_eligibility": self.edge_eligibility,
            "nearest_neighbor_distance": self.nearest_neighbor_distance,
            "direct_to_representative": self.direct_to_representative,
            "component_diameter": self.component_diameter,
            "nearest_neighbor_frame": self.nearest_neighbor_frame,
            "nearest_neighbor_face_index": self.nearest_neighbor_face_index,
            "nearest_neighbor_edge_eligibility": self.nearest_neighbor_edge_eligibility,
            "connected_via_representative": self.connected_via_representative,
            "split_reason": self.split_reason,
            "appearance_mode": (
                list(self.appearance_mode) if self.appearance_mode is not None else None
            ),
            "yaw_micro_band": self.yaw_micro_band,
            "pitch_micro_band": self.pitch_micro_band,
            "mouth_openness_band": self.mouth_openness_band,
            "smile_proxy_band": self.smile_proxy_band,
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
    # Issue #204 — cluster-health metrics for spotting giant-component
    # pathologies before they hit the pruning policy.
    largest_cluster_size: int = 0
    """Member count of the biggest (post-split) cluster."""
    largest_cluster_ratio: float = 0.0
    """``largest_cluster_size / total_faces`` (0.0 when no faces)."""
    giant_component_warning: bool = False
    """``True`` when the cluster topology looks pathological per the
    rules in :func:`_compute_cluster_health`:

    * ``largest_cluster_ratio > 0.25``, OR
    * ``largest_cluster_size > 1000``, OR
    * ``multi_face_clusters <= 2 AND total_faces > 1000``.

    Surfaces in the JSON output so callers can audit topology issues
    without reading every per-record diagnostic."""
    component_spread_p95: float | None = None
    """95th-percentile per-member ``nearest_neighbor_distance`` across the
    report; high values mean members are reaching far for their nearest
    redundancy neighbour and the clusters are likely too sprawling."""
    component_diameter_max: float | None = None
    """Largest per-cluster diameter observed after splitting; capped by
    the splitter's diameter limit when no cluster exceeded it."""
    # Issue #208 — readiness FAIL safety cap diagnostic. ``None`` when the
    # cap did not fire (either readiness wasn't supplied OR the prune
    # ratio was within bounds).
    readiness_safety_cap_applied: bool = False
    """``True`` when ``compute_redundancy`` demoted borderline prune
    candidates to review because readiness was FAIL and the prune ratio
    would otherwise have been extreme."""
    readiness_safety_cap_reason: str | None = None
    """Human-readable explanation of the cap (``None`` when not
    applied)."""
    readiness_safety_cap_demotions: int = 0
    """Number of records whose recommendation was demoted from
    ``prune_candidate`` to ``review`` by the cap."""

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
            "largest_cluster_size": self.largest_cluster_size,
            "largest_cluster_ratio": self.largest_cluster_ratio,
            "giant_component_warning": self.giant_component_warning,
            "component_spread_p95": self.component_spread_p95,
            "component_diameter_max": self.component_diameter_max,
            "readiness_safety_cap_applied": self.readiness_safety_cap_applied,
            "readiness_safety_cap_reason": self.readiness_safety_cap_reason,
            "readiness_safety_cap_demotions": self.readiness_safety_cap_demotions,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Match the LAST run of digits in a frame name (e.g. "frame_000123.png" or
# "shot_07_000123.jpg") so ``_frame_index`` can skip ``findall`` + indexing
# into the materialised list (issue #199 follow-up).
_FRAME_LAST_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


def _frame_index(frame: str) -> int | None:
    """Return the trailing numeric index of a frame name when present."""
    match = _FRAME_LAST_NUMBER_RE.search(frame)
    return int(match.group(1)) if match is not None else None


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


# Per-feature normalisation helpers — hoisted from ``_features_for`` so
# they are not redefined on every record (issue #192 P2). All five share
# the same ``None`` / ``not finite`` short-circuit; the body differs only
# in the clamp + scaling.


def _norm_angle(value: float | None) -> float | None:
    """Clamp an angular feature to [-1, 1] after dividing by 90 deg."""
    if value is None or not math.isfinite(value):
        return None
    return max(-1.0, min(1.0, value / 90.0))


def _norm_unit(value: float | None) -> float | None:
    """Clamp a unit-scale feature to [0, 1.5]."""
    if value is None or not math.isfinite(value):
        return None
    return max(0.0, min(1.5, value))


def _norm_luma(value: float | None) -> float | None:
    """Clamp a 0-255 luminance feature to [0, 1] after dividing by 255."""
    if value is None or not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value / 255.0))


def _norm_ratio(value: float | None) -> float | None:
    """Compress a ratio centred on 1.0 — 2.0 -> ~0.5, 0.5 -> ~-0.5."""
    if value is None or not math.isfinite(value):
        return None
    return math.tanh(value - 1.0)


def _norm_warmth(value: float | None) -> float | None:
    """Compress a colour-warmth signal centred on 0."""
    if value is None or not math.isfinite(value):
        return None
    return math.tanh(value / 60.0)


# Issue #208 — fine pruning bands. Band edges are tuples of ascending
# thresholds; ``_band_index`` returns 0..len(edges) so adjacency = a gap of
# exactly 1. Each band's edges are conservative enough that "obviously
# similar" faces land in the same band while visually distinct ones split.
_YAW_MICRO_DEG: float = 7.5
_PITCH_MICRO_DEG: float = 7.5

_MOUTH_OPENNESS_EDGES: tuple[float, ...] = (0.05, 0.12, 0.22, 0.35)
_SMILE_PROXY_EDGES: tuple[float, ...] = (0.10, 0.22, 0.40)
_EYE_CLOSURE_EDGES: tuple[float, ...] = (0.20, 0.45, 0.70)
_BROW_RAISE_EDGES: tuple[float, ...] = (0.10, 0.25, 0.45)
_EXPRESSION_ASYMMETRY_EDGES: tuple[float, ...] = (0.15, 0.30, 0.50)

# Appearance-mode band edges. These mirror the coverage thresholds where
# they exist (luminance / saturation) and add a signed-warmth band so a
# cool-lighting subject doesn't cluster with a warm-lighting subject.
_LUMINANCE_MODE_EDGES: tuple[float, ...] = (60.0, 100.0, 150.0, 200.0)
_WARMTH_MODE_EDGES: tuple[float, ...] = (-30.0, 0.0, 30.0)
_SATURATION_MODE_EDGES: tuple[float, ...] = (40.0, 80.0, 120.0)
_CONTRAST_MODE_EDGES: tuple[float, ...] = (15.0, 25.0, 40.0)


def _band_index(value: float | None, edges: tuple[float, ...]) -> int | None:
    """Return the 0-indexed bin position of ``value`` against ``edges``.

    ``edges`` is a tuple of ascending thresholds; the returned index is
    ``len(edges)`` for values at or above the last threshold. ``None``
    is returned when ``value`` is missing / non-finite so the gate
    cannot fire on missing data.
    """
    if value is None or not math.isfinite(value):
        return None
    for idx, edge in enumerate(edges):
        if value < edge:
            return idx
    return len(edges)


def _signed_angle_band(value: float | None, width: float) -> int | None:
    """Return the signed-yaw / signed-pitch micro-band index.

    ``round(value / width)`` is signed so the bands stay symmetric around
    0; ``None`` for missing / non-finite values.
    """
    if value is None or not math.isfinite(value):
        return None
    return int(round(value / width))


def _appearance_mode_signature(record: FaceQARecord) -> tuple[int, int, int, int] | None:
    """Return the discrete appearance-mode signature for a face.

    All four channels must be present; a missing channel collapses the
    signature to ``None`` so we don't block edges on partial data.
    """
    luminance = _band_index(record.mean_luminance, _LUMINANCE_MODE_EDGES)
    warmth = _band_index(record.color_warmth, _WARMTH_MODE_EDGES)
    saturation = _band_index(record.saturation, _SATURATION_MODE_EDGES)
    contrast = _band_index(record.contrast, _CONTRAST_MODE_EDGES)
    if None in (luminance, warmth, saturation, contrast):
        return None
    return (luminance, warmth, saturation, contrast)  # type: ignore[return-value]


def _features_for(record: FaceQARecord) -> RepresentationFeatures:
    """Return the normalised representation feature vector for one record."""
    has_identity = _has_identity_guardrail(record)
    identity_outlier = record.identity_quality_flag in {"outlier", "reject"} or (
        record.identity_final_decision in {"outlier", "reject"}
    )

    # Determine bucket-level views once per record. The redundancy
    # ``_features_for`` consumer used to recompute these per pairwise
    # comparison inside the O(N^2) clustering loop via inline imports
    # from coverage; with the new public bucket helpers (issue #192 P1)
    # and the cached labels here, the buckets are computed exactly once
    # per record and reused by every distance / decision call below.
    pose_bucket = None
    pitch_bucket = None
    lighting_bucket = None
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
        # Issue #208 — fine pruning compatibility signals.
        yaw_micro_band=_signed_angle_band(record.yaw, _YAW_MICRO_DEG),
        pitch_micro_band=_signed_angle_band(record.pitch, _PITCH_MICRO_DEG),
        mouth_openness_band=_band_index(record.mouth_openness, _MOUTH_OPENNESS_EDGES),
        smile_proxy_band=_band_index(record.smile_proxy, _SMILE_PROXY_EDGES),
        eye_closure_band=_band_index(record.eye_closure, _EYE_CLOSURE_EDGES),
        brow_raise_band=_band_index(record.brow_raise_proxy, _BROW_RAISE_EDGES),
        expression_asymmetry_band=_band_index(
            record.expression_asymmetry, _EXPRESSION_ASYMMETRY_EDGES
        ),
        appearance_mode=_appearance_mode_signature(record),
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


# Frozen ordering of the continuous-feature attributes used inside the
# per-pair distance calculation. Building a numpy ``(N, F)`` values matrix
# from this order lets ``_distance_with_ctx`` collapse the previous
# ``len(FEATURE_WEIGHTS)`` Python iterations + ``getattr`` calls into a
# handful of vectorised numpy ops (issue #192 P2 high).
_FEATURE_ATTRS: tuple[str, ...] = tuple(FEATURE_WEIGHTS.keys())
_FEATURE_WEIGHTS_ARRAY: np.ndarray = np.array(
    [FEATURE_WEIGHTS[attr] for attr in _FEATURE_ATTRS], dtype=np.float32
)
_FEATURE_IS_IMAGE: np.ndarray = np.array(
    [attr in _IMAGE_DERIVED_FEATURES for attr in _FEATURE_ATTRS], dtype=bool
)
# Bucket-penalty terms — kept as a small tuple, indexed by name. The
# bucket comparisons are cheap (string equality) so they stay in Python.
_BUCKET_PENALTY_TERMS: tuple[tuple[str, float], ...] = (
    ("expression_bucket", 0.4),
    ("lighting_bucket", 0.3),
)


@dataclass(frozen=True)
class _PairwiseContext:
    """Precomputed per-record state consumed by ``_distance_with_ctx``.

    ``values`` is a ``(N, F)`` float32 matrix of normalised feature values
    with ``np.nan`` in the slots where a record had ``None`` for that
    attribute. ``masks`` is the boolean availability mirror so the inner
    pair function can build the joint-availability vector with one AND.
    ``frame_provenance`` is a per-record boolean: True when the record's
    image metrics are frame-derived. ``expression_buckets`` /
    ``lighting_buckets`` are per-record string lists used for the bucket
    penalty.
    """

    values: np.ndarray
    masks: np.ndarray
    frame_provenance: np.ndarray
    expression_buckets: list[str | None]
    lighting_buckets: list[str | None]


def _build_pairwise_context(features: list[RepresentationFeatures]) -> _PairwiseContext:
    """Precompute per-record arrays consumed by the inner distance loop.

    Replaces the ``len(features) * len(FEATURE_WEIGHTS)`` ``getattr`` calls
    + ``_provenance_scale`` recomputes the previous shape paid per pair
    with one linear pass over records here (issue #192 P2 high).
    """
    n = len(features)
    f = len(_FEATURE_ATTRS)
    values = np.full((n, f), np.nan, dtype=np.float32)
    for row, feature in enumerate(features):
        for col, attr in enumerate(_FEATURE_ATTRS):
            value = getattr(feature, attr)
            if value is not None and math.isfinite(value):
                values[row, col] = value
    masks = ~np.isnan(values)
    frame_provenance = np.array(
        [feat.image_metrics_provenance == _FRAME_PROVENANCE for feat in features],
        dtype=bool,
    )
    expression_buckets = [feat.expression_bucket for feat in features]
    lighting_buckets = [feat.lighting_bucket for feat in features]
    return _PairwiseContext(
        values=values,
        masks=masks,
        frame_provenance=frame_provenance,
        expression_buckets=expression_buckets,
        lighting_buckets=lighting_buckets,
    )


def _distance_with_ctx(ctx: _PairwiseContext, i: int, j: int) -> tuple[float, int]:
    """Return ``(weighted_distance, compared_dimensions)`` for record pair ``(i, j)``.

    Vectorised counterpart of :func:`_representation_distance`. The slower
    Python-attribute shape is preserved as a backward-compatible public
    helper below for callers (and tests) that operate on individual
    ``RepresentationFeatures`` objects.
    """
    diff = ctx.values[i] - ctx.values[j]
    mask = ctx.masks[i] & ctx.masks[j]
    both_frame = bool(ctx.frame_provenance[i] and ctx.frame_provenance[j])
    if both_frame:
        scale = np.ones_like(_FEATURE_WEIGHTS_ARRAY)
    else:
        scale = np.where(_FEATURE_IS_IMAGE, _FALLBACK_PROVENANCE_WEIGHT_SCALE, 1.0)
    eff_w = _FEATURE_WEIGHTS_ARRAY * scale * mask
    denominator = float(eff_w.sum())
    if denominator == 0.0:
        return math.inf, 0
    sq = np.where(mask, diff * diff, 0.0)
    numerator = float((eff_w * sq).sum())
    compared = int(mask.sum())
    distance = math.sqrt(numerator / denominator)

    bucket_penalty = 0.0
    bucket_total = 0.0
    exp_i, exp_j = ctx.expression_buckets[i], ctx.expression_buckets[j]
    lit_i, lit_j = ctx.lighting_buckets[i], ctx.lighting_buckets[j]
    for attr, weight in _BUCKET_PENALTY_TERMS:
        if attr == "expression_bucket":
            va, vb = exp_i, exp_j
        else:
            va, vb = lit_i, lit_j
        if va is None or vb is None or va == "unknown" or vb == "unknown":
            continue
        bucket_total += weight
        if va != vb:
            # Lighting bucket is image-derived; reduced weight when either
            # side used a non-frame fallback.
            if attr == "lighting_bucket" and not both_frame:
                bucket_penalty += weight * _FALLBACK_PROVENANCE_WEIGHT_SCALE
            else:
                bucket_penalty += weight
    if bucket_total > 0:
        distance += (bucket_penalty / bucket_total) * 0.5
    return distance, compared


def _representation_distance(
    a: RepresentationFeatures,
    b: RepresentationFeatures,
) -> tuple[float, int]:
    """Return ``(weighted_distance, compared_dimensions)`` for one pair.

    The clustering loop now uses :func:`_distance_with_ctx` for hot-path
    vectorisation; this slower Python-attribute form is preserved as a
    public-style helper because tests (and external probes that operate
    on individual ``RepresentationFeatures`` objects) still call it
    directly.
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
    for attr, weight in _BUCKET_PENALTY_TERMS:
        va = getattr(a, attr)
        vb = getattr(b, attr)
        if va is None or vb is None or va == "unknown" or vb == "unknown":
            continue
        bucket_total += weight
        if va != vb:
            bucket_penalty += weight * _provenance_scale(a, b, attr)
    distance = math.sqrt(numerator / denominator)
    if bucket_total > 0:
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
# Edge eligibility — issue #199
#
# The constrained-redundancy pipeline gates EVERY union-find merge through
# ``can_create_redundancy_edge`` so the resulting graph represents SAFE
# redundancy relationships only. Questionable pairs no longer get silently
# unioned and then handled later inside the post-cluster decision layer —
# the graph itself encodes the safety constraints.
# ---------------------------------------------------------------------------


_MIN_COMPARED_FOR_EDGE: int = 3
_TEMPORAL_CONFIDENCE_FOR_REDUNDANT_EDGE: float = 0.8
_TEMPORAL_CONFIDENCE_FOR_OBVIOUS_EDGE: float = 0.7


# Ordered yaw / pitch bucket labels used by ``_bucket_compatibility`` to
# distinguish ADJACENT bucket pairs (allowed at stricter distance) from
# NON-ADJACENT pairs (blocked unless the pair is an obvious duplicate).
# Issue #204: chains like ``frontal -> slight -> profile -> extreme``
# previously collapsed into one component via transitive union; the
# adjacency-aware gate cuts that chain at the first non-adjacent hop.
_YAW_BUCKET_ORDER: tuple[str, ...] = (
    "left_extreme",
    "left_profile",
    "left_slight",
    "frontal",
    "right_slight",
    "right_profile",
    "right_extreme",
)
_PITCH_BUCKET_ORDER: tuple[str, ...] = (
    "down_extreme",
    "down",
    "neutral",
    "up",
    "up_extreme",
)
# Scale applied to ``representation_distance_threshold`` when adjacent
# pose/pitch buckets are bridged — the pair must be ~40% closer than a
# same-bucket pair to qualify.
_ADJACENT_BUCKET_DISTANCE_SCALE: float = 0.6


def _bucket_index(order: tuple[str, ...], bucket: str | None) -> int | None:
    """Return the position of ``bucket`` in ``order``, or ``None`` when
    the bucket is missing / unknown / outside the known sequence."""
    if bucket is None or bucket == "unknown":
        return None
    try:
        return order.index(bucket)
    except ValueError:
        return None


def _bucket_compatibility(
    features_a: RepresentationFeatures,
    features_b: RepresentationFeatures,
    *,
    distance: float,
    temporal_confidence: float,
    config: AggressivenessConfig,
) -> tuple[bool, str]:
    """Return ``(compatible, reason)`` for the hard bucket constraints.

    Issue #204 made these constraints first-class on the redundancy graph
    so transitive chains across bucket boundaries can no longer silently
    collapse into one giant component:

    * Pose (yaw) bucket mismatch BLOCKS the edge by default; adjacent
      buckets in :data:`_YAW_BUCKET_ORDER` may bridge only at stricter
      distance (``threshold * _ADJACENT_BUCKET_DISTANCE_SCALE``).
    * Pitch bucket mismatch uses the same adjacency rule.
    * Expression bucket mismatch BLOCKS the edge (no adjacency notion).
    * Lighting bucket mismatch BLOCKS the edge.
    * Obvious duplicates (``distance <= obvious_duplicate_threshold`` AND
      ``temporal_confidence >= obvious floor``) may OVERRIDE any bucket
      mismatch — the post-cluster review path catches the bucket-skew on
      the diagnostic side.
    * Unknown buckets do NOT hard-block (we can't tell if they match) but
      remain subject to the existing continuous-feature distance check.
    """
    obvious_override = (
        distance <= config.obvious_duplicate_threshold
        and temporal_confidence >= _TEMPORAL_CONFIDENCE_FOR_OBVIOUS_EDGE
    )
    strict_limit = config.representation_distance_threshold * _ADJACENT_BUCKET_DISTANCE_SCALE

    yaw_a = _bucket_index(_YAW_BUCKET_ORDER, features_a.pose_bucket)
    yaw_b = _bucket_index(_YAW_BUCKET_ORDER, features_b.pose_bucket)
    if yaw_a is not None and yaw_b is not None and yaw_a != yaw_b:
        gap = abs(yaw_a - yaw_b)
        if gap == 1:
            if not (distance <= strict_limit or obvious_override):
                return False, (
                    f"adjacent yaw buckets ({features_a.pose_bucket} vs "
                    f"{features_b.pose_bucket}) require stricter distance "
                    f"({distance:.2f} > {strict_limit:.2f})"
                )
        elif not obvious_override:
            return False, (
                f"yaw bucket mismatch ({features_a.pose_bucket} vs "
                f"{features_b.pose_bucket}) blocks redundancy edge"
            )

    pitch_a = _bucket_index(_PITCH_BUCKET_ORDER, features_a.pitch_bucket)
    pitch_b = _bucket_index(_PITCH_BUCKET_ORDER, features_b.pitch_bucket)
    if pitch_a is not None and pitch_b is not None and pitch_a != pitch_b:
        gap = abs(pitch_a - pitch_b)
        if gap == 1:
            if not (distance <= strict_limit or obvious_override):
                return False, (
                    f"adjacent pitch buckets ({features_a.pitch_bucket} vs "
                    f"{features_b.pitch_bucket}) require stricter distance "
                    f"({distance:.2f} > {strict_limit:.2f})"
                )
        elif not obvious_override:
            return False, (
                f"pitch bucket mismatch ({features_a.pitch_bucket} vs "
                f"{features_b.pitch_bucket}) blocks redundancy edge"
            )

    exp_a = features_a.expression_bucket
    exp_b = features_b.expression_bucket
    if (
        exp_a is not None
        and exp_b is not None
        and exp_a != "unknown"
        and exp_b != "unknown"
        and exp_a != exp_b
        and not obvious_override
    ):
        return False, f"expression bucket mismatch ({exp_a} vs {exp_b}) blocks redundancy edge"

    lit_a = features_a.lighting_bucket
    lit_b = features_b.lighting_bucket
    if (
        lit_a is not None
        and lit_b is not None
        and lit_a != "unknown"
        and lit_b != "unknown"
        and lit_a != lit_b
        and not obvious_override
    ):
        return False, f"lighting bucket mismatch ({lit_a} vs {lit_b}) blocks redundancy edge"

    return True, ""


# Issue #208 — fine pruning compatibility constants.
# ``_FINE_ADJACENT_DISTANCE_SCALE`` is the multiplier on
# ``representation_distance_threshold`` required for adjacent fine pose
# bands; non-adjacent fine bands ONLY pass when an obvious-duplicate
# override fires.
_FINE_ADJACENT_DISTANCE_SCALE: float = 0.7
# Expression bands tolerate a single-band gap by default — the underlying
# numeric features are noisy enough that a hard "same band only" rule
# would over-protect. Two-band gaps trip the gate unless the obvious-
# duplicate override fires.
_EXPRESSION_BAND_GAP_LIMIT: int = 1
# Each expression band that's checked: (RepresentationFeatures attr name,
# human-readable label used in the diagnostic reason string).
_FINE_EXPRESSION_BANDS: tuple[tuple[str, str], ...] = (
    ("mouth_openness_band", "mouth openness"),
    ("smile_proxy_band", "smile intensity"),
    ("eye_closure_band", "eye closure"),
    ("brow_raise_band", "brow raise"),
    ("expression_asymmetry_band", "expression asymmetry"),
)


def _is_visually_obvious(
    features_a: RepresentationFeatures,
    features_b: RepresentationFeatures,
) -> tuple[bool, str | None]:
    """Return ``(visually_obvious, mismatch_reason)`` for one pair.

    Issue #208 follow-up. The earlier shape allowed ANY fine
    pose / expression / appearance mismatch to be bypassed when the
    pair's continuous representation distance was below
    ``obvious_duplicate_threshold``. In practice that let visually
    distinct red-carpet looks with very similar continuous descriptors
    fall through the obvious-override path AND through the
    readiness-FAIL safety cap, undermining the whole fine gate.

    The fix: an "obvious duplicate" must ALSO be visually obvious.
    This helper computes that second predicate purely from the fine
    bands already cached on :class:`RepresentationFeatures`:

    * Same-or-adjacent fine yaw band (``gap <= 1``).
    * Same-or-adjacent fine pitch band (``gap <= 1``).
    * Each fine expression band gap within
      :data:`_EXPRESSION_BAND_GAP_LIMIT`.
    * Appearance mode is exactly equal OR differs by at most one bin in
      exactly one channel — single-dimension lighting drift across a
      bin boundary is tolerated, multi-channel drift is not.

    Returns the first concrete mismatch reason (or ``None`` when every
    fine signal is compatible) so the caller can surface a precise
    ``representation-obvious but visually distinct`` diagnostic.

    Missing fine signals do not count as a mismatch — a face we can't
    classify on a given band stays eligible for the override on that
    band so partial-data faces are not unfairly held back.
    """
    for attr, label in (
        ("yaw_micro_band", "yaw"),
        ("pitch_micro_band", "pitch"),
    ):
        va = getattr(features_a, attr)
        vb = getattr(features_b, attr)
        if va is None or vb is None or va == vb:
            continue
        if abs(va - vb) > 1:
            return False, f"fine {label} band gap ({va} vs {vb}, gap={abs(va - vb)})"

    for attr, label in _FINE_EXPRESSION_BANDS:
        va = getattr(features_a, attr)
        vb = getattr(features_b, attr)
        if va is None or vb is None:
            continue
        gap = abs(va - vb)
        if gap > _EXPRESSION_BAND_GAP_LIMIT:
            return False, f"fine {label} band gap ({va} vs {vb}, gap={gap})"

    mode_a = features_a.appearance_mode
    mode_b = features_b.appearance_mode
    if mode_a is not None and mode_b is not None and mode_a != mode_b:
        # Tolerate single-bin drift in exactly one channel; reject any
        # multi-channel drift OR a >1 bin gap in any channel.
        diffs = [abs(a - b) for a, b in zip(mode_a, mode_b, strict=False)]
        max_diff = max(diffs)
        non_zero = sum(1 for d in diffs if d > 0)
        if max_diff > 1 or non_zero > 1:
            return False, f"appearance mode mismatch ({mode_a} vs {mode_b})"

    return True, None


def _fine_pruning_compatibility(
    features_a: RepresentationFeatures,
    features_b: RepresentationFeatures,
    *,
    distance: float,
    temporal_confidence: float,
    config: AggressivenessConfig,
) -> tuple[bool, str]:
    """Return ``(compatible, reason)`` for the fine pruning constraints.

    Issue #208 adds three layers of fine pruning compatibility ON TOP of
    the coarse ``_bucket_compatibility`` gate so visually distinct
    training examples are not silently merged into pruning clusters:

    * **Fine pose bands** — same-band passes, adjacent bands require
      stricter distance (``threshold * _FINE_ADJACENT_DISTANCE_SCALE``),
      non-adjacent bands block unless the obvious-duplicate override
      fires.
    * **Fine expression bands** — same / adjacent band passes,
      gap >= 2 blocks unless obvious.
    * **Appearance mode** — different appearance-mode signatures block
      unless obvious so different hairstyles / makeup / event /
      lighting setups don't fold into one pruning cluster.

    Obvious duplicates may override the fine constraints ONLY when the
    pair is also visually obvious (issue #208 follow-up): same-or-
    adjacent fine pose AND fine expression AND a compatible appearance
    mode. A pair that is representation-obvious but visually distinct
    is rejected with an explicit
    ``representation-obvious but visually distinct`` reason so the
    downstream diagnostic surfaces the exact mismatch that produced the
    earlier "1610 prune candidates on a 1665-face red-carpet set"
    pathology.
    """
    representation_obvious = (
        distance <= config.obvious_duplicate_threshold
        and temporal_confidence >= _TEMPORAL_CONFIDENCE_FOR_OBVIOUS_EDGE
    )
    visually_obvious, visual_mismatch_reason = _is_visually_obvious(features_a, features_b)
    # Issue #208 follow-up: an "obvious duplicate" is only honoured when
    # the pair is ALSO visually compatible. A pair that is
    # representation-obvious but visually distinct is the exact
    # pathology that produced "1610 prune candidates on a 1665-face
    # red-carpet set" — surface the veto explicitly so the downstream
    # diagnostic explains why the override didn't fire.
    if representation_obvious and not visually_obvious:
        return False, (
            "representation-obvious but visually distinct — "
            f"{visual_mismatch_reason} overrides distance-only duplicate signal"
        )
    obvious_override = representation_obvious and visually_obvious
    strict_limit = config.representation_distance_threshold * _FINE_ADJACENT_DISTANCE_SCALE

    # Fine yaw / pitch micro-bands. These are SIGNED so adjacency is
    # meaningful (left -> centre -> right; up -> neutral -> down).
    for attr, label in (
        ("yaw_micro_band", "yaw"),
        ("pitch_micro_band", "pitch"),
    ):
        va = getattr(features_a, attr)
        vb = getattr(features_b, attr)
        if va is None or vb is None:
            continue
        if va == vb:
            continue
        gap = abs(va - vb)
        if gap == 1:
            if not (distance <= strict_limit or obvious_override):
                return False, (
                    f"adjacent fine {label} bands ({va} vs {vb}) require "
                    f"stricter distance ({distance:.2f} > {strict_limit:.2f})"
                )
        elif not obvious_override:
            return False, (
                f"fine {label} band mismatch ({va} vs {vb}, gap={gap}) blocks pruning edge"
            )

    # Fine expression bands. Two-band gap (or wider) blocks; one-band gap
    # is tolerated since the numeric expression features are noisy.
    for attr, label in _FINE_EXPRESSION_BANDS:
        va = getattr(features_a, attr)
        vb = getattr(features_b, attr)
        if va is None or vb is None:
            continue
        gap = abs(va - vb)
        if gap > _EXPRESSION_BAND_GAP_LIMIT and not obvious_override:
            return False, (
                f"fine {label} band mismatch ({va} vs {vb}, gap={gap}) blocks pruning edge"
            )

    # Appearance mode — discrete (luminance, warmth, saturation, contrast)
    # signature. A different signature means different hairstyle /
    # makeup / event / lighting setup — visually distinct, must NOT
    # collapse into one pruning cluster unless the pair is an obvious
    # duplicate.
    mode_a = features_a.appearance_mode
    mode_b = features_b.appearance_mode
    if mode_a is not None and mode_b is not None and mode_a != mode_b and not obvious_override:
        return False, (f"appearance mode mismatch ({mode_a} vs {mode_b}) blocks pruning edge")

    return True, ""


def can_create_redundancy_edge(
    *,
    features_a: RepresentationFeatures,
    features_b: RepresentationFeatures,
    distance: float,
    compared: int,
    temporal_confidence: float,
    config: AggressivenessConfig,
) -> tuple[bool, str]:
    """Return ``(eligible, reason)`` for one candidate redundancy edge.

    The ticket's contract is:

    * Enough representation metrics must be comparable.
    * Neither face is an identity reject / outlier (the edge collapses two
      faces into the same cluster, which would inherit the outlier into
      the cluster's representation; outlier handling stays in the
      review layer downstream).
    * Identity state is safe, or missing identity forces review-only
      behaviour — a redundancy edge SHOULD NOT be created across a
      missing-identity face because we can't verify it's the same
      subject.
    * Representation distance is under the selected threshold (either
      obvious-duplicate floor OR the redundancy threshold with
      temporal-confidence gate).
    * Obvious duplicates still allowed at lower temporal confidence so
      true duplicates are not rescued by frame-distance alone.

    A pair that fails any rule returns ``(False, reason)`` — the reason
    is surfaced in the post-cluster diagnostics so the user can audit
    which constraint blocked the merge.
    """
    if compared < _MIN_COMPARED_FOR_EDGE:
        return False, f"too few comparable metrics ({compared} < {_MIN_COMPARED_FOR_EDGE})"
    if features_a.identity_outlier or features_b.identity_outlier:
        return False, "identity outlier on at least one side — graph excludes outliers"

    obvious = distance <= config.obvious_duplicate_threshold
    high_temporal = temporal_confidence >= _TEMPORAL_CONFIDENCE_FOR_OBVIOUS_EDGE

    # Missing-identity gate (issue #199): a missing-identity pair MAY still
    # cluster when the match is an obvious, temporally-safe duplicate; the
    # post-cluster decision layer will route the entire cluster to review
    # via the existing ``missing identity embedding guardrail`` rule. For
    # any other case (redundant-but-not-obvious distance, distant temporal)
    # the edge is blocked outright so missing-identity faces cannot be
    # confidently merged with anyone.
    missing_identity = not features_a.has_identity or not features_b.has_identity
    if missing_identity and not (obvious and high_temporal):
        return False, (
            "missing identity guardrail on at least one side — only obvious "
            "high-confidence duplicates may cluster across missing identity"
        )

    # Issue #204: hard bucket-compatibility gate. Pose / pitch / expression
    # / lighting mismatches block normal redundancy edges; obvious
    # duplicates may override only when temporal confidence is high
    # (handled inside ``_bucket_compatibility``). Surfacing the bucket
    # reason here keeps the downstream diagnostic explicit.
    bucket_ok, bucket_reason = _bucket_compatibility(
        features_a,
        features_b,
        distance=distance,
        temporal_confidence=temporal_confidence,
        config=config,
    )
    if not bucket_ok:
        return False, bucket_reason

    # Issue #208: fine pruning compatibility. The coarse bucket gate
    # blocks frontal-vs-profile, smile-vs-neutral, etc.; the fine gate
    # blocks micro-yaw drift inside ``frontal``, mouth-shape variation
    # inside ``smile``, and different appearance modes (hairstyle /
    # makeup / event) inside the same coarse bucket. Together they
    # answer the stricter pruning question "would removing this face
    # lose a visually distinct training example?".
    fine_ok, fine_reason = _fine_pruning_compatibility(
        features_a,
        features_b,
        distance=distance,
        temporal_confidence=temporal_confidence,
        config=config,
    )
    if not fine_ok:
        return False, fine_reason

    if obvious:
        if temporal_confidence < _TEMPORAL_CONFIDENCE_FOR_OBVIOUS_EDGE:
            return False, (
                f"obvious-distance match but temporal_confidence "
                f"{temporal_confidence:.2f} < {_TEMPORAL_CONFIDENCE_FOR_OBVIOUS_EDGE:.2f}"
            )
        if missing_identity:
            return True, (
                f"obvious duplicate (distance={distance:.2f}) — clustered "
                "across missing-identity for downstream review"
            )
        return True, f"obvious duplicate (distance={distance:.2f})"
    if distance > config.representation_distance_threshold:
        return False, (
            f"distance {distance:.2f} > threshold {config.representation_distance_threshold:.2f}"
        )
    if temporal_confidence < _TEMPORAL_CONFIDENCE_FOR_REDUNDANT_EDGE:
        return False, (
            f"redundant-distance match but temporal_confidence "
            f"{temporal_confidence:.2f} < {_TEMPORAL_CONFIDENCE_FOR_REDUNDANT_EDGE:.2f}"
        )
    return True, (
        f"redundant + safe temporal (distance={distance:.2f}, "
        f"temporal_confidence={temporal_confidence:.2f})"
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


# Issue #204 component-diameter rules. A cluster is "too sprawling" when
# any pairwise distance inside it exceeds
# ``representation_distance_threshold * _COMPONENT_DIAMETER_SCALE``; a
# member is treated as "directly compatible with the representative" when
# its distance to the rep is under
# ``representation_distance_threshold * _MEMBER_TO_REP_SCALE``. Outside
# that band the member is split into its own sub-component.
_COMPONENT_DIAMETER_SCALE: float = 1.5
_MEMBER_TO_REP_SCALE: float = 1.25
# Only run the diameter pass on components above this size — pairs and
# triples can't form chains long enough to exceed the diameter band.
_COMPONENT_SPLIT_MIN_SIZE: int = 4
# Issue #204 follow-up. The diameter pass guarantees sub-cluster
# diameter <= diameter_limit, but for very large connected components
# (the giant-component pathology that motivated the ticket in the first
# place) even the on-demand pairwise checks add a quadratic pass over
# the worst component. Above this size we KEEP the component intact and
# log the bypass on each member via ``split_reason`` so the giant-
# component warning + diagnostics still flag the topology issue without
# burning minutes of CPU on a worst-case split.
_MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT: int = 2000


@dataclass(frozen=True)
class _SplitDiagnostics:
    """Per-member diagnostics produced by ``_split_high_diameter_components``."""

    diameter_by_member: dict[int, float]
    nearest_neighbor_by_member: dict[int, float]
    nearest_neighbor_index_by_member: dict[int, int | None]
    direct_to_rep_by_member: dict[int, bool]
    connected_via_rep_edge_by_member: dict[int, bool]
    split_reason_by_member: dict[int, str | None]


def _split_high_diameter_components(
    components: list[list[int]],
    features: list[RepresentationFeatures],
    ctx: _PairwiseContext,
    config: AggressivenessConfig,
    edge_reasons: dict[tuple[int, int], tuple[bool, str]],
) -> tuple[list[list[int]], _SplitDiagnostics]:
    """Split components whose internal spread exceeds the diameter limit.

    Returns ``(new_components, _SplitDiagnostics)``. The diagnostics
    expose enough per-member state to distinguish "redundant with
    representative" from "reachable only through a neighbour chain" and
    surface WHY the splitter touched (or didn't touch) a given
    component (``split_reason``):

    * ``diameter_by_member`` — max pairwise distance within the
      post-split component the member ended up in.
    * ``nearest_neighbor_by_member`` — distance to the closest other
      member within the post-split component (0.0 for singletons).
    * ``nearest_neighbor_index_by_member`` — face-list index of that
      neighbour (``None`` for singletons).
    * ``direct_to_rep_by_member`` — whether the member's pairwise
      distance to the chosen representative is within
      :data:`_MEMBER_TO_REP_SCALE` of the threshold (distance-based).
    * ``connected_via_rep_edge_by_member`` — whether the member had an
      ELIGIBLE redundancy edge to the representative in the original
      graph (graph-based; ``False`` flags transitive-chain membership).
    * ``split_reason_by_member`` — None when no split was needed,
      ``"split from oversize component due to diameter"`` when the
      diameter pass relocated the member, and
      ``"component too large for diameter check ({size} > {cap}) — kept as-is"``
      when the size cap fired.

    Splitter contract (issue #204 follow-up): every emitted sub-cluster
    is GUARANTEED to have diameter <= ``diameter_limit`` UNLESS the
    parent component exceeded
    :data:`_MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT`. The attachment loop
    uses the stricter "distance to seed <= rep_limit AND distance to
    every already-attached member <= diameter_limit" rule so a member
    can't pull two attached members across the diameter envelope.
    """
    diameter_by_member: dict[int, float] = {}
    nearest_neighbor_by_member: dict[int, float] = {}
    nearest_neighbor_index_by_member: dict[int, int | None] = {}
    direct_to_rep_by_member: dict[int, bool] = {}
    connected_via_rep_edge_by_member: dict[int, bool] = {}
    split_reason_by_member: dict[int, str | None] = {}
    new_components: list[list[int]] = []

    diameter_limit = config.representation_distance_threshold * _COMPONENT_DIAMETER_SCALE
    rep_limit = config.representation_distance_threshold * _MEMBER_TO_REP_SCALE

    def _distance(i: int, j: int) -> float:
        """On-demand pairwise distance using the shared context."""
        dist, _compared = _distance_with_ctx(ctx, i, j)
        return dist

    def _component_diameter(members: list[int]) -> float:
        """Compute the diameter of one component by scanning each pair once."""
        if len(members) <= 1:
            return 0.0
        worst = 0.0
        for i_pos, i in enumerate(members):
            for j in members[i_pos + 1 :]:
                d = _distance(i, j)
                if d > worst:
                    worst = d
        return worst

    def _has_eligible_rep_edge(member: int, rep: int) -> bool:
        """Was there an ELIGIBLE edge between ``member`` and ``rep`` in the
        original graph? Reads from ``edge_reasons`` keyed off (min,max)."""
        if member == rep:
            return True
        key = (member, rep) if member < rep else (rep, member)
        entry = edge_reasons.get(key)
        return bool(entry and entry[0])

    def _annotate(members: list[int], reason: str | None) -> None:
        """Choose a representative + populate every member's diagnostics.

        Pairwise distances are computed on demand so we never materialise
        the full N*(N-1)/2 matrix.
        """
        rep = max(members, key=lambda idx: (features[idx].quality_score, -idx))
        if len(members) == 1:
            sole = members[0]
            diameter_by_member[sole] = 0.0
            nearest_neighbor_by_member[sole] = 0.0
            nearest_neighbor_index_by_member[sole] = None
            direct_to_rep_by_member[sole] = True
            connected_via_rep_edge_by_member[sole] = True
            split_reason_by_member[sole] = reason
            return

        # Per-pair distances are reused for: diameter, per-member nearest
        # neighbour, per-member distance-to-rep. Materialising them ONCE
        # in a (member -> {neighbour -> distance}) map keeps the inner
        # loop O(N^2) but reuses each pair instead of recomputing it
        # three times.
        pair_distance: dict[int, dict[int, float]] = {m: {} for m in members}
        for i_pos, i in enumerate(members):
            for j in members[i_pos + 1 :]:
                d = _distance(i, j)
                pair_distance[i][j] = d
                pair_distance[j][i] = d

        diameter = (
            max((max(neighbours.values()) for neighbours in pair_distance.values()), default=0.0)
            if len(members) > 1
            else 0.0
        )
        for member in members:
            diameter_by_member[member] = diameter
            split_reason_by_member[member] = reason
            neighbours = pair_distance[member]
            if not neighbours:
                nearest_neighbor_by_member[member] = math.inf
                nearest_neighbor_index_by_member[member] = None
            else:
                nearest_idx = min(neighbours, key=neighbours.__getitem__)
                nearest_neighbor_by_member[member] = neighbours[nearest_idx]
                nearest_neighbor_index_by_member[member] = nearest_idx
            if member == rep:
                direct_to_rep_by_member[member] = True
                connected_via_rep_edge_by_member[member] = True
                continue
            dist_to_rep = pair_distance[member].get(rep, math.inf)
            direct_to_rep_by_member[member] = dist_to_rep <= rep_limit
            connected_via_rep_edge_by_member[member] = _has_eligible_rep_edge(member, rep)

    def _annotate_oversize(members: list[int], reason: str | None) -> None:
        """Light-weight annotation for components above
        :data:`_MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT`.

        Issue #204 follow-up. ``_annotate`` builds a full per-pair
        distance dict (O(N^2) time AND memory); ``_annotate_oversize``
        only computes the O(N) per-member distance to a single chosen
        representative so the cap branch genuinely avoids the
        giant-component pathology. The cost of the missing data:

        * ``diameter_by_member`` is left ``inf`` — the splitter chose
          to bypass the diameter pass, so a definitive diameter would
          require the same O(N^2) work the bypass exists to avoid.
        * ``nearest_neighbor_by_member`` / index are left ``inf`` /
          ``None`` for the same reason.
        * ``direct_to_rep_by_member`` is the per-member distance check
          to the chosen representative (cheap, O(N) total).
        * ``connected_via_rep_edge_by_member`` is the cheap
          ``edge_reasons`` lookup the regular path uses.
        * ``split_reason_by_member`` carries the cap-bypass reason so
          the report surfaces the topology issue.

        ``compute_redundancy``'s downstream consumers treat ``inf`` /
        ``None`` diagnostics as "missing" and fall back to safer
        recommendations, so the lightweight annotation never produces
        a more aggressive prune decision than the regular path would.
        """
        rep = max(members, key=lambda idx: (features[idx].quality_score, -idx))
        for member in members:
            diameter_by_member[member] = math.inf
            nearest_neighbor_by_member[member] = math.inf
            nearest_neighbor_index_by_member[member] = None
            split_reason_by_member[member] = reason
            if member == rep:
                direct_to_rep_by_member[member] = True
                connected_via_rep_edge_by_member[member] = True
                continue
            dist_to_rep = _distance(member, rep)
            direct_to_rep_by_member[member] = dist_to_rep <= rep_limit
            connected_via_rep_edge_by_member[member] = _has_eligible_rep_edge(member, rep)

    for members in components:
        # Small components can't form chains long enough to exceed the
        # diameter band; annotate directly without the splitter logic.
        if len(members) < _COMPONENT_SPLIT_MIN_SIZE:
            _annotate(members, reason=None)
            new_components.append(members)
            continue

        # Issue #204 follow-up — cap the work on truly huge components.
        # ``_annotate`` would build a full N*(N-1)/2 pair_distance dict
        # (~43M pairs / ~3GB on a 9,304-face giant component); above this
        # cap we KEEP the component intact AND use the O(N) light-weight
        # annotation path so the giant-component pathology cannot crash
        # the process. Cluster-health metrics + ``split_reason`` still
        # surface the topology issue so the user can re-tune
        # aggressiveness.
        if len(members) > _MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT:
            reason = (
                f"component too large for diameter check ({len(members)} > "
                f"{_MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT}) — kept as-is"
            )
            _annotate_oversize(members, reason=reason)
            new_components.append(members)
            continue

        # Check the cheap diameter first — most components are small
        # enough that the full pairwise scan inside ``_component_diameter``
        # is the same as what ``_annotate`` would do anyway, so when the
        # check passes we hand off to ``_annotate`` for the diagnostics.
        diameter = _component_diameter(members)
        if diameter <= diameter_limit:
            _annotate(members, reason=None)
            new_components.append(members)
            continue

        # Strict greedy split (issue #204 follow-up). For each sub-cluster:
        #   * seed = highest-quality remaining member.
        #   * attach a candidate ONLY when distance(candidate, seed) <=
        #     rep_limit AND distance(candidate, every already-attached
        #     member) <= diameter_limit.
        # The conjunction guarantees the sub-cluster's diameter never
        # exceeds ``diameter_limit`` regardless of how generous
        # ``rep_limit`` is.
        remaining: set[int] = set(members)
        while remaining:
            seed = max(
                remaining,
                key=lambda idx: (features[idx].quality_score, -idx),
            )
            sub = [seed]
            remaining.discard(seed)
            # Distances within the current sub: ``sub_distances[idx]`` is the
            # cached distance from ``idx`` to the seed for the rep_limit check.
            for candidate in sorted(remaining):
                seed_distance = _distance(seed, candidate)
                if seed_distance > rep_limit:
                    continue
                # Stricter rule: must also stay within diameter_limit of
                # EVERY already-attached member, not just the seed.
                exceeds_diameter = False
                for attached in sub[1:]:
                    if _distance(attached, candidate) > diameter_limit:
                        exceeds_diameter = True
                        break
                if exceeds_diameter:
                    continue
                sub.append(candidate)
            for attached in sub[1:]:
                remaining.discard(attached)
            sub.sort()
            _annotate(
                sub,
                reason="split from oversize component due to diameter",
            )
            new_components.append(sub)

    new_components.sort(key=lambda c: c[0])
    return (
        new_components,
        _SplitDiagnostics(
            diameter_by_member=diameter_by_member,
            nearest_neighbor_by_member=nearest_neighbor_by_member,
            nearest_neighbor_index_by_member=nearest_neighbor_index_by_member,
            direct_to_rep_by_member=direct_to_rep_by_member,
            connected_via_rep_edge_by_member=connected_via_rep_edge_by_member,
            split_reason_by_member=split_reason_by_member,
        ),
    )


def _build_redundancy_clusters(
    features: list[RepresentationFeatures],
    config: AggressivenessConfig,
    *,
    progress_callback: T.Callable[[int], None] | None = None,
) -> tuple[
    list[list[int]],
    dict[tuple[int, int], tuple[float, int, float, int | None]],
    dict[tuple[int, int], tuple[bool, str]],
    _PairwiseContext,
]:
    """Union-find clusters by redundancy edges plus per-pair edge data.

    ``progress_callback`` (when supplied) receives ``1`` after each pairwise
    comparison so the dispatcher can drive a determinate ``tqdm`` bar over
    the ``n * (n - 1) // 2`` comparisons performed here.
    """
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

    # Precompute per-record arrays ONCE so the O(N^2) clustering loop below
    # consumes them via array indexing instead of repeated ``getattr`` +
    # ``_provenance_scale`` work per pair (issue #192 P2 high).
    ctx = _build_pairwise_context(features)
    frame_indices = [feat.frame_index for feat in features]

    # Issue #199: eligible edges carry per-pair distance/compared/temporal
    # data PLUS the eligibility reason so the post-cluster decision layer can
    # surface why a candidate was folded into the graph.
    #
    # Issue #204 follow-up: do NOT materialise diagnostics for every rejected
    # pair. Large facesets still require the unavoidable O(N^2) comparison
    # pass to discover eligible redundancy edges, but keeping only eligible
    # edges prevents the worst-case O(N^2) memory blow-up before the later
    # oversize-component cap can run.
    edges: dict[tuple[int, int], tuple[float, int, float, int | None]] = {}
    edge_reasons: dict[tuple[int, int], tuple[bool, str]] = {}
    for i in range(n):
        fi = frame_indices[i]
        for j in range(i + 1, n):
            distance, compared = _distance_with_ctx(ctx, i, j)
            t_conf, frame_distance = _temporal_confidence(fi, frame_indices[j], config)
            eligible, reason = can_create_redundancy_edge(
                features_a=features[i],
                features_b=features[j],
                distance=distance,
                compared=compared,
                temporal_confidence=t_conf,
                config=config,
            )
            if eligible:
                edges[(i, j)] = (distance, compared, t_conf, frame_distance)
                edge_reasons[(i, j)] = (True, reason)
                union(i, j)
            if progress_callback is not None:
                progress_callback(1)
    clusters_by_root: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters_by_root.setdefault(root, []).append(idx)
    components = sorted(clusters_by_root.values(), key=lambda c: c[0])
    return components, edges, edge_reasons, ctx


# ---------------------------------------------------------------------------
# Effective coverage + decisions
# ---------------------------------------------------------------------------


def _bucket_of(record: FaceQARecord, dimension: str) -> str:
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
    from collections import Counter as _Counter

    result: dict[str, EffectiveCoverageDimension] = {}
    for dimension in EFFECTIVE_COVERAGE_DIMENSIONS:
        raw: _Counter[str] = _Counter()
        cluster_buckets: dict[str, set[int]] = {}
        for idx, record in enumerate(records):
            bucket = _bucket_of(record, dimension)
            raw[bucket] += 1
            cluster_buckets.setdefault(bucket, set()).add(cluster_of[idx])
        effective = {bucket: len(clusters) for bucket, clusters in cluster_buckets.items()}
        ratios = {
            bucket: round(raw[bucket] / max(effective.get(bucket, 1), 1), 4) for bucket in raw
        }
        result[dimension] = EffectiveCoverageDimension(
            dimension=dimension,
            raw_counts=dict(raw),
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
    """Return a review reason if a representative cannot safely default to keep.

    A representative that lacks the identity guardrail cannot safely default
    to KEEP — we have no way to verify it's the same subject as the other
    cluster members (issue #199 follow-up). The whole cluster falls into
    review via this rule combined with the member-side check below.
    """
    if features.identity_outlier:
        return "identity outlier — review even when selected as representative"
    if not features.has_identity:
        return "missing identity embedding guardrail — review even when representative"
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
    rep_features: RepresentationFeatures | None = None,
) -> tuple[str, str, int]:
    """Return ``(recommendation, reason, budget_after)`` for one cluster member.

    Order of decisions:
    - missing metrics → review
    - identity outlier → review
    - **missing identity embedding (member OR representative)** → review
      (guardrail: without identity verification we cannot confidently keep
      or prune *any* member, even when it looks like an alternate or sits
      in a protected bucket). ``rep_features`` is optional for backward
      compatibility with existing tests; when supplied, the
      missing-identity check covers BOTH sides of the edge so the
      "whole cluster to review" intent from issue #199 actually holds.
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
    if not member_features.has_identity or (
        rep_features is not None and not rep_features.has_identity
    ):
        # Identity is the cross-subject guardrail. Without it on EITHER
        # side of the edge we cannot confidently keep an alternate, spend
        # protection budget on it, or prune it as redundant — every
        # outcome routes to review so the whole cluster lands in the
        # review bucket together (issue #199 follow-up).
        side = "member" if not member_features.has_identity else "representative"
        return (
            REVIEW,
            f"missing identity embedding guardrail on {side} — review before keep or prune",
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


# ---------------------------------------------------------------------------
# Issue #208 follow-up 2 — post-cluster visual-span splitter.
#
# Pairwise gates decide whether two faces MAY connect; visual-span
# splitting decides whether the resulting connected component is still
# coherent as a single pruning cluster. Without this second pass a
# large red-carpet faceset can keep one technically-connected component
# whose internal yaw / expression / appearance bands sweep across many
# visually-distinct modes — the same kind of pathology the distance-
# diameter splitter fixes for numeric distance, but for fine visual
# modes.
# ---------------------------------------------------------------------------

# Only run the visual-span pass on components above this size. Small
# clusters cannot accumulate enough visual span to require splitting and
# the pass is O(N) per cluster — cheap but not free.
_VISUAL_SPAN_SPLIT_MIN_SIZE: int = 50

# Per-band span limits (max-min across cluster members). A cluster
# exceeding ANY of these gets split.
_VISUAL_SPAN_LIMITS: dict[str, int] = {
    "yaw_micro_band": 2,
    "pitch_micro_band": 2,
    "mouth_openness_band": 1,
    "smile_proxy_band": 1,
}
# Appearance-mode drift is computed channel-wise via the same
# "exact OR one bin in one channel" rule as the obvious-override
# tightening; if the cluster contains TWO appearance signatures that
# violate that rule, the cluster is also split.

# Visual-mode key used to group members during the split. Adjacent
# bins collapse via ``// 2`` so neighbouring micro-bands merge into a
# single key bucket — keeps the splitter from over-fragmenting.
_VISUAL_MODE_KEY_ATTRS: tuple[str, ...] = (
    "yaw_micro_band",
    "pitch_micro_band",
    "mouth_openness_band",
    "smile_proxy_band",
)


def _visual_mode_key(features: RepresentationFeatures) -> tuple[T.Any, ...]:
    """Return the discrete visual-mode key for one face.

    Adjacent micro-bins are folded together (``band // 2`` for signed
    bands, ``// 2`` for the unsigned bands too) so the key groups
    visually-similar faces into the SAME bucket instead of fragmenting
    on every per-band boundary. Appearance mode is included verbatim
    since it is already a small discrete signature.

    ``None`` band values become ``None`` in the key so partial-data
    faces cluster together rather than scattering across all possible
    bins.
    """
    parts: list[T.Any] = []
    for attr in _VISUAL_MODE_KEY_ATTRS:
        value = getattr(features, attr)
        if value is None:
            parts.append(None)
        else:
            # Signed bands ``//`` in Python rounds toward -inf — that's
            # fine here, the goal is "neighbouring bins share a bucket".
            parts.append(value // 2)
    parts.append(features.appearance_mode)
    return tuple(parts)


def _appearance_modes_compatible(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> bool:
    """Return ``True`` when two appearance modes are equal OR drift by
    exactly one bin in exactly one channel.

    Matches the obvious-override "visually obvious" rule so the
    visual-span splitter and the obvious-override gate agree on what
    counts as a compatible appearance.
    """
    if a is None or b is None or a == b:
        return True
    diffs = [abs(x - y) for x, y in zip(a, b, strict=False)]
    return max(diffs) <= 1 and sum(1 for d in diffs if d > 0) <= 1


def _visual_span_exceeds_limits(
    members: list[int],
    features: list[RepresentationFeatures],
) -> str | None:
    """Return the first concrete span violation, or ``None`` when the
    cluster's visual span fits within :data:`_VISUAL_SPAN_LIMITS`."""
    for attr, limit in _VISUAL_SPAN_LIMITS.items():
        values = [getattr(features[m], attr) for m in members]
        present = [v for v in values if v is not None]
        if len(present) < 2:
            continue
        span = max(present) - min(present)
        if span > limit:
            return f"{attr} span={span} > {limit}"

    # Appearance-mode diversity: if ANY pair in the cluster is not
    # appearance-compatible, the span is too broad. O(N^2) is fine
    # because we only get here when the per-band span check did not
    # already fire — by then the cluster is small enough that even
    # an O(N^2) scan over its appearance signatures is cheap relative
    # to the rest of the pipeline. Short-circuit on the first mismatch.
    modes = {
        features[m].appearance_mode for m in members if features[m].appearance_mode is not None
    }
    if len(modes) > 1:
        mode_list = sorted(modes)
        for i_pos, a in enumerate(mode_list):
            for b in mode_list[i_pos + 1 :]:
                if not _appearance_modes_compatible(a, b):
                    return f"appearance modes {a} vs {b} drift > 1 bin"
    return None


def _split_high_visual_span_components(
    components: list[list[int]],
    features: list[RepresentationFeatures],
    split_reason_by_member: dict[int, str | None],
) -> tuple[list[list[int]], list[list[int]]]:
    """Split components whose internal visual-mode span is too broad.

    Walks ``components`` AFTER the distance-diameter splitter, leaves
    small components alone, and for each large component:

    1. Computes the per-band span + appearance diversity via
       :func:`_visual_span_exceeds_limits`.
    2. If a limit is exceeded, partitions members by
       :func:`_visual_mode_key` — adjacent bins fold via ``// 2`` so
       neighbouring micro-bands stay together but distinct visual
       modes split apart.
    3. Records the split reason on every relocated member so the
       per-record diagnostic surfaces the visual-mode rationale.

    Components that don't violate the limits pass through unchanged.
    """
    out: list[list[int]] = []
    refreshed_subs: list[list[int]] = []  # Sub-components needing diagnostic refresh.
    for members in components:
        if len(members) < _VISUAL_SPAN_SPLIT_MIN_SIZE:
            out.append(members)
            continue
        violation = _visual_span_exceeds_limits(members, features)
        if violation is None:
            out.append(members)
            continue

        groups: dict[tuple[T.Any, ...], list[int]] = {}
        for member in members:
            key = _visual_mode_key(features[member])
            groups.setdefault(key, []).append(member)

        # Single-bucket grouping means the visual-mode key collapses
        # everything together — we'd produce the same component, so
        # there is nothing to split. Leave the cluster intact (the
        # downstream diagnostics still surface the topology issue via
        # cluster-health metrics) but record the bypass on every member
        # so reviewers can audit. No diagnostic refresh needed since
        # the component membership did not change.
        if len(groups) <= 1:
            for member in members:
                if split_reason_by_member.get(member) is None:
                    split_reason_by_member[member] = (
                        f"visual-span {violation} but mode-key collapsed to one bucket "
                        "— left intact"
                    )
            out.append(members)
            continue

        reason = f"split from oversize component due to visual-mode span ({violation})"
        for sub in groups.values():
            sub.sort()
            for member in sub:
                split_reason_by_member[member] = reason
            out.append(sub)
            refreshed_subs.append(sub)
    out.sort(key=lambda c: c[0])
    return out, refreshed_subs


def _refresh_component_diagnostics(
    members: list[int],
    features: list[RepresentationFeatures],
    ctx: _PairwiseContext,
    config: AggressivenessConfig,
    edge_reasons: dict[tuple[int, int], tuple[bool, str]],
    *,
    diameter_by_member: dict[int, float],
    nearest_neighbor_by_member: dict[int, float],
    nearest_neighbor_index_by_member: dict[int, int | None],
    direct_to_rep_by_member: dict[int, bool],
    connected_via_rep_edge_by_member: dict[int, bool],
) -> None:
    """Refresh per-member diagnostics for one FINAL component.

    Issue #208 follow-up 3. The visual-span splitter runs after the
    diameter splitter and can re-group a parent component into several
    smaller visual-mode sub-components. When that happens, diagnostics
    created by the diameter pass (nearest neighbour, component diameter,
    direct-to-representative, and graph-edge-to-representative) describe
    the PRE-visual-split parent, not the final cluster rendered in the
    report. Recompute those fields for each visual sub-component while
    preserving the existing ``split_reason_by_member`` entries populated
    by the splitters.
    """
    rep_limit = config.representation_distance_threshold * _MEMBER_TO_REP_SCALE

    def _distance(i: int, j: int) -> float:
        dist, _compared = _distance_with_ctx(ctx, i, j)
        return dist

    def _has_eligible_rep_edge(member: int, rep: int) -> bool:
        if member == rep:
            return True
        key = (member, rep) if member < rep else (rep, member)
        entry = edge_reasons.get(key)
        return bool(entry and entry[0])

    rep = max(members, key=lambda idx: (features[idx].quality_score, -idx))

    if len(members) == 1:
        sole = members[0]
        diameter_by_member[sole] = 0.0
        nearest_neighbor_by_member[sole] = 0.0
        nearest_neighbor_index_by_member[sole] = None
        direct_to_rep_by_member[sole] = True
        connected_via_rep_edge_by_member[sole] = True
        return

    # Keep the final-diagnostic refresh bounded. Visual splitting usually
    # produces much smaller groups; if a group somehow remains huge, use
    # the same O(N) representative-only fallback as the diameter cap.
    if len(members) > _MAX_COMPONENT_SIZE_FOR_DIAMETER_SPLIT:
        for member in members:
            diameter_by_member[member] = math.inf
            nearest_neighbor_by_member[member] = math.inf
            nearest_neighbor_index_by_member[member] = None
            if member == rep:
                direct_to_rep_by_member[member] = True
                connected_via_rep_edge_by_member[member] = True
                continue
            dist_to_rep = _distance(member, rep)
            direct_to_rep_by_member[member] = dist_to_rep <= rep_limit
            connected_via_rep_edge_by_member[member] = _has_eligible_rep_edge(member, rep)
        return

    pair_distance: dict[int, dict[int, float]] = {m: {} for m in members}
    for i_pos, i in enumerate(members):
        for j in members[i_pos + 1 :]:
            d = _distance(i, j)
            pair_distance[i][j] = d
            pair_distance[j][i] = d

    diameter = max(
        (max(neighbours.values()) for neighbours in pair_distance.values()), default=0.0
    )
    for member in members:
        diameter_by_member[member] = diameter
        neighbours = pair_distance[member]
        if not neighbours:
            nearest_neighbor_by_member[member] = math.inf
            nearest_neighbor_index_by_member[member] = None
        else:
            nearest_idx = min(neighbours, key=neighbours.__getitem__)
            nearest_neighbor_by_member[member] = neighbours[nearest_idx]
            nearest_neighbor_index_by_member[member] = nearest_idx

        if member == rep:
            direct_to_rep_by_member[member] = True
            connected_via_rep_edge_by_member[member] = True
            continue
        dist_to_rep = pair_distance[member].get(rep, math.inf)
        direct_to_rep_by_member[member] = dist_to_rep <= rep_limit
        connected_via_rep_edge_by_member[member] = _has_eligible_rep_edge(member, rep)


def _finite_or_none(value: float | None) -> float | None:
    """Return ``value`` when it is a finite float; otherwise ``None``.

    Used by ``compute_redundancy`` when projecting the splitter's
    diameter / nearest-neighbour diagnostics onto ``RedundancyRecord``:
    the oversize-component cap path (issue #204 follow-up) leaves these
    as ``math.inf`` to signal "uncomputed", but ``math.inf`` does not
    round-trip through standard JSON, so the report-level fields are
    surfaced as ``None`` instead.
    """
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


_GIANT_COMPONENT_RATIO_LIMIT: float = 0.25
_GIANT_COMPONENT_SIZE_LIMIT: int = 1000
_SPARSE_MULTI_CLUSTER_LIMIT: int = 2
_LARGE_FACESET_THRESHOLD: int = 1000


def _percentile(values: list[float], pct: float) -> float | None:
    """Plain numpy-free percentile so the helper stays usable on tiny lists."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


# Issue #208 — readiness FAIL safety cap parameters. The cap fires when:
#   1. ``coverage`` is supplied AND
#   2. ``ReadinessScores.overall_readiness_score`` is below the FAIL band
#      (``_READINESS_FAIL_SCORE``) AND
#   3. The proposed prune ratio exceeds ``_READINESS_FAIL_PRUNE_RATIO``.
# Under those conditions, borderline prune candidates are demoted to
# review so an under-developed faceset doesn't get gutted before the
# user reviews the topology.
_READINESS_FAIL_SCORE: float = 50.0
_READINESS_FAIL_PRUNE_RATIO: float = 0.80
# Reasons that match "obvious duplicate" survive the demotion — the cap
# protects visually distinct samples, not true near-duplicates.
_OBVIOUS_DUPLICATE_REASON_SUBSTRINGS: tuple[str, ...] = ("obvious duplicate",)


def _apply_readiness_safety_cap(
    output_records: list[RedundancyRecord],
    coverage: FacesetCoverageReport | None,
    total_faces: int,
) -> tuple[bool, str | None, int]:
    """Demote borderline ``prune_candidate`` rows to ``review`` when the
    safety cap conditions fire.

    Returns ``(applied, reason, demoted_count)`` — ``applied`` is True
    when at least one demotion was made; ``reason`` carries the human-
    readable cap explanation for the report header.
    """
    if coverage is None or total_faces == 0:
        return False, None, 0

    # Local import to avoid the circular ``coverage -> redundancy``
    # dependency that would arise if scoring were imported at module
    # top-level.
    from lib.faceqa.scoring import compute_readiness_scores

    try:
        scores = compute_readiness_scores(coverage)
    except Exception:  # pylint: disable=broad-except
        logger.debug("Readiness safety cap: scoring failed; cap not applied.")
        return False, None, 0

    overall = scores.overall_readiness_score
    if overall >= _READINESS_FAIL_SCORE:
        return False, None, 0

    proposed_prune = sum(1 for r in output_records if r.recommendation == PRUNE)
    prune_ratio = proposed_prune / total_faces
    if prune_ratio <= _READINESS_FAIL_PRUNE_RATIO:
        return False, None, 0

    demoted = 0
    for record in output_records:
        if record.recommendation != PRUNE:
            continue
        eligibility = (record.edge_eligibility or "").lower()
        if any(substr in eligibility for substr in _OBVIOUS_DUPLICATE_REASON_SUBSTRINGS):
            # Obvious duplicates stay prune — the cap is about
            # protecting VISUALLY DISTINCT samples on an
            # under-developed faceset.
            continue
        record.recommendation = REVIEW
        cap_note = (
            f"readiness FAIL safety cap: demoted to review "
            f"(readiness={overall:.1f}, prune ratio "
            f"{prune_ratio:.0%} > {_READINESS_FAIL_PRUNE_RATIO:.0%})"
        )
        record.reason = f"{record.reason}; {cap_note}" if record.reason else cap_note
        demoted += 1

    if demoted == 0:
        return False, None, 0

    reason = (
        f"readiness FAIL (overall={overall:.1f} < {_READINESS_FAIL_SCORE:.0f}) "
        f"AND proposed prune ratio {prune_ratio:.0%} > "
        f"{_READINESS_FAIL_PRUNE_RATIO:.0%}; demoted {demoted} borderline "
        f"prune candidate(s) to review."
    )
    logger.info("Redundancy %s", reason)
    return True, reason, demoted


def _compute_cluster_health(
    components: list[list[int]],
    total_faces: int,
    multi_face_clusters: int,
    diameter_by_member: dict[int, float],
    nearest_neighbor_by_member: dict[int, float],
) -> tuple[int, float, bool, float | None, float | None]:
    """Return cluster-health metrics for the report header.

    Returns ``(largest_cluster_size, largest_cluster_ratio,
    giant_component_warning, component_spread_p95,
    component_diameter_max)``. Issue #204: a giant component is a sign
    that the topology over-merged and the prune policy should be
    audited even when the per-face decisions look conservative.
    """
    if not components or total_faces == 0:
        return 0, 0.0, False, None, None

    largest = max(len(c) for c in components)
    ratio = largest / total_faces
    sparse_clusters_on_big_faceset = (
        multi_face_clusters <= _SPARSE_MULTI_CLUSTER_LIMIT
        and total_faces > _LARGE_FACESET_THRESHOLD
    )
    warning = (
        ratio > _GIANT_COMPONENT_RATIO_LIMIT
        or largest > _GIANT_COMPONENT_SIZE_LIMIT
        or sparse_clusters_on_big_faceset
    )
    # Spread is computed over non-singleton clusters so singletons (which
    # contribute 0.0 nearest-neighbour distance trivially) don't drag the
    # p95 down.
    non_singleton_spreads = [
        nearest_neighbor_by_member[member]
        for component in components
        if len(component) > 1
        for member in component
        if member in nearest_neighbor_by_member
        and math.isfinite(nearest_neighbor_by_member[member])
    ]
    spread_p95 = _percentile(non_singleton_spreads, 95.0)
    diameter_values = [
        diameter_by_member[member]
        for component in components
        for member in component
        if member in diameter_by_member and math.isfinite(diameter_by_member[member])
    ]
    diameter_max = max(diameter_values, default=None)
    return largest, round(ratio, 4), warning, spread_p95, diameter_max


def compute_redundancy(
    records: T.Sequence[FaceQARecord],
    *,
    aggressiveness: str = "balanced",
    coverage: FacesetCoverageReport | None = None,
    min_bucket_pct: float = 5.0,
    progress_callback: T.Callable[[int], None] | None = None,
) -> RedundancyReport:
    """Compute coverage-aware representation redundancy clusters and recs.

    ``progress_callback`` is forwarded into the pairwise-clustering loop so
    the dispatcher can drive a determinate ``tqdm`` bar; the total tick count
    matches ``n * (n - 1) // 2`` pairwise comparisons.
    """
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
    components, edges, edge_reasons, ctx = _build_redundancy_clusters(
        features, config, progress_callback=progress_callback
    )
    # Issue #204: split any component whose internal spread exceeds the
    # diameter limit so transitive chains across bucket boundaries can't
    # masquerade as one giant cluster. The splitter populates a
    # ``_SplitDiagnostics`` record so per-member diagnostics
    # (nearest neighbour, direct-to-rep, graph-edge to rep, split reason)
    # flow into each ``RedundancyRecord`` below.
    components, split_diag = _split_high_diameter_components(
        components, features, ctx, config, edge_reasons
    )
    diameter_by_member = split_diag.diameter_by_member
    nearest_neighbor_by_member = split_diag.nearest_neighbor_by_member
    nearest_neighbor_index_by_member = split_diag.nearest_neighbor_index_by_member
    direct_to_rep_by_member = split_diag.direct_to_rep_by_member
    connected_via_rep_edge_by_member = split_diag.connected_via_rep_edge_by_member
    split_reason_by_member = split_diag.split_reason_by_member
    # Issue #208 follow-up 2: post-cluster visual-span splitter. Walks
    # the diameter-split components and breaks any LARGE cluster whose
    # internal fine pose / expression / appearance span exceeds the
    # per-band limits, even when the pairwise gates allowed the
    # connections individually. Member-indexed diagnostic maps stay
    # valid across this pass (the splitter only re-groups indices).
    components, _visual_split_subs = _split_high_visual_span_components(
        components, features, split_reason_by_member
    )
    # Issue #208 follow-up 3: refresh per-member diagnostics for each
    # sub-component the visual splitter produced. Without this, the
    # diameter / nearest-neighbour / direct-to-rep fields would still
    # describe the PRE-split parent component — pointing to faces no
    # longer in the final cluster.
    for _sub in _visual_split_subs:
        _refresh_component_diagnostics(
            _sub,
            features,
            ctx,
            config,
            edge_reasons,
            diameter_by_member=diameter_by_member,
            nearest_neighbor_by_member=nearest_neighbor_by_member,
            nearest_neighbor_index_by_member=nearest_neighbor_index_by_member,
            direct_to_rep_by_member=direct_to_rep_by_member,
            connected_via_rep_edge_by_member=connected_via_rep_edge_by_member,
        )

    def _edge_reason_for(member_index: int | None, rep_index: int | None) -> str | None:
        """Return the eligibility reason for the (rep_index, member_index) edge.

        ``edge_reasons`` is keyed off ascending index pairs ``(i, j)`` and
        now stores eligible graph edges only. Missing entries therefore mean
        singleton clusters, absent indices, or pairs that failed eligibility
        during the streaming union-find pass.
        """
        if member_index is None or rep_index is None or member_index == rep_index:
            return None
        key = (rep_index, member_index) if rep_index < member_index else (member_index, rep_index)
        entry = edge_reasons.get(key)
        if entry is None:
            return None
        eligible, reason = entry
        return reason if eligible else None

    def _pair_diagnostics_for(
        member_index: int,
        rep_index: int,
    ) -> tuple[float, int, float, int | None, bool, str | None]:
        """Return pair diagnostics, recomputing non-eligible pairs on demand.

        The clustering pass keeps only eligible edges to avoid O(N^2)
        memory. Non-representative records still need their direct
        member-to-representative distance and temporal confidence for the
        recommendation layer, so calculate that one pair lazily when the
        edge was not cached.
        """

        if member_index == rep_index:
            return 0.0, len(_FEATURE_ATTRS), 1.0, 0, True, None

        key = (rep_index, member_index) if rep_index < member_index else (member_index, rep_index)
        cached = edges.get(key)
        if cached is not None:
            return (*cached, True, _edge_reason_for(member_index, rep_index))

        distance, compared = _distance_with_ctx(ctx, member_index, rep_index)
        t_conf, frame_distance = _temporal_confidence(
            features[member_index].frame_index,
            features[rep_index].frame_index,
            config,
        )
        eligible, reason = can_create_redundancy_edge(
            features_a=features[member_index],
            features_b=features[rep_index],
            distance=distance,
            compared=compared,
            temporal_confidence=t_conf,
            config=config,
        )
        return distance, compared, t_conf, frame_distance, eligible, reason

    def _neighbor_frame_for(records: list[FaceQARecord], neighbor_idx: int | None) -> str | None:
        if neighbor_idx is None:
            return None
        return records[neighbor_idx].frame

    def _neighbor_face_index_for(
        records: list[FaceQARecord], neighbor_idx: int | None
    ) -> int | None:
        if neighbor_idx is None:
            return None
        return records[neighbor_idx].face_index

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
        cluster_diameter = diameter_by_member.get(rep_index)
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
                nearest_neighbor_distance=_finite_or_none(
                    nearest_neighbor_by_member.get(rep_index, 0.0)
                ),
                direct_to_representative=True,
                component_diameter=_finite_or_none(cluster_diameter),
                nearest_neighbor_frame=_neighbor_frame_for(
                    record_list,
                    nearest_neighbor_index_by_member.get(rep_index),
                ),
                nearest_neighbor_face_index=_neighbor_face_index_for(
                    record_list,
                    nearest_neighbor_index_by_member.get(rep_index),
                ),
                nearest_neighbor_edge_eligibility=_edge_reason_for(
                    nearest_neighbor_index_by_member.get(rep_index), rep_index
                ),
                connected_via_representative=True,
                split_reason=split_reason_by_member.get(rep_index),
                appearance_mode=rep_features.appearance_mode,
                yaw_micro_band=rep_features.yaw_micro_band,
                pitch_micro_band=rep_features.pitch_micro_band,
                mouth_openness_band=rep_features.mouth_openness_band,
                smile_proxy_band=rep_features.smile_proxy_band,
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
            distance, compared, t_conf, t_dist, direct_rep_eligible, direct_rep_edge_reason = (
                _pair_diagnostics_for(member, rep_index)
            )

            if not direct_rep_eligible:
                recommendation, reason, budget = (
                    REVIEW,
                    (
                        "cluster-linked but not directly redundant with representative"
                        + (f": {direct_rep_edge_reason}" if direct_rep_edge_reason else "")
                    ),
                    budget,
                )
            else:
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
                    rep_features=rep_features,
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
                    # Issue #199 follow-up: surface the per-pair diagnostics
                    # so the JSON output explains WHY each member was kept,
                    # reviewed, or marked as a prune candidate. Representatives
                    # keep the 0/None defaults — they have no edge of their
                    # own to describe.
                    compared_metrics=compared,
                    edge_eligibility=_edge_reason_for(member, rep_index),
                    # Issue #204 — per-member component diagnostics.
                    nearest_neighbor_distance=_finite_or_none(
                        nearest_neighbor_by_member.get(member)
                    ),
                    direct_to_representative=direct_to_rep_by_member.get(member, False),
                    component_diameter=_finite_or_none(diameter_by_member.get(member)),
                    nearest_neighbor_frame=_neighbor_frame_for(
                        record_list,
                        nearest_neighbor_index_by_member.get(member),
                    ),
                    nearest_neighbor_face_index=_neighbor_face_index_for(
                        record_list,
                        nearest_neighbor_index_by_member.get(member),
                    ),
                    nearest_neighbor_edge_eligibility=_edge_reason_for(
                        nearest_neighbor_index_by_member.get(member), member
                    ),
                    connected_via_representative=connected_via_rep_edge_by_member.get(
                        member, False
                    ),
                    split_reason=split_reason_by_member.get(member),
                    appearance_mode=member_features.appearance_mode,
                    yaw_micro_band=member_features.yaw_micro_band,
                    pitch_micro_band=member_features.pitch_micro_band,
                    mouth_openness_band=member_features.mouth_openness_band,
                    smile_proxy_band=member_features.smile_proxy_band,
                )
            )

    # Issue #208 — readiness FAIL safety cap. When the supplied coverage
    # report scores below the FAIL threshold AND the proposed prune
    # ratio is extreme, demote BORDERLINE prune candidates to review so
    # an under-developed faceset doesn't get gutted before the user has
    # a chance to look at the topology. Obvious duplicates (where the
    # member's ``edge_eligibility`` reason names "obvious duplicate")
    # are preserved as prune candidates — the cap protects visually
    # distinct samples, not true near-duplicates.
    cap_applied, cap_reason, cap_demotions = _apply_readiness_safety_cap(
        output_records, coverage, n
    )

    keep = sum(1 for r in output_records if r.recommendation == KEEP)
    review = sum(1 for r in output_records if r.recommendation == REVIEW)
    prune = sum(1 for r in output_records if r.recommendation == PRUNE)
    multi = sum(1 for c in components if len(c) > 1)
    (
        largest_cluster_size,
        largest_cluster_ratio,
        giant_component_warning,
        component_spread_p95,
        component_diameter_max,
    ) = _compute_cluster_health(
        components,
        n,
        multi,
        diameter_by_member,
        nearest_neighbor_by_member,
    )
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
        largest_cluster_size=largest_cluster_size,
        largest_cluster_ratio=largest_cluster_ratio,
        giant_component_warning=giant_component_warning,
        component_spread_p95=component_spread_p95,
        component_diameter_max=component_diameter_max,
        readiness_safety_cap_applied=cap_applied,
        readiness_safety_cap_reason=cap_reason,
        readiness_safety_cap_demotions=cap_demotions,
        records=output_records,
    )


__all__ = get_module_objects(__name__)


# Silence the "imported but unused" lint flag for `np` — we import it because
# downstream tooling and tests rely on the same numerical environment that the
# rest of the FaceQA layer uses, even though this module's hot path is pure
# Python math.
_ = np
