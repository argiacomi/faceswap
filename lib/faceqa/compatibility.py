#!/usr/bin/env python3
"""Source-target compatibility scoring for FaceQA coverage reports."""

from __future__ import annotations

import json
import math
import typing as T
from dataclasses import dataclass, field

from lib.faceqa.coverage import FacesetCoverageReport
from lib.faceqa.scoring import compute_readiness_scores
from lib.utils import get_module_objects

# Compatibility component weights (sum to 1).
COMPATIBILITY_WEIGHTS: dict[str, float] = {
    "pose": 0.40,
    "expression": 0.25,
    "lighting": 0.25,
    "quality": 0.10,
}

# Numerical tolerance for proportion comparisons.
PROPORTION_EPSILON = 1e-9

# When listing coverage gaps in the human report, only emit buckets where the
# target proportion exceeds this threshold (so trivial blips don't generate noise).
GAP_MIN_TARGET_PROPORTION = 0.02

# A bucket is "deficient" when source coverage is below this fraction of target demand.
GAP_DEFICIT_RATIO = 0.5

# Maximum number of bucket gaps to emit per dimension.
MAX_GAPS_PER_DIMENSION = 6

# Confidence is log-anchored to this face count (per side).
CONFIDENCE_FACE_ANCHOR = 500

# Human-readable label fragments used to phrase coverage gaps.
_BUCKET_PHRASINGS: dict[str, dict[str, str]] = {
    "pose": {
        "left_extreme": "left-extreme yaw views",
        "left_profile": "left-profile views",
        "left_slight": "left-slight yaw views",
        "frontal": "frontal views",
        "right_slight": "right-slight yaw views",
        "right_profile": "right-profile views",
        "right_extreme": "right-extreme yaw views",
    },
    "pitch": {
        "down_extreme": "extreme downward pitch",
        "down": "downward pitch",
        "neutral": "neutral pitch",
        "up": "upward pitch",
        "up_extreme": "extreme upward pitch",
    },
    "expression": {
        "neutral": "neutral expressions",
        "slight_open": "slightly-parted-lip expressions",
        "talking_open": "talking / open-mouth expressions",
        "smile": "smiling expressions",
        "eyes_closed": "blinking / eyes-closed frames",
        "expressive": "expressive / asymmetric frames",
    },
    "lighting": {
        "dark": "low-light conditions",
        "overexposed": "overexposed / very bright lighting",
        "side_lit": "strong side-lighting",
        "top_lit": "strong overhead / under-lighting",
        "high_contrast": "high-contrast lighting",
        "warm": "warm-toned lighting",
        "cool": "cool-toned lighting",
        "flat_frontal": "flat / frontal lighting",
    },
}


@dataclass
class BucketGap:
    """Per-bucket deficiency between target demand and source coverage."""

    bucket: str
    target_proportion: float
    source_proportion: float
    deficit: float

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "bucket": self.bucket,
            "target_proportion": self.target_proportion,
            "source_proportion": self.source_proportion,
            "deficit": self.deficit,
        }


@dataclass
class DimensionCompatibility:
    """Compatibility breakdown for one dimension (pose, expression, lighting, ...)."""

    name: str
    score: float
    weight: float
    target_total: int
    source_total: int
    target_proportions: dict[str, float] = field(default_factory=dict)
    source_proportions: dict[str, float] = field(default_factory=dict)
    deficiencies: list[BucketGap] = field(default_factory=list)

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "target_total": self.target_total,
            "source_total": self.source_total,
            "target_proportions": self.target_proportions,
            "source_proportions": self.source_proportions,
            "deficiencies": [gap.to_dict() for gap in self.deficiencies],
        }


@dataclass
class CompatibilityReport:
    """Source-target compatibility report."""

    source_path: str = ""
    target_path: str = ""
    source_total_faces: int = 0
    target_total_faces: int = 0
    source_target_compatibility_score: float = 0.0
    pose_compatibility_score: float = 0.0
    expression_compatibility_score: float = 0.0
    lighting_compatibility_score: float = 0.0
    quality_compatibility_score: float = 0.0
    confidence: float = 0.0
    dimensions: dict[str, DimensionCompatibility] = field(default_factory=dict)
    coverage_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "source_total_faces": self.source_total_faces,
            "target_total_faces": self.target_total_faces,
            "source_target_compatibility_score": self.source_target_compatibility_score,
            "pose_compatibility_score": self.pose_compatibility_score,
            "expression_compatibility_score": self.expression_compatibility_score,
            "lighting_compatibility_score": self.lighting_compatibility_score,
            "quality_compatibility_score": self.quality_compatibility_score,
            "confidence": self.confidence,
            "dimensions": {name: dim.to_dict() for name, dim in self.dimensions.items()},
            "coverage_gaps": self.coverage_gaps,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# FaceQA Source-Target Compatibility",
            "",
            f"- **Source**: `{self.source_path}`",
            f"- **Target**: `{self.target_path}`",
            f"- **Source faces**: {self.source_total_faces}",
            f"- **Target faces**: {self.target_total_faces}",
            "",
            "## Compatibility Summary",
            "",
            f"- **Overall score**: {self.source_target_compatibility_score:.1f} / 100",
            f"- **Pose**: {self.pose_compatibility_score:.1f}",
            f"- **Expression**: {self.expression_compatibility_score:.1f}",
            f"- **Lighting**: {self.lighting_compatibility_score:.1f}",
            f"- **Quality**: {self.quality_compatibility_score:.1f}",
            f"- **Confidence**: {self.confidence:.1f}",
        ]
        if self.dimensions:
            lines.extend(
                [
                    "",
                    "| Dimension | Score | Weight | Source Faces | Target Faces |",
                    "|-----------|------:|------:|-------------:|-------------:|",
                ]
            )
            for dim in self.dimensions.values():
                lines.append(
                    "| "
                    f"{dim.name} | {dim.score:.1f} | {dim.weight:.2f} | "
                    f"{dim.source_total} | {dim.target_total} |"
                )
        if self.coverage_gaps:
            lines.extend(["", "## Coverage Gaps", ""])
            lines.extend(f"- {gap}" for gap in self.coverage_gaps)
        for dim in self.dimensions.values():
            if not dim.deficiencies:
                continue
            lines.extend(["", f"### {dim.name.title()} deficiencies", ""])
            lines.extend(
                [
                    "| Bucket | Target % | Source % | Deficit |",
                    "|--------|--------:|--------:|--------:|",
                ]
            )
            for gap in dim.deficiencies:
                lines.append(
                    "| "
                    f"{gap.bucket} | {gap.target_proportion * 100:.1f} | "
                    f"{gap.source_proportion * 100:.1f} | {gap.deficit * 100:.1f} |"
                )
        lines.append("")
        return "\n".join(lines)


def _proportions(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {bucket: 0.0 for bucket in counts}
    return {bucket: value / total for bucket, value in counts.items()}


def _coverage_score(
    target_props: dict[str, float],
    source_props: dict[str, float],
) -> tuple[float, list[BucketGap]]:
    """Return a 0-100 compatibility score plus the list of deficient buckets."""
    if not target_props or sum(target_props.values()) <= 0:
        # No target demand → vacuously compatible.
        return 100.0, []
    score = 0.0
    gaps: list[BucketGap] = []
    for bucket, target_prop in target_props.items():
        if target_prop <= 0:
            continue
        source_prop = source_props.get(bucket, 0.0)
        coverage_factor = min(1.0, source_prop / max(target_prop, PROPORTION_EPSILON))
        score += target_prop * coverage_factor
        if target_prop >= GAP_MIN_TARGET_PROPORTION and coverage_factor < GAP_DEFICIT_RATIO:
            gaps.append(
                BucketGap(
                    bucket=bucket,
                    target_proportion=round(target_prop, 4),
                    source_proportion=round(source_prop, 4),
                    deficit=round(target_prop - source_prop, 4),
                )
            )
    gaps.sort(key=lambda gap: gap.deficit, reverse=True)
    return round(100.0 * score, 2), gaps[:MAX_GAPS_PER_DIMENSION]


def _bucket_counts(
    coverage_payload: dict[str, T.Any] | None,
) -> tuple[dict[str, int], int]:
    """Return bucket counts and the count of classified (non-unknown) records."""
    if not coverage_payload:
        return {}, 0
    counts = T.cast(dict[str, int], coverage_payload.get("counts", {}))
    classified = int(coverage_payload.get("classified_faces", sum(counts.values())))
    return dict(counts), classified


def _dimension_from_distributions(
    *,
    name: str,
    target_coverage: dict[str, T.Any] | None,
    source_coverage: dict[str, T.Any] | None,
    weight: float,
) -> DimensionCompatibility:
    target_counts, target_total = _bucket_counts(target_coverage)
    source_counts, source_total = _bucket_counts(source_coverage)
    bucket_universe = sorted(set(target_counts) | set(source_counts))
    target_props = {
        b: target_counts.get(b, 0) / target_total if target_total else 0.0 for b in bucket_universe
    }
    source_props = {
        b: source_counts.get(b, 0) / source_total if source_total else 0.0 for b in bucket_universe
    }
    score, gaps = _coverage_score(target_props, source_props)
    if target_total == 0:
        # No target demand → undefined; we don't claim full marks blindly.
        score = 0.0 if source_total == 0 else 100.0
    return DimensionCompatibility(
        name=name,
        score=score,
        weight=weight,
        target_total=target_total,
        source_total=source_total,
        target_proportions={k: round(v, 4) for k, v in target_props.items()},
        source_proportions={k: round(v, 4) for k, v in source_props.items()},
        deficiencies=gaps,
    )


def _quality_dimension(
    *,
    source_coverage: FacesetCoverageReport,
    target_coverage: FacesetCoverageReport,
    weight: float,
) -> DimensionCompatibility:
    source_quality = compute_readiness_scores(source_coverage).components["quality"]
    target_quality = compute_readiness_scores(target_coverage).components["quality"]
    if target_quality.score <= 0:
        score = 100.0 if source_quality.score > 0 else 0.0
    else:
        score = round(100.0 * min(1.0, source_quality.score / target_quality.score), 2)
    return DimensionCompatibility(
        name="quality",
        score=score,
        weight=weight,
        target_total=target_coverage.total_faces,
        source_total=source_coverage.total_faces,
        target_proportions={"quality_score": round(target_quality.score, 4)},
        source_proportions={"quality_score": round(source_quality.score, 4)},
        deficiencies=[],
    )


def _confidence(
    source_coverage: FacesetCoverageReport, target_coverage: FacesetCoverageReport
) -> float:
    smaller = min(source_coverage.total_faces, target_coverage.total_faces)
    if smaller <= 0:
        return 0.0
    return round(
        min(100.0, 100.0 * math.log(1 + smaller) / math.log(1 + CONFIDENCE_FACE_ANCHOR)), 2
    )


def _phrase_gap(dimension: str, bucket: str, gap: BucketGap) -> str:
    label = _BUCKET_PHRASINGS.get(dimension, {}).get(bucket, bucket)
    if gap.source_proportion <= PROPORTION_EPSILON:
        adverb = "not present in source"
    elif gap.source_proportion < gap.target_proportion * 0.25:
        adverb = "barely present in source"
    else:
        adverb = "underrepresented in source"
    return (
        f"Target contains {label} ({gap.target_proportion * 100:.0f}% of target) {adverb} "
        f"({gap.source_proportion * 100:.0f}% of source)."
    )


def _gap_phrases(dimensions: dict[str, DimensionCompatibility]) -> list[str]:
    phrases: list[str] = []
    for name, dim in dimensions.items():
        for gap in dim.deficiencies:
            phrases.append(_phrase_gap(name, gap.bucket, gap))
    return phrases


def compute_compatibility(
    source_coverage: FacesetCoverageReport,
    target_coverage: FacesetCoverageReport,
    *,
    source_path: str = "",
    target_path: str = "",
) -> CompatibilityReport:
    """Return a deterministic source-target compatibility report."""
    dimensions: dict[str, DimensionCompatibility] = {
        "pose": _dimension_from_distributions(
            name="pose",
            target_coverage=target_coverage.joint_pose_coverage,
            source_coverage=source_coverage.joint_pose_coverage,
            weight=COMPATIBILITY_WEIGHTS["pose"],
        ),
        "expression": _dimension_from_distributions(
            name="expression",
            target_coverage=target_coverage.expression_coverage,
            source_coverage=source_coverage.expression_coverage,
            weight=COMPATIBILITY_WEIGHTS["expression"],
        ),
        "lighting": _dimension_from_distributions(
            name="lighting",
            target_coverage=target_coverage.lighting_coverage,
            source_coverage=source_coverage.lighting_coverage,
            weight=COMPATIBILITY_WEIGHTS["lighting"],
        ),
        "quality": _quality_dimension(
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            weight=COMPATIBILITY_WEIGHTS["quality"],
        ),
    }
    overall = sum(dim.score * dim.weight for dim in dimensions.values())
    confidence = _confidence(source_coverage, target_coverage)
    coverage_gaps = _gap_phrases(dimensions)
    return CompatibilityReport(
        source_path=source_path,
        target_path=target_path,
        source_total_faces=source_coverage.total_faces,
        target_total_faces=target_coverage.total_faces,
        source_target_compatibility_score=round(overall, 2),
        pose_compatibility_score=dimensions["pose"].score,
        expression_compatibility_score=dimensions["expression"].score,
        lighting_compatibility_score=dimensions["lighting"].score,
        quality_compatibility_score=dimensions["quality"].score,
        confidence=confidence,
        dimensions=dimensions,
        coverage_gaps=coverage_gaps,
    )


__all__ = get_module_objects(__name__)
