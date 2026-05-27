#!/usr/bin/env python3
"""Predictive training-readiness scoring for FaceQA coverage reports."""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass, field

from lib.faceqa.coverage import PITCH_BUCKETS, YAW_BUCKETS, FacesetCoverageReport
from lib.faceqa.expression import EXPRESSION_BUCKETS
from lib.faceqa.lighting import LIGHTING_BUCKETS
from lib.utils import get_module_objects

# Per-component sub-metric weights (sum to 1).
WEIGHT_ENTROPY = 0.40
WEIGHT_OCCUPIED = 0.40
WEIGHT_MIN_SAMPLES = 0.20

# Per-bin minimum sample threshold for "covered" bins. A bin must have at
# least this many samples to count toward the min-samples-per-bin score.
MIN_SAMPLES_PER_BIN = 5

# Component weights for the overall readiness score (sum to 1).
COMPONENT_WEIGHTS: dict[str, float] = {
    "pose": 0.35,
    "expression": 0.25,
    "lighting": 0.20,
    "quality": 0.20,
}

# Quality-component penalties. Each entry maps a coverage signal to the
# maximum score deduction (in points) when the signal is at 100%.
QUALITY_PENALTIES: dict[str, float] = {
    "blur_unusable": 25.0,
    "low_resolution": 20.0,
    "misalignment": 15.0,
    "severe_occlusion": 15.0,
    "identity_outlier": 15.0,
    "duplicates": 10.0,
}

# Confidence scales between 0 and 100 based on the number of classified
# faces relative to this anchor count (log-scaled).
CONFIDENCE_FACE_ANCHOR = 500

# Score interpretation thresholds.
STRENGTH_SCORE_THRESHOLD = 75.0
RISK_SCORE_THRESHOLD = 60.0


@dataclass
class ComponentScore:
    """Score breakdown for a single readiness component."""

    name: str
    score: float
    entropy_coverage: float
    occupied_coverage: float
    min_sample_coverage: float
    weight: float
    classified_faces: int = 0
    total_bins: int = 0
    signal_coverage: float = 1.0

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "name": self.name,
            "score": self.score,
            "entropy_coverage": self.entropy_coverage,
            "occupied_coverage": self.occupied_coverage,
            "min_sample_coverage": self.min_sample_coverage,
            "weight": self.weight,
            "classified_faces": self.classified_faces,
            "total_bins": self.total_bins,
            "signal_coverage": self.signal_coverage,
        }


@dataclass
class ReadinessScores:
    """Predictive readiness scores derived from a coverage report."""

    components: dict[str, ComponentScore] = field(default_factory=dict)
    overall_readiness_score: float = 0.0
    confidence: float = 0.0
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    expected_training_risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "overall_readiness_score": self.overall_readiness_score,
            "confidence": self.confidence,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "strengths": self.strengths,
            "risks": self.risks,
            "expected_training_risks": self.expected_training_risks,
        }


def _normalized_entropy(entropy: float, total_bins: int) -> float:
    if total_bins <= 1:
        return 0.0
    return max(0.0, min(1.0, entropy / math.log2(total_bins)))


def _min_samples_ratio(counts: dict[str, int], total_bins: int) -> float:
    if total_bins <= 0:
        return 0.0
    well_sampled = sum(1 for value in counts.values() if value >= MIN_SAMPLES_PER_BIN)
    return well_sampled / total_bins


def _component_from_distribution(
    *,
    name: str,
    counts: dict[str, int],
    total_bins: int,
    entropy: float,
    classified: int,
    weight: float,
) -> ComponentScore:
    if classified == 0:
        return ComponentScore(
            name=name,
            score=0.0,
            entropy_coverage=0.0,
            occupied_coverage=0.0,
            min_sample_coverage=0.0,
            weight=weight,
            classified_faces=0,
            total_bins=total_bins,
        )
    occupied = sum(1 for value in counts.values() if value > 0)
    entropy_cov = _normalized_entropy(entropy, total_bins)
    occupied_cov = occupied / total_bins if total_bins else 0.0
    min_samples_cov = _min_samples_ratio(counts, total_bins)
    score = 100.0 * (
        WEIGHT_ENTROPY * entropy_cov
        + WEIGHT_OCCUPIED * occupied_cov
        + WEIGHT_MIN_SAMPLES * min_samples_cov
    )
    return ComponentScore(
        name=name,
        score=round(score, 2),
        entropy_coverage=round(entropy_cov, 4),
        occupied_coverage=round(occupied_cov, 4),
        min_sample_coverage=round(min_samples_cov, 4),
        weight=weight,
        classified_faces=classified,
        total_bins=total_bins,
    )


def _pose_component(coverage: FacesetCoverageReport) -> ComponentScore:
    joint = coverage.joint_pose_coverage or {}
    counts = T.cast(dict[str, int], joint.get("counts", {}))
    total_bins = int(joint.get("total_cells", len(YAW_BUCKETS) * len(PITCH_BUCKETS)))
    entropy = float(joint.get("pose_entropy", 0.0))
    classified = int(joint.get("classified_faces", 0))
    return _component_from_distribution(
        name="pose",
        counts=counts,
        total_bins=total_bins,
        entropy=entropy,
        classified=classified,
        weight=COMPONENT_WEIGHTS["pose"],
    )


def _expression_component(coverage: FacesetCoverageReport) -> ComponentScore:
    expression = coverage.expression_coverage or {}
    counts = T.cast(dict[str, int], expression.get("counts", {}))
    total_bins = int(expression.get("total_bins", len(EXPRESSION_BUCKETS)))
    entropy = float(expression.get("expression_entropy", 0.0))
    classified = int(expression.get("classified_faces", 0))
    return _component_from_distribution(
        name="expression",
        counts=counts,
        total_bins=total_bins,
        entropy=entropy,
        classified=classified,
        weight=COMPONENT_WEIGHTS["expression"],
    )


def _lighting_component(coverage: FacesetCoverageReport) -> ComponentScore:
    lighting = coverage.lighting_coverage or {}
    counts = T.cast(dict[str, int], lighting.get("counts", {}))
    total_bins = int(lighting.get("total_bins", len(LIGHTING_BUCKETS)))
    entropy = float(lighting.get("lighting_entropy", 0.0))
    classified = int(lighting.get("classified_faces", 0))
    return _component_from_distribution(
        name="lighting",
        counts=counts,
        total_bins=total_bins,
        entropy=entropy,
        classified=classified,
        weight=COMPONENT_WEIGHTS["lighting"],
    )


def _ratio(coverage: FacesetCoverageReport, dimension: str, buckets: set[str]) -> float:
    total = coverage.total_faces
    if total == 0:
        return 0.0
    counts = coverage.bucket_counts.get(dimension, {})
    return sum(counts.get(bucket, 0) for bucket in buckets) / total


def _quality_signal_coverage(coverage: FacesetCoverageReport) -> float:
    """Return the fraction of expected quality signals that have any data.

    Quality penalties default to zero when their underlying signals are missing,
    so a faceset with no duplicate/identity/occlusion data could otherwise score
    "high quality" purely from absence of evidence. Reporting this ratio lets us
    discount confidence rather than inflate the score itself.
    """
    bucket_dimensions = ("blur", "resolution", "misalignment", "occlusion")
    available = 0
    for dimension in bucket_dimensions:
        counts = coverage.bucket_counts.get(dimension, {})
        non_unknown = sum(value for key, value in counts.items() if key != "unknown")
        if non_unknown > 0:
            available += 1
    if coverage.identity_outlier_ratio is not None:
        available += 1
    if coverage.duplicate_ratio is not None:
        available += 1
    expected = len(bucket_dimensions) + 2
    return round(available / expected, 4) if expected else 0.0


def _quality_component(coverage: FacesetCoverageReport) -> ComponentScore:
    if coverage.total_faces == 0:
        return ComponentScore(
            name="quality",
            score=0.0,
            entropy_coverage=0.0,
            occupied_coverage=0.0,
            min_sample_coverage=0.0,
            weight=COMPONENT_WEIGHTS["quality"],
            classified_faces=0,
            total_bins=0,
            signal_coverage=0.0,
        )
    ratios = {
        "blur_unusable": _ratio(coverage, "blur", {"unusable"}),
        "low_resolution": _ratio(coverage, "resolution", {"low", "tiny"}),
        "misalignment": _ratio(coverage, "misalignment", {"high", "extreme"}),
        "severe_occlusion": _ratio(coverage, "occlusion", {"severe"}),
        "identity_outlier": coverage.identity_outlier_ratio or 0.0,
        "duplicates": coverage.duplicate_ratio or 0.0,
    }
    penalty = sum(QUALITY_PENALTIES[name] * ratio for name, ratio in ratios.items())
    score = max(0.0, min(100.0, 100.0 - penalty))
    return ComponentScore(
        name="quality",
        score=round(score, 2),
        entropy_coverage=0.0,
        occupied_coverage=0.0,
        min_sample_coverage=0.0,
        weight=COMPONENT_WEIGHTS["quality"],
        classified_faces=coverage.total_faces,
        total_bins=0,
        signal_coverage=_quality_signal_coverage(coverage),
    )


def _confidence(coverage: FacesetCoverageReport) -> float:
    total = coverage.total_faces
    if total <= 0:
        return 0.0
    return round(min(100.0, 100.0 * math.log(1 + total) / math.log(1 + CONFIDENCE_FACE_ANCHOR)), 2)


_RISK_RULES: tuple[tuple[str, str, float, str], ...] = (
    (
        "pose",
        "score",
        RISK_SCORE_THRESHOLD,
        "Profile and off-axis face reconstruction will likely fail.",
    ),
    (
        "expression",
        "score",
        RISK_SCORE_THRESHOLD,
        "Mouth and smile reconstruction will be limited.",
    ),
    (
        "lighting",
        "score",
        RISK_SCORE_THRESHOLD,
        "Swap quality will drop in unseen lighting conditions.",
    ),
    (
        "quality",
        "score",
        RISK_SCORE_THRESHOLD,
        "Per-frame defects (blur, low resolution, occlusion) will degrade fidelity.",
    ),
)


def _expected_training_risks(
    coverage: FacesetCoverageReport,
    components: dict[str, ComponentScore],
) -> list[str]:
    risks: list[str] = []
    for component, attribute, threshold, message in _RISK_RULES:
        score_obj = components.get(component)
        if score_obj is None:
            continue
        value = float(getattr(score_obj, attribute))
        if value < threshold and score_obj.classified_faces > 0:
            risks.append(message)
    if coverage.duplicate_ratio is not None and coverage.duplicate_ratio > 0.5:
        risks.append(
            "Training will likely overfit to repeated frames "
            f"({coverage.duplicate_ratio:.0%} marked as prune candidates)."
        )
    if coverage.identity_outlier_ratio is not None and coverage.identity_outlier_ratio > 0.3:
        risks.append(
            "Identity drift risk: "
            f"{coverage.identity_outlier_ratio:.0%} of faces are outliers or rejects."
        )
    return risks


def _component_summary_lines(
    components: dict[str, ComponentScore],
    threshold: float,
    *,
    high: bool,
) -> list[str]:
    """Return lines for components above or below a score threshold."""
    items = sorted(
        components.values(),
        key=lambda c: c.score,
        reverse=high,
    )
    result: list[str] = []
    for component in items:
        if component.classified_faces == 0:
            continue
        meets_threshold = component.score >= threshold if high else component.score < threshold
        if meets_threshold:
            result.append(f"{component.name} ({component.score:.1f})")
    return result


def compute_readiness_scores(coverage: FacesetCoverageReport) -> ReadinessScores:
    """Compute deterministic readiness scores from a coverage report."""
    components: dict[str, ComponentScore] = {
        "pose": _pose_component(coverage),
        "expression": _expression_component(coverage),
        "lighting": _lighting_component(coverage),
        "quality": _quality_component(coverage),
    }
    overall = sum(comp.score * comp.weight for comp in components.values())
    overall_score = round(overall, 2)
    quality_signal_coverage = components["quality"].signal_coverage
    confidence = round(_confidence(coverage) * quality_signal_coverage, 2)
    strengths = _component_summary_lines(components, STRENGTH_SCORE_THRESHOLD, high=True)
    risks = _component_summary_lines(components, RISK_SCORE_THRESHOLD, high=False)
    expected_risks = _expected_training_risks(coverage, components)
    return ReadinessScores(
        components=components,
        overall_readiness_score=overall_score,
        confidence=confidence,
        strengths=strengths,
        risks=risks,
        expected_training_risks=expected_risks,
    )


__all__ = get_module_objects(__name__)
