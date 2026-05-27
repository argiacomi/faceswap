#!/usr/bin/env python3
"""Faceset readiness report generation."""

from __future__ import annotations

import json
import typing as T
from dataclasses import dataclass, field

from lib.faceqa.coverage import FacesetCoverageReport
from lib.utils import get_module_objects

SCHEMA_VERSION = 1
REPORT_TYPE = "faceqa_coverage"
DEFAULT_MIN_BUCKET_PCT = 5.0
MIN_USABLE_FACES = 50
UNUSABLE_BLUR_WARN_RATIO = 0.20
LOW_RES_WARN_RATIO = 0.20
MISALIGNMENT_WARN_RATIO = 0.10
DUPLICATE_WARN_RATIO = 0.50
OUTLIER_WARN_RATIO = 0.30
POSE_FALLBACK_WARN_RATIO = 0.40
LOW_CONFIDENCE_POSE_WARN_RATIO = 0.20
JOINT_POSE_COVERAGE_WARN_PCT = 50.0
JOINT_POSE_ENTROPY_WARN_BITS = 3.0
MISSING_POSE_CELL_REPORT_LIMIT = 8
EXPRESSION_COVERAGE_WARN_PCT = 60.0
EXPRESSION_ENTROPY_WARN_BITS = 1.5
EXPRESSION_BUCKET_WARN_PCT = 5.0
MISSING_EXPRESSION_BIN_REPORT_LIMIT = 6
LIGHTING_COVERAGE_WARN_PCT = 50.0
LIGHTING_ENTROPY_WARN_BITS = 1.5
MISSING_LIGHTING_BIN_REPORT_LIMIT = 6

LIGHTING_GUIDANCE: dict[str, str] = {
    "dark": "low-light frames",
    "overexposed": "very bright / overexposed frames",
    "side_lit": "side-lit frames (strong left/right asymmetry)",
    "top_lit": "frames with strong overhead or under-lighting",
    "high_contrast": "high-contrast frames",
    "warm": "warm-toned frames",
    "cool": "cool-toned frames",
    "flat_frontal": "flat / frontal-lit frames",
}

EXPRESSION_GUIDANCE: dict[str, str] = {
    "neutral": "neutral / resting face frames",
    "slight_open": "frames with slightly parted lips",
    "talking_open": "open-mouth or talking frames",
    "smile": "smiling frames",
    "eyes_closed": "blinking or eyes-closed frames",
    "expressive": "expressive / asymmetric frames (raised brows, smirks)",
}


@dataclass
class ReadinessReport:
    """Human and machine-readable readiness assessment."""

    alignments: str = ""
    sidecar: str | None = None
    schema_version: int = SCHEMA_VERSION
    report_type: str = REPORT_TYPE
    total_faces: int = 0
    usable_faces: int = 0
    coverage: dict[str, dict[str, T.Any]] = field(default_factory=dict)
    metric_summary: dict[str, dict[str, float | None]] = field(default_factory=dict)
    duplicate_ratio: float | None = None
    identity_outlier_ratio: float | None = None
    mask_qa_distribution: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    joint_pose_coverage: dict[str, T.Any] = field(default_factory=dict)
    expression_coverage: dict[str, T.Any] = field(default_factory=dict)
    lighting_coverage: dict[str, T.Any] = field(default_factory=dict)
    underrepresented_buckets: list[dict[str, str | float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, T.Any]:
        """Return a stable JSON-serializable report."""
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "alignments": self.alignments,
            "sidecar": self.sidecar,
            "total_faces": self.total_faces,
            "usable_faces": self.usable_faces,
            "coverage": self.coverage,
            "metric_summary": self.metric_summary,
            "duplicate_ratio": self.duplicate_ratio,
            "identity_outlier_ratio": self.identity_outlier_ratio,
            "mask_qa_distribution": self.mask_qa_distribution,
            "source_counts": self.source_counts,
            "joint_pose_coverage": self.joint_pose_coverage,
            "expression_coverage": self.expression_coverage,
            "lighting_coverage": self.lighting_coverage,
            "underrepresented_buckets": self.underrepresented_buckets,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the report as JSON text."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Return a Markdown audit summary."""
        lines = [
            "# FaceQA Coverage Report",
            "",
            f"- **Alignments**: `{self.alignments}`",
            f"- **Sidecar**: `{self.sidecar}`" if self.sidecar else "- **Sidecar**: not used",
            f"- **Total faces**: {self.total_faces}",
            f"- **Usable faces**: {self.usable_faces}",
        ]
        if self.duplicate_ratio is not None:
            lines.append(f"- **Duplicate ratio**: {self.duplicate_ratio:.1%}")
        if self.identity_outlier_ratio is not None:
            lines.append(f"- **Identity outlier ratio**: {self.identity_outlier_ratio:.1%}")

        for dimension, payload in self.coverage.items():
            counts = payload.get("counts", {})
            percentages = payload.get("percentages", {})
            if not counts:
                continue
            lines.extend(["", f"## {dimension.replace('_', ' ').title()}", ""])
            lines.extend(["| Bucket | Count | % |", "|--------|------:|--:|"])
            for bucket, count in counts.items():
                pct = float(percentages.get(bucket, 0.0))
                lines.append(f"| {bucket} | {count} | {pct:.1f} |")

        joint = self.joint_pose_coverage
        if joint:
            lines.extend(
                [
                    "",
                    "## Joint Pose Coverage (signed yaw x pitch)",
                    "",
                    f"- **Occupied cells**: {joint.get('occupied_pose_cells', 0)} "
                    f"of {joint.get('total_cells', 0)}",
                    f"- **Empty cells**: {joint.get('empty_pose_cells', 0)}",
                    f"- **Bin coverage**: {float(joint.get('pose_bin_coverage_pct', 0.0)):.1f}%",
                    f"- **Pose entropy (bits)**: {float(joint.get('pose_entropy', 0.0)):.3f}",
                ]
            )
            missing = list(joint.get("missing_cells", []))
            if missing:
                preview = ", ".join(missing[:MISSING_POSE_CELL_REPORT_LIMIT])
                if len(missing) > MISSING_POSE_CELL_REPORT_LIMIT:
                    preview += f", … (+{len(missing) - MISSING_POSE_CELL_REPORT_LIMIT} more)"
                lines.append(f"- **Missing pose regions**: {preview}")

        expression = self.expression_coverage
        if expression:
            occupied = expression.get("occupied_expression_bins", 0)
            total_bins = expression.get("total_bins", 0)
            bin_pct = float(expression.get("expression_bin_coverage_pct", 0.0))
            entropy = float(expression.get("expression_entropy", 0.0))
            lines.extend(
                [
                    "",
                    "## Expression Coverage",
                    "",
                    f"- **Occupied bins**: {occupied} of {total_bins}",
                    f"- **Empty bins**: {expression.get('empty_expression_bins', 0)}",
                    f"- **Bin coverage**: {bin_pct:.1f}%",
                    f"- **Expression entropy (bits)**: {entropy:.3f}",
                ]
            )
            missing_bins = list(expression.get("missing_bins", []))
            if missing_bins:
                preview = ", ".join(missing_bins[:MISSING_EXPRESSION_BIN_REPORT_LIMIT])
                if len(missing_bins) > MISSING_EXPRESSION_BIN_REPORT_LIMIT:
                    preview += (
                        f", … (+{len(missing_bins) - MISSING_EXPRESSION_BIN_REPORT_LIMIT} more)"
                    )
                lines.append(f"- **Missing expression regions**: {preview}")

        lighting = self.lighting_coverage
        if lighting:
            occupied = lighting.get("occupied_lighting_bins", 0)
            total_bins = lighting.get("total_bins", 0)
            bin_pct = float(lighting.get("lighting_bin_coverage_pct", 0.0))
            entropy = float(lighting.get("lighting_entropy", 0.0))
            lines.extend(
                [
                    "",
                    "## Lighting Coverage",
                    "",
                    f"- **Occupied bins**: {occupied} of {total_bins}",
                    f"- **Empty bins**: {lighting.get('empty_lighting_bins', 0)}",
                    f"- **Bin coverage**: {bin_pct:.1f}%",
                    f"- **Lighting entropy (bits)**: {entropy:.3f}",
                ]
            )
            missing_bins = list(lighting.get("missing_bins", []))
            if missing_bins:
                preview = ", ".join(missing_bins[:MISSING_LIGHTING_BIN_REPORT_LIMIT])
                if len(missing_bins) > MISSING_LIGHTING_BIN_REPORT_LIMIT:
                    preview += (
                        f", … (+{len(missing_bins) - MISSING_LIGHTING_BIN_REPORT_LIMIT} more)"
                    )
                lines.append(f"- **Missing lighting conditions**: {preview}")

        if self.metric_summary:
            lines.extend(["", "## Metric Summary", ""])
            lines.extend(["| Metric | Min | Median | Max |", "|--------|----:|-------:|----:|"])
            for metric, values in self.metric_summary.items():
                lines.append(
                    "| "
                    f"{metric} | {_fmt(values.get('min'))} | "
                    f"{_fmt(values.get('median'))} | {_fmt(values.get('max'))} |"
                )

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.recommendations:
            lines.extend(["", "## Recommendations", ""])
            lines.extend(f"- {recommendation}" for recommendation in self.recommendations)

        lines.append("")
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    """Format optional metric values for Markdown tables."""
    return "unknown" if value is None else f"{value:.3f}"


def _underrepresented(
    coverage: FacesetCoverageReport, min_bucket_pct: float
) -> list[dict[str, str | float]]:
    """Return non-risk coverage buckets below the configured threshold."""
    included = {
        "pose": {
            "left_extreme",
            "left_profile",
            "left_slight",
            "frontal",
            "right_slight",
            "right_profile",
            "right_extreme",
        },
        "pitch": {"down_extreme", "down", "neutral", "up", "up_extreme"},
        "lighting": {
            "dark",
            "overexposed",
            "side_lit",
            "top_lit",
            "high_contrast",
            "warm",
            "cool",
            "flat_frontal",
        },
        "expression": {
            "neutral",
            "slight_open",
            "talking_open",
            "smile",
            "eyes_closed",
            "expressive",
        },
    }
    result: list[dict[str, str | float]] = []
    for dimension, buckets in coverage.bucket_percentages.items():
        if dimension not in included:
            continue
        allowed = included[dimension]
        for bucket, pct in buckets.items():
            if bucket == "unknown" or (allowed is not None and bucket not in allowed):
                continue
            if pct < min_bucket_pct:
                result.append({"dimension": dimension, "bucket": bucket, "percentage": pct})
    return result


def _ratio(coverage: FacesetCoverageReport, dimension: str, buckets: set[str]) -> float:
    """Return the ratio of total faces in the selected buckets."""
    if coverage.total_faces == 0:
        return 0.0
    counts = coverage.bucket_counts.get(dimension, {})
    return sum(counts.get(bucket, 0) for bucket in buckets) / coverage.total_faces


def _build_warnings(
    coverage: FacesetCoverageReport,
    underrepresented: list[dict[str, str | float]],
    min_bucket_pct: float,
) -> list[str]:
    """Build ordered warning strings."""
    warnings: list[str] = []
    if coverage.total_faces == 0:
        return ["No faces were found in the supplied alignments file."]
    if coverage.usable_faces < MIN_USABLE_FACES:
        warnings.append(
            f"Faceset has only {coverage.usable_faces} usable faces "
            f"(minimum recommended: {MIN_USABLE_FACES})."
        )
    if coverage.duplicate_ratio is not None and coverage.duplicate_ratio > DUPLICATE_WARN_RATIO:
        warnings.append(
            f"High duplicate ratio: {coverage.duplicate_ratio:.1%} of faces are prune candidates."
        )
    if (
        coverage.identity_outlier_ratio is not None
        and coverage.identity_outlier_ratio > OUTLIER_WARN_RATIO
    ):
        warnings.append(
            "High identity outlier ratio: "
            f"{coverage.identity_outlier_ratio:.1%} of faces are outliers or rejects."
        )
    blur_ratio = _ratio(coverage, "blur", {"unusable"})
    if blur_ratio > UNUSABLE_BLUR_WARN_RATIO:
        warnings.append(f"High unusable-blur count: {blur_ratio:.1%} of faces are very blurry.")
    low_res_ratio = _ratio(coverage, "resolution", {"low", "tiny"})
    if low_res_ratio > LOW_RES_WARN_RATIO:
        warnings.append(f"Low-resolution risk: {low_res_ratio:.1%} of faces are low or tiny.")
    misalignment_ratio = _ratio(coverage, "misalignment", {"high", "extreme"})
    if misalignment_ratio > MISALIGNMENT_WARN_RATIO:
        warnings.append(
            "Misalignment risk: "
            f"{misalignment_ratio:.1%} of faces are far from the average face geometry."
        )
    fallback_pose_ratio = _ratio(coverage, "pose_sources", {"alignment"})
    if fallback_pose_ratio > POSE_FALLBACK_WARN_RATIO:
        warnings.append(
            f"Pose fallback risk: {fallback_pose_ratio:.1%} of faces use alignment-derived pose."
        )
    low_confidence_pose_ratio = _ratio(coverage, "pose_confidence", {"low"})
    if low_confidence_pose_ratio > LOW_CONFIDENCE_POSE_WARN_RATIO:
        warnings.append(
            "SPIGA/alignment pose disagreement: "
            f"{low_confidence_pose_ratio:.1%} of faces have low-confidence SPIGA pose."
        )
    for entry in underrepresented:
        warnings.append(
            "Under-represented bucket: "
            f"{entry['dimension']}/{entry['bucket']} at {entry['percentage']:.1f}% "
            f"(threshold: {min_bucket_pct:.1f}%)."
        )
    joint = coverage.joint_pose_coverage
    if joint:
        bin_pct = float(joint.get("pose_bin_coverage_pct", 0.0))
        if bin_pct < JOINT_POSE_COVERAGE_WARN_PCT:
            warnings.append(
                "Sparse pose coverage: only "
                f"{bin_pct:.1f}% of yaw/pitch cells are populated "
                f"({joint.get('occupied_pose_cells', 0)} of "
                f"{joint.get('total_cells', 0)})."
            )
        entropy = float(joint.get("pose_entropy", 0.0))
        if joint.get("classified_faces", 0) and entropy < JOINT_POSE_ENTROPY_WARN_BITS:
            warnings.append(
                f"Low pose entropy: {entropy:.2f} bits over occupied cells — "
                "pose distribution is concentrated in a few regions."
            )
    expression = coverage.expression_coverage
    if expression and expression.get("classified_faces", 0):
        bin_pct = float(expression.get("expression_bin_coverage_pct", 0.0))
        if bin_pct < EXPRESSION_COVERAGE_WARN_PCT:
            warnings.append(
                "Sparse expression coverage: only "
                f"{bin_pct:.1f}% of expression bins are populated "
                f"({expression.get('occupied_expression_bins', 0)} of "
                f"{expression.get('total_bins', 0)})."
            )
        entropy = float(expression.get("expression_entropy", 0.0))
        if entropy < EXPRESSION_ENTROPY_WARN_BITS:
            warnings.append(
                f"Low expression entropy: {entropy:.2f} bits — "
                "facial expressions are concentrated in a few states."
            )
    lighting = coverage.lighting_coverage
    if lighting and lighting.get("classified_faces", 0):
        bin_pct = float(lighting.get("lighting_bin_coverage_pct", 0.0))
        if bin_pct < LIGHTING_COVERAGE_WARN_PCT:
            warnings.append(
                "Sparse lighting coverage: only "
                f"{bin_pct:.1f}% of lighting bins are populated "
                f"({lighting.get('occupied_lighting_bins', 0)} of "
                f"{lighting.get('total_bins', 0)})."
            )
        entropy = float(lighting.get("lighting_entropy", 0.0))
        if entropy < LIGHTING_ENTROPY_WARN_BITS:
            warnings.append(
                f"Low lighting entropy: {entropy:.2f} bits — "
                "lighting conditions are concentrated in a narrow range."
            )
    return warnings


def _build_recommendations(
    coverage: FacesetCoverageReport,
    underrepresented: list[dict[str, str | float]],
) -> list[str]:
    """Build actionable recommendations."""
    if coverage.total_faces == 0:
        return ["Run extraction before auditing faceset coverage."]

    recommendations: list[str] = []
    if coverage.usable_faces < MIN_USABLE_FACES:
        recommendations.append("Collect more source material before training.")
    if coverage.duplicate_ratio is not None and coverage.duplicate_ratio > DUPLICATE_WARN_RATIO:
        recommendations.append("Review or prune duplicate candidates before training.")
    if (
        coverage.identity_outlier_ratio is not None
        and coverage.identity_outlier_ratio > OUTLIER_WARN_RATIO
    ):
        recommendations.append("Review identity outliers for mixed-subject frames.")
    if _ratio(coverage, "blur", {"unusable"}) > UNUSABLE_BLUR_WARN_RATIO:
        recommendations.append("Filter very blurry faces or re-extract from sharper footage.")
    if _ratio(coverage, "resolution", {"low", "tiny"}) > LOW_RES_WARN_RATIO:
        recommendations.append("Add higher-resolution source footage for stronger face detail.")
    if _ratio(coverage, "misalignment", {"high", "extreme"}) > MISALIGNMENT_WARN_RATIO:
        recommendations.append("Review high-distance alignments for bad landmarks or false faces.")
    if _ratio(coverage, "pose_sources", {"alignment"}) > POSE_FALLBACK_WARN_RATIO:
        recommendations.append(
            "Review faces without SPIGA pose backfill; missing or unreadable source frames can force alignment pose."
        )
    if _ratio(coverage, "pose_confidence", {"low"}) > LOW_CONFIDENCE_POSE_WARN_RATIO:
        recommendations.append(
            "Review low-confidence pose samples where SPIGA and alignment pose disagree."
        )

    pose_buckets = [
        str(item["bucket"]) for item in underrepresented if item["dimension"] == "pose"
    ]
    if pose_buckets:
        recommendations.append("Balance yaw coverage: " + ", ".join(pose_buckets) + ".")
    pitch_buckets = [
        str(item["bucket"]) for item in underrepresented if item["dimension"] == "pitch"
    ]
    if pitch_buckets:
        recommendations.append(
            "Add frames at these pitch angles: " + ", ".join(pitch_buckets) + "."
        )
    lighting_under = [
        str(item["bucket"]) for item in underrepresented if item["dimension"] == "lighting"
    ]
    joint = coverage.joint_pose_coverage
    missing_cells = list(joint.get("missing_cells", [])) if joint else []
    if missing_cells and joint.get("classified_faces", 0):
        preview = missing_cells[:MISSING_POSE_CELL_REPORT_LIMIT]
        remainder = len(missing_cells) - len(preview)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        recommendations.append(
            "Collect frames for missing yaw/pitch regions: " + ", ".join(preview) + suffix + "."
        )
    expression_recs = _expression_recommendations(coverage, underrepresented)
    recommendations.extend(expression_recs)
    lighting_recs = _lighting_recommendations(coverage, lighting_under)
    recommendations.extend(lighting_recs)

    if not recommendations:
        recommendations.append("Faceset coverage looks adequate for an initial training run.")
    return recommendations


def _expression_recommendations(
    coverage: FacesetCoverageReport,
    underrepresented: list[dict[str, str | float]],
) -> list[str]:
    """Build targeted recommendations for expression coverage gaps."""
    expression = coverage.expression_coverage
    if not expression or not expression.get("classified_faces", 0):
        return []
    recommendations: list[str] = []
    missing_bins = [str(item) for item in expression.get("missing_bins", [])]
    under_expression = [
        str(item["bucket"])
        for item in underrepresented
        if item["dimension"] == "expression" and str(item["bucket"]) not in missing_bins
    ]
    if missing_bins:
        guidance = [EXPRESSION_GUIDANCE.get(name, name) for name in missing_bins]
        recommendations.append("Collect missing expression frames: " + "; ".join(guidance) + ".")
    if under_expression:
        guidance = [EXPRESSION_GUIDANCE.get(name, name) for name in under_expression]
        recommendations.append(
            "Increase under-represented expressions: " + "; ".join(guidance) + "."
        )
    return recommendations


def _lighting_recommendations(
    coverage: FacesetCoverageReport,
    under_lighting: list[str],
) -> list[str]:
    """Build targeted recommendations for lighting coverage gaps."""
    lighting = coverage.lighting_coverage
    if not lighting or not lighting.get("classified_faces", 0):
        return []
    recommendations: list[str] = []
    missing_bins = [str(item) for item in lighting.get("missing_bins", [])]
    under_unique = [name for name in under_lighting if name not in missing_bins]
    if missing_bins:
        guidance = [LIGHTING_GUIDANCE.get(name, name) for name in missing_bins]
        recommendations.append("Collect missing lighting conditions: " + "; ".join(guidance) + ".")
    if under_unique:
        guidance = [LIGHTING_GUIDANCE.get(name, name) for name in under_unique]
        recommendations.append("Increase under-represented lighting: " + "; ".join(guidance) + ".")
    return recommendations


def generate_readiness_report(
    coverage: FacesetCoverageReport,
    *,
    alignments: str = "",
    sidecar: str | None = None,
    min_bucket_pct: float = DEFAULT_MIN_BUCKET_PCT,
) -> ReadinessReport:
    """Generate a readiness report from precomputed coverage."""
    underrepresented = _underrepresented(coverage, min_bucket_pct)
    warnings = _build_warnings(coverage, underrepresented, min_bucket_pct)
    recommendations = _build_recommendations(coverage, underrepresented)
    return ReadinessReport(
        alignments=alignments,
        sidecar=sidecar,
        total_faces=coverage.total_faces,
        usable_faces=coverage.usable_faces,
        coverage=coverage.coverage_dict(),
        metric_summary=coverage.metric_summary,
        duplicate_ratio=coverage.duplicate_ratio,
        identity_outlier_ratio=coverage.identity_outlier_ratio,
        mask_qa_distribution=coverage.mask_qa_distribution,
        source_counts=coverage.source_counts,
        joint_pose_coverage=coverage.joint_pose_coverage,
        expression_coverage=coverage.expression_coverage,
        lighting_coverage=coverage.lighting_coverage,
        underrepresented_buckets=underrepresented,
        warnings=warnings,
        recommendations=recommendations,
    )


__all__ = get_module_objects(__name__)
