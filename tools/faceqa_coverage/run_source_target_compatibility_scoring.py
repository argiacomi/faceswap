#!/usr/bin/env python3
"""Score source-target faceset compatibility from FaceQA coverage.

Run offline against two alignments files (and optional FaceQA sidecars and source
frames directories) and emit:

- Markdown summary printed to stdout
- JSON artifact at the chosen output path
- Markdown artifact at the chosen output path

Example
-------

::

    python run_source_target_compatibility_scoring.py \
        --source /path/to/source_alignments.fsa \
        --target /path/to/target_alignments.fsa \
        --source-frames-dir /path/to/source_frames \
        --target-frames-dir /path/to/target_frames \
        --output-dir reports/

Outputs ``reports/source_target_compatibility.json`` and
``reports/source_target_compatibility.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lib.align.faceset_qa import load as load_sidecar
from lib.align.faceset_qa import sidecar_path
from lib.faceqa.compatibility import compute_compatibility
from lib.faceqa.coverage import (
    FacesetCoverageReport,
    SpigaPoseBackfiller,
    compute_coverage,
    records_from_alignments,
)
from tools.alignments.media import Frames

logger = logging.getLogger("faceqa.compatibility")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source-target FaceQA compatibility scores from two alignments files."
        ),
    )
    parser.add_argument(
        "--source", required=True, help="Path to the source faceset alignments file."
    )
    parser.add_argument(
        "--target", required=True, help="Path to the target faceset alignments file."
    )
    parser.add_argument(
        "--source-sidecar",
        default=None,
        help=(
            "Optional explicit path to a source FaceQA sidecar JSON. "
            "Defaults to the auto-derived sibling file if present."
        ),
    )
    parser.add_argument(
        "--target-sidecar",
        default=None,
        help=(
            "Optional explicit path to a target FaceQA sidecar JSON. "
            "Defaults to the auto-derived sibling file if present."
        ),
    )
    parser.add_argument(
        "--source-frames-dir",
        default=None,
        help=(
            "Optional directory of source frames used for SPIGA pose backfill. "
            "Skipped when omitted."
        ),
    )
    parser.add_argument(
        "--target-frames-dir",
        default=None,
        help=(
            "Optional directory of target frames used for SPIGA pose backfill. "
            "Skipped when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write JSON and Markdown artifacts. "
            "Defaults to the source alignments parent directory."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="source_target_compatibility",
        help="Filename stem for emitted artifacts (default: source_target_compatibility).",
    )
    parser.add_argument(
        "--exclude-duplicates",
        action="store_true",
        help="Exclude duplicate-candidate faces from coverage counts.",
    )
    parser.add_argument(
        "--exclude-outliers",
        action="store_true",
        help="Exclude identity outliers from coverage counts.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the printed Markdown summary; only emit artifacts.",
    )
    return parser.parse_args(argv)


def _resolve_sidecar(alignments: Path, explicit: str | None) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise SystemExit(f"FaceQA sidecar not found: {candidate}")
        return candidate
    candidate = Path(sidecar_path(str(alignments)))
    return candidate if candidate.is_file() else None


def _build_coverage(
    *,
    label: str,
    alignments: Path,
    sidecar: Path | None,
    frames_dir: str | None,
    exclude_duplicates: bool,
    exclude_outliers: bool,
) -> FacesetCoverageReport:
    if not alignments.is_file():
        raise SystemExit(f"{label} alignments file not found: {alignments}")
    qa_file = load_sidecar(str(sidecar)) if sidecar is not None else None
    logger.info(
        "[%s] alignments=%s sidecar=%s frames_dir=%s",
        label,
        alignments,
        sidecar,
        frames_dir,
    )
    pose_backfiller = SpigaPoseBackfiller(Frames(frames_dir)) if frames_dir else None
    records = records_from_alignments(
        alignments,
        qa_file=qa_file,
        pose_backfiller=pose_backfiller,
    )
    return compute_coverage(
        records,
        exclude_duplicates=exclude_duplicates,
        exclude_outliers=exclude_outliers,
        sidecar_used=qa_file is not None,
    )


def _resolve_output_dir(args: argparse.Namespace, source_alignments: Path) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir else source_alignments.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)

    source_alignments = Path(args.source)
    target_alignments = Path(args.target)
    source_sidecar = _resolve_sidecar(source_alignments, args.source_sidecar)
    target_sidecar = _resolve_sidecar(target_alignments, args.target_sidecar)

    source_coverage = _build_coverage(
        label="source",
        alignments=source_alignments,
        sidecar=source_sidecar,
        frames_dir=args.source_frames_dir,
        exclude_duplicates=args.exclude_duplicates,
        exclude_outliers=args.exclude_outliers,
    )
    target_coverage = _build_coverage(
        label="target",
        alignments=target_alignments,
        sidecar=target_sidecar,
        frames_dir=args.target_frames_dir,
        exclude_duplicates=args.exclude_duplicates,
        exclude_outliers=args.exclude_outliers,
    )

    report = compute_compatibility(
        source_coverage,
        target_coverage,
        source_path=str(source_alignments),
        target_path=str(target_alignments),
    )

    output_dir = _resolve_output_dir(args, source_alignments)
    output_json = output_dir / f"{args.output_prefix}.json"
    output_markdown = output_dir / f"{args.output_prefix}.md"
    output_json.write_text(report.to_json() + "\n", encoding="utf-8")
    output_markdown.write_text(report.to_markdown(), encoding="utf-8")

    if not args.quiet:
        sys.stdout.write(report.to_markdown())
        if not report.to_markdown().endswith("\n"):
            sys.stdout.write("\n")

    logger.info("Wrote JSON report: %s", output_json)
    logger.info("Wrote Markdown report: %s", output_markdown)
    logger.info(
        "Overall compatibility: %.1f (pose=%.1f, expression=%.1f, lighting=%.1f, quality=%.1f, confidence=%.1f)",
        report.source_target_compatibility_score,
        report.pose_compatibility_score,
        report.expression_compatibility_score,
        report.lighting_compatibility_score,
        report.quality_compatibility_score,
        report.confidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
