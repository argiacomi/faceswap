#!/usr/bin/env python3
"""FaceQA coverage audit tool."""

from __future__ import annotations

import logging
import typing as T
from argparse import Namespace
from pathlib import Path

from lib.align.faceset_qa import FaceQAFile, FaceQARecord, sidecar_path
from lib.align.faceset_qa import load as load_sidecar
from lib.align.faceset_qa import save as save_sidecar
from lib.align.objects import FileAlignments
from lib.faceqa.coverage import SpigaPoseBackfiller, compute_coverage, records_from_alignments
from lib.faceqa.readiness import generate_readiness_report
from lib.utils import FaceswapError, get_module_objects
from tools.alignments.media import Frames

logger = logging.getLogger(__name__)


class Faceqa_Coverage:  # pylint:disable=invalid-name
    """Run a FaceQA coverage audit."""

    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._args = arguments

    def process(self) -> None:
        """Run the audit and write JSON and Markdown outputs."""
        alignments = Path(self._args.alignments)
        if not alignments.is_file():
            raise FaceswapError(f"Alignments file not found: {alignments}")

        sidecar = self._sidecar_path(alignments)
        qa_file = load_sidecar(str(sidecar)) if sidecar.is_file() else None
        if qa_file is None:
            logger.info("No FaceQA sidecar found. Deriving available metrics from alignments.")
        else:
            logger.info("Loaded %s FaceQA sidecar records from '%s'.", len(qa_file.faces), sidecar)

        backfill_added = False
        pose_backfiller = self._pose_backfiller()

        def track_pose_backfill(
            record: FaceQARecord,
            face: FileAlignments,
        ) -> dict[str, T.Any] | None:
            nonlocal backfill_added
            pose = pose_backfiller(record, face)
            if pose is not None:
                backfill_added = True
            return pose

        records = records_from_alignments(
            alignments,
            qa_file=qa_file,
            pose_backfiller=track_pose_backfill,
        )
        if backfill_added:
            save_sidecar(
                str(sidecar),
                FaceQAFile(generated_by="faceqa_coverage", faces=records),
            )
            logger.info("Persisted SPIGA pose backfill metadata to '%s'.", sidecar)
        coverage = compute_coverage(
            records,
            exclude_duplicates=bool(self._args.exclude_duplicates),
            exclude_outliers=bool(self._args.exclude_outliers),
            sidecar_used=qa_file is not None,
        )
        report = generate_readiness_report(
            coverage,
            alignments=str(alignments),
            sidecar=str(sidecar) if sidecar.is_file() else None,
            min_bucket_pct=float(self._args.min_bucket_pct),
        )

        output_json, output_markdown = self._output_paths(alignments)
        output_json.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
        output_markdown.write_text(report.to_markdown(), encoding="utf-8")

        logger.info(
            "Coverage audit complete: %s total faces, %s usable faces.",
            report.total_faces,
            report.usable_faces,
        )
        for warning in report.warnings[:5]:
            logger.warning("%s", warning)
        for recommendation in report.recommendations[:5]:
            logger.info("Recommendation: %s", recommendation)
        logger.info("Wrote JSON report to '%s'.", output_json)
        logger.info("Wrote Markdown report to '%s'.", output_markdown)

    def _sidecar_path(self, alignments: Path) -> Path:
        """Return an explicit or inferred sidecar path."""
        explicit = getattr(self._args, "sidecar", None)
        if explicit:
            path = Path(explicit)
            if not path.is_file():
                raise FaceswapError(f"FaceQA sidecar not found: {path}")
            return path
        return Path(sidecar_path(str(alignments)))

    def _pose_backfiller(self) -> SpigaPoseBackfiller:
        """Return the pose backfiller used for automatic FaceQA pose selection."""
        return SpigaPoseBackfiller(Frames(self._args.frames_dir))

    def _output_paths(self, alignments: Path) -> tuple[Path, Path]:
        """Return report output paths, creating parent folders as needed."""
        output_json = Path(
            self._args.output_json
            or alignments.with_name(f"{alignments.stem}_faceqa_coverage.json")
        )
        output_markdown = Path(
            self._args.output_markdown
            or alignments.with_name(f"{alignments.stem}_faceqa_coverage.md")
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        return output_json, output_markdown


__all__ = get_module_objects(__name__)
