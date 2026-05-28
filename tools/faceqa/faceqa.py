#!/usr/bin/env python3
"""Unified FaceQA tool: coverage audit, source-target compatibility, and duplicates."""

from __future__ import annotations

import logging
import math
import typing as T
from argparse import Namespace
from pathlib import Path

from lib.align.faceset_qa import FaceQAFile, FaceQARecord, sidecar_path
from lib.align.faceset_qa import load as load_sidecar
from lib.align.faceset_qa import save as save_sidecar
from lib.align.objects import FileAlignments
from lib.faceqa.compatibility import compute_compatibility
from lib.faceqa.coverage import (
    FacesetCoverageReport,
    SpigaPoseBackfiller,
    compute_coverage,
    records_from_alignments,
)
from lib.faceqa.duplicate_outputs import (
    DEFAULT_SCREEN_WIDTH,
    DEFAULT_TILE_SIZE,
    render_contact_sheets,
    write_sorted_folders,
)
from lib.faceqa.duplicates import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DuplicateReport,
    cluster_duplicates,
    write_manifests,
)
from lib.faceqa.readiness import generate_readiness_report
from lib.utils import FaceswapError, get_module_objects
from tools.alignments.media import Frames

logger = logging.getLogger(__name__)


class Faceqa:  # pylint:disable=invalid-name
    """Dispatch the requested FaceQA workflow."""

    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._args = arguments

    def process(self) -> None:
        """Dispatch the requested ``--mode`` to its handler."""
        mode = getattr(self._args, "mode", "coverage")
        if mode == "coverage":
            self._run_coverage()
        elif mode == "duplicates":
            self._run_duplicates()
        elif mode == "compatibility":
            self._run_compatibility()
        else:
            raise FaceswapError(
                f"Unknown FaceQA mode '{mode}'. Expected: coverage | duplicates | compatibility."
            )

    # ------------------------------------------------------------------
    # Coverage audit (issues #150-#154)
    # ------------------------------------------------------------------

    def _run_coverage(self) -> None:
        alignments = self._require_alignments("alignments")
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
            pose: dict[str, T.Any] | None = pose_backfiller(record, face)
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
                FaceQAFile(generated_by="faceqa", faces=records),
            )
            logger.info("Persisted SPIGA pose backfill metadata to '%s'.", sidecar)

        coverage = compute_coverage(
            records,
            exclude_duplicates=bool(getattr(self._args, "exclude_duplicates", False)),
            exclude_outliers=bool(getattr(self._args, "exclude_outliers", False)),
            sidecar_used=qa_file is not None,
        )
        report = generate_readiness_report(
            coverage,
            alignments=str(alignments),
            sidecar=str(sidecar) if sidecar.is_file() else None,
            min_bucket_pct=float(getattr(self._args, "min_bucket_pct", 5.0)),
        )

        output_json, output_markdown = self._coverage_output_paths(alignments)
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

    # ------------------------------------------------------------------
    # Duplicate clustering (issue #147)
    # ------------------------------------------------------------------

    def _run_duplicates(self) -> None:
        alignments = self._require_alignments("alignments")
        faces_dir = self._require_path("faces_dir", "Aligned faces directory")
        output_dir = self._duplicates_output_dir(alignments)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self._load_or_build_duplicate_report(alignments, output_dir)
        layout = write_sorted_folders(
            report,
            faces_dir=faces_dir,
            output_dir=output_dir,
            symlink=bool(getattr(self._args, "symlink", False)),
        )
        write_manifests(report, output_dir)
        tile_size = int(getattr(self._args, "contact_sheet_tile_size", DEFAULT_TILE_SIZE))
        columns = math.floor(DEFAULT_SCREEN_WIDTH / tile_size)
        sheets = render_contact_sheets(
            report,
            faces_dir=faces_dir,
            output_dir=layout.contact_sheets_dir,
            tile_size=tile_size,
            columns=columns,
            format_="png",
        )
        logger.info(
            "Duplicate pipeline complete: %d faces in %d clusters (%d multi-face).",
            report.total_faces,
            report.cluster_count,
            report.multi_face_clusters,
        )
        logger.info(
            "Recommendations: keep=%d, review=%d, prune_candidate=%d.",
            report.keep_count,
            report.review_count,
            report.prune_candidate_count,
        )
        logger.info("Contact sheets rendered: %d", len(sheets))
        logger.info("Output folders: %s", layout.to_dict())

    def _load_or_build_duplicate_report(
        self,
        alignments: Path,
        output_dir: Path,
    ) -> DuplicateReport:
        """Return a duplicate report from disk if provided, otherwise build one."""
        report_path = getattr(self._args, "duplicates_report", None)
        if report_path:
            return self._load_duplicate_report_from_disk(Path(report_path))
        report = cluster_duplicates(
            alignments,
            identity_model=getattr(self._args, "identity_model", None) or None,
            similarity_threshold=float(
                getattr(self._args, "similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
            ),
            temporal_window=self._optional_int("temporal_window"),
        )
        json_path = output_dir / "faceqa_duplicates.json"
        json_path.write_text(report.to_json() + "\n", encoding="utf-8")
        logger.info("Wrote duplicate report: %s", json_path)
        return report

    @staticmethod
    def _load_duplicate_report_from_disk(path: Path) -> DuplicateReport:
        if not path.is_file():
            raise FaceswapError(f"Duplicate report not found: {path}")
        import json

        from lib.faceqa.duplicates import DuplicateRecord

        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [DuplicateRecord(**item) for item in payload.get("records", [])]
        report = DuplicateReport(
            alignments=str(payload.get("alignments", "")),
            identity_model=payload.get("identity_model"),
            similarity_threshold=float(
                payload.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
            ),
            temporal_window=payload.get("temporal_window"),
            total_faces=int(payload.get("total_faces", len(records))),
            cluster_count=int(payload.get("cluster_count", 0)),
            multi_face_clusters=int(payload.get("multi_face_clusters", 0)),
            keep_count=int(payload.get("keep_count", 0)),
            review_count=int(payload.get("review_count", 0)),
            prune_candidate_count=int(payload.get("prune_candidate_count", 0)),
            skipped_no_embedding=int(payload.get("skipped_no_embedding", 0)),
            records=records,
        )
        return report

    # ------------------------------------------------------------------
    # Source-target compatibility (issue #158)
    # ------------------------------------------------------------------

    def _run_compatibility(self) -> None:
        source = self._require_alignments("source_alignments")
        target = self._require_alignments("target_alignments")
        source_coverage = self._coverage_for_compatibility(
            source,
            sidecar_arg="source_sidecar",
            frames_arg="source_frames_dir",
        )
        target_coverage = self._coverage_for_compatibility(
            target,
            sidecar_arg="target_sidecar",
            frames_arg="target_frames_dir",
        )
        report = compute_compatibility(
            source_coverage,
            target_coverage,
            source_path=str(source),
            target_path=str(target),
        )
        output_dir = self._compatibility_output_dir(source)
        prefix = getattr(self._args, "output_prefix", "source_target_compatibility")
        output_json = output_dir / f"{prefix}.json"
        output_markdown = output_dir / f"{prefix}.md"
        output_json.write_text(report.to_json() + "\n", encoding="utf-8")
        output_markdown.write_text(report.to_markdown(), encoding="utf-8")
        logger.info(
            "Overall compatibility: %.1f (pose=%.1f, expression=%.1f, "
            "lighting=%.1f, quality=%.1f, confidence=%.1f)",
            report.source_target_compatibility_score,
            report.pose_compatibility_score,
            report.expression_compatibility_score,
            report.lighting_compatibility_score,
            report.quality_compatibility_score,
            report.confidence,
        )
        logger.info("Wrote JSON: %s", output_json)
        logger.info("Wrote Markdown: %s", output_markdown)

    def _coverage_for_compatibility(
        self,
        alignments: Path,
        *,
        sidecar_arg: str,
        frames_arg: str,
    ) -> FacesetCoverageReport:
        sidecar_value = getattr(self._args, sidecar_arg, None)
        sidecar = Path(sidecar_value) if sidecar_value else Path(sidecar_path(str(alignments)))
        qa_file = load_sidecar(str(sidecar)) if sidecar.is_file() else None
        frames_dir = getattr(self._args, frames_arg, None)
        pose_backfiller = SpigaPoseBackfiller(Frames(frames_dir)) if frames_dir else None
        records = records_from_alignments(
            alignments,
            qa_file=qa_file,
            pose_backfiller=pose_backfiller,
        )
        return compute_coverage(
            records,
            exclude_duplicates=bool(getattr(self._args, "exclude_duplicates", False)),
            exclude_outliers=bool(getattr(self._args, "exclude_outliers", False)),
            sidecar_used=qa_file is not None,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _require_alignments(self, attr: str) -> Path:
        value = getattr(self._args, attr, None)
        if not value:
            raise FaceswapError(
                f"Missing required alignments argument: --{attr.replace('_', '-')}"
            )
        path = Path(value)
        if not path.is_file():
            raise FaceswapError(f"Alignments file not found: {path}")
        return path

    def _require_path(self, attr: str, label: str) -> Path:
        value = getattr(self._args, attr, None)
        if not value:
            raise FaceswapError(
                f"Missing required argument for {label}: --{attr.replace('_', '-')}"
            )
        path = Path(value)
        if not path.exists():
            raise FaceswapError(f"{label} not found: {path}")
        return path

    def _optional_int(self, attr: str) -> int | None:
        value = getattr(self._args, attr, None)
        if value is None or value == "" or value == -1:
            return None
        return int(T.cast(int, value))

    def _sidecar_path(self, alignments: Path) -> Path:
        explicit = getattr(self._args, "sidecar", None)
        if explicit:
            path = Path(explicit)
            if not path.is_file():
                raise FaceswapError(f"FaceQA sidecar not found: {path}")
            return path
        return Path(sidecar_path(str(alignments)))

    def _pose_backfiller(self) -> SpigaPoseBackfiller:
        return SpigaPoseBackfiller(Frames(self._args.frames_dir))

    def _coverage_output_paths(self, alignments: Path) -> tuple[Path, Path]:
        output_json = Path(
            getattr(self._args, "output_json", None)
            or alignments.with_name(f"{alignments.stem}_faceqa_coverage.json")
        )
        output_markdown = Path(
            getattr(self._args, "output_markdown", None)
            or alignments.with_name(f"{alignments.stem}_faceqa_coverage.md")
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        return output_json, output_markdown

    def _duplicates_output_dir(self, alignments: Path) -> Path:
        value = getattr(self._args, "output_dir", None)
        return (
            Path(value) if value else alignments.with_name(f"{alignments.stem}_faceqa_duplicates")
        )

    def _compatibility_output_dir(self, source: Path) -> Path:
        value = getattr(self._args, "output_dir", None)
        out = Path(value) if value else source.parent
        out.mkdir(parents=True, exist_ok=True)
        return out


__all__ = get_module_objects(__name__)
