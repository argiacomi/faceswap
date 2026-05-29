#!/usr/bin/env python3
"""Unified FaceQA tool dispatcher.

Modes:

- ``coverage``: audit faceset coverage / readiness. ``--suggest-pruning``
  layers coverage-aware representation-redundancy recommendations on top.
- ``compatibility``: score source-target faceset compatibility.

The legacy identity-first duplicate mode has been removed in favour of
coverage-integrated redundancy (see :mod:`lib.faceqa.redundancy`).
"""

from __future__ import annotations

import logging
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
    backfill_identity,
    compute_coverage,
    compute_identity_quality,
    identity_coverage_status,
    records_from_alignments,
)
from lib.faceqa.readiness import generate_readiness_report
from lib.faceqa.redundancy import RedundancyReport, compute_redundancy
from lib.faceqa.redundancy_outputs import (
    render_contact_sheets,
    write_manifests,
    write_sorted_folders,
)
from lib.utils import FaceswapError, get_module_objects
from tools.alignments.media import Frames

logger = logging.getLogger(__name__)


class Faceqa:  # pylint:disable=invalid-name
    """Dispatch the requested FaceQA workflow."""

    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._args = arguments

    def process(self) -> None:
        mode = getattr(self._args, "mode", "coverage")
        if mode == "coverage":
            self._run_coverage()
        elif mode == "compatibility":
            self._run_compatibility()
        else:
            raise FaceswapError(
                f"Unknown FaceQA mode '{mode}'. Expected: coverage | compatibility."
            )

    # ------------------------------------------------------------------
    # Coverage audit (+ optional pruning suggestions)
    # ------------------------------------------------------------------

    def _run_coverage(self) -> None:
        alignments = self._require_alignments("alignments")
        min_bucket_pct = float(getattr(self._args, "min_bucket_pct", 5.0))
        suggest_pruning = bool(getattr(self._args, "suggest_pruning", False))

        # Identity is both a coverage signal and the redundancy guardrail. Run
        # embedding backfill before records are built so coverage, readiness,
        # and pruning all observe the same enriched alignments state.
        if suggest_pruning or getattr(self._args, "frames_dir", None):
            self._run_identity_backfill(alignments)

        sidecar = self._sidecar_path(alignments)
        qa_file = load_sidecar(str(sidecar)) if sidecar.is_file() else None

        if qa_file is None:
            logger.info("No FaceQA sidecar found. Deriving metrics from alignments.")
        else:
            logger.info("Loaded %s FaceQA sidecar records from '%s'.", len(qa_file.faces), sidecar)

        backfill_added = False
        pose_backfiller = self._pose_backfiller()
        pose_backfill_callback: (
            T.Callable[[FaceQARecord, FileAlignments], dict[str, T.Any] | None] | None
        ) = None

        if pose_backfiller is not None:

            def _track_pose_backfill(
                record: FaceQARecord,
                face: FileAlignments,
            ) -> dict[str, T.Any] | None:
                nonlocal backfill_added
                pose: dict[str, T.Any] | None = pose_backfiller(record, face)
                if pose is not None:
                    backfill_added = True
                return pose

            pose_backfill_callback = _track_pose_backfill

        records = records_from_alignments(
            alignments,
            qa_file=qa_file,
            pose_backfiller=pose_backfill_callback,
        )

        identity_quality = compute_identity_quality(records, alignments)
        if identity_quality.disabled_reason:
            logger.info(
                "Identity quality classification skipped: %s", identity_quality.disabled_reason
            )
        else:
            logger.info(
                "Identity quality (%s): %d vectors, %d classified "
                "(inlier=%d, borderline=%d, outlier=%d, reject=%d).",
                identity_quality.model,
                identity_quality.vectors_available,
                identity_quality.classified,
                identity_quality.inlier,
                identity_quality.borderline,
                identity_quality.outlier,
                identity_quality.reject,
            )

        if backfill_added or identity_quality.updated:
            save_sidecar(str(sidecar), FaceQAFile(generated_by="faceqa", faces=records))
            logger.info("Persisted FaceQA enrichment metadata to '%s'.", sidecar)

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
            min_bucket_pct=min_bucket_pct,
        )

        if suggest_pruning:
            redundancy = compute_redundancy(
                records,
                coverage=coverage,
                aggressiveness=str(getattr(self._args, "prune_aggressiveness", "balanced")),
                min_bucket_pct=min_bucket_pct,
            )
            report.pruning_suggestions = redundancy.to_dict()
            logger.info(
                "Pruning suggestions (%s): keep=%d, review=%d, prune_candidate=%d.",
                redundancy.aggressiveness,
                redundancy.keep_count,
                redundancy.review_count,
                redundancy.prune_candidate_count,
            )
            self._maybe_write_prune_outputs(redundancy)

        output_json, output_markdown = self._coverage_output_paths(alignments)
        output_json.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
        output_markdown.write_text(report.to_markdown(), encoding="utf-8")
        logger.info(
            "Coverage audit complete: %s total faces, %s usable faces.",
            report.total_faces,
            report.usable_faces,
        )
        logger.info("Wrote JSON: %s", output_json)
        logger.info("Wrote Markdown: %s", output_markdown)

    def _run_identity_backfill(self, alignments: Path) -> None:
        """Backfill missing identity embeddings before coverage reporting.

        Identity is a FaceQA coverage signal and the pruning guardrail. If the
        selected identity model is already complete, no source frames are
        required. Otherwise ``--frames-dir`` is needed to fill missing vectors.
        """
        status = identity_coverage_status(alignments)
        if status.complete:
            logger.info(
                "Identity coverage already complete for '%s' (%d/%d faces).",
                status.model,
                status.available_faces,
                status.total_faces,
            )
            return

        frames_dir = getattr(self._args, "frames_dir", None)
        if not frames_dir:
            raise FaceswapError(
                "--suggest-pruning requires --frames-dir when identity embeddings "
                "are incomplete, so FaceQA can backfill identity before coverage "
                "and redundancy clustering."
            )

        report = backfill_identity(
            alignments,
            frames_loader=Frames(frames_dir),
            model=status.model,
        )
        if report.disabled_reason:
            logger.warning(
                "Identity backfill disabled (%s); coverage/pruning will rely on "
                "the existing identity coverage only.",
                report.disabled_reason,
            )
            return
        logger.info(
            "Identity backfill (%s): %d faces, %d already present, "
            "%d backfilled, %d skipped (frame: %d, failed: %d), persisted=%s.",
            report.model,
            report.total_faces,
            report.already_present,
            report.backfilled,
            report.skipped_no_frame + report.skipped_failed,
            report.skipped_no_frame,
            report.skipped_failed,
            report.persisted,
        )

    def _maybe_write_prune_outputs(self, redundancy: RedundancyReport) -> None:
        prune_dir_value = getattr(self._args, "output_dir", None)
        if not prune_dir_value:
            return
        faces_dir_value = getattr(self._args, "faces_dir", None)
        if not faces_dir_value:
            raise FaceswapError(
                "--prune-output-dir requires --faces-dir so aligned-face images "
                "can be copied into the sorted folders."
            )
        faces_dir = Path(faces_dir_value)
        if not faces_dir.is_dir():
            raise FaceswapError(f"Aligned faces directory not found: {faces_dir}")
        prune_dir = Path(prune_dir_value)
        prune_dir.mkdir(parents=True, exist_ok=True)
        (prune_dir / "faceqa_redundancy.json").write_text(
            redundancy.to_json() + "\n", encoding="utf-8"
        )
        layout = write_sorted_folders(redundancy, faces_dir=faces_dir, output_dir=prune_dir)
        write_manifests(redundancy, prune_dir)
        sheets = render_contact_sheets(
            redundancy, faces_dir=faces_dir, output_dir=layout.contact_sheets_dir
        )
        logger.info(
            "Wrote pruning artefacts to '%s' (%d contact sheets).",
            prune_dir,
            len(sheets),
        )

    # ------------------------------------------------------------------
    # Source-target compatibility
    # ------------------------------------------------------------------

    def _run_compatibility(self) -> None:
        source = self._require_alignments("source_alignments")
        target = self._require_alignments("target_alignments")
        source_coverage = self._coverage_for_compatibility(source, sidecar_arg="source_sidecar")
        target_coverage = self._coverage_for_compatibility(target, sidecar_arg="target_sidecar")
        report = compute_compatibility(
            source_coverage,
            target_coverage,
            source_path=str(source),
            target_path=str(target),
        )
        output_dir = self._compatibility_output_dir(source)
        output_json = output_dir / "source_target_compatibility.json"
        output_markdown = output_dir / "source_target_compatibility.md"
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
    ) -> FacesetCoverageReport:
        sidecar_value = getattr(self._args, sidecar_arg, None)
        sidecar = Path(sidecar_value) if sidecar_value else Path(sidecar_path(str(alignments)))
        qa_file = load_sidecar(str(sidecar)) if sidecar.is_file() else None
        records = records_from_alignments(alignments, qa_file=qa_file)
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

    def _sidecar_path(self, alignments: Path) -> Path:
        explicit = getattr(self._args, "sidecar", None)
        if explicit:
            path = Path(explicit)
            if not path.is_file():
                raise FaceswapError(f"FaceQA sidecar not found: {path}")
            return path
        return Path(sidecar_path(str(alignments)))

    def _pose_backfiller(self) -> SpigaPoseBackfiller | None:
        frames_dir = getattr(self._args, "frames_dir", None)
        if not frames_dir:
            return None
        return SpigaPoseBackfiller(Frames(frames_dir))

    def _coverage_output_paths(self, alignments: Path) -> tuple[Path, Path]:
        output_dir_value = getattr(self._args, "output_dir", None)
        output_dir = Path(output_dir_value) if output_dir_value else alignments.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = alignments.stem
        return (
            output_dir / f"{stem}_faceqa_coverage.json",
            output_dir / f"{stem}_faceqa_coverage.md",
        )

    def _compatibility_output_dir(self, source: Path) -> Path:
        output_dir_value = getattr(self._args, "output_dir", None)
        output_dir = Path(output_dir_value) if output_dir_value else source.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


__all__ = get_module_objects(__name__)
