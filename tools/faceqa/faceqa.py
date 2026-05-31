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
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from lib.align.objects import FileAlignments
from lib.faceqa.compatibility import compute_compatibility
from lib.faceqa.coverage import (
    FacesetCoverageReport,
    FrameImageMetricsBackfiller,
    IdentityBackfillReport,
    SpigaPoseBackfiller,
    backfill_identity_entries,
    compute_coverage,
    compute_identity_quality,
    load_alignments_envelope,
    records_from_alignments,
    save_alignments_envelope,
)
from lib.faceqa.readiness import generate_readiness_report, integrate_deep_readiness
from lib.faceqa.record import FaceQARecord
from lib.faceqa.redundancy import RedundancyReport, compute_redundancy
from lib.faceqa.redundancy_outputs import (
    render_contact_sheets,
    write_sorted_folders,
)
from lib.utils import FaceswapError, get_module_objects
from tools.alignments.media import Frames

logger = logging.getLogger(__name__)


@contextmanager
def _faceqa_progress(*, total: int, desc: str, unit: str):
    """Yield a ``tqdm.update`` callable for one FaceQA stage.

    Wrapping every stage in this helper means:

    * The GUI ``ProgressParser`` consumes determinate FaceQA progress without
      any GUI-specific protocol code (see issue #187).
    * Stages with zero work (e.g. identity backfill when coverage is already
      complete) skip the bar entirely so the CLI output stays tidy.
    * A FaceQA stage failure does not leak an open ``tqdm`` instance.
    """
    if total <= 0:
        yield lambda _n=1: None
        return
    bar = tqdm(total=total, desc=desc, unit=unit, leave=False)
    try:
        yield bar.update
    finally:
        bar.close()


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
        sort_prune = bool(getattr(self._args, "sort_prune", False))
        contact_sheets = bool(getattr(self._args, "contact_sheets", False))

        if not getattr(self._args, "frames_dir", None):
            raise FaceswapError(
                "--frames-dir is required for FaceQA coverage: aligned crops, "
                "SPIGA pose backfill, and identity embedding backfill all need "
                "the source frames to produce a complete report."
            )
        if (sort_prune or contact_sheets) and not suggest_pruning:
            raise FaceswapError(
                "--sort-prune / --contact-sheets only emit artefacts for "
                "pruning recommendations; enable --suggest-pruning to compute "
                "them first."
            )
        if (sort_prune or contact_sheets) and not getattr(self._args, "faces_dir", None):
            raise FaceswapError(
                "--sort-prune / --contact-sheets require --faces-dir so the "
                "extracted aligned-face images can be sorted or rendered."
            )

        # Build the shared frames loader ONCE per coverage run and publish
        # it on ``self`` so every helper that ``_run_coverage`` invokes
        # (``_pose_backfiller`` / ``_metrics_backfiller`` /
        # ``_run_identity_backfill``) reuses the SAME instance — the folder
        # / video index is walked once instead of three times (issue #196).
        self._use_frames_loader(self._frames_loader())

        raw_envelope, entries = load_alignments_envelope(alignments)

        # Identity is both a coverage signal and the redundancy guardrail. Run
        # embedding backfill against the same in-memory envelope that later
        # FaceQA enrichment stages mutate, then commit everything once.
        identity_backfill = self._run_identity_backfill(entries)
        # Bind ``total_faces`` ONCE — reused by every per-stage tqdm bar
        # below and by the sort-prune progress total (issue #196).
        total_faces = sum(len(entry.faces) for entry in entries.values())

        pose_backfill_added = False
        pose_backfiller = self._pose_backfiller()
        pose_backfill_callback: (
            T.Callable[[FaceQARecord, FileAlignments], dict[str, T.Any] | None] | None
        ) = None

        if pose_backfiller is not None:

            def _track_pose_backfill(
                record: FaceQARecord,
                face: FileAlignments,
            ) -> dict[str, T.Any] | None:
                nonlocal pose_backfill_added
                pose: dict[str, T.Any] | None = pose_backfiller(record, face)
                if pose is not None:
                    pose_backfill_added = True
                return pose

            pose_backfill_callback = _track_pose_backfill

        image_metrics_changed = False

        def _track_image_metrics_change() -> None:
            nonlocal image_metrics_changed
            image_metrics_changed = True

        metrics_backfiller = self._metrics_backfiller()
        with _faceqa_progress(total=total_faces, desc="FaceQA metrics", unit="face") as tick:
            records = records_from_alignments(
                entries,
                pose_backfiller=pose_backfill_callback,
                metrics_backfiller=metrics_backfiller,
                progress_callback=tick,
                metadata_change_callback=_track_image_metrics_change,
            )

        if metrics_backfiller is not None and metrics_backfiller.disabled_reason:
            logger.warning(
                "Frame image-metrics backfill disabled mid-run (%s); remaining "
                "faces fall back to alignments thumbnails for blur/lighting.",
                metrics_backfiller.disabled_reason,
            )

        with _faceqa_progress(
            total=total_faces, desc="FaceQA identity quality", unit="face"
        ) as tick:
            identity_quality = compute_identity_quality(records, entries, progress_callback=tick)
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

        # Commit once, after every enrichment stage has completed successfully.
        if (
            (identity_backfill.backfilled > 0 and identity_backfill.disabled_reason is None)
            or pose_backfill_added
            or image_metrics_changed
            or identity_quality.updated
        ):
            save_alignments_envelope(alignments, raw_envelope, entries)
            logger.info(
                "Persisted FaceQA enrichment (pose/identity/image_metrics) into alignments '%s'.",
                alignments,
            )

        coverage = compute_coverage(
            records,
            exclude_duplicates=bool(getattr(self._args, "exclude_duplicates", False)),
            exclude_outliers=bool(getattr(self._args, "exclude_outliers", False)),
            entries=entries,
        )

        report = generate_readiness_report(
            coverage,
            alignments=str(alignments),
            min_bucket_pct=min_bucket_pct,
        )

        deep_pruning_signals = self._run_deep_analysis(entries, report, total_faces=total_faces)

        output_dir = self._output_dir(alignments)

        if suggest_pruning:
            pair_count = len(records) * (len(records) - 1) // 2
            with _faceqa_progress(total=pair_count, desc="FaceQA redundancy", unit="pair") as tick:
                redundancy = compute_redundancy(
                    records,
                    coverage=coverage,
                    aggressiveness=str(getattr(self._args, "prune_aggressiveness", "balanced")),
                    min_bucket_pct=min_bucket_pct,
                    deep_pruning_signals=deep_pruning_signals,
                    progress_callback=tick,
                )
            report.pruning_suggestions = redundancy.to_dict()
            logger.info(
                "Pruning suggestions (%s): keep=%d, review=%d, prune_candidate=%d.",
                redundancy.aggressiveness,
                redundancy.keep_count,
                redundancy.review_count,
                redundancy.prune_candidate_count,
            )
            self._emit_pruning_artifacts(redundancy, output_dir)

        output_json, output_markdown = self._coverage_output_paths(alignments, output_dir)
        output_json.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
        output_markdown.write_text(report.to_markdown(), encoding="utf-8")
        logger.info(
            "Coverage audit complete: %s total faces, %s usable faces.",
            report.total_faces,
            report.usable_faces,
        )
        logger.info("Wrote JSON: %s", output_json)
        logger.info("Wrote Markdown: %s", output_markdown)

    def _run_deep_analysis(
        self,
        entries: dict[str, T.Any],
        report: T.Any,
        *,
        total_faces: int,
    ) -> dict[tuple[str, int], dict[str, T.Any]] | None:
        """Optionally run the DECA deep audit and attach it to the report.

        Gated on ``--deep-analysis deca``; when ``none`` (the default) this is
        a no-op so the standard coverage workflow stays lightweight and never
        imports torch or the DECA package. The DECA encoder, metrics, and
        landmark-vs-DECA comparison live in :mod:`lib.faceqa.deep`.
        """
        deep_analysis = str(getattr(self._args, "deep_analysis", "none"))
        if deep_analysis != "deca":
            return None

        # Imported here (not at module top) so the default lightweight path
        # pays no torch / DECA import cost.
        from lib.faceqa.deep.audit import build_encoder, run_deep_audit
        from lib.faceqa.deep.weights import default_weights_path

        frames_loader = self._frames_loader()
        if frames_loader is None:
            raise FaceswapError(
                "--frames-dir is required for --deep-analysis deca so FaceQA can "
                "reconstruct aligned crops for the DECA encoder."
            )

        # ``build_encoder`` downloads and SHA256-validates the DECA checkpoint
        # through the standard faceswap model cache when it is not present.
        encoder = build_encoder()

        with _faceqa_progress(
            total=total_faces, desc="FaceQA deep audit (DECA)", unit="face"
        ) as tick:
            deep_report = run_deep_audit(
                entries,
                encoder=encoder,
                frames_loader=frames_loader,
                readiness_scores=report.readiness_scores,
                weights_path=default_weights_path(),
                progress_callback=tick,
            )
        report.deep_audit = deep_report.to_dict()
        integrate_deep_readiness(report)
        logger.info(
            "DECA deep audit: %d/%d faces encoded, status=%s, readiness=%s.",
            deep_report.faces_encoded,
            deep_report.faces_total,
            deep_report.status,
            deep_report.deca_readiness.get("score") if deep_report.deca_readiness else None,
        )
        return {
            (str(signal["frame"]), int(signal["face_index"])): signal
            for signal in deep_report.pruning_signals
        }

    def _run_identity_backfill(
        self,
        entries: dict[str, T.Any],
    ) -> IdentityBackfillReport:
        """Backfill missing identity embeddings into pre-loaded alignments entries."""
        total_faces = sum(len(entry.faces) for entry in entries.values())
        frames_loader = self._frames_loader()
        if frames_loader is None:
            raise FaceswapError(
                "--frames-dir is required when identity embeddings are incomplete, "
                "so FaceQA can backfill identity before coverage and redundancy "
                "clustering."
            )

        with _faceqa_progress(
            total=total_faces,
            desc="FaceQA identity backfill",
            unit="face",
        ) as tick:
            report = backfill_identity_entries(
                entries,
                frames_loader=frames_loader,
                progress_callback=tick,
            )

        if report.disabled_reason:
            logger.warning(
                "Identity backfill disabled (%s); coverage/pruning will rely on "
                "the existing identity coverage only.",
                report.disabled_reason,
            )
            return report

        if report.backfilled == 0 and report.already_present == report.total_faces:
            logger.info(
                "Identity coverage already complete for '%s' (%d/%d faces).",
                report.model,
                report.already_present,
                report.total_faces,
            )
            return report

        logger.info(
            "Identity backfill (%s): %d faces, %d already present, "
            "%d backfilled, %d skipped (frame: %d, failed: %d), pending alignments commit.",
            report.model,
            report.total_faces,
            report.already_present,
            report.backfilled,
            report.skipped_no_frame + report.skipped_failed,
            report.skipped_no_frame,
            report.skipped_failed,
        )
        return report

    def _emit_pruning_artifacts(
        self,
        redundancy: RedundancyReport,
        output_dir: Path,
    ) -> None:
        """Optionally write the sort-prune folders and / or contact sheets.

        Coverage JSON is the single machine-readable source of truth for the
        recommendations (see ``report.pruning_suggestions``). This method
        materialises *visual* artefacts on top of that.
        """
        sort_prune = bool(getattr(self._args, "sort_prune", False))
        contact_sheets = bool(getattr(self._args, "contact_sheets", False))
        if not (sort_prune or contact_sheets):
            return

        faces_dir = Path(self._args.faces_dir)
        if not faces_dir.is_dir():
            raise FaceswapError(f"Aligned faces directory not found: {faces_dir}")

        pruning_dir = output_dir / "pruning"
        pruning_dir.mkdir(parents=True, exist_ok=True)

        if sort_prune:
            # Default sort-prune is MOVE (destructive: originals are
            # relocated into bucket subdirs of faces_dir). ``--keep`` opts
            # into COPY mode (non-destructive: faces are duplicated into
            # pruning/ under output_dir and the source folder is untouched).
            # ``keep_originals`` thus defaults to ``False`` when missing.
            keep_originals = bool(getattr(self._args, "keep_originals", False))
            target_root = pruning_dir if keep_originals else faces_dir
            with _faceqa_progress(
                total=len(redundancy.records),
                desc="FaceQA sort-prune",
                unit="face",
            ) as tick:
                write_sorted_folders(
                    redundancy,
                    faces_dir=faces_dir,
                    output_dir=target_root,
                    copy=keep_originals,
                    progress_callback=tick,
                )
            logger.info(
                "Sort-prune: %s aligned faces into '%s' (keep=%s).",
                "copied" if keep_originals else "moved",
                target_root,
                keep_originals,
            )

        if contact_sheets:
            sheets_dir = pruning_dir / "contact_sheets"
            multi_face_clusters = sum(
                1
                for record in redundancy.records
                if record.representative and record.cluster_size > 1
            )
            with _faceqa_progress(
                total=multi_face_clusters,
                desc="FaceQA contact sheets",
                unit="sheet",
            ) as tick:
                sheets = render_contact_sheets(
                    redundancy,
                    faces_dir=faces_dir,
                    output_dir=sheets_dir,
                    progress_callback=tick,
                )
            logger.info(
                "Contact sheets: rendered %d cluster sheet(s) under '%s'.",
                len(sheets),
                sheets_dir,
            )

    # ------------------------------------------------------------------
    # Source-target compatibility
    # ------------------------------------------------------------------

    def _run_compatibility(self) -> None:
        source = self._require_alignments("source_alignments")
        target = self._require_alignments("target_alignments")
        source_coverage = self._coverage_for_compatibility(
            source, desc="FaceQA compatibility source"
        )
        target_coverage = self._coverage_for_compatibility(
            target, desc="FaceQA compatibility target"
        )
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
        desc: str = "FaceQA compatibility",
    ) -> FacesetCoverageReport:
        # Compatibility is read-only on alignments; the envelope is
        # intentionally discarded since no enrichment is triggered for this
        # faceset (issue #196).
        _envelope, entries = load_alignments_envelope(alignments)
        total_faces = sum(len(entry.faces) for entry in entries.values())
        with _faceqa_progress(total=total_faces, desc=desc, unit="face") as tick:
            records = records_from_alignments(entries, progress_callback=tick)
        return compute_coverage(
            records,
            exclude_duplicates=bool(getattr(self._args, "exclude_duplicates", False)),
            exclude_outliers=bool(getattr(self._args, "exclude_outliers", False)),
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

    def _frames_loader(self) -> Frames | None:
        """Return the shared :class:`Frames` loader for this FaceQA run.

        ``_run_coverage`` calls this once at the top of the run and caches
        the result on ``self`` via :meth:`_use_frames_loader`. Subsequent
        calls from ``_pose_backfiller`` / ``_metrics_backfiller`` /
        ``_run_identity_backfill`` therefore reuse the SAME loader, walking
        the folder / video index once instead of three times (issue #196).

        Returns ``None`` when ``--frames-dir`` is missing.
        """
        cached = getattr(self, "_shared_frames_loader", None)
        if cached is not None:
            return cached  # type: ignore[no-any-return]
        frames_dir = getattr(self._args, "frames_dir", None)
        if not frames_dir:
            return None
        return Frames(frames_dir)

    def _use_frames_loader(self, loader: Frames | None) -> None:
        """Publish a pre-built loader so subsequent ``_frames_loader()`` calls
        reuse it instead of constructing fresh instances (issue #196)."""
        self._shared_frames_loader = loader

    def _pose_backfiller(self) -> SpigaPoseBackfiller | None:
        loader = self._frames_loader()
        if loader is None:
            return None
        return SpigaPoseBackfiller(loader)

    def _metrics_backfiller(self) -> FrameImageMetricsBackfiller | None:
        loader = self._frames_loader()
        if loader is None:
            return None
        return FrameImageMetricsBackfiller(loader)

    def _output_dir(self, alignments: Path) -> Path:
        """Resolve the single FaceQA output directory."""
        output_dir_value = getattr(self._args, "output_dir", None)
        output_dir = Path(output_dir_value) if output_dir_value else alignments.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _coverage_output_paths(self, alignments: Path, output_dir: Path) -> tuple[Path, Path]:
        stem = alignments.stem
        return (
            output_dir / f"{stem}_faceqa_coverage.json",
            output_dir / f"{stem}_faceqa_coverage.md",
        )

    def _compatibility_output_dir(self, source: Path) -> Path:
        return self._output_dir(source)


__all__ = get_module_objects(__name__)
