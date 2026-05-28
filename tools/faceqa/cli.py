#!/usr/bin/env python3
"""Command line arguments for the unified FaceQA tool."""

from __future__ import annotations

import gettext
import typing as T

from lib.cli.actions import (
    DirFullPaths,
    DirOrFileFullPaths,
    FileFullPaths,
    SaveFileFullPaths,
    Slider,
)
from lib.cli.args import FaceSwapArgs
from lib.utils import get_module_objects

_LANG = gettext.translation("tools.faceqa.cli", localedir="locales", fallback=True)
_ = _LANG.gettext

_HELPTEXT = _(
    "Unified FaceQA tool: coverage audit, duplicate clustering, and source-target "
    "compatibility scoring."
)


class FaceqaArgs(FaceSwapArgs):  # pylint:disable=invalid-name
    """Unified FaceQA CLI arguments."""

    @staticmethod
    def get_info() -> str:
        return _(
            "FaceQA tool\n"
            "Supports three modes selected via --mode:\n"
            "  coverage       Audit faceset coverage / readiness from alignments.\n"
            "  duplicates     Detect duplicate clusters, recommend keep/review/prune.\n"
            "  compatibility  Score whether a source faceset can support a target."
        )

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        return [
            {
                "opts": ("--mode",),
                "type": str,
                "dest": "mode",
                "choices": ("coverage", "duplicates", "compatibility"),
                "default": "coverage",
                "group": _("mode"),
                "help": _("Which FaceQA workflow to run."),
            },
            {
                "opts": ("-a", "--alignments"),
                "action": FileFullPaths,
                "type": str,
                "dest": "alignments",
                "group": _("data"),
                "default": None,
                "filetypes": "alignments",
                "help": _("Alignments (.fsa) for coverage and duplicates modes."),
            },
            {
                "opts": ("-s", "--sidecar"),
                "action": FileFullPaths,
                "type": str,
                "dest": "sidecar",
                "group": _("data"),
                "default": None,
                "filetypes": "json",
                "help": _("Optional FaceQA sidecar JSON."),
            },
            {
                "opts": ("-f", "--frames-dir", "--source-images"),
                "action": DirOrFileFullPaths,
                "type": str,
                "dest": "frames_dir",
                "group": _("data"),
                "default": None,
                "filetypes": "video",
                "help": _("Source frames or video for SPIGA pose backfill."),
            },
            {
                "opts": ("--faces-dir",),
                "action": DirFullPaths,
                "type": str,
                "dest": "faces_dir",
                "group": _("data"),
                "default": None,
                "help": _("Aligned-face directory used by --mode duplicates."),
            },
            {
                "opts": ("-o", "--output-json"),
                "action": SaveFileFullPaths,
                "type": str,
                "dest": "output_json",
                "group": _("output"),
                "default": None,
                "filetypes": "json",
                "help": _("JSON coverage report path."),
            },
            {
                "opts": ("-m", "--output-markdown"),
                "action": SaveFileFullPaths,
                "type": str,
                "dest": "output_markdown",
                "group": _("output"),
                "default": None,
                "filetypes": "markdown",
                "help": _("Markdown coverage report path."),
            },
            {
                "opts": ("--output-dir",),
                "action": DirFullPaths,
                "type": str,
                "dest": "output_dir",
                "group": _("output"),
                "default": None,
                "help": _("Output directory for duplicates and compatibility modes."),
            },
            {
                "opts": ("--output-prefix",),
                "type": str,
                "dest": "output_prefix",
                "group": _("output"),
                "default": "source_target_compatibility",
                "help": _("Filename stem for compatibility report artifacts."),
            },
            {
                "opts": ("--exclude-duplicates",),
                "action": "store_true",
                "dest": "exclude_duplicates",
                "default": False,
                "group": _("filters"),
                "help": _("Exclude duplicate prune candidates from usable_faces."),
            },
            {
                "opts": ("--exclude-outliers",),
                "action": "store_true",
                "dest": "exclude_outliers",
                "default": False,
                "group": _("filters"),
                "help": _("Exclude identity outliers and rejects from usable_faces."),
            },
            {
                "opts": ("-p", "--min-bucket-pct"),
                "action": Slider,
                "type": float,
                "dest": "min_bucket_pct",
                "default": 5.0,
                "min_max": (0.0, 50.0),
                "rounding": 1,
                "group": _("thresholds"),
                "help": _("Bucket %% below which coverage buckets are flagged."),
            },
            {
                "opts": ("--suggest-pruning",),
                "action": "store_true",
                "dest": "suggest_pruning",
                "default": False,
                "group": _("pruning"),
                "help": _(
                    "Run coverage-aware representation redundancy and emit "
                    "keep/review/prune_candidate recommendations alongside coverage."
                ),
            },
            {
                "opts": ("--prune-aggressiveness",),
                "type": str,
                "dest": "prune_aggressiveness",
                "default": "balanced",
                "choices": ("conservative", "balanced", "aggressive"),
                "group": _("pruning"),
                "help": _(
                    "Pruning aggressiveness preset. Conservative prunes only "
                    "obvious duplicates; aggressive prunes wider redundancy."
                ),
            },
            {
                "opts": ("--identity-model",),
                "type": str,
                "dest": "identity_model",
                "group": _("duplicates"),
                "default": None,
                "help": _("Identity embedding model key (auto-detected when omitted)."),
            },
            {
                "opts": ("--similarity-threshold",),
                "action": Slider,
                "type": float,
                "dest": "similarity_threshold",
                "default": 0.85,
                "min_max": (0.5, 0.999),
                "rounding": 3,
                "group": _("duplicates"),
                "help": _("Cosine threshold for grouping duplicates."),
            },
            {
                "opts": ("--temporal-window",),
                "action": Slider,
                "type": int,
                "dest": "temporal_window",
                "default": -1,
                "min_max": (-1, 1000),
                "rounding": 1,
                "group": _("duplicates"),
                "help": _("Temporal frame window for duplicate pairing (-1 = disabled)."),
            },
            {
                "opts": ("--duplicates-report",),
                "action": FileFullPaths,
                "type": str,
                "dest": "duplicates_report",
                "group": _("duplicates"),
                "default": None,
                "filetypes": "json",
                "help": _("Reuse an existing duplicate report JSON instead of recomputing."),
            },
            {
                "opts": ("--symlink",),
                "action": "store_true",
                "dest": "symlink",
                "default": False,
                "group": _("duplicates"),
                "help": _("Symlink aligned faces into sorted folders rather than copy."),
            },
            {
                "opts": ("--contact-sheet-tile-size",),
                "action": Slider,
                "type": int,
                "dest": "contact_sheet_tile_size",
                "default": 256,
                "min_max": (64, 1024),
                "rounding": 32,
                "group": _("duplicates"),
                "help": _("Per-face tile size for contact sheets."),
            },
            {
                "opts": ("--source-alignments",),
                "action": FileFullPaths,
                "type": str,
                "dest": "source_alignments",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "alignments",
                "help": _("Source faceset alignments for --mode compatibility."),
            },
            {
                "opts": ("--target-alignments",),
                "action": FileFullPaths,
                "type": str,
                "dest": "target_alignments",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "alignments",
                "help": _("Target faceset alignments for --mode compatibility."),
            },
            {
                "opts": ("--source-sidecar",),
                "action": FileFullPaths,
                "type": str,
                "dest": "source_sidecar",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "json",
                "help": _("Optional source FaceQA sidecar JSON."),
            },
            {
                "opts": ("--target-sidecar",),
                "action": FileFullPaths,
                "type": str,
                "dest": "target_sidecar",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "json",
                "help": _("Optional target FaceQA sidecar JSON."),
            },
            {
                "opts": ("--source-frames-dir",),
                "action": DirOrFileFullPaths,
                "type": str,
                "dest": "source_frames_dir",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "video",
                "help": _("Optional source frames directory for SPIGA pose backfill."),
            },
            {
                "opts": ("--target-frames-dir",),
                "action": DirOrFileFullPaths,
                "type": str,
                "dest": "target_frames_dir",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "video",
                "help": _("Optional target frames directory for SPIGA pose backfill."),
            },
        ]


__all__ = get_module_objects(__name__)
