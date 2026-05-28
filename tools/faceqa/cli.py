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
    "Unified FaceQA tool: coverage audit (with optional coverage-aware pruning "
    "suggestions) and source-target compatibility scoring."
)


class FaceqaArgs(FaceSwapArgs):  # pylint:disable=invalid-name
    """Unified FaceQA CLI arguments."""

    @staticmethod
    def get_info() -> str:
        return _(
            "FaceQA tool\n"
            "Supports two modes selected via --mode:\n"
            "  coverage       Audit faceset coverage / readiness from alignments.\n"
            "                 Pass --suggest-pruning to add coverage-aware\n"
            "                 representation-redundancy recommendations.\n"
            "  compatibility  Score whether a source faceset can support a target."
        )

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        return [
            {
                "opts": ("--mode",),
                "type": str,
                "dest": "mode",
                "choices": ("coverage", "compatibility"),
                "default": "coverage",
                "group": _("mode"),
                "help": _("Which FaceQA workflow to run."),
            },
            # ----------------------------------------------------------------
            # Coverage-mode inputs
            # ----------------------------------------------------------------
            {
                "opts": ("-a", "--alignments"),
                "action": FileFullPaths,
                "type": str,
                "dest": "alignments",
                "group": _("data"),
                "default": None,
                "filetypes": "alignments",
                "help": _("Alignments (.fsa) for --mode coverage."),
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
                "help": _(
                    "Aligned-face directory. Required only when "
                    "--suggest-pruning is used with --prune-output-dir."
                ),
            },
            # ----------------------------------------------------------------
            # Coverage-mode output
            # ----------------------------------------------------------------
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
            # ----------------------------------------------------------------
            # Coverage filters and thresholds
            # ----------------------------------------------------------------
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
                "help": _(
                    "Bucket %% below which coverage buckets are flagged "
                    "as under-represented. Also drives the protection floor "
                    "used by --suggest-pruning."
                ),
            },
            # ----------------------------------------------------------------
            # Pruning suggestions
            # ----------------------------------------------------------------
            {
                "opts": ("--suggest-pruning",),
                "action": "store_true",
                "dest": "suggest_pruning",
                "default": False,
                "group": _("pruning"),
                "help": _(
                    "Run coverage-aware representation redundancy and emit "
                    "keep/review/prune_candidate recommendations alongside "
                    "coverage."
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
                "opts": ("--prune-output-dir",),
                "action": DirFullPaths,
                "type": str,
                "dest": "prune_output_dir",
                "group": _("pruning"),
                "default": None,
                "help": _(
                    "Directory to write sorted folders (keep/review/"
                    "prune_candidate), CSV/JSONL manifests, and contact "
                    "sheets when --suggest-pruning is enabled. Requires "
                    "--faces-dir."
                ),
            },
            # ----------------------------------------------------------------
            # Compatibility-mode options
            # ----------------------------------------------------------------
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
                "opts": ("--compatibility-output-dir",),
                "action": DirFullPaths,
                "type": str,
                "dest": "compatibility_output_dir",
                "group": _("compatibility"),
                "default": None,
                "help": _(
                    "Output directory for compatibility JSON/Markdown "
                    "reports. Defaults to the source alignments directory."
                ),
            },
        ]


__all__ = get_module_objects(__name__)
