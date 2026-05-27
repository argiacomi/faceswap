#!/usr/bin/env python3
"""Command line arguments for the FaceQA coverage audit tool."""

from __future__ import annotations

import gettext
import typing as T

from lib.cli.actions import FileFullPaths, SaveFileFullPaths, Slider
from lib.cli.args import FaceSwapArgs
from lib.utils import get_module_objects

_LANG = gettext.translation("tools.faceqa_coverage.cli", localedir="locales", fallback=True)
_ = _LANG.gettext

_HELPTEXT = _("Audit an extracted faceset for FaceQA coverage and training readiness.")


class Faceqa_CoverageArgs(FaceSwapArgs):  # pylint:disable=invalid-name
    """FaceQA coverage CLI arguments."""

    @staticmethod
    def get_info() -> str:
        """Return command information."""
        return _(
            "FaceQA Coverage tool\n"
            "Reads an alignments file and optional FaceQA sidecar, then writes "
            "JSON and Markdown reports describing quality, pose, resolution, "
            "identity, duplicate, mask, and readiness risks."
        )

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """Return the tool argument list."""
        return [
            {
                "opts": ("-a", "--alignments"),
                "action": FileFullPaths,
                "type": str,
                "dest": "alignments",
                "group": _("data"),
                "required": True,
                "filetypes": "alignments",
                "help": _("Path to the alignments (.fsa) file to audit."),
            },
            {
                "opts": ("-s", "--sidecar"),
                "action": FileFullPaths,
                "type": str,
                "dest": "sidecar",
                "group": _("data"),
                "default": None,
                "filetypes": "json",
                "help": _(
                    "Optional FaceQA sidecar JSON. If omitted, the tool looks for "
                    "<alignments_stem>_faceset_qa.json beside the alignments file."
                ),
            },
            {
                "opts": ("-o", "--output-json"),
                "action": SaveFileFullPaths,
                "type": str,
                "dest": "output_json",
                "group": _("output"),
                "default": None,
                "filetypes": "json",
                "help": _(
                    "Path for the machine-readable JSON report. Defaults to "
                    "<alignments_stem>_faceqa_coverage.json."
                ),
            },
            {
                "opts": ("-m", "--output-markdown"),
                "action": SaveFileFullPaths,
                "type": str,
                "dest": "output_markdown",
                "group": _("output"),
                "default": None,
                "filetypes": "markdown",
                "help": _(
                    "Path for the human-readable Markdown report. Defaults to "
                    "<alignments_stem>_faceqa_coverage.md."
                ),
            },
            {
                "opts": ("--exclude-duplicates",),
                "action": "store_true",
                "dest": "exclude_duplicates",
                "default": False,
                "group": _("filters"),
                "help": _(
                    "Exclude duplicate prune candidates from usable_faces. "
                    "Bucket counts still cover the full faceset."
                ),
            },
            {
                "opts": ("--exclude-outliers",),
                "action": "store_true",
                "dest": "exclude_outliers",
                "default": False,
                "group": _("filters"),
                "help": _(
                    "Exclude identity outliers and rejects from usable_faces. "
                    "Bucket counts still cover the full faceset."
                ),
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
                    "Bucket percentage below which coverage buckets are flagged "
                    "as under-represented."
                ),
            },
        ]


__all__ = get_module_objects(__name__)
