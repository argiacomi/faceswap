#!/usr/bin/env python3
"""Command line arguments for the unified FaceQA tool."""

from __future__ import annotations

import gettext
import typing as T

from lib.cli.actions import (
    DirFullPaths,
    DirOrFileFullPaths,
    FileFullPaths,
    Radio,
    Slider,
)
from lib.cli.args import FaceSwapArgs
from lib.utils import get_module_objects

# LOCALES
_LANG = gettext.translation("tools.faceqa.cli", localedir="locales", fallback=True)
_ = _LANG.gettext


_HELPTEXT = _(
    "Unified FaceQA tool: coverage audit with optional coverage-aware pruning "
    "suggestions, and source-target compatibility scoring."
)


class FaceqaArgs(FaceSwapArgs):  # pylint:disable=invalid-name
    """Class to parse the command line arguments for FaceQA."""

    @staticmethod
    def get_info() -> str:
        """Obtain command information.

        Returns
        -------
        str
            The help text for displaying in argparse's help output.
        """
        return _(
            "FaceQA tool\n"
            "Audit faceset coverage/readiness, optionally emit coverage-aware "
            "representation-redundancy pruning recommendations, or compare "
            "source and target faceset compatibility."
        )

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """Collect the argparse argument options.

        Returns
        -------
        list[dict[str, typing.Any]]
            The argparse command line options for processing by argparse.
        """
        frames_dir = _(
            " Original frames/images (-r) required for FaceQA backfill "
            "of any missing pose or identity data before coverage is reported."
        )
        faces_dir = _(
            " Faces folder (-c) only required when when --sort-prune or --contact-sheets are set."
        )
        output_dir = _(
            " Output directory (-o) for coverage/compatibility reports and pruning artifacts."
        )

        argument_list: list[dict[str, T.Any]] = []
        argument_list.append(
            {
                "opts": ("--mode",),
                "action": Radio,
                "type": str,
                "dest": "mode",
                "choices": ("coverage", "compatibility"),
                "default": "coverage",
                "group": _("processing"),
                "help": _(
                    "R|Choose which FaceQA workflow to run."
                    "\nL|'coverage': Audit faceset coverage/readiness from an alignments "
                    "file and source frames. Pass --suggest-pruning to add "
                    "coverage-aware keep/review/prune_candidate recommendations.{0}{2}"
                    "\nL|'compatibility': Score whether a source faceset can support a "
                    "target faceset.{2}"
                ).format(frames_dir, faces_dir, output_dir),
            }
        )
        argument_list.append(
            {
                "opts": ("-a", "--alignments"),
                "action": FileFullPaths,
                "type": str,
                "dest": "alignments",
                "group": _("data"),
                "default": None,
                "filetypes": "alignments",
                "help": _(
                    "Alignments (.fsa) for 'coverage' mode. Required when "
                    "--mode coverage is selected."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("-r", "-frames_folder", "--frames-dir"),
                "action": DirOrFileFullPaths,
                "type": str,
                "dest": "frames_dir",
                "filetypes": "video",
                "group": _("data"),
                "default": None,
                "help": _(
                    "Directory containing source frames, or a source video file, that "
                    "the faces were extracted from. Required for 'coverage' mode so "
                    "SPIGA pose and missing identity embeddings can be backfilled "
                    "before reporting."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("-c", "-faces_folder", "--faces-dir"),
                "action": DirFullPaths,
                "type": str,
                "dest": "faces_dir",
                "group": _("data"),
                "default": None,
                "help": _(
                    "Directory containing extracted/aligned faces. Not required for "
                    "coverage metrics; only used when --suggest-pruning writes sorted "
                    "folders or contact sheets."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("-o", "--output-dir"),
                "action": DirFullPaths,
                "type": str,
                "dest": "output_dir",
                "group": _("output"),
                "default": None,
                "help": _(
                    "Single output directory for FaceQA JSON/Markdown reports and, "
                    "when --suggest-pruning is enabled, pruning manifests, sorted "
                    "folders and contact sheets. Defaults to the relevant alignments "
                    "directory when omitted."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--exclude-duplicates",),
                "action": "store_true",
                "dest": "exclude_duplicates",
                "default": False,
                "group": _("filters"),
                "help": _(
                    "Exclude duplicate prune candidates from usable_faces after "
                    "coverage-aware pruning recommendations have been generated."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--exclude-outliers",),
                "action": "store_true",
                "dest": "exclude_outliers",
                "default": False,
                "group": _("filters"),
                "help": _("Exclude identity outliers and rejects from usable_faces."),
            }
        )
        argument_list.append(
            {
                "opts": ("-p", "--min-bucket-pct"),
                "action": Slider,
                "type": float,
                "dest": "min_bucket_pct",
                "default": 5.0,
                "min_max": (0.0, 50.0),
                "rounding": 1,
                "group": _("coverage"),
                "help": _(
                    "Bucket %% below which coverage buckets are flagged as "
                    "under-represented. Also drives the coverage protection floor "
                    "used by --suggest-pruning."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--suggest-pruning",),
                "action": "store_true",
                "dest": "suggest_pruning",
                "default": False,
                "group": _("pruning"),
                "help": _(
                    "Run coverage-aware representation redundancy and emit "
                    "keep/review/prune_candidate recommendations alongside the "
                    "coverage report. Sorted folders/contact sheets are written only "
                    "when a faces folder (-c) is supplied."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--prune-aggressiveness",),
                "action": Radio,
                "type": str,
                "dest": "prune_aggressiveness",
                "default": "balanced",
                "choices": ("conservative", "balanced", "aggressive"),
                "group": _("pruning"),
                "help": _(
                    "R|Pruning aggressiveness preset."
                    "\nL|'conservative': Prune only obvious redundant surplus."
                    "\nL|'balanced': Default coverage-aware pruning policy."
                    "\nL|'aggressive': Prune wider representation redundancy while "
                    "preserving effective coverage."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--source-alignments",),
                "action": FileFullPaths,
                "type": str,
                "dest": "source_alignments",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "alignments",
                "help": _(
                    "Source faceset alignments for 'compatibility' mode. Required "
                    "when --mode compatibility is selected."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--target-alignments",),
                "action": FileFullPaths,
                "type": str,
                "dest": "target_alignments",
                "group": _("compatibility"),
                "default": None,
                "filetypes": "alignments",
                "help": _(
                    "Target faceset alignments for 'compatibility' mode. Required "
                    "when --mode compatibility is selected."
                ),
            }
        )
        return argument_list


__all__ = get_module_objects(__name__)
