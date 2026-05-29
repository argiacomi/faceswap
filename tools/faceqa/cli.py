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
            " Source frames folder/video (-r) is required for coverage so FaceQA can "
            "reconstruct aligned crops, backfill SPIGA pose, and fill missing identity "
            "embeddings before reporting."
        )
        faces_dir = _(
            " Pass a faces folder (-c) ONLY when --sort-prune or --contact-sheets is set; "
            "neither coverage nor --suggest-pruning computation needs it."
        )
        output_dir = _(
            " Use the output directory (-o) for the coverage report, contact sheets, and any "
            "sorted pruning folders."
        )

        argument_list: list[dict[str, T.Any]] = []
        argument_list.append(
            {
                "opts": ("--mode",),
                "action": Radio,
                "type": str,
                "dest": "mode",
                "group": _("processing"),
                "choices": ("coverage", "compatibility"),
                "default": "coverage",
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
                "opts": ("-c", "-faces_folder"),
                "action": DirFullPaths,
                "dest": "faces_dir",
                "group": ("data"),
                "help": ("Directory containing extracted faces."),
            }
        )
        argument_list.append(
            {
                "opts": ("-r", "-frames_folder"),
                "action": DirOrFileFullPaths,
                "dest": "frames_dir",
                "required": False,
                "filetypes": "video",
                "group": _("data"),
                "help": _("Directory containing source frames that faces were extracted from."),
            }
        )
        argument_list.append(
            {
                "opts": ("-o", "--output-dir"),
                "action": DirFullPaths,
                "dest": "output_dir",
                "group": _("Data"),
                "help": _(
                    "Output directory. Location to save FaceQA reports, manifests, sorted "
                    "folders, and contact sheets. Defaults to the relevant alignments "
                    "directory when omitted."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--suggest-pruning",),
                "action": "store_true",
                "dest": "suggest_pruning",
                "group": _("pruning"),
                "default": False,
                "help": _(
                    "Compute coverage-aware representation redundancy and embed "
                    "keep/review/prune_candidate recommendations inside the coverage "
                    "report. Pure computation — no extra files are written. Use "
                    "--sort-prune or --contact-sheets for the visual / file artefacts."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--sort-prune",),
                "action": "store_true",
                "dest": "sort_prune",
                "group": _("pruning"),
                "default": False,
                "help": _(
                    "After --suggest-pruning, sort extracted faces into "
                    "keep/review/prune_candidate folders under the output directory "
                    "(or move them inside the faces folder when --keep is disabled). "
                    "Requires --faces-dir."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--contact-sheets",),
                "action": "store_true",
                "dest": "contact_sheets",
                "group": _("pruning"),
                "default": False,
                "help": _(
                    "After --suggest-pruning, render one contact sheet per redundancy "
                    "cluster into the output directory. Requires --faces-dir."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--no-keep",),
                "action": "store_false",
                "dest": "keep_originals",
                "group": _("pruning"),
                "default": True,
                "help": _(
                    "Used with --sort-prune: MOVE original extracted faces from "
                    "--faces-dir into keep/review/prune_candidate subfolders "
                    "instead of copying them into --output-dir. Destructive — "
                    "originals are relocated, not duplicated."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--prune-aggressiveness",),
                "action": Radio,
                "type": str,
                "dest": "prune_aggressiveness",
                "group": _("pruning"),
                "default": "balanced",
                "choices": ("conservative", "balanced", "aggressive"),
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
                "opts": ("--exclude-duplicates",),
                "action": "store_true",
                "dest": "exclude_duplicates",
                "default": False,
                "group": _("coverage"),
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
                "group": _("coverage"),
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
