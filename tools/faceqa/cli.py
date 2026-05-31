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
            "Audit faceset coverage/readiness from alignments plus source frames, "
            "optionally embed coverage-aware keep/review/prune_candidate "
            "recommendations in the coverage report, or compare source and target "
            "faceset compatibility."
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
            " Source frames/images or video (-r / --frames-folder) are required for "
            "coverage so FaceQA can reconstruct aligned crops, backfill SPIGA pose, "
            "and fill missing identity embeddings before reporting."
        )
        faces_dir = _(
            " Pass a faces folder (-c / --faces-folder) ONLY with --suggest-pruning plus "
            "--sort-prune or --contact-sheets; coverage and pruning computation do not "
            "need extracted faces."
        )
        output_dir = _(
            " Use the output directory (-o) for the coverage JSON/Markdown report and "
            "optional review artifacts. FaceQA does not write sidecars, standalone "
            "faceqa_redundancy.json, or keep/review/prune CSV/JSONL manifests."
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
                "help": _(
                    "R|Choose which FaceQA workflow to run."
                    "\nL|'coverage': Audit faceset coverage/readiness from an alignments "
                    "file and source frames. Pass --suggest-pruning to add "
                    "coverage-aware keep/review/prune_candidate recommendations inside "
                    "the coverage report; no standalone redundancy JSON or CSV/JSONL "
                    "manifests are written.{0}{2}"
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
                "opts": ("-c", "--faces_folder"),
                "action": DirFullPaths,
                "dest": "faces_dir",
                "group": ("data"),
                "help": _(
                    "Directory containing extracted faces. Required only when "
                    "--suggest-pruning is combined with --sort-prune or "
                    "--contact-sheets. Not used for coverage computation."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("-r", "--frames_folder"),
                "action": DirOrFileFullPaths,
                "dest": "frames_dir",
                "required": False,
                "filetypes": "video",
                "group": _("data"),
                "help": _(
                    "Source frames/images folder or video. Required for coverage so "
                    "FaceQA can compute authoritative frame-derived metrics, SPIGA "
                    "pose backfill, and identity backfill."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("-o", "--output-dir"),
                "action": DirFullPaths,
                "dest": "output_dir",
                "group": _("data"),
                "help": _(
                    "Output directory for FaceQA reports and optional review artifacts. "
                    "Writes <stem>_faceqa_coverage.json/md plus optional pruning folders "
                    "and contact sheets. Does not write sidecars, standalone "
                    "faceqa_redundancy.json, or keep/review/prune CSV/JSONL manifests. "
                    "Defaults to the relevant alignments directory when omitted."
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
                    "report. Pure computation — no extra pruning files are written: no "
                    "standalone faceqa_redundancy.json and no keep/review/prune "
                    "CSV/JSONL manifests. Use --sort-prune or --contact-sheets for "
                    "optional review artifacts."
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
                    "keep/review/prune_candidate folders. By default, moves originals "
                    "into bucket folders inside --faces-dir; with --keep, copies files "
                    "under --output-dir/pruning instead. Requires --suggest-pruning and "
                    "--faces-dir."
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
                    "cluster under --output-dir/pruning/contact_sheets. Requires "
                    "--faces-dir."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--keep",),
                "action": "store_true",
                "dest": "keep_originals",
                "group": _("pruning"),
                "default": False,
                "help": _(
                    "Used with --sort-prune: COPY extracted faces from --faces-dir "
                    "into keep/review/prune_candidate subfolders under "
                    "--output-dir/pruning. By default, --sort-prune moves originals "
                    "inside --faces-dir; --keep is the non-destructive copy mode."
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
                "opts": ("--deep-analysis",),
                "action": Radio,
                "type": str,
                "dest": "deep_analysis",
                "group": _("deep analysis"),
                "default": "none",
                "choices": ("none", "deca"),
                "help": _(
                    "R|Optional deep dataset-quality analysis layered on top of "
                    "'coverage' mode. The default keeps FaceQA lightweight."
                    "\nL|'none': Standard landmark/thumbnail coverage only (default)."
                    "\nL|'deca': Additionally run the DECA encoder to derive 3D "
                    "expression / pose / lighting coefficients and report "
                    "expression-space, lighting-space, latent-entropy and cluster "
                    "coverage, a DECA readiness sub-score, and a landmark-vs-DECA "
                    "comparison inside the coverage report's 'deep_audit' block. "
                    "The DECA checkpoint (deca_model.tar) is downloaded and "
                    "SHA256-validated through the standard faceswap model cache on "
                    "first use."
                ),
            }
        )
        argument_list.append(
            {
                "opts": ("--deep-device",),
                "action": Radio,
                "type": str,
                "dest": "deep_device",
                "group": _("deep analysis"),
                "default": "auto",
                "choices": ("auto", "cuda", "mps", "cpu"),
                "help": _(
                    "R|Torch device for --deep-analysis deca. "
                    "\nL|'auto': use CUDA when available, then Apple MPS, then CPU "
                    "(default)."
                    "\nL|'cuda' / 'mps': require that accelerator and fail clearly "
                    "if unavailable."
                    "\nL|'cpu': force CPU inference."
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
