#!/usr/bin/env python3
"""Build landmark quality dataset manifests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.landmarks.datasets import (
    SUPPORTED_DATASETS,
    audit_existing_manifest,
    build_cofw_manifest,
    build_directory_manifest,
    build_manifest,
    build_wflw_manifest,
)
from lib.landmarks.datasets.aflw2000_3d import build_aflw2000_3d_manifest
from lib.landmarks.datasets.cofw68 import resolve_cofw68_json
from lib.landmarks.datasets.merl_rav import build_merl_rav_manifest
from lib.landmarks.datasets.polish import polish_landmark_dataset_artifacts
from lib.landmarks.datasets.sources import DEFAULT_CACHE_DIR, resolve_wflw_official_source
from lib.landmarks.datasets.visual_overlays import write_indexed_region_overlays
from lib.landmarks.datasets.w300 import build_300w_manifest

logger = logging.getLogger(__name__)
CLI_SUPPORTED_DATASETS = tuple(dict.fromkeys((*SUPPORTED_DATASETS, "300w")))


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated CLI value."""
    if value is None:
        return None
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or None


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the landmark dataset builder argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=CLI_SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Explicit extracted dataset/source directory. Overrides cache and download.",
    )
    parser.add_argument(
        "--source-zip",
        default=None,
        help="Explicit dataset archive. Overrides cache and download.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", default="default")
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated condition/scenario groups to include. "
        "If omitted, all dataset-provided conditions are kept.",
    )
    parser.add_argument(
        "--samples-per-scenario",
        type=int,
        default=None,
        help="Maximum samples to keep per scenario. Omit or pass 0 to keep all.",
    )
    parser.add_argument(
        "--manifest-mode",
        choices=("replace", "merge"),
        default="replace",
        help="Replace manifest contents or merge this dataset/scenario "
        "slice into an existing manifest.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow the same source image to appear in multiple scenario groups.",
    )
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        help="Write GT, indexed, and region visual landmark overlays when source "
        "images are readable.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate and rewrite dataset_audit.json for an existing manifest "
        "without regenerating data.",
    )
    parser.add_argument("--wflw-annotations", default=None)
    parser.add_argument(
        "--wflw-download-official",
        action="store_true",
        help="Download official WFLW annotations and images into the dataset cache.",
    )
    parser.add_argument(
        "--cofw-json",
        default=None,
        help="Explicit COFW-68 JSON export. If omitted, --dataset cofw resolves "
        "or builds .fs_cache/landmark_quality/cofw/cofw_68.json from the "
        "Caltech COFW color archive and the COFW-68 annotation benchmark.",
    )
    parser.add_argument(
        "--aflw-source-dir",
        default=None,
        help="Explicit AFLW image directory for MERL-RAV label matching.",
    )
    parser.add_argument(
        "--aflw-source-zip",
        default=None,
        help="Explicit AFLW image archive for MERL-RAV label matching.",
    )
    parser.add_argument(
        "--aflw-download-url",
        default=None,
        help="Override AFLW image archive URL for MERL-RAV label matching.",
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Standard dataset cache directory used before attempting download.",
    )
    parser.add_argument(
        "--download-url",
        default=None,
        help="Override dataset download URL when cache/source args are missing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh cached downloads instead of reusing cached archives.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable network fallback. Cache or explicit sources must exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate source-mode combinations before building a manifest."""
    merl_aflw_args = (args.aflw_source_dir, args.aflw_source_zip, args.aflw_download_url)
    if args.dataset != "merl-rav" and any(merl_aflw_args):
        raise ValueError(
            "--aflw-source-dir/--aflw-source-zip/--aflw-download-url are only valid with --dataset merl-rav"
        )
    if args.audit_only:
        if any(
            (
                args.wflw_annotations,
                args.wflw_download_official,
                args.cofw_json,
                args.source_dir,
                args.source_zip,
                *merl_aflw_args,
            )
        ):
            raise ValueError(
                "--audit-only only accepts --dataset, --output-dir, "
                "--allow-overlap, --write-overlays, and logging options"
            )
        if args.no_download or args.force_download or args.download_url:
            raise ValueError("--audit-only does not use download or cache source options")
        return

    source_modes = [
        bool(args.wflw_annotations),
        bool(args.wflw_download_official),
        bool(args.cofw_json),
        bool(args.source_dir),
        bool(args.source_zip),
    ]
    if sum(source_modes) > 1:
        raise ValueError(
            "Pass only one source mode: --wflw-annotations, "
            "--wflw-download-official, --cofw-json, --source-dir, or --source-zip"
        )
    aflw_source_modes = [bool(value) for value in merl_aflw_args]
    if sum(aflw_source_modes) > 1:
        raise ValueError(
            "Pass only one AFLW image source mode: --aflw-source-dir, --aflw-source-zip, or --aflw-download-url"
        )
    if args.dataset == "wflw" and args.cofw_json:
        raise ValueError("--cofw-json is only valid with --dataset cofw")
    if args.dataset != "cofw" and args.cofw_json:
        raise ValueError("--cofw-json is only valid with --dataset cofw")
    if args.dataset == "cofw" and (args.wflw_annotations or args.wflw_download_official):
        raise ValueError("WFLW source options are only valid with --dataset wflw")
    if args.dataset != "wflw" and args.wflw_download_official:
        raise ValueError("--wflw-download-official is only valid with --dataset wflw")
    if args.dataset == "directory":
        if not args.source_dir:
            raise ValueError("--dataset directory requires --source-dir")
        if (
            args.source_zip
            or args.wflw_annotations
            or args.wflw_download_official
            or args.cofw_json
        ):
            raise ValueError("--dataset directory only supports --source-dir")
    if args.dataset in {"merl-rav", "aflw2000-3d", "300w"} and (
        args.wflw_annotations or args.wflw_download_official or args.cofw_json
    ):
        raise ValueError(f"--dataset {args.dataset} only supports --source-dir/--source-zip")
    if args.no_download and args.force_download:
        raise ValueError("--no-download and --force-download cannot be combined")


def _wflw_source_dir(args: argparse.Namespace) -> str | None:
    """Return an explicit WFLW source dir, resolving official multipart downloads."""
    if not args.wflw_download_official:
        return args.source_dir
    return str(
        resolve_wflw_official_source(
            cache_dir=args.cache_dir,
            force_download=args.force_download,
            no_download=args.no_download,
        )
    )


def _cofw_json(args: argparse.Namespace) -> str | None:
    """Return an explicit or auto-materialized COFW-68 JSON export path."""
    if args.cofw_json or args.source_dir or args.source_zip or args.download_url:
        return args.cofw_json
    return str(
        resolve_cofw68_json(
            cache_dir=args.cache_dir,
            force_download=args.force_download,
            no_download=args.no_download,
        )
    )


def _polish_output(args: argparse.Namespace) -> None:
    """Write donor-parity audit/source notes and optional indexed/region overlays."""
    if args.write_overlays:
        write_indexed_region_overlays(args.output_dir)
    polish_landmark_dataset_artifacts(args.output_dir, allow_overlap=args.allow_overlap)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    _validate_args(args)
    scenarios = _parse_csv(args.scenarios)

    if args.audit_only:
        manifest = audit_existing_manifest(args.output_dir, allow_overlap=args.allow_overlap)
    elif args.dataset == "wflw":
        manifest = build_wflw_manifest(
            args.wflw_annotations,
            args.output_dir,
            image_root=args.image_root,
            source_dir=_wflw_source_dir(args),
            source_zip=args.source_zip,
            cache_dir=args.cache_dir,
            download_url=args.download_url,
            force_download=args.force_download,
            no_download=args.no_download,
            scenario=args.scenario,
            scenarios=scenarios,
            samples_per_scenario=args.samples_per_scenario,
            manifest_mode=args.manifest_mode,
            allow_overlap=args.allow_overlap,
            write_overlays=args.write_overlays,
        )
    elif args.dataset == "cofw":
        manifest = build_cofw_manifest(
            _cofw_json(args),
            args.output_dir,
            image_root=args.image_root,
            source_dir=args.source_dir,
            source_zip=args.source_zip,
            cache_dir=args.cache_dir,
            download_url=args.download_url,
            force_download=args.force_download,
            no_download=args.no_download,
            scenario=args.scenario,
            scenarios=scenarios,
            samples_per_scenario=args.samples_per_scenario,
            manifest_mode=args.manifest_mode,
            allow_overlap=args.allow_overlap,
            write_overlays=args.write_overlays,
        )
    elif args.dataset == "300w":
        manifest = build_300w_manifest(
            args.output_dir,
            source_dir=args.source_dir,
            source_zip=args.source_zip,
            cache_dir=args.cache_dir,
            download_url=args.download_url,
            force_download=args.force_download,
            no_download=args.no_download,
            scenario=args.scenario,
            scenarios=scenarios,
            samples_per_scenario=args.samples_per_scenario,
            manifest_mode=args.manifest_mode,
            allow_overlap=args.allow_overlap,
            write_overlays=args.write_overlays,
        )
    elif args.dataset == "merl-rav":
        manifest = build_merl_rav_manifest(
            args.output_dir,
            source_dir=args.source_dir,
            source_zip=args.source_zip,
            cache_dir=args.cache_dir,
            download_url=args.download_url,
            aflw_source_dir=args.aflw_source_dir,
            aflw_source_zip=args.aflw_source_zip,
            aflw_download_url=args.aflw_download_url,
            force_download=args.force_download,
            no_download=args.no_download,
            scenario=args.scenario,
            scenarios=scenarios,
            samples_per_scenario=args.samples_per_scenario,
            manifest_mode=args.manifest_mode,
            allow_overlap=args.allow_overlap,
            write_overlays=args.write_overlays,
        )
    elif args.dataset == "aflw2000-3d":
        manifest = build_aflw2000_3d_manifest(
            args.output_dir,
            source_dir=args.source_dir,
            source_zip=args.source_zip,
            cache_dir=args.cache_dir,
            download_url=args.download_url,
            force_download=args.force_download,
            no_download=args.no_download,
            scenario=args.scenario,
            scenarios=scenarios,
            samples_per_scenario=args.samples_per_scenario,
            manifest_mode=args.manifest_mode,
            allow_overlap=args.allow_overlap,
            write_overlays=args.write_overlays,
        )
    elif args.recursive:
        manifest = build_directory_manifest(
            args.source_dir,
            args.output_dir,
            dataset=args.dataset,
            scenario=args.scenario,
            scenarios=scenarios,
            samples_per_scenario=args.samples_per_scenario,
            manifest_mode=args.manifest_mode,
            allow_overlap=args.allow_overlap,
            write_overlays=args.write_overlays,
        )
    else:
        manifest = build_manifest(
            Path(args.source_dir),
            args.output_dir,
            dataset=args.dataset,
            scenario=args.scenario,
            scenarios=scenarios,
            samples_per_scenario=args.samples_per_scenario,
            manifest_mode=args.manifest_mode,
            allow_overlap=args.allow_overlap,
            write_overlays=args.write_overlays,
        )
    _polish_output(args)
    logger.info("Wrote landmark manifest: %s", manifest)
    print(f"Wrote landmark manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
