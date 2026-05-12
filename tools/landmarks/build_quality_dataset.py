#!/usr/bin/env python3
"""Build landmark quality dataset manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lib.landmarks.datasets import (
    SUPPORTED_DATASETS,
    audit_existing_manifest,
    build_cofw_manifest,
    build_directory_manifest,
    build_manifest,
    build_wflw_manifest,
)
from lib.landmarks.datasets.sources import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated CLI value."""
    if value is None:
        return None
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or None


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the landmark dataset builder argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
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
        help="Write visual landmark overlays when source images are readable.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate and rewrite dataset_audit.json for an existing manifest "
        "without regenerating data.",
    )
    parser.add_argument("--wflw-annotations", default=None)
    parser.add_argument("--cofw-json", default=None)
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
    if args.audit_only:
        if any((args.wflw_annotations, args.cofw_json, args.source_dir, args.source_zip)):
            raise ValueError(
                "--audit-only only accepts --dataset, --output-dir, "
                "--allow-overlap, and logging options"
            )
        if args.no_download or args.force_download or args.download_url:
            raise ValueError("--audit-only does not use download or cache source options")
        return

    source_modes = [
        bool(args.wflw_annotations),
        bool(args.cofw_json),
        bool(args.source_dir),
        bool(args.source_zip),
    ]
    if sum(source_modes) > 1:
        # image_root is an adjunct, not a source mode. Avoid ambiguous datasets.
        raise ValueError(
            "Pass only one source mode: --wflw-annotations, "
            "--cofw-json, --source-dir, or --source-zip"
        )
    if args.dataset == "wflw" and args.cofw_json:
        raise ValueError("--cofw-json is only valid with --dataset cofw")
    if args.dataset == "cofw" and args.wflw_annotations:
        raise ValueError("--wflw-annotations is only valid with --dataset wflw")
    if args.dataset == "directory":
        if not args.source_dir:
            raise ValueError("--dataset directory requires --source-dir")
        if args.source_zip or args.wflw_annotations or args.cofw_json:
            raise ValueError("--dataset directory only supports --source-dir")
    if args.no_download and args.force_download:
        raise ValueError("--no-download and --force-download cannot be combined")


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
    elif args.dataset == "cofw":
        manifest = build_cofw_manifest(
            args.cofw_json,
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
    logger.info("Wrote landmark manifest: %s", manifest)
    print(f"Wrote landmark manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
