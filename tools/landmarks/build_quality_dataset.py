#!/usr/bin/env python3
"""Build landmark quality dataset manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lib.landmarks.datasets import (
    SUPPORTED_DATASETS,
    build_cofw_manifest,
    build_directory_manifest,
    build_manifest,
    build_wflw_manifest,
)
from lib.landmarks.datasets.sources import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


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
    source_modes = [
        bool(args.wflw_annotations),
        bool(args.cofw_json),
        bool(args.source_dir),
        bool(args.source_zip),
    ]
    if sum(source_modes) > 1:
        # image_root is an adjunct, not a source mode. Avoid ambiguous datasets.
        raise ValueError(
            "Pass only one source mode: --wflw-annotations, --cofw-json, --source-dir, or --source-zip"
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

    if args.dataset == "wflw":
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
        )
    elif args.recursive:
        manifest = build_directory_manifest(
            args.source_dir,
            args.output_dir,
            dataset=args.dataset,
            scenario=args.scenario,
        )
    else:
        manifest = build_manifest(
            Path(args.source_dir),
            args.output_dir,
            dataset=args.dataset,
            scenario=args.scenario,
        )
    logger.info("Wrote landmark manifest: %s", manifest)
    print(f"Wrote landmark manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
