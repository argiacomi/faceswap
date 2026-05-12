#!/usr/bin/env python3
"""Build landmark quality dataset manifests."""

from __future__ import annotations

import argparse

from lib.landmarks.datasets import (
    SUPPORTED_DATASETS,
    build_cofw_manifest,
    build_directory_manifest,
    build_manifest,
    build_wflw_manifest,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--wflw-annotations", default=None)
    parser.add_argument("--cofw-json", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args(argv)
    if args.wflw_annotations:
        manifest = build_wflw_manifest(
            args.wflw_annotations,
            args.output_dir,
            image_root=args.image_root,
            scenario=args.scenario,
        )
    elif args.cofw_json:
        manifest = build_cofw_manifest(
            args.cofw_json,
            args.output_dir,
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
            args.source_dir,
            args.output_dir,
            dataset=args.dataset,
            scenario=args.scenario,
        )
    print(f"Wrote landmark manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
