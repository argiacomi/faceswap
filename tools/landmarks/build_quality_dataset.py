#!/usr/bin/env python3
"""Build landmark quality dataset manifests."""

from __future__ import annotations

import argparse

from lib.landmarks.datasets import SUPPORTED_DATASETS, build_manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", default="default")
    args = parser.parse_args(argv)
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
