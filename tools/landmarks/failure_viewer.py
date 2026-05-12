#!/usr/bin/env python3
"""Create worst-first landmark failure contact sheets."""

from __future__ import annotations

import argparse
import json

from lib.landmarks.eval.visualize import write_contact_sheet


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--output", default="outputs/landmark_debug/contact_sheet.png")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args(argv)
    with open(args.metrics, encoding="utf-8") as infile:
        payload = json.load(infile)
    rows = sorted(payload.get("rows", []), key=lambda row: row.get("nme", 0), reverse=True)
    images = [f"{args.debug_dir}/{row['sample_id']}.png" for row in rows[: args.limit]]
    write_contact_sheet(images, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
