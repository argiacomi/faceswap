#!/usr/bin/env python3
"""Generate a compact landmark ensemble report from metrics JSON."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", default="outputs/landmark_quality/report.json")
    args = parser.parse_args(argv)
    with open(args.metrics, encoding="utf-8") as infile:
        metrics = json.load(infile)
    overall = metrics.get("overall", {})
    best_single = min(
        (
            (name, data)
            for name, data in overall.items()
            if name
            not in {
                "plain_average",
                "static_weighted",
                "static_weighted_outliers",
                "static_weighted_downweight",
                "weighted_median",
            }
        ),
        key=lambda item: item[1].get("nme", float("inf")),
        default=(None, {}),
    )
    report = {"overall": overall, "best_single_model": best_single[0], "deltas": {}}
    for variant in (
        "plain_average",
        "static_weighted",
        "static_weighted_outliers",
        "static_weighted_downweight",
        "weighted_median",
    ):
        if variant in overall and best_single[0] is not None:
            report["deltas"][variant] = (
                overall[variant].get("nme", 0) - best_single[1].get("nme", 0)
            )
    with open(args.output, "w", encoding="utf-8") as outfile:
        json.dump(report, outfile, indent=2, sort_keys=True)
        outfile.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
