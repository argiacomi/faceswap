#!/usr/bin/env python3
"""Evaluate a trained stacked residual landmark regressor and apply promotion gates (#223).

Loads the same cached candidate/sample contexts used for training, applies the
regressor's clipped correction per sample, and reports NME change, win/loss rate,
catastrophic-outlier rate, residual magnitude, and clip rate broken out by runtime
bucket and coarse hard-case slice. Promotion gates turn that report into a
pass/fail decision that blocks regressions on easy/frontal cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import typing as T
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    parse_candidates,
)
from lib.landmarks.ensemble.stacked_regressor import load_stacked_regressor
from lib.landmarks.ensemble.stacked_regressor_evaluation import (
    DEFAULT_CATASTROPHIC_NME,
    evaluate_promotion_gates,
    evaluate_stacked_candidate,
)
from lib.landmarks.ensemble.weights import load_weights

logger = logging.getLogger("evaluate_stacked_landmark_regressor")

EVALUATION_REPORT_JSON = "stacked_regressor_evaluation.json"


def _show_progress() -> bool:
    return logger.isEnabledFor(logging.INFO) and sys.stderr.isatty()


def _context_progress(values: T.Sequence[T.Any], desc: str) -> T.Iterable[T.Any]:
    return T.cast(
        T.Iterable[T.Any],
        tqdm(
            values,
            total=len(values),
            desc=desc,
            unit="sample",
            disable=not _show_progress(),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path)
    parser.add_argument("--gt-cache-dir", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-cache-dir", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--catastrophic-nme", type=float, default=DEFAULT_CATASTROPHIC_NME)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument("--allow-image-backfill", action="store_true")
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Return a non-zero exit code if the promotion gates fail.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_all_contexts(args: argparse.Namespace, candidates: T.Sequence[str]) -> list[T.Any]:
    from lib.landmarks.ensemble.runtime_resolver_scorer_data import load_contexts

    contexts: list[T.Any] = []
    for source, manifest, cache_dir in (
        ("gt", args.gt_manifest, args.gt_cache_dir),
        ("production", args.production_manifest, args.production_cache_dir),
    ):
        if manifest is None or cache_dir is None:
            continue
        contexts.extend(
            load_contexts(
                manifest_path=manifest,
                cache_dir=cache_dir,
                weights_path=args.weights,
                candidates=candidates,
                source=source,
                failure_threshold=args.failure_threshold,
                outlier_threshold=args.outlier_threshold,
                allow_image_backfill=args.allow_image_backfill,
                progress=_context_progress,
            )
        )
    return contexts


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)
    if args.gt_manifest is None and args.production_manifest is None:
        parser.error("at least one of --gt-manifest / --production-manifest is required")

    regressor = load_stacked_regressor(args.regressor)
    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    contexts = _load_all_contexts(args, candidates)
    if not contexts:
        logger.error("no sample contexts loaded; nothing to evaluate")
        return 1

    report = evaluate_stacked_candidate(
        contexts,
        regressor,
        catastrophic_nme=args.catastrophic_nme,
    )
    gate = evaluate_promotion_gates(report)

    payload = {
        "regressor": str(args.regressor),
        "output_mode": regressor.output_mode,
        "base_candidate_policy": regressor.base_candidate_policy,
        "report": report.to_dict(),
        "promotion_gate": {
            "passed": gate.passed,
            "reasons": list(gate.reasons),
            "details": gate.details,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / EVALUATION_REPORT_JSON
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote stacked regressor evaluation report to %s", report_path)
    logger.info(
        "Promotion gate: %s%s",
        "PASS" if gate.passed else "FAIL",
        "" if gate.passed else f" ({'; '.join(gate.reasons)})",
    )
    if args.fail_on_gate and not gate.passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EVALUATION_REPORT_JSON", "main"]
