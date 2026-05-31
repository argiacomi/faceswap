#!/usr/bin/env python3
"""Build a hard-case alignment-validation set from a landmark manifest (#82).

Reads an existing manifest (typically AFLW2000-3D, which preserves the 3DDFA
``Pose_Para`` vector in ``metadata``) and writes a filtered manifest plus the
standard geometry outputs into ``outputs/landmarks/hard_alignment_validation``
by default. Hard-case sample tagging is pose-driven. Yaw/profile buckets cover
ordinary profile and large-yaw cases, while explicit roll buckets isolate the
rotated-face failures seen in geometry validation.

The v1 flow is entirely automated: no manual landmarks, no real-failure
collection. Dataset landmarks remain the ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.landmarks.evaluation.hard_slices import (
    DEFAULT_EXTREME_ROLL_DEGREES,
    DEFAULT_FRONTAL_YAW_DEGREES,
    DEFAULT_PROFILE_MAX_DEGREES,
    DEFAULT_PROFILE_MIN_DEGREES,
    DEFAULT_ROLL_DEGREES,
    HardSliceThresholds,
    slice_manifest_samples,
)
from tools.landmarks.run_landmark_resolver_pipeline import DEFAULT_MODELS

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Path = Path("outputs/landmarks/hard_alignment_validation")


def _load_manifest(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or payload.get("scenarios") or []
    if not isinstance(samples, list):
        raise ValueError(f"manifest {path} has no 'samples' (or 'scenarios') list")
    return payload, samples


def _resolve_sample_paths(samples: list[dict], base_dir: Path) -> list[dict]:
    """Rewrite relative ``image`` / ``landmarks`` paths to absolute under ``base_dir``.

    The output manifest may live in a different directory than the source, so
    relative path references must be made portable before serialization.
    """
    resolved: list[dict] = []
    for sample in samples:
        new_sample = dict(sample)
        for key in ("image", "landmarks", "ground_truth"):
            value = sample.get(key)
            if not value:
                continue
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            new_sample[key] = str(candidate)
        resolved.append(new_sample)
    return resolved


def _write_manifest(path: Path, payload: dict, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = dict(payload)
    out_payload["samples"] = samples
    out_payload.pop("scenarios", None)
    path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")


def build_hard_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    thresholds: HardSliceThresholds,
    include_unposed: bool,
    hard_only: bool,
) -> dict[str, object]:
    """Filter ``manifest_path`` into a hard-case manifest under ``output_dir``."""
    payload, samples = _load_manifest(manifest_path)
    sliced, counts = slice_manifest_samples(
        samples,
        thresholds=thresholds,
        include_unposed=include_unposed,
        hard_only=hard_only,
    )
    sliced = _resolve_sample_paths(sliced, manifest_path.parent)
    output_manifest = output_dir / "manifest.json"
    _write_manifest(output_manifest, payload, sliced)
    summary = {
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_manifest),
        "thresholds": {
            "frontal_degrees": thresholds.frontal_degrees,
            "profile_min_degrees": thresholds.profile_min_degrees,
            "profile_max_degrees": thresholds.profile_max_degrees,
            "roll_degrees": thresholds.roll_degrees,
            "extreme_roll_degrees": thresholds.extreme_roll_degrees,
        },
        "selected_sample_count": len(sliced),
        "source_sample_count": len(samples),
        "bucket_counts": dict(counts),
        "hard_only": hard_only,
        "include_unposed": include_unposed,
    }
    (output_dir / "hard_slice_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _run_geometry_eval(
    *,
    hard_manifest: Path,
    cache_dir: Path,
    output_dir: Path,
    models: str,
    variants: str,
    weights_path: Path | None,
    outlier_threshold: float,
    aligned_size: int,
    region_failure_threshold: float,
) -> None:
    """Invoke the geometry-metrics CLI against the freshly built hard-case manifest."""
    from tools.landmarks.evaluate_alignment_geometry import main as geometry_main

    argv = [
        "--manifest",
        str(hard_manifest),
        "--cache-dir",
        str(cache_dir),
        "--models",
        models,
        "--output-dir",
        str(output_dir),
        "--aligned-size",
        str(aligned_size),
        "--region-failure-threshold",
        str(region_failure_threshold),
    ]
    if variants:
        argv.extend(["--variants", variants])
    if weights_path:
        argv.extend(["--weights", str(weights_path)])
        argv.extend(["--outlier-threshold", str(outlier_threshold)])
    geometry_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Source landmark manifest.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Destination directory (defaults to outputs/landmarks/hard_alignment_validation).",
    )
    parser.add_argument(
        "--frontal-degrees",
        type=float,
        default=DEFAULT_FRONTAL_YAW_DEGREES,
        help="|yaw| below this is labeled 'frontal'.",
    )
    parser.add_argument("--profile-min-degrees", type=float, default=DEFAULT_PROFILE_MIN_DEGREES)
    parser.add_argument("--profile-max-degrees", type=float, default=DEFAULT_PROFILE_MAX_DEGREES)
    parser.add_argument(
        "--roll-degrees",
        type=float,
        default=DEFAULT_ROLL_DEGREES,
        help="|roll| at or above this is routed to roll-specific hard buckets.",
    )
    parser.add_argument(
        "--extreme-roll-degrees",
        type=float,
        default=DEFAULT_EXTREME_ROLL_DEGREES,
        help="|roll| at or above this is labeled 'extreme_roll' for non-yaw-hard samples.",
    )
    parser.add_argument(
        "--include-unposed",
        action="store_true",
        help="Keep samples that lack Pose_Para (tagged as 'no_pose').",
    )
    parser.add_argument(
        "--hard-only",
        action="store_true",
        default=True,
        help="Keep only hard-case buckets (default).",
    )
    parser.add_argument(
        "--no-hard-only",
        dest="hard_only",
        action="store_false",
        help="Disable filtering; tag every sample but keep them all.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help=(
            "Optional populated prediction cache. When supplied the geometry-metrics "
            "CLI is invoked against the freshly built hard-case manifest."
        ),
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--variants", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument("--aligned-size", type=int, default=512)
    parser.add_argument("--region-failure-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging("INFO")

    thresholds = HardSliceThresholds(
        frontal_degrees=args.frontal_degrees,
        profile_min_degrees=args.profile_min_degrees,
        profile_max_degrees=args.profile_max_degrees,
        roll_degrees=args.roll_degrees,
        extreme_roll_degrees=args.extreme_roll_degrees,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_hard_manifest(
        Path(args.manifest),
        output_dir,
        thresholds=thresholds,
        include_unposed=args.include_unposed,
        hard_only=args.hard_only,
    )
    logger.info(
        "Selected %d/%d samples into %s",
        summary["selected_sample_count"],
        summary["source_sample_count"],
        summary["output_manifest"],
    )
    if args.cache_dir:
        _run_geometry_eval(
            hard_manifest=Path(summary["output_manifest"]),  # type: ignore[arg-type]
            cache_dir=Path(args.cache_dir),
            output_dir=output_dir,
            models=args.models,
            variants=args.variants,
            weights_path=Path(args.weights) if args.weights else None,
            outlier_threshold=args.outlier_threshold,
            aligned_size=args.aligned_size,
            region_failure_threshold=args.region_failure_threshold,
        )
        logger.info("Wrote geometry metrics to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
