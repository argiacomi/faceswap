#!/usr/bin/env python3
"""Write GT + prediction debug overlays for one dataset:condition bucket."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import typing as T
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.fusion_variants import fuse_variant
from lib.landmarks.datasets.manifest_io import load_manifest
from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.evaluation.visualize import write_contact_sheet

COLORS: dict[str, tuple[int, int, int]] = {
    "GT": (0, 255, 0),
    "hrnet": (255, 0, 0),
    "spiga": (0, 0, 255),
    "orformer": (255, 255, 0),
    "static_weighted": (255, 0, 255),
    "static_weighted_hard_drop": (0, 165, 255),
    "static_weighted_downweight": (255, 128, 0),
    "weighted_median": (128, 0, 255),
}


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")


def bucket_for(sample: T.Any) -> str:
    return f"{sample.dataset or 'unspecified'}:{sample.condition or 'unspecified'}"


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    height, width = image.shape[:2]
    pts = np.asarray(points, dtype="float32")
    for x_raw, y_raw in pts[:, :2]:
        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), radius, color, thickness=-1)


def draw_legend(
    image: np.ndarray,
    labels: T.Sequence[str],
) -> None:
    x = 10
    y = 18
    for label in labels:
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(image, (x, y - 10), (x + 10, y), color, thickness=-1)
        cv2.putText(
            image,
            label,
            (x + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18


def write_overlay(
    *,
    image_path: str,
    output_path: Path,
    predictions: T.Mapping[str, np.ndarray],
    radius: int,
) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    overlay = image.copy()
    for label, points in predictions.items():
        draw_points(
            overlay,
            points,
            color=COLORS.get(label, (255, 255, 255)),
            radius=radius,
        )
    draw_legend(overlay, list(predictions))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket", default="aflw2000-3d:occlusion")
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument("--variants", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested model is missing for a matched sample.",
    )
    args = parser.parse_args(argv)

    models = parse_csv(args.models)
    variants = parse_csv(args.variants)
    weights = load_weights(args.weights) if args.weights else {}
    cache = DiskPredictionCache(args.cache_dir)
    samples = load_manifest(args.manifest)

    matched = [sample for sample in samples if bucket_for(sample) == args.bucket]
    if not matched:
        available = sorted({bucket_for(sample) for sample in samples})
        print(f"No samples found for bucket: {args.bucket}", file=sys.stderr)
        print("Available buckets:", file=sys.stderr)
        for bucket in available:
            print(f"  {bucket}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / safe_name(args.bucket)
    records: list[dict[str, T.Any]] = []
    overlay_paths: list[str] = []

    for index, sample in enumerate(matched[: args.limit], start=1):
        truth = np.load(sample.landmarks).astype("float32")
        predictions: dict[str, np.ndarray] = {"GT": truth}

        available = set(cache.available_models(sample.sample_id))
        missing = [model for model in models if model not in available]
        if missing and args.strict:
            raise FileNotFoundError(
                f"{sample.sample_id} is missing cached predictions for {missing}"
            )

        active_models = tuple(model for model in models if model in available)
        for model in active_models:
            predictions[model] = cache.read(sample.sample_id, model).landmarks

        if variants and len(active_models) >= 2:
            cached = [cache.read(sample.sample_id, model) for model in active_models]
            for variant in variants:
                fused = fuse_variant(
                    variant,
                    cached,
                    models=active_models,
                    weights=weights,
                    outlier_threshold=args.outlier_threshold,
                )
                predictions[variant] = fused

        out_path = overlay_dir / f"{index:04d}_{safe_name(sample.sample_id)}.png"
        write_overlay(
            image_path=sample.image,
            output_path=out_path,
            predictions=predictions,
            radius=args.radius,
        )
        overlay_paths.append(str(out_path))
        records.append(
            {
                "rank": index,
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "condition": sample.condition,
                "bucket": bucket_for(sample),
                "image": sample.image,
                "landmarks": sample.landmarks,
                "models_drawn": ",".join(active_models),
                "missing_models": ",".join(missing),
                "variants_drawn": ",".join(variants),
                "overlay": str(out_path),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_json = output_dir / "debug_bucket_overlays.json"
    index_csv = output_dir / "debug_bucket_overlays.csv"
    index_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    if overlay_paths:
        write_contact_sheet(overlay_paths, output_dir / "contact_sheet.png")

    print(f"Wrote {len(records)} overlays to {overlay_dir}")
    print(f"Index: {index_json}")
    print(f"Contact sheet: {output_dir / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
