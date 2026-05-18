#!/usr/bin/env python3
"""Debug high visible-landmark NME for AFLW2000-3D buckets.

Writes:
  - sample_summary.csv
  - per_landmark_errors.csv
  - contact_sheet_<visibility_mode>.png
  - overlays/<visibility_mode>/*.png

Overlay colors, BGR:
  - GT visible: green
  - GT hidden: gray
  - prediction: magenta
  - large-error visible GT points: red rings
  - normalizer eye corners 36/45: cyan
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import typing as T
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.metrics import per_landmark_error
from lib.landmarks.datasets.manifest_io import load_manifest
from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.evaluation.harness import _fuse_variant
from lib.landmarks.evaluation.nme_metrics import evaluate_prediction
from lib.landmarks.evaluation.visualize import write_contact_sheet

Color = tuple[int, int, int]

COLOR_GT_VISIBLE: Color = (0, 255, 0)
COLOR_GT_HIDDEN: Color = (128, 128, 128)
COLOR_PRED: Color = (255, 0, 255)
COLOR_BAD: Color = (0, 0, 255)
COLOR_NORMALIZER: Color = (255, 255, 0)
COLOR_TEXT: Color = (255, 255, 255)
COLOR_BG: Color = (0, 0, 0)

EYE_CORNER_A = 36
EYE_CORNER_B = 45


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "sample"


def bucket_for(sample: T.Any) -> str:
    return f"{sample.dataset or 'unspecified'}:{sample.condition or 'unspecified'}"


def load_truth(sample: T.Any) -> np.ndarray:
    return np.load(sample.landmarks).astype("float32")


def gt_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    left, top = np.min(points[:, :2], axis=0)
    right, bottom = np.max(points[:, :2], axis=0)
    return float(left), float(top), float(right), float(bottom)


def bbox_stats(points: np.ndarray) -> dict[str, float]:
    left, top, right, bottom = gt_bbox(points)
    width = max(float(right - left), 0.0)
    height = max(float(bottom - top), 0.0)
    diag = float(math.hypot(width, height))
    sqrt_area = float(math.sqrt(max(width * height, 1e-6)))
    return {
        "bbox_left": left,
        "bbox_top": top,
        "bbox_right": right,
        "bbox_bottom": bottom,
        "bbox_width_px": width,
        "bbox_height_px": height,
        "bbox_diag_px": diag,
        "sqrt_area_px": sqrt_area,
    }


def interocular_px(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[EYE_CORNER_A, :2] - points[EYE_CORNER_B, :2]))


def visibility_mask(sample: T.Any, mode: str, landmark_count: int) -> np.ndarray:
    if sample.visibility is None:
        base = np.ones((landmark_count,), dtype=bool)
    else:
        base = np.asarray(sample.visibility, dtype=bool)
        if base.shape[0] != landmark_count:
            raise ValueError(
                f"{sample.sample_id}: visibility length {base.shape[0]} "
                f"does not match landmark count {landmark_count}"
            )

    if mode == "current":
        return base
    if mode == "inverted":
        inverted = ~base
        return inverted if inverted.any() else base
    if mode == "all":
        return np.ones((landmark_count,), dtype=bool)
    if mode == "none":
        return np.zeros((landmark_count,), dtype=bool)
    raise ValueError(f"unknown visibility mode: {mode}")


def mean_visible_error(errors: np.ndarray, visibility: np.ndarray) -> float:
    if visibility.any():
        return float(errors[visibility].mean())
    return float(errors.mean())


def max_visible_error(errors: np.ndarray, visibility: np.ndarray) -> float:
    if visibility.any():
        return float(errors[visibility].max())
    return float(errors.max())


def normalized(value_px: float, denominator: float) -> float:
    return float(value_px / denominator) if denominator > 0 else float("nan")


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    *,
    color: Color,
    radius: int,
    indexes: T.Iterable[int] | None = None,
    thickness: int = -1,
) -> None:
    height, width = image.shape[:2]
    idxs = range(len(points)) if indexes is None else indexes
    for idx in idxs:
        if idx < 0 or idx >= len(points):
            continue
        x_raw, y_raw = points[idx, :2]
        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), radius, color, thickness)


def draw_text_block(canvas: np.ndarray, lines: T.Sequence[str]) -> None:
    y = 18
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        y += 18


def make_canvas(image: np.ndarray, header_lines: T.Sequence[str]) -> np.ndarray:
    header_h = max(112, 20 + 18 * len(header_lines))
    height, width = image.shape[:2]
    canvas = np.zeros((height + header_h, width, 3), dtype=np.uint8)
    canvas[:header_h, :] = COLOR_BG
    canvas[header_h:, :] = image
    draw_text_block(canvas, header_lines)
    return canvas


def draw_overlay(
    *,
    image_path: str,
    output_path: Path,
    truth: np.ndarray,
    prediction: np.ndarray,
    visibility: np.ndarray,
    large_error_indexes: T.Sequence[int],
    header_lines: T.Sequence[str],
    point_radius: int,
) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    overlay = image.copy()
    visible_indexes = np.flatnonzero(visibility).astype(int).tolist()
    hidden_indexes = np.flatnonzero(~visibility).astype(int).tolist()

    draw_points(
        overlay,
        truth,
        color=COLOR_GT_HIDDEN,
        radius=max(point_radius - 1, 1),
        indexes=hidden_indexes,
    )
    draw_points(
        overlay,
        truth,
        color=COLOR_GT_VISIBLE,
        radius=point_radius,
        indexes=visible_indexes,
    )
    draw_points(
        overlay,
        prediction,
        color=COLOR_PRED,
        radius=point_radius,
    )

    for idx in large_error_indexes:
        if idx < 0 or idx >= len(truth):
            continue
        x_raw, y_raw = truth[idx, :2]
        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            cv2.circle(overlay, (x, y), point_radius + 5, COLOR_BAD, 1)

    draw_points(
        overlay,
        truth,
        color=COLOR_NORMALIZER,
        radius=point_radius + 2,
        indexes=[EYE_CORNER_A, EYE_CORNER_B],
    )

    canvas = make_canvas(overlay, header_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def prediction_records_for_sample(
    *,
    sample: T.Any,
    truth: np.ndarray,
    cache: DiskPredictionCache,
    models: T.Sequence[str],
    variants: T.Sequence[str],
    weights: dict[str, list[float]],
    outlier_threshold: float,
) -> dict[str, np.ndarray]:
    available = set(cache.available_models(sample.sample_id))
    predictions: dict[str, np.ndarray] = {}

    active_models = tuple(model for model in models if model in available)
    for model in active_models:
        predictions[model] = cache.read(sample.sample_id, model).landmarks

    if variants and len(active_models) >= 2:
        cached_predictions = [cache.read(sample.sample_id, model) for model in active_models]
        for variant in variants:
            fused, _rejected = _fuse_variant(
                variant,
                cached_predictions,
                active_models,
                weights,
                outlier_threshold=outlier_threshold,
            )
            predictions[variant] = fused.points

    return predictions


def summarize_one_prediction(
    *,
    sample: T.Any,
    label: str,
    pred: np.ndarray,
    truth: np.ndarray,
    visibility: np.ndarray,
    mode: str,
    large_error_ratio: float,
) -> tuple[dict[str, T.Any], list[dict[str, T.Any]]]:
    errors = per_landmark_error(pred, truth)
    mean_visible_px = mean_visible_error(errors, visibility)
    max_visible_px = max_visible_error(errors, visibility)
    stats = bbox_stats(truth)
    interocular = interocular_px(truth)
    normalizer_used = float(sample.normalizer) if sample.normalizer else interocular
    metrics = evaluate_prediction(
        pred,
        truth,
        normalizer=sample.normalizer,
        visibility=visibility,
    )

    large_error_indexes = np.flatnonzero(
        visibility & ((errors / max(normalizer_used, 1e-6)) > large_error_ratio)
    ).astype(int)

    visible_errors = errors[visibility] if visibility.any() else errors
    sorted_visible_indexes = (
        np.flatnonzero(visibility).astype(int)
        if visibility.any()
        else np.arange(errors.shape[0], dtype=int)
    )
    worst_order = sorted(
        sorted_visible_indexes.tolist(),
        key=lambda idx: float(errors[idx]),
        reverse=True,
    )

    summary = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "condition": sample.condition,
        "bucket": bucket_for(sample),
        "visibility_mode": mode,
        "label": label,
        "nme": float(metrics["nme"]),
        "failure": bool(metrics["failure"]),
        "visible_count": int(visibility.sum()),
        "hidden_count": int((~visibility).sum()),
        "interocular_px": interocular,
        "normalizer_used_px": normalizer_used,
        "mean_visible_error_px": mean_visible_px,
        "mean_all_error_px": float(errors.mean()),
        "max_visible_error_px": max_visible_px,
        "visible_error_p50_px": float(np.percentile(visible_errors, 50)),
        "visible_error_p90_px": float(np.percentile(visible_errors, 90)),
        "visible_error_p95_px": float(np.percentile(visible_errors, 95)),
        "visible_error_p99_px": float(np.percentile(visible_errors, 99)),
        "nme_interocular": normalized(mean_visible_px, interocular),
        "nme_normalizer_used": normalized(mean_visible_px, normalizer_used),
        "nme_bbox_diag": normalized(mean_visible_px, stats["bbox_diag_px"]),
        "nme_face_width": normalized(mean_visible_px, stats["bbox_width_px"]),
        "nme_sqrt_area": normalized(mean_visible_px, stats["sqrt_area_px"]),
        "large_visible_landmark_count": int(large_error_indexes.size),
        "large_visible_landmarks": ",".join(str(int(idx)) for idx in large_error_indexes),
        "worst_visible_landmarks": ",".join(str(int(idx)) for idx in worst_order[:10]),
        **stats,
    }

    per_landmark_rows = []
    for idx, err in enumerate(errors):
        per_landmark_rows.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "condition": sample.condition,
                "bucket": bucket_for(sample),
                "visibility_mode": mode,
                "label": label,
                "landmark_index": idx,
                "visible": bool(visibility[idx]),
                "error_px": float(err),
                "error_interocular_nme": normalized(float(err), interocular),
                "error_normalizer_nme": normalized(float(err), normalizer_used),
                "gt_x": float(truth[idx, 0]),
                "gt_y": float(truth[idx, 1]),
                "pred_x": float(pred[idx, 0]),
                "pred_y": float(pred[idx, 1]),
            }
        )

    return summary, per_landmark_rows


def write_csv(path: Path, rows: list[dict[str, T.Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: T.Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument("--visibility-modes", default="current,inverted,all")
    parser.add_argument("--min-nme", type=float, default=1.0)
    parser.add_argument("--large-error-ratio", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=85)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument(
        "--rank-label",
        default="best",
        choices=(
            "best",
            "hrnet",
            "spiga",
            "orformer",
            "static_weighted",
            "static_weighted_hard_drop",
            "static_weighted_downweight",
            "weighted_median",
        ),
        help="Which label to use for filtering/ranking overlays. 'best' chooses lowest NME per sample.",
    )
    args = parser.parse_args(argv)

    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.point_radius <= 0:
        raise SystemExit("--point-radius must be > 0")

    models = parse_csv(args.models)
    variants = parse_csv(args.variants)
    modes = parse_csv(args.visibility_modes)
    weights = load_weights(args.weights) if args.weights else {}
    cache = DiskPredictionCache(args.cache_dir)
    samples = load_manifest(args.manifest)
    output_dir = Path(args.output_dir)

    matched_samples = [sample for sample in samples if bucket_for(sample) == args.bucket]
    if not matched_samples:
        available = sorted({bucket_for(sample) for sample in samples})
        print(f"No samples found for bucket: {args.bucket}", file=sys.stderr)
        print("Available buckets:", file=sys.stderr)
        for bucket in available:
            print(f"  {bucket}", file=sys.stderr)
        return 1

    summary_rows: list[dict[str, T.Any]] = []
    per_landmark_rows: list[dict[str, T.Any]] = []
    overlay_jobs: list[dict[str, T.Any]] = []

    for sample in matched_samples:
        truth = load_truth(sample)
        predictions = prediction_records_for_sample(
            sample=sample,
            truth=truth,
            cache=cache,
            models=models,
            variants=variants,
            weights=weights,
            outlier_threshold=args.outlier_threshold,
        )

        if not predictions:
            continue

        for mode in modes:
            visibility = visibility_mask(sample, mode, truth.shape[0])
            sample_label_summaries: list[dict[str, T.Any]] = []
            sample_label_predictions: dict[str, np.ndarray] = {}

            for label, pred in predictions.items():
                summary, landmark_rows = summarize_one_prediction(
                    sample=sample,
                    label=label,
                    pred=pred,
                    truth=truth,
                    visibility=visibility,
                    mode=mode,
                    large_error_ratio=args.large_error_ratio,
                )
                sample_label_summaries.append(summary)
                sample_label_predictions[label] = pred
                summary_rows.append(summary)
                per_landmark_rows.extend(landmark_rows)

            if args.rank_label == "best":
                chosen = min(sample_label_summaries, key=lambda row: float(row["nme"]))
            else:
                candidates = [
                    row for row in sample_label_summaries if str(row["label"]) == args.rank_label
                ]
                if not candidates:
                    continue
                chosen = candidates[0]

            if float(chosen["nme"]) < args.min_nme:
                continue

            overlay_jobs.append(
                {
                    "sample": sample,
                    "truth": truth,
                    "visibility": visibility,
                    "visibility_mode": mode,
                    "chosen": chosen,
                    "prediction": sample_label_predictions[str(chosen["label"])],
                }
            )

    overlay_jobs.sort(key=lambda job: float(job["chosen"]["nme"]), reverse=True)
    overlay_jobs = overlay_jobs[: args.limit]

    overlay_records: list[dict[str, T.Any]] = []
    overlays_by_mode: dict[str, list[str]] = {mode: [] for mode in modes}

    for rank, job in enumerate(overlay_jobs, start=1):
        sample = job["sample"]
        truth = job["truth"]
        visibility = job["visibility"]
        chosen = job["chosen"]
        mode = str(job["visibility_mode"])
        pred = job["prediction"]

        large_indexes = [
            int(value)
            for value in str(chosen["large_visible_landmarks"]).split(",")
            if value.strip()
        ]

        header = [
            f"{rank:03d} {sample.sample_id}",
            f"mode={mode} label={chosen['label']} nme={float(chosen['nme']):.3f}",
            (
                f"visible={chosen['visible_count']}/68 "
                f"interocular={float(chosen['interocular_px']):.2f}px "
                f"normalizer={float(chosen['normalizer_used_px']):.2f}px"
            ),
            (
                f"mean_visible={float(chosen['mean_visible_error_px']):.2f}px "
                f"all={float(chosen['mean_all_error_px']):.2f}px "
                f"max_visible={float(chosen['max_visible_error_px']):.2f}px"
            ),
            (
                f"bbox_diag={float(chosen['bbox_diag_px']):.2f}px "
                f"sqrt_area={float(chosen['sqrt_area_px']):.2f}px "
                f"nme_bbox_diag={float(chosen['nme_bbox_diag']):.3f}"
            ),
            "green=GT visible gray=GT hidden magenta=pred red=large visible error cyan=36/45",
        ]

        output_path = (
            output_dir
            / "overlays"
            / mode
            / f"{rank:03d}_{safe_name(sample.sample_id)}_{safe_name(str(chosen['label']))}_{float(chosen['nme']):.3f}.png"
        )
        draw_overlay(
            image_path=sample.image,
            output_path=output_path,
            truth=truth,
            prediction=pred,
            visibility=visibility,
            large_error_indexes=large_indexes,
            header_lines=header,
            point_radius=args.point_radius,
        )

        record = dict(chosen)
        record["rank"] = rank
        record["image"] = sample.image
        record["landmarks"] = sample.landmarks
        record["overlay"] = str(output_path)
        overlay_records.append(record)
        overlays_by_mode.setdefault(mode, []).append(str(output_path))

    write_csv(output_dir / "sample_summary.csv", summary_rows)
    write_csv(output_dir / "per_landmark_errors.csv", per_landmark_rows)
    write_csv(output_dir / "overlay_index.csv", overlay_records)
    write_json(
        output_dir / "run_config.json",
        {
            "manifest": args.manifest,
            "cache_dir": args.cache_dir,
            "bucket": args.bucket,
            "models": list(models),
            "variants": list(variants),
            "weights": args.weights,
            "visibility_modes": list(modes),
            "min_nme": args.min_nme,
            "large_error_ratio": args.large_error_ratio,
            "rank_label": args.rank_label,
            "matched_sample_count": len(matched_samples),
            "overlay_count": len(overlay_records),
        },
    )

    for mode, paths in overlays_by_mode.items():
        if paths:
            write_contact_sheet(paths, output_dir / f"contact_sheet_{mode}.png")

    print(f"Matched bucket samples: {len(matched_samples)}")
    print(f"Wrote sample summary: {output_dir / 'sample_summary.csv'}")
    print(f"Wrote per-landmark errors: {output_dir / 'per_landmark_errors.csv'}")
    print(f"Wrote overlay index: {output_dir / 'overlay_index.csv'}")
    print(f"Wrote overlays: {output_dir / 'overlays'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
