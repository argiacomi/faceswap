#!/usr/bin/env python3
"""Write fixture or model-run predictions into the landmark disk cache."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import typing as T
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from lib.landmarks.adapters import LandmarkAdapter, build_landmark_adapter
from lib.landmarks.cache.prediction_cache import DiskPredictionCache, config_hash
from lib.landmarks.coordinates import roi_to_matrix
from lib.landmarks.core.schema import LandmarkPrediction, normalize_landmarks
from lib.landmarks.evaluation.harness import LandmarkSample, load_manifest

logger = logging.getLogger(__name__)


def _split_models(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _show_progress() -> bool:
    """Return whether tqdm progress bars should be shown."""
    return logger.isEnabledFor(logging.INFO)


def _parse_prediction_roots(values: T.Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--prediction-root must be in model=path form")
        model, root = value.split("=", 1)
        model = model.strip().lower()
        if not model:
            raise ValueError("--prediction-root model name cannot be empty")
        roots[model] = Path(root).expanduser()
    return roots


def _manifest_prediction(entry: dict[str, T.Any], model: str) -> str:
    for key in ("predictions", "model_predictions", "prediction_fixtures", "fixtures"):
        predictions = entry.get(key)
        if isinstance(predictions, dict) and model in predictions:
            return str(predictions[model])
    model_key = f"{model}_prediction"
    if model_key in entry:
        return str(entry[model_key])
    return ""


def _load_manifest_entries(path: str | Path) -> list[dict[str, T.Any]]:
    manifest = Path(path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(entries, list):
        raise ValueError("manifest samples must be a list")
    return [entry for entry in entries if isinstance(entry, dict)]


def _entry_id(entry: dict[str, T.Any]) -> str:
    """Return the manifest sample id for raw entry metadata."""
    return str(entry.get("sample_id") or entry.get("id") or entry.get("name") or "")


def _prediction_path(
    *,
    sample_id: str,
    model: str,
    manifest_path: Path,
    entry: dict[str, T.Any],
    prediction_roots: T.Mapping[str, Path],
) -> Path:
    manifest_prediction = _manifest_prediction(entry, model)
    if manifest_prediction:
        return (manifest_path.parent / manifest_prediction).resolve()
    root = prediction_roots.get(model)
    if root is None:
        raise FileNotFoundError(f"no prediction path for sample '{sample_id}' model '{model}'")
    candidates = (
        root / f"{sample_id}.npy",
        root / sample_id / f"{model}.npy",
        root / model / f"{sample_id}.npy",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing prediction for sample '{sample_id}' model '{model}' under {root}"
    )


def _write_prediction(
    cache: DiskPredictionCache,
    sample_id: str,
    model: str,
    prediction_path: Path,
    *,
    schema: str,
    coordinate_space: str,
    checkpoint: str,
    refresh: bool,
) -> bool:
    points = np.load(str(prediction_path)).astype("float32")
    prediction = LandmarkPrediction(
        landmarks=points,
        schema=schema,
        model_name=model,
        source_landmark_count=points.shape[0],
        coordinate_space=coordinate_space,
    )
    config = {
        "schema": schema,
        "coordinate_space": coordinate_space,
        "source": str(prediction_path),
    }
    fresh = cache.is_fresh(sample_id, prediction, checkpoint=checkpoint, config=config)
    cache.write(
        sample_id,
        prediction,
        checkpoint=checkpoint,
        config=config,
        refresh=refresh,
    )
    return refresh or not fresh


def _checkpoint_value(args: argparse.Namespace) -> str:
    """Return the effective checkpoint identifier for cache metadata."""
    return args.checkpoint_tag or args.checkpoint


def _bbox_values(raw: T.Any) -> tuple[float, float, float, float] | None:
    """Normalize common bbox payloads to ``left, top, right, bottom``.

    Thin wrapper over :func:`lib.landmarks.datasets.manifest_io.coerce_bbox` kept under
    the legacy name so the cache-build code below stays unchanged. The
    canonical coercer handles dicts (ltrb/xywh keys) plus 4+ length
    sequences, including the xywh-fallback case the cache previously
    open-coded.
    """
    from lib.landmarks.datasets.manifest_io import coerce_bbox

    return coerce_bbox(raw)


def _entry_face_bbox(entry: dict[str, T.Any]) -> tuple[float, float, float, float] | None:
    """Return an explicit face bbox only, ignoring generic annotation bboxes."""
    bbox = _bbox_values(entry.get("face_bbox"))
    if bbox is not None:
        return bbox
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        return _bbox_values(metadata.get("face_bbox"))
    return None


def _entry_bbox(entry: dict[str, T.Any]) -> tuple[float, float, float, float] | None:
    """Return an explicit face bbox from a manifest entry when present."""
    face_bbox = _entry_face_bbox(entry)
    if face_bbox is not None:
        return face_bbox
    bbox = _bbox_values(entry.get("bbox"))
    if bbox is not None:
        return bbox
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        bbox = _bbox_values(metadata.get("bbox"))
        if bbox is not None:
            return bbox
    return None


def _square_roi_from_bbox(
    bbox: tuple[float, float, float, float],
    *,
    scale: float = 1.25,
) -> np.ndarray:
    """Return a square ROI around ``bbox``."""
    left, top, right, bottom = bbox
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    side = max(width, height) * scale
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    half = side * 0.5
    return np.asarray(
        [center_x - half, center_y - half, center_x + half, center_y + half],
        dtype="float32",
    )


def _scale_roi(
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
) -> np.ndarray:
    """Return ``bbox`` as-is, or square-scaled when ``scale`` is not 1.0."""
    if scale <= 0:
        raise ValueError("gt_roi_scale must be greater than zero")
    if np.isclose(scale, 1.0):
        return np.asarray(bbox, dtype="float32")
    return _square_roi_from_bbox(bbox, scale=scale)


def _roi_from_truth(sample: LandmarkSample, *, scale: float = 1.0) -> np.ndarray:
    """Derive a validation-only raw ROI from ground-truth landmarks."""
    try:
        raw = np.load(sample.landmarks).astype("float32")
        points = normalize_landmarks(raw)
    except Exception as err:
        raise ValueError(
            f"sample '{sample.sample_id}' has no usable face_bbox and GT-derived ROI failed: {err}"
        ) from err
    if points.ndim != 2 or points.shape[1] != 2 or points.size == 0:
        raise ValueError(f"sample '{sample.sample_id}' has invalid GT landmarks for ROI")
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.size == 0:
        raise ValueError(f"sample '{sample.sample_id}' has no finite GT landmarks for ROI")
    left, top = np.min(finite, axis=0)
    right, bottom = np.max(finite, axis=0)
    return _scale_roi((float(left), float(top), float(right), float(bottom)), scale=scale)


def _is_wflw_98_entry(sample: LandmarkSample, entry: dict[str, T.Any]) -> bool:
    """Return whether an entry is a native WFLW 98-point validation sample."""
    if sample.dataset.lower() != "wflw":
        return False
    return str(entry.get("source_schema", "")).lower() == "2d_98"


def _base_roi_for_sample(
    sample: LandmarkSample,
    entry: dict[str, T.Any],
    *,
    allow_gt_roi: bool,
    gt_roi_scale: float = 1.0,
) -> tuple[np.ndarray, str]:
    """Return the raw face ROI and source label for one manifest sample."""
    face_bbox = _entry_face_bbox(entry)
    if face_bbox is not None:
        return np.asarray(face_bbox, dtype="float32"), "manifest_face_bbox"
    if _is_wflw_98_entry(sample, entry):
        if not allow_gt_roi:
            raise ValueError(
                f"sample '{sample.sample_id}' is a WFLW 98-point sample without face_bbox; "
                "GT-derived ROI is required because WFLW annotation bbox is not a model crop ROI"
            )
        return _roi_from_truth(sample, scale=gt_roi_scale), "gt_landmarks_wflw_98"
    bbox = _entry_bbox(entry)
    if bbox is not None:
        return np.asarray(bbox, dtype="float32"), "manifest_bbox"
    if not allow_gt_roi:
        raise ValueError(
            f"sample '{sample.sample_id}' is missing face_bbox/bbox and GT-derived ROI is disabled"
        )
    return _roi_from_truth(sample, scale=gt_roi_scale), "gt_landmarks"


def _model_roi_for_adapter(adapter: LandmarkAdapter, raw_roi: np.ndarray) -> np.ndarray:
    """Return the model crop ROI, letting Faceswap plugins own crop expansion."""
    plugin = getattr(adapter, "plugin", None)
    if plugin is not None and hasattr(plugin, "pre_process"):
        adjusted = plugin.pre_process(np.asarray(raw_roi, dtype="float32")[None])
        return np.asarray(adjusted[0], dtype="float32")
    values = tuple(float(value) for value in np.asarray(raw_roi, dtype="float32").reshape(4))
    return _square_roi_from_bbox(values, scale=1.25)


def _load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image with OpenCV and fail loudly on missing/unreadable input."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def _crop_square(image: np.ndarray, roi: np.ndarray, size: int) -> np.ndarray:
    """Crop a possibly out-of-bounds square ROI and resize to model input size."""
    left, top, right, bottom = [int(round(float(value))) for value in roi]
    side = max(right - left, bottom - top, 1)
    right = left + side
    bottom = top + side
    crop = np.zeros((side, side, image.shape[2]), dtype=image.dtype)
    image_h, image_w = image.shape[:2]
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, image_w)
    src_bottom = min(bottom, image_h)
    if src_right > src_left and src_bottom > src_top:
        dst_left = src_left - left
        dst_top = src_top - top
        crop[
            dst_top : dst_top + (src_bottom - src_top),
            dst_left : dst_left + (src_right - src_left),
        ] = image[src_top:src_bottom, src_left:src_right]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def _fresh_cached_run(
    cache: DiskPredictionCache,
    *,
    sample_id: str,
    model: str,
    checkpoint: str,
    config: dict[str, T.Any],
) -> bool:
    """Return whether an existing run-model cache entry can be reused."""
    prediction_path = cache.prediction_path(sample_id, model)
    if not prediction_path.is_file():
        return False
    metadata = cache.load_metadata(sample_id).get(model, {})
    if not metadata:
        return False
    return (
        metadata.get("checkpoint") == checkpoint
        and metadata.get("config_hash") == config_hash(config)
        and metadata.get("schema") == "2d_68"
        and metadata.get("coordinate_space") == "frame"
    )


@dataclass(frozen=True)
class _RunItem:
    sample: LandmarkSample
    entry: dict[str, T.Any]
    base_roi: np.ndarray
    roi_source: str
    model_roi: np.ndarray
    matrix: np.ndarray
    config: dict[str, T.Any]


def _model_config(
    *,
    model: str,
    adapter: LandmarkAdapter,
    sample: LandmarkSample,
    base_roi: np.ndarray,
    model_roi: np.ndarray,
    roi_source: str,
    device: str,
) -> dict[str, T.Any]:
    """Return cache config metadata for a model/sample prediction run."""
    plugin = getattr(adapter, "plugin", None)
    return {
        "mode": "run-models",
        "model": model,
        "adapter_schema": adapter.config.schema,
        "adapter_coordinate_space": adapter.config.coordinate_space,
        "plugin": type(plugin).__name__ if plugin is not None else type(adapter).__name__,
        "image": str(Path(sample.image).resolve()),
        "base_roi": [float(value) for value in base_roi.tolist()],
        "model_roi": [float(value) for value in model_roi.tolist()],
        "roi_source": roi_source,
        "device": device,
    }


def _build_model_adapters(models: T.Sequence[str], *, device: str) -> dict[str, LandmarkAdapter]:
    """Build and load model adapters for CLI prediction generation."""
    adapters = {
        model: build_landmark_adapter(
            model, device=device, input_is_rgb=False, input_scale=(0, 255)
        )
        for model in models
    }
    for adapter in adapters.values():
        if hasattr(adapter, "load_model"):
            adapter.load_model()  # type: ignore[attr-defined]
    return adapters


def _prepare_run_items(
    *,
    samples: T.Sequence[LandmarkSample],
    entries: T.Mapping[str, dict[str, T.Any]],
    model: str,
    adapter: LandmarkAdapter,
    cache: DiskPredictionCache,
    checkpoint: str,
    refresh: bool,
    allow_gt_roi: bool,
    gt_roi_scale: float = 1.0,
    device: str,
) -> tuple[list[_RunItem], int]:
    """Build run plans and count reusable cache hits."""
    items: list[_RunItem] = []
    reused = 0
    for sample in samples:
        entry = entries.get(sample.sample_id, {})
        base_roi, roi_source = _base_roi_for_sample(
            sample,
            entry,
            allow_gt_roi=allow_gt_roi,
            gt_roi_scale=gt_roi_scale,
        )
        model_roi = _model_roi_for_adapter(adapter, base_roi)
        matrix = roi_to_matrix(model_roi)
        config = _model_config(
            model=model,
            adapter=adapter,
            sample=sample,
            base_roi=base_roi,
            model_roi=model_roi,
            roi_source=roi_source,
            device=device,
        )
        if not refresh and _fresh_cached_run(
            cache,
            sample_id=sample.sample_id,
            model=model,
            checkpoint=checkpoint,
            config=config,
        ):
            reused += 1
            continue
        items.append(
            _RunItem(
                sample=sample,
                entry=entry,
                base_roi=base_roi,
                roi_source=roi_source,
                model_roi=model_roi,
                matrix=matrix,
                config=config,
            )
        )
    return items, reused


def _run_model_predictions(
    *,
    manifest_path: Path,
    models: T.Sequence[str],
    cache_dir: Path,
    checkpoint: str,
    batch_size: int,
    device: str,
    refresh: bool,
    allow_gt_roi: bool,
    gt_roi_scale: float = 1.0,
    adapters: dict[str, LandmarkAdapter] | None = None,
) -> tuple[int, int]:
    """Run selected model adapters from a manifest into the disk cache."""
    samples = load_manifest(manifest_path)
    raw_entries = {_entry_id(entry): entry for entry in _load_manifest_entries(manifest_path)}
    cache = DiskPredictionCache(cache_dir)
    loaded_adapters = (
        adapters if adapters is not None else _build_model_adapters(models, device=device)
    )
    written = 0
    reused = 0
    show_progress = _show_progress()
    for model in models:
        adapter = loaded_adapters[model]
        items, model_reused = _prepare_run_items(
            samples=samples,
            entries=raw_entries,
            model=model,
            adapter=adapter,
            cache=cache,
            checkpoint=checkpoint,
            refresh=refresh,
            allow_gt_roi=allow_gt_roi,
            gt_roi_scale=gt_roi_scale,
            device=device,
        )
        reused += model_reused
        if model_reused:
            logger.debug("Reused %d fresh cached %s predictions", model_reused, model)
        input_size = int(getattr(getattr(adapter, "plugin", None), "input_size", 256))
        progress = tqdm(
            total=len(items),
            desc=f"Cache predictions [{model}]",
            unit="sample",
            disable=not show_progress or not items,
        )
        try:
            for start in range(0, len(items), batch_size):
                batch_items = items[start : start + batch_size]
                images = []
                matrices = []
                for item in batch_items:
                    frame = _load_image_bgr(item.sample.image)
                    images.append(_crop_square(frame, item.model_roi, input_size))
                    matrices.append(item.matrix)
                batch = np.stack(images, axis=0)
                batch_matrices = np.stack(matrices, axis=0).astype("float32", copy=False)
                predictions = adapter.predict_batch(batch, matrices=batch_matrices)
                if len(predictions) != len(batch_items):
                    raise ValueError(
                        f"adapter '{model}' returned {len(predictions)} predictions for batch of {len(batch_items)}"  # noqa: E501
                    )
                for item, prediction in zip(batch_items, predictions, strict=True):
                    canonical = prediction.canonical_68()
                    if canonical.coordinate_space != "frame":
                        raise ValueError(
                            f"adapter '{model}' returned coordinate space '{canonical.coordinate_space}', expected frame"  # noqa: E501
                        )
                    cache.write(
                        item.sample.sample_id,
                        canonical,
                        checkpoint=checkpoint,
                        config=item.config,
                        refresh=True,
                    )
                    written += 1
                progress.update(len(batch_items))
        finally:
            progress.close()
    return written, reused


def _import_manifest_predictions(args: argparse.Namespace) -> tuple[int, int]:
    """Import existing prediction arrays referenced by a manifest."""
    roots = _parse_prediction_roots(args.prediction_root)
    manifest_path = Path(args.manifest).expanduser().resolve()
    samples = load_manifest(manifest_path)
    entries = {_entry_id(entry): entry for entry in _load_manifest_entries(manifest_path)}
    cache = DiskPredictionCache(args.cache_dir)
    written = 0
    reused = 0
    models = _split_models(args.models)
    for sample in tqdm(
        samples,
        desc="Import predictions",
        unit="sample",
        disable=not _show_progress(),
    ):
        entry = entries.get(sample.sample_id, {})
        for model in models:
            path = _prediction_path(
                sample_id=sample.sample_id,
                model=model,
                manifest_path=manifest_path,
                entry=entry,
                prediction_roots=roots,
            )
            changed = _write_prediction(
                cache,
                sample.sample_id,
                model,
                path,
                schema=args.schema,
                coordinate_space=args.coordinate_space,
                checkpoint=_checkpoint_value(args),
                refresh=args.refresh,
            )
            written += int(changed)
            reused += int(not changed)
    return written, reused


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--prediction-root", action="append", default=[])
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--prediction", default="")
    parser.add_argument("--schema", default="2d_68")
    parser.add_argument("--coordinate-space", default="frame")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Run selected model adapters against --manifest instead of importing .npy predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for model adapters, e.g. auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--no-gt-roi",
        action="store_true",
        help="Require face_bbox/bbox in the manifest instead of deriving validation ROIs from GT landmarks.",
    )
    parser.add_argument(
        "--gt-roi-scale",
        type=float,
        default=1.0,
        help="Optional validation-only scale for GT-derived ROIs before model preprocessing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.gt_roi_scale <= 0:
        raise SystemExit("--gt-roi-scale must be greater than zero")

    if args.manifest:
        if not args.models:
            raise SystemExit("--models is required with --manifest")
        if args.run_models:
            written, reused = _run_model_predictions(
                manifest_path=Path(args.manifest).expanduser().resolve(),
                models=_split_models(args.models),
                cache_dir=Path(args.cache_dir),
                checkpoint=_checkpoint_value(args),
                batch_size=args.batch_size,
                device=args.device,
                refresh=args.refresh,
                allow_gt_roi=not args.no_gt_roi,
                gt_roi_scale=args.gt_roi_scale,
            )
        else:
            written, reused = _import_manifest_predictions(args)
        print(f"Cached predictions: wrote={written} reused={reused}")
        return 0

    if args.run_models:
        raise SystemExit("--run-models requires --manifest")
    if not (args.sample_id and args.model and args.prediction):
        raise SystemExit("--sample-id, --model, and --prediction are required without --manifest")
    cache = DiskPredictionCache(args.cache_dir)
    changed = _write_prediction(
        cache,
        args.sample_id,
        args.model.strip().lower(),
        Path(args.prediction).expanduser().resolve(),
        schema=args.schema,
        coordinate_space=args.coordinate_space,
        checkpoint=_checkpoint_value(args),
        refresh=args.refresh,
    )
    message = (
        "Cached predictions: wrote=1 reused=0"
        if changed
        else "Cached predictions: wrote=0 reused=1"
    )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
