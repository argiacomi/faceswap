#!/usr/bin/env python3
"""Canonical manifest + bbox / visibility IO for the landmark pipeline.

Before this module existed each layer had its own near-copy of the bbox
parser: ``harness._coerce_bbox`` coerced a 4-tuple; ``geometry_metrics
._normalize_bbox`` added xywh fallback; ``cache_predictions._bbox_values``
handled dict payloads plus xywh too. The three implementations drifted —
COFW-68 manifests storing ``(x, y, w, h)`` flowed cleanly through one
layer but emerged as a degenerate ``(x, y, w, h)`` ltrb downstream.

This module owns the canonical ltrb coercion, the visibility coercion, and
the :class:`LandmarkSample` / :func:`load_manifest` IO. Callers that used
to consume one of the legacy helpers should import from here instead.
"""

from __future__ import annotations

import json
import logging
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Schemas that ``lib.landmarks.core.schema.to_canonical_68`` can remap to the
# canonical 68-point layout. Ground truth in any other schema (for example the
# MenpoBenchmark 39-point profile set) cannot be scored against 68-point model
# output and must be skipped by canonical-68 eval/scorer/static-weight paths.
CANONICAL_68_COMPATIBLE_SCHEMAS = frozenset({"2d_68", "2d_98"})


@dataclass(frozen=True)
class LandmarkSample:
    """One landmark evaluation manifest entry, with manifest metadata attached.

    ``face_bbox`` is always coerced to ``(left, top, right, bottom)`` order
    (xywh inputs are converted at load time). ``visibility`` is a 68-bool
    tuple when the manifest carries per-landmark visibility flags; otherwise
    ``None``.
    """

    sample_id: str
    image: str
    landmarks: str
    dataset: str = ""
    condition: str = ""
    source_schema: str = ""
    normalizer: float | None = None
    face_bbox: tuple[float, float, float, float] | None = None
    visibility: tuple[bool, ...] | None = None
    metadata: dict[str, T.Any] = field(default_factory=dict)


def coerce_bbox(value: T.Any) -> tuple[float, float, float, float] | None:
    """Coerce a manifest bbox payload to ``(left, top, right, bottom)``.

    Accepts:

    * ``None`` → ``None``.
    * Mappings with ``left/top/right/bottom`` keys.
    * Mappings with ``x/y/w/h`` keys.
    * 4+ length sequences. If the third/fourth values look like width/height
      (i.e. they would produce a non-positive bbox if interpreted as right /
      bottom) the inputs are treated as xywh and converted.

    Anything else (non-iterable, fewer than 4 numeric values, all-zero
    width/height) returns ``None``. The output is always positive-width
    ltrb so downstream geometry code never has to second-guess the shape.
    """
    if value is None:
        return None
    if isinstance(value, T.Mapping):
        keys = set(value)
        if {"left", "top", "right", "bottom"}.issubset(keys):
            try:
                return tuple(float(value[key]) for key in ("left", "top", "right", "bottom"))  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None
        if {"x", "y", "w", "h"}.issubset(keys):
            try:
                left = float(value["x"])
                top = float(value["y"])
                width = float(value["w"])
                height = float(value["h"])
            except (TypeError, ValueError):
                return None
            if width <= 0 or height <= 0:
                return None
            return (left, top, left + width, top + height)
        return None
    try:
        flat = np.asarray(value, dtype="float64").reshape(-1)
    except (TypeError, ValueError):
        return None
    if flat.size < 4:
        return None
    left, top, third, fourth = (float(item) for item in flat[:4])
    if third > left and fourth > top:
        return (left, top, third, fourth)
    # Looks like xywh: third = width, fourth = height.
    if third > 0 and fourth > 0:
        return (left, top, left + third, top + fourth)
    return None


def coerce_visibility(value: T.Any) -> tuple[bool, ...] | None:
    """Coerce a manifest visibility payload to a bool tuple, or ``None``."""
    if value is None:
        return None
    try:
        flags = tuple(bool(item) for item in value)
    except TypeError:
        return None
    if not flags:
        return None
    return flags


def bbox_from_truth_fallback(truth: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return the axis-aligned bbox of ``truth`` landmarks as the fallback bbox.

    Used by analysis tools that need *some* bbox for normalization when the
    manifest doesn't carry one; consumers that need a real detector bbox
    should fail loudly rather than rely on this fallback.
    """
    points = np.asarray(truth, dtype="float64")
    if points.ndim != 2 or points.shape[1] < 2 or points.size == 0:
        return None
    left, top = np.min(points, axis=0)[:2]
    right, bottom = np.max(points, axis=0)[:2]
    if right <= left or bottom <= top:
        return None
    return (float(left), float(top), float(right), float(bottom))


def load_manifest(path: str | Path) -> list[LandmarkSample]:
    """Load a manifest JSON file into :class:`LandmarkSample` records.

    Relative ``image`` / ``landmarks`` paths are resolved against the manifest
    file's parent so callers can work with absolute paths regardless of where
    the manifest lives.
    """
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    samples: list[LandmarkSample] = []
    for entry in payload.get("samples", payload.get("scenarios", [])):
        landmarks = str(entry.get("landmarks") or entry.get("ground_truth") or "")
        if not landmarks:
            raise ValueError(f"manifest entry {entry!r} missing landmarks path")
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        bbox = coerce_bbox(entry.get("face_bbox", metadata.get("face_bbox")))
        if bbox is None:
            bbox = coerce_bbox(entry.get("bbox", metadata.get("bbox")))
        visibility = coerce_visibility(entry.get("visibility", metadata.get("visibility")))
        samples.append(
            LandmarkSample(
                sample_id=str(entry.get("sample_id") or entry.get("id") or entry.get("name")),
                image=str((base / str(entry.get("image", ""))).resolve()),
                landmarks=str((base / landmarks).resolve()),
                dataset=str(entry.get("dataset", "")),
                condition=str(entry.get("condition", entry.get("scenario", ""))),
                source_schema=str(entry.get("source_schema", metadata.get("source_schema", ""))),
                normalizer=entry.get("normalizer", metadata.get("normalizer")),
                face_bbox=bbox,
                visibility=visibility,
                metadata=dict(metadata),
            )
        )
    return samples


def bbox_for_sample(
    sample: LandmarkSample, *, allow_truth_fallback: bool = True
) -> tuple[float, float, float, float] | None:
    """Resolve a usable bbox for ``sample``.

    Returns the manifest-provided ``face_bbox`` first, then falls back to the
    extent of the GT landmarks when ``allow_truth_fallback`` is True. Returns
    ``None`` when no bbox can be determined (missing truth file etc.).
    """
    if sample.face_bbox is not None:
        return sample.face_bbox
    if not allow_truth_fallback:
        return None
    try:
        truth = np.load(sample.landmarks).astype("float32")
    except OSError:
        return None
    return bbox_from_truth_fallback(truth)


def sample_is_canonical_68(sample: LandmarkSample) -> bool:
    """Return ``True`` if the sample ground truth can map to the canonical 68 points.

    The decision is based on the *shape* of the stored GT array (68 points, or 98
    points which :func:`lib.landmarks.core.schema.to_canonical_68` remaps), not the
    declared ``source_schema`` label. The label is unreliable for this purpose: for
    example AFLW2000-3D ground truth is labelled ``3d_68`` but is stored as a
    ``(68, 2)`` array that canonical-68 scoring handles fine, while the MenpoBenchmark
    39-point profile set is stored as ``(39, 2)`` and cannot be scored against
    68-point model output.
    """
    try:
        truth = np.load(sample.landmarks)
    except OSError:
        return False
    return getattr(truth, "ndim", 0) == 2 and int(truth.shape[0]) in (68, 98)


def filter_canonical_68_samples(
    samples: T.Sequence[LandmarkSample], *, context: str = ""
) -> list[LandmarkSample]:
    """Drop samples whose GT cannot be scored against canonical 68-point output.

    Canonical-68 NME / scorer / static-weight code only understands 68-point (or
    98-point, which is remapped to 68) ground truth. Samples in any other schema are
    skipped with a single summarizing warning so that a mixed manifest does not crash
    these paths. The schema is not silently coerced.
    """
    kept: list[LandmarkSample] = []
    skipped_by_schema: dict[str, int] = {}
    for sample in samples:
        if sample_is_canonical_68(sample):
            kept.append(sample)
        else:
            key = (sample.source_schema or "unknown").strip() or "unknown"
            skipped_by_schema[key] = skipped_by_schema.get(key, 0) + 1
    if skipped_by_schema:
        where = f" in {context}" if context else ""
        logger.warning(
            "Skipping %d non-canonical-68 ground-truth sample(s)%s (counts by schema: %s). "
            "Canonical-68 NME/scorer/static-weight paths only support 68/98-point ground "
            "truth; opt out of 39-point profile data or add real partial-schema metrics.",
            sum(skipped_by_schema.values()),
            where,
            skipped_by_schema,
        )
    return kept


__all__ = [
    "CANONICAL_68_COMPATIBLE_SCHEMAS",
    "LandmarkSample",
    "bbox_for_sample",
    "bbox_from_truth_fallback",
    "coerce_bbox",
    "coerce_visibility",
    "filter_canonical_68_samples",
    "load_manifest",
    "sample_is_canonical_68",
]
