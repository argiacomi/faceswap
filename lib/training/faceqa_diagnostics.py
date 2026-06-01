#!/usr/bin/env python3
"""FaceQA training metadata and rolling loss diagnostics."""

from __future__ import annotations

import json
import logging
import os
import typing as T
from dataclasses import dataclass

import torch

from lib.align.objects import FileAlignments, PNGHeader
from lib.faceqa.buckets import (
    blur_bucket,
    expression_bucket,
    is_identity_outlier,
    lighting_bucket,
    mask_qa_bucket,
    pitch_bucket,
    pose_bucket,
    resolution_bucket,
)
from lib.faceqa.coverage import record_from_alignment
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from .loss import BatchLoss

logger = logging.getLogger(__name__)

EMA_ALPHA = 0.05
WORST_BUCKET_LIMIT = 5
LOG_INTERVAL = 100

LOW_CARDINALITY_DIMENSIONS = frozenset(
    {
        "side",
        "yaw_pose",
        "pitch",
        "expression",
        "lighting",
        "blur",
        "resolution",
        "mask_qa",
        "duplicate",
        "identity_outlier",
    }
)


@dataclass(frozen=True)
class FaceQASampleMetadata:
    """Compact FaceQA metadata for one training sample."""

    side: str
    filename: str
    source_file: str
    source_id: str
    face_index: int
    has_faceqa: bool = False
    yaw_pose_bucket: str = "unknown"
    pitch_bucket: str = "unknown"
    expression_bucket: str = "unknown"
    lighting_bucket: str = "unknown"
    blur_bucket: str = "unknown"
    resolution_bucket: str = "unknown"
    mask_qa_bucket: str = "unknown"
    duplicate_bucket: str = "unknown"
    identity_outlier_bucket: str = "unknown"

    @classmethod
    def missing(cls, side: str, filename: str) -> T.Self:  # type: ignore[name-defined]
        """Return a sample placeholder when no FaceQA metadata is available."""
        basename = os.path.basename(filename)
        return cls(
            side=side.upper(),
            filename=filename,
            source_file=basename,
            source_id=basename,
            face_index=-1,
        )

    @classmethod
    def from_png_header(cls, side: str, filename: str, header: PNGHeader) -> T.Self:  # type: ignore[name-defined]
        """Build metadata from the FaceSwap PNG header embedded in a training image."""
        source_file = header.source.source_filename or os.path.basename(filename)
        face_index = int(header.source.face_index)
        alignment = header.alignments
        faceqa_payload = alignment.metadata.get("faceqa")
        if not isinstance(faceqa_payload, dict) or not faceqa_payload:
            return cls(
                side=side.upper(),
                filename=filename,
                source_file=source_file,
                source_id=f"{source_file}:{face_index}",
                face_index=face_index,
            )
        face = FileAlignments(
            x=alignment.x,
            y=alignment.y,
            w=alignment.w,
            h=alignment.h,
            landmarks_xy=alignment.landmarks_xy,
            mask=alignment.mask,
            identity=alignment.identity,
            metadata=alignment.metadata,
            thumb=None,
        )
        record = record_from_alignment(source_file, face_index, face)
        duplicate = "unknown"
        if record.duplicate_cluster_id or record.duplicate_cluster is not None:
            duplicate = "duplicate"
        else:
            duplicate = "unique"
        outlier = "outlier" if is_identity_outlier(record) else "inlier"
        source_id = f"{source_file}:{face_index}"
        return cls(
            side=side.upper(),
            filename=filename,
            source_file=source_file,
            source_id=source_id,
            face_index=face_index,
            has_faceqa=True,
            yaw_pose_bucket=pose_bucket(record),
            pitch_bucket=pitch_bucket(record),
            expression_bucket=expression_bucket(record),
            lighting_bucket=lighting_bucket(record),
            blur_bucket=blur_bucket(record),
            resolution_bucket=resolution_bucket(record),
            mask_qa_bucket=mask_qa_bucket(record),
            duplicate_bucket=duplicate,
            identity_outlier_bucket=outlier,
        )

    def dimensions(self) -> dict[str, str]:
        """Return dimension-to-bucket labels for aggregation."""
        return {
            "side": self.side,
            "source_file": self.source_file,
            "source_id": self.source_id,
            "yaw_pose": self.yaw_pose_bucket,
            "pitch": self.pitch_bucket,
            "expression": self.expression_bucket,
            "lighting": self.lighting_bucket,
            "blur": self.blur_bucket,
            "resolution": self.resolution_bucket,
            "mask_qa": self.mask_qa_bucket,
            "duplicate": self.duplicate_bucket,
            "identity_outlier": self.identity_outlier_bucket,
        }


@dataclass
class _BucketLoss:
    """Rolling loss state for one side/dimension/bucket."""

    side: str
    dimension: str
    bucket: str
    count: int = 0
    current: float = 0.0
    mean: float = 0.0
    ema: float | None = None
    previous_ema: float | None = None

    @property
    def trend(self) -> float:
        """Return current EMA trend. Positive values indicate worsening loss."""
        if self.ema is None or self.previous_ema is None:
            return 0.0
        return self.ema - self.previous_ema

    def update(self, value: float) -> None:
        """Update the rolling loss state."""
        self.count += 1
        self.current = value
        self.mean += (value - self.mean) / self.count
        self.previous_ema = self.ema
        self.ema = value if self.ema is None else EMA_ALPHA * value + (1.0 - EMA_ALPHA) * self.ema

    def to_dict(self) -> dict[str, T.Any]:
        """Return a JSON-serializable diagnostic row."""
        return {
            "side": self.side,
            "dimension": self.dimension,
            "bucket": self.bucket,
            "count": self.count,
            "current": self.current,
            "mean": self.mean,
            "ema": self.ema,
            "trend": self.trend,
        }


class FaceQALossDiagnostics:
    """Aggregate per-sample training losses by FaceQA metadata buckets."""

    def __init__(
        self, jsonl_path: str | None = None, worst_limit: int = WORST_BUCKET_LIMIT
    ) -> None:
        self._jsonl_path = jsonl_path
        self._worst_limit = worst_limit
        self._buckets: dict[tuple[str, str, str], _BucketLoss] = {}

    @staticmethod
    def _side_losses(loss: BatchLoss) -> list[float]:
        """Return one weighted scalar loss per sample for a side."""
        per_output = [
            T.cast(torch.Tensor, sum(parts.values())).detach().cpu() for parts in loss.weighted
        ]
        total = T.cast(torch.Tensor, sum(per_output))
        if loss.mask is not None:
            total += loss.mask.detach().cpu()
        return [float(value) for value in total.tolist()]

    def _update_bucket(self, side: str, dimension: str, bucket: str, loss: float) -> None:
        """Update one bucket accumulator."""
        key = (side, dimension, bucket)
        if key not in self._buckets:
            self._buckets[key] = _BucketLoss(side=side, dimension=dimension, bucket=bucket)
        self._buckets[key].update(loss)

    def _ranked(
        self, key: T.Callable[[_BucketLoss], float], *, include_high_cardinality: bool = False
    ) -> list[_BucketLoss]:
        """Return top bucket diagnostics by a score function."""
        buckets = [
            bucket
            for bucket in self._buckets.values()
            if include_high_cardinality or bucket.dimension in LOW_CARDINALITY_DIMENSIONS
        ]
        return sorted(buckets, key=key, reverse=True)[: self._worst_limit]

    def _summary(
        self,
        *,
        iteration: int,
        metadata_count: int,
        total_count: int,
    ) -> dict[str, T.Any]:
        """Return the current compact diagnostic snapshot."""
        side_stats = {
            bucket.bucket: bucket
            for bucket in self._buckets.values()
            if bucket.dimension == "side" and bucket.ema is not None
        }
        side_a = side_stats.get("A")
        side_b = side_stats.get("B")
        imbalance = (
            None
            if side_a is None or side_b is None or side_a.ema is None or side_b.ema is None
            else abs(side_a.ema - side_b.ema)
        )
        worst_current = self._ranked(lambda bucket: bucket.ema or 0.0)
        worst_trend = self._ranked(lambda bucket: bucket.trend)
        worst_sources = self._ranked(
            lambda bucket: bucket.ema or 0.0, include_high_cardinality=True
        )
        worst_sources = [
            bucket for bucket in worst_sources if bucket.dimension in {"source_file", "source_id"}
        ][: self._worst_limit]
        return {
            "iteration": iteration,
            "metadata_samples": metadata_count,
            "total_samples": total_count,
            "metadata_coverage": metadata_count / total_count if total_count else 0.0,
            "side_ema": {
                side: bucket.ema for side, bucket in side_stats.items() if bucket.ema is not None
            },
            "ab_reconstruction_imbalance": imbalance,
            "worst_current": [bucket.to_dict() for bucket in worst_current],
            "worst_trend": [bucket.to_dict() for bucket in worst_trend],
            "worst_sources": [bucket.to_dict() for bucket in worst_sources],
        }

    def _write_jsonl(self, payload: dict[str, T.Any]) -> None:
        """Append one diagnostic snapshot to the JSONL stream."""
        if not self._jsonl_path:
            return
        os.makedirs(os.path.dirname(self._jsonl_path), exist_ok=True)
        with open(self._jsonl_path, "a", encoding="utf-8") as out_file:
            out_file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _tensorboard_scalars(self, payload: dict[str, T.Any]) -> dict[str, float]:
        """Return bounded TensorBoard scalars for the current snapshot."""
        scalars = {
            "metadata_coverage": float(payload["metadata_coverage"]),
            "metadata_samples": float(payload["metadata_samples"]),
        }
        imbalance = payload["ab_reconstruction_imbalance"]
        if imbalance is not None:
            scalars["ab_reconstruction_imbalance"] = float(imbalance)
        for idx, bucket in enumerate(payload["worst_current"], start=1):
            scalars[f"worst_current_loss_{idx}"] = float(bucket["ema"] or 0.0)
            scalars[f"worst_current_count_{idx}"] = float(bucket["count"])
        for idx, bucket in enumerate(payload["worst_trend"], start=1):
            scalars[f"worst_trend_delta_{idx}"] = float(bucket["trend"])
            scalars[f"worst_trend_count_{idx}"] = float(bucket["count"])
        return scalars

    def update(
        self,
        losses: list[BatchLoss],
        metadata: list[list[FaceQASampleMetadata]] | None,
        iteration: int,
    ) -> dict[str, float]:
        """Update diagnostics and return bounded TensorBoard scalars."""
        if metadata is None:
            return {}
        metadata_count = 0
        total_count = 0
        for side_loss, side_metadata in zip(losses, metadata, strict=False):
            side_values = self._side_losses(side_loss)
            for sample_loss, sample in zip(side_values, side_metadata, strict=False):
                total_count += 1
                if not sample.has_faceqa:
                    continue
                metadata_count += 1
                for dimension, bucket in sample.dimensions().items():
                    self._update_bucket(sample.side, dimension, bucket, sample_loss)
        if metadata_count == 0:
            return {}
        payload = self._summary(
            iteration=iteration,
            metadata_count=metadata_count,
            total_count=total_count,
        )
        self._write_jsonl(payload)
        if iteration % LOG_INTERVAL == 0:
            worst = payload["worst_current"][:3]
            logger.info(
                "FaceQA training diagnostics: metadata %.1f%%, A/B imbalance=%s, worst=%s",
                payload["metadata_coverage"] * 100.0,
                payload["ab_reconstruction_imbalance"],
                [
                    f"{item['side']}:{item['dimension']}={item['bucket']} "
                    f"ema={item['ema']:.5f} n={item['count']}"
                    for item in worst
                ],
            )
        return self._tensorboard_scalars(payload)


__all__ = get_module_objects(__name__)
