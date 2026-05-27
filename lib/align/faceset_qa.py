#!/usr/bin/env python3
"""Faceset QA sidecar schema and I/O helpers."""

from __future__ import annotations

import os
import typing as T
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone

from lib.serializer import get_serializer
from lib.utils import get_module_objects

from .objects import DataclassDict


@dataclass(repr=False)
class FaceQARecord(DataclassDict):
    """Quality-assessment metadata for one extracted face."""

    frame: str = ""
    face_index: int = 0

    identity_score: float | None = None
    identity_cluster: int | None = None
    identity_model: str | None = None
    identity_consensus: float | None = None
    identity_threshold_used: float | None = None
    identity_quality_flag: str | None = None

    identity_primary_model: str | None = None
    identity_verifier_model: str | None = None
    identity_primary_score: float | None = None
    identity_verifier_score: float | None = None
    identity_agreement: str | None = None
    identity_decision_reason: str | None = None
    identity_verifier_trigger: str | None = None
    identity_final_decision: str | None = None

    quality_score: float | None = None
    quality_flags: list[str] = field(default_factory=list)

    duplicate_cluster: int | None = None
    recommended_keep: bool | None = None
    duplicate_cluster_id: str | None = None
    duplicate_representative: bool | None = None
    duplicate_keep_recommendation: str | None = None
    duplicate_score: float | None = None
    duplicate_cluster_size: int | None = None
    duplicate_reason: str | None = None

    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    pose_source: str | None = None
    pose_model: str | None = None
    pose_confidence: str | None = None
    pose_delta_yaw: float | None = None
    pose_delta_pitch: float | None = None
    pose_delta_roll: float | None = None
    pose_max_abs_delta: float | None = None
    alignment_yaw: float | None = None
    alignment_pitch: float | None = None
    alignment_roll: float | None = None
    spiga_yaw: float | None = None
    spiga_pitch: float | None = None
    spiga_roll: float | None = None
    spiga_pose_source: str | None = None
    spiga_pose_model: str | None = None
    spiga_pose_error: str | None = None
    average_distance: float | None = None

    blur_score: float | None = None
    blur_fft_score: float | None = None
    black_pixel_ratio: float | None = None
    color_gray: float | None = None
    color_luma: float | None = None
    color_green: float | None = None
    color_orange: float | None = None
    mean_luminance: float | None = None
    luminance_variance: float | None = None
    contrast: float | None = None
    left_right_ratio: float | None = None
    top_bottom_ratio: float | None = None
    saturation: float | None = None
    color_warmth: float | None = None
    resolution: list[int] = field(default_factory=list)

    occlusion_score: float | None = None
    expression_bucket: str | None = None
    mouth_openness: float | None = None
    mouth_width_ratio: float | None = None
    smile_proxy: float | None = None
    eye_closure: float | None = None
    brow_raise_proxy: float | None = None
    expression_asymmetry: float | None = None

    mask_qa_ref: str | None = None

    reviewed_state: str | None = None
    manual_decision: str | None = None
    manual_decision_reason: str | None = None


@dataclass(repr=False)
class FaceQAFile(DataclassDict):
    """Root object for a ``*_faceset_qa.json`` sidecar file."""

    schema_version: int = 1
    generated_by: str = "faceswap"
    generated_at: str = ""
    faces: list[FaceQARecord] = field(default_factory=list)


def sidecar_path(alignments_path: str) -> str:
    """Return the default FaceQA sidecar path for an alignments file."""
    stem, _ = os.path.splitext(alignments_path)
    return f"{stem}_faceset_qa.json"


def _known_fields(cls: type[DataclassDict]) -> set[str]:
    """Return dataclass field names for tolerant sidecar loading."""
    return {field_.name for field_ in fields(cls)}


def load(path: str) -> FaceQAFile | None:
    """Load a FaceQA sidecar, returning ``None`` when the file is absent.

    Additive fields from newer sidecars are ignored so this initial consumer can
    read reports produced by newer FaceQA tooling without crashing.
    """
    if not os.path.isfile(path):
        return None

    raw: dict[str, T.Any] = get_serializer("json").load(path)
    root_fields = _known_fields(FaceQAFile)
    record_fields = _known_fields(FaceQARecord)
    root = {key: val for key, val in raw.items() if key in root_fields and key != "faces"}
    faces = [
        FaceQARecord.from_dict({key: val for key, val in item.items() if key in record_fields})
        for item in raw.get("faces", [])
        if isinstance(item, dict)
    ]
    return FaceQAFile(faces=faces, **root)


def save(path: str, qa_file: FaceQAFile, *, timestamp: bool = True) -> None:
    """Write a FaceQA sidecar."""
    if timestamp:
        qa_file.generated_at = datetime.now(timezone.utc).isoformat()
    get_serializer("json").save(path, qa_file.to_dict())


def build_index(qa_file: FaceQAFile) -> dict[tuple[str, int], FaceQARecord]:
    """Return sidecar records keyed by ``(frame, face_index)``."""
    return {(record.frame, record.face_index): record for record in qa_file.faces}


__all__ = get_module_objects(__name__)
