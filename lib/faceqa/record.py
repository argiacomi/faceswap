#!/usr/bin/env python3
"""In-memory FaceQA record dataclass.

FaceQA enrichment persists into ``face.metadata['faceqa']``
directly - so the record dataclass moves to a neutral home
in the FaceQA package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lib.align.objects import DataclassDict
from lib.utils import get_module_objects


@dataclass(repr=False)
class FaceQARecord(DataclassDict):
    """In-memory quality-assessment record for one extracted face.

    The record is built per FaceQA pass from the alignments file plus the
    FaceQA metadata envelope in ``face.metadata['faceqa']``; it is the unit
    consumed by coverage, redundancy, readiness, and compatibility.
    """

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
    image_metrics_provenance: str | None = None
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


__all__ = get_module_objects(__name__)
