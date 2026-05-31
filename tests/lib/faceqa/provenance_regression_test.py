#!/usr/bin/env python3
"""Regression tests for FaceQA provenance and identity guardrails."""

from __future__ import annotations

import json
import typing as T

import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.coverage import (
    IMAGE_METRICS_PROVENANCE_FRAME,
    IMAGE_METRICS_PROVENANCE_THUMBNAIL,
    compute_coverage,
    records_from_alignments,
)
from lib.faceqa.readiness import generate_readiness_report
from lib.faceqa.record import FaceQARecord
from lib.faceqa.redundancy import compute_redundancy
from lib.serializer import get_serializer


def _face(width: int = 80, height: int = 90) -> FileAlignments:
    """Return a minimal alignments face."""
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    return FileAlignments(x=0, y=0, w=width, h=height, landmarks_xy=landmarks)


def test_stale_top_level_image_metrics_are_overridden_by_frame_metrics(tmp_path) -> None:
    """Authoritative frame metrics should replace stale embedded top-level fields."""
    face = _face()
    face.metadata = {
        "faceqa": {
            "blur_score": 1.0,
            "mean_luminance": 10.0,
            "contrast": 2.0,
            "image_metrics_provenance": IMAGE_METRICS_PROVENANCE_THUMBNAIL,
            "image_metrics": {
                "provenance": IMAGE_METRICS_PROVENANCE_THUMBNAIL,
                "blur_score": 1.0,
                "mean_luminance": 10.0,
                "contrast": 2.0,
            },
        }
    }
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {"frame_000001.png": AlignmentsEntry(faces=[face]).to_dict()},
        },
    )

    def frame_metrics(_: FileAlignments, __: str, ___: int) -> dict[str, T.Any]:
        return {
            "provenance": IMAGE_METRICS_PROVENANCE_FRAME,
            "blur_score": 9.0,
            "mean_luminance": 200.0,
            "contrast": 33.0,
        }

    records = records_from_alignments(alignments, metrics_backfiller=frame_metrics)

    assert records[0].image_metrics_provenance == IMAGE_METRICS_PROVENANCE_FRAME
    assert records[0].blur_score == 9.0
    assert records[0].mean_luminance == 200.0
    assert records[0].contrast == 33.0


def test_fallback_provenance_downweights_representative_quality() -> None:
    """Fallback-derived blur terms should not dominate representative choice."""
    records = [
        FaceQARecord(
            frame="frame_000001.png",
            face_index=0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            blur_score=20.0,
            average_distance=0.0,
            resolution=[96, 96],
            image_metrics_provenance=IMAGE_METRICS_PROVENANCE_THUMBNAIL,
            identity_final_decision="inlier",
        ),
        FaceQARecord(
            frame="frame_000002.png",
            face_index=0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            blur_score=10.0,
            average_distance=0.0,
            resolution=[96, 96],
            image_metrics_provenance=IMAGE_METRICS_PROVENANCE_FRAME,
            identity_final_decision="inlier",
        ),
    ]

    report = compute_redundancy(records, aggressiveness="aggressive")
    representatives = [record for record in report.records if record.representative]

    assert len(representatives) == 1
    assert representatives[0].frame == "frame_000002.png"


def test_fallback_lighting_bucket_penalty_is_reduced() -> None:
    """Fallback lighting differences should not force a prune recommendation."""
    common = dict(
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        mouth_openness=0.1,
        smile_proxy=0.1,
        eye_closure=0.0,
        expression_bucket="neutral",
        identity_final_decision="inlier",
    )
    records = [
        FaceQARecord(
            frame="frame_000001.png",
            face_index=0,
            mean_luminance=59.0,
            contrast=10.0,
            left_right_ratio=1.0,
            top_bottom_ratio=1.0,
            color_warmth=0.0,
            saturation=50.0,
            image_metrics_provenance=IMAGE_METRICS_PROVENANCE_THUMBNAIL,
            **common,  # type: ignore[arg-type]
        ),
        FaceQARecord(
            frame="frame_000002.png",
            face_index=0,
            mean_luminance=61.0,
            contrast=10.0,
            left_right_ratio=1.0,
            top_bottom_ratio=1.0,
            color_warmth=0.0,
            saturation=50.0,
            image_metrics_provenance=IMAGE_METRICS_PROVENANCE_THUMBNAIL,
            **common,  # type: ignore[arg-type]
        ),
    ]

    report = compute_redundancy(records, aggressiveness="aggressive")
    recommendations = {record.frame: record.recommendation for record in report.records}

    assert report.cluster_count == 1
    assert recommendations["frame_000002.png"] == "keep"


def test_unclassified_identity_embeddings_route_to_review() -> None:
    """Vectors without identity QA decisions should not satisfy the pruning guardrail."""
    records = [
        FaceQARecord(
            frame="frame_000001.png",
            face_index=0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            mouth_openness=0.1,
            smile_proxy=0.1,
            eye_closure=0.0,
            expression_bucket="neutral",
            identity_model="arcface",
        ),
        FaceQARecord(
            frame="frame_000002.png",
            face_index=0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            mouth_openness=0.1,
            smile_proxy=0.1,
            eye_closure=0.0,
            expression_bucket="neutral",
            identity_model="arcface",
        ),
    ]

    report = compute_redundancy(records, aggressiveness="aggressive")

    assert report.cluster_count == 1
    assert any(record.recommendation == "review" for record in report.records)
    assert all(not record.has_identity for record in report.records)


def test_readiness_json_includes_identity_coverage_ratios() -> None:
    """Identity signal ratios should be exposed in readiness JSON and Markdown."""
    coverage = compute_coverage(
        [
            FaceQARecord(
                frame="frame_000001.png",
                face_index=0,
                identity_model="arcface",
                identity_final_decision="inlier",
            ),
            FaceQARecord(frame="frame_000002.png", face_index=0),
        ]
    )

    report = generate_readiness_report(coverage)
    payload = json.loads(report.to_json())
    markdown = report.to_markdown()

    assert payload["identity_embedding_coverage_ratio"] == 0.5
    assert payload["identity_decision_coverage_ratio"] == 0.5
    assert payload["identity_unknown_ratio"] == 0.5
    assert "Identity signal coverage" in markdown
