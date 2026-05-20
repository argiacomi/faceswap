#!/usr/bin/env python3
"""Tests for exporting reviewed production Faceswap alignment manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer
from tools.landmarks.build_faceswap_validated_manifest import build_manifest, main

_LEGACY_LANDMARK_POSE_BUCKET_KEY = "_".join(("landmark", "pose", "bucket"))


def _points(offset: float = 0.0) -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")
    points[:, 0] = np.linspace(20.0 + offset, 120.0 + offset, 68)
    points[:, 1] = np.linspace(30.0, 160.0, 68)
    return points


def _write_alignments(path: Path) -> None:
    serializer = get_serializer("compressed")
    payload = {
        "__meta__": {"version": 2.4},
        "__data__": {
            "IMG_000123.png": AlignmentsEntry(
                faces=[
                    FileAlignments(
                        x=10,
                        y=20,
                        w=100,
                        h=120,
                        landmarks_xy=_points(),
                        metadata={
                            "landmark_ensemble": {
                                "selected_candidate": "spiga",
                                "runtime_bucket": "profile_left",
                                "bucket": "profile_left",
                                "candidate_priority": ["spiga", "orformer", "hrnet"],
                                "vetoed": ["hrnet"],
                                "veto_reasons": {"hrnet": ["roll_veto"]},
                                "roll_estimate": 12.3,
                                "yaw_estimate": -48.0,
                                "risk_route": "high_risk",
                                "dominant_candidate_yaw": -52.0,
                                "max_disagreement_px": 14.2,
                                "max_disagreement_bbox_fraction": 0.08,
                                "runtime_bucket_side": "left",
                                "runtime_bucket_side_source": "image_geometry",
                                "runtime_bucket_severity": "profile",
                                "runtime_bucket_severity_source": "image_geometry",
                                "cloud_area_ratio": {"spiga": 0.72, "hrnet": 0.91},
                                "hull_area_ratio": {"spiga": 0.34, "hrnet": 0.37},
                                "points_outside_expanded_bbox_fraction": {
                                    "spiga": 0.01,
                                    "hrnet": 0.02,
                                },
                                "landmark_consensus_distance": {"spiga": 0.04, "hrnet": 0.08},
                                "roi_center_consensus_distance": {
                                    "spiga": 0.03,
                                    "hrnet": 0.05,
                                },
                                "model_predictions_available": {
                                    "hrnet": True,
                                    "spiga": True,
                                    "orformer": True,
                                },
                                "promoted_candidate_id": "sha256:abc",
                                "setup_mode": "strict",
                                "setup_path": "/tmp/best_setup.json",
                                "strict": True,
                                "weight_source": "runtime_resolver",
                                "weights_path": "best_weights.json",
                            }
                        },
                    )
                ]
            ).to_dict(),
            "IMG_000124.png": AlignmentsEntry(
                faces=[
                    FileAlignments(
                        x=15,
                        y=25,
                        w=90,
                        h=110,
                        landmarks_xy=_points(5.0),
                        metadata={
                            "landmark_ensemble": {
                                "selected_candidate": "static_weighted_hard_drop",
                                "setup_mode": "off",
                                "weight_source": "adapter_config",
                            }
                        },
                    )
                ]
            ).to_dict(),
        },
    }
    serializer.save(str(path), payload)


def _write_review_labels(path: Path) -> None:
    path.write_text(
        "sample_id,image_path,face_index,review_status,issue_type,notes\n"
        f"IMG_000123_face0,{path.parent / 'images' / 'IMG_000123.png'},0,accepted,,\n"
        f"IMG_000124_face0,{path.parent / 'images' / 'IMG_000124.png'},0,rejected,"
        'bad_profile_alignment,"jaw/eyes off"\n',
        encoding="utf-8",
    )


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    (images / "IMG_000123.png").write_bytes(b"")
    (images / "IMG_000124.png").write_bytes(b"")
    alignments = tmp_path / "alignments.fsa"
    _write_alignments(alignments)
    review_labels = tmp_path / "review_labels.csv"
    _write_review_labels(review_labels)
    output = tmp_path / "production_validated"
    return images, alignments, review_labels, output


def test_build_faceswap_validated_manifest_exports_accepted_samples(tmp_path: Path) -> None:
    images, alignments, review_labels, output = _build_fixture(tmp_path)

    rc = main(
        [
            "--images",
            str(images),
            "--alignments",
            str(alignments),
            "--output-dir",
            str(output),
            "--review-labels",
            str(review_labels),
            "--dataset-name",
            "production_validated",
            "--require-landmark-ensemble-metadata",
            "--only-reviewed",
            "accepted",
        ]
    )

    assert rc == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "production_validated"
    assert manifest["metadata"]["label_quality"] == "human_validated"
    assert len(manifest["samples"]) == 1
    sample = manifest["samples"][0]
    assert sample["sample_id"] == "IMG_000123_face0"
    assert sample["condition"] == "profile_left"
    assert sample["conditions"] == ["profile_left"]
    assert sample["metadata"]["runtime_bucket"] == "profile_left"
    assert sample["metadata"]["review_status"] == "accepted"
    assert sample["metadata"]["landmark_ensemble_selected_candidate"] == "spiga"
    assert sample["metadata"]["landmark_ensemble"]["runtime_bucket"] == "profile_left"
    assert sample["metadata"]["landmark_ensemble"]["bucket"] == "profile_left"
    assert _LEGACY_LANDMARK_POSE_BUCKET_KEY not in sample["metadata"]["landmark_ensemble"]
    assert sample["metadata"]["landmark_ensemble"]["candidate_priority"] == [
        "spiga",
        "orformer",
        "hrnet",
    ]
    assert sample["metadata"]["landmark_ensemble"]["vetoed"] == ["hrnet"]
    assert sample["metadata"]["landmark_ensemble"]["veto_reasons"] == {"hrnet": ["roll_veto"]}
    assert sample["metadata"]["landmark_ensemble"]["cloud_area_ratio"]["spiga"] == 0.72
    assert sample["metadata"]["landmark_ensemble"]["hull_area_ratio"]["spiga"] == 0.34
    assert (
        sample["metadata"]["landmark_ensemble"]["points_outside_expanded_bbox_fraction"]["spiga"]
        == 0.01
    )
    assert sample["metadata"]["landmark_ensemble"]["landmark_consensus_distance"]["spiga"] == 0.04
    assert (
        sample["metadata"]["landmark_ensemble"]["roi_center_consensus_distance"]["spiga"] == 0.03
    )
    assert sample["metadata"]["landmark_ensemble"]["model_predictions_available"]["spiga"]
    assert sample["metadata"]["landmark_ensemble"]["setup_mode"] == "strict"
    assert sample["metadata"]["landmark_ensemble"]["setup_path"] == "/tmp/best_setup.json"
    assert sample["metadata"]["landmark_ensemble"]["strict"] is True
    assert sample["metadata"]["landmark_ensemble"]["weights_path"] == "best_weights.json"
    assert sample["face_bbox"] == [10.0, 20.0, 110.0, 140.0]
    landmarks = np.load(str(output / sample["landmarks"]))
    np.testing.assert_allclose(landmarks, _points())

    resolver_lines = (output / "resolver_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(resolver_lines) == 1
    resolver = json.loads(resolver_lines[0])
    assert resolver["sample_id"] == "IMG_000123_face0"
    assert resolver["landmark_ensemble"]["runtime_bucket"] == "profile_left"
    assert resolver["landmark_ensemble"]["bucket"] == "profile_left"

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["faces_total"] == 2
    assert audit["samples_written"] == 1
    assert audit["review_status_counts"] == {"accepted": 1, "rejected": 1}
    assert audit["skipped"] == {"review_status:rejected": 1}


def test_build_faceswap_validated_manifest_allows_missing_resolver_metadata(
    tmp_path: Path,
) -> None:
    images, alignments, review_labels, output = _build_fixture(tmp_path)

    assert (
        main(
            [
                "--images",
                str(images),
                "--alignments",
                str(alignments),
                "--output-dir",
                str(output),
                "--review-labels",
                str(review_labels),
                "--only-reviewed",
                "all",
            ]
        )
        == 0
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [sample["sample_id"] for sample in manifest["samples"]] == [
        "IMG_000123_face0",
        "IMG_000124_face0",
    ]
    assert manifest["samples"][1]["condition"] == "unknown"
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["missing_landmark_ensemble_metadata"] == []
    assert audit["skipped"] == {}


def test_build_faceswap_validated_manifest_applies_runtime_bucket_overrides(
    tmp_path: Path,
) -> None:
    images, alignments, review_labels, output = _build_fixture(tmp_path)
    overrides = tmp_path / "runtime_bucket_overrides.csv"
    overrides.write_text(
        "sample_id,runtime_bucket,reason\n"
        "IMG_000123_face0,intermediate,visual review: frontal/intermediate\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--images",
                str(images),
                "--alignments",
                str(alignments),
                "--output-dir",
                str(output),
                "--review-labels",
                str(review_labels),
                "--runtime-bucket-overrides",
                str(overrides),
                "--require-landmark-ensemble-metadata",
                "--only-reviewed",
                "accepted",
            ]
        )
        == 0
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    sample = manifest["samples"][0]
    assert sample["condition"] == "intermediate"
    assert sample["conditions"] == ["intermediate"]
    assert sample["metadata"]["runtime_bucket"] == "intermediate"
    assert sample["metadata"]["landmark_ensemble"]["runtime_bucket"] == "intermediate"
    assert sample["metadata"]["landmark_ensemble"]["bucket"] == "intermediate"
    assert sample["metadata"]["landmark_ensemble"]["runtime_bucket_original"] == "profile_left"
    assert (
        sample["metadata"]["landmark_ensemble"]["runtime_bucket_override_source"]
        == "runtime_bucket_overrides"
    )

    resolver = json.loads(
        (output / "resolver_metadata.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert resolver["condition"] == "intermediate"
    assert resolver["landmark_ensemble"]["runtime_bucket"] == "intermediate"

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["condition_counts"] == {"intermediate": 1}
    assert audit["runtime_bucket_overrides_applied"] == [
        {
            "sample_id": "IMG_000123_face0",
            "frame": "IMG_000123.png",
            "runtime_bucket": "intermediate",
            "reason": "visual review: frontal/intermediate",
        }
    ]


def test_build_faceswap_validated_manifest_can_require_resolver_metadata(tmp_path: Path) -> None:
    images, alignments, review_labels, output = _build_fixture(tmp_path)

    assert (
        main(
            [
                "--images",
                str(images),
                "--alignments",
                str(alignments),
                "--output-dir",
                str(output),
                "--review-labels",
                str(review_labels),
                "--only-reviewed",
                "all",
                "--require-landmark-ensemble-metadata",
            ]
        )
        == 0
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [sample["sample_id"] for sample in manifest["samples"]] == ["IMG_000123_face0"]
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["missing_landmark_ensemble_metadata"] == ["IMG_000124_face0"]
    assert audit["skipped"]["missing_landmark_ensemble_metadata"] == 1


def test_build_faceswap_validated_manifest_can_extract_missing_resolver_metadata(
    tmp_path: Path,
) -> None:
    images, alignments, review_labels, output = _build_fixture(tmp_path)
    extracted: list[str] = []

    def _extractor(
        _image_path: Path,
        face: FileAlignments,
        _frame_name: str,
        face_index: int,
    ) -> dict[str, object]:
        extracted.append(f"face{face_index}")
        assert np.asarray(face.landmarks_xy).shape == (68, 2)
        return {
            "selected_candidate": "spiga",
            "runtime_bucket": "profile_right",
            "bucket": "profile_right",
            "candidate_priority": ["spiga", "static_weighted"],
            "vetoed": [],
            "veto_reasons": {},
            "roll_estimate": -3.5,
            "risk_route": "high_risk",
            "dominant_candidate_yaw": -54.0,
            "max_disagreement_px": 14.2,
            "max_disagreement_bbox_fraction": 0.08,
            "runtime_bucket_side": "right",
            "runtime_bucket_side_source": "landmark_pose_yaw",
            "runtime_bucket_severity": "profile",
            "runtime_bucket_severity_source": "landmark_pose_yaw",
            "cloud_area_ratio": {"spiga": 0.72},
            "hull_area_ratio": {"spiga": 0.34},
            "points_outside_expanded_bbox_fraction": {"spiga": 0.01},
            "landmark_consensus_distance": {"spiga": 0.04},
            "roi_center_consensus_distance": {"spiga": 0.03},
            "model_predictions_available": {"spiga": True},
            "promoted_candidate_id": "sha256:def",
            "setup_mode": "strict",
            "setup_path": "/tmp/best_setup.json",
            "strict": True,
            "weight_source": "runtime_resolver",
            "weights_path": "best_weights.json",
            "yaw_estimate": 52.0,
        }

    audit = build_manifest(
        images,
        alignments,
        output,
        review_labels_path=review_labels,
        only_reviewed="all",
        require_landmark_ensemble_metadata=True,
        extract_landmark_ensemble_metadata=True,
        metadata_extractor=_extractor,
    )

    assert extracted == ["face0"]
    assert audit["samples_written"] == 2
    assert audit["extracted_landmark_ensemble_metadata"] == ["IMG_000124_face0"]
    assert audit["missing_landmark_ensemble_metadata"] == []

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    second = manifest["samples"][1]
    assert second["condition"] == "profile_right"
    assert second["conditions"] == ["profile_right"]
    assert second["metadata"]["runtime_bucket"] == "profile_right"
    assert second["metadata"]["landmark_ensemble_selected_candidate"] == "spiga"
    assert second["metadata"]["landmark_ensemble"]["runtime_bucket"] == "profile_right"
    assert second["metadata"]["landmark_ensemble"]["bucket"] == "profile_right"
    assert _LEGACY_LANDMARK_POSE_BUCKET_KEY not in second["metadata"]["landmark_ensemble"]
    assert second["metadata"]["landmark_ensemble"]["candidate_priority"] == [
        "spiga",
        "static_weighted",
    ]
    assert second["metadata"]["landmark_ensemble"]["model_predictions_available"] == {
        "spiga": True
    }
    assert second["metadata"]["landmark_ensemble"]["setup_mode"] == "strict"
    assert second["metadata"]["landmark_ensemble"]["strict"] is True
    landmarks = np.load(str(output / second["landmarks"]))
    np.testing.assert_allclose(landmarks, _points(5.0))


def test_build_faceswap_validated_manifest_requires_setup_for_cli_extraction(
    tmp_path: Path,
) -> None:
    images, alignments, review_labels, output = _build_fixture(tmp_path)

    try:
        main(
            [
                "--images",
                str(images),
                "--alignments",
                str(alignments),
                "--output-dir",
                str(output),
                "--review-labels",
                str(review_labels),
                "--only-reviewed",
                "all",
                "--extract-landmark-ensemble-metadata",
            ]
        )
    except SystemExit as err:
        assert "--landmark-ensemble-setup-path" in str(err)
    else:
        raise AssertionError("expected missing promoted setup path to fail")
