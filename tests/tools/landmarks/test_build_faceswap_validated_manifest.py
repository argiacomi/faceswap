#!/usr/bin/env python3
"""Tests for exporting reviewed production Faceswap alignment manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer
from tools.landmarks.build_faceswap_validated_manifest import main


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
                                "bucket": "profile_left",
                                "candidate_priority": ["spiga", "orformer", "hrnet"],
                                "vetoed": ["hrnet"],
                                "veto_reasons": {"hrnet": ["roll_veto"]},
                                "roll_estimate": 12.3,
                                "yaw_estimate": -48.0,
                                "cloud_area_ratio": {"spiga": 0.72, "hrnet": 0.91},
                                "landmark_consensus_distance": {"spiga": 0.04, "hrnet": 0.08},
                                "model_predictions_available": {
                                    "hrnet": True,
                                    "spiga": True,
                                    "orformer": True,
                                },
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
    assert sample["metadata"]["review_status"] == "accepted"
    assert sample["metadata"]["landmark_ensemble_selected_candidate"] == "spiga"
    assert sample["face_bbox"] == [10.0, 20.0, 110.0, 140.0]
    landmarks = np.load(str(output / sample["landmarks"]))
    np.testing.assert_allclose(landmarks, _points())

    resolver_lines = (output / "resolver_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(resolver_lines) == 1
    resolver = json.loads(resolver_lines[0])
    assert resolver["sample_id"] == "IMG_000123_face0"
    assert resolver["landmark_ensemble"]["bucket"] == "profile_left"

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["faces_total"] == 2
    assert audit["samples_written"] == 1
    assert audit["review_status_counts"] == {"accepted": 1, "rejected": 1}
    assert audit["skipped"] == {"review_status:rejected": 1}


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
