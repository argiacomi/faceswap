#!/usr/bin/env python3
"""Tests for FaceQA coverage reports."""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np

from lib.align.faceset_qa import FaceQAFile, FaceQARecord, load, sidecar_path
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.coverage import SpigaPoseBackfiller, compute_coverage, records_from_alignments
from lib.faceqa.readiness import generate_readiness_report
from lib.serializer import get_serializer
from tools.faceqa_coverage.faceqa_coverage import Faceqa_Coverage


def _face(width: int = 80, height: int = 90) -> FileAlignments:
    """Return a minimal alignments face."""
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    return FileAlignments(x=0, y=0, w=width, h=height, landmarks_xy=landmarks)


def test_compute_coverage_counts_unknown_fields() -> None:
    """Missing QA fields should bucket as unknown rather than crashing."""
    report = compute_coverage([FaceQARecord(frame="frame.png", face_index=0)])

    assert report.total_faces == 1
    assert report.usable_faces == 1
    assert report.bucket_counts["blur"]["unknown"] == 1
    assert report.bucket_counts["pose"]["unknown"] == 1
    assert report.bucket_counts["resolution"]["unknown"] == 1


def test_compute_coverage_uses_sidecar_risk_metadata() -> None:
    """Duplicate and identity sidecar fields should affect usable counts."""
    records = [
        FaceQARecord(
            frame="frame.png",
            face_index=0,
            blur_score=0.5,
            duplicate_keep_recommendation="prune_candidate",
            identity_quality_flag="outlier",
            resolution=[40, 40],
            average_distance=0.5,
        )
    ]
    report = compute_coverage(records, exclude_duplicates=True, exclude_outliers=True)

    assert report.usable_faces == 0
    assert report.bucket_counts["blur"]["unusable"] == 1
    assert report.bucket_counts["resolution"]["tiny"] == 1
    assert report.bucket_counts["misalignment"]["extreme"] == 1
    assert report.duplicate_ratio == 1.0
    assert report.identity_outlier_ratio == 1.0


def test_sidecar_load_ignores_unknown_fields(tmp_path) -> None:
    """Future additive sidecar fields should not break this reader."""
    path = tmp_path / "alignments_faceset_qa.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "future_root": True,
                "faces": [
                    {
                        "frame": "frame.png",
                        "face_index": 0,
                        "yaw": 10.0,
                        "future_face_field": "ignored",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    qa_file = load(str(path))

    assert isinstance(qa_file, FaceQAFile)
    assert qa_file.faces[0].yaw == 10.0


def test_faceqa_coverage_tool_writes_reports_from_alignments(tmp_path, monkeypatch) -> None:
    """The tool should write JSON and Markdown reports from alignments alone."""
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                "frame_000001.png": AlignmentsEntry(faces=[_face()]).to_dict(),
            },
        },
    )
    output_json = tmp_path / "coverage.json"
    output_md = tmp_path / "coverage.md"
    args = Namespace(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        sidecar=None,
        output_json=str(output_json),
        output_markdown=str(output_md),
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
    )
    monkeypatch.setattr(
        Faceqa_Coverage,
        "_pose_backfiller",
        lambda _: lambda __, ___: None,
    )

    Faceqa_Coverage(args).process()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["alignments"] == str(alignments)
    assert payload["total_faces"] == 1
    assert "pose" in payload["coverage"]
    assert "pose_sources" in payload["coverage"]
    assert "pose_confidence" in payload["coverage"]
    assert "blur" in payload["coverage"]
    markdown = output_md.read_text(encoding="utf-8")
    assert markdown.startswith("# FaceQA Coverage Report")
    assert "## Pose Sources" in markdown
    assert "## Pose Confidence" in markdown


def test_records_from_alignments_prefers_spiga_pose_metadata(tmp_path) -> None:
    """FaceQA should prefer native SPIGA pose over landmark-derived pose."""
    face = _face()
    face.metadata = {
        "spiga": {
            "pose": {
                "yaw": 42.0,
                "pitch": -8.0,
                "roll": 3.0,
                "source": "spiga",
                "model": "spiga",
                "delta": {"yaw": 2.0, "pitch": -1.0, "roll": 0.5},
            }
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

    records = records_from_alignments(alignments)

    assert records[0].yaw == 42.0
    assert records[0].pitch == -8.0
    assert records[0].roll == 3.0
    assert records[0].pose_source == "spiga"
    assert records[0].pose_model == "spiga"
    assert records[0].alignment_yaw is not None
    assert records[0].spiga_yaw is not None
    assert records[0].pose_delta_yaw is not None
    assert records[0].pose_delta_pitch is not None
    assert records[0].pose_delta_roll is not None
    assert records[0].pose_delta_yaw == records[0].spiga_yaw - records[0].alignment_yaw
    assert records[0].pose_max_abs_delta == max(
        abs(records[0].pose_delta_yaw),
        abs(records[0].pose_delta_pitch),
        abs(records[0].pose_delta_roll),
    )
    assert records[0].pose_confidence == "low"


def test_records_from_alignments_backfills_missing_spiga_pose(tmp_path) -> None:
    """Missing SPIGA pose should be backfilled before coverage pose selection."""
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                "frame_000001.png": AlignmentsEntry(faces=[_face()]).to_dict(),
            },
        },
    )

    def backfill(_: FaceQARecord, __: FileAlignments) -> dict[str, object]:
        return {
            "yaw": 15.0,
            "pitch": -2.0,
            "roll": 4.0,
            "source": "spiga_backfill",
            "model": "spiga",
        }

    records = records_from_alignments(alignments, pose_backfiller=backfill)

    assert records[0].yaw == 15.0
    assert records[0].pitch == -2.0
    assert records[0].roll == 4.0
    assert records[0].spiga_yaw == 15.0
    assert records[0].pose_source == "spiga_backfill"
    assert records[0].pose_model == "spiga"


def test_spiga_pose_backfiller_uses_source_frame_alignment_not_thumbnail() -> None:
    """SPIGA backfill should reconstruct aligned faces from source frames."""

    class _Frames:
        loaded: list[str] = []

        def load_image(self, frame: str) -> np.ndarray:
            self.loaded.append(frame)
            image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
            image[:, :, 1] = 255
            return image

    class _Plugin:
        input_size = 64

        def __init__(self) -> None:
            self.feed: np.ndarray | None = None
            self.last_debug_metadata: list[dict[str, object]] = []

        def process(self, batch: np.ndarray) -> np.ndarray:
            self.feed = batch
            return np.zeros((1, 71, 2), dtype=np.float32)

        def post_process(self, _: np.ndarray) -> np.ndarray:
            self.last_debug_metadata = [
                {"pose": {"yaw": 11.0, "pitch": -2.0, "roll": 3.0, "model": "spiga"}}
            ]
            return np.zeros((1, 68, 2), dtype=np.float32)

    frames = _Frames()
    plugin = _Plugin()
    backfiller = SpigaPoseBackfiller(frames)
    backfiller._plugin = plugin  # pylint:disable=protected-access
    face = _face()
    face.thumb = None

    pose = backfiller(FaceQARecord(frame="frame_000001.png", face_index=0), face)

    assert frames.loaded == ["frame_000001.png"]
    assert plugin.feed is not None
    assert plugin.feed.shape == (1, 64, 64, 3)
    assert pose == {
        "yaw": 11.0,
        "pitch": -2.0,
        "roll": 3.0,
        "model": "spiga",
        "source": "spiga_backfill",
    }


def test_records_from_alignments_uses_alignment_when_spiga_pose_invalid(tmp_path) -> None:
    """Invalid SPIGA pose candidates should not become selected coverage pose."""
    face = _face()
    face.metadata = {
        "spiga": {
            "pose": {
                "yaw": 500.0,
                "pitch": 0.0,
                "roll": 0.0,
                "source": "spiga",
                "model": "spiga",
            }
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

    records = records_from_alignments(alignments)

    assert records[0].spiga_yaw == 500.0
    assert records[0].pose_source == "alignment"
    assert records[0].pose_model == "aligned_face"
    assert records[0].pose_confidence == "fallback"
    assert records[0].yaw == records[0].alignment_yaw


def test_compute_coverage_reports_pose_source_and_confidence() -> None:
    """Coverage should summarize selected pose provenance and confidence."""
    report = compute_coverage(
        [
            FaceQARecord(
                frame="frame.png",
                face_index=0,
                yaw=15.0,
                pitch=-2.0,
                roll=4.0,
                pose_source="spiga_backfill",
                pose_confidence="high",
            )
        ]
    )

    assert report.bucket_counts["pose_sources"]["spiga_backfill"] == 1
    assert report.bucket_counts["pose_confidence"]["high"] == 1
    assert report.coverage_dict()["pose_sources"]["percentages"]["spiga_backfill"] == 100.0


def test_readiness_report_warns_for_pose_fallback_and_disagreement() -> None:
    """Readiness warnings should flag fallback pose and low-confidence SPIGA pose."""
    records = [
        FaceQARecord(pose_source="alignment", pose_confidence="fallback", yaw=0.0),
        FaceQARecord(pose_source="alignment", pose_confidence="fallback", yaw=0.0),
        FaceQARecord(pose_source="alignment", pose_confidence="fallback", yaw=0.0),
        FaceQARecord(pose_source="spiga", pose_confidence="low", yaw=30.0),
        FaceQARecord(pose_source="spiga", pose_confidence="low", yaw=-30.0),
    ]

    report = generate_readiness_report(compute_coverage(records))

    assert any("Pose fallback risk" in warning for warning in report.warnings)
    assert any("SPIGA/alignment pose disagreement" in warning for warning in report.warnings)


def test_faceqa_coverage_tool_persists_backfilled_spiga_pose(tmp_path, monkeypatch) -> None:
    """The coverage tool should persist backfilled SPIGA pose to the sidecar."""
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                "frame_000001.png": AlignmentsEntry(faces=[_face()]).to_dict(),
            },
        },
    )
    output_json = tmp_path / "coverage.json"
    output_md = tmp_path / "coverage.md"
    args = Namespace(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        sidecar=None,
        output_json=str(output_json),
        output_markdown=str(output_md),
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
    )

    def backfill(_: FaceQARecord, __: FileAlignments) -> dict[str, object]:
        return {
            "yaw": 21.0,
            "pitch": -3.0,
            "roll": 7.0,
            "source": "spiga_backfill",
            "model": "spiga",
        }

    monkeypatch.setattr(
        Faceqa_Coverage,
        "_pose_backfiller",
        lambda _: backfill,
    )

    Faceqa_Coverage(args).process()

    qa_file = load(sidecar_path(str(alignments)))
    assert qa_file is not None
    assert qa_file.faces[0].spiga_pose_source == "spiga_backfill"
    assert qa_file.faces[0].pose_source == "spiga_backfill"
    assert qa_file.faces[0].yaw == 21.0


def test_faceqa_coverage_tool_does_not_recompute_existing_spiga_pose(
    tmp_path,
    monkeypatch,
) -> None:
    """Existing valid SPIGA pose metadata should be reused."""
    face = _face()
    face.metadata = {
        "spiga": {
            "pose": {
                "yaw": 12.0,
                "pitch": 1.0,
                "roll": -2.0,
                "source": "spiga",
                "model": "spiga",
            }
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
    output_json = tmp_path / "coverage.json"
    output_md = tmp_path / "coverage.md"
    args = Namespace(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        sidecar=None,
        output_json=str(output_json),
        output_markdown=str(output_md),
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
    )
    calls = 0

    def backfill(_: FaceQARecord, __: FileAlignments) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    monkeypatch.setattr(
        Faceqa_Coverage,
        "_pose_backfiller",
        lambda _: backfill,
    )

    Faceqa_Coverage(args).process()

    assert calls == 0
    assert not (tmp_path / "alignments_faceset_qa.json").exists()


def test_compute_coverage_signed_pose_and_pitch_buckets() -> None:
    """Signed yaw and pitch coverage should partition records by side and tilt."""
    records = [
        FaceQARecord(frame="left_extreme.png", face_index=0, yaw=-75.0, pitch=-40.0),
        FaceQARecord(frame="left_profile.png", face_index=0, yaw=-45.0, pitch=-20.0),
        FaceQARecord(frame="left_slight.png", face_index=0, yaw=-20.0, pitch=-5.0),
        FaceQARecord(frame="frontal.png", face_index=0, yaw=0.0, pitch=0.0),
        FaceQARecord(frame="right_slight.png", face_index=0, yaw=20.0, pitch=20.0),
        FaceQARecord(frame="right_profile.png", face_index=0, yaw=45.0, pitch=35.0),
        FaceQARecord(frame="right_extreme.png", face_index=0, yaw=75.0, pitch=10.0),
    ]

    report = compute_coverage(records)

    pose = report.bucket_counts["pose"]
    assert pose["left_extreme"] == 1
    assert pose["left_profile"] == 1
    assert pose["left_slight"] == 1
    assert pose["frontal"] == 1
    assert pose["right_slight"] == 1
    assert pose["right_profile"] == 1
    assert pose["right_extreme"] == 1
    pitch_counts = report.bucket_counts["pitch"]
    assert pitch_counts["down_extreme"] == 1
    assert pitch_counts["down"] == 1
    assert pitch_counts["neutral"] == 3
    assert pitch_counts["up"] == 1
    assert pitch_counts["up_extreme"] == 1
    assert "signed_yaw" not in report.bucket_counts
    assert "slight" not in pose
    assert "profile" not in pose
    assert "extreme" not in pose


def test_compute_coverage_joint_pose_coverage_counts_and_entropy() -> None:
    """Joint coverage should track occupied/empty cells and pose entropy."""
    records = [
        FaceQARecord(frame="a.png", face_index=0, yaw=0.0, pitch=0.0),
        FaceQARecord(frame="b.png", face_index=0, yaw=0.0, pitch=0.0),
        FaceQARecord(frame="c.png", face_index=0, yaw=45.0, pitch=20.0),
        FaceQARecord(frame="d.png", face_index=0, yaw=-45.0, pitch=-20.0),
        FaceQARecord(frame="e.png", face_index=0, yaw=None),
    ]

    report = compute_coverage(records)

    joint = report.joint_pose_coverage
    assert joint["total_cells"] == 7 * 5
    assert joint["occupied_pose_cells"] == 3
    assert joint["empty_pose_cells"] == 32
    assert joint["classified_faces"] == 4
    assert joint["unknown_faces"] == 1
    assert joint["counts"]["frontal+neutral"] == 2
    assert joint["counts"]["right_profile+up"] == 1
    assert joint["counts"]["left_profile+down"] == 1
    assert joint["pose_bin_coverage_pct"] == round(3 / 35 * 100, 2)
    expected_entropy = -(0.5 * np.log2(0.5) + 0.25 * np.log2(0.25) * 2)
    assert abs(joint["pose_entropy"] - round(expected_entropy, 4)) < 1e-3
    assert "left_extreme+up_extreme" in joint["missing_cells"]
    assert "frontal+neutral" not in joint["missing_cells"]


def test_compute_coverage_joint_pose_coverage_empty_records() -> None:
    """Joint coverage handles records with no pose data without dividing by zero."""
    records = [FaceQARecord(frame="f.png", face_index=0)]

    report = compute_coverage(records)

    joint = report.joint_pose_coverage
    assert joint["classified_faces"] == 0
    assert joint["unknown_faces"] == 1
    assert joint["occupied_pose_cells"] == 0
    assert joint["empty_pose_cells"] == joint["total_cells"]
    assert joint["pose_entropy"] == 0.0
    assert joint["pose_bin_coverage_pct"] == 0.0


def test_readiness_report_exposes_joint_pose_coverage_in_json() -> None:
    """Joint pose coverage should round-trip through the JSON report."""
    records = [
        FaceQARecord(frame="a.png", face_index=0, yaw=0.0, pitch=0.0),
        FaceQARecord(frame="b.png", face_index=0, yaw=-45.0, pitch=20.0),
    ]

    report = generate_readiness_report(compute_coverage(records))
    payload = json.loads(report.to_json())

    assert "joint_pose_coverage" in payload
    joint = payload["joint_pose_coverage"]
    assert joint["counts"]["frontal+neutral"] == 1
    assert joint["counts"]["left_profile+up"] == 1
    assert joint["occupied_pose_cells"] == 2
    assert joint["empty_pose_cells"] == joint["total_cells"] - 2
    assert "missing_cells" in joint


def test_readiness_report_recommends_missing_pose_regions() -> None:
    """Recommendations should mention specific missing yaw/pitch cells."""
    records = [
        FaceQARecord(frame="a.png", face_index=0, yaw=0.0, pitch=0.0),
        FaceQARecord(frame="b.png", face_index=0, yaw=0.0, pitch=0.0),
        FaceQARecord(frame="c.png", face_index=0, yaw=0.0, pitch=0.0),
    ]

    report = generate_readiness_report(compute_coverage(records))

    assert any(
        "missing yaw/pitch regions" in recommendation for recommendation in report.recommendations
    )
    assert any("Sparse pose coverage" in warning for warning in report.warnings)
    assert any("Low pose entropy" in warning for warning in report.warnings)


def test_readiness_report_recommends_balanced_yaw_and_pitch() -> None:
    """Under-represented yaw/pitch buckets should drive recommendations."""
    records = [
        FaceQARecord(frame=f"a{i}.png", face_index=0, yaw=0.0, pitch=0.0) for i in range(20)
    ]
    records.append(FaceQARecord(frame="r.png", face_index=0, yaw=70.0, pitch=40.0))

    report = generate_readiness_report(compute_coverage(records))

    assert any(
        "Balance yaw coverage" in recommendation for recommendation in report.recommendations
    )
    assert any("pitch angles" in recommendation for recommendation in report.recommendations)


def test_faceqa_coverage_tool_writes_joint_pose_metrics(tmp_path, monkeypatch) -> None:
    """The coverage tool should write joint pose metrics into the JSON report."""
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                "frame_000001.png": AlignmentsEntry(faces=[_face()]).to_dict(),
            },
        },
    )
    output_json = tmp_path / "coverage.json"
    output_md = tmp_path / "coverage.md"
    args = Namespace(
        alignments=str(alignments),
        frames_dir=str(tmp_path),
        sidecar=None,
        output_json=str(output_json),
        output_markdown=str(output_md),
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
    )
    monkeypatch.setattr(
        Faceqa_Coverage,
        "_pose_backfiller",
        lambda _: lambda __, ___: None,
    )

    Faceqa_Coverage(args).process()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "signed_yaw" not in payload["coverage"]
    pose_buckets = payload["coverage"]["pose"]["counts"]
    assert set(pose_buckets).issuperset(
        {
            "left_extreme",
            "left_profile",
            "left_slight",
            "frontal",
            "right_slight",
            "right_profile",
            "right_extreme",
            "unknown",
        }
    )
    assert "pitch" in payload["coverage"]
    joint = payload["joint_pose_coverage"]
    assert joint["total_cells"] == 35
    assert "pose_entropy" in joint
    assert "missing_cells" in joint
    markdown = output_md.read_text(encoding="utf-8")
    assert "## Pose" in markdown
    assert "## Pitch" in markdown
    assert "## Joint Pose Coverage" in markdown


def test_compute_coverage_expression_bins_and_entropy() -> None:
    """Expression coverage should report counts, entropy, and empty bins."""
    records = [
        FaceQARecord(frame=f"n{i}.png", face_index=0, expression_bucket="neutral")
        for i in range(4)
    ] + [
        FaceQARecord(frame="s.png", face_index=0, expression_bucket="smile"),
        FaceQARecord(frame="t.png", face_index=0, expression_bucket="talking_open"),
        FaceQARecord(frame="u.png", face_index=0, expression_bucket=None),
    ]

    report = compute_coverage(records)
    expression = report.expression_coverage

    assert expression["total_bins"] == 6
    assert expression["counts"]["neutral"] == 4
    assert expression["counts"]["smile"] == 1
    assert expression["counts"]["talking_open"] == 1
    assert expression["occupied_expression_bins"] == 3
    assert expression["empty_expression_bins"] == 3
    assert expression["classified_faces"] == 6
    assert expression["unknown_faces"] == 1
    assert expression["expression_bin_coverage_pct"] == round(3 / 6 * 100, 2)
    expected_entropy = -((4 / 6) * np.log2(4 / 6) + 2 * (1 / 6) * np.log2(1 / 6))
    assert abs(expression["expression_entropy"] - round(expected_entropy, 4)) < 1e-3
    assert "eyes_closed" in expression["missing_bins"]
    assert "neutral" not in expression["missing_bins"]


def test_compute_coverage_expression_handles_empty_distribution() -> None:
    """Expression coverage should not divide by zero when nothing classifies."""
    report = compute_coverage([FaceQARecord(frame="f.png", face_index=0)])
    expression = report.expression_coverage

    assert expression["classified_faces"] == 0
    assert expression["unknown_faces"] == 1
    assert expression["occupied_expression_bins"] == 0
    assert expression["expression_entropy"] == 0.0
    assert expression["expression_bin_coverage_pct"] == 0.0


def test_records_from_alignments_derives_expression_from_landmarks(tmp_path) -> None:
    """records_from_alignments should derive expression features from landmarks."""
    from tests.lib.faceqa.expression_test import _neutral_landmarks

    face = FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=_neutral_landmarks())
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {"frame_000001.png": AlignmentsEntry(faces=[face]).to_dict()},
        },
    )

    records = records_from_alignments(alignments)

    assert records[0].expression_bucket == "neutral"
    assert records[0].mouth_openness is not None
    assert records[0].mouth_width_ratio is not None
    assert records[0].smile_proxy is not None
    assert records[0].eye_closure is not None
    assert records[0].brow_raise_proxy is not None
    assert records[0].expression_asymmetry is not None


def test_records_from_alignments_overrides_sidecar_expression(tmp_path) -> None:
    """Landmark-derived expression should supersede sidecar expression labels."""
    from lib.align.faceset_qa import FaceQAFile
    from tests.lib.faceqa.expression_test import _neutral_landmarks

    face = FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=_neutral_landmarks())
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {"frame_000001.png": AlignmentsEntry(faces=[face]).to_dict()},
        },
    )
    qa_file = FaceQAFile(
        faces=[FaceQARecord(frame="frame_000001.png", face_index=0, expression_bucket="smile")]
    )

    records = records_from_alignments(alignments, qa_file=qa_file)

    assert records[0].expression_bucket == "neutral"


def test_readiness_report_warns_on_sparse_expression_coverage() -> None:
    """Readiness warnings should flag sparse expression coverage and low entropy."""
    records = [
        FaceQARecord(frame=f"n{i}.png", face_index=0, expression_bucket="neutral")
        for i in range(10)
    ]

    report = generate_readiness_report(compute_coverage(records))

    assert any("Sparse expression coverage" in warning for warning in report.warnings)
    assert any("Low expression entropy" in warning for warning in report.warnings)


def test_readiness_report_recommends_missing_expression_regions() -> None:
    """Recommendations should call out missing expression buckets by name."""
    records = [
        FaceQARecord(frame=f"n{i}.png", face_index=0, expression_bucket="neutral")
        for i in range(10)
    ]

    report = generate_readiness_report(compute_coverage(records))

    assert any(
        "Collect missing expression frames" in recommendation
        and "smiling frames" in recommendation
        and "open-mouth" in recommendation
        for recommendation in report.recommendations
    )


def test_readiness_report_exposes_expression_coverage_in_json() -> None:
    """Expression coverage should round-trip through JSON output."""
    records = [
        FaceQARecord(frame="a.png", face_index=0, expression_bucket="neutral"),
        FaceQARecord(frame="b.png", face_index=0, expression_bucket="smile"),
    ]

    report = generate_readiness_report(compute_coverage(records))
    payload = json.loads(report.to_json())

    assert "expression_coverage" in payload
    assert payload["expression_coverage"]["counts"]["neutral"] == 1
    assert payload["expression_coverage"]["counts"]["smile"] == 1
    assert payload["expression_coverage"]["occupied_expression_bins"] == 2
    assert "missing_bins" in payload["expression_coverage"]
    markdown = report.to_markdown()
    assert "## Expression Coverage" in markdown
    assert "Bin coverage" in markdown


def test_sidecar_path() -> None:
    """Default FaceQA sidecar path should be derived from alignments stem."""
    assert sidecar_path("/tmp/project/alignments.fsa") == "/tmp/project/alignments_faceset_qa.json"
