#!/usr/bin/env python3
"""End-to-end test for the run_source_target_compatibility_scoring.py CLI."""

from __future__ import annotations

import json

import numpy as np

import tools.faceqa_coverage.run_source_target_compatibility_scoring as runner
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer


def _alignments_with_one_face(path) -> None:
    """Write a minimal alignments file with a single 80x90 face."""
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    face = FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=landmarks)
    get_serializer("compressed").save(
        str(path),
        {
            "__meta__": {"version": 2.4},
            "__data__": {"frame_000001.png": AlignmentsEntry(faces=[face]).to_dict()},
        },
    )


def test_runner_writes_json_and_markdown(tmp_path, capsys) -> None:
    source = tmp_path / "source.fsa"
    target = tmp_path / "target.fsa"
    _alignments_with_one_face(source)
    _alignments_with_one_face(target)
    output_dir = tmp_path / "reports"

    exit_code = runner.main(
        [
            "--source",
            str(source),
            "--target",
            str(target),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    output_json = output_dir / "source_target_compatibility.json"
    output_md = output_dir / "source_target_compatibility.md"
    assert output_json.is_file()
    assert output_md.is_file()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "source_target_compatibility_score" in payload
    assert "pose_compatibility_score" in payload
    assert "coverage_gaps" in payload

    captured = capsys.readouterr().out
    assert "# FaceQA Source-Target Compatibility" in captured
    assert "Compatibility Summary" in captured


def test_runner_quiet_suppresses_stdout(tmp_path, capsys) -> None:
    source = tmp_path / "source.fsa"
    target = tmp_path / "target.fsa"
    _alignments_with_one_face(source)
    _alignments_with_one_face(target)

    exit_code = runner.main(
        [
            "--source",
            str(source),
            "--target",
            str(target),
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Compatibility Summary" not in captured


def test_runner_errors_on_missing_alignments(tmp_path) -> None:
    target = tmp_path / "target.fsa"
    _alignments_with_one_face(target)

    try:
        runner.main(
            [
                "--source",
                str(tmp_path / "does_not_exist.fsa"),
                "--target",
                str(target),
                "--output-dir",
                str(tmp_path),
            ]
        )
    except SystemExit as err:
        assert "alignments file not found" in str(err)
    else:
        raise AssertionError("Expected SystemExit for missing source alignments")
