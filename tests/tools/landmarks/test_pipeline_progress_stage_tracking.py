#!/usr/bin/env python3
"""Regression tests for pipeline progress coverage on non-child-command stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tools.landmarks.run_landmark_resolver_pipeline as pipeline


def _progress_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_root=tmp_path / "static_run",
        production_root=tmp_path / "production_run",
        output_root=tmp_path / "resolver_run",
        start_at="build_splits",
        stop_after="build_production_manifest",
        progress=False,
        no_progress=True,
    )


def test_progress_log_tracks_inline_and_reused_manifest_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """build_splits / fit_static_weights / build_production_manifest stay tracked.

    These stages are easy to regress because build_splits and fit_static_weights run
    inline in run_landmark_resolver_pipeline.py, while build_production_manifest may
    reuse already-materialized production artifacts. The outer run_pipeline() progress
    wrapper must still emit start+finish events for all of them.
    """

    tracked_stages = (
        "build_splits",
        "fit_static_weights",
        "build_production_manifest",
    )
    seen: list[str] = []

    def _fake_execute_stage(
        stage: str,
        _args: argparse.Namespace,
        _paths: pipeline.PipelinePaths,
    ) -> pipeline.StageResult:
        assert stage in tracked_stages
        seen.append(stage)
        return pipeline.StageResult(stage, "ok", notes=[f"{stage}=tracked"])

    def _fake_summary_payload(
        _args: argparse.Namespace,
        _paths: pipeline.PipelinePaths,
        results: list[pipeline.StageResult],
    ) -> dict[str, object]:
        return {
            "status": "pass",
            "stages": [{"name": result.name, "status": result.status} for result in results],
        }

    monkeypatch.setattr(pipeline, "_execute_stage", _fake_execute_stage)
    monkeypatch.setattr(pipeline, "_summary_payload", _fake_summary_payload)

    args = _progress_args(tmp_path)
    summary = pipeline.run_pipeline(args)

    assert summary["status"] == "pass"
    assert tuple(seen) == tracked_stages

    progress_log = pipeline.PipelinePaths(
        args.run_root,
        args.production_root,
        args.output_root,
    ).progress_log
    events = [json.loads(line) for line in progress_log.read_text(encoding="utf-8").splitlines()]

    assert [(event["event"], event["stage"], event["status"]) for event in events] == [
        ("start", "build_splits", "running"),
        ("finish", "build_splits", "ok"),
        ("start", "fit_static_weights", "running"),
        ("finish", "fit_static_weights", "ok"),
        ("start", "build_production_manifest", "running"),
        ("finish", "build_production_manifest", "ok"),
    ]
    assert {event["total"] for event in events} == {len(tracked_stages)}
    assert [event["index"] for event in events] == [1, 1, 2, 2, 3, 3]
