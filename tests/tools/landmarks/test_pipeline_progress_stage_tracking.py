#!/usr/bin/env python3
"""Regression tests for tqdm pipeline progress coverage on non-child-command stages."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import tools.landmarks.pipeline_progress as progress_mod
import tools.landmarks.run_landmark_resolver_pipeline as pipeline


class FakeTqdm:
    instances: list["FakeTqdm"] = []

    def __init__(self, *, total: int, desc: str, unit: str, leave: bool) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self.n = 0
        self.postfixes: list[str] = []
        self.closed = False
        FakeTqdm.instances.append(self)

    def set_postfix_str(self, value: str, *, refresh: bool = True) -> None:
        assert refresh is True
        self.postfixes.append(value)

    def close(self) -> None:
        self.closed = True


def _progress_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_root=tmp_path / "static_run",
        production_root=tmp_path / "production_run",
        output_root=tmp_path / "resolver_run",
        start_at="build_splits",
        stop_after="build_production_manifest",
        progress=True,
        no_progress=False,
        dry_run=False,
        write_summary_md=False,
    )


def test_tqdm_progress_bar_tracks_inline_and_reused_manifest_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """build_splits / fit_static_weights / build_production_manifest stay on tqdm.

    These stages are easy to regress because build_splits and fit_static_weights run
    inline in run_landmark_resolver_pipeline.py, while build_production_manifest may
    reuse already-materialized production artifacts. The outer run_pipeline() progress
    wrapper must still drive the visible tqdm bar for all of them.
    """

    tracked_stages = (
        "build_splits",
        "fit_static_weights",
        "build_production_manifest",
    )
    seen: list[str] = []
    FakeTqdm.instances.clear()
    progress_mod.PipelineProgress._BAR = None

    fake_tqdm_module = types.ModuleType("tqdm")
    fake_tqdm_module.tqdm = FakeTqdm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm_module)

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

    assert len(FakeTqdm.instances) == 1
    bar = FakeTqdm.instances[0]
    assert bar.total == len(tracked_stages)
    assert bar.desc == "landmark pipeline"
    assert bar.unit == "stage"
    assert bar.closed is True
    assert bar.n == len(tracked_stages)
    assert bar.postfixes == [
        "build_splits running",
        "build_splits ok",
        "fit_static_weights running",
        "fit_static_weights ok",
        "build_production_manifest running",
        "build_production_manifest ok",
    ]

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
