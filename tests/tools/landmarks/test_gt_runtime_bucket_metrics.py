#!/usr/bin/env python3
"""Tests for ``lib/landmarks/search/gt_runtime_bucket_metrics.py`` and the
companion CLI ``tools/landmarks/build_gt_runtime_bucket_metrics.py``
(issue #205).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from lib.landmarks.search.gt_runtime_bucket_metrics import (
    GT_RUNTIME_BUCKET_CSV_COLUMNS,
    SINGLE_MODEL_CANDIDATES,
    aggregate_runtime_bucket_metrics,
    load_candidate_table_csv,
    write_runtime_bucket_csv,
    write_runtime_bucket_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(
    *,
    sample_id: str,
    candidate: str,
    nme: float,
    failure: bool = False,
    runtime_bucket: str = "frontal",
) -> dict[str, Any]:
    """Minimal candidate-table row dict consumed by the aggregator."""
    return {
        "sample_id": sample_id,
        "candidate": candidate,
        "nme": nme,
        "failure": failure,
        "runtime_bucket": runtime_bucket,
    }


def _synthetic_rows() -> list[dict[str, Any]]:
    """Synthetic candidate table covering two runtime buckets and a
    representative mix of single-model + ensemble candidates so the
    aggregation logic gets fully exercised."""
    rows: list[dict[str, Any]] = []
    # ``frontal`` bucket: spiga wins overall + among single-models.
    for i in range(5):
        rows.append(_row(sample_id=f"front_{i}", candidate="spiga", nme=0.030 + i * 0.001))
        rows.append(_row(sample_id=f"front_{i}", candidate="hrnet", nme=0.040 + i * 0.001))
        rows.append(_row(sample_id=f"front_{i}", candidate="fan", nme=0.045 + i * 0.001))
        rows.append(
            _row(sample_id=f"front_{i}", candidate="weighted_median", nme=0.038 + i * 0.001)
        )
        rows.append(
            _row(
                sample_id=f"front_{i}",
                candidate="static_weighted_downweight",
                nme=0.039 + i * 0.001,
            )
        )

    # ``profile_left`` bucket: weighted_median wins; fan is best single.
    for i in range(4):
        rows.append(
            _row(
                sample_id=f"prof_{i}",
                candidate="weighted_median",
                nme=0.060 + i * 0.001,
                runtime_bucket="profile_left",
            )
        )
        rows.append(
            _row(
                sample_id=f"prof_{i}",
                candidate="fan",
                nme=0.065 + i * 0.001,
                runtime_bucket="profile_left",
                failure=(i == 0),
            )
        )
        rows.append(
            _row(
                sample_id=f"prof_{i}",
                candidate="spiga",
                nme=0.080 + i * 0.001,
                runtime_bucket="profile_left",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregator_groups_by_runtime_bucket_and_candidate() -> None:
    rows = _synthetic_rows()

    metrics = aggregate_runtime_bucket_metrics(rows)

    assert set(metrics.keys()) == {"frontal", "profile_left"}
    frontal = metrics["frontal"]
    assert frontal.sample_count == 5
    assert set(frontal.per_candidate.keys()) == {
        "spiga",
        "hrnet",
        "fan",
        "weighted_median",
        "static_weighted_downweight",
    }
    # Each candidate aggregate covers all 5 samples in the bucket.
    for candidate_metrics in frontal.per_candidate.values():
        assert candidate_metrics.sample_count == 5


def test_best_candidate_and_best_single_use_mean_nme_with_tie_breaks() -> None:
    rows = _synthetic_rows()

    metrics = aggregate_runtime_bucket_metrics(rows)

    # Frontal: spiga has the lowest mean NME overall AND among singles.
    frontal = metrics["frontal"]
    assert frontal.best_candidate == "spiga"
    assert frontal.best_single_candidate == "spiga"
    assert frontal.best_candidate_mean_nme == pytest.approx(0.032, rel=1e-3)

    # Profile-left: weighted_median wins overall; fan is best single
    # despite having a failure (mean is still lower than spiga's).
    profile = metrics["profile_left"]
    assert profile.best_candidate == "weighted_median"
    assert profile.best_single_candidate == "fan"
    assert profile.best_candidate_mean_nme < profile.best_single_mean_nme  # type: ignore[operator]


def test_selected_candidate_metrics_only_populate_when_present() -> None:
    rows = _synthetic_rows()

    with_sel = aggregate_runtime_bucket_metrics(rows, selected_candidate="weighted_median")
    without_sel = aggregate_runtime_bucket_metrics(rows)
    missing_sel = aggregate_runtime_bucket_metrics(rows, selected_candidate="not_present")

    assert with_sel["frontal"].selected_candidate == "weighted_median"
    assert with_sel["frontal"].selected_candidate_mean_nme is not None
    assert without_sel["frontal"].selected_candidate is None
    assert missing_sel["frontal"].selected_candidate is None


def test_unknown_bucket_sorts_last_in_payload() -> None:
    rows = [
        _row(sample_id="a", candidate="spiga", nme=0.05, runtime_bucket="unknown"),
        _row(sample_id="b", candidate="spiga", nme=0.05, runtime_bucket="frontal"),
    ]

    metrics = aggregate_runtime_bucket_metrics(rows)

    keys = list(metrics.keys())
    assert keys == ["frontal", "unknown"]


def test_rows_with_non_finite_nme_are_skipped() -> None:
    rows = [
        _row(sample_id="a", candidate="spiga", nme=float("nan")),
        _row(sample_id="b", candidate="spiga", nme=0.05),
    ]

    metrics = aggregate_runtime_bucket_metrics(rows)

    assert metrics["frontal"].per_candidate["spiga"].sample_count == 1


def test_p90_nme_and_failure_rate_present_per_candidate() -> None:
    rows = [
        _row(sample_id=f"s_{i}", candidate="spiga", nme=0.02 + i * 0.01, failure=(i >= 8))
        for i in range(10)
    ]

    metrics = aggregate_runtime_bucket_metrics(rows)

    spiga = metrics["frontal"].per_candidate["spiga"]
    assert spiga.sample_count == 10
    assert spiga.p90_nme == pytest.approx(0.029 + 0.01 * 9 - 0.01, rel=0.1)
    assert spiga.failure_rate == pytest.approx(0.2, rel=1e-3)


def test_static_weighted_downweight_and_weighted_median_surface_at_bucket_level() -> None:
    rows = _synthetic_rows()

    metrics = aggregate_runtime_bucket_metrics(rows)

    frontal = metrics["frontal"]
    assert frontal.static_weighted_downweight_mean_nme is not None
    assert frontal.weighted_median_mean_nme is not None
    # When a top-level candidate is absent from the bucket the field is None.
    profile = metrics["profile_left"]
    assert profile.static_weighted_downweight_mean_nme is None
    assert profile.weighted_median_mean_nme is not None


def test_default_single_model_set_matches_runtime_resolver_defaults() -> None:
    """The aggregator's default single-model list must stay in sync with
    the runtime resolver default candidates so external defaults updates
    can't silently drift."""
    assert {"fan", "hrnet", "spiga", "orformer"} == SINGLE_MODEL_CANDIDATES


# ---------------------------------------------------------------------------
# Writers + CSV round trip
# ---------------------------------------------------------------------------


def test_write_runtime_bucket_json_round_trips(tmp_path: Path) -> None:
    metrics = aggregate_runtime_bucket_metrics(_synthetic_rows())

    out = write_runtime_bucket_json(metrics, tmp_path / "gt_runtime_bucket_metrics.json")
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert set(payload.keys()) == {"frontal", "profile_left"}
    assert payload["frontal"]["best_candidate"] == "spiga"
    assert "per_candidate" in payload["frontal"]


def test_write_runtime_bucket_csv_has_one_row_per_bucket_candidate(tmp_path: Path) -> None:
    rows = _synthetic_rows()
    metrics = aggregate_runtime_bucket_metrics(rows, selected_candidate="weighted_median")

    out = write_runtime_bucket_csv(metrics, tmp_path / "gt_runtime_bucket_metrics.csv")

    with out.open(encoding="utf-8") as handle:
        rows_read = list(csv.DictReader(handle))
    expected_total = sum(len(m.per_candidate) for m in metrics.values())
    assert len(rows_read) == expected_total
    # Header is stable.
    with out.open(encoding="utf-8") as handle:
        headers = next(csv.reader(handle))
    assert tuple(headers) == GT_RUNTIME_BUCKET_CSV_COLUMNS
    # is_selected_candidate flag is 1 only for weighted_median rows.
    sel_rows = [row for row in rows_read if row["is_selected_candidate"] == "1"]
    assert sel_rows
    assert all(row["candidate"] == "weighted_median" for row in sel_rows)


def test_load_candidate_table_csv_coerces_types(tmp_path: Path) -> None:
    src = tmp_path / "candidate_table.csv"
    src.write_text(
        "sample_id,candidate,nme,failure,runtime_bucket\n"
        "a,spiga,0.040,0,frontal\n"
        "a,fan,0.060,1,frontal\n",
        encoding="utf-8",
    )

    rows = load_candidate_table_csv(src)

    assert len(rows) == 2
    assert rows[0]["nme"] == pytest.approx(0.040)
    assert rows[0]["failure"] is False
    assert rows[1]["failure"] is True


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_writes_json_and_csv_artifacts(tmp_path: Path) -> None:
    from tools.landmarks.build_gt_runtime_bucket_metrics import main

    table_path = tmp_path / "candidate_table.csv"
    out_dir = tmp_path / "candidate_search"
    with table_path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "candidate", "nme", "failure", "runtime_bucket"),
        )
        writer.writeheader()
        writer.writerow(
            dict(sample_id="a", candidate="spiga", nme=0.04, failure=0, runtime_bucket="frontal")
        )
        writer.writerow(
            dict(sample_id="a", candidate="fan", nme=0.06, failure=0, runtime_bucket="frontal")
        )

    rc = main(
        [
            "--candidate-table",
            str(table_path),
            "--output-dir",
            str(out_dir),
            "--selected-candidate",
            "spiga",
        ]
    )

    assert rc == 0
    assert (out_dir / "gt_runtime_bucket_metrics.json").exists()
    assert (out_dir / "gt_runtime_bucket_metrics.csv").exists()
    payload = json.loads((out_dir / "gt_runtime_bucket_metrics.json").read_text())
    assert payload["frontal"]["selected_candidate"] == "spiga"


def test_cli_errors_when_candidate_table_missing(tmp_path: Path) -> None:
    from tools.landmarks.build_gt_runtime_bucket_metrics import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--candidate-table",
                str(tmp_path / "missing.csv"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert "candidate-table not found" in str(exc.value)


# ---------------------------------------------------------------------------
# Pipeline in-process wiring — ``_emit_gt_runtime_bucket_artifacts``
# ---------------------------------------------------------------------------


def test_pipeline_helper_writes_artifacts_into_candidate_dir(tmp_path: Path) -> None:
    """``_emit_gt_runtime_bucket_artifacts`` reads
    ``candidate_dir/gt_runtime_bucket_candidate_table.csv`` and drops the JSON+CSV
    aggregates into ``candidate_dir`` per the ticket layout."""
    from tools.landmarks.run_landmark_resolver_pipeline import (
        _emit_gt_runtime_bucket_artifacts,
    )

    candidate_dir = tmp_path / "candidate_search"
    candidate_dir.mkdir(parents=True)

    # Minimal candidate_table.csv covering two buckets.
    table = candidate_dir / "gt_runtime_bucket_candidate_table.csv"
    with table.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "candidate", "nme", "failure", "runtime_bucket"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "a",
                "candidate": "spiga",
                "nme": 0.04,
                "failure": 0,
                "runtime_bucket": "frontal",
            }
        )
        writer.writerow(
            {
                "sample_id": "a",
                "candidate": "weighted_median",
                "nme": 0.05,
                "failure": 0,
                "runtime_bucket": "frontal",
            }
        )

    # best_setup.json declares ``weighted_median`` so the report tags it.
    best_setup = candidate_dir / "best_setup.json"
    best_setup.write_text(
        json.dumps({"candidate": {"strategy": "weighted_median"}}),
        encoding="utf-8",
    )

    from types import SimpleNamespace

    paths = SimpleNamespace(
        best_setup=candidate_dir / "best_setup.json",
        candidate_dir=candidate_dir,
    )
    _emit_gt_runtime_bucket_artifacts(paths)  # type: ignore[arg-type]

    json_path = candidate_dir / "gt_runtime_bucket_metrics.json"
    csv_path = candidate_dir / "gt_runtime_bucket_metrics.csv"
    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["frontal"]["selected_candidate"] == "weighted_median"


def test_pipeline_helper_is_noop_when_table_missing(tmp_path: Path) -> None:
    """No candidate_table means the helper logs + returns; no artifacts."""
    from tools.landmarks.run_landmark_resolver_pipeline import (
        _emit_gt_runtime_bucket_artifacts,
    )

    candidate_dir = tmp_path / "candidate_search"
    candidate_dir.mkdir(parents=True)

    from types import SimpleNamespace

    paths = SimpleNamespace(
        best_setup=candidate_dir / "best_setup.json",
        candidate_dir=candidate_dir,
    )
    _emit_gt_runtime_bucket_artifacts(paths)  # type: ignore[arg-type]

    assert not (candidate_dir / "gt_runtime_bucket_metrics.json").exists()
    assert not (candidate_dir / "gt_runtime_bucket_metrics.csv").exists()


def test_pipeline_helper_survives_aggregator_failure(tmp_path: Path) -> None:
    """A malformed candidate_table.csv must NOT raise — the helper is
    crash-safe so candidate-search aggregation stays green."""
    from tools.landmarks.run_landmark_resolver_pipeline import (
        _emit_gt_runtime_bucket_artifacts,
    )

    candidate_dir = tmp_path / "candidate_search"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "gt_runtime_bucket_candidate_table.csv").write_text(
        "garbage\n", encoding="utf-8"
    )

    from types import SimpleNamespace

    paths = SimpleNamespace(
        best_setup=candidate_dir / "best_setup.json",
        candidate_dir=candidate_dir,
    )
    # Crash-safe: this call must NOT raise.
    _emit_gt_runtime_bucket_artifacts(paths)  # type: ignore[arg-type]
