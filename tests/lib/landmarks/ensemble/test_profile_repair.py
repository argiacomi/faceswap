#!/usr/bin/env python3
"""Tests for the profile_visible_side_repaired candidate generator (#219)."""

from __future__ import annotations

import types
from typing import cast

import numpy as np

from lib.landmarks.ensemble import profile_repair as repair


def _candidate(name: str, landmarks: np.ndarray) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, landmarks=landmarks, is_fusion=False)


def _metric(veto: tuple[str, ...] = (), plausibility: float = 0.1) -> types.SimpleNamespace:
    return types.SimpleNamespace(geometry_veto_reasons=veto, shape_plausibility_score=plausibility)


def _profile_face() -> np.ndarray:
    rng = np.random.default_rng(1)
    face: np.ndarray = rng.uniform(20.0, 80.0, size=(68, 2)).astype("float32")
    # Give the jaw a left-to-right increasing x so smoothing is well-defined.
    face[0:17, 0] = np.linspace(10.0, 90.0, 17)
    face[repair.NOSE_TIP_68] = (50.0, 50.0)
    return face


# --- construction --------------------------------------------------------


def test_repair_is_finite_68x2_and_named() -> None:
    repaired = repair.make_profile_visible_side_repair(_profile_face(), visible_side="left")
    assert repaired.shape == (68, 2)
    assert np.all(np.isfinite(repaired))
    assert repair.PROFILE_REPAIR_CANDIDATE_NAME == "profile_visible_side_repaired"


def test_repair_unknown_side_returns_source_unchanged() -> None:
    face = _profile_face()
    repaired = repair.make_profile_visible_side_repair(face, visible_side="")
    assert np.allclose(repaired, face[:68, :2])


def test_repair_passes_gates() -> None:
    repaired = repair.make_profile_visible_side_repair(_profile_face(), visible_side="right")
    assert repair.repair_candidate_passes_gates(repaired) is True


def test_repair_rejects_non_finite() -> None:
    bad = _profile_face()
    bad[5] = (np.nan, np.inf)
    assert repair.repair_candidate_passes_gates(bad) is False


def test_repair_jawline_monotonic() -> None:
    repaired = repair.make_profile_visible_side_repair(_profile_face(), visible_side="left")
    jaw_x = repaired[0:17, 0]
    diffs = np.diff(jaw_x)
    assert np.all(diffs >= -1e-5) or np.all(diffs <= 1e-5)


# --- eligibility ---------------------------------------------------------


def test_is_profile_repair_context() -> None:
    assert repair.is_profile_repair_context(("profile_left", "")) is True
    assert repair.is_profile_repair_context(("", "occlusion")) is True
    assert repair.is_profile_repair_context(("frontal", "normal")) is False


# --- source selection ----------------------------------------------------


def test_choose_source_prefers_lowest_visible_distance() -> None:
    a = _candidate("plain_average", _profile_face())
    b = _candidate("spiga", _profile_face())
    metrics = {"plain_average": _metric(), "spiga": _metric()}
    extra = {
        "plain_average": {"visible_side_consensus_distance": 0.5},
        "spiga": {"visible_side_consensus_distance": 0.1},
    }
    chosen = repair.choose_profile_repair_source([a, b], metrics, extra)
    assert chosen is not None
    assert chosen.name == "spiga"


def test_choose_source_none_when_no_usable_landmarks() -> None:
    bad = _candidate("x", np.zeros((10, 2), dtype="float32"))
    assert repair.choose_profile_repair_source([bad], {"x": _metric()}, {}) is None


# --- build ---------------------------------------------------------------


def test_build_returns_none_for_frontal() -> None:
    candidates = [_candidate("hrnet", _profile_face())]
    metrics = {"hrnet": _metric()}
    out = repair.build_profile_repair_landmarks(
        candidates,
        metrics,
        runtime_bucket="frontal",
        condition="normal",
        yaw_estimate=0.0,
    )
    assert out is None


def test_build_returns_landmarks_for_profile() -> None:
    candidates = [_candidate("spiga", _profile_face()), _candidate("hrnet", _profile_face())]
    metrics = {"spiga": _metric(), "hrnet": _metric()}
    out = repair.build_profile_repair_landmarks(
        candidates,
        metrics,
        runtime_bucket="profile_left",
        condition="profile_left",
        yaw_estimate=-40.0,
    )
    assert out is not None
    repaired, source_name, visible_side = out
    assert repaired.shape == (68, 2)
    assert np.all(np.isfinite(repaired))
    assert source_name in {"spiga", "hrnet"}
    assert visible_side == "left"


def test_provenance_and_features() -> None:
    prov = repair.profile_repair_provenance(
        source_candidate="spiga", visible_side="left", reason="all_candidates_vetoed"
    )
    assert prov["profile_repair_source_candidate"] == "spiga"
    assert prov["profile_repair_method"] == repair.PROFILE_REPAIR_METHOD
    features = repair.profile_repair_features(visible_side="left", source_rank=2, shape_score=0.3)
    assert features["candidate_is_profile_repaired"] == 1.0
    assert features["profile_repair_visible_side_left"] == 1.0
    assert features["profile_repair_visible_side_right"] == 0.0
    assert features["profile_repair_source_rank"] == 2.0


def test_candidate_table_exposes_repair_provenance_columns() -> None:
    import types as _types

    from lib.landmarks.ensemble import runtime_resolver_scorer_data as scorer_data

    for column in (
        "profile_repair_used",
        "profile_repair_source_candidate",
        "profile_repair_visible_side",
        "profile_repair_method",
        "profile_repair_reason",
    ):
        assert column in scorer_data.CANDIDATE_TABLE_COLUMNS

    provenance = repair.profile_repair_provenance(
        source_candidate="spiga",
        visible_side="left",
        reason="all_candidates_hard_invalid_v3",
    )
    context = _types.SimpleNamespace(profile_repair_provenance=provenance)
    # Populated for the repair candidate.
    repaired_cols = scorer_data._candidate_repair_provenance_columns(
        cast("scorer_data.SampleCandidateContext", context),
        repair.PROFILE_REPAIR_CANDIDATE_NAME,
    )
    assert repaired_cols["profile_repair_source_candidate"] == "spiga"
    assert repaired_cols["profile_repair_reason"] == "all_candidates_hard_invalid_v3"
    assert repaired_cols["profile_repair_used"] == 1
    # Empty for ordinary candidates.
    other_cols = scorer_data._candidate_repair_provenance_columns(
        cast("scorer_data.SampleCandidateContext", context), "spiga"
    )
    assert other_cols["profile_repair_source_candidate"] == ""
    assert other_cols["profile_repair_used"] == ""


# --- item 4: anchor exclusion -------------------------------------------


def test_is_profile_repair_context_excludes_anchor() -> None:
    assert repair.is_profile_repair_context(("profile_left", "anchor")) is False
    assert repair.is_profile_repair_context(("anchor",)) is False
    assert (
        repair.is_profile_repair_context({"condition": "anchor", "runtime_bucket": "profile_left"})
        is False
    )
    assert repair.is_profile_repair_context(("profile_left", "profile")) is True


# --- item 3: strict-monotonic, self-intersection-free generator ----------


def _plausible_profile_face() -> np.ndarray:
    face: np.ndarray = np.zeros((68, 2), dtype="float64")
    face[0:17, 0] = np.linspace(40, 160, 17)
    face[0:17, 1] = 120 + 30 * np.sin(np.linspace(0, np.pi, 17))
    face[17:22, 0] = np.linspace(50, 90, 5)
    face[17:22, 1] = 70
    face[22:27, 0] = np.linspace(110, 150, 5)
    face[22:27, 1] = 70
    face[27:31, 0] = 100
    face[27:31, 1] = np.linspace(75, 100, 4)
    face[31:36, 0] = np.linspace(88, 112, 5)
    face[31:36, 1] = 105
    face[36:42, 0] = np.linspace(60, 80, 6)
    face[36:42, 1] = 85
    face[42:48, 0] = np.linspace(120, 140, 6)
    face[42:48, 1] = 85
    face[48:60, 0] = np.linspace(75, 125, 12)
    face[48:60, 1] = 130
    face[60:68, 0] = np.linspace(82, 118, 8)
    face[60:68, 1] = 130
    return face


def test_repair_jaw_strictly_monotonic_and_no_self_intersection() -> None:
    from lib.landmarks.evaluation.shape_plausibility import evaluate_shape_plausibility

    crossing = _plausible_profile_face()
    # A self-crossing zig-zag occluded jaw: the generator must still emit a strictly
    # x-monotonic (function-graph) jaw, which cannot self-intersect.
    crossing[0:8, 0] = crossing[8, 0] + np.array([6, -6, 6, -6, 6, -6, 6, -6], dtype="float64")
    for side in ("left", "right"):
        repaired = repair.make_profile_visible_side_repair(crossing.copy(), visible_side=side)
        diffs = np.diff(repaired[0:17, 0])
        assert bool(np.all(diffs > 0) or np.all(diffs < 0)), "jaw must be strictly monotonic"
        metrics = evaluate_shape_plausibility(repaired).metrics
        assert metrics.get("self_intersection_count", 0.0) == 0.0


def test_repair_side_safeguard_overrides_wrong_yaw_label() -> None:
    # Subject-left jaw (9..16) is wide/real; subject-right jaw (0..7) is collapsed.
    face = _plausible_profile_face()
    chin_x = float(face[8, 0])
    face[0:8, 0] = chin_x - (chin_x - face[0:8, 0]) * 0.05  # collapse occluded right half
    # Geometry clearly says visible side is "left"; a wrong "right" label must not flip it.
    as_left = repair.make_profile_visible_side_repair(face.copy(), visible_side="left")
    as_right = repair.make_profile_visible_side_repair(face.copy(), visible_side="right")
    assert np.allclose(as_left, as_right)
