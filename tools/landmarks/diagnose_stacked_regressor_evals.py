#!/usr/bin/env python3
"""Diagnose why stacked residual regressor MSE improves but promotion gates fail.

Example:
    python tools/landmarks/diagnose_stacked_regressor_evals.py \
      --eval-root "$STACKED_EVAL_ROOT" \
      --context-cache "$CONTEXT_CACHE" \
      --output-dir "$STACKED_EVAL_ROOT/diagnostics"

This scans evaluation folders for stacked_regressor_evaluation.json, reloads each
regressor, replays the cached contexts, and writes:
  - stacked_diagnostics_summary.md
  - stacked_diagnostics_summary.json
  - optional per-sample jsonl rows with --write-samples
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pickle
import sys
import typing as T
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.ensemble.runtime_features import stacked_regression_feature_map
from lib.landmarks.ensemble.stacked_regressor import (
    STACKED_REGION_INDICES,
    apply_residual,
    load_stacked_regressor,
)
from lib.landmarks.ensemble.stacked_regressor_evaluation import (
    DEFAULT_CATASTROPHIC_NME,
)
from lib.landmarks.ensemble.stacked_regressor_training import (
    _base_extent_diagonal,
    _target_for_mode,
    select_base_candidate,
)


@dataclass
class GroupAccumulator:
    base_nme: list[float] = field(default_factory=list)
    corrected_nme: list[float] = field(default_factory=list)
    nme_delta: list[float] = field(default_factory=list)  # corrected - base; negative is good
    wins: list[float] = field(default_factory=list)  # base - corrected; positive
    losses: list[float] = field(default_factory=list)  # corrected - base; positive
    ties: int = 0
    sample_clip: list[float] = field(default_factory=list)
    point_clip_rate: list[float] = field(default_factory=list)
    residual_norm_mean: list[float] = field(default_factory=list)
    raw_target_mse: list[float] = field(default_factory=list)
    base_landmark_mse: list[float] = field(default_factory=list)
    corrected_landmark_mse: list[float] = field(default_factory=list)
    landmark_mse_delta: list[float] = field(default_factory=list)
    raw_landmark_mse_delta: list[float] = field(default_factory=list)
    base_catastrophic: int = 0
    corrected_catastrophic: int = 0


@dataclass
class LandmarkAccumulator:
    count: int = 0
    error_delta_sum: np.ndarray = field(default_factory=lambda: np.zeros(68, dtype="float64"))
    wrong_move_sum: np.ndarray = field(default_factory=lambda: np.zeros(68, dtype="float64"))
    residual_norm_sum: np.ndarray = field(default_factory=lambda: np.zeros(68, dtype="float64"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regressor-root", type=Path)
    parser.add_argument("--pattern", default="**/stacked_regressor_evaluation.json")
    parser.add_argument("--catastrophic-nme", type=float, default=DEFAULT_CATASTROPHIC_NME)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--write-samples", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of evaluation JSONs to analyze in parallel. "
            "Use 1 for sequential execution. Each worker loads the context cache once."
        ),
    )
    return parser


def _load_context_cache(path: Path) -> list[T.Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    def flatten(value: T.Any) -> list[T.Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            if "contexts" in value:
                return flatten(value["contexts"])
            out: list[T.Any] = []
            for item in value.values():
                out.extend(flatten(item))
            return out
        if isinstance(value, (list, tuple, set)):
            out = []
            for item in value:
                if isinstance(item, (list, tuple, set, dict)):
                    out.extend(flatten(item))
                else:
                    out.append(item)
            return out
        return [value]

    contexts = flatten(payload)
    if not contexts:
        raise SystemExit(f"No contexts found in {path}")
    return contexts


def _slice_for_bucket(bucket: str) -> str:
    name = (bucket or "").lower()
    if "profile" in name:
        return "profile"
    if "large_yaw" in name or "yaw" in name:
        return "large_yaw"
    if "roll" in name:
        return "rolled"
    if name in {"frontal", "intermediate", "no_pose", ""}:
        return "frontal"
    return "other"


def _nme(pred: np.ndarray, gt: np.ndarray, normalizer: T.Any, visibility: T.Any) -> float | None:
    try:
        norm = float(normalizer)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(norm) or norm <= 0:
        return None

    distances = np.linalg.norm(pred - gt, axis=1)
    if visibility is not None and len(visibility) == distances.shape[0]:
        mask = np.asarray(visibility, dtype=bool)
        if mask.any():
            distances = distances[mask]
    return float(np.mean(distances) / norm)


def _finite_landmarks(value: T.Any) -> np.ndarray | None:
    arr: np.ndarray = np.asarray(value, dtype="float64")
    if arr.shape != (68, 2) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else float("nan")


def _corr(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    aa = np.asarray(a, dtype="float64")
    bb = np.asarray(b, dtype="float64")
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _summarize_group(acc: GroupAccumulator) -> dict[str, T.Any]:
    n = len(acc.base_nme)
    loss_mean = _mean(acc.losses)
    win_mean = _mean(acc.wins)
    return {
        "count": n,
        "base_nme_mean": _mean(acc.base_nme),
        "corrected_nme_mean": _mean(acc.corrected_nme),
        "nme_mean_change": _mean(acc.nme_delta),
        "base_nme_median": _median(acc.base_nme),
        "corrected_nme_median": _median(acc.corrected_nme),
        "nme_median_change": _median(acc.corrected_nme) - _median(acc.base_nme)
        if n
        else float("nan"),
        "win_rate": len(acc.wins) / n if n else float("nan"),
        "loss_rate": len(acc.losses) / n if n else float("nan"),
        "tie_rate": acc.ties / n if n else float("nan"),
        "win_mean_size": win_mean,
        "win_median_size": _median(acc.wins),
        "win_p95_size": _percentile(acc.wins, 95),
        "loss_mean_size": loss_mean,
        "loss_median_size": _median(acc.losses),
        "loss_p95_size": _percentile(acc.losses, 95),
        "loss_to_win_mean_ratio": loss_mean / win_mean
        if math.isfinite(loss_mean) and math.isfinite(win_mean) and win_mean > 0
        else float("nan"),
        "base_catastrophic_rate": acc.base_catastrophic / n if n else float("nan"),
        "corrected_catastrophic_rate": acc.corrected_catastrophic / n if n else float("nan"),
        "catastrophic_rate_change": (
            (acc.corrected_catastrophic - acc.base_catastrophic) / n if n else float("nan")
        ),
        "sample_clip_rate": _mean(acc.sample_clip),
        "point_clip_rate_mean": _mean(acc.point_clip_rate),
        "residual_norm_mean": _mean(acc.residual_norm_mean),
        "raw_target_mse_mean": _mean(acc.raw_target_mse),
        "base_landmark_mse_mean": _mean(acc.base_landmark_mse),
        "corrected_landmark_mse_mean": _mean(acc.corrected_landmark_mse),
        "landmark_mse_mean_change": _mean(acc.landmark_mse_delta),
        "raw_unclipped_landmark_mse_mean_change": _mean(acc.raw_landmark_mse_delta),
        "corr_nme_delta_vs_landmark_mse_delta": _corr(acc.nme_delta, acc.landmark_mse_delta),
        "corr_nme_delta_vs_raw_target_mse": _corr(acc.nme_delta, acc.raw_target_mse),
    }


def _add_group_sample(
    acc: GroupAccumulator,
    *,
    base_nme: float,
    corrected_nme: float,
    catastrophic_nme: float,
    sample_clip: bool,
    point_clip_rate: float,
    residual_norm_mean: float,
    raw_target_mse: float,
    base_landmark_mse: float,
    corrected_landmark_mse: float,
    raw_landmark_mse: float,
) -> None:
    delta = corrected_nme - base_nme
    acc.base_nme.append(base_nme)
    acc.corrected_nme.append(corrected_nme)
    acc.nme_delta.append(delta)
    if delta < -1e-9:
        acc.wins.append(-delta)
    elif delta > 1e-9:
        acc.losses.append(delta)
    else:
        acc.ties += 1
    acc.sample_clip.append(1.0 if sample_clip else 0.0)
    acc.point_clip_rate.append(point_clip_rate)
    acc.residual_norm_mean.append(residual_norm_mean)
    acc.raw_target_mse.append(raw_target_mse)
    acc.base_landmark_mse.append(base_landmark_mse)
    acc.corrected_landmark_mse.append(corrected_landmark_mse)
    acc.landmark_mse_delta.append(corrected_landmark_mse - base_landmark_mse)
    acc.raw_landmark_mse_delta.append(raw_landmark_mse - base_landmark_mse)
    acc.base_catastrophic += 1 if base_nme > catastrophic_nme else 0
    acc.corrected_catastrophic += 1 if corrected_nme > catastrophic_nme else 0


def _resolve_regressor_path(
    eval_json: Path,
    payload: dict[str, T.Any],
    regressor_root: Path | None,
) -> Path | None:
    candidates: list[Path] = []

    raw = str(payload.get("regressor", "") or "")
    if raw:
        p = Path(raw)
        candidates.extend([p, eval_json.parent / p, Path.cwd() / p])

    if regressor_root is not None:
        candidates.append(regressor_root / eval_json.parent.name / "stacked_regressor.json")

    # Layout fallback:
    # .../stacked_regressor_evaluation/<name>/stacked_regressor_evaluation.json
    # .../stacked_regressor/<name>/stacked_regressor.json
    parts = list(eval_json.parts)
    if "stacked_regressor_evaluation" in parts:
        idx = parts.index("stacked_regressor_evaluation")
        guessed = Path(
            *parts[:idx], "stacked_regressor", eval_json.parent.name, "stacked_regressor.json"
        )
        candidates.append(guessed)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _analyze_one(
    *,
    name: str,
    eval_json: Path,
    regressor_path: Path,
    contexts: list[T.Any],
    catastrophic_nme: float,
    write_samples_path: Path | None,
) -> dict[str, T.Any]:
    regressor = load_stacked_regressor(regressor_path)
    groups: dict[str, GroupAccumulator] = defaultdict(GroupAccumulator)
    landmark_acc = LandmarkAccumulator()
    sample_rows: list[dict[str, T.Any]] = []

    sample_file = write_samples_path.open("w", encoding="utf-8") if write_samples_path else None
    try:
        for context in contexts:
            gt = _finite_landmarks(getattr(context, "truth_landmarks", None))
            if gt is None:
                continue

            base_candidate = select_base_candidate(context, regressor.base_candidate_policy)
            if base_candidate is None:
                continue

            base = _finite_landmarks(getattr(base_candidate, "landmarks", None))
            if base is None:
                continue

            normalizer = getattr(context, "normalizer", None)
            visibility = getattr(context, "visibility", None)
            base_nme = _nme(base, gt, normalizer, visibility)
            if base_nme is None:
                continue

            diag = _base_extent_diagonal(base)
            model_landmarks = {
                candidate.name: np.asarray(candidate.landmarks, dtype="float64")
                for candidate in getattr(context, "candidates", ())
                if not getattr(candidate, "is_fusion", False)
            }
            features = stacked_regression_feature_map(
                base_landmarks=base,
                model_landmarks=model_landmarks,
                reference_bbox=None,
                runtime_bucket=str(getattr(context, "runtime_bucket", "") or ""),
                roll_estimate=getattr(context, "roll_estimate", None),
                yaw_estimate=getattr(context, "yaw_estimate", None),
                candidate_yaw_disagreement=getattr(context, "candidate_yaw_disagreement", None),
                max_disagreement_px=getattr(context, "max_disagreement_px", None),
                hard_case_tags=tuple(getattr(context, "hard_case_tags", ()) or ()),
                model_predictions_available=getattr(context, "model_predictions_available", None),
            )

            raw_output = regressor.predict(features)
            target = _target_for_mode(regressor.output_mode, base, gt, diag)
            raw_target_mse = float(np.mean((raw_output - target) ** 2))

            raw_result = apply_residual(
                base,
                raw_output,
                output_mode=regressor.output_mode,
                clip_fraction=0.0,
                bbox_diagonal=diag,
            )
            result = apply_residual(
                base,
                raw_output,
                output_mode=regressor.output_mode,
                clip_fraction=regressor.residual_clip_fraction,
                bbox_diagonal=diag,
            )

            corrected = np.asarray(result.landmarks, dtype="float64")
            raw_corrected = np.asarray(raw_result.landmarks, dtype="float64")
            corrected_nme = _nme(corrected, gt, normalizer, visibility)
            if corrected_nme is None:
                continue

            raw_residual = raw_corrected - base
            clipped_residual = corrected - base
            raw_residual_norm = np.linalg.norm(raw_residual, axis=1)
            clipped_residual_norm = np.linalg.norm(clipped_residual, axis=1)
            clip_px = float(regressor.residual_clip_fraction) * diag
            point_clip_rate = float(np.mean(raw_residual_norm > clip_px)) if clip_px > 0 else 0.0

            base_landmark_mse = float(np.mean(((base - gt) / diag) ** 2))
            corrected_landmark_mse = float(np.mean(((corrected - gt) / diag) ** 2))
            raw_landmark_mse = float(np.mean(((raw_corrected - gt) / diag) ** 2))

            bucket = str(getattr(context, "runtime_bucket", "") or "unknown")
            slice_name = _slice_for_bucket(bucket)

            for key in ("overall", f"slice:{slice_name}", f"bucket:{bucket}"):
                _add_group_sample(
                    groups[key],
                    base_nme=base_nme,
                    corrected_nme=corrected_nme,
                    catastrophic_nme=catastrophic_nme,
                    sample_clip=result.clip_applied,
                    point_clip_rate=point_clip_rate,
                    residual_norm_mean=result.residual_norm_mean,
                    raw_target_mse=raw_target_mse,
                    base_landmark_mse=base_landmark_mse,
                    corrected_landmark_mse=corrected_landmark_mse,
                    raw_landmark_mse=raw_landmark_mse,
                )

            base_error = np.linalg.norm(base - gt, axis=1)
            corrected_error = np.linalg.norm(corrected - gt, axis=1)
            landmark_norm = (
                float(normalizer) if normalizer is not None and float(normalizer) > 0 else diag
            )
            landmark_error_delta = (corrected_error - base_error) / landmark_norm

            ideal = gt - base
            dot = np.sum(clipped_residual * ideal, axis=1)
            moved = clipped_residual_norm > (1e-9 * diag)
            has_target = np.linalg.norm(ideal, axis=1) > (1e-9 * diag)
            wrong_move = np.where(moved & has_target, dot <= 0.0, False)

            landmark_acc.count += 1
            landmark_acc.error_delta_sum += landmark_error_delta
            landmark_acc.wrong_move_sum += wrong_move.astype("float64")
            landmark_acc.residual_norm_sum += clipped_residual_norm / diag

            row = {
                "sample_id": str(getattr(context, "sample_id", "")),
                "bucket": bucket,
                "slice": slice_name,
                "base_candidate": str(getattr(base_candidate, "name", "")),
                "base_nme": base_nme,
                "corrected_nme": corrected_nme,
                "nme_delta": corrected_nme - base_nme,
                "sample_clip": bool(result.clip_applied),
                "point_clip_rate": point_clip_rate,
                "raw_target_mse": raw_target_mse,
                "base_landmark_mse": base_landmark_mse,
                "corrected_landmark_mse": corrected_landmark_mse,
                "raw_unclipped_landmark_mse": raw_landmark_mse,
            }
            if sample_file:
                sample_file.write(json.dumps(row, sort_keys=True) + "\n")
            else:
                sample_rows.append(row)
    finally:
        if sample_file:
            sample_file.close()

    group_summary = {key: _summarize_group(acc) for key, acc in sorted(groups.items())}

    if landmark_acc.count:
        mean_error_delta = landmark_acc.error_delta_sum / landmark_acc.count
        wrong_move_rate = landmark_acc.wrong_move_sum / landmark_acc.count
        mean_residual_norm = landmark_acc.residual_norm_sum / landmark_acc.count
    else:
        mean_error_delta = np.full(68, np.nan)
        wrong_move_rate = np.full(68, np.nan)
        mean_residual_norm = np.full(68, np.nan)

    worst_landmarks = sorted(
        [
            {
                "landmark": int(i),
                "mean_nme_contribution_delta": float(mean_error_delta[i]),
                "wrong_move_rate": float(wrong_move_rate[i]),
                "mean_residual_norm": float(mean_residual_norm[i]),
            }
            for i in range(68)
        ],
        key=lambda x: x["mean_nme_contribution_delta"],
        reverse=True,
    )

    region_summary = {}
    for region, indices in STACKED_REGION_INDICES.items():
        idx = list(indices)
        region_summary[region] = {
            "mean_nme_contribution_delta": float(np.nanmean(mean_error_delta[idx])),
            "wrong_move_rate": float(np.nanmean(wrong_move_rate[idx])),
            "mean_residual_norm": float(np.nanmean(mean_residual_norm[idx])),
        }

    return {
        "name": name,
        "eval_json": str(eval_json),
        "regressor": str(regressor_path),
        "output_mode": regressor.output_mode,
        "base_candidate_policy": regressor.base_candidate_policy,
        "residual_clip_fraction": regressor.residual_clip_fraction,
        "groups": group_summary,
        "worst_landmarks": worst_landmarks,
        "region_summary": region_summary,
        "sample_rows": sample_rows[:0],  # kept out of summary by default
    }


def _fmt(value: T.Any, digits: int = 6) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(f):
        return "nan"
    return f"{f:.{digits}f}"


_WORKER_CONTEXTS: list[T.Any] | None = None
_WORKER_CATASTROPHIC_NME: float = DEFAULT_CATASTROPHIC_NME
_WORKER_WRITE_SAMPLES: bool = False
_WORKER_SAMPLES_DIR: Path | None = None


def _init_worker(
    context_cache: str,
    catastrophic_nme: float,
    write_samples: bool,
    samples_dir: str | None,
) -> None:
    """Load heavy shared inputs once per worker process."""
    global _WORKER_CONTEXTS
    global _WORKER_CATASTROPHIC_NME
    global _WORKER_WRITE_SAMPLES
    global _WORKER_SAMPLES_DIR

    _WORKER_CONTEXTS = _load_context_cache(Path(context_cache))
    _WORKER_CATASTROPHIC_NME = float(catastrophic_nme)
    _WORKER_WRITE_SAMPLES = bool(write_samples)
    _WORKER_SAMPLES_DIR = Path(samples_dir) if samples_dir else None


def _analyze_task_worker(task: dict[str, str]) -> dict[str, T.Any]:
    """Analyze one evaluation report inside a worker process."""
    if _WORKER_CONTEXTS is None:
        raise RuntimeError("worker contexts were not initialized")

    name = task["name"]
    sample_path = (
        _WORKER_SAMPLES_DIR / f"{name}.jsonl"
        if _WORKER_WRITE_SAMPLES and _WORKER_SAMPLES_DIR is not None
        else None
    )
    return _analyze_one(
        name=name,
        eval_json=Path(task["eval_json"]),
        regressor_path=Path(task["regressor"]),
        contexts=_WORKER_CONTEXTS,
        catastrophic_nme=_WORKER_CATASTROPHIC_NME,
        write_samples_path=sample_path,
    )


def _analyze_task_sequential(
    task: dict[str, str],
    *,
    contexts: list[T.Any],
    catastrophic_nme: float,
    write_samples: bool,
    samples_dir: Path | None,
) -> dict[str, T.Any]:
    name = task["name"]
    sample_path = (
        samples_dir / f"{name}.jsonl" if write_samples and samples_dir is not None else None
    )
    return _analyze_one(
        name=name,
        eval_json=Path(task["eval_json"]),
        regressor_path=Path(task["regressor"]),
        contexts=contexts,
        catastrophic_nme=catastrophic_nme,
        write_samples_path=sample_path,
    )


def _markdown(results: list[dict[str, T.Any]], top_k: int) -> str:
    lines: list[str] = []
    lines.append("# Stacked regressor diagnostics")
    lines.append("")
    lines.append("## Run ranking")
    lines.append("")
    lines.append(
        "| run | mode | base | clip | n | mean ΔNME | median ΔNME | win | loss | cat Δ | sample clip | point clip |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ranked = sorted(
        results,
        key=lambda r: (
            r["groups"].get("overall", {}).get("catastrophic_rate_change", float("inf")),
            r["groups"].get("overall", {}).get("nme_mean_change", float("inf")),
        ),
    )
    for r in ranked:
        o = r["groups"].get("overall", {})
        lines.append(
            f"| {r['name']} | {r['output_mode']} | {r['base_candidate_policy']} | "
            f"{_fmt(r['residual_clip_fraction'], 4)} | {o.get('count', 0)} | "
            f"{_fmt(o.get('nme_mean_change'))} | {_fmt(o.get('nme_median_change'))} | "
            f"{_fmt(o.get('win_rate'), 3)} | {_fmt(o.get('loss_rate'), 3)} | "
            f"{_fmt(o.get('catastrophic_rate_change'), 4)} | "
            f"{_fmt(o.get('sample_clip_rate'), 3)} | {_fmt(o.get('point_clip_rate_mean'), 3)} |"
        )

    for r in ranked:
        o = r["groups"].get("overall", {})
        lines.append("")
        lines.append(f"## {r['name']}")
        lines.append("")
        lines.append("### 1. Does stacked_residual improve NME anywhere?")
        lines.append("")
        groups = [
            (k, v)
            for k, v in r["groups"].items()
            if k.startswith("slice:") or k.startswith("bucket:")
        ]
        improved = [
            (k, v)
            for k, v in groups
            if float(v.get("nme_mean_change", float("nan"))) < 0
            or float(v.get("nme_median_change", float("nan"))) < 0
        ]
        if improved:
            lines.append("| group | n | mean ΔNME | median ΔNME | win | loss | cat Δ |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for k, v in sorted(improved, key=lambda item: item[1].get("nme_mean_change", 999))[
                :top_k
            ]:
                lines.append(
                    f"| {k} | {v.get('count', 0)} | {_fmt(v.get('nme_mean_change'))} | "
                    f"{_fmt(v.get('nme_median_change'))} | {_fmt(v.get('win_rate'), 3)} | "
                    f"{_fmt(v.get('loss_rate'), 3)} | {_fmt(v.get('catastrophic_rate_change'), 4)} |"
                )
        else:
            lines.append("No bucket or slice shows mean/median NME improvement.")

        lines.append("")
        lines.append("### 2. Are wins tiny and losses larger?")
        lines.append("")
        lines.append(
            f"- win mean size: `{_fmt(o.get('win_mean_size'))}`\n"
            f"- loss mean size: `{_fmt(o.get('loss_mean_size'))}`\n"
            f"- loss/win mean ratio: `{_fmt(o.get('loss_to_win_mean_ratio'), 3)}`\n"
            f"- win rate: `{_fmt(o.get('win_rate'), 3)}`\n"
            f"- loss rate: `{_fmt(o.get('loss_rate'), 3)}`"
        )

        lines.append("")
        lines.append("### 3. Is the model clipped on most samples?")
        lines.append("")
        lines.append(
            f"- sample clip rate: `{_fmt(o.get('sample_clip_rate'), 3)}`\n"
            f"- mean point clip rate: `{_fmt(o.get('point_clip_rate_mean'), 3)}`\n"
            f"- mean residual norm: `{_fmt(o.get('residual_norm_mean'))}`"
        )

        lines.append("")
        lines.append("### 4. Are residuals moving the wrong landmarks?")
        lines.append("")
        lines.append("| region | mean NME contribution Δ | wrong move rate | mean residual norm |")
        lines.append("|---|---:|---:|---:|")
        for region, stats in sorted(
            r["region_summary"].items(),
            key=lambda item: item[1]["mean_nme_contribution_delta"],
            reverse=True,
        ):
            lines.append(
                f"| {region} | {_fmt(stats['mean_nme_contribution_delta'])} | "
                f"{_fmt(stats['wrong_move_rate'], 3)} | {_fmt(stats['mean_residual_norm'])} |"
            )

        lines.append("")
        lines.append(f"Top {top_k} worsening landmarks:")
        lines.append("")
        lines.append(
            "| landmark | mean NME contribution Δ | wrong move rate | mean residual norm |"
        )
        lines.append("|---:|---:|---:|---:|")
        for item in r["worst_landmarks"][:top_k]:
            lines.append(
                f"| {item['landmark']} | {_fmt(item['mean_nme_contribution_delta'])} | "
                f"{_fmt(item['wrong_move_rate'], 3)} | {_fmt(item['mean_residual_norm'])} |"
            )

        lines.append("")
        lines.append("### 5. Is the base candidate already better than the target suggests?")
        lines.append("")
        lines.append(
            f"- base landmark MSE: `{_fmt(o.get('base_landmark_mse_mean'))}`\n"
            f"- corrected landmark MSE: `{_fmt(o.get('corrected_landmark_mse_mean'))}`\n"
            f"- corrected MSE change: `{_fmt(o.get('landmark_mse_mean_change'))}`\n"
            f"- raw unclipped MSE change: `{_fmt(o.get('raw_unclipped_landmark_mse_mean_change'))}`"
        )

        lines.append("")
        lines.append("### 6. Are MSE coordinates aligned with promotion NME?")
        lines.append("")
        lines.append(
            f"- raw target MSE mean: `{_fmt(o.get('raw_target_mse_mean'))}`\n"
            f"- corr(NME Δ, landmark MSE Δ): `{_fmt(o.get('corr_nme_delta_vs_landmark_mse_delta'), 3)}`\n"
            f"- corr(NME Δ, raw target MSE): `{_fmt(o.get('corr_nme_delta_vs_raw_target_mse'), 3)}`"
        )

    return "\n".join(lines) + "\n"


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    eval_paths = sorted(args.eval_root.glob(args.pattern))
    if not eval_paths:
        raise SystemExit(
            f"No evaluation reports found under {args.eval_root} using {args.pattern!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    tasks: list[dict[str, str]] = []

    samples_dir = args.output_dir / "samples"
    if args.write_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    for eval_json in eval_paths:
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
        regressor_path = _resolve_regressor_path(eval_json, payload, args.regressor_root)
        if regressor_path is None:
            skipped.append(
                {"eval_json": str(eval_json), "reason": "could not resolve regressor path"}
            )
            continue

        name = eval_json.parent.name
        tasks.append(
            {
                "name": name,
                "eval_json": str(eval_json),
                "regressor": str(regressor_path),
            }
        )

    jobs = max(1, int(args.jobs))
    if jobs > 1 and tasks:
        jobs = min(jobs, len(tasks), os.cpu_count() or jobs)
        print(
            f"[diagnose] analyzing {len(tasks)} report(s) with {jobs} worker(s); "
            "each worker loads the context cache once"
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_worker,
            initargs=(
                str(args.context_cache),
                float(args.catastrophic_nme),
                bool(args.write_samples),
                str(samples_dir) if args.write_samples else None,
            ),
        ) as executor:
            future_to_task = {executor.submit(_analyze_task_worker, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as err:
                    skipped.append(
                        {
                            "eval_json": task["eval_json"],
                            "reason": f"analysis failed: {err}",
                        }
                    )
                    print(f"[diagnose] failed {task['name']}: {err}")
                    continue
                print(f"[diagnose] done {task['name']}")
                results.append(result)
    else:
        contexts = _load_context_cache(args.context_cache)
        for task in tasks:
            print(f"[diagnose] {task['name']}: {task['regressor']}")
            results.append(
                _analyze_task_sequential(
                    task,
                    contexts=contexts,
                    catastrophic_nme=args.catastrophic_nme,
                    write_samples=args.write_samples,
                    samples_dir=samples_dir if args.write_samples else None,
                )
            )

    results.sort(key=lambda result: result["name"])

    summary = {
        "eval_root": str(args.eval_root),
        "context_cache": str(args.context_cache),
        "count": len(results),
        "skipped": skipped,
        "results": results,
    }

    json_path = args.output_dir / "stacked_diagnostics_summary.json"
    md_path = args.output_dir / "stacked_diagnostics_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(results, args.top_k), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if skipped:
        print(f"Skipped {len(skipped)} report(s); see JSON summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
