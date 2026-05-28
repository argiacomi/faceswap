#!/usr/bin/env python3
"""Duplicate-cluster detection and representative selection for FaceQA.

The pipeline answers, for an extracted faceset:

- Which faces are near-duplicates of each other?
- Which face is the best visual representative of each cluster?
- Which faces are safe to prune?
- Which faces should be routed to human review?

Decisions are deterministic and emit advisory recommendations
(``keep`` / ``review`` / ``prune_candidate``) so the downstream tool can
build a training-ready faceset without losing the originals.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lib.align.objects import FileAlignments
from lib.faceqa.coverage import load_alignments_records
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

# Default cosine-similarity threshold for treating two embeddings as a duplicate pair.
# 0.85 is conservative — most legitimately-different frames of the same subject sit
# above this threshold for any modern face-identity model.
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Default temporal window (frames) used when frame names parse to monotonic indices.
# When ``None`` no temporal constraint is applied (clustering is purely embedding-based).
DEFAULT_TEMPORAL_WINDOW: int | None = None

# Representative-selection weights. Larger value = signal is more important.
QUALITY_WEIGHTS: dict[str, float] = {
    "blur": 1.0,
    "resolution": 0.8,
    "pose": 0.6,
    "distance": 0.6,
    "black_pixel": 0.4,
}

# Cluster size threshold for review-vs-prune routing of close-quality non-representatives.
# Non-representative faces whose normalised quality is within this margin of the best
# face in the cluster are routed to ``review`` instead of ``prune_candidate``.
REVIEW_QUALITY_MARGIN = 0.15

# Recommendation tokens (advisory).
KEEP = "keep"
REVIEW = "review"
PRUNE = "prune_candidate"

_FRAME_NUMBER_RE = re.compile(r"(\d+)")


@dataclass
class DuplicateRecord:
    """One face's duplicate-pipeline outcome."""

    frame: str
    face_index: int
    cluster_id: int
    cluster_size: int
    representative: bool
    recommendation: str
    quality_score: float
    similarity_to_representative: float
    reason: str
    identity_model: str | None = None

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "frame": self.frame,
            "face_index": self.face_index,
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "representative": self.representative,
            "recommendation": self.recommendation,
            "quality_score": self.quality_score,
            "similarity_to_representative": self.similarity_to_representative,
            "reason": self.reason,
            "identity_model": self.identity_model,
        }


@dataclass
class DuplicateReport:
    """Aggregate duplicate-pipeline report."""

    alignments: str = ""
    identity_model: str | None = None
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    temporal_window: int | None = None
    total_faces: int = 0
    cluster_count: int = 0
    multi_face_clusters: int = 0
    keep_count: int = 0
    review_count: int = 0
    prune_candidate_count: int = 0
    skipped_no_embedding: int = 0
    records: list[DuplicateRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "alignments": self.alignments,
            "identity_model": self.identity_model,
            "similarity_threshold": self.similarity_threshold,
            "temporal_window": self.temporal_window,
            "total_faces": self.total_faces,
            "cluster_count": self.cluster_count,
            "multi_face_clusters": self.multi_face_clusters,
            "keep_count": self.keep_count,
            "review_count": self.review_count,
            "prune_candidate_count": self.prune_candidate_count,
            "skipped_no_embedding": self.skipped_no_embedding,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_index(frame: str) -> int | None:
    """Return the trailing numeric index of a frame name when present."""
    match = _FRAME_NUMBER_RE.findall(frame)
    if not match:
        return None
    return int(match[-1])


def _pick_identity_model(
    faces: T.Iterable[FileAlignments], *, model: str | None
) -> tuple[str | None, int]:
    """Pick the embedding model name to cluster on and count its coverage."""
    counts: dict[str, int] = {}
    for face in faces:
        for key, vec in face.identity.items():
            if vec is not None and len(np.asarray(vec).ravel()) > 0:
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None, 0
    if model is not None:
        return (model, counts.get(model, 0))
    # Otherwise pick the model with the largest coverage; tie-break by name for determinism.
    best_model = max(counts.items(), key=lambda item: (item[1], -ord(item[0][0])))[0]
    return best_model, counts[best_model]


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    result: np.ndarray = matrix / norms
    return result


def _quality_score(face: FileAlignments) -> float:
    """Return a deterministic per-face quality score for representative selection.

    Higher is better. Uses Sort-style signals already present in alignments
    (blur via Laplacian of the thumb is *not* recomputed here; we rely on
    intrinsic geometry instead so this works without a decoded thumb).
    """
    width = max(int(face.w), 1)
    height = max(int(face.h), 1)
    side = float(min(width, height))
    pose = face.metadata.get("pose") if isinstance(face.metadata, dict) else None
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    if isinstance(pose, dict):
        yaw = abs(float(pose.get("yaw") or 0.0))
        pitch = abs(float(pose.get("pitch") or 0.0))
        roll = abs(float(pose.get("roll") or 0.0))
    blur = 0.0
    distance = 0.0
    black = 0.0
    faceqa = face.metadata.get("faceqa") if isinstance(face.metadata, dict) else None
    if isinstance(faceqa, dict):
        blur = float(faceqa.get("blur_score") or 0.0)
        distance = float(faceqa.get("average_distance") or 0.0)
        black = float(faceqa.get("black_pixel_ratio") or 0.0)
    blur_term = math.tanh(blur / 6.0)
    res_term = math.tanh(side / 512.0)
    pose_term = 1.0 - math.tanh((yaw + pitch + roll) / 90.0)
    distance_term = 1.0 - math.tanh(distance * 4.0)
    black_term = 1.0 - math.tanh(black * 4.0)
    weights = QUALITY_WEIGHTS
    score = (
        weights["blur"] * blur_term
        + weights["resolution"] * res_term
        + weights["pose"] * pose_term
        + weights["distance"] * distance_term
        + weights["black_pixel"] * black_term
    )
    total_weight = sum(weights.values())
    return round(score / total_weight, 6) if total_weight else 0.0


def _build_similarity_graph(
    embeddings: np.ndarray,
    frame_indices: list[int | None],
    *,
    similarity_threshold: float,
    temporal_window: int | None,
) -> list[list[int]]:
    """Return an adjacency list of indices that are duplicate-edges of each other."""
    normalised = _l2_normalise(embeddings)
    similarity = normalised @ normalised.T
    np.fill_diagonal(similarity, 0.0)
    adjacency: list[list[int]] = [[] for _ in range(embeddings.shape[0])]
    candidates = np.argwhere(similarity >= similarity_threshold)
    for raw_i, raw_j in candidates:
        i = int(raw_i)
        j = int(raw_j)
        if i >= j:
            continue
        if temporal_window is not None:
            idx_i = frame_indices[i]
            idx_j = frame_indices[j]
            if idx_i is not None and idx_j is not None and abs(idx_i - idx_j) > temporal_window:
                continue
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency


def _connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    """Return components as sorted-index lists for deterministic output."""
    visited = [False] * len(adjacency)
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbour in adjacency[node]:
                if not visited[neighbour]:
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _classify(
    cluster: list[int],
    quality_scores: list[float],
    *,
    representative_index: int,
) -> dict[int, tuple[str, str]]:
    """Map cluster member index → (recommendation, reason)."""
    decisions: dict[int, tuple[str, str]] = {}
    if len(cluster) == 1:
        decisions[cluster[0]] = (KEEP, "single-face cluster (no duplicates detected)")
        return decisions
    rep_score = quality_scores[representative_index]
    for member in cluster:
        if member == representative_index:
            decisions[member] = (KEEP, "best-quality representative of the cluster")
            continue
        score = quality_scores[member]
        delta = rep_score - score
        if delta <= REVIEW_QUALITY_MARGIN:
            decisions[member] = (REVIEW, "near-duplicate with near-best quality — review")
        else:
            decisions[member] = (
                PRUNE,
                f"duplicate of representative (quality delta {delta:.2f})",
            )
    return decisions


def _records_for_cluster(
    cluster: list[int],
    *,
    cluster_id: int,
    embeddings: np.ndarray,
    quality_scores: list[float],
    face_keys: list[tuple[str, int]],
    identity_model: str | None,
) -> list[DuplicateRecord]:
    representative_index = max(cluster, key=lambda idx: (quality_scores[idx], -idx))
    decisions = _classify(cluster, quality_scores, representative_index=representative_index)
    normalised = _l2_normalise(embeddings)
    rep_embedding = normalised[representative_index]
    records: list[DuplicateRecord] = []
    ordered = sorted(
        cluster,
        key=lambda idx: (
            0 if idx == representative_index else 1,
            -quality_scores[idx],
            idx,
        ),
    )
    for member in ordered:
        recommendation, reason = decisions[member]
        frame, face_index = face_keys[member]
        similarity = float(normalised[member] @ rep_embedding)
        records.append(
            DuplicateRecord(
                frame=frame,
                face_index=face_index,
                cluster_id=cluster_id,
                cluster_size=len(cluster),
                representative=bool(member == representative_index),
                recommendation=recommendation,
                quality_score=round(quality_scores[member], 4),
                similarity_to_representative=round(similarity, 4),
                reason=reason,
                identity_model=identity_model,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


def cluster_duplicates(
    alignments_path: str | Path,
    *,
    identity_model: str | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    temporal_window: int | None = DEFAULT_TEMPORAL_WINDOW,
) -> DuplicateReport:
    """Cluster the faces in ``alignments_path`` and return a deterministic report."""
    rows = load_alignments_records(alignments_path)
    selected_model, coverage = _pick_identity_model(
        (face for _, _, face in rows), model=identity_model
    )
    if selected_model is None or coverage == 0:
        # No embeddings at all: every face becomes its own review-routed singleton.
        records: list[DuplicateRecord] = []
        for cluster_id, (frame, idx, face) in enumerate(rows):
            records.append(
                DuplicateRecord(
                    frame=frame,
                    face_index=idx,
                    cluster_id=cluster_id,
                    cluster_size=1,
                    representative=True,
                    recommendation=REVIEW,
                    quality_score=_quality_score(face),
                    similarity_to_representative=1.0,
                    reason="no identity embedding available; routed to review",
                    identity_model=None,
                )
            )
        return _summarise(
            DuplicateReport(
                alignments=str(alignments_path),
                identity_model=None,
                similarity_threshold=similarity_threshold,
                temporal_window=temporal_window,
                total_faces=len(records),
                records=records,
                skipped_no_embedding=len(records),
            )
        )

    embeddings_list: list[np.ndarray] = []
    face_keys: list[tuple[str, int]] = []
    frame_indices: list[int | None] = []
    quality_scores: list[float] = []
    skipped: list[tuple[str, int, FileAlignments]] = []
    for frame, idx, face in rows:
        vector = face.identity.get(selected_model)
        if vector is None or len(np.asarray(vector).ravel()) == 0:
            skipped.append((frame, idx, face))
            continue
        embeddings_list.append(np.asarray(vector, dtype=np.float32).ravel())
        face_keys.append((frame, idx))
        frame_indices.append(_frame_index(frame))
        quality_scores.append(_quality_score(face))

    embeddings = (
        np.stack(embeddings_list) if embeddings_list else np.empty((0, 0), dtype=np.float32)
    )
    output_records: list[DuplicateRecord] = []
    cluster_id_seq = 0
    if embeddings.size:
        adjacency = _build_similarity_graph(
            embeddings,
            frame_indices,
            similarity_threshold=similarity_threshold,
            temporal_window=temporal_window,
        )
        components = _connected_components(adjacency)
        # Deterministic component ordering by (first frame index, first face index).
        components.sort(
            key=lambda c: (face_keys[c[0]][0], face_keys[c[0]][1]),
        )
        for component in components:
            output_records.extend(
                _records_for_cluster(
                    component,
                    cluster_id=cluster_id_seq,
                    embeddings=embeddings,
                    quality_scores=quality_scores,
                    face_keys=face_keys,
                    identity_model=selected_model,
                )
            )
            cluster_id_seq += 1
    for frame, idx, face in skipped:
        output_records.append(
            DuplicateRecord(
                frame=frame,
                face_index=idx,
                cluster_id=cluster_id_seq,
                cluster_size=1,
                representative=True,
                recommendation=REVIEW,
                quality_score=_quality_score(face),
                similarity_to_representative=1.0,
                reason=f"face missing '{selected_model}' embedding; routed to review",
                identity_model=selected_model,
            )
        )
        cluster_id_seq += 1
    report = DuplicateReport(
        alignments=str(alignments_path),
        identity_model=selected_model,
        similarity_threshold=similarity_threshold,
        temporal_window=temporal_window,
        total_faces=len(output_records),
        records=output_records,
        skipped_no_embedding=len(skipped),
    )
    return _summarise(report)


def _summarise(report: DuplicateReport) -> DuplicateReport:
    """Populate aggregate counts on the report from its records list."""
    cluster_ids: set[int] = set()
    multi: set[int] = set()
    keep = review = prune = 0
    for record in report.records:
        cluster_ids.add(record.cluster_id)
        if record.cluster_size > 1:
            multi.add(record.cluster_id)
        if record.recommendation == KEEP:
            keep += 1
        elif record.recommendation == REVIEW:
            review += 1
        elif record.recommendation == PRUNE:
            prune += 1
    report.cluster_count = len(cluster_ids)
    report.multi_face_clusters = len(multi)
    report.keep_count = keep
    report.review_count = review
    report.prune_candidate_count = prune
    return report


# ---------------------------------------------------------------------------
# Output helpers — manifests
# ---------------------------------------------------------------------------


_MANIFEST_FIELDS: tuple[str, ...] = (
    "frame",
    "face_index",
    "cluster_id",
    "cluster_size",
    "representative",
    "recommendation",
    "quality_score",
    "similarity_to_representative",
    "reason",
    "identity_model",
)


def write_manifests(report: DuplicateReport, output_dir: str | Path) -> dict[str, Path]:
    """Write ``keep`` and ``prune_candidate`` manifests in CSV + JSONL.

    Review-routed faces are folded into ``keep`` (training-safe default per the
    issue spec). Returns the dict of artefact paths actually written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    keep_records = [r for r in report.records if r.recommendation in (KEEP, REVIEW)]
    prune_records = [r for r in report.records if r.recommendation == PRUNE]
    artifacts["keep_csv"] = _write_csv(out / "keep.csv", keep_records)
    artifacts["keep_jsonl"] = _write_jsonl(out / "keep.jsonl", keep_records)
    artifacts["prune_csv"] = _write_csv(out / "prune_candidates.csv", prune_records)
    artifacts["prune_jsonl"] = _write_jsonl(out / "prune_candidates.jsonl", prune_records)
    return artifacts


def _write_csv(path: Path, records: list[DuplicateRecord]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field_: record.to_dict()[field_] for field_ in _MANIFEST_FIELDS})
    return path


def _write_jsonl(path: Path, records: list[DuplicateRecord]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")
    return path


__all__ = get_module_objects(__name__)
