#!/usr/bin/env python3
"""Shared target-row generation for runtime resolver scorer tools."""

from __future__ import annotations

import typing as T

from tools.landmarks.pipeline_conventions import normalize_source_label
from tools.landmarks.runtime_resolver_scorer_data import (
    DEFAULT_HIGH_GAP_THRESHOLD,
    CandidateQualityRow,
    SampleCandidateContext,
    candidate_table_rows,
    rows_for_context,
)

TaggedRow = tuple[CandidateQualityRow, str]


def tagged_quality_rows(
    contexts: T.Sequence[SampleCandidateContext],
    *,
    high_gap_threshold: float = DEFAULT_HIGH_GAP_THRESHOLD,
) -> list[TaggedRow]:
    """Return scorer target rows tagged with explicit metadata source labels."""
    rows: list[TaggedRow] = []
    for context in contexts:
        source = normalize_source_label(context.source)
        rows.extend(
            (row, source)
            for row in rows_for_context(context, high_gap_threshold=high_gap_threshold)
        )
    return rows


def scorer_candidate_table_rows(
    contexts: T.Sequence[SampleCandidateContext],
) -> list[dict[str, T.Any]]:
    """Return canonical scorer candidate diagnostic rows."""
    return candidate_table_rows(contexts)


def untag_quality_rows(rows: T.Sequence[TaggedRow]) -> list[CandidateQualityRow]:
    """Strip source labels from tagged scorer rows."""
    return [row for row, _source in rows]


def source_quality_rows(rows: T.Sequence[TaggedRow], source: str) -> list[CandidateQualityRow]:
    """Return scorer rows matching a canonical metadata source label."""
    canonical = normalize_source_label(source)
    return [row for row, row_source in rows if normalize_source_label(row_source) == canonical]


__all__ = [
    "TaggedRow",
    "scorer_candidate_table_rows",
    "source_quality_rows",
    "tagged_quality_rows",
    "untag_quality_rows",
]
