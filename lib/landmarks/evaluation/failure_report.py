#!/usr/bin/env python3
"""Lightweight failure debug reports for static landmark pipeline compatibility."""

from __future__ import annotations

import json
from pathlib import Path


def write_failure_report_from_metrics(
    *,
    metrics_path: Path,
    output_dir: Path,
    limit: int,
) -> None:
    """Write small worst-case JSON artifacts from a metrics.json file.

    This intentionally avoids overlay/contact-sheet generation. The deleted
    ``failure_viewer`` CLI was a debug visualization surface; the legacy static
    pipeline only needs stable debug artifacts for smoke tests and summaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    rows = list(payload.get("rows", [])) if isinstance(payload, dict) else []
    worst = sorted(rows, key=lambda row: float(row.get("nme", 0.0) or 0.0), reverse=True)[:limit]
    regressions = [
        row
        for row in rows
        if str(row.get("model", "")) == "ensemble"
        and row.get("delta_vs_best_single") not in ("", None)
        and float(row.get("delta_vs_best_single", 0.0) or 0.0) > 0.0
    ][:limit]
    (output_dir / "worst_cases.json").write_text(
        json.dumps(worst, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ensemble_regressions.json").write_text(
        json.dumps(regressions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["write_failure_report_from_metrics"]
