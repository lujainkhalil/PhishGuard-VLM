"""
Aggregate statistics for crawl manifest lists (counts, class balance, text length).
"""

from __future__ import annotations

import logging
from typing import Any

from data_pipeline.preprocessing.build import resolve_project_path

logger = logging.getLogger(__name__)


def _text_length_for_record(record: dict[str, Any], project_root: Any) -> int | None:
    from pathlib import Path

    tp = record.get("text_path")
    if not tp:
        return None
    path = resolve_project_path(tp, Path(project_root))
    try:
        if not path.is_file():
            return None
        return len(path.read_text(encoding="utf-8", errors="ignore").strip())
    except OSError:
        return None


def compute_crawl_statistics(
    records: list[dict[str, Any]],
    project_root: Any,
    *,
    label_key: str = "label",
) -> dict[str, Any]:
    """
    Compute summary stats for a crawl manifest list.

    Returns:
        Dict with total_records, by_label, ok_count, avg_text_length (over rows with readable text),
        text_lengths_sampled.
    """
    from pathlib import Path

    root = Path(project_root)
    by_label: dict[str, int] = {}
    ok_count = 0
    lengths: list[int] = []

    for r in records:
        lab = r.get(label_key)
        lk = (lab if isinstance(lab, str) else "unknown").strip().lower() or "unknown"
        by_label[lk] = by_label.get(lk, 0) + 1
        if r.get("status") == "ok":
            ok_count += 1
        n = _text_length_for_record(r, root)
        if n is not None:
            lengths.append(n)

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "total_records": len(records),
        "ok_status_count": ok_count,
        "by_label": dict(sorted(by_label.items())),
        "avg_text_length": round(avg_len, 2),
        "text_length_samples": len(lengths),
    }


def log_crawl_statistics(phase: str, records: list[dict[str, Any]], project_root: Any) -> dict[str, Any]:
    """Log and return :func:`compute_crawl_statistics` output."""
    stats = compute_crawl_statistics(records, project_root)
    logger.info(
        "[%s] total=%d ok_status=%d by_label=%s avg_text_len=%.2f (n=%d files read)",
        phase,
        stats["total_records"],
        stats["ok_status_count"],
        stats["by_label"],
        stats["avg_text_length"],
        stats["text_length_samples"],
    )
    return stats
