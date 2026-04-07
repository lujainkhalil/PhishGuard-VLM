"""
Filter raw crawl manifest rows using ``configs/data.yaml``-aligned rules.

Counts drops per rule for logging (training-ready subset).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_pipeline.preprocessing.build import resolve_project_path
from data_pipeline.preprocessing.validation import screenshot_file_is_valid

logger = logging.getLogger(__name__)


@dataclass
class CrawlFilterReport:
    """Per-rule removal counts for :func:`filter_crawl_records_for_training`."""

    rows_in: int = 0
    removed_non_ok_status: int = 0
    removed_bad_label: int = 0
    removed_missing_text_path: int = 0
    removed_short_text: int = 0
    removed_missing_screenshot: int = 0
    removed_invalid_screenshot: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "removed_non_ok_status": self.removed_non_ok_status,
            "removed_bad_label": self.removed_bad_label,
            "removed_missing_text_path": self.removed_missing_text_path,
            "removed_short_text": self.removed_short_text,
            "removed_missing_screenshot": self.removed_missing_screenshot,
            "removed_invalid_screenshot": self.removed_invalid_screenshot,
            "rows_out": self.rows_out,
            "notes": list(self.notes),
        }


def _read_text_len(text_path: Path) -> int | None:
    try:
        if not text_path.is_file():
            return None
        return len(text_path.read_text(encoding="utf-8", errors="ignore").strip())
    except OSError:
        return None


def filter_crawl_records_for_training(
    records: list[dict[str, Any]],
    project_root: Path,
    *,
    min_text_length: int,
    min_screenshot_bytes: int,
    min_image_edge_px: int,
    allowed_labels: set[str],
    require_ok_status: bool = True,
    validate_screenshot_image: bool = True,
) -> tuple[list[dict[str, Any]], CrawlFilterReport]:
    """
    Keep rows suitable for multimodal training materialization.

    Rules (each increment exactly one counter when a row is dropped):

    1. ``require_ok_status``: ``status`` must be ``ok``.
    2. ``label`` (normalized lower) must be in ``allowed_labels``.
    3. ``text_path`` must exist and UTF-8 text length >= ``min_text_length``.
    4. ``screenshot_path`` must exist; file must pass
       :func:`screenshot_file_is_valid` with byte and edge thresholds.

    Args:
        records: Crawl manifest dicts.
        project_root: Repo root for relative paths.
        min_text_length: Minimum stripped text length.
        min_screenshot_bytes: Minimum screenshot file size in bytes.
        min_image_edge_px: Minimum width and height in pixels.
        allowed_labels: Allowed label strings (compare lowercased).
        require_ok_status: If True, drop rows where status != ``ok``.
        validate_screenshot_image: If True, PIL-validate screenshot dimensions/bytes.

    Returns:
        (kept_records, report)
    """
    report = CrawlFilterReport()
    report.rows_in = len(records)
    allowed = {x.lower() for x in allowed_labels}
    out: list[dict[str, Any]] = []

    for r in records:
        if require_ok_status and r.get("status") != "ok":
            report.removed_non_ok_status += 1
            continue
        label = r.get("label")
        if not isinstance(label, str) or label.strip().lower() not in allowed:
            report.removed_bad_label += 1
            continue

        tp_raw = r.get("text_path")
        if not tp_raw:
            report.removed_missing_text_path += 1
            continue
        tp = resolve_project_path(tp_raw, project_root)
        tlen = _read_text_len(tp)
        if tlen is None:
            report.removed_missing_text_path += 1
            continue
        if tlen < min_text_length:
            report.removed_short_text += 1
            continue

        sp_raw = r.get("screenshot_path")
        if not sp_raw:
            report.removed_missing_screenshot += 1
            continue
        sp = resolve_project_path(sp_raw, project_root)
        if not sp.is_file():
            report.removed_missing_screenshot += 1
            continue
        if validate_screenshot_image and not screenshot_file_is_valid(
            sp,
            min_bytes=min_screenshot_bytes,
            min_edge_px=min_image_edge_px,
        ):
            report.removed_invalid_screenshot += 1
            continue

        out.append(r)

    report.rows_out = len(out)
    logger.info(
        "Crawl filter: in=%d out=%d removed: non_ok=%d bad_label=%d no_text=%d short_text=%d "
        "no_shot=%d bad_shot=%d",
        report.rows_in,
        report.rows_out,
        report.removed_non_ok_status,
        report.removed_bad_label,
        report.removed_missing_text_path,
        report.removed_short_text,
        report.removed_missing_screenshot,
        report.removed_invalid_screenshot,
    )
    return out, report


def load_collection_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve collection + quality keys from loaded ``data.yaml`` dict.

    Returns keys: min_text_length, min_screenshot_bytes, min_image_edge_px, allowed_labels.
    """
    coll = config.get("collection") or {}
    quality = config.get("quality") or {}

    min_text = coll.get("min_text_length")
    if min_text is None:
        min_text = int(quality.get("min_text_length", 50))
    else:
        min_text = int(min_text)

    min_bytes = coll.get("min_screenshot_bytes")
    if min_bytes is None:
        min_bytes = int(quality.get("min_screenshot_bytes", 100))
    else:
        min_bytes = int(min_bytes)

    min_edge = coll.get("min_image_edge_px")
    if min_edge is None:
        min_edge = int(quality.get("min_image_edge_px", 1))
    else:
        min_edge = int(min_edge)

    raw_labels = coll.get("allowed_labels")
    if raw_labels is None:
        allowed = {"phishing", "benign"}
    elif isinstance(raw_labels, str):
        allowed = {raw_labels.strip().lower()}
    else:
        allowed = {str(x).strip().lower() for x in raw_labels if str(x).strip()}

    return {
        "min_text_length": min_text,
        "min_screenshot_bytes": min_bytes,
        "min_image_edge_px": min_edge,
        "allowed_labels": allowed,
        "validate_screenshot_image": bool(
            coll.get("validate_screenshot_image", quality.get("validate_screenshot_image", True))
        ),
        "require_ok_status": bool(coll.get("require_ok_status", True)),
    }
