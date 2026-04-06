"""
Validate processed dataset manifests: text, screenshots, duplicates, length floors, and statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.feeds.utils import normalize_url

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationReport:
    """Counts and optional notes from :func:`validate_processed_manifest`."""

    rows_in: int = 0
    dropped_empty_text: int = 0
    dropped_short_text: int = 0
    dropped_missing_image_path: int = 0
    dropped_invalid_screenshot: int = 0
    dropped_duplicate_url: int = 0
    dropped_duplicate_normalized_url: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "dropped_empty_text": self.dropped_empty_text,
            "dropped_short_text": self.dropped_short_text,
            "dropped_missing_image_path": self.dropped_missing_image_path,
            "dropped_invalid_screenshot": self.dropped_invalid_screenshot,
            "dropped_duplicate_url": self.dropped_duplicate_url,
            "dropped_duplicate_normalized_url": self.dropped_duplicate_normalized_url,
            "rows_out": self.rows_out,
            "notes": list(self.notes),
        }


def screenshot_file_is_valid(
    path: Path,
    *,
    min_bytes: int = 100,
    min_edge_px: int = 1,
) -> bool:
    """
    True if file exists, has minimum size, and PIL can open a non-degenerate RGB image.

    Used to drop failed or empty screenshot captures.
    """
    try:
        if not path.is_file():
            return False
        if path.stat().st_size < min_bytes:
            return False
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            im2 = im2.convert("RGB")
            w, h = im2.size
        if w < min_edge_px or h < min_edge_px:
            return False
    except Exception:
        return False
    return True


def _resolve_path(p: str | Path, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def _row_text_content(row: pd.Series, root: Path) -> str:
    if "text" in row.index and pd.notna(row.get("text")):
        return str(row["text"]).strip()
    tp = row.get("text_path")
    if tp and str(tp).strip():
        p = _resolve_path(str(tp).strip(), root)
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                return ""
    return ""


def _row_image_path(row: pd.Series, root: Path) -> Path | None:
    for key in ("image_path", "screenshot_path"):
        if key in row.index and pd.notna(row.get(key)) and str(row.get(key) or "").strip():
            return _resolve_path(str(row[key]).strip(), root)
    return None


def validate_processed_manifest(
    df: pd.DataFrame,
    project_root: Path,
    *,
    min_text_length: int = 50,
    min_screenshot_bytes: int = 100,
    min_image_edge_px: int = 1,
    dedupe_by_url: bool = True,
    dedupe_by_normalized_url: bool = True,
    url_column: str = "url",
) -> tuple[pd.DataFrame, DatasetValidationReport]:
    """
    Filter a processed manifest DataFrame in place logically (returns a copy):

    - Drops rows with empty / whitespace-only text (after loading from ``text`` or ``text_path``).
    - Drops rows shorter than ``min_text_length`` characters.
    - Drops rows with missing image path or invalid screenshot file.
    - Drops duplicate pages by ``url`` and optionally by normalized URL.

    Does not mutate the input DataFrame.
    """
    report = DatasetValidationReport()
    if df is None or df.empty:
        report.rows_out = 0
        return df.copy() if df is not None else pd.DataFrame(), report

    root = Path(project_root).resolve()
    work = df.copy()
    report.rows_in = len(work)

    # --- Text: empty / short (vectorized if inline ``text`` exists) ---
    if "text" in work.columns:
        work["_val_text"] = work["text"].fillna("").astype(str).str.strip()
    else:
        texts = [_row_text_content(row, root) for _, row in work.iterrows()]
        work["_val_text"] = pd.Series(texts, index=work.index, dtype=object)

    empty_mask = work["_val_text"].str.len() == 0
    report.dropped_empty_text = int(empty_mask.sum())
    work = work.loc[~empty_mask].copy()

    short_mask = work["_val_text"].str.len() < min_text_length
    report.dropped_short_text = int(short_mask.sum())
    work = work.loc[~short_mask].copy()

    # --- Screenshots ---
    def _img_for_row(row: pd.Series) -> Path | None:
        return _row_image_path(row, root)

    work["_val_img_path"] = [_img_for_row(row) for _, row in work.iterrows()]

    missing_img = work["_val_img_path"].isna()
    report.dropped_missing_image_path = int(missing_img.sum())
    work = work.loc[~missing_img].copy()

    valid_mask = work["_val_img_path"].apply(
        lambda p: screenshot_file_is_valid(
            p, min_bytes=min_screenshot_bytes, min_edge_px=min_image_edge_px
        )
    )
    invalid = ~valid_mask
    report.dropped_invalid_screenshot = int(invalid.sum())
    work = work.loc[valid_mask].copy()

    # --- Dedupe URL ---
    if dedupe_by_url and url_column in work.columns:
        n_before = len(work)
        work = work.drop_duplicates(subset=[url_column], keep="first")
        report.dropped_duplicate_url = n_before - len(work)

    if dedupe_by_normalized_url and url_column in work.columns:
        norms = work[url_column].astype(str).map(lambda u: normalize_url(u.strip()) or u.strip().lower())
        work["_val_norm_url"] = norms
        n_before = len(work)
        work = work.drop_duplicates(subset=["_val_norm_url"], keep="first")
        report.dropped_duplicate_normalized_url = n_before - len(work)
        work = work.drop(columns=["_val_norm_url"], errors="ignore")

    work = work.drop(columns=[c for c in ("_val_text", "_val_img_path") if c in work.columns], errors="ignore")
    report.rows_out = len(work)
    return work.reset_index(drop=True), report


def log_manifest_statistics(df: pd.DataFrame, *, prefix: str = "Dataset") -> None:
    """
    Log row counts, split and label distributions, and text-length summaries (uses ``text`` column if present).
    """
    if df is None or df.empty:
        logger.info("%s statistics: empty manifest", prefix)
        return

    n = len(df)
    logger.info("%s statistics: total_rows=%d", prefix, n)

    if "split" in df.columns:
        for split, cnt in df["split"].value_counts().sort_index().items():
            logger.info("%s statistics: split=%s count=%d", prefix, split, int(cnt))

    if "label" in df.columns:
        for lab, cnt in df["label"].value_counts().items():
            logger.info("%s statistics: label=%s count=%d", prefix, lab, int(cnt))

    if "text" in df.columns:
        lens = df["text"].astype(str).str.len()
        logger.info(
            "%s statistics: text_len min=%d max=%d mean=%.1f median=%.1f",
            prefix,
            int(lens.min()),
            int(lens.max()),
            float(lens.mean()),
            float(lens.median()),
        )
    elif "text_path" in df.columns:
        logger.info("%s statistics: text in external files (text_path); length stats skipped", prefix)


def log_validation_report(report: DatasetValidationReport, *, prefix: str = "Validation") -> None:
    """Log drop counts from a validation run."""
    d = report.to_dict()
    logger.info(
        "%s: rows_in=%d rows_out=%d dropped(empty_text=%d, short_text=%d, missing_image=%d, "
        "bad_screenshot=%d, dup_url=%d, dup_norm_url=%d)",
        prefix,
        d["rows_in"],
        d["rows_out"],
        d["dropped_empty_text"],
        d["dropped_short_text"],
        d["dropped_missing_image_path"],
        d["dropped_invalid_screenshot"],
        d["dropped_duplicate_url"],
        d["dropped_duplicate_normalized_url"],
    )
    for note in report.notes:
        logger.info("%s note: %s", prefix, note)
