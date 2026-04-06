"""
Load URLs from feed output files (JSON/CSV from run_feed_fetch) for the crawler.
"""

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_urls_from_feeds(
    feeds_dir: str | Path,
    *,
    glob_patterns: list[str] | None = None,
) -> list[tuple[str, str, str, str | None]]:
    """
    Load (url, label, source, fetched_at) from feed JSON and CSV files in feeds_dir.

    ``fetched_at`` is an ISO timestamp string when the feed row provides it; else None.

    Args:
        feeds_dir: Directory containing openphish.json, phishtank.json, etc.
        glob_patterns: File patterns to include; default ["*.json", "*.csv"].

    Returns:
        List of (url, label, source, fetched_at). Duplicates by URL are preserved; caller may dedupe.
    """
    feeds_dir = Path(feeds_dir)
    if not feeds_dir.is_dir():
        logger.warning("Feeds dir does not exist: %s", feeds_dir)
        return []

    patterns = glob_patterns or ["*.json", "*.csv"]
    entries: list[tuple[str, str, str, str | None]] = []
    seen_files: set[Path] = set()

    for pattern in patterns:
        for path in feeds_dir.glob(pattern):
            if path in seen_files:
                continue
            seen_files.add(path)
            if path.suffix.lower() == ".json":
                entries.extend(_load_json(path))
            elif path.suffix.lower() == ".csv":
                entries.extend(_load_csv(path))

    return entries


def _coerce_fetched_at(raw: object) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _load_json(path: Path) -> list[tuple[str, str, str, str | None]]:
    """Load URLs from a JSON feed file (list of objects with url, optional label/source/fetched_at)."""
    out: list[tuple[str, str, str, str | None]] = []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return []
    if not isinstance(data, list):
        logger.warning("Expected list in %s", path)
        return []
    source = path.stem.lower()
    for item in data:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url or not isinstance(url, str):
            continue
        url = url.strip()
        if not url:
            continue
        label = item.get("label", "phishing")
        if not isinstance(label, str):
            label = "phishing"
        src = item.get("source", source)
        if not isinstance(src, str):
            src = source
        fetched_at = _coerce_fetched_at(item.get("fetched_at"))
        out.append((url, label, src, fetched_at))
    logger.info("Loaded %d URLs from %s", len(out), path.name)
    return out


def _load_csv(path: Path) -> list[tuple[str, str, str, str | None]]:
    """Load URLs from a CSV feed file (must have 'url' column; optional fetched_at)."""
    out: list[tuple[str, str, str, str | None]] = []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "url" not in (reader.fieldnames or []):
                logger.warning("CSV %s has no 'url' column", path)
                return []
            source = path.stem.lower()
            for row in reader:
                url = (row.get("url") or "").strip()
                if not url:
                    continue
                label = (row.get("label") or "phishing").strip() or "phishing"
                src = (row.get("source") or source).strip() or source
                fetched_at = _coerce_fetched_at(row.get("fetched_at"))
                out.append((url, label, src, fetched_at))
    except Exception as e:
        logger.warning("Failed to load CSV %s: %s", path, e)
        return []
    logger.info("Loaded %d URLs from %s", len(out), path.name)
    return out
