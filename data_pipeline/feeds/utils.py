"""
Shared utilities for phishing feed collectors: URL normalization, deduplication, and storage.
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedEntry:
    """A single URL entry from a feed (phishing or benign)."""

    url: str
    label: str = "phishing"
    source: str = ""
    fetched_at: str = ""
    # Optional metadata (e.g. PhishTank phish_id, target)
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Export as a flat dict for JSON/CSV storage."""
        d: dict[str, Any] = {
            "url": self.url,
            "label": self.label,
            "source": self.source,
            "fetched_at": self.fetched_at,
        }
        if self.extra:
            d.update(self.extra)
        return d


def normalize_url(url: str) -> str | None:
    """
    Normalize a URL for consistent deduplication and storage.

    - Strip whitespace
    - Lowercase scheme and host
    - Remove default ports (80, 443)
    - Remove fragment
    - Empty or invalid URLs return None
    """
    if not url or not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw:
        return None
    # Ensure scheme for urlparse
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "http").lower()
        netloc = (parsed.netloc or "").lower()
        if not netloc:
            return None
        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        path = parsed.path or "/"
        normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
        return normalized
    except Exception as e:
        logger.debug("Failed to normalize URL %r: %s", url[:80], e)
        return None


def deduplicate_entries(entries: list[FeedEntry], key: str = "url") -> list[FeedEntry]:
    """
    Remove duplicate entries by key (default: url), preserving first occurrence.
    """
    seen: set[str] = set()
    out: list[FeedEntry] = []
    for e in entries:
        k = getattr(e, key, e.url)
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    if len(out) < len(entries):
        logger.info("Deduplication removed %d duplicate(s)", len(entries) - len(out))
    return out


def write_entries_json(entries: list[FeedEntry], path: str | Path) -> Path:
    """Write entries to a JSON file (list of objects)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.to_dict() for e in entries]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d entries to %s", len(data), path)
    return path


def write_entries_csv(entries: list[FeedEntry], path: str | Path) -> Path:
    """
    Write entries to a CSV file. Uses all keys from the first entry's to_dict();
    optional extra fields are flattened into columns.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        # Write header only
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "label", "source", "fetched_at"])
        logger.info("Wrote 0 entries to %s", path)
        return path
    rows = [e.to_dict() for e in entries]
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d entries to %s", len(rows), path)
    return path
