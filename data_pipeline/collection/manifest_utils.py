"""
Load and write crawl manifest JSON (same schema as ``run_crawl.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_manifest_list(manifest_path: Path) -> list[dict[str, Any]]:
    """
    Load crawl manifest as a list of records.

    Returns an empty list if the file is missing or invalid.
    """
    if not manifest_path.is_file():
        return []
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Could not read manifest %s: %s", manifest_path, e)
        return []
    if not isinstance(data, list):
        logger.warning("Manifest %s: expected JSON array", manifest_path)
        return []
    return [r for r in data if isinstance(r, dict)]


def load_manifest_by_url(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """
    Load manifest JSON into url -> record (last wins if duplicate URLs in file).
    """
    out: dict[str, dict[str, Any]] = {}
    for row in load_manifest_list(manifest_path):
        u = row.get("url")
        if u:
            out[str(u)] = row
    return out


def write_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    """Write crawl manifest JSON array with stable formatting."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
