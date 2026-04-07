"""
Merge crawl manifest records with URL and optional screenshot content-hash deduplication.

Preserves order: base records first; appended records skip duplicates.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from data_pipeline.collection.manifest_utils import load_manifest_list

logger = logging.getLogger(__name__)


def _resolve_asset_path(path: str | None, project_root: Path) -> Path | None:
    if not path or not isinstance(path, str):
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def screenshot_bytes_hash(record: dict[str, Any], project_root: Path) -> str | None:
    """
    SHA-256 of screenshot file bytes for ``status == ok`` rows with a readable file.

    Returns None if hashing is not possible (missing path, not ok, I/O error).
    """
    if record.get("status") != "ok":
        return None
    shot = record.get("screenshot_path")
    resolved = _resolve_asset_path(shot, project_root) if shot else None
    if resolved is None or not resolved.is_file():
        return None
    try:
        h = hashlib.sha256()
        with open(resolved, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.debug("Could not hash screenshot for %s: %s", record.get("url"), e)
        return None


def merge_crawl_record_lists(
    base: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    project_root: Path,
    dedupe_url: bool = True,
    dedupe_content_hash: bool = False,
) -> list[dict[str, Any]]:
    """
    Merge ``incoming`` into ``base`` while preserving existing rows first.

    - **URL dedupe:** if a URL already appears in output, skip the later row.
    - **Content hash dedupe:** if ``dedupe_content_hash`` and a new row's screenshot
      hash matches an earlier row's hash, skip the new row (URLs may differ).

    Args:
        base: Existing records (order preserved).
        incoming: New records to append when not duplicate.
        project_root: Root for resolving relative screenshot paths.
        dedupe_url: Drop incoming rows whose URL was already seen.
        dedupe_content_hash: Also drop incoming rows whose screenshot hash collides.

    Returns:
        New list (does not mutate inputs).
    """
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    out: list[dict[str, Any]] = []

    def consider(row: dict[str, Any], *, from_incoming: bool) -> None:
        url = row.get("url")
        if not url:
            logger.debug("Skipping row without url: %s", row)
            return
        url_s = str(url)
        if dedupe_url and url_s in seen_urls:
            if from_incoming:
                logger.debug("Dedupe URL (skip incoming): %s", url_s[:80])
            return
        if dedupe_content_hash:
            ch = screenshot_bytes_hash(row, project_root)
            if ch and ch in seen_hashes:
                if from_incoming:
                    logger.debug("Dedupe content hash (skip incoming): %s", url_s[:80])
                return
        out.append(row)
        seen_urls.add(url_s)
        if dedupe_content_hash:
            ch = screenshot_bytes_hash(row, project_root)
            if ch:
                seen_hashes.add(ch)

    for row in base:
        consider(row, from_incoming=False)
    skipped_incoming = 0
    for row in incoming:
        before = len(out)
        consider(row, from_incoming=True)
        if len(out) == before:
            skipped_incoming += 1
    if skipped_incoming:
        logger.info("Merge: skipped %d incoming rows (dedupe rules)", skipped_incoming)
    logger.info("Merge: %d base + %d incoming -> %d total rows", len(base), len(incoming), len(out))
    return out


def merge_multiple_manifest_files(
    paths: list[Path],
    *,
    project_root: Path,
    dedupe_url: bool = True,
    dedupe_content_hash: bool = False,
) -> list[dict[str, Any]]:
    """
    Load manifests in order and fold each into the accumulated list (first file is base).
    """
    if not paths:
        return []
    acc: list[dict[str, Any]] = []
    for i, p in enumerate(paths):
        batch = load_manifest_list(p)
        if i == 0:
            acc = list(batch)
            continue
        acc = merge_crawl_record_lists(
            acc,
            batch,
            project_root=project_root,
            dedupe_url=dedupe_url,
            dedupe_content_hash=dedupe_content_hash,
        )
    return acc
