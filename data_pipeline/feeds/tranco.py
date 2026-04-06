"""
Tranco top-sites list — benign URL feed for balancing phishing crawls.

Downloads the official Tranco top-1M CSV (zipped), maps domains to ``https://`` URLs,
normalizes, deduplicates, and writes JSON/CSV compatible with :mod:`feed_loader`.

Official list: https://tranco-list.eu/ (see ``download_url`` in config).
"""

from __future__ import annotations

import csv
import io
import logging
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests

from .utils import FeedEntry, deduplicate_entries, normalize_url, write_entries_csv, write_entries_json

logger = logging.getLogger(__name__)

DEFAULT_DOWNLOAD_URL = "https://tranco-list.eu/top-1m.csv.zip"
USER_AGENT = "PhishGuard-VLM/1.0 (research; benign top-list crawl)"
DEFAULT_TIMEOUT_SEC = 600


def _domain_to_benign_url(domain: str) -> str | None:
    d = domain.strip().lower().rstrip(".")
    if not d or "*" in d or " " in d or "/" in d:
        return None
    return normalize_url(f"https://{d}/")


def parse_tranco_csv_from_text(
    text: str,
    *,
    min_urls: int = 10_000,
    max_urls: int | None = None,
    fetched_at: str | None = None,
) -> list[FeedEntry]:
    """
    Parse Tranco CSV body (``rank,domain`` header + rows) into benign :class:`FeedEntry`\\ s.

    Stops after ``max_urls`` unique URLs, or end of file. If fewer than ``min_urls``
    valid rows exist, returns as many as found (caller may warn).
    """
    ts = fetched_at or datetime.now(timezone.utc).isoformat()
    if max_urls is not None:
        cap = max(int(max_urls), min_urls)
    else:
        cap = 1_000_000

    reader = csv.reader(io.StringIO(text))
    rows_iter = iter(reader)
    first = next(rows_iter, None)
    if first is None:
        return []
    # Skip header if present (Tranco: rank,domain)
    c0 = str(first[0]).strip().lstrip("\ufeff").lower()
    c1h = str(first[1]).strip().lower() if len(first) >= 2 else ""
    if len(first) >= 2 and (c0 in ("rank", "#", "position") or c1h == "domain"):
        start = rows_iter
    else:
        start = iter([first, *rows_iter])

    entries: list[FeedEntry] = []
    seen: set[str] = set()
    for row in start:
        if len(row) < 2:
            continue
        domain = str(row[1]).strip()
        nu = _domain_to_benign_url(domain)
        if nu is None or nu in seen:
            continue
        seen.add(nu)
        entries.append(
            FeedEntry(
                url=nu,
                label="benign",
                source="tranco",
                fetched_at=ts,
                extra={"tranco_rank": str(row[0]).strip()},
            )
        )
        if len(entries) >= cap:
            break

    if len(entries) < min_urls:
        logger.warning(
            "Tranco: only %d URLs parsed (requested minimum %d)",
            len(entries),
            min_urls,
        )
    else:
        logger.info("Tranco: parsed %d benign URLs (min=%d cap=%s)", len(entries), min_urls, cap)
    return entries


def parse_tranco_zip_bytes(
    data: bytes,
    *,
    min_urls: int = 10_000,
    max_urls: int | None = None,
    fetched_at: str | None = None,
) -> list[FeedEntry]:
    """Parse a Tranco ``top-1m.csv.zip`` in memory."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        csv_member = next((n for n in names if n.lower().endswith(".csv")), None)
        if not csv_member:
            raise ValueError(f"Tranco zip has no .csv member: {names[:5]}")
        with zf.open(csv_member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="").read()
    return parse_tranco_csv_from_text(
        text, min_urls=min_urls, max_urls=max_urls, fetched_at=fetched_at
    )


def fetch_tranco(
    download_url: str = DEFAULT_DOWNLOAD_URL,
    *,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    min_urls: int = 10_000,
    max_urls: int | None = None,
) -> list[FeedEntry]:
    """
    Download Tranco top-1M zip and return benign :class:`FeedEntry`\\ s.

    Streams to a temp file to avoid loading multi‑MB zips entirely in RAM twice.
    """
    logger.info("Fetching Tranco list from %s", download_url)
    fetched_at = datetime.now(timezone.utc).isoformat()
    with requests.get(
        download_url,
        timeout=timeout_sec,
        headers={"User-Agent": USER_AGENT},
        stream=True,
    ) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tpath = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    tmp.write(chunk)
    try:
        data = tpath.read_bytes()
        entries = parse_tranco_zip_bytes(
            data, min_urls=min_urls, max_urls=max_urls, fetched_at=fetched_at
        )
    finally:
        try:
            tpath.unlink(missing_ok=True)
        except OSError:
            pass
    return entries


def collect_tranco(
    output_path: str | Path,
    *,
    download_url: str = DEFAULT_DOWNLOAD_URL,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    min_urls: int = 10_000,
    max_urls: int | None = None,
    output_format: Literal["json", "csv"] = "json",
    deduplicate: bool = True,
) -> list[FeedEntry]:
    """Fetch Tranco, optionally dedupe, write ``tranco.json`` / ``tranco.csv``."""
    entries = fetch_tranco(
        download_url,
        timeout_sec=timeout_sec,
        min_urls=min_urls,
        max_urls=max_urls,
    )
    if not entries:
        logger.warning("Tranco returned no URLs")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            output_path.write_text("[]", encoding="utf-8")
        else:
            output_path.write_text("url,label,source,fetched_at\n", encoding="utf-8")
        return []

    if deduplicate:
        entries = deduplicate_entries(entries)

    path = Path(output_path)
    if output_format == "csv":
        write_entries_csv(entries, path)
    else:
        write_entries_json(entries, path)
    return entries
