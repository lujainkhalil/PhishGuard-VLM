"""
OpenPhish community feed collector.

Fetches the free community feed (one URL per line), normalizes URLs,
removes duplicates, and optionally writes to JSON or CSV.

Feed: https://openphish.com/feed.txt (or GitHub mirror)
Update frequency: ~every 12 hours.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests

from .utils import FeedEntry, deduplicate_entries, normalize_url, write_entries_csv, write_entries_json

logger = logging.getLogger(__name__)

DEFAULT_FEED_URL = "https://openphish.com/feed.txt"
# GitHub mirror (same community feed; merge with primary to maximize coverage if snapshots differ).
DEFAULT_FEED_URLS = (
    "https://openphish.com/feed.txt",
    "https://raw.githubusercontent.com/openphish/public_feed/main/feed.txt",
)
USER_AGENT = "PhishGuard-VLM/1.0 (research; phishing detection)"
REQUEST_TIMEOUT = 120


def fetch_openphish(
    feed_url: str = DEFAULT_FEED_URL,
    timeout: int = REQUEST_TIMEOUT,
) -> list[FeedEntry]:
    """
    Fetch phishing URLs from the OpenPhish community feed.

    Args:
        feed_url: URL of the feed (one URL per line).
        timeout: HTTP request timeout in seconds.

    Returns:
        List of FeedEntry instances (unnormalized; call normalize_and_dedupe next
        or use collect_openphish for a full run).

    Raises:
        requests.RequestException: On network or HTTP errors.
    """
    logger.info("Fetching OpenPhish feed from %s", feed_url)
    response = requests.get(
        feed_url,
        timeout=(30, timeout),
        headers={"User-Agent": USER_AGENT},
        stream=True,
    )
    response.raise_for_status()
    fetched_at = datetime.now(timezone.utc).isoformat()
    entries: list[FeedEntry] = []
    seen_raw: set[str] = set()
    line_count = 0
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        line_count += 1
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        normalized = normalize_url(raw)
        if normalized is None:
            logger.debug("Skipped invalid or unparseable URL at line %d", line_count)
            continue
        entries.append(
            FeedEntry(
                url=normalized,
                label="phishing",
                source="openphish",
                fetched_at=fetched_at,
            )
        )
    logger.info("Fetched %d URLs from OpenPhish (%d lines)", len(entries), line_count)
    return entries


def _fetch_openphish_merged(urls: list[str], *, timeout: int) -> list[FeedEntry]:
    """Fetch multiple OpenPhish mirrors and merge unique URLs (first source wins order)."""
    seen: set[str] = set()
    merged: list[FeedEntry] = []
    for fu in urls:
        fu = fu.strip()
        if not fu:
            continue
        try:
            batch = fetch_openphish(feed_url=fu, timeout=timeout)
        except Exception as e:
            logger.warning("OpenPhish source failed (%s): %s", fu, e)
            continue
        for e in batch:
            if e.url not in seen:
                seen.add(e.url)
                merged.append(e)
    if merged:
        logger.info("OpenPhish merged total: %d unique URLs from %d source(s)", len(merged), len(urls))
    return merged


def collect_openphish(
    output_path: str | Path,
    feed_url: str | None = None,
    feed_urls: list[str] | tuple[str, ...] | None = None,
    timeout: int = REQUEST_TIMEOUT,
    output_format: Literal["json", "csv"] = "json",
    deduplicate: bool = True,
) -> list[FeedEntry]:
    """
    Fetch OpenPhish feed(s), normalize, deduplicate, and write to file.

    Reads the **entire** text feed(s) from each URL (no row cap). Multiple URLs are
    merged with URL-level deduplication (e.g. primary + GitHub mirror).

    Args:
        output_path: Path for the output file (JSON or CSV).
        feed_url: Single feed URL (deprecated if ``feed_urls`` is set).
        feed_urls: Ordered list of feed URLs to fetch and merge.
        timeout: Per-request read timeout in seconds (connect timeout is separate, 30s).
        output_format: Output file format.
        deduplicate: Whether to remove duplicate URLs (by normalized url).

    Returns:
        List of FeedEntry after normalization and optional deduplication.
    """
    if feed_urls is not None:
        urls = [str(u).strip() for u in feed_urls if u and str(u).strip()]
        if not urls:
            urls = list(DEFAULT_FEED_URLS)
    elif feed_url is not None:
        urls = [feed_url]
    else:
        urls = list(DEFAULT_FEED_URLS)

    entries = _fetch_openphish_merged(urls, timeout=timeout)
    if not entries:
        logger.warning("OpenPhish feed returned no URLs")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("[]")
        else:
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write("url,label,source,fetched_at\n")
        return []

    if deduplicate:
        entries = deduplicate_entries(entries)

    path = Path(output_path)
    if output_format == "csv":
        write_entries_csv(entries, path)
    else:
        write_entries_json(entries, path)
    return entries
