"""
PhishTank online-valid feed collector.

Fetches the downloadable database (online-valid.json), normalizes URLs,
removes duplicates, and optionally writes to JSON or CSV.

Feed: https://data.phishtank.com/data/online-valid.json
With API key: https://data.phishtank.com/data/<app_key>/online-valid.json
Update frequency: hourly. Register at https://phishtank.org/api_register.php
for an application key (higher rate limits).
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests

from .utils import FeedEntry, deduplicate_entries, normalize_url, write_entries_csv, write_entries_json

logger = logging.getLogger(__name__)

BASE_URL = "https://data.phishtank.com/data"
USER_AGENT = "PhishGuard-VLM/1.0 (research; phishing detection)"
REQUEST_TIMEOUT = 360  # Full online-valid.json can be very large


def _build_feed_url(app_key: str | None) -> str:
    """Build online-valid.json URL; use app_key in path if provided."""
    if app_key and app_key.strip():
        return f"{BASE_URL}/{app_key.strip()}/online-valid.json"
    return f"{BASE_URL}/online-valid.json"


def _parse_entry(entry: dict[str, Any], fetched_at: str) -> FeedEntry | None:
    """Parse one PhishTank entry into a FeedEntry. Returns None if URL invalid."""
    url_raw = entry.get("url")
    if not url_raw or not isinstance(url_raw, str):
        return None
    normalized = normalize_url(url_raw.strip())
    if normalized is None:
        return None
    extra: dict[str, Any] = {}
    if "phish_id" in entry:
        extra["phish_id"] = entry["phish_id"]
    if "target" in entry and entry["target"]:
        extra["target"] = str(entry["target"]).strip()
    if "submission_time" in entry:
        extra["submission_time"] = entry["submission_time"]
    if "verification_time" in entry:
        extra["verification_time"] = entry["verification_time"]
    return FeedEntry(
        url=normalized,
        label="phishing",
        source="phishtank",
        fetched_at=fetched_at,
        extra=extra if extra else None,
    )


def fetch_phishtank(
    app_key: str | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> list[FeedEntry]:
    """
    Fetch phishing URLs from the PhishTank online-valid feed.

    Args:
        app_key: Optional PhishTank application key (env PHISHTANK_API_KEY
            used if not provided). Higher rate limits with a key.
        timeout: HTTP request timeout in seconds (feed can be large).

    Returns:
        List of FeedEntry instances. Use collect_phishtank for fetch + write.

    Raises:
        requests.RequestException: On network or HTTP errors.
        ValueError: If the response is not valid JSON or missing expected structure.
    """
    key = app_key or os.environ.get("PHISHTANK_API_KEY", "").strip() or None
    feed_url = _build_feed_url(key)
    if not key:
        logger.warning(
            "PhishTank: no app key; using public URL (rate limited). "
            "Set PHISHTANK_API_KEY for higher limits."
        )
    logger.info("Fetching PhishTank feed from %s", feed_url.split("/")[0] + "/...")
    response = requests.get(
        feed_url,
        timeout=(45, timeout),
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    try:
        data = response.json()
    except Exception as e:
        raise ValueError(f"PhishTank response is not valid JSON: {e}") from e
    fetched_at = datetime.now(timezone.utc).isoformat()
    entries: list[FeedEntry] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                e = _parse_entry(item, fetched_at)
                if e is not None:
                    entries.append(e)
    elif isinstance(data, dict):
        # Some versions wrap in {"entries": [...]} or similar
        raw = data.get("entries", data.get("data", data))
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    e = _parse_entry(item, fetched_at)
                    if e is not None:
                        entries.append(e)
        else:
            raise ValueError("PhishTank JSON: expected list or dict with entries/data list")
    else:
        raise ValueError("PhishTank JSON: expected list or dict")
    logger.info("Fetched %d URLs from PhishTank", len(entries))
    return entries


def collect_phishtank(
    output_path: str | Path,
    app_key: str | None = None,
    timeout: int = REQUEST_TIMEOUT,
    output_format: Literal["json", "csv"] = "json",
    deduplicate: bool = True,
) -> list[FeedEntry]:
    """
    Fetch PhishTank feed, normalize, deduplicate, and write to file.

    Args:
        output_path: Path for the output file (JSON or CSV).
        app_key: PhishTank application key (or set PHISHTANK_API_KEY).
        timeout: Request timeout in seconds.
        output_format: Output file format.
        deduplicate: Whether to remove duplicate URLs.

    Returns:
        List of FeedEntry after normalization and optional deduplication.
    """
    entries = fetch_phishtank(app_key=app_key, timeout=timeout)
    if not entries:
        logger.warning("PhishTank feed returned no URLs")
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
