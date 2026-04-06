"""
Feed collectors: OpenPhish, PhishTank (phishing), Tranco (benign top sites).

Each collector fetches URLs, normalizes them, removes duplicates,
and can store results in JSON or CSV.
"""

from .openphish import collect_openphish, fetch_openphish
from .phishtank import collect_phishtank, fetch_phishtank
from .tranco import collect_tranco, fetch_tranco, parse_tranco_csv_from_text, parse_tranco_zip_bytes
from .utils import FeedEntry, deduplicate_entries, normalize_url, write_entries_csv, write_entries_json

__all__ = [
    "FeedEntry",
    "collect_openphish",
    "collect_phishtank",
    "collect_tranco",
    "deduplicate_entries",
    "fetch_openphish",
    "fetch_phishtank",
    "fetch_tranco",
    "normalize_url",
    "parse_tranco_csv_from_text",
    "parse_tranco_zip_bytes",
    "write_entries_csv",
    "write_entries_json",
]
