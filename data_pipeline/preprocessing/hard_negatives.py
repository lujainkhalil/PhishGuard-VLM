"""
Merge **hard negative** crawl rows: benign pages that look phishing-like (login forms, branded
logos, security cues). They stay ``label=benign`` but are tagged for training (optional oversampling).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from data_pipeline.feeds.utils import normalize_url

logger = logging.getLogger(__name__)

# Allowed categories for manifest / YAML metadata
HARD_NEGATIVE_CATEGORIES = frozenset(
    {"login_form", "branded", "phishing_lookalike", "general"}
)


def normalize_hard_negative_category(raw: str | None, *, default: str = "general") -> str:
    s = (raw or default).strip().lower().replace(" ", "_")
    if s in HARD_NEGATIVE_CATEGORIES:
        return s
    if s in ("brand", "branded_page"):
        return "branded"
    if s in ("login", "signin", "sign_in"):
        return "login_form"
    if s in ("lookalike", "phish_like", "phishing_like"):
        return "phishing_lookalike"
    logger.debug("Unknown hard_negative_category %r; using %r", raw, default)
    return default


def _norm_url_key(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    nu = normalize_url(u)
    return nu if nu else u


def merge_hard_negative_crawls(
    primary_records: list[dict],
    extra_record_groups: list[list[dict]],
    *,
    default_category: str = "general",
) -> list[dict]:
    """
    Append hard-negative crawl rows that are not already present (by normalized URL).

    Each appended row is forced to ``label=benign``, ``source=hard_negative``, and
    ``hard_negative_category`` set from the record or ``default_category``.
    """
    seen: set[str] = set()
    for r in primary_records:
        k = _norm_url_key(str(r.get("final_url") or r.get("url") or ""))
        if k:
            seen.add(k)

    out = list(primary_records)
    added = 0
    for group in extra_record_groups:
        for r in group:
            url = str(r.get("final_url") or r.get("url") or "").strip()
            k = _norm_url_key(url)
            if not k or k in seen:
                continue
            rec = dict(r)
            lab = (rec.get("label") or "benign").strip().lower()
            if lab == "phishing":
                logger.warning(
                    "Hard-negative manifest contained label=phishing for %s; forcing benign.",
                    url[:80],
                )
            rec["label"] = "benign"
            rec["source"] = "hard_negative"
            cat = rec.get("hard_negative_category") or rec.get("hard_negative_type")
            rec["hard_negative_category"] = normalize_hard_negative_category(
                str(cat) if cat is not None else None,
                default=default_category,
            )
            out.append(rec)
            seen.add(k)
            added += 1

    if added:
        logger.info("Merged %d hard-negative crawl rows (benign, phishing-like).", added)
    return out


def force_hard_negatives_train_split(records: list[dict]) -> None:
    """In-place: rows with ``source == hard_negative`` are assigned ``split=train``."""
    for r in records:
        if r.get("source") == "hard_negative":
            r["split"] = "train"


def load_hard_negative_manifest_paths(
    paths: list[str | Path],
    *,
    project_root: Path | None = None,
) -> list[list[dict]]:
    """Load each JSON crawl manifest; skip missing files with a warning."""
    from .build import load_crawl_manifest

    groups: list[list[dict]] = []
    root = project_root or Path.cwd()
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            logger.warning("Hard-negative crawl manifest not found (skip): %s", path)
            continue
        recs = load_crawl_manifest(path)
        if recs:
            groups.append(recs)
    return groups
