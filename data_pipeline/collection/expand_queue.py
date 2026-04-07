"""
Build ordered crawl work queues from an existing manifest plus new URL sources.
"""

from __future__ import annotations

from typing import Any


def build_expansion_work_queue(
    existing_rows: list[dict[str, Any]],
    new_entries: list[tuple[str, str, str, str | None]],
) -> list[tuple[str, str, str, str | None]]:
    """
    Preserve manifest order first, then append unseen URLs from ``new_entries``.

    Each tuple is ``(url, label, source, fetched_at)``. Duplicate URLs are skipped
    when they already appear in ``existing_rows`` or earlier in ``new_entries``.
    """
    seen: set[str] = set()
    out: list[tuple[str, str, str, str | None]] = []

    for r in existing_rows:
        u = r.get("url")
        if not u:
            continue
        url_s = str(u).strip()
        if not url_s or url_s in seen:
            continue
        seen.add(url_s)
        lab = r.get("label", "phishing")
        src = r.get("source", "manifest")
        fat = r.get("fetched_at")
        fetched = fat if isinstance(fat, str) and fat.strip() else None
        out.append((url_s, str(lab) if isinstance(lab, str) else "phishing", str(src) if isinstance(src, str) else "manifest", fetched))

    for url, label, source, fetched_at in new_entries:
        url_s = str(url).strip()
        if not url_s or url_s in seen:
            continue
        seen.add(url_s)
        out.append((url_s, label, source, fetched_at))

    return out
