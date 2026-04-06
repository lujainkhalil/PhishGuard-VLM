"""
Live Playwright crawls of known-reachable sites.

Requires network and a working Chromium install (``playwright install chromium``).
Target: at least 80% success rate on this allowlist (flaky CI may occasionally dip).
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

from data_pipeline.crawler.crawler import crawl_url_with_retries

# Mix of high-uptime sites (including google.com / bbc.com per product checks) plus stable fallbacks.
REACHABLE_URLS = [
    "https://www.google.com",
    "https://www.bbc.com",
    "https://example.com",
    "https://www.wikipedia.org/",
    "https://www.cloudflare.com/",
    "https://www.debian.org/",
]


@pytest.mark.integration
@pytest.mark.network
def test_reachable_urls_success_rate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    shot = tmp_path / "screenshots"
    pages = tmp_path / "pages"
    ok = 0
    failures: list[str] = []
    for url in REACHABLE_URLS:
        r = crawl_url_with_retries(
            url,
            screenshot_dir=shot,
            pages_dir=pages,
            timeout_ms=60_000,
            max_attempts=3,
            retry_backoff_ms=500,
        )
        if r.status == "ok" and r.screenshot_path and r.text_path:
            ok += 1
        else:
            failures.append(f"{url} -> {r.status}: {(r.error or '')[:180]}")

    rate = ok / len(REACHABLE_URLS)
    assert rate >= 0.8, f"success rate {rate:.0%} < 80%; failures={failures}"
