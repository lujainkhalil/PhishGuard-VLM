"""Retry policy for crawl_url_with_retries (crawl_url mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from data_pipeline.crawler.crawler import CrawlResult, crawl_url_with_retries


def _fail_timeout() -> CrawlResult:
    return CrawlResult(
        url="https://example.com",
        final_url="https://example.com",
        status="timeout",
        screenshot_path=None,
        text_path=None,
        error="Timeout",
        error_category="timeout",
        permanent_failure=False,
    )


def _ok() -> CrawlResult:
    return CrawlResult(
        url="https://example.com",
        final_url="https://example.com",
        status="ok",
        screenshot_path="/tmp/x.png",
        text_path="/tmp/x.txt",
        error=None,
        error_category=None,
        permanent_failure=False,
    )


@patch("data_pipeline.crawler.crawler.crawl_url")
def test_three_attempts_then_success(mock_crawl, tmp_path: Path) -> None:
    mock_crawl.side_effect = [_fail_timeout(), _fail_timeout(), _ok()]
    s, p = tmp_path / "s", tmp_path / "p"
    r = crawl_url_with_retries(
        "https://example.com",
        screenshot_dir=s,
        pages_dir=p,
        max_attempts=3,
        retry_backoff_ms=0,
    )
    assert r.status == "ok"
    assert mock_crawl.call_count == 3


@patch("data_pipeline.crawler.crawler.crawl_url")
def test_permanent_dns_no_extra_attempts(mock_crawl, tmp_path: Path) -> None:
    mock_crawl.return_value = CrawlResult(
        url="https://dead.invalid",
        final_url="https://dead.invalid",
        status="error",
        screenshot_path=None,
        text_path=None,
        error="net::ERR_NAME_NOT_RESOLVED",
        error_category="dns",
        permanent_failure=True,
    )
    s, p = tmp_path / "s", tmp_path / "p"
    r = crawl_url_with_retries(
        "https://dead.invalid",
        screenshot_dir=s,
        pages_dir=p,
        max_attempts=3,
        retry_backoff_ms=0,
    )
    assert r.status == "error"
    assert mock_crawl.call_count == 1


@patch("data_pipeline.crawler.crawler.crawl_url")
def test_legacy_max_retries_plus_one(mock_crawl, tmp_path: Path) -> None:
    mock_crawl.side_effect = [_fail_timeout(), _ok()]
    s, p = tmp_path / "s", tmp_path / "p"
    r = crawl_url_with_retries(
        "https://example.com",
        screenshot_dir=s,
        pages_dir=p,
        max_retries=1,
        retry_backoff_ms=0,
    )
    assert r.status == "ok"
    assert mock_crawl.call_count == 2
