"""
Playwright-based crawler: load URL, full-page screenshot, sanitized DOM text.

Used by the data pipeline and can be invoked by the inference API for on-demand URL analysis.
"""

from .crawler import CrawlResult, classify_crawl_error, crawl_url, crawl_url_with_retries

__all__ = [
    "CrawlResult",
    "classify_crawl_error",
    "crawl_url",
    "crawl_url_with_retries",
]
