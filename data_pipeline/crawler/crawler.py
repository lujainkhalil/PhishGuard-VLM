"""
Playwright-based webpage crawler: load URL, capture full-page screenshot, extract DOM text.

Handles timeouts, JavaScript rendering, retries, and errors gracefully.
Output: screenshot path, sanitized text path, final URL, status.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

logger = logging.getLogger(__name__)


def classify_crawl_error(exc: BaseException, *, message: str | None = None) -> tuple[str, bool, str]:
    """
    Classify a Playwright/navigation failure.

    Returns:
        error_category: short machine tag (dns, timeout, ssl, …).
        permanent_failure: if True, skip further attempts this run and on resume (e.g. DNS NXDOMAIN).
        human_summary: concise reason for logs and manifests.
    """
    raw = message if message is not None else str(exc)
    msg = raw
    exc_s = str(exc) if exc is not None else ""
    if "ERR_NAME_NOT_RESOLVED" in msg or "NAME_NOT_RESOLVED" in msg or "ERR_NAME_NOT_RESOLVED" in exc_s:
        return "dns", True, "DNS resolution failed (host not found / NXDOMAIN)"
    if isinstance(exc, PlaywrightTimeoutError):
        return "timeout", False, "Playwright operation timed out"
    if "timeout" in msg.lower() and "exceeded" in msg.lower():
        return "timeout", False, "Navigation or operation timed out"
    if "ERR_INTERNET_DISCONNECTED" in msg:
        return "network", False, "Network disconnected"
    if "ERR_CONNECTION_REFUSED" in msg:
        return "connection_refused", False, "Connection refused"
    if "ERR_CONNECTION_RESET" in msg:
        return "connection_reset", False, "Connection reset"
    if "ERR_CONNECTION_TIMED_OUT" in msg or "ERR_TIMED_OUT" in msg:
        return "connection_timeout", False, "TCP connection timed out"
    if "ERR_SSL" in msg or "ERR_CERT_" in msg:
        return "ssl", True, "SSL/TLS or certificate error"
    if "ERR_BLOCKED_BY_CLIENT" in msg or "ERR_BLOCKED" in msg:
        return "blocked", True, "Request blocked"
    if "net::ERR_" in msg:
        token = "unknown"
        try:
            token = msg.split("net::", 1)[1].split()[0].rstrip(")")
        except IndexError:
            pass
        return "network", False, f"Network error ({token})"
    return "other", False, (raw[:400] + "…") if len(raw) > 400 else raw


@dataclass
class CrawlResult:
    """Result of crawling a single URL."""

    url: str
    final_url: str
    status: str  # "ok" | "timeout" | "error" | "blocked"
    screenshot_path: str | None
    text_path: str | None
    error: str | None = None
    redirect_count: int = 0
    error_category: str | None = None
    permanent_failure: bool = False


def _safe_filename(url: str, suffix: str = "", max_len: int = 64) -> str:
    """Generate a safe filesystem name from URL (hash + optional suffix)."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^\w\-.]", "_", digest)
    if suffix:
        return f"{safe}{suffix}"
    return safe


def _extract_sanitized_text(html: str) -> str:
    """
    Extract visible text from HTML, stripping scripts, styles, and normalizing whitespace.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def crawl_url(
    url: str,
    *,
    screenshot_dir: str | Path,
    pages_dir: str | Path,
    timeout_ms: int = 45_000,
    viewport: dict | None = None,
    wait_until: str = "domcontentloaded",
    extra_wait_ms: int = 500,
) -> CrawlResult:
    """
    Open URL with Playwright, capture full-page screenshot, extract sanitized DOM text.

    Args:
        url: Page URL (will be normalized with http:// if no scheme).
        screenshot_dir: Directory to save full-page screenshot (PNG).
        pages_dir: Directory to save extracted text (.txt).
        timeout_ms: Navigation and operation timeout in milliseconds.
        viewport: Optional {"width": int, "height": int}. Default 1920x1080.
        wait_until: Playwright load state: "load" | "domcontentloaded" | "networkidle".
        extra_wait_ms: Extra milliseconds to wait after load for JS rendering.

    Returns:
        CrawlResult with paths (or None on failure), final_url, status, and optional error.
    """
    screenshot_dir = Path(screenshot_dir)
    pages_dir = Path(pages_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    if not url.strip():
        return CrawlResult(
            url=url,
            final_url=url,
            status="error",
            screenshot_path=None,
            text_path=None,
            error="Empty URL",
            error_category="invalid_url",
            permanent_failure=True,
        )
    if "://" not in url:
        url = "http://" + url

    base_name = _safe_filename(url)
    screenshot_path = screenshot_dir / f"{base_name}.png"
    text_path = pages_dir / f"{base_name}.txt"

    vp = viewport or {"width": 1920, "height": 1080}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    viewport=vp,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    ignore_https_errors=True,
                )
                context.set_default_timeout(timeout_ms)
                page = context.new_page()

                redirect_count = 0

                # Track redirects via response
                def on_response(response):
                    nonlocal redirect_count
                    if response.status in (301, 302, 303, 307, 308):
                        redirect_count += 1

                page.on("response", on_response)

                try:
                    page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                except PlaywrightTimeoutError as e:
                    cat, perm, human = classify_crawl_error(e)
                    logger.info(
                        "Crawl navigation failed url=%s status=timeout category=%s permanent=%s summary=%s detail=%s",
                        url[:160],
                        cat,
                        perm,
                        human,
                        str(e)[:500],
                    )
                    return CrawlResult(
                        url=url,
                        final_url=url,
                        status="timeout",
                        screenshot_path=None,
                        text_path=None,
                        error=str(e),
                        redirect_count=redirect_count,
                        error_category=cat,
                        permanent_failure=perm,
                    )
                except Exception as e:
                    cat, perm, human = classify_crawl_error(e)
                    log_fn = logger.info if perm else logger.warning
                    log_fn(
                        "Crawl navigation failed url=%s status=error category=%s permanent=%s summary=%s detail=%s",
                        url[:160],
                        cat,
                        perm,
                        human,
                        str(e)[:500],
                    )
                    return CrawlResult(
                        url=url,
                        final_url=url,
                        status="error",
                        screenshot_path=None,
                        text_path=None,
                        error=str(e),
                        redirect_count=redirect_count,
                        error_category=cat,
                        permanent_failure=perm,
                    )

                if extra_wait_ms > 0:
                    try:
                        page.wait_for_timeout(extra_wait_ms)
                    except Exception:
                        pass

                final_url = page.url

                try:
                    page.screenshot(path=str(screenshot_path), full_page=True)
                except PlaywrightTimeoutError as e:
                    cat, perm, human = classify_crawl_error(e)
                    logger.info(
                        "Crawl screenshot failed url=%s category=%s permanent=%s summary=%s detail=%s",
                        url[:160],
                        cat,
                        perm,
                        human,
                        str(e)[:500],
                    )
                    return CrawlResult(
                        url=url,
                        final_url=final_url,
                        status="timeout",
                        screenshot_path=None,
                        text_path=None,
                        error=f"screenshot: {e}",
                        redirect_count=redirect_count,
                        error_category="screenshot_timeout",
                        permanent_failure=False,
                    )
                except Exception as e:
                    cat, perm, human = classify_crawl_error(e)
                    logger.warning(
                        "Crawl screenshot failed url=%s category=%s permanent=%s summary=%s detail=%s",
                        url[:160],
                        cat or "screenshot",
                        perm,
                        human,
                        str(e)[:500],
                    )
                    return CrawlResult(
                        url=url,
                        final_url=final_url,
                        status="error",
                        screenshot_path=None,
                        text_path=None,
                        error=str(e),
                        redirect_count=redirect_count,
                        error_category=cat or "screenshot",
                        permanent_failure=False,
                    )

                try:
                    html = page.content()
                except Exception as e:
                    logger.info(
                        "Content extraction failed url=%s (continuing with empty HTML): %s",
                        url[:160],
                        str(e)[:400],
                    )
                    html = ""

                text = _extract_sanitized_text(html)
                text_path.write_text(text, encoding="utf-8")

                return CrawlResult(
                    url=url,
                    final_url=final_url,
                    status="ok",
                    screenshot_path=str(screenshot_path),
                    text_path=str(text_path),
                    redirect_count=redirect_count,
                )
            finally:
                browser.close()
    except Exception as e:
        cat, perm, human = classify_crawl_error(e)
        logger.exception(
            "Crawler browser-level error url=%s category=%s permanent=%s summary=%s",
            url[:160],
            cat,
            perm,
            human,
        )
        return CrawlResult(
            url=url,
            final_url=url,
            status="error",
            screenshot_path=None,
            text_path=None,
            error=str(e),
            error_category=cat,
            permanent_failure=perm,
        )


def crawl_url_with_retries(
    url: str,
    *,
    screenshot_dir: str | Path,
    pages_dir: str | Path,
    timeout_ms: int = 45_000,
    viewport: dict | None = None,
    max_attempts: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int = 750,
    wait_until: str = "domcontentloaded",
    extra_wait_ms: int = 500,
) -> CrawlResult:
    """
    Crawl a URL with up to ``max_attempts`` full browser runs (default **3**).

    After each failed attempt (except **permanent** failures such as DNS NXDOMAIN), waits
    ``retry_backoff_ms`` then retries. Permanent failures return immediately.

    ``max_retries`` is deprecated: when ``max_attempts`` is omitted, uses
    ``max_retries + 1`` attempts (legacy config), else **3** attempts.
    """
    if max_attempts is not None:
        n = max(1, int(max_attempts))
    elif max_retries is not None:
        n = max(1, int(max_retries) + 1)
    else:
        n = 3

    last: CrawlResult | None = None
    for attempt in range(n):
        result = crawl_url(
            url,
            screenshot_dir=screenshot_dir,
            pages_dir=pages_dir,
            timeout_ms=timeout_ms,
            viewport=viewport,
            wait_until=wait_until,
            extra_wait_ms=extra_wait_ms,
        )
        last = result
        if result.status == "ok":
            if attempt > 0:
                logger.info(
                    "Crawl succeeded url=%s after %d/%d attempts",
                    url[:140],
                    attempt + 1,
                    n,
                )
            return result
        if result.permanent_failure:
            logger.info(
                "Crawl stopped url=%s after %d/%d attempts reason=permanent_failure category=%s detail=%s",
                url[:140],
                attempt + 1,
                n,
                result.error_category,
                (result.error or "")[:350],
            )
            return result
        if attempt < n - 1:
            logger.info(
                "Crawl attempt %d/%d failed url=%s status=%s category=%s backoff_ms=%d detail=%s",
                attempt + 1,
                n,
                url[:140],
                result.status,
                result.error_category,
                retry_backoff_ms,
                (result.error or "")[:400],
            )
            if retry_backoff_ms > 0:
                time.sleep(retry_backoff_ms / 1000.0)

    if last is not None:
        logger.info(
            "Crawl exhausted url=%s attempts=%d final_status=%s category=%s detail=%s",
            url[:140],
            n,
            last.status,
            last.error_category,
            (last.error or "")[:400],
        )
        return last
    return CrawlResult(
        url=url,
        final_url=url,
        status="error",
        screenshot_path=None,
        text_path=None,
        error="No attempts made",
        error_category="internal",
        permanent_failure=False,
    )
