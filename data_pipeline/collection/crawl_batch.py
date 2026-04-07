"""
Batch crawl execution: shared queue processor for ``run_crawl`` and expansion pipelines.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.crawler import crawl_url_with_retries

from .manifest_utils import write_manifest

logger = logging.getLogger(__name__)


def execute_crawl_queue(
    unique: list[tuple[str, str, str, str | None]],
    prior: dict[str, dict[str, Any]],
    *,
    manifest_path: Path,
    screenshots_dir: Path,
    pages_dir: Path,
    timeout_ms: int,
    viewport: dict[str, int] | None,
    max_attempts: int,
    retry_backoff_ms: int,
    batch_size: int,
    resume: bool = True,
) -> dict[str, int]:
    """
    Crawl each URL in ``unique`` (url, label, source, fetched_at), updating ``prior`` and writing manifest.

    When ``resume`` is True, skips URLs already ``ok`` or ``permanent_failure`` in ``prior``.

    Args:
        unique: Ordered work queue.
        prior: Mutable url -> last manifest row (updated as crawls complete).
        manifest_path: JSON output path.
        screenshots_dir: Screenshot output directory.
        pages_dir: Extracted text output directory.
        timeout_ms: Navigation timeout.
        viewport: Playwright viewport dict.
        max_attempts: Browser attempts per URL.
        retry_backoff_ms: Delay between retries.
        batch_size: Flush manifest every N crawl attempts (0 = end only).
        resume: Use prior to skip completed / permanent-failure URLs.

    Returns:
        Stats: ok_count, skipped_ok, skipped_permanent, crawl_attempts.
    """
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    ok_count = 0
    skipped_ok = 0
    skipped_permanent = 0
    crawls_since_save = 0
    bs = max(0, batch_size)
    crawl_attempts = 0

    def flush_manifest() -> None:
        write_manifest(manifest_path, manifest_entries)

    vp = viewport or {"width": 1920, "height": 1080}

    for i, (url, label, source, fetched_at) in enumerate(unique):
        prev_row = prior.get(url, {}) if resume else {}
        if resume and prev_row.get("status") == "ok":
            row = prev_row
            manifest_entries.append(row)
            skipped_ok += 1
            if row.get("status") == "ok":
                ok_count += 1
            logger.info("[%d/%d] Skip (already ok): %s", i + 1, len(unique), url[:70])
            continue

        if resume and prev_row.get("permanent_failure"):
            row = prev_row
            manifest_entries.append(row)
            skipped_permanent += 1
            logger.info(
                "[%d/%d] Skip (permanent failure, category=%s): %s",
                i + 1,
                len(unique),
                prev_row.get("error_category", "?"),
                url[:70],
            )
            continue

        logger.info(
            "[%d/%d] Crawling %s (timeout_ms=%d max_attempts=%d)",
            i + 1,
            len(unique),
            url[:70],
            timeout_ms,
            max_attempts,
        )
        crawl_attempts += 1
        try:
            result = crawl_url_with_retries(
                url,
                screenshot_dir=screenshots_dir,
                pages_dir=pages_dir,
                timeout_ms=timeout_ms,
                viewport=vp,
                max_attempts=max_attempts,
                retry_backoff_ms=retry_backoff_ms,
                wait_until="domcontentloaded",
                extra_wait_ms=500,
            )
        except Exception as e:
            logger.exception("Unexpected error crawling %s: %s", url[:80], e)
            result = None

        crawled_at = datetime.now(timezone.utc).isoformat()
        if result is None:
            row = {
                "url": url,
                "final_url": url,
                "status": "error",
                "screenshot_path": None,
                "text_path": None,
                "label": label,
                "source": source,
                "error": "unexpected crawler exception",
                "error_category": "internal",
                "permanent_failure": False,
                "redirect_count": 0,
                "crawled_at": crawled_at,
            }
        else:
            row = {
                "url": result.url,
                "final_url": result.final_url,
                "status": result.status,
                "screenshot_path": result.screenshot_path,
                "text_path": result.text_path,
                "label": label,
                "source": source,
                "error": result.error,
                "error_category": result.error_category,
                "permanent_failure": bool(result.permanent_failure),
                "redirect_count": result.redirect_count,
                "crawled_at": crawled_at,
            }
        if fetched_at:
            row["fetched_at"] = fetched_at

        manifest_entries.append(row)
        prior[url] = row
        if row.get("status") == "ok":
            ok_count += 1

        crawls_since_save += 1
        if bs > 0 and crawls_since_save >= bs:
            flush_manifest()
            crawls_since_save = 0
            logger.info("Checkpoint: wrote %d manifest rows to %s", len(manifest_entries), manifest_path)

    flush_manifest()
    failed = len(unique) - ok_count
    logger.info(
        "Crawl complete: %d ok, %d not ok, %d skipped (prior ok), %d skipped (permanent failure). Manifest: %s",
        ok_count,
        failed,
        skipped_ok,
        skipped_permanent,
        manifest_path,
    )
    return {
        "ok_count": ok_count,
        "skipped_ok": skipped_ok,
        "skipped_permanent": skipped_permanent,
        "crawl_attempts": crawl_attempts,
        "queue_size": len(unique),
    }

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def execute_crawl_queue_parallel(
    unique: list[tuple[str, str, str, str | None]],
    prior: dict[str, dict[str, Any]],
    *,
    manifest_path: Path,
    screenshots_dir: Path,
    pages_dir: Path,
    timeout_ms: int,
    viewport: dict[str, int] | None,
    max_attempts: int,
    retry_backoff_ms: int,
    batch_size: int,
    resume: bool = True,
    workers: int = 5,
) -> dict[str, int]:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    vp = viewport or {"width": 1920, "height": 1080}
    lock = threading.Lock()
    manifest_entries: list[dict[str, Any]] = []
    stats = {"ok": 0, "skipped_ok": 0, "skipped_permanent": 0, "attempts": 0}
    completed_count = 0

    # Pre-populate skipped entries
    work_queue = []
    for url, label, source, fetched_at in unique:
        prev = prior.get(url, {}) if resume else {}
        if resume and prev.get("status") == "ok":
            with lock:
                manifest_entries.append(prev)
                stats["skipped_ok"] += 1
                stats["ok"] += 1
            continue
        if resume and prev.get("permanent_failure"):
            with lock:
                manifest_entries.append(prev)
                stats["skipped_permanent"] += 1
            continue
        work_queue.append((url, label, source, fetched_at))

    total = len(work_queue) + stats["skipped_ok"] + stats["skipped_permanent"]
    logger.info("Parallel crawl: %d to crawl, %d skipped (workers=%d)", 
                len(work_queue), stats["skipped_ok"] + stats["skipped_permanent"], workers)

    def crawl_one(item: tuple[str, str, str, str | None]) -> dict[str, Any]:
        url, label, source, fetched_at = item
        result = crawl_url_with_retries(
            url,
            screenshot_dir=screenshots_dir,
            pages_dir=pages_dir,
            timeout_ms=timeout_ms,
            viewport=vp,
            max_attempts=max_attempts,
            retry_backoff_ms=retry_backoff_ms,
        )
        row: dict[str, Any] = {
            "url": url,
            "final_url": result.final_url,
            "status": result.status,
            "label": label,
            "source": source,
            "fetched_at": fetched_at,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "screenshot_path": result.screenshot_path,
            "text_path": result.text_path,
            "redirect_count": result.redirect_count,
            "error": result.error,
            "error_category": result.error_category,
            "permanent_failure": result.permanent_failure,
        }
        return row

    bs = max(1, batch_size)
    crawls_since_save = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(crawl_one, item): item for item in work_queue}
        done_count = stats["skipped_ok"] + stats["skipped_permanent"]

        for future in as_completed(futures):
            done_count += 1
            try:
                row = future.result()
            except Exception as e:
                url, label, source, fetched_at = futures[future]
                logger.error("Unhandled crawl error for %s: %s", url[:80], e)
                row = {
                    "url": url, "final_url": url, "status": "error",
                    "label": label, "source": source, "fetched_at": fetched_at,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "screenshot_path": None, "text_path": None,
                    "redirect_count": 0, "error": str(e),
                    "error_category": "internal", "permanent_failure": False,
                }

            with lock:
                manifest_entries.append(row)
                stats["attempts"] += 1
                if row.get("status") == "ok":
                    stats["ok"] += 1
                crawls_since_save += 1
                should_save = bs > 0 and crawls_since_save >= bs
                if should_save:
                    crawls_since_save = 0

            if should_save:
                with lock:
                    write_manifest(manifest_path, manifest_entries)
                logger.info("Progress [%d/%d] ok=%d", done_count, total, stats["ok"])

    write_manifest(manifest_path, manifest_entries)
    logger.info("Crawl complete: ok=%d / attempted=%d", stats["ok"], stats["attempts"])
    return stats