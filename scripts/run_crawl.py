#!/usr/bin/env python3
"""
Crawl URLs from feed output (data/raw/feeds), save full-page screenshots and DOM text.

Usage:
    python scripts/run_crawl.py
    python scripts/run_crawl.py --feeds-dir data/raw/feeds --limit 10
    python scripts/run_crawl.py --config configs/data.yaml
    python scripts/run_crawl.py --batch-size 25 --resume

Reads crawl timeout, viewport, max_attempts / max_retries, retry_backoff_ms from configs/data.yaml.

With resume (default), URLs with status "ok" are skipped, and URLs marked ``permanent_failure``
(e.g. DNS NXDOMAIN) are skipped without recrawling.
Results are written to the manifest every --batch-size crawl attempts (0 = only at end).
Per-URL failures are recorded and do not stop the pipeline.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.crawler import crawl_url_with_retries
from data_pipeline.crawler.feed_loader import load_urls_from_feeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_manifest_by_url(manifest_path: Path) -> dict[str, dict]:
    """Load manifest JSON array into url -> record (last wins if duplicates)."""
    if not manifest_path.is_file():
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Could not read manifest %s: %s", manifest_path, e)
        return {}
    if not isinstance(data, list):
        return {}
    out: dict[str, dict] = {}
    for row in data:
        if isinstance(row, dict) and row.get("url"):
            out[str(row["url"])] = row
    return out


def write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def load_crawl_config(config_path: Path) -> dict:
    """Load crawl section from data config YAML."""
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("crawl") or {}
    except Exception as e:
        logger.warning("Could not load config %s: %s", config_path, e)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl feed URLs with Playwright")
    parser.add_argument(
        "--feeds-dir",
        type=Path,
        default=Path("data/raw/feeds"),
        help="Directory containing feed JSON/CSV files",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("data/screenshots"),
        help="Output directory for full-page screenshots",
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=Path("data/pages"),
        help="Output directory for extracted DOM text",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data config YAML for timeout, viewport, max_retries",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max URLs to crawl (0 = no limit)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/crawl_manifest.json"),
        help="Output path for crawl manifest (url, status, paths, etc.)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Save manifest after this many crawl attempts (0 = save only at end)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Crawl every URL even if it already succeeded in the manifest",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="Override config crawl.timeout_ms",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Override config crawl.max_attempts (full browser runs per URL, default 3)",
    )
    parser.add_argument(
        "--retry-backoff-ms",
        type=int,
        default=None,
        help="Override config crawl.retry_backoff_ms",
    )
    args = parser.parse_args()

    config = load_crawl_config(args.config)
    timeout_ms = int(args.timeout_ms if args.timeout_ms is not None else config.get("timeout_ms", 60_000))
    viewport = config.get("viewport") or {"width": 1920, "height": 1080}
    if args.max_attempts is not None:
        max_attempts = max(1, int(args.max_attempts))
    elif config.get("max_attempts") is not None:
        max_attempts = max(1, int(config["max_attempts"]))
    else:
        max_attempts = max(1, int(config.get("max_retries", 2)) + 1)
    retry_backoff_ms = int(
        args.retry_backoff_ms if args.retry_backoff_ms is not None else config.get("retry_backoff_ms", 750)
    )

    entries = load_urls_from_feeds(args.feeds_dir)
    if not entries:
        logger.error("No URLs found in %s. Run scripts/run_feed_fetch.py first.", args.feeds_dir)
        return 1

    # Deduplicate by URL, keeping first (url, label, source, fetched_at)
    seen: set[str] = set()
    unique: list[tuple[str, str, str, str | None]] = []
    for url, label, source, fetched_at in entries:
        if url in seen:
            continue
        seen.add(url)
        unique.append((url, label, source, fetched_at))
    if len(unique) < len(entries):
        logger.info("Deduped to %d unique URLs", len(unique))

    if args.limit > 0:
        unique = unique[: args.limit]
        logger.info("Limited to %d URLs", len(unique))

    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
    args.pages_dir.mkdir(parents=True, exist_ok=True)

    resume = not args.no_resume
    prior = load_manifest_by_url(args.manifest) if resume else {}
    manifest_entries: list[dict] = []
    ok_count = 0
    skipped_ok = 0
    skipped_permanent = 0
    crawls_since_save = 0
    batch_size = max(0, args.batch_size)

    def flush_manifest() -> None:
        write_manifest(args.manifest, manifest_entries)

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
        try:
            result = crawl_url_with_retries(
                url,
                screenshot_dir=args.screenshots_dir,
                pages_dir=args.pages_dir,
                timeout_ms=timeout_ms,
                viewport=viewport,
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
        if batch_size > 0 and crawls_since_save >= batch_size:
            flush_manifest()
            crawls_since_save = 0
            logger.info("Checkpoint: wrote %d manifest rows to %s", len(manifest_entries), args.manifest)

    flush_manifest()
    failed = len(unique) - ok_count
    logger.info(
        "Crawl complete: %d ok, %d not ok, %d skipped (prior ok), %d skipped (permanent failure). Manifest: %s",
        ok_count,
        failed,
        skipped_ok,
        skipped_permanent,
        args.manifest,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
