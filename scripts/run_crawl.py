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
import logging
import sys
import json
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.collection.crawl_batch import execute_crawl_queue, execute_crawl_queue_parallel

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.collection.crawl_batch import execute_crawl_queue
from data_pipeline.collection.manifest_utils import load_manifest_by_url
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
    parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Parallel crawl workers (default 1; use 5 for fast crawl)",
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
    if args.workers > 1:
        execute_crawl_queue_parallel(
            unique,
            prior,
            manifest_path=args.manifest,
            screenshots_dir=args.screenshots_dir,
            pages_dir=args.pages_dir,
            timeout_ms=timeout_ms,
            viewport=viewport,
            max_attempts=max_attempts,
            retry_backoff_ms=retry_backoff_ms,
            batch_size=max(0, args.batch_size),
            resume=resume,
            workers=args.workers,
        )
    else:
        execute_crawl_queue(
            unique,
            prior,
            manifest_path=args.manifest,
            screenshots_dir=args.screenshots_dir,
            pages_dir=args.pages_dir,
            timeout_ms=timeout_ms,
            viewport=viewport,
            max_attempts=max_attempts,
            retry_backoff_ms=retry_backoff_ms,
            batch_size=max(0, args.batch_size),
            resume=resume,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
