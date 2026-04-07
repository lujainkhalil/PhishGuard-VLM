#!/usr/bin/env python3
"""
Large-scale crawl expansion: existing ``crawl_manifest.json`` + feeds + extra URL files.

Builds a unified work queue (manifest order first, then new URLs), resumes ok/permanent
rows like ``run_crawl.py``, and writes the same JSON schema to ``--manifest``.

Usage:
    python scripts/run_crawl_expand.py --url-file data/imports/more_phish.json
    python scripts/run_crawl_expand.py --feeds-dir data/raw/feeds --url-file extra.csv --manifest data/crawl_manifest.json
    python scripts/run_crawl_expand.py --manifest data/crawl_manifest.json --no-feeds --url-file urls.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.collection.crawl_batch import execute_crawl_queue
from data_pipeline.collection.expand_queue import build_expansion_work_queue
from data_pipeline.collection.manifest_utils import load_manifest_by_url, load_manifest_list
from data_pipeline.collection.stats import log_crawl_statistics
from data_pipeline.crawler.feed_loader import load_urls_from_feeds, load_urls_from_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_crawl_section(config_path: Path) -> dict:
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("crawl") or {}
    except Exception as e:
        logger.warning("Could not load config %s: %s", config_path, e)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand crawl manifest with feeds + URL import files")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data YAML for crawl.timeout_ms, viewport, max_attempts",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/crawl_manifest.json"),
        help="Read/write crawl manifest (same schema as run_crawl.py)",
    )
    parser.add_argument(
        "--feeds-dir",
        type=Path,
        default=Path("data/raw/feeds"),
        help="Feed JSON/CSV directory (skipped if --no-feeds)",
    )
    parser.add_argument(
        "--no-feeds",
        action="store_true",
        help="Do not load URLs from --feeds-dir",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Additional labeled URL list (.json or .csv); repeatable",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("data/screenshots"),
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=Path("data/pages"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max **new** URLs from feeds/files after merge (0 = no limit)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Flush manifest every N crawl attempts (0 = end only)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-crawl every URL (ignore prior ok rows)",
    )
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--retry-backoff-ms", type=int, default=None)
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else _project_root / args.manifest
    existing_list = load_manifest_list(manifest_path)
    log_crawl_statistics("before_expand", existing_list, _project_root)

    new_tuples: list[tuple[str, str, str, str | None]] = []
    if not args.no_feeds:
        fd = args.feeds_dir if args.feeds_dir.is_absolute() else _project_root / args.feeds_dir
        new_tuples.extend(load_urls_from_feeds(fd))
    for uf in args.url_file:
        p = uf if uf.is_absolute() else _project_root / uf
        batch = load_urls_from_file(p)
        logger.info("Loaded %d URLs from %s", len(batch), p)
        new_tuples.extend(batch)

    seen_new: set[str] = set()
    deduped_new: list[tuple[str, str, str, str | None]] = []
    for t in new_tuples:
        if t[0] in seen_new:
            continue
        seen_new.add(t[0])
        deduped_new.append(t)
    new_tuples = deduped_new

    if args.limit > 0:
        existing_urls = {str(r.get("url", "")) for r in existing_list if r.get("url")}
        added = 0
        limited: list[tuple[str, str, str, str | None]] = []
        for t in new_tuples:
            if t[0] in existing_urls:
                continue
            if added >= args.limit:
                break
            limited.append(t)
            added += 1
        new_tuples = limited
        logger.info("Limited to %d new URLs from feeds/files", len(new_tuples))

    work_queue = build_expansion_work_queue(existing_list, new_tuples)
    if not work_queue:
        logger.error("No URLs to crawl. Add feeds, url-file, or populate manifest.")
        return 1

    from collections import Counter

    q_labels = Counter(lab.lower() for _, lab, _, _ in work_queue)
    logger.info("Work queue: %d URLs, label counts: %s", len(work_queue), dict(q_labels))

    cfg = load_crawl_section(args.config if args.config.is_absolute() else _project_root / args.config)
    timeout_ms = int(args.timeout_ms if args.timeout_ms is not None else cfg.get("timeout_ms", 60_000))
    viewport = cfg.get("viewport") or {"width": 1920, "height": 1080}
    if args.max_attempts is not None:
        max_attempts = max(1, int(args.max_attempts))
    elif cfg.get("max_attempts") is not None:
        max_attempts = max(1, int(cfg["max_attempts"]))
    else:
        max_attempts = max(1, int(cfg.get("max_retries", 2)) + 1)
    retry_backoff_ms = int(
        args.retry_backoff_ms if args.retry_backoff_ms is not None else cfg.get("retry_backoff_ms", 750)
    )

    prior = load_manifest_by_url(manifest_path) if not args.no_resume else {}
    resume = not args.no_resume

    shots = args.screenshots_dir if args.screenshots_dir.is_absolute() else _project_root / args.screenshots_dir
    pages = args.pages_dir if args.pages_dir.is_absolute() else _project_root / args.pages_dir

    execute_crawl_queue(
        work_queue,
        prior,
        manifest_path=manifest_path,
        screenshots_dir=shots,
        pages_dir=pages,
        timeout_ms=timeout_ms,
        viewport=viewport,
        max_attempts=max_attempts,
        retry_backoff_ms=retry_backoff_ms,
        batch_size=max(0, args.batch_size),
        resume=resume,
    )

    final_list = load_manifest_list(manifest_path)
    log_crawl_statistics("after_expand", final_list, _project_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
