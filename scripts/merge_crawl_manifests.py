#!/usr/bin/env python3
"""
Merge multiple crawl manifest JSON files (same schema as run_crawl.py) into one list.

By default concatenates in order (no dedupe). With ``--dedupe-url`` / ``--dedupe-content-hash``,
preserves **first** file's rows then appends only novel rows (see ``data_pipeline.collection.merge``).

Usage:
    python scripts/merge_crawl_manifests.py -o data/crawl_merged.json \\
        data/crawl_phishing.json data/crawl_benign.json
    python scripts/merge_crawl_manifests.py -o data/crawl_manifest.json --dedupe-url \\
        data/crawl_manifest.json data/new_batch.json
    python scripts/run_preprocess.py --crawl-manifest data/crawl_merged.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.collection.merge import merge_multiple_manifest_files
from data_pipeline.preprocessing.build import load_crawl_manifests_concat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge crawl manifest JSON arrays")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSON path (merged array)",
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input crawl manifest JSON files (order preserved)",
    )
    ap.add_argument(
        "--dedupe-url",
        action="store_true",
        help="Skip later rows whose URL already appeared (first file wins within each stage)",
    )
    ap.add_argument(
        "--dedupe-content-hash",
        action="store_true",
        help="Also skip rows whose screenshot bytes hash matches an earlier kept row",
    )
    args = ap.parse_args()

    out = args.output if args.output.is_absolute() else _project_root / args.output
    paths = [p if p.is_absolute() else _project_root / p for p in args.inputs]
    if args.dedupe_url or args.dedupe_content_hash:
        merged = merge_multiple_manifest_files(
            paths,
            project_root=_project_root,
            dedupe_url=args.dedupe_url,
            dedupe_content_hash=args.dedupe_content_hash,
        )
    else:
        merged = load_crawl_manifests_concat(paths, project_root=_project_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d rows to %s", len(merged), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
