#!/usr/bin/env python3
"""
Fetch phishing feeds (OpenPhish, PhishTank) and benign Tranco top list into data/raw/feeds/.

OpenPhish: reads the **full** text feed from each configured URL (no row cap) and merges
unique URLs across mirrors. PhishTank: downloads the full ``online-valid.json`` dump.
Tranco: downloads the official top-1M CSV zip and emits at least ``min_urls`` benign https URLs.

Usage:
    python scripts/run_feed_fetch.py
    python scripts/run_feed_fetch.py --config configs/data.yaml
    python scripts/run_feed_fetch.py --output-dir data/raw/feeds --format json
    python scripts/run_feed_fetch.py --skip-tranco
    python scripts/run_feed_fetch.py --tranco-only

Requires PHISHTANK_API_KEY in the environment for reliable PhishTank access (recommended).
"""

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.feeds import collect_openphish, collect_phishtank, collect_tranco

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch OpenPhish, PhishTank, and Tranco feeds")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/feeds"),
        help="Directory for output files (default: data/raw/feeds)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data config with feeds.* settings",
    )
    parser.add_argument("--openphish-only", action="store_true", help="Fetch only OpenPhish")
    parser.add_argument("--phishtank-only", action="store_true", help="Fetch only PhishTank")
    parser.add_argument("--tranco-only", action="store_true", help="Fetch only Tranco benign list")
    parser.add_argument("--skip-tranco", action="store_true", help="Do not fetch Tranco")
    parser.add_argument("--skip-openphish", action="store_true", help="Do not fetch OpenPhish")
    parser.add_argument("--skip-phishtank", action="store_true", help="Do not fetch PhishTank")
    parser.add_argument(
        "--tranco-min-urls",
        type=int,
        default=None,
        help="Override config: minimum benign URLs from Tranco (default from config, e.g. 10000)",
    )
    parser.add_argument(
        "--tranco-max-urls",
        type=int,
        default=None,
        help="Override config: cap rows from Tranco CSV (after min_urls)",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else _project_root / args.config
    data_cfg = load_yaml(cfg_path)
    feeds_root = data_cfg.get("feeds") or {}
    phish_cfg = feeds_root.get("phishing") or {}
    benign_cfg = feeds_root.get("benign") or {}

    op_cfg = phish_cfg.get("openphish") or {}
    if isinstance(op_cfg, str):
        op_cfg = {}
    pt_cfg = phish_cfg.get("phishtank") or {}
    if isinstance(pt_cfg, str):
        pt_cfg = {}
    tr_cfg = benign_cfg.get("tranco") or {}
    if isinstance(tr_cfg, str):
        tr_cfg = {}

    op_urls = op_cfg.get("feed_urls")
    if isinstance(op_urls, str):
        op_urls = [op_urls]
    op_timeout = int(op_cfg.get("timeout_sec", 120))

    pt_timeout = int(pt_cfg.get("timeout_sec", 360))

    tr_enabled = bool(tr_cfg.get("enabled", True))
    tr_min = int(args.tranco_min_urls if args.tranco_min_urls is not None else tr_cfg.get("min_urls", 10_000))
    tr_max = args.tranco_max_urls if args.tranco_max_urls is not None else tr_cfg.get("max_urls")
    tr_max = int(tr_max) if tr_max is not None else None
    tr_url = str(tr_cfg.get("download_url", "https://tranco-list.eu/top-1m.csv.zip"))
    tr_timeout = int(tr_cfg.get("timeout_sec", 600))

    out_dir = args.output_dir if args.output_dir.is_absolute() else _project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    only_one = sum(
        1 for x in (args.openphish_only, args.phishtank_only, args.tranco_only) if x
    )
    if only_one > 1:
        logger.error("Use at most one of --openphish-only, --phishtank-only, --tranco-only")
        return 2

    do_op = not args.skip_openphish and not args.phishtank_only and not args.tranco_only
    do_pt = not args.skip_phishtank and not args.openphish_only and not args.tranco_only
    do_tr = (
        tr_enabled
        and not args.skip_tranco
        and not args.openphish_only
        and not args.phishtank_only
    )

    if args.openphish_only:
        do_op, do_pt, do_tr = True, False, False
    if args.phishtank_only:
        do_op, do_pt, do_tr = False, True, False
    if args.tranco_only:
        do_op, do_pt, do_tr = False, False, True

    exit_code = 0

    if do_op:
        try:
            openphish_path = out_dir / f"openphish.{fmt}"
            entries = collect_openphish(
                str(openphish_path),
                feed_urls=op_urls if op_urls else None,
                timeout=op_timeout,
                output_format=fmt,
            )
            logger.info("OpenPhish: %d entries written to %s", len(entries), openphish_path)
        except Exception as e:
            logger.exception("OpenPhish fetch failed: %s", e)
            exit_code = 1

    if do_pt:
        try:
            phishtank_path = out_dir / f"phishtank.{fmt}"
            entries = collect_phishtank(
                str(phishtank_path),
                timeout=pt_timeout,
                output_format=fmt,
            )
            logger.info("PhishTank: %d entries written to %s", len(entries), phishtank_path)
        except Exception as e:
            logger.exception("PhishTank fetch failed: %s", e)
            exit_code = 1

    if do_tr:
        try:
            tranco_path = out_dir / f"tranco.{fmt}"
            entries = collect_tranco(
                str(tranco_path),
                download_url=tr_url,
                timeout_sec=tr_timeout,
                min_urls=tr_min,
                max_urls=tr_max,
                output_format=fmt,
            )
            logger.info("Tranco: %d benign entries written to %s", len(entries), tranco_path)
            if len(entries) < tr_min:
                logger.error(
                    "Tranco returned fewer than min_urls (%d < %d). Check network or list URL.",
                    len(entries),
                    tr_min,
                )
                exit_code = 1
        except Exception as e:
            logger.exception("Tranco fetch failed: %s", e)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
