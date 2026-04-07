#!/usr/bin/env python3
"""
Validate crawl manifest rows using ``configs/data.yaml`` (collection + quality).

Filters out rows failing text length, screenshot size, label allowlist, or non-ok status.
Logs per-rule removal counts and dataset statistics, then writes the filtered JSON.

Optionally runs ``scripts/run_preprocess.py`` to refresh ``data/processed/``.

Usage:
    python scripts/validate_filter_crawl_manifest.py --input data/crawl_manifest.json --output data/crawl_manifest.json
    python scripts/validate_filter_crawl_manifest.py --input data/crawl_manifest.json --output data/crawl_filtered.json --report-json report.json
    python scripts/validate_filter_crawl_manifest.py --run-preprocess
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.collection.crawl_record_filter import (
    filter_crawl_records_for_training,
    load_collection_filter_config,
)
from data_pipeline.collection.manifest_utils import load_manifest_list, write_manifest
from data_pipeline.collection.stats import compute_crawl_statistics, log_crawl_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _rel_to_root(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def load_full_config(config_path: Path) -> dict:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter crawl manifest by data.yaml rules + log stats")
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--input", type=Path, default=Path("data/crawl_manifest.json"), help="Input crawl JSON")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output crawl JSON (default: overwrite --input)",
    )
    ap.add_argument("--report-json", type=Path, default=None, help="Write filter report + stats JSON")
    ap.add_argument(
        "--run-preprocess",
        action="store_true",
        help="After write, run scripts/run_preprocess.py on --output manifest",
    )
    ap.add_argument(
        "--preprocess-output",
        type=Path,
        default=None,
        help="Override --output for run_preprocess (default: config preprocessing.manifest_path)",
    )
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else _project_root / args.config
    config = load_full_config(cfg_path)
    fcfg = load_collection_filter_config(config)

    in_path = args.input if args.input.is_absolute() else _project_root / args.input
    out_path = args.output
    if out_path is None:
        out_path = in_path
    else:
        out_path = out_path if out_path.is_absolute() else _project_root / out_path

    records = load_manifest_list(in_path)
    if not records:
        logger.error("No records in %s", in_path)
        return 1

    log_crawl_statistics("before_filter", records, _project_root)
    kept, report = filter_crawl_records_for_training(
        records,
        _project_root,
        min_text_length=fcfg["min_text_length"],
        min_screenshot_bytes=fcfg["min_screenshot_bytes"],
        min_image_edge_px=fcfg["min_image_edge_px"],
        allowed_labels=fcfg["allowed_labels"],
        require_ok_status=fcfg["require_ok_status"],
        validate_screenshot_image=fcfg["validate_screenshot_image"],
    )
    log_crawl_statistics("after_filter", kept, _project_root)

    write_manifest(out_path, kept)
    logger.info("Wrote %d rows to %s", len(kept), out_path)

    if args.report_json:
        rp = args.report_json if args.report_json.is_absolute() else _project_root / args.report_json
        rp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "filter_report": report.to_dict(),
            "stats_before": compute_crawl_statistics(records, _project_root),
            "stats_after": compute_crawl_statistics(kept, _project_root),
        }
        rp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote report %s", rp)

    if args.run_preprocess:
        pre = config.get("preprocessing") or {}
        manifest_parquet = args.preprocess_output
        if manifest_parquet is None:
            mp = Path(pre.get("manifest_path", "data/processed/manifest.parquet"))
            manifest_parquet = mp if mp.is_absolute() else _project_root / mp
        else:
            manifest_parquet = (
                args.preprocess_output
                if args.preprocess_output.is_absolute()
                else _project_root / args.preprocess_output
            )

        cmd = [
            sys.executable,
            str(_project_root / "scripts/run_preprocess.py"),
            "--crawl-manifest",
            _rel_to_root(out_path, _project_root),
            "--output",
            _rel_to_root(manifest_parquet, _project_root),
            "--config",
            _rel_to_root(cfg_path, _project_root),
        ]
        logger.info("Running preprocess: %s", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(_project_root))
        if r.returncode != 0:
            logger.error("run_preprocess.py failed with code %s", r.returncode)
            return r.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
