#!/usr/bin/env python3
"""
Validate an existing manifest (Parquet/CSV): drop bad rows, log statistics, optionally overwrite.

Uses ``configs/data.yaml`` ``dataset_validation`` block by default.

Usage:
    python scripts/validate_dataset.py --manifest data/processed/manifest.parquet
    python scripts/validate_dataset.py --manifest data/processed/manifest.csv --dry-run
    python scripts/validate_dataset.py --manifest in.parquet --output out.parquet
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

from data_pipeline.preprocessing import save_manifest
from models.training import load_manifest
from data_pipeline.preprocessing.validation import (
    log_manifest_statistics,
    log_validation_report,
    validate_processed_manifest,
)

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
    ap = argparse.ArgumentParser(description="Validate processed dataset manifest")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None, help="Write cleaned manifest (default: overwrite input)")
    ap.add_argument("--dry-run", action="store_true", help="Log only; do not write manifest")
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config if args.config.is_absolute() else _project_root / args.config)
    dval = cfg.get("dataset_validation") or {}
    quality = cfg.get("quality") or {}

    mp = args.manifest if args.manifest.is_absolute() else _project_root / args.manifest
    df = load_manifest(mp)
    if df.empty:
        logger.error("Manifest empty or missing: %s", mp)
        return 1

    data_root = args.data_root or _project_root / (cfg.get("paths") or {}).get("data_root", "data")

    log_manifest_statistics(df, prefix="Before validation")

    df_out, rep = validate_processed_manifest(
        df,
        data_root if data_root.is_dir() else _project_root,
        min_text_length=int(dval.get("min_text_length", quality.get("min_text_length", 50))),
        min_screenshot_bytes=int(dval.get("min_screenshot_bytes", 100)),
        min_image_edge_px=int(dval.get("min_image_edge_px", 1)),
        dedupe_by_url=bool(dval.get("dedupe_by_url", True)),
        dedupe_by_normalized_url=bool(dval.get("dedupe_by_normalized_url", True)),
    )
    log_validation_report(rep, prefix="Dataset validation")

    log_manifest_statistics(df_out, prefix="After validation")

    rep_path = mp.parent / f"{mp.stem}.dataset_validation.json"
    rep_path.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    logger.info("Wrote report: %s", rep_path)

    if args.dry_run:
        logger.info("Dry run: not writing manifest")
        return 0

    out = args.output or mp
    out = out if out.is_absolute() else _project_root / out
    save_manifest(df_out, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
