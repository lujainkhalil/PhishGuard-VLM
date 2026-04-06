#!/usr/bin/env python3
"""
Build multimodal dataset from crawl manifest: train/validation/test splits with metadata.

By default, materializes cleaned text and resized RGB images under ``data/processed`` (or
``--processed-dir`` / parent of ``--output``), writes Parquet with columns including
``url``, ``text``, ``label``, ``image_path``, ``screenshot_path``, ``text_path``, ``split``.

Reads configs/data.yaml for ratios, quality filters, ``image_size``, and manifest path.

Usage:
    python scripts/run_preprocess.py
    python scripts/run_preprocess.py --crawl-manifest data/crawl_manifest.json --output data/processed/manifest.parquet
    python scripts/run_preprocess.py --crawl-manifest data/crawl_phish.json --merge-manifest data/crawl_benign.json
    python scripts/run_preprocess.py --no-materialize   # index only; keep raw crawl paths
"""

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.preprocessing import build_dataset, save_manifest, save_split_manifests
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


def load_config(config_path: Path) -> dict:
    """Load full data config YAML."""
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load config %s: %s", config_path, e)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset from crawl manifest")
    parser.add_argument(
        "--crawl-manifest",
        type=Path,
        default=Path("data/crawl_manifest.json"),
        help="Primary crawl manifest JSON from run_crawl.py",
    )
    parser.add_argument(
        "--merge-manifest",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Additional crawl JSON to concatenate before filtering (repeatable). "
        "Order: primary first, then each merge file; URL dedupe keeps the first row per URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path (default: from config preprocessing.manifest_path)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data config for ratios and quality filters",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help="(Deprecated: splits are written by default.) Force split file output if combined with defaults",
    )
    parser.add_argument(
        "--no-split-files",
        action="store_true",
        help="Do not write per-split Parquet files under the splits directory",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=("stratified", "stratified_domain", "temporal", "auto"),
        help="Override config: stratified | stratified_domain (default) | temporal | auto",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Directory for train/validation/test.parquet (default: <manifest-parent>/splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Only filter/split; keep raw crawl paths (no processed images/text under data/processed)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory for processed images/text (default: parent of --output)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pre = config.get("preprocessing") or {}
    quality = config.get("quality") or {}
    dedup = config.get("dedup") or {}
    dval = config.get("dataset_validation") or {}
    hn_cfg = config.get("hard_negatives") or {}

    output_path = args.output or Path(pre.get("manifest_path", "data/processed/manifest.parquet"))
    output_path = output_path if output_path.is_absolute() else _project_root / output_path

    if args.processed_dir is not None:
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else _project_root / args.processed_dir
    elif pre.get("processed_dir"):
        pd_cfg = Path(pre["processed_dir"])
        processed_dir = pd_cfg if pd_cfg.is_absolute() else _project_root / pd_cfg
    else:
        processed_dir = output_path.parent

    crawl_manifest = args.crawl_manifest
    if not crawl_manifest.is_absolute():
        crawl_manifest = _project_root / crawl_manifest

    merge_extra: list[Path] = []
    for p in args.merge_manifest:
        merge_extra.append(p if p.is_absolute() else _project_root / p)

    ts_keys = pre.get("split_timestamp_keys")
    if ts_keys is not None and not isinstance(ts_keys, (list, tuple)):
        ts_keys = None

    split_mode = args.split_mode or pre.get("split_mode", "stratified_domain")

    hn_paths: list = []
    if hn_cfg.get("enabled", False):
        raw_paths = hn_cfg.get("crawl_manifest_paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        hn_paths = [Path(p) for p in raw_paths]

    df = build_dataset(
        crawl_manifest,
        additional_crawl_manifest_paths=merge_extra if merge_extra else None,
        min_text_length=quality.get("min_text_length", 50),
        max_redirects=quality.get("max_redirects", 5),
        exclude_http_errors=quality.get("exclude_http_errors", True),
        dedup_by_url=dedup.get("by_url", True),
        dedup_by_normalized_url=dedup.get("by_normalized_url", True),
        train_ratio=pre.get("train_ratio", 0.8),
        val_ratio=pre.get("val_ratio", 0.1),
        test_ratio=pre.get("test_ratio", 0.1),
        split_mode=split_mode,
        split_timestamp_keys=tuple(ts_keys) if ts_keys else None,
        auto_temporal_min_fraction=float(pre.get("auto_temporal_min_fraction", 0.5)),
        seed=args.seed,
        project_root=_project_root,
        processed_dir=processed_dir,
        image_size=int(pre.get("image_size", 336)),
        materialize=not args.no_materialize,
        validate_screenshot_image=quality.get("validate_screenshot_image", True),
        min_screenshot_bytes=int(quality.get("min_screenshot_bytes", 100)),
        min_image_edge_px=int(quality.get("min_image_edge_px", 1)),
        hard_negative_manifest_paths=hn_paths if hn_paths else None,
        hard_negatives_default_category=str(hn_cfg.get("default_category", "general")),
        hard_negatives_force_train_split=bool(hn_cfg.get("force_train_split", True)),
    )

    if df.empty:
        logger.error("No records after preprocessing. Check crawl manifest and filters.")
        return 1

    log_manifest_statistics(df, prefix="After build (pre manifest validation)")

    data_root_for_validation = _project_root / (config.get("paths") or {}).get("data_root", "data")

    if dval.get("enabled", True):
        df, val_report = validate_processed_manifest(
            df,
            data_root_for_validation,
            min_text_length=int(dval.get("min_text_length", quality.get("min_text_length", 50))),
            min_screenshot_bytes=int(dval.get("min_screenshot_bytes", 100)),
            min_image_edge_px=int(dval.get("min_image_edge_px", 1)),
            dedupe_by_url=bool(dval.get("dedupe_by_url", True)),
            dedupe_by_normalized_url=bool(dval.get("dedupe_by_normalized_url", True)),
        )
        log_validation_report(val_report, prefix="Dataset validation")
        if dval.get("write_report_json", True):
            try:
                import json

                rep_path = output_path.parent / f"{output_path.stem}.dataset_validation.json"
                rep_path.write_text(json.dumps(val_report.to_dict(), indent=2), encoding="utf-8")
                logger.info("Wrote validation report: %s", rep_path)
            except Exception as e:
                logger.warning("Could not write validation JSON: %s", e)

    if df.empty:
        logger.error("No records after dataset validation.")
        return 1

    log_manifest_statistics(df, prefix="Final manifest")

    save_manifest(df, output_path)

    write_split_files = not args.no_split_files
    if args.write_splits and args.no_split_files:
        logger.warning("--write-splits ignored because --no-split-files was set")

    if write_split_files:
        if args.splits_dir is not None:
            splits_dir = args.splits_dir if args.splits_dir.is_absolute() else _project_root / args.splits_dir
        else:
            sub = pre.get("splits_subdir", "splits")
            splits_dir = output_path.parent / sub
        paths = save_split_manifests(df, splits_dir)
        if paths:
            logger.info("Wrote %d split file(s) under %s", len(paths), splits_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
