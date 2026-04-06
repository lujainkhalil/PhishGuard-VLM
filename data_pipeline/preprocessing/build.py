"""
Build a multimodal dataset from crawl manifest: screenshots, text, domain metadata, labels.

Applies quality filters, deduplication, stratified splits, and optional materialization
(processed images + cleaned text under data/processed).
"""

import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from data_pipeline.feeds.utils import normalize_url

from .image_processing import prepare_image
from .splits import assign_splits
from .text_processing import clean_and_normalize_text, extract_visible_text_from_html
from .hard_negatives import (
    force_hard_negatives_train_split,
    load_hard_negative_manifest_paths,
    merge_hard_negative_crawls,
)
from .validation import screenshot_file_is_valid

logger = logging.getLogger(__name__)


def _domain_from_url(url: str) -> str:
    """Extract domain (netloc) from URL for metadata."""
    if not url or not isinstance(url, str):
        return ""
    try:
        if "://" not in url:
            url = "http://" + url
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def asset_stem_for_url(url: str) -> str:
    """Stable 16-char hex id for outputs (matches crawler hash convention)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def resolve_project_path(path: str | Path, project_root: Path) -> Path:
    """Resolve crawl artifact path relative to project root or pass through absolute paths."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def load_crawl_manifest(path: str | Path) -> list[dict]:
    """Load crawl manifest JSON; return list of records."""
    path = Path(path)
    if not path.exists():
        logger.warning("Crawl manifest not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("Crawl manifest expected list, got %s", type(data))
        return []
    return data


def load_crawl_manifests_concat(
    paths: list[str | Path],
    *,
    project_root: Path | None = None,
) -> list[dict]:
    """
    Load several crawl JSON manifests and concatenate (primary order preserved).

    Paths may be relative to ``project_root``. Duplicate URLs are **not** removed here;
    :func:`deduplicate_by_url` / :func:`deduplicate_by_normalized_url` run later in
    :func:`build_dataset`. First occurrence wins on dedupe.
    """
    root = project_root or Path.cwd()
    out: list[dict] = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        batch = load_crawl_manifest(path)
        if batch:
            logger.info("Loaded %d crawl rows from %s", len(batch), path)
        out.extend(batch)
    return out


def apply_quality_filters(
    records: list[dict],
    *,
    min_text_length: int = 50,
    max_redirects: int = 5,
    exclude_http_errors: bool = True,
    text_path_key: str = "text_path",
    screenshot_path_key: str = "screenshot_path",
    project_root: Path | None = None,
    validate_screenshot_image: bool = True,
    min_screenshot_bytes: int = 100,
    min_image_edge_px: int = 1,
) -> list[dict]:
    """
    Filter to successful crawls and apply quality rules.

    - Keeps only status == "ok" if exclude_http_errors.
    - Drops rows with redirect_count > max_redirects.
    - Requires screenshot file to exist (multimodal training).
    - Drops rows where extracted text length < min_text_length (reads file).
    """
    out: list[dict] = []
    for r in records:
        if exclude_http_errors and r.get("status") != "ok":
            continue
        if r.get("redirect_count", 0) > max_redirects:
            continue
        shot = r.get(screenshot_path_key)
        if not shot:
            continue
        shot_path = Path(shot) if project_root is None else resolve_project_path(shot, project_root)
        if not shot_path.is_file():
            continue
        if validate_screenshot_image and not screenshot_file_is_valid(
            shot_path, min_bytes=min_screenshot_bytes, min_edge_px=min_image_edge_px
        ):
            continue
        text_path = r.get(text_path_key)
        if text_path:
            tp = Path(text_path) if project_root is None else resolve_project_path(text_path, project_root)
            if not tp.is_file():
                if min_text_length > 0:
                    continue
            else:
                try:
                    text_len = len(tp.read_text(encoding="utf-8", errors="ignore").strip())
                    if text_len < min_text_length:
                        continue
                except Exception:
                    continue
        elif min_text_length > 0:
            continue
        out.append(r)
    return out


def deduplicate_by_url(records: list[dict], url_key: str = "url") -> list[dict]:
    """Keep first occurrence of each URL."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        u = r.get(url_key) or ""
        if u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out


def deduplicate_by_normalized_url(records: list[dict], url_key: str = "url") -> list[dict]:
    """Drop rows whose normalized URL was already seen (reduces http/https duplicate leakage)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        raw = (r.get(url_key) or "").strip()
        nu = normalize_url(raw) if raw else None
        key = nu if nu else raw
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def materialize_processed_records(
    records: list[dict],
    *,
    project_root: Path,
    processed_dir: Path,
    image_size: int,
    data_subdir: str = "data",
) -> list[dict]:
    """
    Write resized RGB images and cleaned text; set text, image_path, text_path, screenshot_path.

    Paths in returned rows are relative to the project ``data/`` directory
    (e.g. ``processed/images/{stem}.png``) so training can use ``data_root=<project>/data``.
    """
    images_dir = processed_dir / "images"
    text_out_dir = processed_dir / "text"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_out_dir.mkdir(parents=True, exist_ok=True)
    data_base = (project_root / data_subdir).resolve()
    try:
        processed_part = processed_dir.resolve().relative_to(data_base)
    except ValueError:
        processed_part = Path(processed_dir.name)

    out: list[dict] = []
    for r in records:
        url = (r.get("url") or "").strip()
        stem = asset_stem_for_url(url) if url else None
        if not stem:
            logger.warning("Skipping record with empty url")
            continue
        src_img = resolve_project_path(r["screenshot_path"], project_root)
        if not src_img.is_file():
            logger.warning("Skip missing screenshot: %s", src_img)
            continue

        html_path = r.get("html_path")
        raw_text = ""
        try:
            if html_path:
                hp = resolve_project_path(html_path, project_root)
                if hp.is_file():
                    raw_text = extract_visible_text_from_html(
                        hp.read_text(encoding="utf-8", errors="ignore")
                    )
            if not raw_text and r.get("text_path"):
                tp = resolve_project_path(r["text_path"], project_root)
                if tp.is_file():
                    raw_text = clean_and_normalize_text(
                        tp.read_text(encoding="utf-8", errors="ignore")
                    )
        except Exception as e:
            logger.warning("Text read failed for %s: %s", url[:60], e)
            continue

        if len(raw_text.strip()) < 1:
            logger.warning("Skip empty text after cleaning: %s", url[:60])
            continue

        dst_img = images_dir / f"{stem}.png"
        dst_txt = text_out_dir / f"{stem}.txt"
        try:
            prepare_image(src_img, dst_img, image_size)
        except Exception as e:
            logger.warning("Image prepare failed for %s: %s", url[:60], e)
            continue

        try:
            dst_txt.write_text(raw_text, encoding="utf-8")
        except Exception as e:
            logger.warning("Text write failed for %s: %s", url[:60], e)
            continue

        img_rel_s = str(processed_part / "images" / f"{stem}.png").replace("\\", "/")
        txt_rel_s = str(processed_part / "text" / f"{stem}.txt").replace("\\", "/")

        rec = dict(r)
        rec["text"] = raw_text
        rec["image_path"] = img_rel_s
        rec["screenshot_path"] = img_rel_s
        rec["text_path"] = txt_rel_s
        if "html_path" in rec:
            del rec["html_path"]
        out.append(rec)

    if len(out) < len(records):
        logger.info("Materialized %d / %d records (skipped failures)", len(out), len(records))
    else:
        logger.info("Materialized %d processed records", len(out))
    return out


def build_dataset(
    crawl_manifest_path: str | Path,
    *,
    additional_crawl_manifest_paths: list[str | Path] | None = None,
    min_text_length: int = 50,
    max_redirects: int = 5,
    exclude_http_errors: bool = True,
    dedup_by_url: bool = True,
    dedup_by_normalized_url: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_mode: str = "stratified_domain",
    split_timestamp_keys: tuple[str, ...] | list[str] | None = None,
    auto_temporal_min_fraction: float = 0.5,
    seed: int = 42,
    project_root: Path | None = None,
    processed_dir: Path | None = None,
    image_size: int = 336,
    materialize: bool = True,
    validate_screenshot_image: bool = True,
    min_screenshot_bytes: int = 100,
    min_image_edge_px: int = 1,
    hard_negative_manifest_paths: list[str | Path] | None = None,
    hard_negatives_default_category: str = "general",
    hard_negatives_force_train_split: bool = True,
) -> pd.DataFrame:
    """
    Load crawl manifest, filter, dedupe, add domain metadata, assign splits, optionally materialize.

    Splitting uses ``split_mode``: ``stratified_domain`` (default) keeps each domain in a single
    split; ``temporal`` orders by timestamp per class; ``auto`` chooses temporal when enough
    rows have ``crawled_at`` / ``fetched_at`` / etc.

    When ``materialize`` is True (default), writes under ``processed_dir`` and sets
    ``text``, ``image_path``, and paths relative to ``data/`` for the training loader.

    Returns a DataFrame with columns including: url, text, label, image_path, screenshot_path,
    text_path, final_url, split, source, domain, redirect_count, hard_negative_category
    (for benign phishing-like pages), and optional timestamp fields.

    Optional ``hard_negative_manifest_paths``: additional crawl JSON list(s) merged after the main
    crawl (same schema). Rows are forced ``label=benign``, ``source=hard_negative``, tagged with
    ``hard_negative_category``. When ``hard_negatives_force_train_split`` is True, those rows are
    assigned to the train split only.

    Optional ``additional_crawl_manifest_paths``: more crawl JSON files (e.g. benign Tranco crawl +
    phishing OpenPhish crawl) concatenated **before** quality filters. Order is primary manifest
    first, then each additional file in order; URL dedupe keeps the **first** row per URL.
    """
    root = project_root or Path.cwd()
    extra = list(additional_crawl_manifest_paths or [])
    if extra:
        paths = [crawl_manifest_path, *extra]
        records = load_crawl_manifests_concat(paths, project_root=root)
    else:
        records = load_crawl_manifest(
            crawl_manifest_path
            if Path(crawl_manifest_path).is_absolute()
            else root / crawl_manifest_path
        )
    if not records:
        logger.warning("No records in crawl manifest(s)")
        return pd.DataFrame()

    n_before = len(records)
    records = apply_quality_filters(
        records,
        min_text_length=min_text_length,
        max_redirects=max_redirects,
        exclude_http_errors=exclude_http_errors,
        project_root=root,
        validate_screenshot_image=validate_screenshot_image,
        min_screenshot_bytes=min_screenshot_bytes,
        min_image_edge_px=min_image_edge_px,
    )
    n_after_filter = len(records)
    if n_after_filter < n_before:
        logger.info("Quality filters: %d -> %d records", n_before, n_after_filter)

    n_after_dedup_url = len(records)
    if dedup_by_url:
        records = deduplicate_by_url(records)
        if len(records) < n_after_dedup_url:
            logger.info("Dedup by URL: %d -> %d records", n_after_dedup_url, len(records))

    n_before_norm = len(records)
    if dedup_by_normalized_url:
        records = deduplicate_by_normalized_url(records)
        if len(records) < n_before_norm:
            logger.info("Dedup by normalized URL: %d -> %d records", n_before_norm, len(records))

    hn_paths = hard_negative_manifest_paths or []
    if hn_paths:
        extra_groups = load_hard_negative_manifest_paths(hn_paths, project_root=root)
        filtered_extras: list[list[dict]] = []
        for g in extra_groups:
            filtered_extras.append(
                apply_quality_filters(
                    g,
                    min_text_length=min_text_length,
                    max_redirects=max_redirects,
                    exclude_http_errors=exclude_http_errors,
                    project_root=root,
                    validate_screenshot_image=validate_screenshot_image,
                    min_screenshot_bytes=min_screenshot_bytes,
                    min_image_edge_px=min_image_edge_px,
                )
            )
        records = merge_hard_negative_crawls(
            records,
            filtered_extras,
            default_category=hard_negatives_default_category,
        )

    for r in records:
        r["domain"] = _domain_from_url(r.get("final_url") or r.get("url") or "")

    records = assign_splits(
        records,
        mode=split_mode,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        timestamp_keys=split_timestamp_keys,
        seed=seed,
        auto_temporal_min_fraction=auto_temporal_min_fraction,
    )

    if hard_negatives_force_train_split:
        force_hard_negatives_train_split(records)

    if materialize:
        out_proc = processed_dir or (root / "data" / "processed")
        out_proc = Path(out_proc)
        records = materialize_processed_records(
            records,
            project_root=root,
            processed_dir=out_proc,
            image_size=image_size,
        )
        if not records:
            return pd.DataFrame()

    df = pd.DataFrame(records)
    preferred = [
        "url",
        "text",
        "label",
        "image_path",
        "screenshot_path",
        "text_path",
        "final_url",
        "split",
        "source",
        "hard_negative_category",
        "domain",
        "redirect_count",
    ]
    ordered = [c for c in preferred if c in df.columns]
    extra = [c for c in df.columns if c not in preferred]
    if ordered or extra:
        df = df[ordered + extra]
    return df


def save_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Write dataset manifest to Parquet (or CSV if path ends with .csv)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    logger.info("Wrote manifest: %s (%d rows)", path, len(df))
    return path


def save_split_manifests(df: pd.DataFrame, splits_dir: str | Path) -> list[Path]:
    """
    Write train.parquet, validation.parquet, and test.parquet under ``splits_dir``.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split_name in ("train", "validation", "test"):
        subset = df[df["split"] == split_name]
        if subset.empty:
            continue
        p = splits_dir / f"{split_name}.parquet"
        save_manifest(subset, p)
        written.append(p)
    return written
