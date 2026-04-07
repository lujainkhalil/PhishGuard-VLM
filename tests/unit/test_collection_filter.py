"""Tests for crawl record filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_pipeline.collection.crawl_record_filter import (
    filter_crawl_records_for_training,
    load_collection_filter_config,
)


def _valid_png(path: Path) -> None:
    pytest.importorskip("PIL.Image")
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path, format="PNG")


def test_filter_drops_short_text(tmp_path: Path) -> None:
    root = tmp_path
    pages = root / "data" / "pages"
    shots = root / "data" / "screenshots"
    pages.mkdir(parents=True)
    shots.mkdir(parents=True)
    txt = pages / "t.txt"
    txt.write_text("short", encoding="utf-8")
    png = shots / "s.png"
    _valid_png(png)
    records = [
        {
            "url": "https://a.com",
            "status": "ok",
            "label": "phishing",
            "text_path": str(txt.relative_to(root)),
            "screenshot_path": str(png.relative_to(root)),
        }
    ]
    kept, rep = filter_crawl_records_for_training(
        records,
        root,
        min_text_length=50,
        min_screenshot_bytes=10,
        min_image_edge_px=1,
        allowed_labels={"phishing", "benign"},
    )
    assert len(kept) == 0
    assert rep.removed_short_text == 1


def test_filter_keeps_good_row(tmp_path: Path) -> None:
    root = tmp_path
    pages = root / "data" / "pages"
    shots = root / "data" / "screenshots"
    pages.mkdir(parents=True)
    shots.mkdir(parents=True)
    txt = pages / "t.txt"
    txt.write_text("x" * 60, encoding="utf-8")
    png = shots / "s.png"
    _valid_png(png)
    records = [
        {
            "url": "https://a.com",
            "status": "ok",
            "label": "benign",
            "text_path": str(txt.relative_to(root)),
            "screenshot_path": str(png.relative_to(root)),
        }
    ]
    kept, rep = filter_crawl_records_for_training(
        records,
        root,
        min_text_length=50,
        min_screenshot_bytes=10,
        min_image_edge_px=1,
        allowed_labels={"phishing", "benign"},
    )
    assert len(kept) == 1
    assert rep.rows_out == 1


def test_filter_bad_label(tmp_path: Path) -> None:
    root = tmp_path
    pages = root / "data" / "pages"
    shots = root / "data" / "screenshots"
    pages.mkdir(parents=True)
    shots.mkdir(parents=True)
    txt = pages / "t.txt"
    txt.write_text("x" * 60, encoding="utf-8")
    png = shots / "s.png"
    _valid_png(png)
    records = [
        {
            "url": "https://a.com",
            "status": "ok",
            "label": "spam",
            "text_path": str(txt.relative_to(root)),
            "screenshot_path": str(png.relative_to(root)),
        }
    ]
    kept, rep = filter_crawl_records_for_training(
        records,
        root,
        min_text_length=50,
        min_screenshot_bytes=10,
        min_image_edge_px=1,
        allowed_labels={"phishing", "benign"},
    )
    assert len(kept) == 0
    assert rep.removed_bad_label == 1


def test_load_collection_filter_config_defaults() -> None:
    cfg = load_collection_filter_config({"quality": {"min_text_length": 40}, "collection": {}})
    assert cfg["min_text_length"] == 40
    assert "phishing" in cfg["allowed_labels"]
