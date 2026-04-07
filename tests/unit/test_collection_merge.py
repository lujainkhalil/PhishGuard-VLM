"""Tests for crawl manifest merge deduplication."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_pipeline.collection.merge import merge_crawl_record_lists, screenshot_bytes_hash


def _write_png(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    pytest.importorskip("PIL.Image")
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path, format="PNG")


def test_merge_dedupe_url_preserves_first(tmp_path: Path) -> None:
    root = tmp_path
    base = [{"url": "https://a.com", "status": "ok", "label": "phishing"}]
    incoming = [{"url": "https://a.com", "status": "ok", "label": "benign", "source": "new"}]
    out = merge_crawl_record_lists(base, incoming, project_root=root, dedupe_url=True, dedupe_content_hash=False)
    assert len(out) == 1
    assert out[0]["label"] == "phishing"


def test_merge_appends_new_url(tmp_path: Path) -> None:
    root = tmp_path
    base = [{"url": "https://a.com", "status": "ok"}]
    incoming = [{"url": "https://b.com", "status": "ok"}]
    out = merge_crawl_record_lists(base, incoming, project_root=root, dedupe_url=True)
    assert len(out) == 2


def test_merge_content_hash_dedupe(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    root = tmp_path
    shot = root / "data" / "screenshots"
    shot.mkdir(parents=True)
    p1 = shot / "a.png"
    p2 = shot / "b.png"
    _write_png(p1)
    _write_png(p2)
    rel = str(p1.relative_to(root))
    rel2 = str(p2.relative_to(root))
    # same bytes -> same hash
    base = [
        {
            "url": "https://a.com",
            "status": "ok",
            "screenshot_path": rel,
        }
    ]
    incoming = [
        {
            "url": "https://b.com",
            "status": "ok",
            "screenshot_path": rel2,
        }
    ]
    h1 = screenshot_bytes_hash(base[0], root)
    h2 = screenshot_bytes_hash(incoming[0], root)
    assert h1 == h2
    out = merge_crawl_record_lists(
        base,
        incoming,
        project_root=root,
        dedupe_url=True,
        dedupe_content_hash=True,
    )
    assert len(out) == 1


def test_screenshot_hash_none_for_non_ok(tmp_path: Path) -> None:
    row = {"url": "https://x.com", "status": "error", "screenshot_path": "nope.png"}
    assert screenshot_bytes_hash(row, tmp_path) is None
