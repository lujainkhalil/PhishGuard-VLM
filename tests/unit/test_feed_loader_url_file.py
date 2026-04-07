"""Tests for load_urls_from_file."""

from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.crawler.feed_loader import load_urls_from_file


def test_load_urls_json(tmp_path: Path) -> None:
    p = tmp_path / "u.json"
    p.write_text(
        json.dumps(
            [
                {"url": "https://a.com", "label": "benign", "source": "test"},
                {"url": "https://b.com"},
            ]
        ),
        encoding="utf-8",
    )
    rows = load_urls_from_file(p)
    assert len(rows) == 2
    assert rows[0][:2] == ("https://a.com", "benign")
    assert rows[1][1] == "phishing"


def test_load_urls_csv(tmp_path: Path) -> None:
    p = tmp_path / "u.csv"
    p.write_text("url,label,source\nhttps://x.com,benign,csvsrc\n", encoding="utf-8")
    rows = load_urls_from_file(p)
    assert rows[0][0] == "https://x.com"
    assert rows[0][1] == "benign"
    assert rows[0][2] == "csvsrc"
