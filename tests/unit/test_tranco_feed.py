"""Tranco CSV / zip parsing (no network)."""

from __future__ import annotations

import io
import zipfile

import pytest

from data_pipeline.feeds.tranco import parse_tranco_csv_from_text, parse_tranco_zip_bytes


def test_parse_tranco_with_header() -> None:
    csv_text = "rank,domain\n1,Example.COM\n2,wikipedia.org\n"
    entries = parse_tranco_csv_from_text(csv_text, min_urls=2, max_urls=10, fetched_at="t0")
    assert len(entries) == 2
    assert entries[0].label == "benign"
    assert entries[0].source == "tranco"
    assert entries[0].url.startswith("https://")
    assert "example.com" in entries[0].url.lower()


def test_parse_tranco_without_header() -> None:
    csv_text = "1,foo.example\n2,bar.example\n"
    entries = parse_tranco_csv_from_text(csv_text, min_urls=2, max_urls=5, fetched_at="t0")
    assert len(entries) == 2


def test_parse_tranco_zip_roundtrip() -> None:
    csv_text = "rank,domain\n1,a.example\n2,b.example\n3,c.example\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("top-1m.csv", csv_text)
    entries = parse_tranco_zip_bytes(buf.getvalue(), min_urls=2, max_urls=10, fetched_at="t1")
    assert len(entries) == 3


def test_parse_tranco_min_warning_logs(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    parse_tranco_csv_from_text("rank,domain\n1,x.example\n", min_urls=5000, max_urls=10_000)
    assert "only 1" in caplog.text and "5000" in caplog.text
