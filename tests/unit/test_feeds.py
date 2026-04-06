"""
Unit tests for data_pipeline.feeds (URL normalization, dedupe, FeedEntry).
"""

import tempfile
from pathlib import Path

import pytest

from data_pipeline.feeds import (
    FeedEntry,
    deduplicate_entries,
    normalize_url,
    write_entries_csv,
    write_entries_json,
)


class TestNormalizeUrl:
    def test_basic(self) -> None:
        assert normalize_url("https://example.com/path") == "https://example.com/path"
        assert normalize_url("http://example.com") == "http://example.com/"

    def test_strip_and_lowercase_host(self) -> None:
        assert normalize_url("  http://EXAMPLE.COM:80/  ") == "http://example.com/"

    def test_default_port_removed(self) -> None:
        assert normalize_url("http://example.com:80/") == "http://example.com/"
        assert normalize_url("https://example.com:443/") == "https://example.com/"

    def test_fragment_removed(self) -> None:
        assert normalize_url("http://example.com/path#section") == "http://example.com/path"

    def test_no_scheme_adds_http(self) -> None:
        assert normalize_url("example.com/path") == "http://example.com/path"

    def test_empty_invalid_return_none(self) -> None:
        assert normalize_url("") is None
        assert normalize_url("   ") is None
        assert normalize_url("://") is None


class TestFeedEntry:
    def test_to_dict(self) -> None:
        e = FeedEntry(
            url="https://x.com",
            label="phishing",
            source="openphish",
            fetched_at="2025-01-01T00:00:00Z",
        )
        d = e.to_dict()
        assert d["url"] == "https://x.com"
        assert d["label"] == "phishing"
        assert d["source"] == "openphish"
        assert d["fetched_at"] == "2025-01-01T00:00:00Z"

    def test_to_dict_with_extra(self) -> None:
        e = FeedEntry(
            url="https://x.com",
            source="phishtank",
            fetched_at="2025-01-01",
            extra={"phish_id": 123, "target": "Bank"},
        )
        d = e.to_dict()
        assert d["url"] == "https://x.com"
        assert d["phish_id"] == 123
        assert d["target"] == "Bank"


class TestDeduplicateEntries:
    def test_removes_duplicate_urls(self) -> None:
        e1 = FeedEntry("https://a.com", source="x", fetched_at="1")
        e2 = FeedEntry("https://a.com", source="y", fetched_at="2")
        e3 = FeedEntry("https://b.com", source="x", fetched_at="1")
        out = deduplicate_entries([e1, e2, e3])
        assert len(out) == 2
        assert out[0].url == "https://a.com"
        assert out[1].url == "https://b.com"


class TestWriteEntries:
    def test_write_json(self) -> None:
        entries = [
            FeedEntry("https://a.com", source="openphish", fetched_at="2025-01-01"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            write_entries_json(entries, path)
            assert path.exists()
            import json
            data = json.loads(path.read_text())
            assert len(data) == 1
            assert data[0]["url"] == "https://a.com"

    def test_write_csv(self) -> None:
        entries = [
            FeedEntry("https://a.com", source="openphish", fetched_at="2025-01-01"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            write_entries_csv(entries, path)
            assert path.exists()
            text = path.read_text()
            assert "url,label,source,fetched_at" in text
            assert "https://a.com" in text

    def test_write_csv_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            write_entries_csv([], path)
            assert path.exists()
            assert "url,label,source,fetched_at" in path.read_text()
