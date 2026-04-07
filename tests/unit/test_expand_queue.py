"""Tests for expansion work queue ordering."""

from __future__ import annotations

from data_pipeline.collection.expand_queue import build_expansion_work_queue


def test_queue_preserves_manifest_then_new() -> None:
    existing = [
        {"url": "https://first.com", "label": "phishing", "source": "a"},
        {"url": "https://second.com", "label": "benign", "source": "b"},
    ]
    new = [("https://third.com", "phishing", "csv", None)]
    q = build_expansion_work_queue(existing, new)
    assert [t[0] for t in q] == ["https://first.com", "https://second.com", "https://third.com"]


def test_queue_skips_duplicate_new_url() -> None:
    existing = [{"url": "https://a.com", "label": "phishing", "source": "m"}]
    new = [("https://a.com", "benign", "x", None), ("https://b.com", "benign", "y", None)]
    q = build_expansion_work_queue(existing, new)
    assert len(q) == 2
    assert q[1][0] == "https://b.com"
