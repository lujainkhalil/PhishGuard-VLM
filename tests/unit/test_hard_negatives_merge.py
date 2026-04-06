"""Unit tests: hard-negative crawl merge (no filesystem crawl)."""

from __future__ import annotations

from data_pipeline.preprocessing.hard_negatives import (
    force_hard_negatives_train_split,
    merge_hard_negative_crawls,
    normalize_hard_negative_category,
)


class TestNormalizeCategory:
    def test_aliases(self) -> None:
        assert normalize_hard_negative_category("login") == "login_form"
        assert normalize_hard_negative_category("branded_page") == "branded"


class TestMergeHardNegativeCrawls:
    def test_forces_benign_and_tags(self) -> None:
        primary = [
            {
                "url": "https://a.com/",
                "final_url": "https://a.com/",
                "label": "benign",
                "source": "feed",
            }
        ]
        extra = [
            {
                "url": "https://b.com/login",
                "final_url": "https://b.com/login",
                "label": "phishing",
                "hard_negative_category": "login_form",
            }
        ]
        out = merge_hard_negative_crawls(primary, [extra], default_category="general")
        assert len(out) == 2
        hn = next(r for r in out if r["url"].endswith("/login"))
        assert hn["label"] == "benign"
        assert hn["source"] == "hard_negative"
        assert hn["hard_negative_category"] == "login_form"

    def test_skips_duplicate_url(self) -> None:
        primary = [{"url": "https://same.com/x", "final_url": "https://same.com/x", "label": "benign"}]
        extra = [{"url": "https://same.com/x", "final_url": "https://same.com/x", "label": "benign"}]
        out = merge_hard_negative_crawls(primary, [extra], default_category="general")
        assert len(out) == 1


class TestForceTrain:
    def test_sets_split(self) -> None:
        rows = [
            {"url": "https://x.com", "split": "validation", "source": "hard_negative"},
            {"url": "https://y.com", "split": "validation", "source": "feed"},
        ]
        force_hard_negatives_train_split(rows)
        assert rows[0]["split"] == "train"
        assert rows[1]["split"] == "validation"
