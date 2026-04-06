"""Unit tests: split helpers (data pipeline)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data_pipeline.preprocessing.splits import (
    assign_splits,
    fraction_with_timestamps,
    parse_record_timestamp,
    stratified_split,
    stratified_domain_split,
)


class TestParseRecordTimestamp:
    def test_iso_z_suffix(self) -> None:
        r = {"fetched_at": "2024-06-01T12:00:00Z"}
        t = parse_record_timestamp(r)
        assert t is not None
        assert t.tzinfo is not None

    def test_datetime_object(self) -> None:
        dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert parse_record_timestamp({"crawled_at": dt}) == dt

    def test_missing_returns_none(self) -> None:
        assert parse_record_timestamp({}) is None


class TestFractionWithTimestamps:
    def test_all_missing(self) -> None:
        assert fraction_with_timestamps([{}, {}]) == 0.0

    def test_half(self) -> None:
        rows = [{"fetched_at": "2024-01-01T00:00:00Z"}, {}]
        assert fraction_with_timestamps(rows) == 0.5


class TestStratifiedSplit:
    def test_all_rows_have_split(self) -> None:
        records = [
            {"url": "a", "label": "benign"},
            {"url": "b", "label": "benign"},
            {"url": "c", "label": "phishing"},
            {"url": "d", "label": "phishing"},
        ]
        out = stratified_split(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=0)
        assert len(out) == 4
        splits = {r["split"] for r in out}
        assert splits <= {"train", "validation", "test"}


class TestStratifiedDomainSplit:
    def test_same_domain_same_split(self) -> None:
        records = [
            {"url": "https://x.com/a", "domain": "x.com", "label": "benign"},
            {"url": "https://x.com/b", "domain": "x.com", "label": "benign"},
            {"url": "https://y.com/", "domain": "y.com", "label": "phishing"},
        ]
        out = stratified_domain_split(
            records, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=42
        )
        by_domain: dict[str, set[str]] = {}
        for r in out:
            d = r["domain"]
            by_domain.setdefault(d, set()).add(r["split"])
        for sset in by_domain.values():
            assert len(sset) == 1


class TestAssignSplits:
    def test_auto_uses_temporal_when_enough_timestamps(self) -> None:
        records = [
            {"url": "a", "label": "benign", "domain": "a.com", "fetched_at": "2020-01-01T00:00:00Z"},
            {"url": "b", "label": "benign", "domain": "b.com", "fetched_at": "2021-01-01T00:00:00Z"},
            {"url": "c", "label": "phishing", "domain": "c.com", "fetched_at": "2022-01-01T00:00:00Z"},
        ]
        out = assign_splits(
            records,
            mode="auto",
            train_ratio=0.34,
            val_ratio=0.33,
            test_ratio=0.33,
            auto_temporal_min_fraction=0.5,
            seed=1,
        )
        assert len(out) == 3
        assert all("split" in r for r in out)

    def test_stratified_mode(self) -> None:
        records = [{"url": str(i), "label": "benign"} for i in range(10)]
        out = assign_splits(records, mode="stratified", seed=0)
        assert len(out) == 10


class TestSplitRatiosValidation:
    def test_invalid_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="1.0"):
            stratified_split(
                [{"x": 1}],
                train_ratio=0.5,
                val_ratio=0.5,
                test_ratio=0.5,
            )
