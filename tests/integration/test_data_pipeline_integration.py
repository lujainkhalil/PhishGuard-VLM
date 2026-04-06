"""
Integration-style tests for the data pipeline: split assignment invariants on larger synthetic sets.
"""

from __future__ import annotations

import pytest

from data_pipeline.preprocessing.splits import assign_splits, temporal_split


@pytest.mark.integration
def test_temporal_split_orders_by_time_per_label() -> None:
    records = []
    for i in range(6):
        records.append(
            {
                "id": i,
                "label": "benign",
                "fetched_at": f"2024-01-{i+1:02d}T00:00:00Z",
            }
        )
    out = temporal_split(
        records,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
    )
    assert len(out) == 6
    by_id = {r["id"]: r["split"] for r in out}
    train_ids = {i for i, s in by_id.items() if s == "train"}
    test_ids = {i for i, s in by_id.items() if s == "test"}
    assert train_ids and test_ids
    assert max(train_ids) <= min(test_ids)


@pytest.mark.integration
def test_assign_splits_unknown_mode_falls_back() -> None:
    records = [{"url": "u", "label": "benign", "domain": "d.com"}]
    out = assign_splits(records, mode="not_a_real_mode", seed=0)
    assert len(out) == 1
    assert out[0]["split"] in ("train", "validation", "test")
