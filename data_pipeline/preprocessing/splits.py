"""
Train / validation / test assignment with reduced leakage (domain-disjoint, temporal options).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from datetime import datetime
logger = logging.getLogger(__name__)

DEFAULT_TIMESTAMP_KEYS = (
    "crawled_at",
    "fetched_at",
    "crawl_date",
    "timestamp",
    "feed_fetched_at",
)


def parse_record_timestamp(record: dict, keys: tuple[str, ...] | list[str] | None = None) -> datetime | None:
    """
    Parse first available ISO-like timestamp from record values.
    """
    keys = tuple(keys) if keys is not None else DEFAULT_TIMESTAMP_KEYS
    for k in keys:
        raw = record.get(k)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return raw
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            continue
    return None


def fraction_with_timestamps(records: list[dict], keys: tuple[str, ...] | list[str] | None = None) -> float:
    if not records:
        return 0.0
    n = sum(1 for r in records if parse_record_timestamp(r, keys) is not None)
    return n / len(records)


def stratified_split(
    records: list[dict],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    label_key: str = "label",
    seed: int = 42,
) -> list[dict]:
    """
    Random stratified split by label (URLs may share domains across splits — more leakage).
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        lab = r.get(label_key) or "unknown"
        by_label[lab].append(r)

    out: list[dict] = []
    for group in by_label.values():
        rng.shuffle(group)
        n = len(group)
        t_end, v_end = _cut_points(n, train_ratio, val_ratio)
        for i, r in enumerate(group):
            rec = dict(r)
            rec["split"] = _split_name(i, t_end, v_end)
            out.append(rec)
    rng.shuffle(out)
    return out


def stratified_domain_split(
    records: list[dict],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    label_key: str = "label",
    domain_key: str = "domain",
    seed: int = 42,
) -> list[dict]:
    """
    Assign splits by domain: all URLs with the same ``domain`` share one split.

    Stratifies **domains** by majority label so each split gets a similar mix of
    domain-groups per class (reduces same-site train/test leakage).
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        d = (r.get(domain_key) or "").strip().lower() or "_unknown"
        groups[d].append(r)

    group_rows: list[tuple[str, list[dict], str]] = []
    for gid, recs in groups.items():
        lab = _majority_label(recs, label_key)
        group_rows.append((gid, recs, lab))

    by_label: dict[str, list[tuple[str, list[dict], str]]] = defaultdict(list)
    for row in group_rows:
        by_label[row[2]].append(row)

    split_by_group: dict[str, str] = {}
    for lab, gl in by_label.items():
        rng.shuffle(gl)
        n = len(gl)
        t_end, v_end = _cut_points(n, train_ratio, val_ratio)
        for i, (gid, _, _) in enumerate(gl):
            split_by_group[gid] = _split_name(i, t_end, v_end)

    out: list[dict] = []
    for gid, recs, _ in group_rows:
        sp = split_by_group[gid]
        for r in recs:
            rec = dict(r)
            rec["split"] = sp
            out.append(rec)
    rng.shuffle(out)
    return out


def temporal_split(
    records: list[dict],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    label_key: str = "label",
    timestamp_keys: tuple[str, ...] | list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Per-label chronological split: older samples → train, then validation, then test.

    Rows without a parseable timestamp are stratified separately with the same ratios
    and concatenated (logged).
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)
    keys = tuple(timestamp_keys) if timestamp_keys is not None else DEFAULT_TIMESTAMP_KEYS
    by_label: dict[str, list[tuple[datetime, dict]]] = defaultdict(list)
    no_ts: list[dict] = []

    for r in records:
        t = parse_record_timestamp(r, keys)
        if t is None:
            no_ts.append(r)
        else:
            lab = r.get(label_key) or "unknown"
            by_label[lab].append((t, r))

    out: list[dict] = []
    for lab, pairs in by_label.items():
        pairs.sort(key=lambda x: x[0])
        items = [p[1] for p in pairs]
        n = len(items)
        t_end, v_end = _cut_points(n, train_ratio, val_ratio)
        for i, r in enumerate(items):
            rec = dict(r)
            rec["split"] = _split_name(i, t_end, v_end)
            out.append(rec)

    if no_ts:
        logger.warning(
            "Temporal split: %d / %d rows lack timestamps; applying stratified split to those rows",
            len(no_ts),
            len(records),
        )
        out.extend(
            stratified_split(
                no_ts,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                label_key=label_key,
                seed=seed + 1,
            )
        )

    random.Random(seed).shuffle(out)
    return out


def assign_splits(
    records: list[dict],
    *,
    mode: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    label_key: str = "label",
    domain_key: str = "domain",
    timestamp_keys: tuple[str, ...] | list[str] | None = None,
    seed: int = 42,
    auto_temporal_min_fraction: float = 0.5,
) -> list[dict]:
    """
    Dispatch split strategy.

    ``mode``: ``stratified`` | ``stratified_domain`` | ``temporal`` | ``auto``.

    ``auto``: use temporal if at least ``auto_temporal_min_fraction`` of rows have
    a parseable timestamp; otherwise ``stratified_domain``.
    """
    keys = tuple(timestamp_keys) if timestamp_keys is not None else DEFAULT_TIMESTAMP_KEYS
    m = (mode or "stratified_domain").strip().lower()

    if m == "auto":
        frac = fraction_with_timestamps(records, keys)
        if frac >= auto_temporal_min_fraction:
            logger.info("Split mode auto → temporal (%.0f%% rows have timestamps)", frac * 100)
            m = "temporal"
        else:
            logger.info(
                "Split mode auto → stratified_domain (only %.0f%% rows have timestamps)",
                frac * 100,
            )
            m = "stratified_domain"

    if m == "temporal":
        return temporal_split(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            label_key=label_key,
            timestamp_keys=keys,
            seed=seed,
        )
    if m == "stratified_domain":
        return stratified_domain_split(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            label_key=label_key,
            domain_key=domain_key,
            seed=seed,
        )
    if m == "stratified":
        return stratified_split(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            label_key=label_key,
            seed=seed,
        )

    logger.warning("Unknown split mode %r; using stratified_domain", mode)
    return stratified_domain_split(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        label_key=label_key,
        domain_key=domain_key,
        seed=seed,
    )


def _check_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {s}")


def _cut_points(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n == 0:
        return 0, 0
    t_end = int(n * train_ratio)
    v_end = t_end + int(n * val_ratio)
    if t_end == 0 and n > 0:
        t_end = 1
    if v_end <= t_end and n > t_end:
        v_end = min(t_end + 1, n)
    if v_end > n:
        v_end = n
    return t_end, v_end


def _split_name(i: int, t_end: int, v_end: int) -> str:
    if i < t_end:
        return "train"
    if i < v_end:
        return "validation"
    return "test"


def _majority_label(records: list[dict], label_key: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        lab = r.get(label_key) or "unknown"
        counts[lab] += 1
    return max(counts, key=counts.get)
