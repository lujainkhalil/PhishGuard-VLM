"""Merging multiple crawl manifests before build_dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def project_with_assets(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    for i, url in enumerate(
        ("https://phish-one.example/path", "https://benign-two.example/"),
        start=1,
    ):
        png = d / f"s{i}.png"
        txt = d / f"t{i}.txt"
        Image.new("RGB", (120, 80), color=(i * 40, 10, 10)).save(png)
        txt.write_text("Some visible page text " * 5, encoding="utf-8")
    return tmp_path


def test_load_crawl_manifests_concat_order(tmp_path: Path) -> None:
    from data_pipeline.preprocessing.build import load_crawl_manifests_concat

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps([{"url": "https://a/", "x": 1}]), encoding="utf-8")
    b.write_text(json.dumps([{"url": "https://b/", "x": 2}]), encoding="utf-8")
    rows = load_crawl_manifests_concat([a, b], project_root=tmp_path)
    assert len(rows) == 2
    assert rows[0]["url"] == "https://a/"


def test_build_dataset_merge_two_ok_rows(project_with_assets: Path) -> None:
    from data_pipeline.preprocessing.build import build_dataset

    root = project_with_assets
    r1 = {
        "url": "https://phish-one.example/path",
        "final_url": "https://phish-one.example/path",
        "status": "ok",
        "screenshot_path": "raw/s1.png",
        "text_path": "raw/t1.txt",
        "label": "phishing",
        "source": "openphish",
        "redirect_count": 0,
    }
    r2 = {
        "url": "https://benign-two.example/",
        "final_url": "https://benign-two.example/",
        "status": "ok",
        "screenshot_path": "raw/s2.png",
        "text_path": "raw/t2.txt",
        "label": "benign",
        "source": "tranco",
        "redirect_count": 0,
    }
    m1 = root / "m1.json"
    m2 = root / "m2.json"
    m1.write_text(json.dumps([r1]), encoding="utf-8")
    m2.write_text(json.dumps([r2]), encoding="utf-8")

    df = build_dataset(
        m1,
        additional_crawl_manifest_paths=[m2],
        project_root=root,
        processed_dir=root / "processed",
        materialize=True,
        split_mode="stratified",
    )
    assert len(df) == 2
    labels = set(df["label"].tolist())
    assert labels == {"phishing", "benign"}
