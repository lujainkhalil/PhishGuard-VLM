"""Unit tests: processed manifest validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from data_pipeline.preprocessing.validation import (
    DatasetValidationReport,
    screenshot_file_is_valid,
    validate_processed_manifest,
)


@pytest.fixture
def tmp_data(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    img_dir = root / "processed" / "images"
    img_dir.mkdir(parents=True)
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = img_dir / f"im{i}.png"
        Image.new("RGB", (64, 64), color).save(p)
    return root


def test_screenshot_file_is_valid(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    Image.new("RGB", (200, 200), (1, 2, 3)).save(p)
    assert screenshot_file_is_valid(p, min_bytes=50) is True
    assert screenshot_file_is_valid(tmp_path / "missing.png") is False


def test_validate_drops_empty_short_duplicate(tmp_data: Path) -> None:
    img_rel = "processed/images/im0.png"
    df = pd.DataFrame(
        [
            {
                "url": "https://a.com",
                "text": "x" * 60,
                "image_path": img_rel,
                "label": "phishing",
                "split": "train",
            },
            {
                "url": "https://b.com",
                "text": "",
                "image_path": img_rel,
                "label": "benign",
                "split": "train",
            },
            {
                "url": "https://c.com",
                "text": "short",
                "image_path": "processed/images/im1.png",
                "label": "benign",
                "split": "train",
            },
            {
                "url": "https://a.com",
                "text": "y" * 60,
                "image_path": img_rel,
                "label": "phishing",
                "split": "test",
            },
        ]
    )
    out, rep = validate_processed_manifest(
        df,
        tmp_data,
        min_text_length=50,
        dedupe_by_url=True,
        dedupe_by_normalized_url=False,
    )
    assert isinstance(rep, DatasetValidationReport)
    assert rep.dropped_empty_text >= 1
    assert rep.dropped_short_text >= 1
    assert rep.dropped_duplicate_url >= 1
    assert len(out) == 1
    assert out.iloc[0]["url"] == "https://a.com"
