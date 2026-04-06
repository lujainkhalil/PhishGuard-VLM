"""
PyTorch Dataset for the preprocessed manifest: screenshot path, text path, label.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_ID = {"benign": 0, "phishing": 1}
ID_TO_LABEL = {0: "benign", 1: "phishing"}


def _load_image(path: str | Path) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    img = Image.open(p).convert("RGB")
    return img


def _load_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore").strip()


class PhishingDataset(Dataset):
    """
    Dataset of (image, webpage text, label) from the processed manifest.

    Manifest columns: screenshot_path or image_path, text_path and/or inline ``text``,
    label, split, url, optional ``hard_negative_category`` (benign phishing-like pages), ...
    Label: "phishing" -> 1, "benign" -> 0.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        *,
        split: str | None = None,
        text_max_length: int = 2048,
        root: Path | str | None = None,
    ):
        """
        Args:
            manifest_df: DataFrame with screenshot_path, text_path, label, split.
            split: If set, filter to this split (train/validation/test).
            text_max_length: Truncate raw text to this many chars (for consistency).
            root: Optional root to prepend to paths.
        """
        self.df = manifest_df.copy()
        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.text_max_length = text_max_length
        self.root = Path(root) if root else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        if "image_path" in self.df.columns and pd.notna(row.get("image_path")):
            screenshot_path = row["image_path"]
        else:
            screenshot_path = row["screenshot_path"]
        text_path = row.get("text_path", "")
        label_str = (row["label"] or "benign").strip().lower()
        label = LABEL_TO_ID.get(label_str, 0)
        if self.root:
            if not Path(screenshot_path).is_absolute():
                screenshot_path = self.root / screenshot_path
            if text_path and not Path(text_path).is_absolute():
                text_path = self.root / text_path
        image = _load_image(screenshot_path)
        if "text" in self.df.columns:
            cell = row["text"]
            text = "" if pd.isna(cell) else str(cell).strip()
            if not text and text_path:
                text = _load_text(text_path)
        else:
            text = _load_text(text_path) if text_path else ""
        if len(text) > self.text_max_length:
            text = text[: self.text_max_length]
        return {
            "image": image,
            "text": text,
            "label": torch.tensor(label, dtype=torch.long),
            "url": row.get("url", ""),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate batch: keep images and texts as lists for processor; stack labels.
    """
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    urls = [b["url"] for b in batch]
    return {
        "images": images,
        "texts": texts,
        "labels": labels,
        "urls": urls,
    }
