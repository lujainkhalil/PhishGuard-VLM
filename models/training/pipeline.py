"""
Training pipeline: dataset loading, dataloaders, and trainer construction.

Single entry point for LoRA fine-tuning with validation and W&B tracking.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PhishingDataset, collate_fn

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load the processed manifest (Parquet or CSV): split, label, image paths, optional inline ``text``."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_datasets(
    manifest_df: pd.DataFrame,
    *,
    text_max_length: int = 2048,
    data_root: Path | str | None = None,
) -> tuple[PhishingDataset, PhishingDataset]:
    """
    Build train and validation datasets from the manifest.

    Returns:
        (train_dataset, val_dataset)
    """
    data_root = Path(data_root) if data_root else None
    train_ds = PhishingDataset(
        manifest_df,
        split="train",
        text_max_length=text_max_length,
        root=data_root,
    )
    val_ds = PhishingDataset(
        manifest_df,
        split="validation",
        text_max_length=text_max_length,
        root=data_root,
    )
    return train_ds, val_ds


def get_balanced_sampler(
    dataset: PhishingDataset,
    *,
    hard_negative_oversample: float = 1.0,
) -> torch.utils.data.Sampler:
    """
    WeightedRandomSampler to oversample minority class (for balanced_sampling=True).

    Benign rows with non-empty ``hard_negative_category`` in the manifest get an extra
    ``hard_negative_oversample`` multiplier (reduces false positives on login/branded pages).
    """
    labels = dataset.df["label"].map({"phishing": 1, "benign": 0}).fillna(0).astype(int).tolist()
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    hne = None
    if "hard_negative_category" in dataset.df.columns:
        hne = dataset.df["hard_negative_category"]
    mult = max(1.0, float(hard_negative_oversample))
    sw_list: list[float] = []
    for i, lab in enumerate(labels):
        w = class_weights[lab].item()
        if mult > 1.0 and lab == 0 and hne is not None:
            cell = hne.iloc[i]
            if not pd.isna(cell) and str(cell).strip():
                w *= mult
        sw_list.append(w)
    sample_weights = torch.tensor(sw_list, dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def build_dataloaders(
    train_dataset: PhishingDataset,
    val_dataset: PhishingDataset,
    *,
    batch_size: int = 8,
    balanced_sampling: bool = True,
    hard_negative_oversample: float = 1.0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders.

    If balanced_sampling is True, train loader uses WeightedRandomSampler.
    """
    train_sampler = (
        get_balanced_sampler(train_dataset, hard_negative_oversample=hard_negative_oversample)
        if balanced_sampling
        else None
    )
    pin = torch.cuda.is_available() and num_workers >= 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not balanced_sampling,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def build_test_dataloader(
    manifest_df: pd.DataFrame,
    *,
    split: str | None = "test",
    text_max_length: int = 2048,
    data_root: Path | str | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """
    Single :class:`DataLoader` for evaluation.

    If ``split`` is set and a ``split`` column exists, filter to that split; otherwise use all rows.
    """
    data_root = Path(data_root) if data_root else None
    use_split = split if split is not None and "split" in manifest_df.columns else None
    ds = PhishingDataset(
        manifest_df,
        split=use_split,
        text_max_length=text_max_length,
        root=data_root,
    )
    pin = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
