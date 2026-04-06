"""
Classification objectives for phishing training (binary logits).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_pos_weight_from_manifest(train_df: pd.DataFrame, *, label_col: str = "label") -> float | None:
    """
    ``pos_weight`` for :class:`~torch.nn.BCEWithLogitsLoss`: weight on the positive (phishing) class.

    PyTorch uses ``pos_weight`` such that loss for positive examples is scaled; setting
    ``pos_weight = n_negative / n_positive`` upweights the minority positive class when phishing is rarer.
    """
    if train_df is None or len(train_df) == 0 or label_col not in train_df.columns:
        return None
    s = train_df[label_col].astype(str).str.lower().str.strip()
    n_pos = int((s == "phishing").sum())
    n_neg = int((s == "benign").sum())
    if n_pos == 0:
        logger.warning("No phishing rows in train subset; pos_weight skipped.")
        return None
    w = float(n_neg) / float(n_pos)
    logger.info("BCE pos_weight (n_neg/n_pos): %.4f (benign=%d, phishing=%d)", w, n_neg, n_pos)
    return w


class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal loss on binary logits (Lin et al.); down-weights easy examples via ``(1 - p_t)^gamma``.
    ``alpha`` balances contribution of positive vs negative labels (typical 0.25 favors negatives less).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (1.0 - pt).pow(self.gamma) * bce
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal).mean()


def build_train_criterion(
    loss_cfg: dict[str, Any] | None,
    *,
    train_df: pd.DataFrame,
    device: torch.device,
) -> nn.Module:
    """
    Build a criterion from ``configs/training.yaml`` ``loss:`` block.

    Keys:
        - ``type``: ``bce`` (default) or ``focal``
        - ``pos_weight``: ``auto`` | ``null`` | number (BCE only; ``auto`` uses train class counts)
        - ``focal_gamma``, ``focal_alpha`` (focal only)
    """
    cfg = loss_cfg or {}
    loss_type = str(cfg.get("type", "bce")).lower().strip()

    if loss_type == "focal":
        gamma = float(cfg.get("focal_gamma", 2.0))
        alpha = float(cfg.get("focal_alpha", 0.25))
        return BinaryFocalLossWithLogits(gamma=gamma, alpha=alpha).to(device)

    pos_raw = cfg.get("pos_weight", "auto")
    pw: torch.Tensor | None = None
    if pos_raw in (None, "none", "off", False):
        pw = None
    elif pos_raw in ("auto", True, "true"):
        w = compute_pos_weight_from_manifest(train_df)
        if w is not None:
            pw = torch.tensor([w], dtype=torch.float32, device=device)
    else:
        pw = torch.tensor([float(pos_raw)], dtype=torch.float32, device=device)

    if pw is None:
        return nn.BCEWithLogitsLoss().to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pw).to(device)
