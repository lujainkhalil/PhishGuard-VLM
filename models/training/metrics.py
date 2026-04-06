"""
Binary classification metrics for validation / logging (delegates to ``evaluation.metrics``).
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.metrics import compute_binary_metrics


def compute_binary_classification_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    y_score: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute accuracy, precision, recall, and F1; include ``roc_auc`` when ``y_score`` is given.

    For a full report including the confusion matrix, use
    :func:`evaluation.metrics.compute_binary_metrics`.
    """
    m = compute_binary_metrics(y_true, y_pred, y_score)
    out: dict[str, float] = {
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }
    if m.roc_auc is not None:
        out["roc_auc"] = m.roc_auc
    return out
