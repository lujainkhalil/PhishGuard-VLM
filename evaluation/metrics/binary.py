"""
Binary classification metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

Works with NumPy arrays or PyTorch tensors. ``y_score`` should be the **positive-class**
(phishing) probability or score for ROC-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy_1d(y: Any) -> np.ndarray:
    """Accept NumPy arrays, Python sequences, or PyTorch tensors (no hard ``torch`` import)."""
    if isinstance(y, np.ndarray):
        return y.astype(np.float64).reshape(-1)
    if hasattr(y, "detach"):
        t = y.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        return np.asarray(t.numpy(), dtype=np.float64).reshape(-1)
    return np.asarray(y, dtype=np.float64).reshape(-1)


@dataclass
class BinaryClassificationMetrics:
    """Container for standard binary classification evaluation outputs."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: np.ndarray
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_samples = int(
            self.true_negative + self.false_positive + self.false_negative + self.true_positive
        )

    def to_dict(self, *, include_matrix: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_positive": self.true_positive,
            "n_samples": self.n_samples,
        }
        if include_matrix:
            d["confusion_matrix"] = self.confusion_matrix.tolist()
        return d


def compute_binary_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    *,
    pos_label: int = 1,
    labels: tuple[int, int] = (0, 1),
) -> BinaryClassificationMetrics:
    """
    Compute accuracy, precision, recall, F1, optional ROC-AUC, and a 2×2 confusion matrix.

    Args:
        y_true: Gold labels in ``{0, 1}``.
        y_pred: Hard predictions in ``{0, 1}``.
        y_score: Continuous scores / probabilities for the **positive** class (required for ROC-AUC).
        pos_label: Positive class id (default ``1`` = phishing).
        labels: Row/column order for the confusion matrix ``(negative, positive)``.
    """
    yt = _to_numpy_1d(y_true).astype(int)
    yp = _to_numpy_1d(y_pred).astype(int)

    acc = float(accuracy_score(yt, yp))
    prec = float(precision_score(yt, yp, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(yt, yp, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(yt, yp, pos_label=pos_label, zero_division=0))

    cm = confusion_matrix(yt, yp, labels=list(labels))
    if cm.shape != (2, 2):
        full = np.zeros((2, 2), dtype=np.int64)
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                full[i, j] = int(np.sum((yt == li) & (yp == lj)))
        cm = full

    # Rows = true, cols = pred; labels order (0, 1) → [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    roc: float | None = None
    if y_score is not None:
        ys = _to_numpy_1d(y_score)
        if len(ys) != len(yt):
            raise ValueError(f"y_score length {len(ys)} != y_true length {len(yt)}")
        if np.unique(yt).size < 2:
            roc = None
        else:
            try:
                roc = float(roc_auc_score(yt, ys))
            except ValueError:
                roc = None

    return BinaryClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        confusion_matrix=cm,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        true_positive=tp,
    )


def compute_binary_metrics_dict(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Same as :func:`compute_binary_metrics` but returns a JSON-friendly dict."""
    return compute_binary_metrics(y_true, y_pred, y_score, **kwargs).to_dict()
