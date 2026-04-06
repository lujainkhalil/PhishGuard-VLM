"""
Threshold search on validation scores to maximize F1 (or other metrics).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def sweep_threshold_f1(
    y_true: np.ndarray | list[Any],
    y_score: np.ndarray | list[Any],
    *,
    n_thresholds: int = 101,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[float, float, dict[str, float]]:
    """
    Grid-search a decision threshold on ``y_score`` to maximize F1 (phishing = positive class 1).

    Returns:
        (best_threshold, best_f1, metrics_at_best) where ``metrics_at_best`` includes
        accuracy, precision, recall, f1 at the chosen threshold.
    """
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    ys = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if len(yt) != len(ys):
        raise ValueError("y_true and y_score length mismatch")
    if len(yt) == 0:
        return 0.5, 0.0, {}

    thresholds = np.linspace(t_min, t_max, n_thresholds)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        yp = (ys >= t).astype(np.int64)
        f1 = float(f1_score(yt, yp, pos_label=1, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    yp_best = (ys >= best_t).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(yt, yp_best)),
        "precision": float(precision_score(yt, yp_best, pos_label=1, zero_division=0)),
        "recall": float(recall_score(yt, yp_best, pos_label=1, zero_division=0)),
        "f1": float(f1_score(yt, yp_best, pos_label=1, zero_division=0)),
    }
    return best_t, best_f1, metrics
