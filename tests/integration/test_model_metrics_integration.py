"""
Integration tests: training metrics delegate to evaluation metrics (numpy / sklearn path).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evaluation.metrics import compute_binary_metrics
from models.training.metrics import compute_binary_classification_metrics


@pytest.mark.integration
@pytest.mark.requires_torch
def test_compute_binary_metrics_perfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_binary_metrics(y_true, y_pred, y_score)
    assert m.accuracy == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0


@pytest.mark.integration
@pytest.mark.requires_torch
def test_compute_binary_classification_metrics_torch_tensors() -> None:
    y_true = torch.tensor([0, 1, 1])
    y_pred = torch.tensor([0, 1, 0])
    y_score = torch.tensor([0.1, 0.7, 0.4])
    d = compute_binary_classification_metrics(y_true, y_pred, y_score)
    assert "accuracy" in d
    assert "f1" in d
    assert "roc_auc" in d
    assert 0.0 <= d["accuracy"] <= 1.0
