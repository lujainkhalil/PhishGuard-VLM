"""Unit tests: threshold sweep (no torch)."""

from __future__ import annotations

import numpy as np

from evaluation.threshold_tuning import sweep_threshold_f1


def test_sweep_threshold_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8])
    t, f1, m = sweep_threshold_f1(y_true, y_score, n_thresholds=50)
    assert f1 == 1.0
    assert 0.0 < t < 1.0
    assert m["accuracy"] == 1.0
