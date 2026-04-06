"""Unit tests: classification head (model pipeline)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.heads.classification import PhishingClassificationHead


@pytest.mark.requires_torch
class TestPhishingClassificationHead:
    def test_forward_shape(self) -> None:
        head = PhishingClassificationHead(128, num_classes=1, mlp_hidden_dim=64, dropout=0.0)
        x = torch.randn(4, 128)
        logits = head(x)
        assert logits.shape == (4, 1)

    def test_logits_to_probability_range(self) -> None:
        logits = torch.tensor([[0.0], [2.0], [-2.0]])
        p = PhishingClassificationHead.logits_to_probability(logits)
        assert p.shape == (3,)
        assert bool((p >= 0).all() and (p <= 1).all())

    def test_probability_to_label_threshold(self) -> None:
        p = torch.tensor([0.49, 0.51])
        y = PhishingClassificationHead.probability_to_label(p, threshold=0.5)
        assert y.tolist() == [0, 1]
