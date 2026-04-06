"""Unit tests: multimodal fusion helpers and modules (no HF / transformers)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[2]
_cm_path = _ROOT / "models" / "fusion" / "cross_modal.py"
_spec = importlib.util.spec_from_file_location("cross_modal_under_test", _cm_path)
assert _spec and _spec.loader
_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cm)

vision_text_masks_from_input_ids = _cm.vision_text_masks_from_input_ids
CrossModalFusion = _cm.CrossModalFusion
WeightedModalPool = _cm.WeightedModalPool


@pytest.mark.requires_torch
class TestVisionTextMasks:
    def test_masks_disjoint(self) -> None:
        # B=2, L=8; image_token_id=99 at positions 1–3 for both rows
        input_ids = torch.tensor(
            [[0, 99, 99, 99, 1, 2, 3, 0], [5, 99, 99, 99, 7, 8, 0, 0]]
        )
        attn = torch.tensor([[1] * 8, [1] * 7 + [0]])
        vm, tm = vision_text_masks_from_input_ids(input_ids, attn, image_token_id=99)
        assert vm.sum().item() == 6
        assert (vm & tm).sum().item() == 0
        assert tm[0].sum().item() == 5


@pytest.mark.requires_torch
class TestWeightedModalPool:
    def test_output_shape(self) -> None:
        d = 32
        b, l = 2, 12
        h = torch.randn(b, l, d)
        vm = torch.zeros(b, l, dtype=torch.bool)
        vm[:, :4] = True
        tm = torch.ones(b, l, dtype=torch.bool)
        tm &= ~vm
        m = WeightedModalPool(d)
        out = m(h, vm, tm)
        assert out.shape == (b, d)


@pytest.mark.requires_torch
class TestCrossModalFusion:
    def test_cross_attention_output_shape(self) -> None:
        d = 64
        heads = 8
        b, tv, lt = 2, 4, 6
        l = tv + lt
        h = torch.randn(b, l, d)
        vm = torch.zeros(b, l, dtype=torch.bool)
        vm[:, :tv] = True
        tm = torch.zeros(b, l, dtype=torch.bool)
        tm[:, tv:] = True
        m = CrossModalFusion(d, num_heads=heads, dropout=0.0, gated=False)
        out = m(h, vm, tm)
        assert out.shape == (b, d)

    def test_gated_requires_global(self) -> None:
        d = 64
        b, tv, lt = 1, 4, 5
        l = tv + lt
        h = torch.randn(b, l, d)
        vm = torch.zeros(b, l, dtype=torch.bool)
        vm[:, :tv] = True
        tm = torch.zeros(b, l, dtype=torch.bool)
        tm[:, tv:] = True
        m = CrossModalFusion(d, num_heads=8, dropout=0.0, gated=True)
        with pytest.raises(ValueError, match="global_pooled"):
            m(h, vm, tm)
        gp = torch.randn(b, d)
        out = m(h, vm, tm, global_pooled=gp)
        assert out.shape == (b, d)
