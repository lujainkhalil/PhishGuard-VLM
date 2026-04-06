"""
Explicit vision–text fusion on top of LLaVA sequence hidden states.

LLaVA already runs deep self-attention over interleaved image and text tokens; this module adds a
lightweight **decision-time** fusion block: modality-specific alignment projections, bidirectional
cross-attention (text queries image tokens; image queries text tokens), optional gating with the
standard last-token pool, or a simple learned scalar mix of mean vision vs mean text.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def vision_text_masks_from_input_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``vision_mask[b, i]`` is True where LLaVA inserted image patch embeddings (``image_token_id``).
    ``text_mask`` is True for real text/prompt tokens (not image, not padding).
    """
    vision = input_ids == image_token_id
    if attention_mask is None:
        text = ~vision
    else:
        text = attention_mask.bool() & ~vision
    return vision, text


def _gather_vision_tokens(hidden: torch.Tensor, vision_mask: torch.Tensor) -> torch.Tensor:
    """(B, L, D) + (B, L) bool -> (B, T_v, D) with fixed T_v per batch row."""
    b = hidden.size(0)
    counts = vision_mask.sum(dim=1)
    if counts.numel() == 0 or counts.min() != counts.max():
        raise ValueError(
            "Expected a fixed number of vision tokens per sample (LLaVA image_seq_length); "
            f"got per-row counts min={counts.min().item()}, max={counts.max().item()}"
        )
    t = int(counts[0].item())
    if t == 0:
        raise ValueError("vision_mask has no True positions — check image_token_id vs input_ids")
    return hidden[vision_mask].reshape(b, t, hidden.size(-1))


def _gather_text_tokens(
    hidden: torch.Tensor, text_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack variable-length text tokens to (B, L_max, D) and return key_padding_mask for nn.MHA
    (True = ignore position).
    """
    b, _, d = hidden.shape
    lengths = text_mask.sum(dim=1)
    max_len = int(lengths.max().item())
    if max_len < 1:
        out = hidden.new_zeros(b, 1, d)
        kpm = torch.ones(b, 1, dtype=torch.bool, device=hidden.device)
        return out, kpm

    out = hidden.new_zeros(b, max_len, d)
    kpm = torch.ones(b, max_len, dtype=torch.bool, device=hidden.device)
    for i in range(b):
        idx = text_mask[i].nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n > 0:
            out[i, :n] = hidden[i, idx]
            kpm[i, :n] = False
    return out, kpm


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, L, D) with (B, L) bool -> (B, D)."""
    m = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    summed = (hidden * m).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(dtype=hidden.dtype)
    return summed / denom


class WeightedModalPool(nn.Module):
    """``sigmoid(logit) * v_mean + (1 - sigmoid(logit)) * t_mean`` with LayerNorm on each branch."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln_v = nn.LayerNorm(embed_dim)
        self.ln_t = nn.LayerNorm(embed_dim)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden: torch.Tensor,
        vision_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        v = self.ln_v(masked_mean(hidden, vision_mask))
        t = self.ln_t(masked_mean(hidden, text_mask))
        w = torch.sigmoid(self.mix_logit)
        return w * v + (1.0 - w) * t


class CrossModalFusion(nn.Module):
    """
    Align vision/text token streams, run bidirectional cross-attention, project to ``embed_dim``.

    If ``gated=True``, blends with ``global_pooled`` (e.g. last-token pool) via a learned scalar gate.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.1,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.gated = gated
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.align_v = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.align_t = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))

        self.cross_t2v = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_v2t = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        fused_in = embed_dim * 4
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        if gated:
            self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_mask: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        global_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v_seq = _gather_vision_tokens(hidden_states, vision_mask)
        t_seq, text_kpm = _gather_text_tokens(hidden_states, text_mask)

        v_seq = self.align_v(v_seq)
        t_seq = self.align_t(t_seq)

        v_pool = v_seq.mean(dim=1, keepdim=True)
        t_pool = masked_mean(hidden_states, text_mask).unsqueeze(1)
        t_seq_att = t_seq

        t2v, _ = self.cross_t2v(query=t_pool, key=v_seq, value=v_seq)
        v2t, _ = self.cross_v2t(query=v_pool, key=t_seq_att, value=t_seq_att, key_padding_mask=text_kpm)

        t2v = t2v.squeeze(1)
        v2t = v2t.squeeze(1)
        v_mean = v_pool.squeeze(1)
        t_mean = t_pool.squeeze(1)

        fused = self.fuse(torch.cat([t2v, v2t, v_mean, t_mean], dim=-1))

        if not self.gated:
            return fused

        if global_pooled is None:
            raise ValueError("gated CrossModalFusion requires global_pooled")
        g = torch.sigmoid(self.gate_logit)
        return g * fused + (1.0 - g) * global_pooled
