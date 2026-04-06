"""
Binary classification head: multimodal embeddings -> MLP -> logit(s) / phishing probability.
"""

import torch
import torch.nn as nn


class PhishingClassificationHead(nn.Module):
    """
    Two-layer MLP on pooled multimodal embeddings with input LayerNorm, GELU, and dropout.

    Forward returns **logits** (use :meth:`logits_to_probability` for phishing probability in [0, 1]
    when ``num_classes == 1``).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 1,
        *,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        h = mlp_hidden_dim if mlp_hidden_dim is not None else max(256, min(2048, embedding_dim // 2))
        self.norm = nn.LayerNorm(embedding_dim) if use_layer_norm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embedding_dim) from the VLM backbone.
        Returns:
            logits: (batch, num_classes); for binary, shape (batch, 1).
        """
        x = self.norm(embeddings)
        return self.mlp(x)

    @staticmethod
    def logits_to_probability(logits: torch.Tensor) -> torch.Tensor:
        """(B, 1) or (B,) logits -> (B,) phishing probability in [0, 1] (binary)."""
        if logits.dim() == 2:
            logits = logits.squeeze(-1)
        return logits.sigmoid()

    @staticmethod
    def probability_to_label(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """(B,) probability -> (B,) long label: 1 if prob >= threshold else 0."""
        return (prob >= threshold).long()
