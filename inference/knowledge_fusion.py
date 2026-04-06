"""
Configurable weighting of knowledge (brand–domain) vs model in :func:`~inference.aggregator.aggregate_signals`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class KnowledgeFusionConfig:
    """
    Tunes how strongly knowledge-module priors influence the fused phishing probability.

    - ``knowledge_weight_multiplier``: scales the knowledge blend weight ``w_k`` from dynamic rules
      (then renormalizes). Values > 1 trust knowledge more; < 1 dampen it.
    - ``max_knowledge_blend_weight``: cap on ``w_k`` after scaling (stability).
    - ``min_model_blend_weight``: floor on ``w_m`` so the VLM always retains some influence.
    """

    knowledge_weight_multiplier: float = 1.0
    max_knowledge_blend_weight: float = 0.62
    min_model_blend_weight: float = 0.22

    @staticmethod
    def from_mapping(m: Mapping[str, Any] | None) -> KnowledgeFusionConfig:
        raw = dict(m or {})
        return KnowledgeFusionConfig(
            knowledge_weight_multiplier=float(raw.get("knowledge_weight_multiplier", 1.0)),
            max_knowledge_blend_weight=float(raw.get("max_knowledge_blend_weight", 0.62)),
            min_model_blend_weight=float(raw.get("min_model_blend_weight", 0.22)),
        )


DEFAULT_KNOWLEDGE_FUSION = KnowledgeFusionConfig()
