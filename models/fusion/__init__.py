"""Multimodal fusion (cross-attention, alignment, weighted pooling) on top of LLaVA hidden states."""

from .cross_modal import CrossModalFusion, WeightedModalPool, vision_text_masks_from_input_ids

__all__ = [
    "CrossModalFusion",
    "WeightedModalPool",
    "vision_text_masks_from_input_ids",
]
