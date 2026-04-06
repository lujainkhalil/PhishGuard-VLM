"""
LLaVA-1.5 backbone: thin alias around :class:`LLaVA15MultimodalWrapper` for backward compatibility.
"""

from __future__ import annotations

from ..wrappers.llava15_multimodal import LLaVA15MultimodalWrapper


class LlavaBackbone(LLaVA15MultimodalWrapper):
    """
    Multimodal encoder (image + text) → pooled embeddings for a classification head.

    Same as :class:`~models.wrappers.LLaVA15MultimodalWrapper`; use :meth:`prepare_inputs`
    for the historical API name.
    """

    def prepare_inputs(self, *args, **kwargs):
        """Alias for :meth:`prepare_multimodal_inputs`."""
        return self.prepare_multimodal_inputs(*args, **kwargs)
