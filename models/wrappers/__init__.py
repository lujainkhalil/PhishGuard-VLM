"""
Reusable VLM wrappers (Hugging Face backbones) decoupled from task heads.
"""

from .llava15_multimodal import (
    DEFAULT_MODEL_ID,
    LLaVA15MultimodalWrapper,
    pool_hidden_state,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "LLaVA15MultimodalWrapper",
    "pool_hidden_state",
]
