"""
Inference utilities: fuse model outputs with knowledge signals; URL → crawl → model pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .aggregator import (
    AggregatedVerdict,
    KnowledgeSignalLike,
    ModelPrediction,
    aggregate_signals,
    knowledge_to_phish_prior,
)
from .knowledge_fusion import DEFAULT_KNOWLEDGE_FUSION, KnowledgeFusionConfig

if TYPE_CHECKING:
    from .pipeline import URLInferencePipeline, URLInferenceResult

__all__ = [
    "AggregatedVerdict",
    "DEFAULT_KNOWLEDGE_FUSION",
    "KnowledgeFusionConfig",
    "KnowledgeSignalLike",
    "ModelPrediction",
    "URLInferencePipeline",
    "URLInferenceResult",
    "aggregate_signals",
    "knowledge_to_phish_prior",
]


def __getattr__(name: str):
    if name == "URLInferencePipeline":
        from .pipeline import URLInferencePipeline

        return URLInferencePipeline
    if name == "URLInferenceResult":
        from .pipeline import URLInferenceResult

        return URLInferenceResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
