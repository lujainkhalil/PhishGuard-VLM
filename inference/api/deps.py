"""FastAPI dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from inference.pipeline import URLInferencePipeline


def get_pipeline(request: Request) -> URLInferencePipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("Inference pipeline is not initialized (application misconfiguration).")
    return pipeline
