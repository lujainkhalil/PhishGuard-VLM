"""HTTP routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from inference.api.deps import get_pipeline
from inference.api.schemas import PredictRequest, PredictResponse

if TYPE_CHECKING:
    from inference.pipeline import URLInferencePipeline

router = APIRouter(tags=["predict"])


@router.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a URL",
    response_description="Binary label, confidence, and fused explanation",
)
async def predict(
    body: PredictRequest,
    pipeline: Annotated["URLInferencePipeline", Depends(get_pipeline)],
) -> PredictResponse:
    try:
        result = await run_in_threadpool(pipeline.analyze, body.url)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e!s}",
        ) from e
    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        explanation=result.explanation,
        cross_modal_consistency=getattr(result, "cross_modal_consistency", None),
        cross_modal=getattr(result, "cross_modal", None),
    )
