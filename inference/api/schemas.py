"""Request / response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Page URL to crawl and classify (scheme optional; http:// added if missing).",
        examples=["https://www.example.com/"],
    )


class PredictResponse(BaseModel):
    label: int = Field(..., description="0 = benign, 1 = phishing")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision strength in [0, 1]")
    explanation: str = Field(..., description="Human-readable rationale")
    cross_modal_consistency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Brand alignment across page text, screenshot (OCR when available), and domain; higher is more consistent.",
    )
    cross_modal: dict[str, Any] | None = Field(
        default=None,
        description="Diagnostics: extracted text/image brand candidates, domain registrable, OCR flag, notes.",
    )
