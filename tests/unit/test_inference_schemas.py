"""Unit tests: API request/response schemas."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from inference.api.schemas import PredictRequest, PredictResponse


def test_predict_request_accepts_url_string() -> None:
    r = PredictRequest(url="https://example.com/path")
    assert r.url == "https://example.com/path"


def test_predict_request_rejects_empty() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictRequest(url="")


def test_predict_response_roundtrip() -> None:
    r = PredictResponse(label=0, confidence=0.75, explanation="Looks fine.")
    d = r.model_dump()
    assert d["label"] == 0
    assert d["confidence"] == 0.75
