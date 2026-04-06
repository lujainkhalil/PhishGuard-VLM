"""
Integration tests: FastAPI /predict and /health with a stub pipeline (no model load).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference.api.routes import router


class StubPipeline:
    phish_threshold = 0.5

    def analyze(
        self,
        url: str,
        *,
        brand_hint: str | None = None,
        official_domains: list[str] | None = None,
    ):
        return SimpleNamespace(
            label=1,
            confidence=0.84,
            explanation="Stub: flagged as phishing for testing.",
        )


@pytest.fixture
def test_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = StubPipeline()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


@pytest.mark.integration
def test_health_ok(test_app: FastAPI) -> None:
    with TestClient(test_app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.integration
def test_predict_returns_label_confidence_explanation(test_app: FastAPI) -> None:
    with TestClient(test_app) as client:
        r = client.post("/predict", json={"url": "https://phishing.example/login"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == 1
    assert isinstance(data["confidence"], (int, float))
    assert data["confidence"] == pytest.approx(0.84)
    assert "Stub" in data["explanation"] or "phishing" in data["explanation"].lower()


@pytest.mark.integration
def test_predict_validation_error_on_missing_url(test_app: FastAPI) -> None:
    with TestClient(test_app) as client:
        r = client.post("/predict", json={})
    assert r.status_code == 422
