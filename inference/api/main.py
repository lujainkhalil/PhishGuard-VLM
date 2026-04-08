"""
FastAPI application factory and ASGI entrypoint.

Run locally::

    uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

Or::

    python scripts/run_api.py
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from inference.api.routes import router
from inference.pipeline import URLInferencePipeline

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    env = os.environ.get("PHISHGUARD_PROJECT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    # inference/api/main.py → repo root
    return Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    root = _project_root()
    import os
    mock = os.environ.get("PHISHGUARD_MOCK", "0") == "1"
    adapter = os.environ.get("PHISHGUARD_ADAPTER_PATH", "").strip()

    if mock:
        logger.warning("Running in MOCK mode")
        from inference.api.mock_pipeline import MockInferencePipeline
        app.state.pipeline = MockInferencePipeline()
    elif adapter:
        logger.info("Loading VLM adapter from %s", adapter)
        from inference.vlm_inference import VLMInferencePipeline
        app.state.pipeline = VLMInferencePipeline(adapter)
    else:
        logger.info("Loading full pipeline from %s", root)
        app.state.pipeline = URLInferencePipeline.from_config(root)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="PhishGuard VLM",
        description="URL → crawl → multimodal classifier + optional knowledge fusion.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    ui_path = _project_root() / "web_app" / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    async def serve_web_ui() -> FileResponse:
        if not ui_path.is_file():
            raise HTTPException(status_code=404, detail="Web UI not found at web_app/static/index.html")
        return FileResponse(ui_path)

    return app


app = create_app()
