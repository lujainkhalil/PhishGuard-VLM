"""
FastAPI package for URL phishing prediction.

ASGI entrypoint: ``inference.api.main:app`` (loads the full stack including PyTorch).
"""

from __future__ import annotations

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name == "app":
        from inference.api.main import app as _app

        return _app
    if name == "create_app":
        from inference.api.main import create_app as _create_app

        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
