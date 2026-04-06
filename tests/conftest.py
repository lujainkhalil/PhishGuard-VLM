"""
Shared pytest configuration and fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parent.parent
