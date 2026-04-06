#!/usr/bin/env python3
"""Start the FastAPI server (inference pipeline loaded at startup)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhishGuard FastAPI service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Dev only: reload on code changes (loads model on each worker restart)",
    )
    args = parser.parse_args()
    if "PHISHGUARD_PROJECT_ROOT" not in os.environ:
        os.environ["PHISHGUARD_PROJECT_ROOT"] = str(_project_root)

    import uvicorn

    uvicorn.run(
        "inference.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
