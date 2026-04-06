#!/usr/bin/env python3
"""Evaluate on the TR-OP benchmark (same as ``python scripts/run_eval.py --tr-op``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_cmd = [sys.executable, str(_root / "scripts" / "run_eval.py"), "--tr-op", *sys.argv[1:]]
sys.exit(subprocess.run(_cmd, cwd=_root).returncode)
