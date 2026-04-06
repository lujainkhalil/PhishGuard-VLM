"""
Resolve benchmark dataset paths (e.g. TR-OP) to a concrete manifest file.
"""

from __future__ import annotations

from pathlib import Path

_MANIFEST_NAMES = (
    "manifest.parquet",
    "manifest.csv",
    "tr_op.parquet",
    "tr-op.parquet",
    "test.parquet",
)


def resolve_benchmark_manifest(project_root: Path, candidate: str | Path) -> Path:
    """
    If ``candidate`` is a file, return it (resolved under ``project_root`` if relative).

    If it is a directory, return the first existing file among common manifest names.
    """
    p = Path(candidate)
    if not p.is_absolute():
        p = project_root / p
    p = p.resolve()
    if p.is_file():
        return p
    if p.is_dir():
        for name in _MANIFEST_NAMES:
            c = p / name
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"No benchmark manifest found in {p}; expected one of: {', '.join(_MANIFEST_NAMES)}"
        )
    raise FileNotFoundError(f"Benchmark path not found: {p}")
