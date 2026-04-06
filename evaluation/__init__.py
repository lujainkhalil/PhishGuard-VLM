"""Evaluation: metrics, benchmarks, error analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .benchmark_paths import resolve_benchmark_manifest
from .metrics import BinaryClassificationMetrics, compute_binary_metrics, compute_binary_metrics_dict
from .threshold_tuning import sweep_threshold_f1

if TYPE_CHECKING:
    from .pipeline import EvaluationArtifacts, run_evaluation_pipeline, run_vlm_inference

__all__ = [
    "BinaryClassificationMetrics",
    "EvaluationArtifacts",
    "compute_binary_metrics",
    "compute_binary_metrics_dict",
    "resolve_benchmark_manifest",
    "run_evaluation_pipeline",
    "run_vlm_inference",
    "sweep_threshold_f1",
]


def __getattr__(name: str):
    if name == "EvaluationArtifacts":
        from .pipeline import EvaluationArtifacts

        return EvaluationArtifacts
    if name == "run_evaluation_pipeline":
        from .pipeline import run_evaluation_pipeline

        return run_evaluation_pipeline
    if name == "run_vlm_inference":
        from .pipeline import run_vlm_inference

        return run_vlm_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
