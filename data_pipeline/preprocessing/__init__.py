"""
Preprocessing: turn crawl output into a multimodal dataset with train/val/test splits.

Heavy manifest helpers in ``build`` are loaded lazily so ``import data_pipeline.preprocessing.text_processing``
does not require pandas.
"""

from __future__ import annotations

from .image_processing import prepare_image
from .splits import assign_splits, parse_record_timestamp, stratified_domain_split, temporal_split
from .text_processing import clean_and_normalize_text, extract_visible_text_from_html

__all__ = [
    "assign_splits",
    "asset_stem_for_url",
    "build_dataset",
    "clean_and_normalize_text",
    "DatasetValidationReport",
    "extract_visible_text_from_html",
    "load_crawl_manifest",
    "load_crawl_manifests_concat",
    "log_manifest_statistics",
    "log_validation_report",
    "materialize_processed_records",
    "parse_record_timestamp",
    "prepare_image",
    "save_manifest",
    "save_split_manifests",
    "screenshot_file_is_valid",
    "stratified_domain_split",
    "temporal_split",
    "validate_processed_manifest",
]

_BUILD_NAMES = frozenset(
    {
        "asset_stem_for_url",
        "build_dataset",
        "load_crawl_manifest",
        "load_crawl_manifests_concat",
        "materialize_processed_records",
        "save_manifest",
        "save_split_manifests",
    }
)

_VALIDATION_NAMES = frozenset(
    {
        "DatasetValidationReport",
        "log_manifest_statistics",
        "log_validation_report",
        "screenshot_file_is_valid",
        "validate_processed_manifest",
    }
)


def __getattr__(name: str):
    if name in _BUILD_NAMES:
        from . import build

        return getattr(build, name)
    if name in _VALIDATION_NAMES:
        from . import validation

        return getattr(validation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
