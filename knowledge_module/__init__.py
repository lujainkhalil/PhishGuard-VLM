"""Knowledge enrichment modules (Wikidata, brand matching, cross-modal consistency)."""

from .cross_modal_consistency import CrossModalConsistency, compute_cross_modal_consistency

__all__ = [
    "CrossModalConsistency",
    "compute_cross_modal_consistency",
]
