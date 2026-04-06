"""
Wikidata SPARQL + search API with disk cache and resilient HTTP handling.
"""

from .client import BrandInfo, DEFAULT_SPARQL_ENDPOINT, WikidataClient

__all__ = [
    "BrandInfo",
    "DEFAULT_SPARQL_ENDPOINT",
    "WikidataClient",
]
