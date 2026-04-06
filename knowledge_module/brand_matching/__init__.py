"""
Brand vs page-domain matching and impersonation risk signals.
"""

from .domain import (
    extract_host,
    host_under_brand_domain,
    normalize_brand_domain_entry,
    registrable_domain,
)
from .domain_suspicion import compute_domain_suspicion
from .matcher import (
    BrandDomainRisk,
    assess_brand_domain_risk,
    assess_url_against_wikidata_brand,
    normalize_official_domains,
)

__all__ = [
    "BrandDomainRisk",
    "assess_brand_domain_risk",
    "assess_url_against_wikidata_brand",
    "compute_domain_suspicion",
    "extract_host",
    "host_under_brand_domain",
    "normalize_brand_domain_entry",
    "normalize_official_domains",
    "registrable_domain",
]
