"""
High-level brand–domain matching and risk assessment for URLs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from .domain import (
    extract_host,
    host_under_brand_domain,
    normalize_brand_domain_entry,
    registrable_domain,
    split_sld_tld,
)
from .domain_suspicion import compute_domain_suspicion
from .impersonation import collect_impersonation_signals

if TYPE_CHECKING:
    from ..wikidata.client import WikidataClient

logger = logging.getLogger(__name__)


def _score_to_level(score: float) -> str:
    if score <= 0.05:
        return "none"
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "medium"
    return "high"


@dataclass
class BrandDomainRisk:
    """Risk signal for a URL given official brand web origins."""

    risk_level: str
    score: float
    page_host: str | None
    page_registrable: str | None
    official_domains: list[str]
    matched_official: bool
    signals: list[str] = field(default_factory=list)
    claimed_brand: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "score": self.score,
            "page_host": self.page_host,
            "page_registrable": self.page_registrable,
            "official_domains": list(self.official_domains),
            "matched_official": self.matched_official,
            "signals": list(self.signals),
            "claimed_brand": self.claimed_brand,
        }


def _compact_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _brand_aligns_official_sld(claimed_brand: str, official_slds: set[str]) -> float:
    """How well the claimed brand string matches at least one official registrable SLD (0–1)."""
    c = _compact_alnum(claimed_brand)
    if len(c) < 3:
        return 0.0
    best = 0.0
    for sld in official_slds:
        if not sld:
            continue
        s = sld.lower()
        if c == s or s in c or c in s:
            best = max(best, 0.92)
            continue
        best = max(best, SequenceMatcher(None, c, s).ratio())
    return best


def collect_claimed_brand_hostname_mismatch(
    claimed_brand: str | None,
    host: str,
    official_brand_domains: list[str],
    *,
    matched_official: bool,
) -> tuple[list[str], float]:
    """
    If the claimed brand clearly corresponds to an official SLD but the page host is not under
    any official domain, emit a strong domain–brand mismatch signal.
    """
    if not claimed_brand or not str(claimed_brand).strip() or matched_official or not host:
        return [], 0.0
    if not official_brand_domains:
        return [], 0.0

    official_slds: set[str] = set()
    for od in official_brand_domains:
        reg = registrable_domain(normalize_brand_domain_entry(od) or od or "")
        sld, _ = split_sld_tld(reg)
        if sld:
            official_slds.add(sld.lower())

    if not official_slds:
        return [], 0.0

    align = _brand_aligns_official_sld(str(claimed_brand).strip(), official_slds)
    if align < 0.82:
        return [], 0.0

    for od in official_brand_domains:
        h = normalize_brand_domain_entry(od)
        if h and host_under_brand_domain(host, h):
            return [], 0.0

    return (["claimed_brand_hostname_mismatch"], 0.78)


def _merge_host_enrichment(
    host: str | None,
    signals: list[str],
    score: float,
    *,
    claimed_brand: str | None,
    official_hosts: list[str],
    matched_official: bool,
) -> tuple[list[str], float]:
    """Append suspicious-domain and brand–hostname mismatch cues; lift risk score."""
    if not host:
        return signals, score

    sus_sigs, sus_score = compute_domain_suspicion(host)
    for s in sus_sigs:
        if s not in signals:
            signals.append(s)
    score = min(1.0, max(score, sus_score * 0.88))

    mm_sigs, mm_score = collect_claimed_brand_hostname_mismatch(
        claimed_brand, host, official_hosts, matched_official=matched_official
    )
    for s in mm_sigs:
        if s not in signals:
            signals.append(s)
    if mm_sigs:
        score = min(1.0, max(score, mm_score))

    return signals, score


def normalize_official_domains(entries: list[str] | None) -> list[str]:
    """Turn URLs or hosts into bare hostnames, deduped, order preserved."""
    if not entries:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for e in entries:
        h = normalize_brand_domain_entry(e)
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def assess_brand_domain_risk(
    url: str,
    official_brand_domains: list[str],
    *,
    claimed_brand: str | None = None,
) -> BrandDomainRisk:
    """
    Extract domain from ``url``, compare to ``official_brand_domains``, run impersonation heuristics.

    ``official_brand_domains`` may contain ``https://www.brand.com/path`` or ``brand.com``.
    """
    hosts = normalize_official_domains(official_brand_domains)
    host = extract_host(url)
    reg = registrable_domain(host) if host else None

    if not hosts:
        signals = ["no_official_domains_configured"]
        score = 0.2
        matched = False
        if host:
            signals, score = _merge_host_enrichment(
                host,
                signals,
                score,
                claimed_brand=claimed_brand,
                official_hosts=[],
                matched_official=False,
            )
        level = _score_to_level(score)
        return BrandDomainRisk(
            risk_level=level,
            score=round(score, 4),
            page_host=host,
            page_registrable=reg or None,
            official_domains=[],
            matched_official=matched,
            signals=signals,
            claimed_brand=claimed_brand,
        )

    if not host:
        return BrandDomainRisk(
            risk_level="medium",
            score=0.45,
            page_host=None,
            page_registrable=None,
            official_domains=hosts,
            matched_official=False,
            signals=["invalid_or_missing_host"],
            claimed_brand=claimed_brand,
        )

    signals, score = collect_impersonation_signals(host, hosts)
    matched = "matches_official_domain" in signals or "matches_official_registrable" in signals
    signals, score = _merge_host_enrichment(
        host,
        list(signals),
        score,
        claimed_brand=claimed_brand,
        official_hosts=hosts,
        matched_official=matched,
    )

    level = _score_to_level(score)
    return BrandDomainRisk(
        risk_level=level,
        score=round(score, 4),
        page_host=host,
        page_registrable=reg or None,
        official_domains=hosts,
        matched_official=matched,
        signals=signals,
        claimed_brand=claimed_brand,
    )


def assess_url_against_wikidata_brand(
    url: str,
    brand_search_name: str,
    client: WikidataClient,
    *,
    use_cache: bool = True,
) -> BrandDomainRisk | None:
    """
    Resolve ``brand_search_name`` via :class:`~knowledge_module.wikidata.WikidataClient`,
    then run :func:`assess_brand_domain_risk` using returned official websites.

    Returns ``None`` if Wikidata returns no usable brand row.
    """
    try:
        info = client.get_brand_info(brand_search_name, use_cache=use_cache)
    except Exception as e:
        logger.warning("Wikidata brand lookup failed: %s", e)
        return None
    if not info or not info.official_websites:
        h = extract_host(url)
        signals = ["wikidata_no_official_website"]
        score = 0.15
        if h:
            signals, score = _merge_host_enrichment(
                h,
                signals,
                score,
                claimed_brand=info.label if info else brand_search_name,
                official_hosts=[],
                matched_official=False,
            )
        level = _score_to_level(score)
        return BrandDomainRisk(
            risk_level=level,
            score=round(score, 4),
            page_host=h,
            page_registrable=registrable_domain(h or "") or None,
            official_domains=[],
            matched_official=False,
            signals=signals,
            claimed_brand=info.label if info else brand_search_name,
        )
    risk = assess_brand_domain_risk(url, info.official_websites, claimed_brand=info.label)
    return risk
