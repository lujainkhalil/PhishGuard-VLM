"""Unit tests: brand matcher + domain suspicion + claim mismatch."""

from __future__ import annotations

from knowledge_module.brand_matching.matcher import assess_brand_domain_risk


class TestAssessBrandDomainRiskEnrichment:
    def test_claimed_brand_mismatch_and_suspicious_tld(self) -> None:
        r = assess_brand_domain_risk(
            "https://steal-creds.phish-login.xyz/secure",
            ["paypal.com", "www.paypal.com"],
            claimed_brand="PayPal",
        )
        assert "claimed_brand_hostname_mismatch" in r.signals
        assert any(s.startswith("suspicious_tld_") for s in r.signals)
        assert r.matched_official is False
        assert r.score >= 0.5

    def test_official_match_no_mismatch(self) -> None:
        r = assess_brand_domain_risk(
            "https://www.paypal.com/signin",
            ["paypal.com"],
            claimed_brand="PayPal",
        )
        assert r.matched_official is True
        assert "claimed_brand_hostname_mismatch" not in r.signals
