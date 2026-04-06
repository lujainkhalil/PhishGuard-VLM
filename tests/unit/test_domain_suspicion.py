"""Unit tests: suspicious-domain heuristics."""

from __future__ import annotations

from knowledge_module.brand_matching.domain_suspicion import compute_domain_suspicion


class TestComputeDomainSuspicion:
    def test_risky_tld(self) -> None:
        sigs, score = compute_domain_suspicion("login-secure.phishbank.xyz")
        assert any(s.startswith("suspicious_tld_") for s in sigs)
        assert score >= 0.4

    def test_ip_literal(self) -> None:
        sigs, score = compute_domain_suspicion("192.168.1.1")
        assert "host_is_ip_literal" in sigs
        assert score >= 0.5

    def test_benign_com_no_strong_signal(self) -> None:
        sigs, score = compute_domain_suspicion("www.paypal.com")
        assert score < 0.35
