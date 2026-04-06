"""Unit tests: cross-modal brand consistency heuristics (no torch)."""

from __future__ import annotations

from knowledge_module.cross_modal_consistency import (
    compute_cross_modal_consistency,
    extract_brand_candidates_from_text,
)


class TestExtractBrandFromText:
    def test_login_phrase(self) -> None:
        t = "Please sign in to PayPal to continue with your payment."
        c = extract_brand_candidates_from_text(t)
        assert any("paypal" in x.lower() for x in c)

    def test_copyright(self) -> None:
        t = "© 2024 Microsoft Corporation. All rights reserved."
        c = extract_brand_candidates_from_text(t)
        assert any("microsoft" in x.lower() for x in c)


class TestComputeConsistency:
    def test_aligned_text_and_domain(self) -> None:
        cm = compute_cross_modal_consistency(
            page_text="Sign in to PayPal",
            screenshot_image=None,
            page_url="https://www.paypal.com/signin",
            reference_brands=None,
        )
        assert cm.consistency_score >= 0.5
        assert cm.domain_registrable == "paypal.com"

    def test_mismatch_brand_vs_domain_lowers_score(self) -> None:
        cm = compute_cross_modal_consistency(
            page_text="Welcome to Chase Online Banking secure login",
            screenshot_image=None,
            page_url="https://totally-unrelated-login.xyz/secure",
            reference_brands=None,
        )
        assert cm.consistency_score < 0.72
