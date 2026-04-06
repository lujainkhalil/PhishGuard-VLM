"""Unit tests: inference aggregation (no crawl / no torch)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from inference.aggregator import (
    ModelPrediction,
    aggregate_signals,
    knowledge_to_phish_prior,
)
from inference.knowledge_fusion import KnowledgeFusionConfig


@dataclass
class FakeKnowledge:
    risk_level: str
    score: float
    matched_official: bool
    signals: list[str] = field(default_factory=list)


class TestKnowledgeToPhishPrior:
    def test_none_is_neutral(self) -> None:
        p = knowledge_to_phish_prior(None)
        assert p == 0.5

    def test_matched_official_low(self) -> None:
        k = FakeKnowledge("low", 0.1, True, [])
        p = knowledge_to_phish_prior(k)
        assert p < 0.25

    def test_suspicious_signal_boosts_prior_when_not_matched(self) -> None:
        base = FakeKnowledge("low", 0.1, False, ["no_official_match_no_strong_impersonation_pattern"])
        boosted = FakeKnowledge("low", 0.1, False, ["suspicious_tld_xyz"])
        assert knowledge_to_phish_prior(boosted) > knowledge_to_phish_prior(base)


class TestAggregateSignals:
    def test_model_only(self) -> None:
        v = aggregate_signals(ModelPrediction(phishing_probability=0.9), None)
        assert v.label == 1
        assert v.knowledge_used is False
        assert v.confidence > 0
        assert "model" in v.explanation.lower() or "vision-language" in v.explanation.lower()

    def test_high_domain_risk_increases_phish(self) -> None:
        k = FakeKnowledge("high", 0.9, False, ["typosquat_candidate"])
        v_low = aggregate_signals(ModelPrediction(phishing_probability=0.45), None)
        v_fused = aggregate_signals(ModelPrediction(phishing_probability=0.45), k)
        assert v_fused.phishing_probability >= v_low.phishing_probability
        assert v_fused.knowledge_used is True

    def test_benign_model_and_official_domain(self) -> None:
        k = FakeKnowledge("low", 0.1, True, ["matches_official_domain"])
        v = aggregate_signals(ModelPrediction(phishing_probability=0.15), k)
        assert v.label == 0
        assert v.knowledge_phish_prior is not None

    def test_low_cross_modal_consistency_nudges_phish(self) -> None:
        base = aggregate_signals(ModelPrediction(phishing_probability=0.48), None)
        cm = SimpleNamespace(
            consistency_score=0.2,
            ocr_used=False,
            to_dict=lambda: {"consistency_score": 0.2},
        )
        v = aggregate_signals(ModelPrediction(phishing_probability=0.48), None, cross_modal=cm)
        assert v.phishing_probability >= base.phishing_probability
        assert v.cross_modal_consistency == pytest.approx(0.2)
        assert v.cross_modal is not None

    def test_higher_knowledge_weight_multiplier_favors_knowledge_prior(self) -> None:
        k = FakeKnowledge("high", 0.95, False, ["typosquat_edit_distance_1_to_paypal.com"])
        low_k = KnowledgeFusionConfig(knowledge_weight_multiplier=0.5)
        high_k = KnowledgeFusionConfig(knowledge_weight_multiplier=1.6)
        v_low = aggregate_signals(ModelPrediction(phishing_probability=0.45), k, fusion=low_k)
        v_high = aggregate_signals(ModelPrediction(phishing_probability=0.45), k, fusion=high_k)
        assert v_high.phishing_probability > v_low.phishing_probability
