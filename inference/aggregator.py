"""
Fuse VLM phishing scores with knowledge-module (e.g. brand–domain) signals.

Produces a final label, calibrated-style confidence, and a short natural-language explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .knowledge_fusion import DEFAULT_KNOWLEDGE_FUSION, KnowledgeFusionConfig


@runtime_checkable
class KnowledgeSignalLike(Protocol):
    """Duck type for :class:`~knowledge_module.brand_matching.BrandDomainRisk` and test doubles."""

    risk_level: str
    score: float
    matched_official: bool
    signals: list[str]


@dataclass
class ModelPrediction:
    """Classifier output (probabilities in [0, 1])."""

    phishing_probability: float
    """Estimated P(phishing) from the model (e.g. sigmoid(logit))."""
    label_hint: int | None = None
    """Optional discrete hint {0, 1} from the model’s own threshold."""


@dataclass
class AggregatedVerdict:
    """Fused decision for downstream APIs and audit logs."""

    label: int
    """0 = benign, 1 = phishing (see ``phish_threshold``)."""
    confidence: float
    """Strength of the decision in [0, 1]; high when probability is far from 0.5."""
    explanation: str
    """Human-readable rationale."""
    phishing_probability: float
    """Fused P(phishing) after combining signals."""
    model_probability: float
    knowledge_phish_prior: float | None
    """Effective P(phishing) contribution from knowledge, or None if absent."""
    knowledge_used: bool = False
    notes: list[str] = field(default_factory=list)
    """Structured bullets mirroring ``explanation`` (for JSON)."""
    cross_modal_consistency: float | None = None
    """[0,1] text/image/domain brand alignment; None if not computed."""
    cross_modal: dict[str, Any] | None = None
    """Structured cross-modal diagnostics (brands, OCR flag, notes)."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "label_name": "phishing" if self.label == 1 else "benign",
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "phishing_probability": round(self.phishing_probability, 4),
            "model_probability": round(self.model_probability, 4),
            "knowledge_phish_prior": None
            if self.knowledge_phish_prior is None
            else round(self.knowledge_phish_prior, 4),
            "knowledge_used": self.knowledge_used,
            "notes": list(self.notes),
        }
        if self.cross_modal_consistency is not None:
            d["cross_modal_consistency"] = round(self.cross_modal_consistency, 4)
        if self.cross_modal is not None:
            d["cross_modal"] = dict(self.cross_modal)
        return d


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _knowledge_signal_prior_adjustment(signals: list[str]) -> float:
    """Extra phishing prior mass from structured domain signals (capped)."""
    delta = 0.0
    for s in signals:
        if s.startswith("suspicious_tld_") or s in ("host_is_ip_literal", "punycode_in_hostname"):
            delta += 0.038
        if s == "claimed_brand_hostname_mismatch":
            delta += 0.12
        if "typosquat" in s or "homoglyph" in s or "leading_label_mimics_brand" in s:
            delta += 0.028
        if "brand_prefix_with_separator" in s or "hyphen_brand_keyword" in s:
            delta += 0.022
    return min(0.24, delta)


def knowledge_to_phish_prior(knowledge: KnowledgeSignalLike | None) -> float:
    """
    Map :class:`~knowledge_module.brand_matching.BrandDomainRisk` (or duck-typed) to P(phish | knowledge).

    Official domain match → low prior; impersonation signals → elevated prior; missing fields default cautiously.
    """
    if knowledge is None:
        return 0.5

    matched = bool(getattr(knowledge, "matched_official", False))
    level = (getattr(knowledge, "risk_level", "low") or "low").lower()
    score = _clamp01(float(getattr(knowledge, "score", 0.0)))
    sigs = list(getattr(knowledge, "signals", None) or [])

    if matched:
        # Legitimate alignment with known corporate sites
        return _clamp01(0.08 + 0.12 * score)

    if level == "high":
        base = _clamp01(0.62 + 0.33 * score)
    elif level == "medium":
        base = _clamp01(0.48 + 0.35 * score)
    elif level == "low":
        base = _clamp01(0.35 + 0.25 * score)
    else:
        base = _clamp01(0.28 + 0.2 * score)

    return _clamp01(base + _knowledge_signal_prior_adjustment(sigs))


def _dynamic_weights(knowledge: KnowledgeSignalLike | None, p_k: float) -> tuple[float, float, float]:
    """
    Return (w_model, w_knowledge, w_neutral) summing to 1.0.

    Shift mass toward knowledge when impersonation is strong or when official domain matches
    (so the model is not the only voice on brand integrity).
    """
    w_m, w_k, w_n = 0.55, 0.35, 0.10
    if knowledge is None:
        return 0.85, 0.0, 0.15

    matched = bool(getattr(knowledge, "matched_official", False))
    level = (getattr(knowledge, "risk_level", "low") or "low").lower()

    if matched:
        # Trust domain evidence; still keep model in the loop
        return 0.50, 0.25, 0.25
    if level == "high" or p_k >= 0.75:
        return 0.42, 0.48, 0.10
    if level == "medium":
        return 0.48, 0.40, 0.12
    return w_m, w_k, w_n


def _apply_knowledge_fusion_weights(
    w_m: float,
    w_k: float,
    w_n: float,
    cfg: KnowledgeFusionConfig,
) -> tuple[float, float, float]:
    """Scale knowledge weight, renormalize, then apply caps/floors."""
    w_k = w_k * cfg.knowledge_weight_multiplier
    s = w_m + w_k + w_n
    if s <= 0:
        return 0.55, 0.35, 0.10
    w_m, w_k, w_n = w_m / s, w_k / s, w_n / s

    if w_k > cfg.max_knowledge_blend_weight:
        excess = w_k - cfg.max_knowledge_blend_weight
        w_k = cfg.max_knowledge_blend_weight
        w_m += excess * 0.78
        w_n += excess * 0.22
        s2 = w_m + w_k + w_n
        w_m, w_k, w_n = w_m / s2, w_k / s2, w_n / s2

    if w_m < cfg.min_model_blend_weight:
        deficit = cfg.min_model_blend_weight - w_m
        take_k = min(w_k, deficit * 0.7)
        take_n = min(w_n, deficit - take_k)
        w_m += take_k + take_n
        w_k -= take_k
        w_n -= take_n
        s3 = w_m + w_k + w_n
        if s3 > 0:
            w_m, w_k, w_n = w_m / s3, w_k / s3, w_n / s3

    return w_m, w_k, w_n


def _confidence_from_probability(p: float) -> float:
    """Distance from decision boundary; symmetric around 0.5."""
    p = _clamp01(p)
    return abs(p - 0.5) * 2.0


def _build_explanation(
    p_m: float,
    p_final: float,
    label: int,
    knowledge: KnowledgeSignalLike | None,
    p_k: float | None,
    notes: list[str],
    cross_modal: Any | None = None,
) -> str:
    parts: list[str] = []
    parts.append(f"The vision-language model scored this page at {_pct(p_m)} likely phishing.")
    if knowledge is not None and p_k is not None:
        lvl = getattr(knowledge, "risk_level", "?")
        parts.append(
            f"Brand and domain analysis suggests {_pct(p_k)} phishing prior (risk level: {lvl})."
        )
        if getattr(knowledge, "matched_official", False):
            parts.append("The hostname appears consistent with an official brand web presence.")
        sigs = getattr(knowledge, "signals", None) or []
        impostor_like = any(
            "typosquat" in s or "homoglyph" in s or "leading_label_mimics_brand" in s for s in sigs
        )
        if impostor_like:
            parts.append("Several domain signals resemble impersonation or typosquatting patterns.")
        if any(s.startswith("suspicious_tld_") or s == "host_is_ip_literal" for s in sigs):
            parts.append("The hostname shows suspicious registration patterns (e.g. risky TLD or IP literal).")
        if any(s == "claimed_brand_hostname_mismatch" for s in sigs):
            parts.append("The claimed brand does not match the page hostname against known official domains.")
    elif knowledge is None:
        parts.append("No knowledge-module signal was supplied; decision relies on the model only.")

    if cross_modal is not None:
        s = _clamp01(float(getattr(cross_modal, "consistency_score", 0.5)))
        parts.append(
            f"Cross-modal brand consistency (text vs image vs domain) is {_pct(s)} aligned."
        )
        if getattr(cross_modal, "ocr_used", False):
            parts.append("Screenshot OCR contributed image-side brand cues.")
        else:
            parts.append("Image-side brand cues used text extraction only (OCR unavailable or empty).")

    parts.append(f"After fusion, estimated phishing probability is {_pct(p_final)} ({'phishing' if label == 1 else 'benign'}).")
    if notes:
        parts.append("Details: " + "; ".join(notes[:5]) + ("…" if len(notes) > 5 else ""))
    return " ".join(parts)


def _pct(x: float) -> str:
    return f"{100.0 * _clamp01(x):.0f}%"


def aggregate_signals(
    model: ModelPrediction,
    knowledge: KnowledgeSignalLike | None = None,
    *,
    cross_modal: Any | None = None,
    fusion: KnowledgeFusionConfig | None = None,
    phish_threshold: float = 0.5,
) -> AggregatedVerdict:
    """
    Combine ``model.phishing_probability`` with optional ``BrandDomainRisk`` (or compatible object).

    Strategy (compact):

    1. Convert knowledge to a phishing **prior** ``p_k`` in [0, 1].
    2. Choose blend weights: more weight on knowledge when impersonation is high or when the
       domain clearly matches an official brand. Optional :class:`KnowledgeFusionConfig` scales
       and caps the knowledge weight.
    3. Fused probability ``p_final = w_m * p_m + w_k * p_k + w_n * 0.5``.
    4. **Agreement boost**: if model and knowledge both strongly agree on phishing or benign,
       nudge ``p_final`` slightly toward that pole (higher confidence).
    5. Optional **cross-modal consistency** (text vs image brand cues vs domain): low alignment
       nudges ``p_final`` toward phishing; strong alignment nudges slightly toward benign.
    6. Label = ``1`` iff ``p_final >= phish_threshold``.

    This is heuristic (not learned calibration); tune weights for your deployment.
    """
    p_m = _clamp01(model.phishing_probability)
    notes: list[str] = []
    fusion_cfg = fusion if fusion is not None else DEFAULT_KNOWLEDGE_FUSION

    if knowledge is not None:
        p_k = knowledge_to_phish_prior(knowledge)
        w_m, w_k, w_n = _dynamic_weights(knowledge, p_k)
        w_m, w_k, w_n = _apply_knowledge_fusion_weights(w_m, w_k, w_n, fusion_cfg)
        p_final = w_m * p_m + w_k * p_k + w_n * 0.5
        knowledge_used = True

        # Agreement nudge
        both_phish = p_m >= 0.65 and p_k >= 0.6
        both_benign = p_m <= 0.35 and p_k <= 0.35
        if both_phish:
            p_final = _clamp01(p_final + 0.06)
            notes.append("model_and_knowledge_agree_phishing")
        elif both_benign:
            p_final = _clamp01(p_final - 0.06)
            notes.append("model_and_knowledge_agree_benign")

        # Strong conflict: very confident model benign but high domain risk
        if p_m <= 0.3 and p_k >= 0.7:
            p_final = _clamp01(0.55 * p_final + 0.45 * p_k)
            notes.append("elevated_phish_due_to_domain_knowledge_despite_low_model_score")
        # Model very phish but official domain
        if p_m >= 0.75 and getattr(knowledge, "matched_official", False):
            p_final = _clamp01(0.62 * p_final + 0.38 * p_k)
            notes.append("tempered_model_phish_due_to_official_domain_signal")
    else:
        p_k = None
        p_final = 0.88 * p_m + 0.12 * 0.5
        knowledge_used = False

    cross_modal_consistency: float | None = None
    cross_modal_dict: dict[str, Any] | None = None
    if cross_modal is not None:
        cross_modal_consistency = _clamp01(float(getattr(cross_modal, "consistency_score", 0.5)))
        to_dict_fn = getattr(cross_modal, "to_dict", None)
        if callable(to_dict_fn):
            cross_modal_dict = to_dict_fn()
        s = cross_modal_consistency
        if s <= 0.33:
            p_final = _clamp01(p_final + 0.06)
            notes.append("low_cross_modal_brand_consistency")
        elif s <= 0.45:
            p_final = _clamp01(p_final + 0.03)
            notes.append("moderate_cross_modal_inconsistency")
        elif s >= 0.85:
            p_final = _clamp01(p_final - 0.03)
            notes.append("high_cross_modal_brand_consistency")

        if knowledge is not None and s <= 0.38:
            p_k2 = knowledge_to_phish_prior(knowledge)
            if p_k2 >= 0.55:
                p_final = _clamp01(0.88 * p_final + 0.12 * p_k2)
                notes.append("cross_modal_weak_plus_domain_knowledge")

    label = 1 if p_final >= phish_threshold else 0
    conf = _confidence_from_probability(p_final)

    if knowledge is not None:
        sigs = list(getattr(knowledge, "signals", None) or [])[:4]
        for s in sigs:
            notes.append(f"domain:{s}")

    explanation = _build_explanation(p_m, p_final, label, knowledge, p_k, notes, cross_modal)

    return AggregatedVerdict(
        label=label,
        confidence=round(conf, 4),
        explanation=explanation,
        phishing_probability=round(p_final, 4),
        model_probability=round(p_m, 4),
        knowledge_phish_prior=None if p_k is None else round(p_k, 4),
        knowledge_used=knowledge_used,
        notes=notes,
        cross_modal_consistency=cross_modal_consistency,
        cross_modal=cross_modal_dict,
    )
