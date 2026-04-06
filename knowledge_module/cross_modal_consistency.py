"""
Cross-modal brand consistency: compare brand-like cues from page text, screenshot (optional OCR),
and the URL registrable domain. Produces a consistency score for fusion with the VLM verdict.

Higher ``consistency_score`` means text and image (when OCR is available) agree with each other
and with the hostname; lower scores flag likely impersonation (e.g. brand in UI, unrelated domain).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .brand_matching.domain import extract_host, registrable_domain, split_sld_tld

logger = logging.getLogger(__name__)

# Phrases that often precede a spoofed brand name on login pages
_BRAND_AFTER_LOGIN = re.compile(
    r"(?i)(?:sign\s*in(?:\s+to)?|log\s*in(?:\s+to)?|welcome\s+to|continue\s+to|"
    r"verify\s+your|secure\s+login\s+for)\s+([A-Za-z0-9][A-Za-z0-9.&'\-\s]{0,40}?)(?:\s*$|[.,!\n]|\s+with|\s+to\s)",
)
_COPYRIGHT_LINE = re.compile(
    r"(?i)©\s*(?:\d{4}\s*)?([A-Za-z][A-Za-z0-9,&'\-\s]{0,45}?)(?:\.|$|\n|©)",
)
_POWERED_BY = re.compile(r"(?i)powered\s+by\s+([A-Za-z][A-Za-z0-9.&'\-\s]{0,35}?)(?:\s|$|[.,!])")

_STOP = frozenset(
    """
    the a an your you our my this that these those here there to from for with without
    and or not is are was were be been being as at on in of if it we us they them their
    account page site web online secure security password email please click continue
    """.split()
)


def _norm_key(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _fuzzy_ratio(a: str, b: str) -> float:
    a, b = _norm_key(a), _norm_key(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return max(
            SequenceMatcher(None, a, b).ratio(),
            min(len(a), len(b)) / max(len(a), len(b)),
        )
    return SequenceMatcher(None, a, b).ratio()


def _clean_candidate(raw: str) -> str | None:
    s = (raw or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) < 2 or len(s) > 48:
        return None
    low = s.lower()
    words = low.split()
    if all(w in _STOP or len(w) < 2 for w in words):
        return None
    if low in _STOP:
        return None
    return s


def extract_brand_candidates_from_text(text: str, *, max_candidates: int = 10) -> list[str]:
    """
    Heuristic brand-like strings from visible page text (login phrases, copyright, powered-by).
    """
    if not text or not isinstance(text, str):
        return []
    seen: set[str] = set()
    out: list[str] = []

    def push(m: str | None) -> None:
        if m is None or len(out) >= max_candidates:
            return
        c = _clean_candidate(m)
        if c is None:
            return
        k = _norm_key(c)
        if not k or k in seen:
            return
        seen.add(k)
        out.append(c)

    for rx in (_BRAND_AFTER_LOGIN, _COPYRIGHT_LINE, _POWERED_BY):
        for m in rx.finditer(text[:8000]):
            push(m.group(1).strip())

    # Title-like runs (conservative): 2–3 capitalized tokens
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text[:4000]):
        push(m.group(1))

    return out[:max_candidates]


def extract_text_from_screenshot_pil(image: Any) -> str:
    """
    Optional OCR over the screenshot. Requires ``pytesseract`` and a system Tesseract binary.

    Returns empty string if OCR is unavailable or fails.
    """
    try:
        import pytesseract
        from PIL import Image

        if not isinstance(image, Image.Image):
            return ""
        im = image.convert("RGB")
        try:
            txt = pytesseract.image_to_string(im, timeout=25)
        except TypeError:
            txt = pytesseract.image_to_string(im)
        from data_pipeline.preprocessing.text_processing import clean_and_normalize_text

        return clean_and_normalize_text(txt)
    except Exception as e:
        logger.debug("Screenshot OCR skipped or failed: %s", e)
        return ""


def extract_brand_candidates_from_image(image: Any, *, max_candidates: int = 8) -> tuple[list[str], bool]:
    """
    Run OCR (if available) and reuse :func:`extract_brand_candidates_from_text`.

    Returns ``(candidates, ocr_ran_successfully)`` where the second flag is True only if
    pytesseract returned non-empty text.
    """
    ocr_text = extract_text_from_screenshot_pil(image)
    if not ocr_text:
        return [], False
    return extract_brand_candidates_from_text(ocr_text, max_candidates=max_candidates), True


def _domain_brand_tokens(url: str) -> tuple[str | None, list[str]]:
    """Registrable eTLD+1 and token hints from the host (SLD, without common prefixes)."""
    host = extract_host(url)
    if not host:
        return None, []
    reg = registrable_domain(host)
    if not reg:
        return None, []
    sld, _tld = split_sld_tld(reg)
    hints: list[str] = []
    if sld:
        hints.append(sld)
        for part in re.split(r"[\-_]+", sld):
            if len(part) >= 3 and part not in ("www", "mail", "login", "signin", "secure", "account"):
                hints.append(part)
    return reg, list(dict.fromkeys(hints))


def _pool_modality_score(text_brands: list[str], image_brands: list[str]) -> float:
    if not text_brands and not image_brands:
        return 0.55
    if not text_brands or not image_brands:
        return 0.52
    best = 0.0
    for t in text_brands:
        for i in image_brands:
            best = max(best, _fuzzy_ratio(t, i))
    if best >= 0.88:
        return 1.0
    if best <= 0.32:
        return 0.22
    return 0.22 + (best - 0.32) * (1.0 - 0.22) / (0.88 - 0.32)


def _pool_domain_score(
    candidates: list[str],
    domain_hints: list[str],
) -> float:
    if not domain_hints:
        return 0.5
    if not candidates:
        return 0.5
    best = 0.0
    for c in candidates:
        for h in domain_hints:
            best = max(best, _fuzzy_ratio(c, h))
    if best >= 0.82:
        return 1.0
    if best <= 0.38:
        return 0.25
    return 0.25 + (best - 0.38) * (1.0 - 0.25) / (0.82 - 0.38)


def _pool_reference_score(candidates: list[str], references: list[str]) -> float:
    if not references:
        return 0.55
    if not candidates:
        return 0.45
    best = 0.0
    for c in candidates:
        for r in references:
            best = max(best, _fuzzy_ratio(c, r))
    if best >= 0.85:
        return 1.0
    if best <= 0.4:
        return 0.28
    return 0.28 + (best - 0.4) * (1.0 - 0.28) / (0.85 - 0.4)


@dataclass
class CrossModalConsistency:
    """
    ``consistency_score`` in [0, 1]: higher when text/image brand cues align with each other
    and with the URL (and optional reference brand names).
    """

    consistency_score: float
    text_brands: list[str] = field(default_factory=list)
    image_brands: list[str] = field(default_factory=list)
    domain_registrable: str | None = None
    ocr_used: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistency_score": round(float(self.consistency_score), 4),
            "text_brands": list(self.text_brands),
            "image_brands": list(self.image_brands),
            "domain_registrable": self.domain_registrable,
            "ocr_used": self.ocr_used,
            "notes": list(self.notes),
        }


def compute_cross_modal_consistency(
    *,
    page_text: str,
    screenshot_image: Any | None,
    page_url: str,
    reference_brands: list[str] | None = None,
    max_text_brands: int = 10,
    max_image_brands: int = 8,
) -> CrossModalConsistency:
    """
    Extract brand-like cues from ``page_text`` and optionally from ``screenshot_image`` (OCR),
    compare to the registrable domain of ``page_url`` and to ``reference_brands`` (e.g. user hint
    or Wikidata label).
    """
    notes: list[str] = []
    text_brands = extract_brand_candidates_from_text(page_text, max_candidates=max_text_brands)
    image_brands: list[str] = []
    ocr_used = False

    if screenshot_image is not None:
        image_brands, ocr_used = extract_brand_candidates_from_image(
            screenshot_image, max_candidates=max_image_brands
        )
        if ocr_used and not image_brands:
            notes.append("ocr_no_brand_candidates")
        elif not ocr_used:
            notes.append("ocr_unavailable_or_empty")
    else:
        notes.append("no_screenshot_for_image_brands")

    reg, domain_hints = _domain_brand_tokens(page_url)
    if reg:
        notes.append(f"domain_registrable:{reg}")
    else:
        notes.append("domain_unparsed")

    refs = [r.strip() for r in (reference_brands or []) if r and str(r).strip()]
    refs = list(dict.fromkeys(refs))[:6]

    all_cand = list(dict.fromkeys(text_brands + image_brands))

    modality = _pool_modality_score(text_brands, image_brands)
    domain_s = _pool_domain_score(all_cand, domain_hints)
    ref_s = _pool_reference_score(all_cand, refs)

    if refs:
        score = 0.32 * modality + 0.38 * domain_s + 0.30 * ref_s
    else:
        score = 0.45 * modality + 0.55 * domain_s

    score = max(0.0, min(1.0, float(score)))

    if modality < 0.35 and text_brands and image_brands:
        notes.append("text_image_brand_mismatch")
    if domain_s < 0.35 and all_cand:
        notes.append("brand_domain_mismatch")

    return CrossModalConsistency(
        consistency_score=round(score, 4),
        text_brands=text_brands,
        image_brands=image_brands,
        domain_registrable=reg,
        ocr_used=ocr_used,
        notes=notes,
    )
