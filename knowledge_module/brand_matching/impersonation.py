"""
Heuristic impersonation / typosquatting signals between page host and official brand domains.
"""

from __future__ import annotations

import re
from .domain import host_under_brand_domain, registrable_domain, split_sld_tld


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (small strings only; intended for domain labels)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins, delete, sub = prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


_CONFUSABLES = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "5": "s",
        "ɑ": "a",
        "ο": "o",
        "і": "i",
        "ｅ": "e",
        "а": "a",
        "о": "o",
        "р": "p",
    }
)


def normalize_for_homoglyph_check(label: str) -> str:
    """Map a few common confusables to ASCII letters (best-effort)."""
    return (label or "").lower().translate(_CONFUSABLES)


def collect_impersonation_signals(
    page_host: str,
    official_domains: list[str],
) -> tuple[list[str], float]:
    """
    Compare page host to official brand domains; return (signal strings, risk score in [0, 1]).

    Score is heuristic: 0 = clear official match; higher = more suspicious.
    """
    page_host = (page_host or "").lower().rstrip(".")
    if not page_host:
        return (["empty_host"], 0.5)

    canon = [d for d in (official_domains or []) if d]
    if not canon:
        return (["no_official_domains_configured"], 0.0)

    for od in canon:
        if host_under_brand_domain(page_host, od):
            return (["matches_official_domain"], 0.0)

    reg_page = registrable_domain(page_host)
    page_labels = page_host.split(".")

    signals: list[str] = []
    score = 0.0

    reg_officials = {registrable_domain(od): od for od in canon}
    reg_list = list(reg_officials.keys())

    # Same registrable domain but host didn't match subdomain rule (shouldn't happen often)
    if reg_page in reg_officials:
        return (["matches_official_registrable"], 0.0)

    sld_page, tld_page = split_sld_tld(reg_page)
    sld_page_l = sld_page.lower()
    sld_norm = normalize_for_homoglyph_check(sld_page)

    for reg_off in reg_list:
        if not reg_off:
            continue
        sld_off, tld_off = split_sld_tld(reg_off)
        sld_off_l = sld_off.lower()
        sld_off_n = normalize_for_homoglyph_check(sld_off)

        # Unicode / confusable spoof: normalized labels match but raw ASCII differs
        if sld_norm == sld_off_n and sld_page_l != sld_off_l:
            signals.append(f"homoglyph_or_confusable_match_to_{sld_off}")
            score = max(score, 0.85)

        # Typosquat on raw ASCII label (do not fold 1->l here, or paypa1 looks like paypal)
        dist = levenshtein(sld_page_l, sld_off_l)
        if sld_page_l != sld_off_l and dist <= 1 and min(len(sld_page_l), len(sld_off_l)) >= 3:
            signals.append(f"typosquat_edit_distance_{dist}_to_{sld_off}")
            score = max(score, 0.88 if dist == 1 else 0.72)
        elif sld_page_l != sld_off_l and dist == 2 and min(len(sld_page_l), len(sld_off_l)) >= 5:
            signals.append(f"typosquat_edit_distance_{dist}_to_{sld_off}")
            score = max(score, 0.55)

        # Same brand-like label, different TLD
        if sld_page_l == sld_off_l and tld_page and tld_off and tld_page != tld_off:
            signals.append(f"same_brand_label_different_tld_{tld_page}_vs_{tld_off}")
            score = max(score, 0.62)

        # Hyphenated brand keyword (paypal-login, secure-paypal)
        if sld_off_l:
            if f"{sld_off_l}-" in sld_page_l or f"-{sld_off_l}" in sld_page_l:
                signals.append(f"hyphen_brand_keyword_{sld_off}")
                score = max(score, 0.78)
            if re.match(rf"^{re.escape(sld_off_l)}[.-].+", sld_page_l) and sld_page_l != sld_off_l:
                signals.append(f"brand_prefix_with_separator_{sld_off}")
                score = max(score, 0.68)

        # Leading label mimics brand but registrable is unrelated (paypal.evil.com)
        if len(page_labels) >= 3:
            first = page_labels[0].lower()
            if first == sld_off_l and reg_page != reg_off and not reg_page.endswith(reg_off):
                signals.append(f"leading_label_mimics_brand_{sld_off}")
                score = max(score, 0.82)

        # Embedded brand as substring in registrable (not exact match)
        if len(sld_off_l) >= 4 and sld_off_l in sld_page_l and sld_page_l != sld_off_l:
            signals.append(f"brand_substring_in_domain_{sld_off}")
            score = max(score, 0.58)

    if not signals:
        signals.append("no_official_match_no_strong_impersonation_pattern")

    return (signals, min(1.0, score))
