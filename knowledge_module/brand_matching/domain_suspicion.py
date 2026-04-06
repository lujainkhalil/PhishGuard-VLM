"""
Heuristic signals for suspicious hostnames (risky TLDs, IP literals, punycode, length).

Used to raise :class:`~knowledge_module.brand_matching.matcher.BrandDomainRisk` scores even when
no official brand list is available or before impersonation rules fire.
"""

from __future__ import annotations

import re
from typing import Final

# Commonly abused or ultra-cheap TLDs in phishing (not exhaustive).
_RISKY_TLDS: Final[frozenset[str]] = frozenset(
    {
        "xyz",
        "top",
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "click",
        "download",
        "loan",
        "racing",
        "review",
        "zip",
        "country",
        "kim",
        "work",
        "party",
        "science",
        "trade",
        "accountant",
        "bid",
        "cricket",
        "date",
        "faith",
        "men",
        "stream",
        "win",
        "gdn",
        "buzz",
        "cam",
        "bar",
        "rest",
        "pw",
        "cc",
        "surf",
        "space",
        "site",
        "online",
        "website",
        "fun",
        "icu",
    }
)

_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def compute_domain_suspicion(host: str) -> tuple[list[str], float]:
    """
    Return ``(signal_strings, risk_score)`` with score in ``[0, 1]``.

    Higher scores indicate more suspicious hosting patterns (not proof of phishing).
    """
    host = (host or "").lower().strip().rstrip(".")
    if not host:
        return (["empty_host_suspicion"], 0.35)

    signals: list[str] = []
    score = 0.0

    host_no_port = host.split(":")[0]

    if _IPV4_RE.match(host_no_port):
        signals.append("host_is_ip_literal")
        score = max(score, 0.58)

    if "xn--" in host:
        signals.append("punycode_in_hostname")
        score = max(score, 0.36)

    labels = [x for x in host_no_port.split(".") if x]
    if len(labels) >= 2:
        tld = labels[-1]
        if tld in _RISKY_TLDS:
            signals.append(f"suspicious_tld_{tld}")
            score = max(score, 0.44)
    elif len(labels) == 1 and labels[0] not in ("localhost",):
        signals.append("single_label_hostname")
        score = max(score, 0.28)

    if len(host_no_port) > 72:
        signals.append("very_long_hostname")
        score = max(score, 0.24)

    # Long numeric-only label (e.g. dga-style) inside hostname
    for lab in labels:
        if len(lab) >= 18 and lab.isdigit():
            signals.append("long_numeric_label_in_hostname")
            score = max(score, 0.32)
            break

    if not signals:
        signals.append("no_suspicious_domain_pattern")

    return (signals, min(1.0, score))
