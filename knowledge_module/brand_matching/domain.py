"""
Extract and normalize host / registrable domain from URLs for brand comparison.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Second-level ccTLD-style suffixes (eTLD+1 is last three labels).
_MULTI_LABEL_SUFFIXES = frozenset(
    {
        "co.uk",
        "com.au",
        "co.jp",
        "co.nz",
        "com.br",
        "com.mx",
        "co.in",
        "com.ar",
        "co.za",
        "com.sg",
        "com.hk",
        "co.kr",
        "com.tw",
        "co.id",
        "com.co",
        "ac.uk",
        "gov.uk",
        "ne.jp",
        "or.jp",
    }
)


def extract_host(url: str) -> str | None:
    """
    Return lowercase hostname without port, or None if missing/invalid.

    Accepts bare hosts; adds a scheme when absent so ``urlparse`` behaves.
    """
    if not url or not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if not host:
            return None
        return host
    except Exception:
        return None


def registrable_domain(host: str) -> str:
    """
    Approximate registrable domain (eTLD+1) without the Public Suffix List.

    Handles common ``*.co.uk``-style suffixes via :data:`_MULTI_LABEL_SUFFIXES`.
    """
    host = (host or "").lower().strip().rstrip(".")
    if not host:
        return ""
    # Strip IPv6 brackets if any
    if host.startswith("["):
        return host
    labels = [x for x in host.split(".") if x]
    if len(labels) < 2:
        return host
    if len(labels) >= 3:
        tail3 = ".".join(labels[-2:])
        if tail3 in _MULTI_LABEL_SUFFIXES:
            return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def host_under_brand_domain(host: str, brand_domain: str) -> bool:
    """
    True if ``host`` equals ``brand_domain`` or is a proper subdomain of it
    (e.g. ``www.paypal.com`` under ``paypal.com``).
    """
    host = (host or "").lower().rstrip(".")
    brand = (brand_domain or "").lower().rstrip(".")
    if not host or not brand:
        return False
    if host == brand:
        return True
    return host.endswith("." + brand)


def normalize_brand_domain_entry(entry: str) -> str | None:
    """Normalize a config/Wikidata URL or host string to a bare hostname."""
    if not entry or not isinstance(entry, str):
        return None
    s = entry.strip().lower()
    if not s:
        return None
    if "://" in s:
        try:
            netloc = urlparse(s).netloc
            if "@" in netloc:
                netloc = netloc.split("@")[-1]
            host = netloc.split(":")[0]
            return host or None
        except Exception:
            return None
    s = s.split("/")[0].split(":")[0]
    return s or None


def split_sld_tld(reg: str) -> tuple[str, str]:
    """Split registrable ``foo.com`` -> (``foo``, ``com``); ``foo.co.uk`` treated as one SLD label group."""
    reg = (reg or "").lower().rstrip(".")
    if not reg:
        return "", ""
    parts = reg.split(".")
    if len(parts) < 2:
        return reg, ""
    sld = parts[0]
    tld = ".".join(parts[1:])
    return sld, tld
