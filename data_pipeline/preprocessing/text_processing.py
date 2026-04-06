"""
Visible text extraction from HTML and text cleaning for model training.
"""

import re
import unicodedata

from bs4 import BeautifulSoup

# Visible content tags to strip (same family as crawler DOM text)
_STRIP_TAGS = ("script", "style", "noscript", "iframe", "svg", "template")


def extract_visible_text_from_html(html: str) -> str:
    """
    Extract human-visible text from raw HTML (scripts/styles removed, whitespace collapsed).
    """
    if not html or not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return clean_and_normalize_text(text)


def clean_and_normalize_text(text: str) -> str:
    """
    Normalize Unicode, remove control characters, collapse whitespace.

    Preserves case (important for phishing cues like brand names).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
