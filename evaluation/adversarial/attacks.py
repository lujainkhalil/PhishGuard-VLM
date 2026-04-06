"""
Simulated adversarial perturbations on webpage text and screenshots (robustness probes).

Used for **adversarial evaluation** (measure drop vs. clean inputs) and optionally at **train
time** via :mod:`models.training.adversarial_augment` to improve robustness. Production systems
should still aim for **minimal degradation** on clean pages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ZWSP = "\u200b"
NBSP = "\u00a0"


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def apply_html_obfuscation(text: str, level: str = "medium", rng: np.random.Generator | None = None) -> str:
    """
    Insert invisible / noisy tokens typical of HTML-text extraction quirks (ZWSP, faux comments, NBSP).

    ``level``: ``light`` | ``medium`` | ``heavy``.
    """
    if not text:
        return text
    rng = rng or _rng(None)
    level = (level or "medium").lower()
    prob_zw = {"light": 0.06, "medium": 0.14, "heavy": 0.26}.get(level, 0.14)
    words = text.split()
    if not words:
        return text
    chunks: list[str] = []
    for w in words:
        chunks.append(w)
        if rng.random() < prob_zw:
            chunks.append(ZWSP * int(rng.integers(1, 4)))
    s = " ".join(chunks)
    if level != "light" and rng.random() < (0.12 if level == "medium" else 0.22):
        pos = int(rng.integers(0, max(1, len(s))))
        s = s[:pos] + " <!--obf-->" + s[pos:]
    if level == "heavy" and rng.random() < 0.18:
        s = s.replace(" ", NBSP, int(rng.integers(1, min(5, max(2, len(words) // 4)))))
    return s


_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _edit_string(s: str, rng: np.random.Generator, max_edits: int) -> str:
    if len(s) < 4 or max_edits <= 0:
        return s
    t = list(s)
    edits = int(rng.integers(1, max_edits + 1))
    for _ in range(edits):
        if not t:
            break
        op = int(rng.integers(0, 3))
        idx = int(rng.integers(0, len(t)))
        if op == 0 and t[idx].isalnum():  # substitute
            pool = "abcdefghijklmnopqrstuvwxyz0123456789"
            t[idx] = str(rng.choice(list(pool)))
        elif op == 1 and len(t) > 4:  # delete
            t.pop(idx)
        else:  # duplicate
            t.insert(idx, t[idx])
    return "".join(t)


def apply_typosquatting_text(
    text: str,
    *,
    max_edit_distance: int = 2,
    rng: np.random.Generator | None = None,
) -> str:
    """
    Perturb URLs in text and a sample of long alphanumeric tokens (typosquat-style noise).
    """
    if not text:
        return text
    rng = rng or _rng(None)

    def repl_url(m: re.Match[str]) -> str:
        u = m.group(0)
        if len(u) < 12:
            return u
        return _edit_string(u, rng, min(max_edit_distance, 2))

    s = _URL_RE.sub(repl_url, text)
    words = re.findall(r"[A-Za-z0-9]{5,}", s)
    if not words:
        return s
    n_touch = min(3, len(words))
    pick = rng.choice(len(words), size=n_touch, replace=False)
    for i in pick:
        w = words[int(i)]
        w2 = _edit_string(w, rng, max_edit_distance)
        if w2 != w:
            s = s.replace(w, w2, 1)
    return s


def apply_logo_manipulation_simulated(
    image: Image.Image,
    level: str = "medium",
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """
    Simulate capture/processing noise and mild blur on the screenshot (logo region not segmented).

    ``level``: ``light`` | ``medium`` | ``heavy``.
    """
    rng = rng or _rng(None)
    level = (level or "medium").lower()
    noise_std, blur_r, bright = {
        "light": (2.5, 0.35, 0.06),
        "medium": (6.0, 0.85, 0.10),
        "heavy": (12.0, 1.25, 0.14),
    }.get(level, (6.0, 0.85, 0.10))

    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0.0, noise_std, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    out = Image.fromarray(arr, mode="RGB")
    if blur_r > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur_r))
    factor = float(1.0 + rng.uniform(-bright, bright))
    out = ImageEnhance.Brightness(out).enhance(factor)
    out = ImageEnhance.Contrast(out).enhance(float(1.0 + rng.uniform(-0.06, 0.06)))
    return out


def load_prompt_injection_templates(path: str | Path) -> list[str]:
    """Load non-comment, non-empty lines from a template file."""
    p = Path(path)
    if not p.is_file():
        return []
    out: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def apply_prompt_injection(
    text: str,
    templates: list[str],
    *,
    placement: str = "end",
    rng: np.random.Generator | None = None,
) -> str:
    """
    Append or prepend an instruction-style string (tests VLM robustness to junk in page text).

    ``placement``: ``start`` | ``end`` | ``both``.
    """
    if not text or not templates:
        return text
    rng = rng or _rng(None)
    tpl = str(rng.choice(templates))
    pl = placement.lower()
    if pl == "start":
        return tpl + "\n\n" + text
    if pl == "both":
        tpl2 = str(rng.choice(templates))
        return tpl + "\n\n" + text + "\n\n" + tpl2
    return text + "\n\n" + tpl


def make_batch_preprocessor(
    attack: str,
    *,
    html_level: str = "medium",
    logo_level: str = "medium",
    typosquat_max_edits: int = 2,
    prompt_templates: list[str] | None = None,
    prompt_placement: str = "end",
    seed: int | None = None,
) -> Any:
    """
    Factory returning a ``batch_preprocessor`` for :func:`evaluation.pipeline.run_vlm_inference`.

    ``attack``: ``baseline`` | ``html_obfuscation`` | ``typosquatting`` | ``logo_manipulation`` | ``prompt_injection``.
    """
    rng = _rng(seed)

    def prep(batch: dict[str, Any]) -> dict[str, Any]:
        images = batch["images"]
        texts = batch["texts"]
        n = len(texts)
        if attack == "baseline":
            return batch
        new_texts = list(texts)
        new_images = list(images)
        for i in range(n):
            if attack == "html_obfuscation":
                new_texts[i] = apply_html_obfuscation(new_texts[i], html_level, rng)
            elif attack == "typosquatting":
                new_texts[i] = apply_typosquatting_text(
                    new_texts[i], max_edit_distance=typosquat_max_edits, rng=rng
                )
            elif attack == "logo_manipulation":
                new_images[i] = apply_logo_manipulation_simulated(
                    new_images[i], logo_level, rng
                )
            elif attack == "prompt_injection":
                new_texts[i] = apply_prompt_injection(
                    new_texts[i],
                    prompt_templates or [],
                    placement=prompt_placement,
                    rng=rng,
                )
        return {**batch, "images": new_images, "texts": new_texts}

    return prep
