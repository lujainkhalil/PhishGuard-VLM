"""
Train-time adversarial-style augmentation (HTML obfuscation, typosquat-like text noise,
simulated logo/screenshot degradation).

Reuses transforms from :mod:`evaluation.adversarial.attacks` so evaluation probes match training
corruption families. **Labels are unchanged** (phishing vs benign); only inputs are perturbed for
robustness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from evaluation.adversarial.attacks import (
    apply_html_obfuscation,
    apply_logo_manipulation_simulated,
    apply_typosquatting_text,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdversarialTrainAugmentConfig:
    enabled: bool = False
    base_probability: float = 0.32
    seed: int | None = 42
    html_enabled: bool = True
    html_probability: float = 0.55
    html_level: str = "medium"
    typo_enabled: bool = True
    typo_probability: float = 0.5
    typo_max_edits: int = 2
    logo_enabled: bool = True
    logo_probability: float = 0.45
    logo_level: str = "light"

    @staticmethod
    def from_mapping(m: dict[str, Any] | None) -> AdversarialTrainAugmentConfig:
        raw = dict(m or {})
        html = dict(raw.get("html_obfuscation") or {})
        typo = dict(raw.get("typosquatting") or {})
        logo = dict(raw.get("logo_manipulation") or {})
        seed = raw.get("seed")
        return AdversarialTrainAugmentConfig(
            enabled=bool(raw.get("enabled", False)),
            base_probability=float(raw.get("base_probability", 0.32)),
            seed=int(seed) if seed is not None else None,
            html_enabled=bool(html.get("enabled", True)),
            html_probability=float(html.get("probability", 0.55)),
            html_level=str(html.get("level", "medium")),
            typo_enabled=bool(typo.get("enabled", True)),
            typo_probability=float(typo.get("probability", 0.5)),
            typo_max_edits=int(typo.get("max_edit_distance", 2)),
            logo_enabled=bool(logo.get("enabled", True)),
            logo_probability=float(logo.get("probability", 0.45)),
            logo_level=str(logo.get("level", "light")),
        )


class AdversarialTrainBatchAugment:
    """
    Callable ``(batch, global_step) -> batch`` for use inside :meth:`PhishingTrainer.train_step`.

    For each sample, with probability ``base_probability``, applies zero or more enabled
    transforms (each with its own conditional probability). RNG mixes in ``global_step`` for
    variety across steps.
    """

    def __init__(self, cfg: AdversarialTrainAugmentConfig) -> None:
        self.cfg = cfg

    def __call__(self, batch: dict[str, Any], global_step: int) -> dict[str, Any]:
        if not self.cfg.enabled:
            return batch
        images = batch.get("images")
        texts = batch.get("texts")
        if images is None or texts is None:
            return batch
        n = len(texts)
        if n == 0:
            return batch

        seed = (self.cfg.seed if self.cfg.seed is not None else 0) + global_step * 1_009_063
        rng = np.random.default_rng(seed)

        new_images = list(images)
        new_texts = list(texts)

        for i in range(n):
            if rng.random() >= self.cfg.base_probability:
                continue
            t = new_texts[i]
            img = new_images[i]

            if self.cfg.html_enabled and rng.random() < self.cfg.html_probability:
                t = apply_html_obfuscation(t, self.cfg.html_level, rng)
            if self.cfg.typo_enabled and rng.random() < self.cfg.typo_probability:
                t = apply_typosquatting_text(
                    t, max_edit_distance=self.cfg.typo_max_edits, rng=rng
                )
            if self.cfg.logo_enabled and rng.random() < self.cfg.logo_probability:
                try:
                    img = apply_logo_manipulation_simulated(img, self.cfg.logo_level, rng)
                except Exception as e:
                    logger.debug("Logo augment skipped for sample %d: %s", i, e)
            new_texts[i] = t
            new_images[i] = img

        return {**batch, "images": new_images, "texts": new_texts}


def build_adversarial_train_augment(
    cfg: dict[str, Any] | None,
) -> Callable[[dict[str, Any], int], dict[str, Any]] | None:
    """Return augment callable or ``None`` if disabled / empty config."""
    c = AdversarialTrainAugmentConfig.from_mapping(cfg)
    if not c.enabled:
        return None
    aug = AdversarialTrainBatchAugment(c)
    logger.info(
        "Adversarial train augmentation on (base_p=%.2f html=%s typo=%s logo=%s).",
        c.base_probability,
        c.html_enabled,
        c.typo_enabled,
        c.logo_enabled,
    )
    return aug
