"""Train-time adversarial augmentation wiring (HTML / typosquat / logo families)."""

from __future__ import annotations

import pytest
from PIL import Image

pytest.importorskip("torch")

from models.training.adversarial_augment import (
    AdversarialTrainAugmentConfig,
    AdversarialTrainBatchAugment,
    build_adversarial_train_augment,
)


def test_from_mapping_typo_max_edit_distance():
    c = AdversarialTrainAugmentConfig.from_mapping(
        {
            "enabled": True,
            "typosquatting": {"max_edit_distance": 1},
        }
    )
    assert c.typo_max_edits == 1


def test_build_returns_none_when_disabled():
    assert build_adversarial_train_augment({"enabled": False}) is None
    assert build_adversarial_train_augment({}) is None
    assert build_adversarial_train_augment(None) is None


def test_augment_changes_batch_deterministic_seed():
    img = Image.new("RGB", (64, 48), color=(120, 80, 200))
    text = "Login at https://secure-example-banking.com please verifyaccount12"
    batch = {
        "images": [img],
        "texts": [text],
        "labels": [1.0],
    }
    cfg = AdversarialTrainAugmentConfig(
        enabled=True,
        base_probability=1.0,
        seed=123,
        html_enabled=True,
        html_probability=1.0,
        html_level="medium",
        typo_enabled=True,
        typo_probability=1.0,
        typo_max_edits=2,
        logo_enabled=True,
        logo_probability=1.0,
        logo_level="light",
    )
    aug = AdversarialTrainBatchAugment(cfg)
    out = aug(dict(batch), global_step=7)
    assert out["texts"][0] != text
    assert list(out["images"][0].getdata()) != list(img.getdata())
    assert out["labels"] == batch["labels"]


def test_second_step_differs_from_first_with_same_seed_config():
    """global_step mixes into RNG so corruption is not identical every step."""
    img = Image.new("RGB", (32, 24), color=(10, 20, 30))
    text = "word " * 20 + "https://verylongurl-example.com/path"
    cfg = AdversarialTrainAugmentConfig(
        enabled=True,
        base_probability=1.0,
        seed=0,
        html_enabled=True,
        html_probability=1.0,
        typo_enabled=False,
        logo_enabled=False,
    )
    aug = AdversarialTrainBatchAugment(cfg)
    t0 = aug({"images": [img], "texts": [text]}, 0)["texts"][0]
    t1 = aug({"images": [img], "texts": [text]}, 1)["texts"][0]
    assert t0 != t1
