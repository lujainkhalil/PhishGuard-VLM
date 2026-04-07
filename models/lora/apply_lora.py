"""
Apply PEFT LoRA to the LLaVA language model and freeze all non-adapter backbone weights.

Optional gradient checkpointing reduces activation memory during fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


def _count_trainable_params(module: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def freeze_llava_except_lora_adapters(
    hf_llava: nn.Module,
    *,
    train_multi_modal_projector: bool = False,
) -> None:
    """
    Set ``requires_grad`` so only LoRA matrices (and optionally ``multi_modal_projector``) train.

    ``hf_llava`` should be ``LlavaForConditionalGeneration`` (the same module as ``backbone.model``).
    """
    for name, param in hf_llava.named_parameters():
        nl = name.lower()
        is_lora = "lora_a" in nl or "lora_b" in nl or ".lora." in nl
        is_projector = train_multi_modal_projector and "multi_modal_projector" in nl
        param.requires_grad = bool(is_lora or is_projector)


def apply_lora_to_llava(
    backbone: Any,
    *,
    r: int = 16,
    alpha: int = 32,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    train_multi_modal_projector: bool = False,
    gradient_checkpointing: bool = True,
) -> Any:
    """
    Wrap the LLaVA **language model** with PEFT LoRA and freeze the rest of the HF module.

    - Trainable inside ``backbone.model``: LoRA adapter weights only, unless
      ``train_multi_modal_projector`` is True (then the vision–language projector also trains).
    - The classification head lives on ``PhishingClassifier`` and remains trainable separately.
    - Enables gradient checkpointing and ``use_cache=False`` when supported to lower VRAM use.

    Args:
        backbone: Module with ``.model`` = ``LlavaForConditionalGeneration`` (e.g. ``LlavaBackbone``).
        r, alpha, target_modules, lora_dropout, bias: Passed to :class:`peft.LoraConfig`.
        train_multi_modal_projector: If True, keep projector weights trainable (higher memory).
        gradient_checkpointing: Enable on the HF model if available.
    """
    target_modules = target_modules or ["q_proj", "v_proj"]
    if not hasattr(backbone, "model"):
        raise ValueError("Backbone must have .model (LlavaForConditionalGeneration)")
    hf_llava = backbone.model
    if not hasattr(hf_llava, "model"):
        raise ValueError("LLaVA .model (inner LlavaModel) not found")
    inner = hf_llava.model
    language_model = getattr(inner, "language_model", None)
    if language_model is None:
        language_model = getattr(hf_llava, "language_model", None)
    if language_model is None:
        raise ValueError("LLaVA language_model not found")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
    )
    language_model = get_peft_model(language_model, config)
    if hasattr(inner, "language_model"):
        inner.language_model = language_model
    else:
        hf_llava.language_model = language_model

    if hasattr(language_model, "print_trainable_parameters"):
        language_model.print_trainable_parameters()

    freeze_llava_except_lora_adapters(
        hf_llava,
        train_multi_modal_projector=train_multi_modal_projector,
    )

    if gradient_checkpointing:
        if hasattr(hf_llava, "config") and hf_llava.config is not None:
            hf_llava.config.use_cache = False
        gc_ok = False
        if hasattr(hf_llava, "gradient_checkpointing_enable"):
            hf_llava.gradient_checkpointing_enable()
            gc_ok = True
        elif hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable()
            gc_ok = True
        if gc_ok:
            logger.info("Gradient checkpointing enabled on LLaVA (use_cache=False) for lower memory use.")
        else:
            logger.warning("Gradient checkpointing not available on this LLaVA build; consider smaller batch size.")

    trainable, total = _count_trainable_params(hf_llava)
    pct = 100.0 * trainable / total if total else 0.0
    extra = " + multi_modal_projector" if train_multi_modal_projector else ""
    logger.info(
        "LLaVA backbone trainable params%s: %s / %s (%.3f%%)",
        extra,
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )
    return backbone
