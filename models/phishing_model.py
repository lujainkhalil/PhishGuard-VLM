"""
Full phishing classifier: LLaVA backbone (+ LoRA) + classification head.

Input: webpage screenshot(s), webpage text.
Output: phishing probability, classification label.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .backbones import LlavaBackbone
from .fusion import CrossModalFusion, WeightedModalPool, vision_text_masks_from_input_ids
from .heads import PhishingClassificationHead
from .lora import apply_lora_to_llava
from .wrappers.llava15_multimodal import pool_hidden_state

logger = logging.getLogger(__name__)

FUSION_MODES = frozenset({"pooled", "cross_attention", "cross_attention_gated", "weighted_pool"})


def fusion_kwargs_from_yaml(fusion_cfg: dict | None) -> dict[str, Any]:
    """Map optional ``fusion:`` block from ``model.yaml`` to :class:`PhishingClassifier` kwargs."""
    fc = fusion_cfg or {}
    return {
        "fusion_mode": str(fc.get("mode", "pooled")).strip().lower(),
        "fusion_num_heads": int(fc.get("num_heads", 8)),
        "fusion_dropout": float(fc.get("dropout", 0.1)),
    }


class PhishingClassifier(nn.Module):
    """
    Multimodal phishing classifier: backbone (image + text) -> pooled hidden -> head -> logit.

    Optional **fusion** (``fusion_mode``) refines the vector fed to the head after the LLM stack:
    ``weighted_pool`` (learned mix of mean-vision vs mean-text), ``cross_attention`` (bidirectional
    cross-attention between vision tokens and text tokens with alignment layers), or
    ``cross_attention_gated`` (same plus a learned blend with the usual last-token pool). The base
    LLaVA model already performs deep multimodal self-attention; this block is an extra fusion stage
    for classification.

    - forward(images, texts, ...) returns logits.
    - predict_proba(logits) -> probability in [0, 1].
    - predict(logits or prob, threshold=0.5) -> binary label 0/1.
    """

    def __init__(
        self,
        *,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        freeze_vision_encoder: bool = True,
        train_projector: bool = True,
        lora_enabled: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        lora_train_multi_modal_projector: bool = False,
        lora_gradient_checkpointing: bool = True,
        head_hidden_size: int = 4096,
        head_mlp_hidden_dim: int | None = None,
        head_dropout: float = 0.1,
        head_use_layer_norm: bool = True,
        num_classes: int = 1,
        torch_dtype: torch.dtype | None = None,
        device_map: str | None = "auto",
        fusion_mode: str = "pooled",
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        torch_dtype = torch_dtype or torch.float16
        self.prompt_template = "Analyze this webpage screenshot and text. Is it phishing? Webpage text:\n{text}"
        self.max_length = 2048

        backbone = LlavaBackbone(
            model_name=model_name,
            revision=revision,
            freeze_vision_encoder=freeze_vision_encoder,
            train_projector=train_projector,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if lora_enabled:
            backbone = apply_lora_to_llava(
                backbone,
                r=lora_r,
                alpha=lora_alpha,
                target_modules=lora_target_modules or ["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                train_multi_modal_projector=lora_train_multi_modal_projector,
                gradient_checkpointing=lora_gradient_checkpointing,
            )
        self.backbone = backbone
        embed_dim = backbone.hidden_size

        mode = str(fusion_mode or "pooled").strip().lower()
        if mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {sorted(FUSION_MODES)}, got {fusion_mode!r}")
        self.fusion_mode = mode

        hf_cfg = backbone.model.config
        self._image_token_id = int(
            getattr(hf_cfg, "image_token_index", getattr(hf_cfg, "image_token_id", 32000))
        )

        self.cross_modal_fusion: CrossModalFusion | None = None
        self.weighted_modal_pool: WeightedModalPool | None = None
        if mode == "weighted_pool":
            self.weighted_modal_pool = WeightedModalPool(embed_dim)
        elif mode == "cross_attention":
            self.cross_modal_fusion = CrossModalFusion(
                embed_dim, num_heads=fusion_num_heads, dropout=fusion_dropout, gated=False
            )
        elif mode == "cross_attention_gated":
            self.cross_modal_fusion = CrossModalFusion(
                embed_dim, num_heads=fusion_num_heads, dropout=fusion_dropout, gated=True
            )

        if head_hidden_size != embed_dim:
            logger.warning(
                "head_hidden_size=%s does not match backbone.hidden_size=%s; using backbone size for the head input",
                head_hidden_size,
                embed_dim,
            )
        self.head = PhishingClassificationHead(
            embed_dim,
            num_classes,
            mlp_hidden_dim=head_mlp_hidden_dim,
            dropout=head_dropout,
            use_layer_norm=head_use_layer_norm,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits (batch, num_classes)."""
        if self.fusion_mode == "pooled":
            pooled = self.backbone(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            out = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            last_h = out.last_hidden_state
            vm, tm = vision_text_masks_from_input_ids(
                input_ids, attention_mask, image_token_id=self._image_token_id
            )
            if self.weighted_modal_pool is not None:
                pooled = self.weighted_modal_pool(last_h, vm, tm)
            elif self.cross_modal_fusion is not None:
                gp = None
                if self.fusion_mode == "cross_attention_gated":
                    gp = pool_hidden_state(last_h, attention_mask, pooling="last_token")
                pooled = self.cross_modal_fusion(last_h, vm, tm, global_pooled=gp)
            else:
                raise RuntimeError("fusion_mode set but no fusion module initialized")

        return self.head(pooled)

    def prepare_inputs(
        self,
        images: list[Any],
        texts: list[str],
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build tokenized inputs from raw images and texts. Moves to device if given."""
        return self.backbone.prepare_inputs(
            images=images,
            texts=texts,
            prompt_template=self.prompt_template,
            max_length=self.max_length,
            device=device,
        )

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, 1) logits -> (B,) phishing probability."""
        return PhishingClassificationHead.logits_to_probability(logits)

    def predict(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Logits -> (B,) binary label: 1 = phishing, 0 = benign."""
        prob = self.predict_proba(logits)
        return PhishingClassificationHead.probability_to_label(prob, threshold=threshold)
