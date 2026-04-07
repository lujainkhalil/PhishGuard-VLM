"""
Modular Hugging Face wrapper for LLaVA-1.5 7B: multimodal (image + text) in, embeddings or LM logits out.

Designed to compose with arbitrary heads (e.g. :class:`PhishingClassificationHead`) without coupling.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

OutputType = Literal["embeddings", "last_hidden_state", "lm_logits"]
PoolingType = Literal["last_token", "mean"]


def pool_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    pooling: PoolingType = "last_token",
) -> torch.Tensor:
    """
    (B, L, D) -> (B, D). ``last_token`` uses the last non-padding position; ``mean`` averages valid tokens.
    """
    if pooling == "mean":
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    if attention_mask is None:
        return last_hidden_state[:, -1, :]
    last_valid = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, last_valid, :]


class LLaVA15MultimodalWrapper(nn.Module):
    """
    Loads ``LlavaForConditionalGeneration`` + ``AutoProcessor``; runs vision + language forward.

    - **Inputs**: ``pixel_values``, ``input_ids``, ``attention_mask`` (from
      :meth:`prepare_multimodal_inputs` or your own tokenization).
    - **Outputs** (``output_type``):
        - ``embeddings``: pooled sequence representation ``(batch, hidden_size)`` for a classification head.
        - ``last_hidden_state``: full sequence ``(batch, seq_len, hidden_size)``.
        - ``lm_logits``: vocabulary logits ``(batch, seq_len, vocab_size)`` from the LM head.

    Exposes ``.model`` (the HF module) so :func:`models.lora.apply_lora_to_llava` keeps working.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        revision: str = "main",
        *,
        freeze_vision_encoder: bool = True,
        train_projector: bool = True,
        torch_dtype: torch.dtype | None = None,
        device_map: str | dict | None = "auto",
    ):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        torch_dtype = torch_dtype or torch.float16

        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

        if freeze_vision_encoder and hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
            for p in self.model.model.vision_tower.parameters():
                p.requires_grad = False
            logger.info("Frozen vision encoder.")

        if not train_projector and hasattr(self.model, "model") and hasattr(self.model.model, "multi_modal_projector"):
            for p in self.model.model.multi_modal_projector.parameters():
                p.requires_grad = False
            logger.info("Frozen multi_modal_projector.")

        self._hidden_size = int(self.model.config.text_config.hidden_size)

    @property
    def hidden_size(self) -> int:
        """Hidden size of the language model (input dim for a classification head)."""
        return self._hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_type: OutputType = "embeddings",
        pooling: PoolingType = "last_token",
        output_hidden_states: bool = True,
    ) -> torch.Tensor:
        """
        Run the multimodal forward pass.

        Returns:
            - ``embeddings``: ``(B, hidden_size)``
            - ``last_hidden_state``: ``(B, L, hidden_size)``
            - ``lm_logits``: ``(B, L, vocab_size)``
        """
        need_hidden = output_type in ("embeddings", "last_hidden_state")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        if output_type == "lm_logits":
            if outputs.logits is None:
                raise RuntimeError("Model returned no logits; check config and inputs.")
            return outputs.logits

        last_hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None
            else outputs.hidden_states[-1]
        )
        if output_type == "last_hidden_state":
            return last_hidden

        return pool_hidden_state(last_hidden, attention_mask, pooling=pooling)

    def prepare_multimodal_inputs(
        self,
        images: list[Any],
        texts: list[str],
        prompt_template: str,
        *,
        max_length: int = 2048,
        truncation: bool = True,
        return_tensors: str = "pt",
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Build tensors from PIL / tensor images and text strings using the processor chat template.

        ``prompt_template`` must contain ``{text}`` (webpage text inserted per sample).
        """
        assert len(images) == len(texts)
        from torch.nn.utils.rnn import pad_sequence

        list_pixel: list[torch.Tensor] = []
        list_input_ids: list[torch.Tensor] = []
        list_attention: list[torch.Tensor] = []
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        for img, text in zip(images, texts):
            body = (text or "")[:max_length] if truncation else (text or "")
            prompt = prompt_template.format(text=body)
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            out = self.processor.apply_chat_template(
                conv,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                padding=True,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                images=[img],
            )
            list_pixel.append(out["pixel_values"])
            list_input_ids.append(out["input_ids"])
            list_attention.append(out["attention_mask"])

        input_ids = pad_sequence(
            [x.squeeze(0) for x in list_input_ids],
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = pad_sequence(
            [x.squeeze(0) for x in list_attention],
            batch_first=True,
            padding_value=0,
        )
        pixel_values = torch.cat(list_pixel, dim=0)
        batch = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}
        if device is not None:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def forward_with_classification_head(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head: nn.Module,
        *,
        pooling: PoolingType = "last_token",
    ) -> torch.Tensor:
        """
        Pool multimodal embeddings and apply ``head`` (e.g. linear classifier). Returns logits from ``head``.
        """
        emb = self.forward(
            pixel_values,
            input_ids,
            attention_mask,
            output_type="embeddings",
            pooling=pooling,
        )
        return head(emb)
