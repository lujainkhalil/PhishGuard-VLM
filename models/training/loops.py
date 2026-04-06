"""
Multimodal forward pass, loss, and validation loop (image + text → logits).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_binary_classification_metrics


def multimodal_forward_and_loss(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    criterion: nn.Module,
    *,
    use_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single forward: prepare image+text inputs, run model, compute binary classification loss.

    Expects ``batch`` with keys ``images``, ``texts``, ``labels`` (long 0/1).

    Returns:
        loss (scalar tensor), logits (detached or not — caller uses .backward on loss only).
    """
    labels = batch["labels"].to(device, dtype=torch.float32).unsqueeze(1)
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            inputs = model.prepare_inputs(batch["images"], batch["texts"], device=device)
            logits = model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            loss = criterion(logits, labels)
    else:
        inputs = model.prepare_inputs(batch["images"], batch["texts"], device=device)
        logits = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        loss = criterion(logits, labels)
    return loss, logits


@torch.no_grad()
def metrics_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    desc: str = "Eval",
    restore_training: bool = True,
    use_amp: bool = False,
) -> tuple[dict[str, float], float]:
    """
    Run inference-style forward over a loader; return classification metrics and mean loss.

    Model is set to ``eval`` during the pass; if ``restore_training`` is True, restored to ``train`` after.
    """
    was_training = model.training
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        loss, logits = multimodal_forward_and_loss(
            model, batch, device, criterion, use_amp=use_amp
        )
        total_loss += loss.item()
        n_batches += 1
        preds = model.predict(logits)
        all_labels.append(batch["labels"].cpu())
        all_preds.append(preds.cpu())

    if n_batches == 0:
        if restore_training and was_training:
            model.train()
        elif restore_training and not was_training:
            model.eval()
        return {}, 0.0

    if restore_training and was_training:
        model.train()
    elif restore_training and not was_training:
        model.eval()

    labels_cat = torch.cat(all_labels, dim=0)
    preds_cat = torch.cat(all_preds, dim=0)
    metrics = compute_binary_classification_metrics(labels_cat, preds_cat)
    mean_loss = total_loss / n_batches if n_batches else 0.0
    return metrics, mean_loss


@torch.no_grad()
def validate_multimodal(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    desc: str = "Validation",
    use_amp: bool = False,
) -> tuple[dict[str, float], float]:
    """
    Full validation pass: average loss plus accuracy, precision, recall, F1.

    Returns:
        metrics dict, mean validation loss.
    """
    return metrics_on_loader(
        model,
        val_loader,
        device,
        criterion,
        desc=desc,
        restore_training=True,
        use_amp=use_amp,
    )
