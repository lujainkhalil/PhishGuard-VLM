"""
End-to-end evaluation: test DataLoader → VLM inference → metrics → files under ``evaluation/results``.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_binary_metrics

logger = logging.getLogger(__name__)

# Batch dict: images (list[PIL]), texts (list[str]), labels (Tensor), urls (list[str])
BatchPreprocessor = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class EvaluationArtifacts:
    """Paths and summary written by :func:`run_evaluation_pipeline`."""

    metrics_path: Path
    predictions_path: Path | None
    n_samples: int
    metrics: dict[str, Any]


def _resolve_torch_device(preferred: torch.device | None = None) -> torch.device:
    if preferred is not None:
        return preferred
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_vlm_inference(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    threshold: float = 0.5,
    batch_preprocessor: BatchPreprocessor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Full forward pass over the loader: logits → phishing probability → hard label.

    If ``batch_preprocessor`` is set, it receives a shallow-copied batch dict (lists for
    ``images``/``texts``/``urls``) and must return the dict with updated ``images``/``texts``.

    Returns:
        y_true (N,), y_pred (N,), y_score (N,), urls (length N).
    """
    model.eval()
    y_list: list[int] = []
    pred_list: list[int] = []
    prob_list: list[float] = []
    url_list: list[str] = []

    for batch in tqdm(dataloader, desc="Inference", leave=False):
        labels = batch["labels"].view(-1).cpu().numpy().astype(int)
        if batch_preprocessor is not None:
            work: dict[str, Any] = {
                "images": list(batch["images"]),
                "texts": list(batch["texts"]),
                "labels": batch["labels"],
                "urls": list(batch.get("urls") or [""] * len(batch["texts"])),
            }
            work = batch_preprocessor(work)
            images = work["images"]
            texts = work["texts"]
            urls = work.get("urls") or [""] * len(labels)
        else:
            images = batch["images"]
            texts = batch["texts"]
            urls = batch.get("urls") or [""] * len(labels)

        inputs = model.prepare_inputs(images, texts, device=device)
        logits = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        probs = model.predict_proba(logits).detach().cpu().numpy().reshape(-1)
        preds = model.predict(logits, threshold=threshold).detach().cpu().numpy().reshape(-1)

        for i in range(len(labels)):
            y_list.append(int(labels[i]))
            pred_list.append(int(preds[i]))
            prob_list.append(float(probs[i]))
            url_list.append(str(urls[i]) if i < len(urls) else "")

    return (
        np.asarray(y_list, dtype=np.int64),
        np.asarray(pred_list, dtype=np.int64),
        np.asarray(prob_list, dtype=np.float64),
        url_list,
    )


def run_evaluation_pipeline(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    output_dir: Path,
    device: torch.device | None = None,
    threshold: float = 0.5,
    run_id: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    save_predictions_csv: bool = True,
    batch_preprocessor: BatchPreprocessor | None = None,
) -> EvaluationArtifacts:
    """
    Run inference, compute accuracy / precision / recall / F1 / ROC-AUC / confusion matrix, save JSON (+ CSV).

    ``output_dir`` is created if missing. Filenames include UTC timestamp and optional ``run_id``.
    """
    device = _resolve_torch_device(device)
    try:
        model.to(device)
    except Exception:
        logger.debug("Model .to(device) skipped (e.g. device_map=auto).")

    y_true, y_pred, y_score, urls = run_vlm_inference(
        model,
        dataloader,
        device=device,
        threshold=threshold,
        batch_preprocessor=batch_preprocessor,
    )
    n = len(y_true)
    if n == 0:
        raise ValueError("No evaluation samples in dataloader.")

    m = compute_binary_metrics(y_true, y_pred, y_score)
    metrics_flat = m.to_dict(include_matrix=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"{stamp}_{run_id}" if run_id else stamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "utc_timestamp": stamp,
        "n_samples": n,
        "threshold": threshold,
        "device": str(device),
        **(extra_meta or {}),
    }
    metrics_path = output_dir / f"eval_metrics_{suffix}.json"
    payload = {"meta": meta, "metrics": metrics_flat}
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote metrics: %s", metrics_path)

    pred_path: Path | None = None
    if save_predictions_csv:
        pred_path = output_dir / f"eval_predictions_{suffix}.csv"
        with open(pred_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["url", "y_true", "y_pred", "phishing_probability", "correct"],
            )
            w.writeheader()
            for i in range(n):
                w.writerow(
                    {
                        "url": urls[i],
                        "y_true": y_true[i],
                        "y_pred": y_pred[i],
                        "phishing_probability": f"{y_score[i]:.6f}",
                        "correct": int(y_true[i] == y_pred[i]),
                    }
                )
        logger.info("Wrote predictions: %s", pred_path)

    return EvaluationArtifacts(
        metrics_path=metrics_path,
        predictions_path=pred_path,
        n_samples=n,
        metrics=metrics_flat,
    )
