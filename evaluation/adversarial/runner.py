"""
Run baseline vs. adversarial inference on the same subset; report metric drops.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from evaluation.metrics import compute_binary_metrics
from evaluation.pipeline import run_vlm_inference
from evaluation.adversarial.attacks import (
    load_prompt_injection_templates,
    make_batch_preprocessor,
)

logger = logging.getLogger(__name__)

# Operational goal: keep drops small (reporting thresholds only).
TARGET_MAX_ACCURACY_DROP = 0.05
TARGET_MAX_F1_DROP = 0.06


def _metric_deltas(baseline: dict[str, Any], attack: dict[str, Any]) -> dict[str, float | None]:
    keys = ("accuracy", "precision", "recall", "f1")
    out: dict[str, float | None] = {}
    for k in keys:
        b = baseline.get(k)
        a = attack.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            out[f"{k}_drop"] = float(b) - float(a)
        else:
            out[f"{k}_drop"] = None
    b_auc = baseline.get("roc_auc")
    a_auc = attack.get("roc_auc")
    if isinstance(b_auc, (int, float)) and isinstance(a_auc, (int, float)):
        out["roc_auc_drop"] = float(b_auc) - float(a_auc)
    else:
        out["roc_auc_drop"] = None
    return out


def run_adversarial_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    attacks: list[str],
    output_dir: Path,
    eval_config: dict[str, Any],
    project_root: Path | None = None,
    threshold: float = 0.5,
    seed: int = 42,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate ``baseline`` then each attack on the **same** loader; save JSON (+ optional CSV row).

    ``eval_config`` should contain the ``adversarial:`` block from ``configs/evaluation.yaml``.
    """
    adv = eval_config.get("adversarial") or {}
    html_cfg = adv.get("html_obfuscation") or {}
    logo_cfg = adv.get("logo_manipulation") or {}
    typo_cfg = adv.get("typosquatting") or {}
    inj_cfg = adv.get("prompt_injection") or {}
    root = Path(project_root) if project_root else Path.cwd()

    templates_path = Path(inj_cfg.get("templates_file", "configs/prompt_injection_templates.txt"))
    if not templates_path.is_absolute():
        templates_path = (root / templates_path).resolve()
    templates = load_prompt_injection_templates(templates_path)
    if "prompt_injection" in attacks and not templates:
        logger.warning("No prompt-injection templates at %s; attack may be a no-op.", templates_path)

    placement = inj_cfg.get("placement")
    if isinstance(placement, list) and placement:
        prompt_placement = str(np.random.default_rng(seed).choice(placement))
    else:
        prompt_placement = "end" if not isinstance(placement, str) else placement

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_b, y_pred_b, y_score_b, urls = run_vlm_inference(
        model, dataloader, device=device, threshold=threshold, batch_preprocessor=None
    )
    if len(y_true_b) == 0:
        raise ValueError("Empty dataloader for adversarial evaluation.")

    baseline_m = compute_binary_metrics(y_true_b, y_pred_b, y_score_b).to_dict(include_matrix=True)

    attack_results: dict[str, Any] = {}
    for attack in attacks:
        if attack == "baseline":
            continue
        pre = make_batch_preprocessor(
            attack,
            html_level=str(html_cfg.get("level", "medium")),
            logo_level=str(logo_cfg.get("level", "medium")),
            typosquat_max_edits=int(typo_cfg.get("max_edit_distance", 2)),
            prompt_templates=templates,
            prompt_placement=prompt_placement,
            seed=seed
            + int(hashlib.sha256(attack.encode()).hexdigest()[:8], 16) % 10_000,
        )
        yt, yp, ys, _ = run_vlm_inference(
            model,
            dataloader,
            device=device,
            threshold=threshold,
            batch_preprocessor=pre,
        )
        m = compute_binary_metrics(yt, yp, ys).to_dict(include_matrix=True)
        deltas = _metric_deltas(baseline_m, m)
        attack_results[attack] = {
            "metrics": m,
            "delta_vs_baseline": deltas,
        }
        acc_drop = deltas.get("accuracy_drop")
        f1_drop = deltas.get("f1_drop")
        meets_target = (
            acc_drop is not None
            and f1_drop is not None
            and acc_drop <= TARGET_MAX_ACCURACY_DROP
            and f1_drop <= TARGET_MAX_F1_DROP
        )
        attack_results[attack]["meets_minimal_degradation_target"] = meets_target
        logger.info(
            "Attack %s: acc=%.4f (drop %.4f), f1=%.4f (drop %.4f)",
            attack,
            m.get("accuracy", 0),
            acc_drop or -1,
            m.get("f1", 0),
            f1_drop or -1,
        )

    worst_f1_drop = (
        max((v["delta_vs_baseline"].get("f1_drop") or 0.0) for v in attack_results.values())
        if attack_results
        else 0.0
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"{stamp}_{run_id}" if run_id else stamp
    report_path = output_dir / f"adversarial_report_{suffix}.json"
    report = {
        "meta": {
            "utc_timestamp": stamp,
            "n_samples": int(len(y_true_b)),
            "threshold": threshold,
            "attacks": attacks,
            "goal": "minimal_degradation_under_perturbation",
            "target_max_accuracy_drop": TARGET_MAX_ACCURACY_DROP,
            "target_max_f1_drop": TARGET_MAX_F1_DROP,
            "worst_f1_drop": float(worst_f1_drop),
            "meets_worst_case_target": bool(worst_f1_drop <= TARGET_MAX_F1_DROP),
        },
        "baseline": baseline_m,
        "attacks": attack_results,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote adversarial report: %s", report_path)

    csv_path = Path(adv.get("output_table", "evaluation/results/tables/adversarial.csv"))
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_base = {
        "run_id": run_id or stamp,
        "n_samples": len(y_true_b),
        "split": "adversarial_eval",
    }
    file_exists = csv_path.is_file()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id",
            "n_samples",
            "split",
            "attack",
            "accuracy",
            "f1",
            "accuracy_drop",
            "f1_drop",
            "meets_target",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(
            {
                **row_base,
                "attack": "baseline",
                "accuracy": baseline_m.get("accuracy"),
                "f1": baseline_m.get("f1"),
                "accuracy_drop": 0.0,
                "f1_drop": 0.0,
                "meets_target": True,
            }
        )
        for name, payload in attack_results.items():
            d = payload["delta_vs_baseline"]
            w.writerow(
                {
                    **row_base,
                    "attack": name,
                    "accuracy": payload["metrics"].get("accuracy"),
                    "f1": payload["metrics"].get("f1"),
                    "accuracy_drop": d.get("accuracy_drop"),
                    "f1_drop": d.get("f1_drop"),
                    "meets_target": payload.get("meets_minimal_degradation_target"),
                }
            )
    logger.info("Appended rows to %s", csv_path)

    return report


def build_subset_dataloader(
    base_loader: DataLoader,
    n_samples: int,
    seed: int,
) -> DataLoader:
    """Subsample dataset indices for faster adversarial sweeps."""
    ds = base_loader.dataset
    n = len(ds)
    if n == 0:
        raise ValueError("Empty dataset")
    k = min(n_samples, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False).tolist()
    subset = Subset(ds, idx)
    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=False,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=getattr(base_loader, "pin_memory", False),
        persistent_workers=getattr(base_loader, "persistent_workers", False),
    )
