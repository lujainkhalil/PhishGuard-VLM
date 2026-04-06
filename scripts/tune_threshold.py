#!/usr/bin/env python3
"""
Find a decision threshold that maximizes F1 on a **validation** manifest (do not use test/TR-OP labels for this).

Writes JSON to ``evaluation/results/tables/`` (or ``--output``) for use with::

    python scripts/run_eval.py --threshold-file evaluation/results/tables/best_threshold.json

Usage:
    python scripts/tune_threshold.py --manifest data/processed/splits/validation.parquet
    python scripts/tune_threshold.py --manifest data/processed/manifest.parquet --split validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from evaluation.pipeline import run_vlm_inference
from evaluation.threshold_tuning import sweep_threshold_f1
from models import PhishingClassifier, fusion_kwargs_from_yaml
from models.training import load_manifest, build_test_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_ID_MAP = {
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
}


def load_yaml(path: Path) -> dict:
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return {}


def load_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state dict: {checkpoint_path}")
    model.load_state_dict(state, strict=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune phishing threshold on validation scores (max F1)")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/splits/validation.parquet"))
    parser.add_argument("--split", type=str, default=None, help="If set, filter manifest rows where split column equals this")
    parser.add_argument("--no-split-filter", action="store_true", help="Use all rows")
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grid-steps", type=int, default=101)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    default_cfg = load_yaml(args.default_config)
    model_cfg = load_yaml(args.config)
    paths = default_cfg.get("paths") or {}
    data_root = args.data_root or _project_root / paths.get("data_root", "data")
    manifest_path = args.manifest if args.manifest.is_absolute() else _project_root / args.manifest

    df = load_manifest(manifest_path)
    split_arg = None if args.no_split_filter else (args.split or "validation")
    loader = build_test_dataloader(
        df,
        split=split_arg,
        text_max_length=2048,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if len(loader.dataset) == 0:
        logger.error("No samples (check manifest path and --split / --no-split-filter).")
        return 1

    m_cfg = model_cfg.get("model") or {}
    model_name = m_cfg.get("name", "llava-1.5-7b")
    model_id = MODEL_ID_MAP.get(model_name, model_name)
    if "/" not in model_id:
        model_id = f"llava-hf/{model_name}-hf"
    lora_cfg = model_cfg.get("lora") or {}
    head_cfg = model_cfg.get("head") or {}
    fusion_cfg = model_cfg.get("fusion") or {}

    model = PhishingClassifier(
        model_name=model_id,
        revision=m_cfg.get("revision", "main"),
        freeze_vision_encoder=m_cfg.get("freeze_vision_encoder", True),
        train_projector=m_cfg.get("train_projector", True),
        lora_enabled=lora_cfg.get("enabled", True),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=lora_cfg.get("target_modules"),
        lora_train_multi_modal_projector=lora_cfg.get("train_multi_modal_projector", False),
        lora_gradient_checkpointing=lora_cfg.get("gradient_checkpointing", True),
        head_hidden_size=head_cfg.get("hidden_size", 4096),
        head_mlp_hidden_dim=head_cfg.get("mlp_hidden_dim"),
        head_dropout=float(head_cfg.get("dropout", 0.1)),
        head_use_layer_norm=head_cfg.get("use_layer_norm", True),
        num_classes=head_cfg.get("num_classes", 1),
        **fusion_kwargs_from_yaml(fusion_cfg),
    )

    ckpt = args.checkpoint or _project_root / paths.get("checkpoints_dir", "models/checkpoints") / "best.pt"
    ckpt = ckpt if ckpt.is_absolute() else _project_root / ckpt
    if not ckpt.is_file():
        logger.error("Checkpoint not found: %s", ckpt)
        return 1
    load_weights(model, ckpt)

    pt = model_cfg.get("prompt_template")
    if isinstance(pt, str) and pt.strip():
        model.prompt_template = pt.strip()
    if "text_max_length" in m_cfg:
        model.max_length = int(m_cfg["text_max_length"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, _, y_score, _ = run_vlm_inference(
        model, loader, device=device, threshold=0.5, batch_preprocessor=None
    )

    best_t, best_f1, metrics = sweep_threshold_f1(
        y_true, y_score, n_thresholds=args.grid_steps
    )

    out_dir = _project_root / "evaluation" / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (out_dir / "best_threshold.json")
    if not out_path.is_absolute():
        out_path = _project_root / out_path

    payload = {
        "threshold": best_t,
        "best_f1_validation": best_f1,
        "metrics_at_threshold": metrics,
        "n_samples": int(len(y_true)),
        "manifest": str(manifest_path),
        "split_filter": split_arg,
        "checkpoint": str(ckpt),
        "grid_steps": args.grid_steps,
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Tune on validation only; apply to test/TR-OP via run_eval.py --threshold-file",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Best threshold=%.4f F1=%.4f (n=%d). Wrote %s",
        best_t,
        best_f1,
        len(y_true),
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
