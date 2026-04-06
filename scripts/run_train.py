#!/usr/bin/env python3
"""
Train the PhishingClassifier (LLaVA-1.5 7B + LoRA + classification head).

Pipeline: dataset loading → train/val dataloaders → training loop with validation,
checkpoint saving, and W&B experiment tracking.

Usage:
    python scripts/run_train.py
    python scripts/run_train.py --config configs/model.yaml --training-config configs/training.yaml
    python scripts/run_train.py --manifest data/processed/manifest.parquet
    WANDB_MODE=disabled python scripts/run_train.py   # disable W&B
"""

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from models import PhishingClassifier, fusion_kwargs_from_yaml
from models.training import (
    PhishingTrainer,
    load_manifest,
    build_datasets,
    build_dataloaders,
)
from models.training.losses import build_train_criterion

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size in YAML (Colab: try 1)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs in YAML")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override checkpoint.eval_steps")
    parser.add_argument("--save-steps", type=int, default=None, help="Override checkpoint.save_steps")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Mixed precision (fp16) on CUDA via GradScaler (saves VRAM)",
    )
    args = parser.parse_args()

    default_cfg = load_yaml(args.default_config)
    model_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.training_config)

    paths = default_cfg.get("paths") or {}
    data_root = args.data_root or _project_root / paths.get("data_root", "data")
    checkpoint_dir = args.checkpoint_dir or _project_root / paths.get("checkpoints_dir", "models/checkpoints")
    manifest_path = args.manifest if args.manifest.is_absolute() else _project_root / args.manifest

    try:
        df = load_manifest(manifest_path)
    except FileNotFoundError:
        logger.error("Manifest not found: %s. Run run_preprocess.py first.", manifest_path)
        return 1

    text_max_length = 2048
    train_ds, val_ds = build_datasets(
        df,
        text_max_length=text_max_length,
        data_root=data_root,
    )
    if len(train_ds) == 0:
        logger.error("No training samples. Check manifest and split column.")
        return 1

    train_cfg_t = dict(train_cfg.get("training") or {})
    if args.batch_size is not None:
        train_cfg_t["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        train_cfg_t["epochs"] = int(args.epochs)
    batch_size = train_cfg_t.get("batch_size", 8)
    balanced = train_cfg_t.get("balanced_sampling", True)
    train_loader, val_loader = build_dataloaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        balanced_sampling=balanced,
        hard_negative_oversample=float(train_cfg_t.get("hard_negative_oversample", 1.0)),
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_cfg = train_cfg.get("loss") or {}
    criterion = build_train_criterion(loss_cfg, train_df=train_ds.df, device=device)

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

    ckpt_cfg = dict(train_cfg.get("checkpoint") or {})
    if args.eval_steps is not None:
        ckpt_cfg["eval_steps"] = int(args.eval_steps)
    if args.save_steps is not None:
        ckpt_cfg["save_steps"] = int(args.save_steps)
    es_cfg = train_cfg.get("early_stopping") or {}
    wandb_cfg = train_cfg.get("wandb") or {}
    adv_aug_cfg = train_cfg.get("adversarial_augmentation") or {}
    wandb_config = {
        "model": model_cfg.get("model"),
        "lora": lora_cfg,
        "fusion": fusion_cfg,
        "training": train_cfg_t,
        "loss": loss_cfg,
        "checkpoint": ckpt_cfg,
        "early_stopping": es_cfg,
        "adversarial_augmentation": adv_aug_cfg,
    }

    trainer = PhishingTrainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=train_cfg_t.get("learning_rate", 2e-5),
        epochs=train_cfg_t.get("epochs", 10),
        warmup_ratio=train_cfg_t.get("warmup_ratio", 0.03),
        weight_decay=train_cfg_t.get("weight_decay", 0.01),
        max_grad_norm=train_cfg_t.get("max_grad_norm", 1.0),
        checkpoint_dir=checkpoint_dir,
        save_steps=ckpt_cfg.get("save_steps", 500),
        eval_steps=ckpt_cfg.get("eval_steps", 250),
        metric_for_best=ckpt_cfg.get("metric_for_best", "f1"),
        wandb_project=wandb_cfg.get("project"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_run_name=args.run_name or wandb_cfg.get("run_name"),
        wandb_config=wandb_config,
        criterion=criterion,
        scheduler_type=train_cfg_t.get("scheduler_type", "linear_warmup"),
        early_stopping_patience=int(es_cfg.get("patience", 0)),
        early_stopping_min_delta=float(es_cfg.get("min_delta", 0.0)),
        log_train_metrics_each_epoch=bool(train_cfg_t.get("log_train_metrics_each_epoch", False)),
        adversarial_augmentation=adv_aug_cfg,
        use_amp=bool(args.use_amp),
    )

    logger.info("Starting training: %d train, %d val", len(train_ds), len(val_ds))
    result = trainer.train()
    logger.info("Training done. %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
