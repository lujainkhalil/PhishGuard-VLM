#!/usr/bin/env python3
"""
Baseline vs. adversarial evaluation on the same test subset: HTML obfuscation, typosquatting,
simulated logo/screenshot noise, prompt injection. Writes JSON under ``evaluation/results`` and
appends ``evaluation/results/tables/adversarial.csv``.

Usage:
    python scripts/run_adversarial_eval.py --checkpoint models/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from evaluation.adversarial.runner import build_subset_dataloader, run_adversarial_evaluation
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("load_state_dict missing keys (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        logger.warning("load_state_dict unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])
    logger.info("Loaded weights from %s", checkpoint_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation (same subset as baseline)")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--no-split-filter", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-test-set",
        action="store_true",
        help="Use entire test loader (ignore adversarial.n_samples_per_attack)",
    )
    args = parser.parse_args()

    default_cfg = load_yaml(args.default_config)
    model_cfg = load_yaml(args.config)
    eval_cfg = load_yaml(args.eval_config)
    adv_cfg = eval_cfg.get("adversarial") or {}

    paths = default_cfg.get("paths") or {}
    data_root = args.data_root or _project_root / paths.get("data_root", "data")
    manifest_path = args.manifest if args.manifest.is_absolute() else _project_root / args.manifest

    results_cfg = eval_cfg.get("results") or {}
    output_dir = args.output_dir or _project_root / Path(results_cfg.get("output_dir", "evaluation/results"))

    try:
        df = load_manifest(manifest_path)
    except FileNotFoundError:
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    split_arg = None if args.no_split_filter else args.split
    loader = build_test_dataloader(
        df,
        split=split_arg,
        text_max_length=2048,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if len(loader.dataset) == 0:
        logger.error("No samples for evaluation (check split / manifest).")
        return 1

    n_cap = int(adv_cfg.get("n_samples_per_attack", 500))
    if not args.full_test_set and n_cap > 0:
        loader = build_subset_dataloader(loader, n_samples=n_cap, seed=args.seed)
        logger.info("Using adversarial subset: n=%d (from n_samples_per_attack)", len(loader.dataset))

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

    if not args.no_checkpoint:
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = _project_root / paths.get("checkpoints_dir", "models/checkpoints") / "best.pt"
        ckpt = ckpt if ckpt.is_absolute() else _project_root / ckpt
        if ckpt.is_file():
            load_weights(model, ckpt)
        else:
            logger.warning("Checkpoint not found at %s — evaluating base/head weights.", ckpt)
    else:
        logger.info("Skipping checkpoint load (--no-checkpoint).")

    pt = model_cfg.get("prompt_template")
    if isinstance(pt, str) and pt.strip():
        model.prompt_template = pt.strip()
    m_cfg_all = model_cfg.get("model") or {}
    if "text_max_length" in m_cfg_all:
        model.max_length = int(m_cfg_all["text_max_length"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attacks = list(adv_cfg.get("attacks") or [])
    if not attacks:
        attacks = [
            "html_obfuscation",
            "typosquatting",
            "logo_manipulation",
            "prompt_injection",
        ]

    report = run_adversarial_evaluation(
        model,
        loader,
        device=device,
        attacks=attacks,
        output_dir=output_dir,
        eval_config=eval_cfg,
        project_root=_project_root,
        threshold=args.threshold,
        seed=args.seed,
        run_id=args.run_id,
    )
    meta = report["meta"]
    logger.info(
        "Adversarial eval done: n=%d worst_f1_drop=%.4f meets_worst_case_target=%s",
        meta["n_samples"],
        meta["worst_f1_drop"],
        meta["meets_worst_case_target"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
