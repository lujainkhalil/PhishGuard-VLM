#!/usr/bin/env python3
"""
Evaluate PhishingClassifier on a held-out manifest split: inference → metrics → ``evaluation/results``.

Usage:
    python scripts/run_eval.py --checkpoint models/checkpoints/best.pt
    python scripts/run_eval.py --manifest data/processed/splits/test.parquet --split test
    python scripts/run_eval.py --tr-op   # TR-OP benchmark (configs evaluation.yaml / data.yaml paths)
    python scripts/run_eval.py --threshold-file evaluation/results/tables/best_threshold.json
    python scripts/run_eval_tr_op.py --threshold-file evaluation/results/tables/best_threshold.json
    python scripts/run_eval.py --no-checkpoint   # base weights only (sanity / ablation)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from evaluation.benchmark_paths import resolve_benchmark_manifest
from evaluation.pipeline import run_evaluation_pipeline
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
    parser = argparse.ArgumentParser(description="Run VLM evaluation on test split")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    parser.add_argument("--split", type=str, default="test", help="Manifest split column value (ignored if no column)")
    parser.add_argument("--no-split-filter", action="store_true", help="Use all rows in manifest")
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt from training (optional)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Do not load fine-tuned weights")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=None,
        help="JSON from scripts/tune_threshold.py (key 'threshold') overrides --threshold",
    )
    parser.add_argument(
        "--tr-op",
        action="store_true",
        help="Evaluate TR-OP benchmark manifest from configs (evaluation.yaml test_sets.tr_op or data.yaml benchmarks.tr_op_path)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-predictions-csv", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    default_cfg = load_yaml(args.default_config)
    model_cfg = load_yaml(args.config)
    eval_cfg = load_yaml(args.eval_config)
    data_cfg = load_yaml(_project_root / "configs/data.yaml")

    paths = default_cfg.get("paths") or {}
    data_root = args.data_root or _project_root / paths.get("data_root", "data")
    manifest_path = args.manifest if args.manifest.is_absolute() else _project_root / args.manifest
    split_arg = None if args.no_split_filter else args.split

    if args.tr_op:
        tr_path = (eval_cfg.get("test_sets") or {}).get("tr_op") or (data_cfg.get("benchmarks") or {}).get(
            "tr_op_path"
        )
        if not tr_path:
            logger.error("TR-OP path missing: set test_sets.tr_op in configs/evaluation.yaml or benchmarks.tr_op_path in configs/data.yaml")
            return 1
        manifest_path = resolve_benchmark_manifest(_project_root, tr_path)
        split_arg = None
        logger.info("TR-OP manifest: %s (all rows; evaluation-only benchmark)", manifest_path)

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

    loaded_ckpt: str | None = None
    if not args.no_checkpoint:
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = _project_root / paths.get("checkpoints_dir", "models/checkpoints") / "best.pt"
        ckpt = ckpt if ckpt.is_absolute() else _project_root / ckpt
        if ckpt.is_file():
            load_weights(model, ckpt)
            loaded_ckpt = str(ckpt)
        else:
            logger.warning("Checkpoint not found at %s — evaluating uninitialized head / base weights.", ckpt)
    else:
        logger.info("Skipping checkpoint load (--no-checkpoint).")

    pt = model_cfg.get("prompt_template")
    if isinstance(pt, str) and pt.strip():
        model.prompt_template = pt.strip()
    m_cfg_all = model_cfg.get("model") or {}
    if "text_max_length" in m_cfg_all:
        model.max_length = int(m_cfg_all["text_max_length"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = float(args.threshold)
    if args.threshold_file:
        tf = args.threshold_file if args.threshold_file.is_absolute() else _project_root / args.threshold_file
        data = json.loads(tf.read_text(encoding="utf-8"))
        threshold = float(data["threshold"])
        logger.info("Using threshold %.4f from %s", threshold, tf)

    meta = {
        "manifest": str(manifest_path),
        "split": split_arg,
        "checkpoint": loaded_ckpt,
        "model_id": model_id,
        "threshold": threshold,
        "tr_op_eval": bool(args.tr_op),
    }

    art = run_evaluation_pipeline(
        model,
        loader,
        output_dir=output_dir,
        device=device,
        threshold=threshold,
        run_id=args.run_id,
        extra_meta=meta,
        save_predictions_csv=not args.no_predictions_csv,
    )

    m = art.metrics
    logger.info(
        "Done: n=%d acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%s",
        art.n_samples,
        m.get("accuracy", 0),
        m.get("precision", 0),
        m.get("recall", 0),
        m.get("f1", 0),
        m.get("roc_auc"),
    )
    logger.info("Metrics file: %s", art.metrics_path)
    if art.predictions_path:
        logger.info("Predictions file: %s", art.predictions_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
