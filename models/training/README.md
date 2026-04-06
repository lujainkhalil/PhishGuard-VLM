# Training

Full training pipeline: LoRA fine-tuning with dataset loading, training loop, validation, checkpoint saving, and W&B experiment tracking.

- **Dataset loading**: `load_manifest()` (Parquet/CSV), `build_datasets()` (train/val by split), `build_dataloaders()` with optional balanced sampling. See `pipeline.py`.
- **Adversarial augmentation**: `configs/training.yaml` `adversarial_augmentation:` (set `enabled: true`) applies optional HTML obfuscation, typosquat-style URL/token noise, and light screenshot degradation per batch in `PhishingTrainer.train_step`, reusing `evaluation/adversarial/attacks.py`. Labels are unchanged.
- **Hard negatives**: Manifest column `hard_negative_category` (from `configs/data.yaml` `hard_negatives:` merged crawl) marks benign login/branded/lookalike pages; `training.hard_negative_oversample` up-weights them in `WeightedRandomSampler` when `balanced_sampling` is on.
- **Loss**: Default BCE-with-logits; `configs/training.yaml` → `loss:` supports `pos_weight: auto` (train-class ratio) and `type: focal` (see `losses.py`). Pass a custom `criterion` into `PhishingTrainer` if needed.
- **Training loop**: `PhishingTrainer` runs the configured loss, gradient clip, optimizer step, LR warmup+decay; evaluates every `eval_steps` and at end of each epoch.
- **Validation**: `evaluate()` computes F1, precision, recall on the validation set.
- **Checkpoint saving**: Best model by `metric_for_best` (default F1) saved as `best.pt`; periodic `checkpoint-{step}.pt`. Checkpoints include model, optimizer, scheduler, step, and optional config.
- **W&B**: When `wandb.project` is set in config, trainer initializes a run, logs config, train loss/lr, validation metrics, and best metric; saves checkpoints to W&B. Use `WANDB_MODE=disabled` to turn off.
