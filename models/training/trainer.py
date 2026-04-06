"""
Training orchestration: optimizer, scheduler, train/val loops, checkpoints, W&B.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import amp as torch_amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adversarial_augment import build_adversarial_train_augment
from .loops import metrics_on_loader, multimodal_forward_and_loss, validate_multimodal
from .schedulers import build_lr_scheduler

logger = logging.getLogger(__name__)


def _maybe_wandb_init(
    project: str | None = None,
    entity: str | None = None,
    name: str | None = None,
    config: dict[str, Any] | None = None,
) -> bool:
    if not project:
        return False
    try:
        import wandb

        wandb.init(project=project, entity=entity or None, name=name, config=config or {})
        return True
    except Exception as e:
        logger.warning("W&B init skipped: %s", e)
        return False


class PhishingTrainer:
    """
    End-to-end training for multimodal phishing classifiers.

    - Loads data via externally built :class:`DataLoader`\\ s (see :mod:`models.training.pipeline`).
    - Each step: image + text forward, binary loss (default BCE-with-logits; optional focal / weighted BCE), backprop, clip, step.
    - Validation: loss + accuracy, precision, recall, F1.
    - LR schedule: ``linear_warmup`` or ``cosine`` (step-based, with warmup).
    - Optional early stopping on epoch-end ``metric_for_best`` (``early_stopping_patience`` > 0).
    - Epoch logs compare batch-averaged train loss with validation; optional full train-set metrics in eval mode.
    - Checkpoints: model, optimizer, scheduler, step; optional W&B logging.

    ``metric_for_best`` should name a validation metric where **higher is better**
    (e.g. ``f1``, ``accuracy``). Validation also reports ``loss`` (lower is better; not used for best selection unless you extend the trainer).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-5,
        epochs: int = 10,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Path | str = "models/checkpoints",
        save_steps: int = 500,
        eval_steps: int = 250,
        metric_for_best: str = "f1",
        device: torch.device | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        scheduler_type: str = "linear_warmup",
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        log_train_metrics_each_epoch: bool = False,
        adversarial_augmentation: dict[str, Any] | None = None,
        use_amp: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.metric_for_best = metric_for_best
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.log_train_metrics_each_epoch = log_train_metrics_each_epoch
        self._best_epoch_end_metric = float("-inf")
        self._epochs_without_epoch_improve = 0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_amp = bool(use_amp) and self.device.type == "cuda"
        self._scaler: torch_amp.GradScaler | None = (
            torch_amp.GradScaler("cuda") if self._use_amp else None
        )

        try:
            self.model.to(self.device)
        except Exception:
            pass

        total_steps = max(1, len(train_loader) * epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        self._trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not self._trainable_params:
            logger.warning("No trainable parameters found; check LoRA and head setup.")
        self.optimizer = AdamW(
            self._trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            self.scheduler_type,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        else:
            self.criterion = criterion.to(self.device)
        self.best_metric = -1.0
        self.global_step = 0

        self._wandb_active = _maybe_wandb_init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
        )
        self._adv_augment: Callable[[dict[str, Any], int], dict[str, Any]] | None = (
            build_adversarial_train_augment(adversarial_augmentation or {})
        )

    def train_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward + loss (caller runs backward / step)."""
        if self._adv_augment is not None:
            batch = self._adv_augment(batch, self.global_step)
        return multimodal_forward_and_loss(
            self.model, batch, self.device, self.criterion, use_amp=self._use_amp
        )

    def evaluate(self) -> dict[str, float]:
        """Validation metrics and mean val loss under keys ``accuracy``, ``precision``, ``recall``, ``f1``, ``loss``."""
        metrics, mean_loss = validate_multimodal(
            self.model,
            self.val_loader,
            self.device,
            self.criterion,
            use_amp=self._use_amp,
        )
        out = {**metrics, "loss": mean_loss}
        return out

    def save_checkpoint(
        self,
        name: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.checkpoint_dir / name
        state: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }
        if extra:
            state["extra"] = extra
        torch.save(state, path)
        logger.info("Saved checkpoint %s", path)
        if self._wandb_active:
            try:
                import wandb

                wandb.save(str(path), base_path=str(self.checkpoint_dir))
            except Exception:
                pass
        return path

    def _log_metrics(self, metrics: dict[str, float], step: int | None = None, prefix: str = "") -> None:
        if not self._wandb_active:
            return
        try:
            import wandb

            d = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(d, step=step or self.global_step)
        except Exception:
            pass

    def _log_validation(self, metrics: dict[str, float], context: str) -> None:
        logger.info(
            "%s — loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f",
            context,
            metrics.get("loss", 0.0),
            metrics.get("accuracy", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1", 0.0),
        )

    def train(self) -> dict[str, float]:
        self.model.train()
        early_stopped = False
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self.train_step(batch)
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    if self.max_grad_norm > 0:
                        self._scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self._trainable_params, self.max_grad_norm)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._trainable_params, self.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])

                if self._wandb_active:
                    self._log_metrics(
                        {"train/loss": loss.item(), "train/lr": self.scheduler.get_last_lr()[0]}
                    )

                if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                    metrics = self.evaluate()
                    self._log_validation(metrics, f"Step {self.global_step} val")
                    self._log_metrics(metrics, prefix="val/")
                    m = metrics.get(self.metric_for_best, -1)
                    if m > self.best_metric:
                        self.best_metric = m
                        self.save_checkpoint("best.pt", extra={"metric_for_best": self.metric_for_best})
                        if self._wandb_active:
                            self._log_metrics({"best/" + self.metric_for_best: m})

                if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}.pt")

            avg_loss = epoch_loss / num_batches if num_batches else 0.0
            train_eval_metrics: dict[str, float] | None = None
            train_eval_mean_loss = avg_loss
            if self.log_train_metrics_each_epoch:
                train_eval_metrics, train_eval_mean_loss = metrics_on_loader(
                    self.model,
                    self.train_loader,
                    self.device,
                    self.criterion,
                    desc=f"Epoch {epoch + 1} train (eval)",
                    restore_training=True,
                    use_amp=self._use_amp,
                )

            val_metrics = self.evaluate()
            lr_now = self.scheduler.get_last_lr()[0]
            epoch_m = val_metrics.get(self.metric_for_best, float("-inf"))

            if epoch_m > self.best_metric:
                self.best_metric = epoch_m
                self.save_checkpoint("best.pt", extra={"metric_for_best": self.metric_for_best})
                if self._wandb_active:
                    self._log_metrics({"best/" + self.metric_for_best: self.best_metric})

            if epoch_m > self._best_epoch_end_metric + self.early_stopping_min_delta:
                self._best_epoch_end_metric = epoch_m
                self._epochs_without_epoch_improve = 0
            else:
                self._epochs_without_epoch_improve += 1

            patience_disp = (
                f"{self._epochs_without_epoch_improve}/{self.early_stopping_patience}"
                if self.early_stopping_patience > 0
                else "off"
            )

            if train_eval_metrics is not None:
                logger.info(
                    "Epoch %d/%d — train (eval): loss=%.4f acc=%.4f prec=%.4f rec=%.4f f1=%.4f | "
                    "val: loss=%.4f acc=%.4f prec=%.4f rec=%.4f f1=%.4f | "
                    "best_%s=%.4f | lr=%.2e | batch_avg_train_loss=%.4f | epoch_patience %s",
                    epoch + 1,
                    self.epochs,
                    train_eval_mean_loss,
                    train_eval_metrics.get("accuracy", 0.0),
                    train_eval_metrics.get("precision", 0.0),
                    train_eval_metrics.get("recall", 0.0),
                    train_eval_metrics.get("f1", 0.0),
                    val_metrics.get("loss", 0.0),
                    val_metrics.get("accuracy", 0.0),
                    val_metrics.get("precision", 0.0),
                    val_metrics.get("recall", 0.0),
                    val_metrics.get("f1", 0.0),
                    self.metric_for_best,
                    self.best_metric,
                    lr_now,
                    avg_loss,
                    patience_disp,
                )
            else:
                logger.info(
                    "Epoch %d/%d — train (batch-avg loss): %.4f | "
                    "val: loss=%.4f acc=%.4f prec=%.4f rec=%.4f f1=%.4f | "
                    "best_%s=%.4f | lr=%.2e | epoch_patience %s",
                    epoch + 1,
                    self.epochs,
                    avg_loss,
                    val_metrics.get("loss", 0.0),
                    val_metrics.get("accuracy", 0.0),
                    val_metrics.get("precision", 0.0),
                    val_metrics.get("recall", 0.0),
                    val_metrics.get("f1", 0.0),
                    self.metric_for_best,
                    self.best_metric,
                    lr_now,
                    patience_disp,
                )

            if self._wandb_active:
                epoch_log: dict[str, float] = {
                    "train/epoch_loss": avg_loss,
                    "val/loss": val_metrics.get("loss", 0.0),
                    "val/accuracy": val_metrics.get("accuracy", 0.0),
                    "val/precision": val_metrics.get("precision", 0.0),
                    "val/recall": val_metrics.get("recall", 0.0),
                    "val/f1": val_metrics.get("f1", 0.0),
                    "train/lr_epoch_end": lr_now,
                    "best/" + self.metric_for_best: self.best_metric,
                }
                if train_eval_metrics is not None:
                    epoch_log["train/eval_loss"] = train_eval_mean_loss
                    epoch_log["train/eval_accuracy"] = train_eval_metrics.get("accuracy", 0.0)
                    epoch_log["train/eval_precision"] = train_eval_metrics.get("precision", 0.0)
                    epoch_log["train/eval_recall"] = train_eval_metrics.get("recall", 0.0)
                    epoch_log["train/eval_f1"] = train_eval_metrics.get("f1", 0.0)
                self._log_metrics(epoch_log, step=self.global_step)

            if (
                self.early_stopping_patience > 0
                and self._epochs_without_epoch_improve >= self.early_stopping_patience
            ):
                logger.info(
                    "Early stopping: no epoch-end improvement in %s for %d epochs (min_delta=%s).",
                    self.metric_for_best,
                    self.early_stopping_patience,
                    self.early_stopping_min_delta,
                )
                early_stopped = True
                break

        if self._wandb_active:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        return {
            "best_" + self.metric_for_best: self.best_metric,
            "early_stopped": early_stopped,
        }
