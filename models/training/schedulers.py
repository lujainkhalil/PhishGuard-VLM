"""
Learning-rate schedulers for multimodal training (step-based).
"""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer

try:
    from transformers import get_cosine_schedule_with_warmup
    from transformers import get_linear_schedule_with_warmup as hf_linear_warmup
except ImportError:  # pragma: no cover
    get_cosine_schedule_with_warmup = None  # type: ignore[misc, assignment]
    hf_linear_warmup = None  # type: ignore[misc, assignment]

try:
    from torch.optim.lr_scheduler import get_linear_schedule_with_warmup as torch_linear_warmup
except ImportError:  # pragma: no cover
    torch_linear_warmup = None  # type: ignore[misc, assignment]


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Any:
    """
    Step-based LR schedule tied to optimizer steps (matches existing trainer loop).

    ``scheduler_type``:
        - ``linear_warmup`` (default): linear decay after warmup (HF or PyTorch helper).
        - ``cosine`` / ``cosine_warmup``: cosine decay after warmup (requires ``transformers``).
    """
    st = (scheduler_type or "linear_warmup").strip().lower()
    if st in ("linear", "linear_warmup", "default"):
        if hf_linear_warmup is not None:
            return hf_linear_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if torch_linear_warmup is not None:
            return torch_linear_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        raise ImportError(
            "linear_warmup schedule needs transformers or a PyTorch build with "
            "get_linear_schedule_with_warmup"
        )

    if st in ("cosine", "cosine_warmup"):
        if get_cosine_schedule_with_warmup is None:
            raise ImportError("cosine schedule requires transformers; pip install transformers")
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    raise ValueError(f"Unknown scheduler_type: {scheduler_type!r} (use linear_warmup | cosine)")
