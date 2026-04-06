# Configs

YAML configuration for all subsystems. Design: one file per concern so experiments can override only what they need.

| File | Purpose |
|------|--------|
| `default.yaml` | Shared defaults (paths, seeds, logging) |
| `data.yaml` | Feed URLs, crawl timeouts, output paths, split ratios |
| `model.yaml` | LLaVA variant, image size, LoRA rank, classification head |
| `training.yaml` | Batch size, LR, epochs, W&B project, checkpointing |
| `evaluation.yaml` | Test set paths, metrics list, adversarial config |

Scripts and the API load config via a single loader that merges `default` + chosen override (e.g. `training_lora.yaml`).
