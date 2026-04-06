# Models

VLM-based phishing classifier: LLaVA-1.5 7B + LoRA + classification head.

## Subdirectories

| Directory | Responsibility |
|-----------|-----------------|
| **backbones/** | LLaVA wrapper (vision encoder + projector + LLM), image and text tokenization. |
| **heads/** | Binary classification head on top of pooled/last hidden state. |
| **lora/** | LoRA config and application (PEFT) on LLM (and optionally projector). |
| **training/** | Training loop, dataloader, checkpointing, W&B logging. |

## Inputs/Outputs

- **Input**: Webpage screenshot (PIL/image) + webpage text string.
- **Output**: Phishing probability (0–1), classification label (0 = benign, 1 = phishing).

## Training

Run `python scripts/run_train.py` after building the processed manifest (`run_preprocess.py`). Config: `configs/model.yaml`, `configs/training.yaml`, `configs/default.yaml`.
