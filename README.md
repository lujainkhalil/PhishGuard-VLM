# PhishGuard-VLM

Research-grade **multimodal phishing detection** using a Vision-Language Model (LLaVA 1.5) to analyze webpage screenshots and text simultaneously.

## Overview

- **Input**: URL
- **Processing**: Dynamic render (Playwright) → screenshot + DOM text → VLM (LLaVA-7B + LoRA) + optional Wikidata brand verification
- **Output**: Phishing/benign label, confidence, explanation, attention heatmap

## Repository Layout

| Directory | Purpose |
|-----------|--------|
| [`configs/`](configs/) | YAML configuration (data, model, training, evaluation) |
| [`data_pipeline/`](data_pipeline/) | Feed fetchers, crawler, preprocessing |
| [`models/`](models/) | LLaVA backbone, classification head, LoRA, training loop |
| [`knowledge_module/`](knowledge_module/) | Wikidata SPARQL client, brand–domain verification |
| [`inference/`](inference/) | End-to-end pipeline, decision aggregation, FastAPI |
| [`evaluation/`](evaluation/) | Metrics, test sets, adversarial robustness |
| [`web_app/`](web_app/) | Demo UI (React or HTML) |
| [`scripts/`](scripts/) | Entrypoints (fetch, crawl, train, eval) |
| [`tests/`](tests/) | Unit, integration, e2e |
| [`docs/`](docs/) | [Architecture](docs/ARCHITECTURE.md), ADRs |

## Tech Stack

Python 3.11 · PyTorch · HuggingFace Transformers · Playwright · FastAPI · W&B · PyTest · Docker

## Quick Start (after implementation)

```bash
# Data (OpenPhish + PhishTank + Tranco benign list; see configs/data.yaml)
python scripts/run_feed_fetch.py
python scripts/run_crawl.py
python scripts/run_preprocess.py   # optional: --merge-manifest for extra crawl JSON files

# Train
python scripts/run_train.py --config configs/training.yaml

# Evaluate
python scripts/run_eval.py

# Run API + demo
uvicorn inference.api.app:app --reload
# Open web_app/frontend or serve static build
```

## Design

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for full system architecture, design choices, and implementation order.
