# Scripts

Entrypoints for data, training, and evaluation. All accept config overrides and log to W&B where applicable.

| Script | Purpose |
|--------|--------|
| `run_feed_fetch.py` | Fetch URLs from all feeds; write to `data/raw/feeds/`. |
| `run_crawl.py` | Crawl URLs from feed output; save screenshots and DOM. |
| `run_preprocess.py` | Build train/val/test from crawl output. |
| `run_train.py` | Train model; config e.g. `configs/training.yaml`. |
| `run_eval.py` | Run evaluation on configured test sets and write results. |
| `run_adversarial.py` | Run adversarial evaluation suite. |
| **utils/** | Shared helpers (logging, path resolution, config loading). |
