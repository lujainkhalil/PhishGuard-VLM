# Measured pipeline run

**Timestamp (UTC):** `20260406_204505`  
**Elapsed:** 16.17 s  

## Pytest

- Exit code: `0`
- Parsed: passed=61, failed=0, skipped=4

## Feed fetch (OpenPhish)

- {
  "exit_code": 0,
  "openphish_entries_reported": 300,
  "log_tail": "2026-04-06 21:45:16 [INFO] data_pipeline.feeds.openphish: Fetching OpenPhish feed from https://openphish.com/feed.txt\n2026-04-06 21:45:16 [INFO] data_pipeline.feeds.openphish: Fetched 300 URLs from OpenPhish (300 lines)\n2026-04-06 21:45:16 [INFO] data_pipeline.feeds.utils: Wrote 300 entries to data/raw/feeds/openphish.json\n2026-04-06 21:45:16 [INFO] __main__: OpenPhish: 300 entries written to data/raw/feeds/openphish.json\n"
}

## Synthetic score metrics (n=500, reproducible RNG)

- At threshold 0.5: {'accuracy': 0.676, 'precision': 0.7269076305220884, 'recall': 0.6581818181818182, 'f1': 0.6908396946564885, 'roc_auc': 0.742189898989899}
- Best F1 threshold sweep: τ=0.2300, F1=0.7492
- Metrics at τ*: {'accuracy': 0.668, 'precision': 0.6408268733850129, 'recall': 0.9018181818181819, 'f1': 0.7492447129909365}

## Tiny MLP head (proxy for train loop)

```json
{
  "status": "skipped",
  "reason": "torch not installed"
}
```

## Full VLM eval

```json
{
  "skipped": true,
  "reason": "pass --attempt-vlm-eval to run (downloads LLaVA; needs disk/GPU)"
}
```

## Crawl smoke test (post-run)

- Attempted **2** OpenPhish URLs via `scripts/run_crawl.py --limit 2`: **0** reached `status: ok` (DNS failure on one, navigation timeout on one). Manifest: `data/crawl_manifest.json`.
- **Preprocess** not run afterward (no `ok` rows to materialize). See `pipeline_extensions.json` for detail.

## Artefacts

- JSON: `evaluation/results/measured_runs/20260406_204505/report.json`
- Extensions: `evaluation/results/measured_runs/20260406_204505/pipeline_extensions.json`
- Synthetic data: `evaluation/results/measured_runs/20260406_204505/synthetic_processed`

> **Note:** End-to-end LLaVA training/evaluation requires `pip install -r requirements.txt`, GPU memory, crawled data, and `python scripts/run_train.py` / `run_eval.py`. This report records what ran successfully in the current environment.