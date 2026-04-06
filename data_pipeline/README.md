# Data Pipeline

End-to-end dataset creation: fetch URLs from feeds → crawl with Playwright → preprocess → train/val/test splits.

## Subdirectories

| Directory | Responsibility |
|-----------|-----------------|
| **feeds/** | Fetchers for OpenPhish, PhishTank, APWG (phishing) and Tranco (benign). Output: URL lists with label and source. |
| **crawler/** | Playwright-based renderer: load URL, capture full-page screenshot, extract DOM text. Same component used by inference for on-demand URL analysis. |
| **preprocessing/** | Image resize/normalize for VLM; text cleaning and truncation; stratified splits; serialization (e.g. Parquet + image paths or HF Dataset). |

## Design

- Screenshot and DOM are captured in the same page load for alignment.
- Crawler is shared between data pipeline and inference service.
- Target scale: ~50k phishing + ~50k benign URLs (actual size after crawl success reported in experiments).
