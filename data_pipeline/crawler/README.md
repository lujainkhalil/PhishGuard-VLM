# Crawler

Playwright-based webpage renderer used by both the data pipeline and the inference API.

- **Input**: URL, optional timeout and viewport (from `configs/data.yaml` when using `scripts/run_crawl.py`).
- **Output**: Full-page screenshot (PNG in `data/screenshots/`), sanitized DOM text (`.txt` in `data/pages/`), final URL after redirects, status.
- **Policies**: Configurable timeouts (default 60s in `configs/data.yaml`), up to **three** full navigation attempts per URL with backoff, structured **error categories** (DNS, timeout, …), **permanent** failures (e.g. DNS NXDOMAIN) are not retried wastefully and are **skipped on resume** in `scripts/run_crawl.py`.

## Modules

- **crawler.py**: `crawl_url()` and `crawl_url_with_retries()` — open URL with Playwright, wait for DOM/JS, capture full-page screenshot and extract text (scripts/styles stripped, whitespace normalized).
- **feed_loader.py**: `load_urls_from_feeds()` — load URLs from feed JSON/CSV files in `data/raw/feeds/`.

## Usage

Pipeline: run `scripts/run_feed_fetch.py`, then `scripts/run_crawl.py`. Screenshots and text are written to `data/screenshots` and `data/pages`; a manifest is written to `data/crawl_manifest.json`.
