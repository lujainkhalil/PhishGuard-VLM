# Feed Fetchers

Fetch URL lists for the crawler (`scripts/run_feed_fetch.py` → `data/raw/feeds/`).

## Phishing

- **OpenPhish**: Streams the **entire** text feed per URL (no in-code row limit). Multiple `feed_urls` in `configs/data.yaml` are fetched and **merged** with URL-level deduplication (primary site + GitHub mirror).
- **PhishTank**: Downloads the full **`online-valid.json`** dump (long read timeout; set `PHISHTANK_API_KEY` for reliable access).

## Benign

- **Tranco**: Downloads the official **top-1M** CSV zip from [tranco-list.eu](https://tranco-list.eu/), maps each domain to `https://{domain}/`, normalizes, and writes **`tranco.json`** / **`tranco.csv`**. Defaults require **at least `min_urls`** (10,000) and cap at **`max_urls`** (see `configs/data.yaml`).

## Output shape

Each row: `{url, label, source, fetched_at, …}` — `label` is `phishing` or `benign`. The crawler’s `feed_loader` loads all `*.json` / `*.csv` in the feeds directory.

## Modules

- `openphish.py`, `phishtank.py`, `tranco.py`, `utils.py`
