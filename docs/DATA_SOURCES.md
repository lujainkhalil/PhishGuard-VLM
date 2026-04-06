# Data Sources

Reference for dataset creation and evaluation. Update with actual URLs, formats, and licensing when implementing.

## Phishing URL Feeds

| Source | URL / API | Format | Notes |
|--------|-----------|--------|--------|
| **OpenPhish** | `https://openphish.com/feed.txt` (+ GitHub mirror in `configs/data.yaml`) | One URL per line; full feed streamed | Free community feed; merged mirrors in `run_feed_fetch.py`. |
| **PhishTank** | https://phishtank.org/ | API (requires API key) | Set `PHISHTANK_API_KEY` in env. |
| **APWG** | (TBD: member/feed URL) | As per APWG docs | May require membership for bulk. |

## Benign URL Feeds

| Source | URL | Format | Notes |
|--------|-----|--------|--------|
| **Tranco** | `https://tranco-list.eu/top-1m.csv.zip` | Zipped CSV `rank,domain` | Benign `https://{domain}/` rows; default ≥10k URLs via `feeds.benign.tranco` in `configs/data.yaml`. |

## Benchmark (Evaluation Only)

| Dataset | Source | Use |
|---------|--------|-----|
| **TR-OP** | KnowPhish paper / authors or replication package | Evaluation only. Same dataset as KnowPhish (92.05% F1). Do not train on TR-OP. Store under `data/benchmarks/tr-op/`. Document license and citation. |

## Knowledge

| Source | URL | Use |
|--------|-----|-----|
| **Wikidata** | https://query.wikidata.org/ (SPARQL) | Brand official domains, aliases, logos. Public; cache responses. |

## Manifest and Class Counts

After preprocessing, the manifest (see ARCHITECTURE §2.3) should include enough metadata to report class counts for training (for balanced sampling or weighted loss) and to construct temporal/zero-shot splits (crawl_date, brand_id).
