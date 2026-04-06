# Preprocessing

Transform raw crawl outputs into a multimodal dataset (screenshot + text + domain + label) with train/validation/test splits.

- **Input**: Crawl manifest JSON from `run_crawl.py` (url, status, screenshot_path, text_path, label, source, redirect_count).
- **Merging phishing + benign**: If you crawled sources into separate manifests, either:
  - `python scripts/run_preprocess.py --crawl-manifest data/crawl_phish.json --merge-manifest data/crawl_benign.json` (primary first; URL dedupe keeps the **first** row per URL), or
  - `python scripts/merge_crawl_manifests.py -o data/crawl_merged.json a.json b.json` then preprocess with a single `--crawl-manifest`.
- **Quality filters** (`quality` in `configs/data.yaml`): Drop failed crawls, max redirects, min text length, and optionally invalid screenshot files (PIL verify + minimum byte size / edge length).
- **Dataset validation** (`dataset_validation` in `configs/data.yaml`): After materialization, `run_preprocess.py` runs `validate_processed_manifest()` — removes empty/short text, unreadable screenshots, and duplicate URLs (and normalized URLs). Logs statistics via `log_manifest_statistics()`; writes `<stem>.dataset_validation.json` next to the manifest when enabled.
- **Standalone**: `python scripts/validate_dataset.py --manifest path.parquet` to clean an existing manifest.
- **Metadata**: Domain extracted from final URL; label (phishing/benign) and source preserved.
- **Splits**: Domain-aware / temporal / stratified modes (see `configs/data.yaml`).
- **Output**: Manifest Parquet/CSV under `data/processed/` with `text`, `image_path`, `text_path`, etc.

Image resizing to VLM resolution uses `prepare_image` during materialization (default 336px square).
