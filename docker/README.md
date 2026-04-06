# Docker

Reproducible **Python 3.11** environment with **PyTorch (CPU)**, **Playwright (Chromium)**, and all `requirements.txt` packages.

## Quick start — API + web UI

From the **repository root**:

```bash
docker compose up --build phishguard
```

Open `http://localhost:8000/` (UI) and `http://localhost:8000/docs` (OpenAPI).  
First start downloads Hugging Face weights into the `huggingface_cache` volume (slow, large disk use).

## Run pipeline commands

`docker compose run` overrides the default API command:

```bash
docker compose run --rm phishguard python scripts/run_crawl.py --help
docker compose run --rm phishguard python scripts/run_preprocess.py --help
docker compose run --rm phishguard python scripts/run_train.py --help
docker compose run --rm phishguard python scripts/run_eval.py --help
docker compose run --rm phishguard python scripts/run_inference.py "https://example.com"
docker compose run --rm phishguard pytest tests/ -q
```

Host directories **`./data`** and **`./models/checkpoints`** are mounted so crawls, manifests, and checkpoints persist.

## GPU image (optional)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker build -f docker/Dockerfile.gpu -t phishguard-vlm:gpu .
docker run --gpus all -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models/checkpoints:/app/models/checkpoints" \
  -e PHISHGUARD_PROJECT_ROOT=/app \
  phishguard-vlm:gpu
```

## Build notes

- **CPU** default: `docker/Dockerfile` installs PyTorch from the official CPU wheel index.
- **Context** is the repo root; large paths are skipped via `.dockerignore` (`data/`, `checkpoints/`) and mounted at runtime instead.
- **Playwright** Chromium is baked into the image (no extra volume) so crawls work immediately after `compose up`.
