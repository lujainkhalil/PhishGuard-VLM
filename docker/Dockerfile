# PhishGuard-VLM — reproducible CPU image (PyTorch CPU + Playwright Chromium).
# For GPU, use docker/Dockerfile.gpu or pass --gpus with a CUDA base (see docker/README.md).

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="PhishGuard-VLM"
LABEL org.opencontainers.image.description="Phishing detection: crawl, preprocess, VLM, FastAPI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PHISHGUARD_PROJECT_ROOT=/app \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Playwright system deps + curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# CPU PyTorch first (smaller, portable); then the rest of requirements
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    playwright install-deps chromium && \
    playwright install chromium

COPY . /app

EXPOSE 8000

# Override with: docker compose run phishguard python scripts/run_preprocess.py ...
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
