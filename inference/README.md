# Inference

End-to-end pipeline and API for live URL analysis.

## Components

| Component | Responsibility |
|-----------|-----------------|
| **pipeline.py** | Orchestrate: crawl URL → preprocess → VLM forward → optional knowledge lookup → aggregate. |
| **aggregator.py** | Combine VLM probability with knowledge signals; produce final label, confidence, explanation, heatmap. |
| **api/** | FastAPI app: e.g. `POST /analyze` with `{"url": "..."}`; return JSON (label, confidence, explanation, heatmap). |

Model and LoRA loaded once at startup; optional response cache per URL.
