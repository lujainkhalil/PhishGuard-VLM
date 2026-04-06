# Design Choices Summary

Short reference for dissertation or reviews: why key decisions were made.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **VLM** | LLaVA-1.5 7B | Strong instruction-following, single model for image+text, good balance of capacity vs GPU memory; 13B optional for scale-up. |
| **Fine-tuning** | LoRA on LLM (and optionally projector) | Fewer trainable parameters, less overfitting, faster iteration; full fine-tune possible if resources allow. |
| **Crawler** | Playwright | Modern JS support, stable screenshots and DOM from same load, better than Selenium for research reproducibility. |
| **Knowledge** | Wikidata SPARQL + cache | Public, structured brand/domain data; cache avoids rate limits and keeps latency low. |
| **Config** | YAML, one file per concern | Reproducible runs; experiments override only data/training/eval as needed. |
| **Experiment tracking** | Weights & Biases | Standard for ML; links config, metrics, and checkpoints for dissertation figures and tables. |
| **API** | FastAPI | Async, OpenAPI, simple to wire inference pipeline and serve the web app. |
| **Evaluation** | Held-out + temporal + zero-shot + adversarial | Matches proposal: performance, generalization, and robustness with <5% degradation target. |
| **Repository layout** | data_pipeline, models, knowledge_module, inference, evaluation, web_app, scripts, tests, configs | Clear boundaries; data pipeline and inference share crawler; evaluation is self-contained for reproducibility. |
