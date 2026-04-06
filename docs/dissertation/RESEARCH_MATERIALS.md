# PhishGuard-VLM — Research Materials for Dissertation

This document supports thesis writing: **experiment logging**, **evaluation reporting**, **design rationale**, and **interpretation of results**. It is aligned with the repository layout (`configs/`, `evaluation/`, `models/training/`, `inference/`, `docs/`). **Numerical results** are intentionally left as placeholders (`—` or `[to be filled]`) so you paste outcomes from your own runs.

---

## Part A — Experiment logging

### A.1 What constitutes an experiment log

A complete experiment record should allow a third party to **reproduce** the run and to **attribute** reported numbers. The implementation records:

| Channel | Content | Location / mechanism |
|--------|---------|----------------------|
| **Structured config** | Model ID, LoRA hyperparameters, head architecture, prompt template, data paths | `configs/model.yaml`, `configs/data.yaml`, `configs/evaluation.yaml` |
| **Training loop** | Training loss (per epoch / step), validation metrics (accuracy, precision, recall, F1, ROC-AUC when scores available), learning rate | Python `logging` + optional **Weights & Biases** (`wandb.log`) |
| **Checkpoints** | Model state, optimizer, scheduler, global step; best checkpoint selected by configurable metric (default F1) | `models/checkpoints/best.pt`, `checkpoint-{step}.pt` |
| **Version control** | Git commit hash | Record manually in run table or W&B run notes |

When W&B is enabled (`wandb_project` in trainer configuration), the run stores **hyperparameters**, **time series** of train/val metrics, and **artifact links** to checkpoints. For dissertation appendices, export run URLs and screenshot key panels (loss curves, validation F1).

### A.2 Recommended fields for each logged run

Use one row per training run (spreadsheet or thesis table):

| Field | Description |
|-------|-------------|
| Run ID | Short identifier (e.g. `exp-llava-lora-r16-2026-03`) |
| Date | UTC date of run |
| Git commit | SHA |
| Config hash | Optional: `sha256sum` of merged YAML or `configs/*.yaml` set |
| Seed | Random seed(s) for splits and training |
| Hardware | GPU model, VRAM, batch size |
| Train / val / test | Row counts and split policy (`stratified`, `stratified_domain`, `temporal`, `auto`) |
| Best epoch / step | From checkpoint metadata |
| Best validation metric | Name (e.g. F1) and value |
| Checkpoint path | Relative path to `best.pt` |
| W&B link | If applicable |

### A.3 Narrative paragraph (dissertation-ready)

> Training runs were configured via version-controlled YAML files specifying the vision-language backbone (LLaVA-1.5 7B), low-rank adaptation of attention projections, and a binary classification head. Metrics on the validation set were computed each epoch; the checkpoint maximizing the selected validation criterion was retained. Optional integration with Weights & Biases provided time-series logs and persistent links for supplementary material. Reproducibility was supported by fixed random seeds for data splitting and training, together with explicit recording of software versions and commit identifiers.

---

## Part B — Evaluation summaries

### B.1 Metrics reported (definitions)

| Metric | Role in phishing detection | Typical use in thesis |
|--------|----------------------------|------------------------|
| **Accuracy** | Overall fraction correct | headline number; can mask imbalance |
| **Precision** | Of predicted phishing, fraction truly phishing | operational cost of false alarms |
| **Recall** | Of true phishing, fraction detected | user safety; missed attacks |
| **F1** | Harmonic mean of precision and recall | single scalar under imbalance |
| **ROC-AUC** | Discrimination across thresholds | threshold-independent ranking quality |

Confusion-matrix counts (TN, FP, FN, TP) support qualitative error analysis and cost-sensitive discussion.

### B.2 Main results table (template)

**Markdown (fill after `run_eval.py`):**

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | *n* |
|-----------|----------|-----------|--------|-----|---------|-----|
| Held-out test | — | — | — | — | — | — |
| TR-OP (benchmark) | — | — | — | — | — | — |
| + Knowledge fusion (inference) | — | — | — | — | — | — |

**LaTeX (paste into thesis):**

```latex
\begin{table}[t]
  \centering
  \caption{PhishGuard-VLM performance on held-out and benchmark test sets.}
  \label{tab:main-results}
  \begin{tabular}{lcccccc}
    \toprule
    Condition & Acc. & Prec. & Rec. & F1 & AUC & $n$ \\
    \midrule
    Held-out test & -- & -- & -- & -- & -- & -- \\
    TR-OP         & -- & -- & -- & -- & -- & -- \\
    \bottomrule
  \end{tabular}
\end{table}
```

### B.3 Adversarial robustness summary (template)

The adversarial runner (`scripts/run_adversarial_eval.py`) evaluates the **same** test subset under: HTML-style obfuscation, typosquatting-style text edits, simulated screenshot noise/blur, and prompt-injection strings appended to page text. Report **baseline** metrics and **per-attack** metrics, plus **drops** (e.g. $\Delta\text{Acc} = \text{Acc}_{\text{clean}} - \text{Acc}_{\text{attack}}$).

| Attack | Accuracy (clean ref.) | Accuracy (attacked) | $\Delta$ Acc | $\Delta$ F1 |
|--------|------------------------|----------------------|--------------|-------------|
| (baseline) | — | — | 0 | 0 |
| HTML obfuscation | — | — | — | — |
| Typosquatting | — | — | — | — |
| Logo / screenshot (sim.) | — | — | — | — |
| Prompt injection | — | — | — | — |

**Dissertation sentence:**

> Robustness probes measured the change in accuracy and F1 relative to unperturbed inputs, using evaluation-only transformations that approximate evasive HTML, domain-like typosquatting noise, capture degradation on screenshots, and adversarial text appended to the extracted document body.

### B.4 Ablations (template)

| Variant | F1 (held-out) | Notes |
|---------|---------------|--------|
| Full (VLM + LoRA + head) | — | primary system |
| No LoRA (frozen backbone, trainable head only if applicable) | — | isolate adaptation |
| Text-only / image-only | — | modality contribution |
| Model without knowledge fusion | — | `aggregate_signals` vs model-only prior |
| No adversarial training (if applicable) | — | train-time vs eval-only robustness |

### B.5 File artefacts to cite in appendices

| Artefact | Path (typical) |
|----------|----------------|
| Standard eval metrics JSON | `evaluation/results/eval_metrics_*.json` |
| Per-URL predictions | `evaluation/results/eval_predictions_*.csv` |
| Adversarial JSON report | `evaluation/results/adversarial_report_*.json` |
| Adversarial CSV table | `evaluation/results/tables/adversarial.csv` |

---

## Part C — Design justifications

### C.1 Vision-language model (LLaVA-1.5 7B)

Phishing pages are inherently **multimodal**: layout, branding, and visual mimicry interact with **lexical** cues (urgency, credential requests, URL mentions). A single **vision-language model** processes screenshot and text jointly, avoiding hand-engineered fusion of separate vision and NLP pipelines. LLaVA-1.5 offers a practical trade-off between **capacity** and **compute** for academic and prototype deployment; larger variants can be discussed as future work.

### C.2 Low-rank adaptation (LoRA)

Full fine-tuning of a multi-billion-parameter VLM is **data- and memory-intensive** and risks **catastrophic forgetting** of general capabilities. **LoRA** restricts trainable weights to low-rank updates of selected linear layers (here, attention projections), while a small **MLP classification head** maps pooled representations to a phishing logit. This yields **fewer trainable parameters**, **faster iteration**, and **checkpoint sizes** suitable for distribution and ablation studies. Gradient checkpointing and controlled use of the multimodal projector further reduce memory use.

### C.3 Data pipeline: crawl, preprocess, splits

**Playwright** provides a reproducible path from URL to **rendered DOM** and **full-page screenshot**, supporting modern JavaScript-heavy pages. Text extraction and normalization align training and inference. **Splits** (stratified, domain-disjoint, temporal, or automatic choice based on timestamp coverage) address **leakage**: the same registrable domain should not appear in both train and test when using domain-aware splitting, and temporal splits support claims about **forward generalization** when timestamps are trustworthy.

### C.4 Knowledge module and aggregation

Brand-centric phishing often hinges on whether the **served hostname** aligns with **official** web properties. The knowledge pathway resolves brand entities (e.g. via **Wikidata**) or accepts explicit official domains, then applies **heuristic impersonation signals**. The **aggregator** combines the model’s phishing probability with a knowledge-derived prior using **dynamic weights** and **agreement / conflict** rules, producing a **fused probability**, **discrete label**, **confidence** (distance from 0.5), and a **natural-language explanation** suitable for user-facing systems and thesis discussion of **interpretability**.

### C.5 Inference and deployment

The **URL inference pipeline** chains crawl → preprocess → model → optional knowledge → aggregation, exposing a single interface for scripts and a **FastAPI** `/predict` endpoint. **Docker** images pin Python and system dependencies (including Chromium for crawling), supporting **reproducible** experiments and deployment narratives in the thesis.

### C.6 Evaluation design

Evaluation spans **standard** classification metrics, **benchmark** comparison (e.g. TR-OP, with external baselines cited from literature where code is unavailable), **ablations**, and **adversarial** probes. The design separates **training** data from **evaluation-only** benchmarks and records outputs under `evaluation/results/` for tables and figures.

---

## Part D — Explaining key results (interpretation guide)

### D.1 Accuracy in a security context

High **accuracy** alone can be misleading if the test set is **imbalanced** (many benign pages, few phishing). The thesis should pair accuracy with **precision/recall/F1** and, where relevant, **per-class** error rates. **False negatives** (missed phishing) and **false positives** (benign flagged) map directly to user risk and operational burden; a short discussion of **cost asymmetry** strengthens the results chapter.

### D.2 ROC-AUC vs thresholded metrics

**ROC-AUC** reflects ranking quality across thresholds and is useful for comparing models **without** committing to a single operating point. **F1**, precision, and recall at a fixed threshold (e.g. 0.5 on fused probability) match **deployed** behaviour. Reporting **both** clarifies whether gains are **global** (better ranking) or **local** (better at default threshold).

### D.3 Adversarial degradation

A **drop** in accuracy under perturbation indicates **brittleness** to that perturbation class. **Simulated** logo manipulation (blur/noise on the full screenshot) approximates capture noise rather than targeted patch attacks; the thesis should state this **scope**. **Prompt-injection** strings test whether junk text in the page body shifts the VLM’s decision—relevant to **robustness** of multimodal prompts. Framing: *degradation quantifies sensitivity; the engineering goal is bounded degradation under realistic noise.*

### D.4 Model–knowledge fusion

When **model probability** and **knowledge prior** **agree**, the fused score tends toward stronger **confidence**. **Conflict** (e.g. low model score but high impersonation risk) triggers explicit rules in the aggregator; discussing one or two **concrete examples** from error analysis (URL, screenshot path, explanation string) illustrates **traceability** for examiners.

### D.5 Limitations and threats to validity

Suggested thesis bullets:

- **Temporal drift**: phishing tactics and benign web design evolve; static test sets become dated.
- **Crawl failures**: pages that timeout or block crawlers are underrepresented; inference returns explicit failure explanations.
- **Knowledge coverage**: Wikidata and heuristics do not cover all brands or all attack types.
- **Simulated attacks**: adversarial evaluation is **approximate**; not a certificate of security.
- **Compute**: 7B VLMs require substantial resources; results may depend on GPU class and batch settings.

---

## Part E — One-page reproducibility checklist (for appendix)

- [ ] Record git commit and `configs/*.yaml` (or hash) with each experiment.  
- [ ] Fix seeds for splits (`data_pipeline/preprocessing/splits.py` / build config) and training.  
- [ ] Store `best.pt` path and validation score used for selection.  
- [ ] Run `scripts/run_eval.py` with the same manifest and checkpoint; archive JSON + CSV under `evaluation/results/`.  
- [ ] Run `scripts/run_adversarial_eval.py` with `configs/evaluation.yaml` adversarial block; archive report + CSV.  
- [ ] Note Docker image tag or `Dockerfile` digest if containerized.  
- [ ] Export W&B run URL or disable W&B and rely on file logs consistently.

---

## Related repository documents

- `docs/DESIGN_CHOICES.md` — concise decision table.  
- `docs/EVALUATION_PLAN.md` — evaluation plan overview.  
- `docs/ARCHITECTURE.md` — system structure.  
- `evaluation/README.md` — evaluation subdirectory roles.  
- `docker/README.md` — containerised reproduction.

---

*End of research materials. Insert empirical numbers from your completed runs; keep config paths and commit IDs versioned with the thesis submission.*
