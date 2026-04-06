# Architecture Review: Gaps and Improvements

Pre-implementation review of the PhishGuard-VLM architecture and repository for: **multimodal training**, **dataset creation**, **adversarial testing**, and **dissertation evaluation**. Items marked **[ADDED]** have been reflected in the architecture or configs.

---

## 1. Multimodal Training

### Gaps identified

| Gap | Risk | Improvement |
|-----|------|-------------|
| **No explicit instruction/prompt template** for the VLM during training | Inconsistent task framing; poor convergence | Define a single canonical prompt template (e.g. "Analyze this webpage screenshot and text. Is it phishing?") and use it for both training and inference. Document in `model.yaml` and in ARCHITECTURE §2.4. **[ADDED]** |
| **Vision encoder / projector trainability** unspecified | Unclear whether to freeze or train; affects reproducibility | Document in config: freeze_vision_encoder (default true), train_projector (default true). **[ADDED]** |
| **Class imbalance** (phishing vs benign in feeds) not addressed | Biased model toward majority class | Support weighted loss or balanced sampling in training config; report class counts in dataset manifest. **[ADDED]** |
| **No modality ablation plan** | Dissertation cannot show contribution of image vs text vs both | Add evaluation mode: image-only, text-only, multimodal. Implement by masking or bypassing the other modality in the pipeline. **[ADDED]** |
| **Explanation training** is optional but underspecified | Inconsistent explanations at inference | Decide: (a) classification-only first, add explanation head later, or (b) joint classification + explanation from the start. Document in ARCHITECTURE. **[ADDED]** |

### Improvements made

- **ARCHITECTURE.md**: §2.4 and §2.5 now specify prompt template location (config), freeze_vision_encoder/train_projector, class balancing, and explanation strategy (classification-first, then generation for explanation).
- **configs/model.yaml**: Added `prompt_template`, `freeze_vision_encoder`, `train_projector`.
- **configs/training.yaml**: Added `class_weights` / `balanced_sampling` and reference to dataset manifest for class counts.

---

## 2. Dataset Creation

### Gaps identified

| Gap | Risk | Improvement |
|-----|------|-------------|
| **TR-OP benchmark** not explicitly integrated | Cannot compare directly to KnowPhish (92.05% F1) as in proposal | Treat TR-OP as a separate evaluation-only dataset: no training on it; document source, license, and how it's loaded in evaluation. **[ADDED]** |
| **Manifest/schema** for processed data not defined | Inconsistent loaders; hard to reproduce splits | Define manifest format (e.g. Parquet/CSV: url, split, label, screenshot_path, text_path, source, crawl_date, content_hash). **[ADDED]** |
| **Quality filters** for crawled pages unspecified | Noisy training data (blank, error, consent walls) | Define filters: min text length, max redirects, exclude HTTP errors; optional: language filter. Document in data pipeline. **[ADDED]** |
| **Deduplication** (URL and content) not specified | Train/val/test leakage; inflated metrics | Deduplicate by URL before split; optionally by content hash to drop near-duplicates. **[ADDED]** |
| **Temporal split** procedure unclear | "Temporal dataset" undefined for evaluation | Define: e.g. "crawl a fresh set of URLs from same feeds at date T2; no overlap with train/val/test by URL." Document in evaluation/test_sets. **[ADDED]** |
| **Zero-shot brand split** procedure unclear | Unclear how "unseen brands" are held out | Define: list of brand entities; assign each sample to a brand (by rule or model); hold out entire brands for test. Document and add brand_id to manifest if available. **[ADDED]** |
| **Feed URLs and formats** are placeholders | Implementation will stall on data ingestion | Add `docs/DATA_SOURCES.md` with real feed URLs, formats, and API keys (env vars). **[ADDED]** |

### Improvements made

- **ARCHITECTURE.md**: §2.1 and §2.3 now include TR-OP usage, manifest schema, quality filters, deduplication, temporal and zero-shot split procedures.
- **data/README.md** and **data_pipeline/preprocessing/README.md**: Reference manifest schema and TR-OP location.
- **configs/data.yaml**: Added quality filters, deduplication, and paths for TR-OP and manifest.
- **docs/DATA_SOURCES.md**: Added (stub) for feed URLs and formats.

---

## 3. Adversarial Testing

### Gaps identified

| Gap | Risk | Improvement |
|-----|------|-------------|
| **Baseline for degradation** unspecified | Cannot compute "degradation vs clean" correctly | For each attack, run on the *same* N samples twice: once clean, once perturbed. Report clean acc, perturbed acc, and degradation. **[ADDED]** |
| **Attack strength/parameters** not configurable | No sensitivity analysis; hard to reproduce | Add per-attack parameters in `evaluation.yaml` (e.g. obfuscation_level, n_typosquat_variants, prompt_injection_templates). **[ADDED]** |
| **Source of adversarial samples** unclear | Where do the 500 samples come from? | Define: use a fixed subset of held-out test (or dedicated adversarial pool) by sample IDs; no overlap with training. **[ADDED]** |
| **Logo manipulation** procedure vague | Unrepeatable or inconsistent results | Specify: e.g. "detect logo region (heuristic or CLIP), overlay a fixed set of replacement logos; or use synthetic logo patch dataset." **[ADDED]** |
| **Prompt injection** placement and strings | Unclear what to inject and where | Define: inject at start and/or end of webpage text; list of canonical strings in config or file; report which variant. **[ADDED]** |
| **Per-attack reporting format** | Dissertation needs a clear table | Table: attack | clean_acc | perturbed_acc | degradation | (optional) F1. Export CSV/JSON from evaluation. **[ADDED]** |

### Improvements made

- **ARCHITECTURE.md**: §2.8 now specifies: same-sample clean vs perturbed comparison, source of adversarial samples, per-attack config and reporting table, and implementation notes for logo manipulation and prompt injection.
- **configs/evaluation.yaml**: Added adversarial sample_source, per-attack params, and output table path.
- **evaluation/adversarial/README.md**: Expanded with baseline comparison and reporting.

---

## 4. Evaluation for the Dissertation

### Gaps identified

| Gap | Risk | Improvement |
|-----|------|-------------|
| **KnowPhish baseline** not formalized | Cannot state "we outperform KnowPhish on TR-OP" | Add explicit evaluation: run PhishGuard-VLM on TR-OP; report same metrics; document KnowPhish numbers in a table (from paper). No need to run KnowPhish code if unavailable. **[ADDED]** |
| **Statistical significance** not mentioned | Reviewers may ask for it | Recommend: report 95% CIs for metrics (bootstrap or normal approx); or McNemar for model comparison if both run on same test set. **[ADDED]** |
| **Result export and figures** unspecified | Hard to regenerate dissertation tables/figures | Define: export run metadata (config hash, seed) + metrics to CSV/JSON; standard figures (ROC, PR, latency histogram) with script or W&B. **[ADDED]** |
| **Ablation plan** missing | Contribution of each component unclear | Plan ablations: (1) full model, (2) no knowledge module, (3) image-only, (4) text-only, (5) no LoRA / base LLaVA. Document in evaluation. **[ADDED]** |
| **Error analysis** not planned | Qualitative discussion in dissertation weak | Add: export misclassified examples (FP/FN) with screenshot path, prediction, label; optional script to build a small review set. **[ADDED]** |
| **Reproducibility checklist** | Incomplete reproducibility statement | Document: seeds (data split, training, eval), PyTorch/CUDA deterministic flags, config versioning. **[ADDED]** |

### Improvements made

- **ARCHITECTURE.md**: §2.9 now includes: KnowPhish comparison on TR-OP, CIs/McNemar, result export and figure specs, ablation list, error-analysis export, reproducibility checklist.
- **docs/EVALUATION_PLAN.md**: New document summarizing evaluation plan, tables, figures, and dissertation checklist.
- **evaluation/README.md**: Links to evaluation plan and result format.
- **configs/evaluation.yaml**: Added output paths for tables/figures and optional reproducibility seed.

---

## 5. Repository Structure Additions

- **docs/DATA_SOURCES.md**: Feed URLs, formats, and env vars for API keys.
- **docs/EVALUATION_PLAN.md**: Dissertation evaluation plan (metrics, baselines, ablations, reporting, reproducibility).
- **evaluation/error_analysis/**: Optional script/output for exporting misclassified examples for qualitative analysis.

---

## 6. Summary

All four areas had material gaps that could block implementation or weaken the dissertation. The updates keep the existing architecture and repo layout intact while adding:

- Clear **multimodal training** choices (prompt, freeze/train, balancing, ablation).
- **Dataset** rigor (manifest, TR-OP, filters, dedup, temporal/zero-shot procedures).
- **Adversarial** rigor (same-sample baseline, configurable attacks, reporting table).
- **Dissertation evaluation** (KnowPhish comparison, significance, export, ablations, error analysis, reproducibility).

No new top-level directories were required; only config and doc updates and one new evaluation subdirectory for error analysis.
