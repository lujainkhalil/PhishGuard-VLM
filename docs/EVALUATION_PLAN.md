# Evaluation Plan (Dissertation)

Summary of evaluation design for reproducible results and dissertation-ready tables and figures.

## Metrics

- **Primary**: Accuracy, Precision, Recall, F1, ROC-AUC (macro / per-class as needed).
- **Optional**: ECE (calibration).
- **Statistical significance**: 95% CIs (bootstrap or normal approx) for key metrics; McNemar for model comparison when both predictions on same test set available.

## Test Sets

| Set | Description | Loader / path |
|-----|-------------|----------------|
| Held-out | Random split from main dataset (seed in config) | `evaluation/test_sets/` |
| Temporal | Fresh crawl at T2; no URL overlap with train/val/test | Config path |
| Zero-shot brand | Brands unseen in training; split by brand_id | Config path |
| TR-OP | Benchmark; evaluation only | `data/benchmarks/tr-op/` |

## Baseline Comparison

- Run PhishGuard-VLM on TR-OP; report same metrics as KnowPhish paper.
- Document KnowPhish numbers (e.g. 92.05% F1) in results table; no need to run KnowPhish code if unavailable.

## Ablations

1. Full model (VLM + knowledge)
2. No knowledge module
3. Image-only
4. Text-only
5. No LoRA (base LLaVA)

Run as separate configs or flags; report metrics for each.

## Adversarial

- Same N samples per attack: run clean, then perturbed; report clean_acc, perturbed_acc, degradation.
- Target: &lt;5% accuracy degradation per attack.
- Export table to CSV/JSON: attack_type | clean_acc | perturbed_acc | degradation | F1.

## Efficiency

- Median and 95th percentile inference latency (end-to-end and model-only).
- Throughput (URLs/s) at batch size 1.

## Result Export

- **Tables**: Run metadata (config hash, seed, git commit) + metrics to `evaluation/results/tables/` (CSV/JSON).
- **Figures**: ROC, PR curve, latency histogram; script or W&B export to `evaluation/results/figures/`.
- **Error analysis**: Misclassified examples (FP/FN) with screenshot path, prediction, label, URL to `evaluation/error_analysis/` or similar.

## Reproducibility Checklist

- [ ] Fixed seed for data split (in config).
- [ ] Fixed seed for training and evaluation.
- [ ] PyTorch/CUDA deterministic flags set where possible.
- [ ] Config hash and git commit recorded with each run.
- [ ] TR-OP and other benchmarks: no training on evaluation-only data.
