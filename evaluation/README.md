# Evaluation

Research-grade evaluation: metrics, multiple test sets, adversarial robustness, efficiency. Full plan: **docs/EVALUATION_PLAN.md**.

## Subdirectories

| Directory | Responsibility |
|-----------|-----------------|
| **metrics/** | Accuracy, precision, recall, F1, ROC-AUC; optional calibration. |
| **test_sets/** | Loaders for held-out, temporal, zero-shot brand datasets. |
| **adversarial/** | Attack implementations (HTML obfuscation, logo manipulation, typosquatting, prompt injection) and runner; report accuracy degradation. |
| **results/** | Tables (CSV/JSON), figures (ROC, PR, latency), W&B links. |
| **error_analysis/** | Export misclassified FP/FN for qualitative analysis. |

**Baseline**: PhishGuard-VLM on TR-OP; document KnowPhish 92.05% F1. **Ablations**: Full, no knowledge, image-only, text-only, no LoRA.

**TR-OP eval**: `python scripts/run_eval.py --tr-op` (manifest under `configs/evaluation.yaml` `test_sets.tr_op` or `configs/data.yaml` `benchmarks.tr_op_path`). **Threshold tuning** (validation only): `python scripts/tune_threshold.py`, then `run_eval.py --threshold-file evaluation/results/tables/best_threshold.json`.

**Measured smoke report** (pytest + optional OpenPhish fetch + sklearn metrics + synthetic manifest): `python scripts/produce_measured_results.py` → `evaluation/results/measured_runs/<stamp>/report.json` and `SUMMARY.md`; pointer in `measured_runs/LATEST.txt`.

## Efficiency

Median and 95th percentile inference latency; throughput (URLs/s) at batch size 1 (and optionally larger).
