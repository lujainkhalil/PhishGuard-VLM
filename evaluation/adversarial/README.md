# Adversarial Evaluation

Implement and run: HTML obfuscation, logo manipulation, typosquatting, prompt injection.

**Baseline**: For each attack, use a fixed sample set (e.g. 500 from held-out or adversarial pool). Run model on **clean** inputs, then on **perturbed** inputs. Report clean_acc, perturbed_acc, **degradation** (clean − perturbed). Target &lt;5% degradation.

**Reporting**: Export table: attack_type | clean_acc | perturbed_acc | degradation | F1 to CSV (see configs/evaluation.yaml output_table). Per-attack parameters (obfuscation level, typosquat variants, prompt-injection templates) in config for sensitivity analysis.
