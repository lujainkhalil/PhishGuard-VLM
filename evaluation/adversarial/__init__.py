"""Adversarial robustness probes for evaluation (text + image perturbations)."""

from evaluation.adversarial.attacks import (
    apply_html_obfuscation,
    apply_logo_manipulation_simulated,
    apply_prompt_injection,
    apply_typosquatting_text,
    load_prompt_injection_templates,
    make_batch_preprocessor,
)
from evaluation.adversarial.runner import build_subset_dataloader, run_adversarial_evaluation

__all__ = [
    "apply_html_obfuscation",
    "apply_logo_manipulation_simulated",
    "apply_prompt_injection",
    "apply_typosquatting_text",
    "build_subset_dataloader",
    "load_prompt_injection_templates",
    "make_batch_preprocessor",
    "run_adversarial_evaluation",
]
