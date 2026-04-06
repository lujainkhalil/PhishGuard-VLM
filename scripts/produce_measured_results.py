#!/usr/bin/env python3
"""
Run as much of the PhishGuard-VLM stack as the current environment allows and write
**measured** numbers to ``evaluation/results/measured_runs/<timestamp>/``.

Stages:
  1. Pytest suite (counts + exit code)
  2. Optional feed fetch (OpenPhish; network)
  3. Synthetic processed manifest + PNGs (for layout validation)
  4. Classification metrics + threshold sweep on fixed score arrays (sklearn)
  5. Optional: tiny ``PhishingClassificationHead`` train loop if PyTorch is installed
  6. Optional: full ``run_eval`` if ``--attempt-vlm-eval`` and torch+deps present (may download LLaVA)

Usage:
    python scripts/produce_measured_results.py
    python scripts/produce_measured_results.py --skip-fetch
    python scripts/produce_measured_results.py --attempt-vlm-eval   # heavy; needs GPU RAM for LLaVA
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_pytest() -> dict:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(_project_root / "tests"), "-q", "--tb=no"],
        cwd=_project_root,
        capture_output=True,
        text=True,
    )
    tail = (r.stdout or "") + (r.stderr or "")
    passed = failed = skipped = 0
    if m := re.search(r"(\d+)\s+passed", tail):
        passed = int(m.group(1))
    if m := re.search(r"(\d+)\s+failed", tail):
        failed = int(m.group(1))
    if m := re.search(r"(\d+)\s+skipped", tail):
        skipped = int(m.group(1))
    return {
        "exit_code": r.returncode,
        "stdout_tail": tail[-2000:],
        "parsed_passed": passed,
        "parsed_failed": failed,
        "parsed_skipped": skipped,
    }


def _run_feed_fetch_openphish() -> dict:
    r = subprocess.run(
        [sys.executable, str(_project_root / "scripts" / "run_feed_fetch.py"), "--openphish-only"],
        cwd=_project_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (r.stdout or "") + (r.stderr or "")
    n = 0
    for line in out.splitlines():
        if "OpenPhish:" in line and "entries" in line:
            try:
                n = int(line.split("OpenPhish:")[1].split("entries")[0].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return {"exit_code": r.returncode, "openphish_entries_reported": n, "log_tail": out[-1500:]}


def _synthetic_manifest(out_dir: Path) -> dict:
    import numpy as np
    from PIL import Image

    import pandas as pd

    img_dir = out_dir / "images"
    txt_dir = out_dir / "text"
    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.random.default_rng(42)
    for i in range(12):
        lab = "phishing" if i % 3 != 0 else "benign"
        split = ["train", "train", "validation", "test"][min(i % 4, 3)]
        stem = f"sample_{i:03d}"
        png = img_dir / f"{stem}.png"
        color = (200, 40, 40) if lab == "phishing" else (40, 120, 200)
        Image.new("RGB", (336, 200), color).save(png)
        tp = txt_dir / f"{stem}.txt"
        text = (
            "Verify your account immediately. Click here to login."
            if lab == "phishing"
            else "Welcome to our documentation and help center."
        )
        tp.write_text(text, encoding="utf-8")
        rows.append(
            {
                "url": f"https://{'evil' if lab == 'phishing' else 'good'}.example/{i}",
                "image_path": str(png.relative_to(out_dir)),
                "text_path": str(tp.relative_to(out_dir)),
                "label": lab,
                "split": split,
            }
        )
    df = pd.DataFrame(rows)
    mp = out_dir / "manifest.parquet"
    try:
        df.to_parquet(mp, index=False)
        fmt = "parquet"
    except Exception as e:
        mp = out_dir / "manifest.csv"
        df.to_csv(mp, index=False)
        fmt = f"csv (parquet unavailable: {e})"
    return {"manifest_path": str(mp), "format": fmt, "n_rows": len(rows), "output_dir": str(out_dir)}


def _metrics_and_threshold() -> dict:
    import numpy as np

    from evaluation.metrics.binary import compute_binary_metrics
    from evaluation.threshold_tuning import sweep_threshold_f1

    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    logits = rng.standard_normal(n) * 1.5 + (y_true * 2.0 - 1.0) * 0.8
    y_score = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (y_score >= 0.5).astype(np.int64)
    m = compute_binary_metrics(y_true, y_pred, y_score)
    d = m.to_dict(include_matrix=True)
    bt, bf1, bm = sweep_threshold_f1(y_true, y_score, n_thresholds=101)
    return {
        "synthetic_scores_n": n,
        "metrics_at_0p5_threshold": {k: d[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if k in d},
        "threshold_sweep_best_threshold": bt,
        "threshold_sweep_best_f1": bf1,
        "threshold_sweep_metrics_at_best": bm,
    }


def _tiny_head_train() -> dict | None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"status": "skipped", "reason": "torch not installed"}

    from models.heads.classification import PhishingClassificationHead

    torch.manual_seed(42)
    d = 256
    n = 400
    x = torch.randn(n, d)
    y = (torch.randn(n) > 0).float().unsqueeze(1)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    head = PhishingClassificationHead(d, num_classes=1, mlp_hidden_dim=64, dropout=0.0)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    head.train()
    losses = []
    for epoch in range(3):
        el = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            logit = head(xb)
            loss = crit(logit, yb)
            loss.backward()
            opt.step()
            el += loss.item()
        losses.append(el / max(1, len(dl)))
    head.eval()
    with torch.no_grad():
        logits = head(x)
        prob = PhishingClassificationHead.logits_to_probability(logits)
        pred = (prob >= 0.5).long()
        y_long = y.squeeze().long()
        acc = float((pred == y_long).float().mean())
    return {
        "status": "ok",
        "embedding_dim": d,
        "epochs": 3,
        "mean_train_loss_last_epoch": losses[-1],
        "train_subset_accuracy": acc,
    }


def _attempt_vlm_eval(manifest_dir: Path) -> dict:
    try:
        r = subprocess.run(
            [
                sys.executable,
                str(_project_root / "scripts" / "run_eval.py"),
                "--manifest",
                str(manifest_dir / "manifest.parquet"),
                "--no-split-filter",
                "--no-checkpoint",
                "--batch-size",
                "1",
                "--no-predictions-csv",
                "--run-id",
                "measured-pipeline",
            ],
            cwd=_project_root,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return {
            "exit_code": r.returncode,
            "stdout_tail": (r.stdout or "")[-3000:],
            "stderr_tail": (r.stderr or "")[-3000:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": 3600}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--attempt-vlm-eval", action="store_true", help="May download multi-GB model")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _project_root / "evaluation" / "results" / "measured_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    synth_dir = run_dir / "synthetic_processed"

    report: dict = {
        "meta": {
            "utc_timestamp": stamp,
            "project_root": str(_project_root),
            "python": sys.version.split()[0],
        },
        "stages": {},
    }

    t0 = time.perf_counter()
    logger.info("Running pytest…")
    report["stages"]["pytest"] = _run_pytest()

    if not args.skip_fetch:
        logger.info("Fetching OpenPhish feed…")
        try:
            report["stages"]["feed_fetch_openphish"] = _run_feed_fetch_openphish()
        except Exception as e:
            report["stages"]["feed_fetch_openphish"] = {"error": str(e)}
    else:
        report["stages"]["feed_fetch_openphish"] = {"skipped": True}

    logger.info("Building synthetic manifest…")
    try:
        report["stages"]["synthetic_manifest"] = _synthetic_manifest(synth_dir)
    except Exception as e:
        report["stages"]["synthetic_manifest"] = {"error": str(e)}

    logger.info("Computing sklearn metrics + threshold sweep…")
    try:
        report["stages"]["metrics_and_threshold"] = _metrics_and_threshold()
    except Exception as e:
        report["stages"]["metrics_and_threshold"] = {"error": str(e)}

    logger.info("Tiny classification head (optional torch)…")
    report["stages"]["tiny_head_train"] = _tiny_head_train()

    if args.attempt_vlm_eval and "error" not in report["stages"].get("synthetic_manifest", {}):
        logger.info("Attempting VLM eval (long-running)…")
        report["stages"]["vlm_eval_no_checkpoint"] = _attempt_vlm_eval(synth_dir)
    else:
        report["stages"]["vlm_eval_no_checkpoint"] = {
            "skipped": True,
            "reason": "pass --attempt-vlm-eval to run (downloads LLaVA; needs disk/GPU)",
        }

    report["meta"]["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    json_path = run_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Human-readable summary
    lines = [
        "# Measured pipeline run",
        "",
        f"**Timestamp (UTC):** `{stamp}`  ",
        f"**Elapsed:** {report['meta']['elapsed_seconds']} s  ",
        "",
        "## Pytest",
        "",
    ]
    py = report["stages"].get("pytest", {})
    lines.append(f"- Exit code: `{py.get('exit_code')}`")
    lines.append(f"- Parsed: passed={py.get('parsed_passed')}, failed={py.get('parsed_failed')}, skipped={py.get('parsed_skipped')}")
    lines.append("")
    lines.append("## Feed fetch (OpenPhish)")
    lines.append("")
    ff = report["stages"].get("feed_fetch_openphish", {})
    lines.append(f"- {json.dumps(ff, indent=2)[:1200]}")
    lines.append("")
    lines.append("## Synthetic score metrics (n=500, reproducible RNG)")
    lines.append("")
    mt = report["stages"].get("metrics_and_threshold", {})
    if "error" not in mt:
        lines.append(f"- At threshold 0.5: {mt.get('metrics_at_0p5_threshold')}")
        lines.append(f"- Best F1 threshold sweep: τ={mt.get('threshold_sweep_best_threshold'):.4f}, F1={mt.get('threshold_sweep_best_f1'):.4f}")
        lines.append(f"- Metrics at τ*: {mt.get('threshold_sweep_metrics_at_best')}")
    else:
        lines.append(f"- Error: `{mt.get('error')}`")
    lines.append("")
    lines.append("## Tiny MLP head (proxy for train loop)")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['stages'].get('tiny_head_train'), indent=2)}\n```")
    lines.append("")
    lines.append("## Full VLM eval")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['stages'].get('vlm_eval_no_checkpoint'), indent=2)[:2000]}\n```")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- JSON: `{json_path.relative_to(_project_root)}`")
    lines.append(f"- Synthetic data: `{synth_dir.relative_to(_project_root)}`")
    lines.append("")
    lines.append(
        "> **Note:** End-to-end LLaVA training/evaluation requires `pip install -r requirements.txt`, "
        "GPU memory, crawled data, and `python scripts/run_train.py` / `run_eval.py`. "
        "This report records what ran successfully in the current environment."
    )

    (run_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    latest = run_dir.parent / "LATEST.txt"
    latest.write_text(stamp + "\n", encoding="utf-8")
    logger.info("Wrote %s and SUMMARY.md (latest → %s)", json_path, latest)
    return 0 if report["stages"].get("pytest", {}).get("exit_code") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
