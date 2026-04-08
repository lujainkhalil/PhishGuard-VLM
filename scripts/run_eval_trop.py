"""
Evaluate PhishGuard on the TR-OP benchmark dataset.
Run on Kaggle with the trained adapter.

Usage:
    python scripts/run_eval_trop.py \
        --trop-dir /path/to/tr-op \
        --adapter-path /path/to/adapter \
        --output results/trop_eval.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_trop_samples(trop_dir: Path, label: int, max_samples: int = None):
    samples = []
    folder = trop_dir / ("openphish_5000" if label == 1 else "tranco_5000")
    dirs = sorted(folder.iterdir())
    if max_samples:
        dirs = dirs[:max_samples]
    for d in dirs:
        if not d.is_dir():
            continue
        shot = d / "shot.png"
        html = d / "html.txt"
        url_f = d / "input_url.txt"
        if not shot.exists():
            continue
        url = url_f.read_text(errors="ignore").strip() if url_f.exists() else d.name
        text = html.read_text(errors="ignore")[:400] if html.exists() else ""
        samples.append({"path": d, "url": url, "text": text,
                        "screenshot": shot, "label": label})
    return samples


def run_evaluation(adapter_path: str, trop_dir: str, output_path: str,
                   max_per_class: int = 500):
    import torch
    from PIL import Image
    from transformers import (LlavaForConditionalGeneration,
                               LlamaTokenizer, CLIPImageProcessor,
                               LlavaProcessor)
    from peft import PeftModel
    from sklearn.metrics import (f1_score, precision_score,
                                  recall_score, accuracy_score,
                                  classification_report)

    logger.info("Loading model from %s", adapter_path)
    model_id = "llava-hf/llava-1.5-7b-hf"

    tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(model_id)
    processor = LlavaProcessor(tokenizer=tokenizer,
                                image_processor=image_processor)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    device = next(model.parameters()).device
    logger.info("Model loaded on %s", device)

    trop = Path(trop_dir)
    phish_samples = load_trop_samples(trop, label=1, max_samples=max_per_class)
    benign_samples = load_trop_samples(trop, label=0, max_samples=max_per_class)
    all_samples = phish_samples + benign_samples
    logger.info("Evaluating %d samples (%d phishing, %d benign)",
                len(all_samples), len(phish_samples), len(benign_samples))

    preds, labels, details = [], [], []

    for i, s in enumerate(all_samples):
        if i % 50 == 0:
            logger.info("Progress: %d/%d", i, len(all_samples))
        try:
            image = Image.open(s["screenshot"]).convert("RGB")
        except:
            image = Image.new("RGB", (336, 336), (200, 200, 200))

        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
             f"Is this webpage phishing? URL: {s['url']}\n"
             f"Text: {s['text'][:200]}\nAnswer PHISHING or BENIGN."}
        ]}]

        try:
            prompt = processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=20, do_sample=False)
            response = processor.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip().upper()
            pred = 1 if "PHISHING" in response else 0
        except Exception as e:
            logger.warning("Failed on %s: %s", s["url"], e)
            pred = 0
            response = "ERROR"

        preds.append(pred)
        labels.append(s["label"])
        details.append({"url": s["url"], "true": s["label"],
                        "pred": pred, "response": response[:100]})

    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    acc = accuracy_score(labels, preds)

    report = classification_report(labels, preds,
                                   target_names=["benign", "phishing"])

    results = {
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "n_samples": len(all_samples),
        "knowphish_baseline_f1": 0.9205,
        "report": report,
        "details": details,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("F1: %.4f (KnowPhish baseline: 0.9205)", f1)
    logger.info("Results saved to %s", output_path)
    print("\n" + report)
    print(f"\nF1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"KnowPhish baseline F1: 0.9205")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trop-dir", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output", default="results/trop_eval.json")
    parser.add_argument("--max-per-class", type=int, default=500)
    args = parser.parse_args()
    run_evaluation(args.adapter_path, args.trop_dir,
                   args.output, args.max_per_class)


if __name__ == "__main__":
    main()