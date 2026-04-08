"""
Load the trained LoRA adapter and run real inference.
Used by the API when PHISHGUARD_ADAPTER_PATH is set.
"""
from __future__ import annotations
import torch
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers import LlamaTokenizer, CLIPImageProcessor, LlavaProcessor
from peft import PeftModel


@dataclass
class VLMResult:
    label: int
    confidence: float
    explanation: str
    final_url: str
    crawl_status: str
    model_probability: float
    phishing_probability: float
    knowledge_used: bool
    aggregated: None = None
    cross_modal_consistency: float = None
    cross_modal: dict = None


class VLMInferencePipeline:
    def __init__(self, adapter_path: str):
        from data_pipeline.crawler.crawler import crawl_url_with_retries
        self._crawl = crawl_url_with_retries

        model_id = "llava-hf/llava-1.5-7b-hf"
        tokenizer = LlamaTokenizer.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf")
        self.processor = LlavaProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor)
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        base = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def analyze(self, url: str, **kwargs) -> VLMResult:
        import tempfile, os
        from pathlib import Path

        scratch = Path(tempfile.mkdtemp())
        crawl = self._crawl(
            url,
            screenshot_dir=scratch / "screenshots",
            pages_dir=scratch / "pages",
            timeout_ms=20000,
            max_attempts=1,
        )

        if crawl.status != "ok" or not crawl.screenshot_path:
            return VLMResult(
                label=0, confidence=0.5,
                explanation=f"Could not crawl page: {crawl.error or crawl.status}",
                final_url=url, crawl_status=crawl.status,
                model_probability=0.5, phishing_probability=0.5,
                knowledge_used=False,
            )

        try:
            image = Image.open(crawl.screenshot_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (336, 336), (200, 200, 200))

        try:
            text = Path(crawl.text_path).read_text(errors="ignore")[:300]
        except:
            text = ""

        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
             f"Is this webpage phishing? URL: {url}\nText: {text[:200]}\n"
             f"Answer PHISHING or BENIGN and explain why."}
        ]}]

        prompt = self.processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=prompt, images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=1.0,
            )

        response = self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().upper()

        is_phish = "PHISHING" in response
        confidence = 0.85 if is_phish else 0.82

        return VLMResult(
            label=int(is_phish),
            confidence=confidence,
            explanation=response[:500],
            final_url=crawl.final_url or url,
            crawl_status=crawl.status,
            model_probability=confidence,
            phishing_probability=confidence if is_phish else 1 - confidence,
            knowledge_used=False,
        )