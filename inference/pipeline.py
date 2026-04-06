"""
End-to-end URL inference: crawl → preprocess → VLM → knowledge → aggregate.

Returns ``label`` (0 benign / 1 phishing), ``confidence``, and ``explanation``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from data_pipeline.crawler.crawler import CrawlResult, crawl_url_with_retries
from data_pipeline.preprocessing.image_processing import resize_pad_rgb
from data_pipeline.preprocessing.text_processing import clean_and_normalize_text
from inference.aggregator import AggregatedVerdict, ModelPrediction, aggregate_signals
from inference.knowledge_fusion import DEFAULT_KNOWLEDGE_FUSION, KnowledgeFusionConfig
from knowledge_module.brand_matching.matcher import assess_brand_domain_risk, assess_url_against_wikidata_brand
from knowledge_module.cross_modal_consistency import compute_cross_modal_consistency
from knowledge_module.wikidata.client import WikidataClient
from models import PhishingClassifier, fusion_kwargs_from_yaml

logger = logging.getLogger(__name__)


@dataclass
class URLInferenceResult:
    """Outcome of analyzing a single URL (may be partial if crawl failed)."""

    label: int
    confidence: float
    explanation: str
    final_url: str
    crawl_status: str
    model_probability: float | None
    phishing_probability: float | None
    knowledge_used: bool
    aggregated: AggregatedVerdict | None
    cross_modal_consistency: float | None = None
    cross_modal: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "label_name": "phishing" if self.label == 1 else "benign",
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "final_url": self.final_url,
            "crawl_status": self.crawl_status,
            "model_probability": None
            if self.model_probability is None
            else round(self.model_probability, 4),
            "phishing_probability": None
            if self.phishing_probability is None
            else round(self.phishing_probability, 4),
            "knowledge_used": self.knowledge_used,
        }
        if self.cross_modal_consistency is not None:
            d["cross_modal_consistency"] = round(self.cross_modal_consistency, 4)
        if self.cross_modal is not None:
            d["cross_modal"] = dict(self.cross_modal)
        if self.aggregated is not None:
            d["verdict"] = self.aggregated.to_dict()
        return d


def _load_checkpoint_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("load_state_dict missing keys (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        logger.warning("load_state_dict unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])


class URLInferencePipeline:
    """
    Loads the classifier once; each :meth:`analyze` crawls the URL, preprocesses, runs model + knowledge, fuses.
    """

    MODEL_ID_MAP = {"llava-1.5-7b": "llava-hf/llava-1.5-7b-hf"}

    def __init__(
        self,
        model: PhishingClassifier,
        device: torch.device,
        *,
        image_size: int = 336,
        text_max_length: int = 2048,
        phish_threshold: float = 0.5,
        crawl_timeout_ms: int = 30_000,
        crawl_viewport: dict | None = None,
        crawl_max_retries: int = 2,
        crawl_max_attempts: int | None = None,
        crawl_retry_backoff_ms: int = 750,
        scratch_dir: Path | str | None = None,
        wikidata_client: WikidataClient | None = None,
        wikidata_cache_dir: Path | str | None = None,
        enable_cross_modal_consistency: bool = True,
        knowledge_fusion: KnowledgeFusionConfig | None = None,
    ):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.phish_threshold = phish_threshold
        self.crawl_timeout_ms = crawl_timeout_ms
        self.crawl_viewport = crawl_viewport or {"width": 1920, "height": 1080}
        self.crawl_max_retries = crawl_max_retries
        self.crawl_max_attempts = crawl_max_attempts
        self.crawl_retry_backoff_ms = crawl_retry_backoff_ms
        self.scratch_dir = Path(scratch_dir) if scratch_dir else Path("data/inference_scratch")
        self._wikidata_explicit = wikidata_client
        self._wikidata_cache_dir = Path(wikidata_cache_dir) if wikidata_cache_dir else None
        self._wikidata_lazy: WikidataClient | None = wikidata_client
        self.enable_cross_modal_consistency = bool(enable_cross_modal_consistency)
        self.knowledge_fusion = knowledge_fusion if knowledge_fusion is not None else DEFAULT_KNOWLEDGE_FUSION

    def _wikidata(self) -> WikidataClient:
        if self._wikidata_explicit is not None:
            return self._wikidata_explicit
        if self._wikidata_lazy is None:
            self._wikidata_lazy = WikidataClient(cache_dir=self._wikidata_cache_dir)
        return self._wikidata_lazy

    @classmethod
    def from_config(
        cls,
        project_root: Path,
        *,
        model_yaml: Path | None = None,
        default_yaml: Path | None = None,
        data_yaml: Path | None = None,
        inference_yaml: Path | None = None,
        checkpoint: Path | None = None,
        no_checkpoint: bool = False,
        wikidata_cache_dir: Path | None = None,
    ) -> URLInferencePipeline:
        """Build pipeline from repo YAML layouts (same conventions as ``scripts/run_eval.py``)."""

        def _load_yaml(p: Path) -> dict[str, Any]:
            try:
                import yaml

                with open(p, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Could not load %s: %s", p, e)
                return {}

        project_root = Path(project_root)
        model_yaml = model_yaml or project_root / "configs/model.yaml"
        default_yaml = default_yaml or project_root / "configs/default.yaml"
        data_yaml = data_yaml or project_root / "configs/data.yaml"

        model_cfg = _load_yaml(model_yaml)
        default_cfg = _load_yaml(default_yaml)
        data_cfg = _load_yaml(data_yaml)
        inf_path = inference_yaml if inference_yaml is not None else project_root / "configs/inference.yaml"
        inference_cfg = _load_yaml(inf_path).get("inference") or {}
        knowledge_fusion = KnowledgeFusionConfig.from_mapping(inference_cfg.get("knowledge_fusion"))

        paths = default_cfg.get("paths") or {}
        crawl_cfg = data_cfg.get("crawl") or {}
        pre_cfg = data_cfg.get("preprocessing") or {}

        m = model_cfg.get("model") or {}
        model_name = m.get("name", "llava-1.5-7b")
        model_id = cls.MODEL_ID_MAP.get(model_name, model_name)
        if "/" not in model_id:
            model_id = f"llava-hf/{model_name}-hf"
        lora_cfg = model_cfg.get("lora") or {}
        head_cfg = model_cfg.get("head") or {}
        fusion_cfg = model_cfg.get("fusion") or {}

        model = PhishingClassifier(
            model_name=model_id,
            revision=m.get("revision", "main"),
            freeze_vision_encoder=m.get("freeze_vision_encoder", True),
            train_projector=m.get("train_projector", True),
            lora_enabled=lora_cfg.get("enabled", True),
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=lora_cfg.get("target_modules"),
            lora_train_multi_modal_projector=lora_cfg.get("train_multi_modal_projector", False),
            lora_gradient_checkpointing=lora_cfg.get("gradient_checkpointing", True),
            head_hidden_size=head_cfg.get("hidden_size", 4096),
            head_mlp_hidden_dim=head_cfg.get("mlp_hidden_dim"),
            head_dropout=float(head_cfg.get("dropout", 0.1)),
            head_use_layer_norm=head_cfg.get("use_layer_norm", True),
            num_classes=head_cfg.get("num_classes", 1),
            **fusion_kwargs_from_yaml(fusion_cfg),
        )

        if not no_checkpoint:
            ckpt = checkpoint
            if ckpt is None:
                ckpt = project_root / paths.get("checkpoints_dir", "models/checkpoints") / "best.pt"
            else:
                ckpt = Path(ckpt)
                if not ckpt.is_absolute():
                    ckpt = project_root / ckpt
            if ckpt.is_file():
                _load_checkpoint_state(model, ckpt)
            else:
                logger.warning("Checkpoint not found at %s — using base weights.", ckpt)

        pt = model_cfg.get("prompt_template")
        if isinstance(pt, str) and pt.strip():
            model.prompt_template = pt.strip()
        if "text_max_length" in m:
            model.max_length = int(m["text_max_length"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache = wikidata_cache_dir
        if cache is None:
            cache = project_root / paths.get("data_root", "data") / "wikidata_cache"

        return cls(
            model,
            device,
            image_size=int(pre_cfg.get("image_size", 336)),
            text_max_length=int(pre_cfg.get("text_max_length", 2048)),
            crawl_timeout_ms=int(crawl_cfg.get("timeout_ms", 60_000)),
            crawl_viewport=crawl_cfg.get("viewport"),
            crawl_max_retries=int(crawl_cfg.get("max_retries", 2)),
            crawl_max_attempts=(
                int(crawl_cfg["max_attempts"]) if crawl_cfg.get("max_attempts") is not None else None
            ),
            crawl_retry_backoff_ms=int(crawl_cfg.get("retry_backoff_ms", 750)),
            scratch_dir=project_root / "data" / "inference_scratch",
            wikidata_client=None,
            wikidata_cache_dir=cache,
            knowledge_fusion=knowledge_fusion,
        )

    def _failure_result(self, crawl: CrawlResult, message: str) -> URLInferenceResult:
        return URLInferenceResult(
            label=0,
            confidence=0.0,
            explanation=message,
            final_url=crawl.final_url or crawl.url,
            crawl_status=crawl.status,
            model_probability=None,
            phishing_probability=None,
            knowledge_used=False,
            aggregated=None,
            cross_modal_consistency=None,
            cross_modal=None,
        )

    def _knowledge_signal(
        self,
        url: str,
        *,
        brand_hint: str | None,
        official_domains: list[str] | None,
    ):
        if brand_hint and brand_hint.strip():
            return assess_url_against_wikidata_brand(
                url.strip(), brand_hint.strip(), self._wikidata()
            )
        if official_domains:
            return assess_brand_domain_risk(url, official_domains)
        return None

    @torch.no_grad()
    def analyze(
        self,
        url: str,
        *,
        brand_hint: str | None = None,
        official_domains: list[str] | None = None,
    ) -> URLInferenceResult:
        """
        Crawl ``url``, clean text and resize screenshot, run the VLM, optionally run brand/domain knowledge, fuse.
        """
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        shot_dir = self.scratch_dir / "screenshots"
        pages_dir = self.scratch_dir / "pages"

        _crawl_kw: dict[str, Any] = {
            "screenshot_dir": shot_dir,
            "pages_dir": pages_dir,
            "timeout_ms": self.crawl_timeout_ms,
            "viewport": self.crawl_viewport,
            "retry_backoff_ms": self.crawl_retry_backoff_ms,
        }
        if self.crawl_max_attempts is not None:
            _crawl_kw["max_attempts"] = self.crawl_max_attempts
        else:
            _crawl_kw["max_retries"] = self.crawl_max_retries
        crawl = crawl_url_with_retries(url, **_crawl_kw)

        if crawl.status != "ok" or not crawl.screenshot_path or not crawl.text_path:
            err = crawl.error or crawl.status
            return self._failure_result(
                crawl,
                f"Page could not be captured ({crawl.status}). {err} No model or knowledge score was produced.",
            )

        final_url = crawl.final_url or url
        try:
            raw_text = Path(crawl.text_path).read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return self._failure_result(crawl, f"Could not read extracted text: {e}")

        text = clean_and_normalize_text(raw_text)
        if self.text_max_length > 0 and len(text) > self.text_max_length:
            text = text[: self.text_max_length]

        try:
            with Image.open(crawl.screenshot_path) as im:
                image = resize_pad_rgb(im, self.image_size)
        except Exception as e:
            return self._failure_result(crawl, f"Could not load screenshot: {e}")

        try:
            self.model.eval()
            try:
                self.model.to(self.device)
            except Exception:
                pass
            inputs = self.model.prepare_inputs([image], [text], device=self.device)
            logits = self.model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            p_model = float(self.model.predict_proba(logits).reshape(-1)[0].item())
        except Exception as e:
            logger.exception("Model forward failed")
            return self._failure_result(crawl, f"Model inference failed: {e}")

        knowledge = self._knowledge_signal(
            final_url,
            brand_hint=brand_hint,
            official_domains=official_domains,
        )

        cross_modal = None
        if self.enable_cross_modal_consistency:
            ref_brands: list[str] = []
            if brand_hint and str(brand_hint).strip():
                ref_brands.append(str(brand_hint).strip())
            if knowledge is not None:
                cb = getattr(knowledge, "claimed_brand", None)
                if cb and str(cb).strip() and str(cb).strip() not in ref_brands:
                    ref_brands.append(str(cb).strip())
            cross_modal = compute_cross_modal_consistency(
                page_text=text,
                screenshot_image=image,
                page_url=final_url,
                reference_brands=ref_brands or None,
            )

        verdict = aggregate_signals(
            ModelPrediction(phishing_probability=p_model),
            knowledge,
            cross_modal=cross_modal,
            fusion=self.knowledge_fusion,
            phish_threshold=self.phish_threshold,
        )

        return URLInferenceResult(
            label=verdict.label,
            confidence=verdict.confidence,
            explanation=verdict.explanation,
            final_url=final_url,
            crawl_status=crawl.status,
            model_probability=verdict.model_probability,
            phishing_probability=verdict.phishing_probability,
            knowledge_used=verdict.knowledge_used,
            aggregated=verdict,
            cross_modal_consistency=verdict.cross_modal_consistency,
            cross_modal=verdict.cross_modal,
        )
