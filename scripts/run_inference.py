#!/usr/bin/env python3
"""
Run full inference on a URL: crawl, preprocess, VLM, optional brand knowledge, aggregate.

Prints JSON with label, confidence, explanation (and optional verdict detail).

Example:
    python scripts/run_inference.py https://example.com
    python scripts/run_inference.py https://suspicious.example --brand "PayPal"
    python scripts/run_inference.py https://foo.com --official paypal.com,ebay.com
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from inference.pipeline import URLInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="PhishGuard URL inference (crawl + VLM + knowledge)")
    parser.add_argument("url", type=str, help="Page URL to analyze")
    parser.add_argument("--brand", type=str, default=None, help="Brand name for Wikidata official-domain check")
    parser.add_argument(
        "--official",
        type=str,
        default=None,
        help="Comma-separated official domains (skips Wikidata), e.g. paypal.com,ebay.com",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    official_list: list[str] | None = None
    if args.official:
        official_list = [x.strip() for x in args.official.split(",") if x.strip()]

    pipe = URLInferencePipeline.from_config(
        _project_root,
        model_yaml=args.config if args.config.is_absolute() else _project_root / args.config,
        default_yaml=args.default_config if args.default_config.is_absolute() else _project_root / args.default_config,
        data_yaml=args.data_config if args.data_config.is_absolute() else _project_root / args.data_config,
        checkpoint=args.checkpoint,
        no_checkpoint=args.no_checkpoint,
    )
    pipe.phish_threshold = args.threshold

    logger.info("Analyzing %s", args.url[:120])
    result = pipe.analyze(args.url, brand_hint=args.brand, official_domains=official_list)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
