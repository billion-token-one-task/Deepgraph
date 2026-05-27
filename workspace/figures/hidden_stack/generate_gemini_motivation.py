#!/usr/bin/env python3
"""One-off Gemini motivation figure with tight prompt (v2 style, paper content)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from paperbanana_wrapper import (  # noqa: E402
    _ensure_paperbanana_env,
    _nearest_supported_ratio,
    _run_gemini_native_image_generation,
)

PROMPT = """
Draw a publication-quality raster scientific diagram (rich illustration, not flat vector art).

LAYOUT — copy a three-column conference framework style:
• LEFT: light blue rounded outer panel, small header "Fixed Inputs" inside panel top.
• CENTER: light orange rounded outer panel with DASHED border, small header "Inference Backends".
• RIGHT: light green rounded outer panel, small header "Benchmark Outcome".
• Horizontal arrows connect columns. Rounded white inner cards, soft shadows, small 3D-style icons.

LEFT PANEL content:
1) Fixed Model Block — server icon — "Qwen2.5-7B-Instruct (frozen weights)"
2) Fixed Prompt Box — document icon — "Q: What is the capital of Australia?" and "A: (same template)" and "Standardized Input"

CENTER PANEL — three stacked color-coded backend cards (NOT generic Backend A/B/C):
• Blue card: "HuggingFace Transformers" — bullets: FP16 precision; Dynamic batching; Default stop rules
• Orange card: "vLLM" — bullets: BF16 precision; Continuous batching; Optimized CUDA kernels
• Green card: "TGI" — bullets: FP16 precision; Tensor parallelism; Custom decoding
Do NOT show tokenize-layer-decode pipelines, speculative decoding, clocks, or latency labels.

RIGHT PANEL content:
• Bar chart titled "Exact Match Accuracy" with three bars: HF 92.1%, vLLM 92.4%, TGI 91.8%
• Three small answer snippet boxes below (color matched), same question, slightly different wording, one differing word in red
• Tiny legend bottom-right corner only: Blue HF, Orange vLLM, Green TGI

TOPIC LOCK: LLM inference backend benchmark evaluation ONLY.

FORBIDDEN — do not draw any of these:
Static Data, Configuration Files, Adaptive Model, Real-time Learning, Performance Metrics, Actionable Insights,
data cleaning, feature engineering, sensor data, ML training, latency, throughput, GPU utilization,
parallel decode, Backend A, Backend B, Backend C, ampersand character, full-width figure title, bottom summary banner.

No ampersand anywhere. No title at top. No bottom motivational sentence.
""".strip()


def main() -> int:
    out = ROOT / "workspace/figures/hidden_stack/fig_motivation_gemini.png"
    _ensure_paperbanana_env()
    ratio = _nearest_supported_ratio("16:9")
    return _run_gemini_native_image_generation(output_path=out, prompt=PROMPT, aspect_ratio=ratio)


if __name__ == "__main__":
    raise SystemExit(main())
