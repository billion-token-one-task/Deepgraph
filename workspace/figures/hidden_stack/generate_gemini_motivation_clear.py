#!/usr/bin/env python3
"""Generate single-column readable motivation figure (overview clear style)."""

from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from paperbanana_wrapper import (  # noqa: E402
    _ensure_paperbanana_env,
    _env_first,
    _http_error_message,
    _image_attempt_count,
    _image_retry_sleep_seconds,
    _is_retriable_image_error,
    _normalize_gemini_native_base_url,
)

STYLE_REF = ROOT / "workspace/figures/hidden_stack/fig_overview_gemini.png"
OUT = ROOT / "workspace/figures/hidden_stack/fig_motivation_gemini.png"

PALETTE = """
- Canvas: off-white #F9F9F9
- HuggingFace: steel blue #5B9BD5
- vLLM: golden amber #EDAD50
- TGI: dusty rose #E27D7D
- Gray section headers, white inner cards, thin #D0D0D0 borders, soft shadows
- Down arrows: soft blue gradient
"""

PROMPT = f"""
Draw a NEW publication-quality motivation figure for an ACL single-column paper.

MATCH the attached overview figure style exactly:
- Gray rounded section headers, white inner cards, muted palette, readable labels, professional ACL look.
- Concrete and easy to read. NO abstract symbols, NO bullet lists, NO ampersand.

CRITICAL LAYOUT — SINGLE COLUMN ONLY (NOT three horizontal columns):
- One narrow vertical stack centered on canvas (~45% width), generous side margins.
- Flow TOP to BOTTOM with downward gradient arrows between sections.
- Aspect ratio feel: tall single-column teaser for intro right column.

NO top figure title. NO bottom banner. NO Step 1/2/3. NO numeric benchmark scores.

{PALETTE}

=== SECTION 1 (top) — gray header "Fixed Inputs" ===
White card 1: server icon + "Qwen2.5-7B-Instruct" + subtitle "frozen weights"
White card 2: document icon + "Q: What is the capital of Australia?" + "Standardized Input"
Small chip row: greedy decoding, fixed extraction

Down arrow

=== SECTION 2 (middle) — gray header "Inference Backends" ===
Three stacked backend cards (vertical, NOT side-by-side columns):

Card A (blue left border): "HuggingFace Transformers"
- One readable row of labeled chips: FP16, eager mode, default stops
- NOT a tokenize-to-decode pipeline diagram

Card B (amber left border): "vLLM"
- Chips: BF16, continuous batching, paged KV cache

Card C (rose left border): "TGI"
- Chips: FP16, tensor parallel, custom decode

Down arrow

=== SECTION 3 (bottom) — gray header "Benchmark Outcome" ===
White card 1 — grouped bar chart titled "Exact Match Accuracy"
- Three backend groups HF vLLM TGI, one colored bar each, slightly different heights
- NO axis numbers, NO percentage labels

White card 2 — CONCRETE answer comparison titled "Same prompt, different strings"
- Three color-coded one-line answers:
  HF: Canberra
  vLLM: The capital is Canberra
  TGI: Canberra, Australia

White card 3 — three small pills in a row:
- Blue "Reference backend"
- Amber "Serving optimized"
- Rose "Production stack"

Tiny legend bottom-right: blue HF, amber vLLM, rose TGI.

Do NOT use left-center-right three-column layout. Everything stacked vertically in one column.
""".strip()


def _run_gemini(*, style_ref: Path, out: Path, prompt: str) -> int:
    api_key = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY", "GEMINI_NATIVE_API_KEY")
    base_url = _normalize_gemini_native_base_url(
        _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL", "GEMINI_NATIVE_BASE_URL")
    )
    model = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "IMAGE_GEN_MODEL_NAME") or "gemini-2.5-flash-image"
    if not api_key or not base_url:
        print("Missing Gemini credentials", file=sys.stderr)
        return 3

    b64 = base64.b64encode(style_ref.read_bytes()).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": b64}},
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    url = f"{base_url}/v1beta/models/{model}:generateContent"
    attempts = _image_attempt_count()
    errors: list[str] = []

    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "DeepGraph-PaperBanana-Wrapper/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            code, detail = _http_error_message(exc)
            msg = f"attempt_{attempt}:HTTP{code}:{detail[:300]}" if code else f"attempt_{attempt}:{detail[:300]}"
            errors.append(msg)
            if attempt < attempts and _is_retriable_image_error(detail, http_code=code):
                import time

                wait = _image_retry_sleep_seconds(attempt)
                print(f"Retry {attempt}/{attempts} after {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"Gemini generation failed: {detail}", file=sys.stderr)
            return 4

        for candidate in body.get("candidates") or []:
            content = candidate.get("content") if isinstance(candidate, dict) else None
            for part in (content.get("parts") if isinstance(content, dict) else None) or []:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(base64.b64decode(str(inline["data"])))
                    print(f"Wrote {out} ({out.stat().st_size} bytes)")
                    return 0
        errors.append(f"attempt_{attempt}:no_image_in_response")
        if attempt < attempts:
            import time

            time.sleep(_image_retry_sleep_seconds(attempt))

    print("Gemini generation failed: " + "; ".join(errors[-3:]), file=sys.stderr)
    return 4


def main() -> int:
    _ensure_paperbanana_env()
    if not STYLE_REF.exists():
        print(f"Missing style reference: {STYLE_REF}", file=sys.stderr)
        return 2
    backup = OUT.with_name(OUT.stem + "_prev.png")
    if OUT.exists():
        import shutil

        shutil.copy2(OUT, backup)
        print(f"Backed up previous output to {backup}")
    return _run_gemini(style_ref=STYLE_REF, out=OUT, prompt=PROMPT)


if __name__ == "__main__":
    raise SystemExit(main())
