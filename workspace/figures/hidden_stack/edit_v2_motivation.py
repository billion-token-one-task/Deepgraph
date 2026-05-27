#!/usr/bin/env python3
"""Edit fig_motivation_gemini_v2.png in-place style via Gemini (image + text)."""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
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

SRC = ROOT / "workspace/figures/hidden_stack/fig_motivation_gemini_v2.png"
OUT = ROOT / "workspace/figures/hidden_stack/fig_motivation_gemini.png"

EDIT_PROMPT = """
Edit the attached v2 scientific figure. RE-LAYOUT into a COMPACT single-column figure (for ACL intro right column).
Keep v2 visual polish: gray section headers, white cards, shadows, rich mini-diagrams inside backend cards, NOT bullet lists.

FORBIDDEN LAYOUT:
- NO three wide columns left-to-right (no Context Setup | Backends | Outcome horizontal split).
- NO tall vertical strip that is much taller than wide — keep aspect ratio roughly square or slightly tall (max ~4:5), NOT elongated poster.

REQUIRED COMPACT TOP-TO-BOTTOM LAYOUT (3 horizontal bands, down arrows between bands only):

BAND 1 — slim header "Fixed Inputs"
- One compact white card (~15% total height): Fixed Model Block + Fixed Prompt side-by-side in same row
- Model: Qwen2.5-7B-Instruct (frozen weights), server icon
- Prompt: Q: What is the capital of Australia? + Standardized Input, document icon

BAND 2 — header "Inference Backends"
- Three backend cards STACKED VERTICALLY but COMPRESSED (~40% total height, each card short):
  Card 1 BLUE "HuggingFace Transformers" — keep v2 Backend A mini-pipeline: Tokenize -> Layer 1 -> ... -> Layer N -> Decode, footer "Eager reference implementation", NO latency text
  Card 2 ORANGE "vLLM" — keep v2 Backend B batched inference -> Shared Compute / paged KV, footer "Continuous batching"
  Card 3 ROSE/GREEN "TGI" — keep v2 Backend C parallel branches -> Serving Output, footer "Production serving stack"
- Keep internal flow diagrams small and dense like v2, NOT full-width tall cards

BAND 3 — header "Benchmark Outcome"
- One compact white card (~35% total height), internal left-right OK inside this card only:
  LEFT: bar chart "Exact Match Accuracy", 3 backend bars (blue/amber/rose), slightly different heights, NO numeric labels
  RIGHT: three small answer chips: Canberra / The capital is Canberra / Canberra, Australia
- Bottom row: three colored pills (HF reference, vLLM serving, TGI production)
- Tiny corner legend: blue HF, amber vLLM, rose TGI

CONTENT: LLM backend evaluation motivation ONLY. Remove all latency, throughput, Backend A/B/C, speculative decoding labels.

STRICT: NO top figure title, NO bottom banner, NO ampersand, NO bullet lists, NO numeric scores on chart.
""".strip()


def _run_gemini_edit(*, src: Path, out: Path, prompt: str) -> int:
    api_key = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY", "GEMINI_NATIVE_API_KEY")
    base_url = _normalize_gemini_native_base_url(
        _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL", "GEMINI_NATIVE_BASE_URL")
    )
    model = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "IMAGE_GEN_MODEL_NAME") or "gemini-2.5-flash-image"
    if not api_key or not base_url:
        print("Missing Gemini credentials", file=sys.stderr)
        return 3

    b64 = base64.b64encode(src.read_bytes()).decode("ascii")
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
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
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
            print(f"Gemini edit failed: {detail}", file=sys.stderr)
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

    print("Gemini edit failed: " + "; ".join(errors[-3:]), file=sys.stderr)
    return 4


def main() -> int:
    _ensure_paperbanana_env()
    if not SRC.exists():
        print(f"Missing source image: {SRC}", file=sys.stderr)
        return 2
    backup = OUT.with_name(OUT.stem + "_prev.png")
    if OUT.exists():
        OUT.replace(backup)
        print(f"Backed up previous output to {backup}")
    return _run_gemini_edit(src=SRC, out=OUT, prompt=EDIT_PROMPT)


if __name__ == "__main__":
    raise SystemExit(main())
