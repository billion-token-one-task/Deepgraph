"""Shared Gemini image generation helpers for Kairos figures."""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

from paperbanana_wrapper import (
    _env_first,
    _http_error_message,
    _image_attempt_count,
    _image_retry_sleep_seconds,
    _is_retriable_image_error,
    _normalize_gemini_native_base_url,
)

PALETTE = """
- Canvas: off-white #F9F9F9
- Event A / primary: steel blue #5B9BD5
- Event B / secondary: golden amber #EDAD50
- Counterfactual / update: dusty rose #E27D7D
- Relation graph: soft lavender #C4A8D8
- Gray section headers, white inner cards, thin #D0D0D0 borders, soft shadows
- NO neon colors, NO ampersand character
"""


def run_gemini(*, style_ref: Path, out: Path, prompt: str, extra_refs: list[Path] | None = None) -> int:
    api_key = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY", "GEMINI_NATIVE_API_KEY")
    base_url = _normalize_gemini_native_base_url(
        _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL", "GEMINI_NATIVE_BASE_URL")
    )
    model = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "IMAGE_GEN_MODEL_NAME") or "gemini-2.5-flash-image"
    if not api_key or not base_url:
        print("Missing Gemini credentials", file=err_stream())
        return 3

    parts: list[dict] = []
    for ref in [*(extra_refs or []), style_ref]:
        parts.append(
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(ref.read_bytes()).decode("ascii")}}
        )
    parts.append({"text": prompt})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
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

                time.sleep(_image_retry_sleep_seconds(attempt))
                continue
            print(f"Gemini failed: {detail}", file=err_stream())
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

    print("Gemini failed: " + "; ".join(errors[-3:]), file=err_stream())
    return 4


def err_stream():
    import sys

    return sys.stderr
