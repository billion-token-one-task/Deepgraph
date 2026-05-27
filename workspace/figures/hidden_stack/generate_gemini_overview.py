#!/usr/bin/env python3
"""Generate BAVD overview figure — API-style neural architecture diagram via Gemini."""

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

STYLE_REF = ROOT / "workspace/figures/hidden_stack/fig_motivation_gemini.png"
COLOR_REF = ROOT / "workspace/figures/hidden_stack/style_ref_api_colors.png"
SRC_EDIT = ROOT / "workspace/figures/hidden_stack/fig_overview_gemini.png"
OUT = ROOT / "workspace/figures/hidden_stack/fig_overview_gemini.png"

PALETTE_RULES = """
PROFESSIONAL COLOR PALETTE (match first attached color-reference figure exactly):
- Canvas background: off-white #F9F9F9, NOT saturated panel blocks
- HuggingFace / primary blue: muted steel blue #5B9BD5 (like Audio in reference)
- vLLM / warm accent: golden amber #EDAD50 (like Text in reference), NOT neon orange
- TGI / secondary accent: dusty rose #E27D7D (like Vision in reference), NOT bright green
- Left panel tint: barely-there cool gray-blue #F5F8FC
- Top-right panel tint: warm cream #FFFBF3 (like reference yellow-beige zones)
- Bottom-right panel tint: warm off-white #FAF7F7 with faint rose hints
- Container borders: thin light gray #D0D0D0, subtle drop shadows
- Main flow arrows: soft blue gradient (not dark gray)
- Residual/neutral: warm gray #B8B8B8, Interaction: soft lavender #C4A8D8
- NO neon teal, NO saturated mint green, NO harsh orange blocks, NO primary RGB colors
""".strip()

RECOLOR_AND_CLEAN_PROMPT = f"""
You receive TWO images.
IMAGE 1 = color/style reference.
IMAGE 2 = BAVD overview to refine.

Apply BOTH tasks to IMAGE 2:

TASK A — Recolor using IMAGE 1 palette:
{PALETTE_RULES}

TASK B — Remove ALL title/header bars (top title, panel headers, sub-zone headers like Component Aggregation, Backend Marginalization, Robust Comparison, Optimized Prototype Pool). Fill with matching subtle background.

KEEP: layout, formulas, icons, matrix, charts, legends, BAVD label, dataset/method/backend names.
NO new titles. NO saturated neon blocks.
""".strip()

RECOLOR_PROMPT = f"""
Edit the attached BAVD overview figure. RECOLOR ONLY — do not change layout, content, or text.

Apply this muted ACL-style palette (replace all current saturated colors):
{PALETTE_RULES}
- Dataset accent greens: muted sage #8FB896

CRITICAL CONTENT LOCK — keep every panel exactly as attached:
- Left: datasets, matrix, Qwen chip
- Top-right: Bernoulli formula, variance stars, prototype pool map
- Bottom-right: Backend Marginalization, BAVD network, Robust Comparison intervals

DO NOT add DIAM, Prediction, Audio/Text/Vision prototypes, or multimodal content from any reference.
DO NOT remove or simplify elements. Change colors and tonal quality only.
""".strip()

CLEAR_PROMPT = f"""
Draw a NEW publication-quality BAVD overview figure that is EASY TO READ (concrete, not abstract).
Match attached motivation figure for clarity: three clear columns, gray mini-headers, white inner cards, block arrows.

READABILITY RULE — every panel must be instantly understandable to an ACL reviewer without decoding symbols:
- Use labeled tables, labeled bar charts, labeled heatmaps, and short plain-English captions (max 6 words each).
- NO abstract elements: NO voronoi maps, NO prototype pools, NO star constellations, NO feature-cloud blobs, NO neural-network decoration, NO cosine-similarity stars.
- NO Step 1/2/3 numbering. NO top figure title. NO ampersand.

{PALETTE_RULES}

=== COLUMN 1 (left, subtle blue tint) ===
Gray header: "Controlled Benchmark Grid"

White card 1 — model chip: Qwen2.5-7B-Instruct, frozen weights, greedy decoding.

White card 2 — READABLE TABLE titled "Methods x Backends x Seeds":
- Column headers (color pills): HuggingFace (blue), vLLM (amber), TGI (rose)
- Row labels: Direct, CoT, Self-Consistency (3 rows enough)
- Each cell: three small seed icons + one-line text "3 replicates"
- Footer dataset row: GSM8K, MuSiQue, StrategyQA, 2Wiki as labeled folder chips

White card 3 — CONCRETE example (not abstract):
- Prompt line: "Q: Capital of Australia?"
- Three answer lines color-coded by backend:
  HF: "Canberra"
  vLLM: "The capital is Canberra"
  TGI: "Canberra, Australia"
- Tiny caption: "Same weights, different strings"

=== COLUMN 2 (center, subtle cream tint) ===
Gray header: "Variance Decomposition"

White card 1 — READABLE stacked horizontal bar chart titled "What drives score variance?":
- Five labeled segments with icons: Method (gear), Backend (server), Dataset (folder), Method-Backend (link), Residual (wave)
- Different segment widths, NO numeric labels

White card 2 — simple mixed-effects schematic (literal boxes, not abstract):
- Center box: "Example correctness"
- Three arrows in TO it from labeled boxes: Method effect, Backend effect, Dataset effect
- One small LaTeX line only: logit(p) = mu + u_m + v_b + s_d

=== COLUMN 3 (right, subtle rose tint) ===
Gray header: "Backend-Aware Reporting"

White card 1 — READABLE grouped bar chart titled "Raw exact-match per backend":
- Two method groups (Method A, Method B)
- Three colored bars per group (HF blue, vLLM amber, TGI rose), different heights, NO axis numbers

White card 2 — READABLE before/after titled "Marginalize over backends":
- Left: three scattered backend dots
- Arrow
- Right: one centered dot labeled "Backend-marginalized score"

White card 3 — READABLE interval chart titled "Robust method comparison":
- Two horizontal intervals with labels Method A and Method B
- Method A: green check + caption "stable rank"
- Method B: red swap icon + caption "rank changes across backends"

Corner legend: blue HF, amber vLLM, rose TGI.

Keep professional muted palette, shadows, rounded cards — but prioritize CLARITY over decorative density.
""".strip()

EDIT_PROMPT = """
Edit the attached scientific figure. Remove ALL titles and title header bars. Keep everything else identical.

REMOVE completely (bar + text):
- Top figure title if any remains
- Panel headers: BENCHMARK INPUTS, VARIANCE DECOMPOSITION, BACKEND-AWARE REPORTING
- Sub-zone headers: Component Aggregation, Optimized Prototype Pool, Backend Marginalization, Robust Comparison

PRESERVE exactly:
- All layout, pastel backgrounds, icons, matrices, formulas, charts, arrows, legends, colors, shadows
- Dataset names, method names, backend names, LaTeX math, BAVD label on network, dashed legend box

Seamlessly fill removed title areas with matching panel background. No new titles or header bars.
""".strip()

PROMPT = f"""
Draw a BRAND NEW publication-quality neural-architecture-style scientific diagram.
Copy the EXACT drawing style of the attached reference figure (soft pastel module backgrounds, 3D stacked layer icons, feature-cloud blobs with geometric symbols, integrated LaTeX math boxes, mini bar charts, cluster diagrams, thick gradient arrows, dashed legend boxes, rounded sub-panels WITHOUT any title headers).

TOPIC: Backend-Aware Variance Decomposition (BAVD) for LLM benchmark evaluation.
Do NOT copy the reference figure's content (multimodal sentiment / API / prototypes). Only copy its visual style and layout grammar.

=== OVERALL LAYOUT (match reference structure) ===
One large outer rounded container. Inside:
- LEFT vertical column (~30% width): benchmark inputs and evaluation matrix
- RIGHT area (~70% width) split into TWO stacked horizontal sub-panels:
  TOP sub-panel with soft yellow/cream background
  BOTTOM sub-panel with soft teal/mint background
Thick shaded arrow from left column into top-right sub-panel, then thick arrow from top-right to bottom-right.

=== LEFT COLUMN (soft powder blue background, NO title header) ===
Four stacked 3D layer icons (like Audio/Text/Vision in reference), each with icon + short label:
- GSM8K (calculator icon, blue layers)
- MuSiQue (book icon, yellow layers)
- StrategyQA (lightbulb icon, orange layers)
- 2Wiki (link icon, green layers)

Next to each stack, draw a feature-cloud blob containing small geometric symbols:
- solid circles = correct examples, dashed circles = incorrect examples (qualitative mix, no counts)

Below stacks, one compact chip: Qwen2.5-7B frozen weights.

Main lower-left visual — method-by-backend matrix inside a rounded white card:
- 5 rows with tiny icons: Direct, CoT, SC, Routing, Oracle
- 3 columns color-coded: HuggingFace blue, vLLM orange, TGI green
- Each cell = mini feature cloud (3-5 symbols, mix of solid/dashed)

Small dashed legend box: solid = correct, dashed = mismatch.

=== TOP RIGHT SUB-PANEL (soft yellow background, NO title header) ===

Left zone (no title bar):
- Cluster of colored dots (blue method, orange backend, green dataset) filtering into centroid
- Tiny LaTeX formula box (no numeric values): y ~ Bernoulli(p), logit(p) = mu + u_m + v_b + ...

Center zone — horizontal spectrum axis (like Negative/Positive in reference):
- Five labeled regions along axis: Method, Backend, Dataset, Interaction, Residual
- Colored star markers on axis at qualitative positions (Method and Residual largest)

Right zone (no title bar, like Optimized Prototype Pool visual):
- Five shape clusters separated by smooth curved boundaries
- Each cluster has distinct color and icon (gear, server, folder, link, wave)
- Small LaTeX: eta_backend = sigma_b^2 / Sigma^2 (symbolic only, no numbers)

=== BOTTOM RIGHT SUB-PANEL (soft teal background, NO title header) ===

Left zone with lightbulb icon (no title bar):
- Three backend prototype stars (blue, orange, green) compared to method query star
- Mini cosine-similarity style double arrows between stars
- Three tiny bar charts below (one per backend, different heights, NO axis numbers)

Center zone — mini neural graph (nodes and edges, like APG in reference):
- Small network labeled BAVD producing gamma/beta style parameters OR interval endpoints
- LaTeX: p_tilde_m = E_b[p_mb]
- Element-wise dot and plus icons connecting blocks

Right zone — vertical output card (like Prediction in reference, no title bar):
- Three horizontal interval segments stacked (methods A, B, C)
- Green check on stable rank, red swap icon on unstable
- Final output star at bottom

Dashed legend bottom-right: blue HuggingFace, orange vLLM, green TGI, star = marginalized score.

=== STYLE RULES (critical) ===
{PALETTE_RULES}
- Match reference: thin borders, LaTeX math in small rounded boxes, 3D stack icons, feature clouds, cluster boundaries, soft blue gradient arrows.
- GRAND and element-rich like reference — dense professional NeurIPS/ACL architecture figure.
- NO Step 1/2/3/4/5 labels, NO numbered pipeline stages.
- NO bullet lists, NO long paragraphs, NO ampersand character.
- NO numeric benchmark scores (no 92%, no accuracy values, no axis tick numbers).
- NO titles or header bars anywhere (no panel titles, no sub-panel titles, no top figure title, no bottom banner).
- NOT a simple 3-column slide — use the reference's left-column + right-stacked-modules layout.
""".strip()


def _run_gemini(
    *,
    style_ref: Path,
    out: Path,
    prompt: str,
    extra_refs: list[Path] | None = None,
) -> int:
    api_key = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY", "GEMINI_NATIVE_API_KEY")
    base_url = _normalize_gemini_native_base_url(
        _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL", "GEMINI_NATIVE_BASE_URL")
    )
    model = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "IMAGE_GEN_MODEL_NAME") or "gemini-2.5-flash-image"
    if not api_key or not base_url:
        print("Missing Gemini credentials", file=sys.stderr)
        return 3

    parts: list[dict] = []
    for idx, ref in enumerate([*(extra_refs or []), style_ref], start=1):
        b64 = base64.b64encode(ref.read_bytes()).decode("ascii")
        parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})
        if extra_refs and idx == 1:
            parts.append({"text": f"IMAGE {idx} (color/style reference)."})
        elif extra_refs and idx == 2:
            parts.append({"text": f"IMAGE {idx} (diagram to edit)."})
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
    import os
    import shutil

    _ensure_paperbanana_env()
    backup = OUT.with_name(OUT.stem + "_prev.png")

    mode = os.environ.get("OVERVIEW_MODE", "clear").strip().lower()
    if mode == "clear":
        if OUT.exists():
            shutil.copy2(OUT, backup)
            print(f"Backed up previous output to {backup}")
        if not STYLE_REF.exists():
            print(f"Missing style reference: {STYLE_REF}", file=sys.stderr)
            return 2
        return _run_gemini(style_ref=STYLE_REF, out=OUT, prompt=CLEAR_PROMPT)

    if mode == "generate":
        if OUT.exists():
            shutil.copy2(OUT, backup)
            print(f"Backed up previous output to {backup}")
        if not STYLE_REF.exists():
            print(f"Missing style reference: {STYLE_REF}", file=sys.stderr)
            return 2
        color_ref = COLOR_REF if COLOR_REF.exists() else STYLE_REF
        return _run_gemini(
            style_ref=color_ref,
            out=OUT,
            prompt=PROMPT,
            extra_refs=[STYLE_REF] if color_ref != STYLE_REF else None,
        )

    edit_src = SRC_EDIT if SRC_EDIT.exists() else backup
    if not edit_src.exists():
        print(f"Missing source for edit: {edit_src}", file=sys.stderr)
        return 2
    if edit_src == SRC_EDIT and SRC_EDIT.exists():
        shutil.copy2(SRC_EDIT, backup)
        print(f"Backed up previous output to {backup}")

    if mode == "recolor":
        color_ref = COLOR_REF if COLOR_REF.exists() else None
        src = ROOT / "workspace/figures/hidden_stack/fig_overview_gemini_prev2.png"
        if not src.exists():
            src = edit_src
        return _run_gemini(
            style_ref=src,
            out=OUT,
            prompt=RECOLOR_PROMPT,
            extra_refs=[color_ref] if color_ref else None,
        )

    if mode == "recolor_clean":
        color_ref = COLOR_REF if COLOR_REF.exists() else STYLE_REF
        src = ROOT / "workspace/figures/hidden_stack/fig_overview_gemini_prev.png"
        if not src.exists():
            src = edit_src
        return _run_gemini(
            style_ref=src,
            out=OUT,
            prompt=RECOLOR_AND_CLEAN_PROMPT,
            extra_refs=[color_ref],
        )

    return _run_gemini(style_ref=edit_src, out=OUT, prompt=EDIT_PROMPT)


if __name__ == "__main__":
    raise SystemExit(main())
