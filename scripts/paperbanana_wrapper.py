#!/usr/bin/env python3
"""DeepGraph bridge for one-shot PaperBanana diagram generation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


DEEPGRAPH_ROOT = Path(__file__).resolve().parents[1]
PAPERBANANA_ROOT = Path.home() / "PaperBanana"
PAPERBANANA_PYTHON = PAPERBANANA_ROOT / ".venv" / "bin" / "python"
PAPERBANANA_ENTRY = PAPERBANANA_ROOT / "skill" / "run.py"
PAPERBANANA_CONFIG = PAPERBANANA_ROOT / "configs" / "model_config.yaml"

SUPPORTED_RATIOS = ("21:9", "16:9", "3:2")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _looks_like_openrouter(key: str) -> bool:
    return key.startswith("sk-or-v1-")


def _looks_like_google(key: str) -> bool:
    return key.startswith("AIza")


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _normalize_openai_compatible_base_url(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith(("/v1", "/v1beta")):
        return value
    return f"{value}/v1"


def _normalize_gemini_native_base_url(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/v1beta"):
        return value[: -len("/v1beta")]
    if value.endswith("/v1"):
        return value[: -len("/v1")]
    return value


def _ensure_paperbanana_env() -> None:
    _load_dotenv(DEEPGRAPH_ROOT / ".env")

    deepgraph_key = os.environ.get("DEEPGRAPH_LLM_API_KEY", "")
    deepgraph_base_url = _normalize_openai_compatible_base_url(os.environ.get("DEEPGRAPH_LLM_BASE_URL", ""))
    deepgraph_model = os.environ.get("DEEPGRAPH_LLM_MODEL", "").strip()
    image_protocol = (_env_first("DEEPGRAPH_PAPERBANANA_IMAGE_PROTOCOL") or "openai_compatible").strip().lower()

    if not os.environ.get("OPENAI_API_KEY"):
        candidate = _env_first("DEEPGRAPH_PAPERBANANA_TEXT_API_KEY", "DEEPGRAPH_LLM_API_KEY")
        if candidate:
            os.environ["OPENAI_API_KEY"] = candidate

    if not os.environ.get("OPENAI_BASE_URL"):
        candidate = _normalize_openai_compatible_base_url(
            _env_first("DEEPGRAPH_PAPERBANANA_TEXT_BASE_URL", "DEEPGRAPH_LLM_BASE_URL")
        )
        if candidate:
            os.environ["OPENAI_BASE_URL"] = candidate

    if image_protocol == "gemini_native":
        if not os.environ.get("GEMINI_NATIVE_API_KEY"):
            candidate = _env_first(
                "DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY",
                "DEEPGRAPH_PAPERBANANA_GEMINI_API_KEY",
            )
            if candidate:
                os.environ["GEMINI_NATIVE_API_KEY"] = candidate
        if not os.environ.get("GEMINI_NATIVE_BASE_URL"):
            candidate = _normalize_gemini_native_base_url(
                _env_first(
                    "DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL",
                    "DEEPGRAPH_PAPERBANANA_GEMINI_BASE_URL",
                )
            )
            if candidate:
                os.environ["GEMINI_NATIVE_BASE_URL"] = candidate
    else:
        if not os.environ.get("OPENROUTER_API_KEY"):
            candidate = _env_first(
                "DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY",
                "DEEPGRAPH_PAPERBANANA_OPENROUTER_API_KEY",
            )
            if candidate:
                os.environ["OPENROUTER_API_KEY"] = candidate
            elif _looks_like_openrouter(deepgraph_key):
                os.environ["OPENROUTER_API_KEY"] = deepgraph_key

        if not os.environ.get("OPENROUTER_BASE_URL"):
            candidate = _normalize_openai_compatible_base_url(
                _env_first(
                    "DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL",
                    "DEEPGRAPH_PAPERBANANA_OPENROUTER_BASE_URL",
                )
            )
            if candidate:
                os.environ["OPENROUTER_BASE_URL"] = candidate

    if not os.environ.get("GOOGLE_API_KEY"):
        candidate = os.environ.get("DEEPGRAPH_PAPERBANANA_GOOGLE_API_KEY", "")
        if candidate:
            os.environ["GOOGLE_API_KEY"] = candidate
        elif _looks_like_google(deepgraph_key):
            os.environ["GOOGLE_API_KEY"] = deepgraph_key

    os.environ.setdefault(
        "MAIN_MODEL_NAME",
        _env_first("DEEPGRAPH_PAPERBANANA_MAIN_MODEL") or deepgraph_model or "gpt-5.4",
    )
    os.environ.setdefault(
        "IMAGE_GEN_MODEL_NAME",
        _env_first(
            "DEEPGRAPH_PAPERBANANA_IMAGE_MODEL",
            "DEEPGRAPH_PAPERBANANA_GEMINI_MODEL",
            "DEEPGRAPH_PAPERBANANA_OPENROUTER_MODEL",
        )
        or "gemini-2.5-flash-image",
    )

    if deepgraph_base_url:
        os.environ.setdefault("DEEPGRAPH_PAPERBANANA_TEXT_BASE_URL", deepgraph_base_url)


def _ensure_model_config() -> None:
    if PAPERBANANA_CONFIG.exists():
        return
    PAPERBANANA_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PAPERBANANA_CONFIG.write_text(
        "\n".join(
            [
                "defaults:",
                '  main_model_name: "gpt-5.4"',
                '  image_gen_model_name: "gemini-2.5-flash-image"',
                "api_keys:",
                '  google_api_key: ""',
                '  gemini_native_api_key: ""',
                '  openai_api_key: ""',
                '  anthropic_api_key: ""',
                '  openrouter_api_key: ""',
                "api_base_urls:",
                '  gemini_native_base_url: ""',
                '  openai_base_url: ""',
                '  openrouter_base_url: ""',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _ratio_value(label: str) -> float:
    try:
        left, right = label.split(":", 1)
        return float(left) / float(right)
    except (ValueError, ZeroDivisionError):
        return 16 / 9


def _nearest_supported_ratio(raw: str | None) -> str:
    if not raw:
        return "16:9"
    raw = str(raw).strip()
    if raw in SUPPORTED_RATIOS:
        return raw
    target = _ratio_value(raw)
    return min(SUPPORTED_RATIOS, key=lambda label: abs(_ratio_value(label) - target))


def _clip(text: Any, limit: int = 800) -> str:
    value = str(text or "").strip()
    return value[:limit]


def _list_block(values: list[Any], *, limit: int = 5) -> str:
    lines: list[str] = []
    for item in values[:limit]:
        if isinstance(item, dict):
            text = item.get("name") or item.get("title") or item.get("model") or json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        text = text.strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def _build_content(spec: dict[str, Any]) -> str:
    fig = spec.get("figure") or {}
    plan = spec.get("experimental_plan") or {}
    sections = [
        "# Paper Context",
        f"Title: {_clip(spec.get('state_title'), 200)}",
        f"Method name: {_clip(spec.get('method_name'), 200)}",
        "",
        "# Figure Goal",
        f"Figure title: {_clip(fig.get('title') or fig.get('figure_id'), 240)}",
        f"Figure type: {_clip(fig.get('plot_type'), 80)}",
        f"Objective: {_clip(fig.get('objective'), 600)}",
        f"Data source: {_clip(fig.get('data_source'), 120)}",
        "",
        "# Method Summary",
        _clip(spec.get("problem_statement"), 1200),
        "",
        _clip(spec.get("existing_weakness"), 800),
        "",
        _clip(spec.get("method_summary"), 1800),
    ]

    contributions = spec.get("contributions") or []
    if contributions:
        sections.extend(["", "# Key Contributions", _list_block(contributions, limit=6)])

    datasets = plan.get("datasets") or []
    if datasets:
        sections.extend(["", "# Datasets", _list_block(datasets, limit=6)])

    baselines = plan.get("baselines") or []
    if baselines:
        sections.extend(["", "# Baselines", _list_block(baselines, limit=6)])

    metric_name = spec.get("baseline_metric_name")
    baseline = spec.get("baseline_metric_value")
    best = spec.get("best_metric_value")
    effect_pct = spec.get("effect_pct")
    verdict = spec.get("verdict")
    if any(value is not None for value in (metric_name, baseline, best, effect_pct, verdict)):
        sections.extend(
            [
                "",
                "# Experiment Snapshot",
                f"Metric: {_clip(metric_name, 120)}",
                f"Baseline: {baseline}",
                f"Best: {best}",
                f"Effect pct: {effect_pct}",
                f"Verdict: {_clip(verdict, 80)}",
            ]
        )

    evidence_summary = _clip(spec.get("evidence_summary"), 1200)
    if evidence_summary:
        sections.extend(["", "# Evidence Summary", evidence_summary])

    return "\n".join(line for line in sections if line is not None).strip()


def _build_caption(spec: dict[str, Any]) -> str:
    fig = spec.get("figure") or {}
    title = _clip(fig.get("title") or spec.get("state_title"), 180)
    objective = _clip(fig.get("objective"), 240)
    if title and objective and objective.lower() not in title.lower():
        return f"{title}. {objective}"
    return title or objective or "Framework overview"


def _is_motivation_overview_spec(spec: dict[str, Any]) -> bool:
    fig = spec.get("figure") or {}
    text = " ".join(
        str(part or "")
        for part in (
            fig.get("figure_id"),
            fig.get("title"),
            fig.get("objective"),
            spec.get("state_title"),
            spec.get("problem_statement"),
        )
    ).lower()
    return any(token in text for token in ("motivation", "overview", "teaser", "problem-method-result", "problem method result"))


def _check_credentials() -> tuple[bool, str]:
    image_protocol = (_env_first("DEEPGRAPH_PAPERBANANA_IMAGE_PROTOCOL") or "").strip().lower()
    if image_protocol == "openai_compatible" and _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY") and _openai_image_base_url():
        return True, "openai_compatible_image"
    if os.environ.get("GEMINI_NATIVE_API_KEY") and os.environ.get("GEMINI_NATIVE_BASE_URL"):
        return True, "gemini_native"
    if os.environ.get("OPENROUTER_API_KEY"):
        return True, "openrouter"
    if os.environ.get("GOOGLE_API_KEY"):
        return True, "google"
    if os.environ.get("OPENAI_API_KEY") and "gpt-image" in os.environ.get("IMAGE_GEN_MODEL_NAME", ""):
        return True, "openai"
    return False, "missing"


def _image_prompt(spec: dict[str, Any], *, caption: str, content: str) -> str:
    fig = spec.get("figure") or {}
    if _is_motivation_overview_spec(spec):
        fig_text = " ".join(str(fig.get(key) or "") for key in ("figure_id", "title", "objective")).lower()
        figure_role = "motivation" if "motivation" in fig_text else "overview"
        cleaned_caption = _clip(caption, 700)
        title = str(fig.get("title") or "").strip()
        if title and cleaned_caption.lower().startswith(f"{title.lower()}."):
            cleaned_caption = cleaned_caption[len(title) + 1 :].strip()
        context_lines = [
            f"Paper title: {_clip(spec.get('state_title'), 200)}",
            f"Method name: {_clip(spec.get('method_name'), 200)}",
            f"Problem statement: {_clip(spec.get('problem_statement'), 800)}",
            f"Existing weakness: {_clip(spec.get('existing_weakness'), 600)}",
            f"Method summary: {_clip(spec.get('method_summary'), 1200)}",
        ]
        contributions = spec.get("contributions") or []
        if contributions:
            context_lines.extend(["Key contributions:", _list_block(contributions, limit=6)])
        cleaned_context = "\n".join(line for line in context_lines if line.strip())
        if figure_role == "motivation":
            schema = (
                "This figure should function as the paper's motivation figure. "
                "It should look like a publication framework figure with clear regions, concise labels, and a visible scientific contrast. "
                "The reader should immediately understand what the problem is, why it matters, what is insufficient in the current setting, and what key contrast motivates the proposed direction."
            )
        else:
            schema = (
                "This figure should function as the paper's overview figure. "
                "It should summarize the main mechanism or conceptual structure of the method in one unified framework diagram. "
                "Use a structured multi-region layout with grouped modules, concise labels, arrows, and a clear semantic flow. "
                "The reader should understand the core idea at a glance."
            )
        return "\n".join(
            [
                "You are an experienced scientific figure designer preparing a camera-ready figure for a machine learning paper.",
                "Carefully read the paper context and the figure intent, fully understand the research content, and produce a figure suitable for academic publication.",
                schema,
                "The figure should be understandable at a glance, even before the reader studies the full paper.",
                "Prefer a wide publication-style framework layout with 3 to 5 clearly separated functional regions across the canvas. Each region should have an obvious role in the scientific story.",
                "Do not make it a plain left-to-right pipeline or a generic flowchart. Create a denser scientific composition with local substructures, internal comparisons, and grouped modules.",
                "Avoid three equally sized vertical slabs with a single arrow passing through them. Prefer one dominant dense working area plus one or two supporting grouped regions, or an asymmetric multi-cluster arrangement.",
                "Take inspiration from strong editorial scientific figures: use asymmetric layout, a clear focal region, supporting side clusters, fan-in or fan-out connectors, and local density variation rather than uniform columns.",
                "A good pattern is: one contextual cluster, one bridge or interface cluster, and one dense main analytical cluster, with a small integrated legend or semantic key in a corner if needed.",
                "Do not center the figure around one giant symbolic object. The main structure should come from grouped panels, modules, and connections rather than a single metaphor shape.",
                "Do not place a large title at the top of the image.",
                "Never add a standalone figure heading such as Figure 1, Motivation, Overview, System Overview, Framework, or any caption-like sentence anywhere in the image.",
                "Do not create giant comparison banners such as Traditional X vs Proposed Y across the top. If a comparison is necessary, express it with local grouped modules and small embedded labels only.",
                "Do not add a detached bottom takeaway box, key insight box, or summary strip outside the main composition.",
                "Do not place a bottom caption, footnote, or explanatory paragraph inside the image.",
                "Short in-figure labels are allowed and encouraged when they improve scientific clarity. Use concise framework-style labels, module names, arrow labels, and compact legends when necessary.",
                "Use a disciplined text hierarchy: small integrated panel headers, short module labels inside boxes, and very short arrow labels. Avoid giant all-caps banner text spanning the full canvas.",
                "If region names are needed, embed them inside the relevant panel and keep them secondary. Do not place oversized text floating above large regions.",
                "All visible text should use Times New Roman or a very close academic serif font. Avoid sans-serif, poster-like display fonts, handwritten styles, or playful typography.",
                "Design it like a strong conference framework figure: organized blocks, grouped regions, rounded rectangles when useful, arrows or connectors where they clarify logic, and a composition that feels authored rather than templated.",
                "Do not over-expand low-level implementation detail. Keep the abstraction at the right level for a publication figure.",
                "Do not let generic background context occupy too much of the canvas. Large low-information background panels are discouraged. Reserve most visual emphasis for the method logic and the main scientific contrast.",
                "Use semantic consistency: similar roles should share consistent color, shape, visual weight, iconography, and placement logic. Assign a small palette of 3 to 4 semantic colors and reuse them consistently.",
                "Each region should contain meaningful internal structure: nested boxes, grouped items, small comparisons, aligned rows, or compact examples. Avoid large empty washes with only one object inside.",
                "Use a few necessary concrete icons or data thumbnails when they improve comprehension, such as document, message, cache, embedding, user, model, dataset, or output icons. They should be clean, intentional, and tied to real modules rather than decorative.",
                "When reusing visual elements, introduce controlled differences so repeated modules are not mechanically identical. Avoid long stacks of near-duplicate cards or repeated clipart blocks.",
                "Allow multiple meaningful visual elements if the figure needs them, but every element must have a clear role in the scientific explanation.",
                "Avoid decorative concept art, giant symbolic brains, clouds, funnels, waves, logo walls, random icon piles, or visually flashy but semantically empty motifs.",
                "Avoid collage-like card stacking. The figure should feel like an engineered layout, not a pile of decorative tiles.",
                "Avoid generic stock shapes that scream AI-generated infographic, such as oversized trapezoid encoders, giant ribbon arrows, or repeated empty neural-network clipart unless grounded in a precise scientific role.",
                "Do not turn the figure into a toy infographic. It should read like a polished framework figure from a top ML paper.",
                "Use whitespace well, but do not oversimplify the figure into a vague sparse composition. A richer multi-block framework figure is acceptable if it improves clarity.",
                "Create visual texture through meaningful structure: nested containers, aligned micro-elements, varied line weights, subtle shadows, and local detail. Do not rely on huge gradients or oversized empty background areas for style.",
                "A compact legend strip or semantic key is allowed when useful. If used, make it small, integrated, and tucked into a corner or margin. Never let the legend become a bottom-wide banner.",
                "Favor non-uniform occupancy: let important regions be denser and larger, and let supporting regions be smaller and more compact. Avoid evenly distributing empty space across the canvas.",
                "The visual style should resemble a modern academic framework diagram template: white background, restrained local tinting, grouped panels, rounded modules, controlled outlines, balanced spacing, moderate line weights, arrows with clear direction, and clean readable labels.",
                "The output should look like a serious NeurIPS/ICLR/ICML figure prepared for publication.",
                f"Figure-specific intent: {_clip(fig.get('objective') or caption, 700)}",
                f"Figure caption context from the paper: {cleaned_caption}",
                f"Paper context to read and use: {_clip(cleaned_context, 2200)}",
            ]
        ).strip()
    labels = [
        str(part)
        for part in [
            spec.get("state_title"),
            spec.get("method_name"),
            fig.get("title"),
            fig.get("objective"),
        ]
        if part
    ]
    return "\n".join(
        [
            "Create one clean publication-quality scientific diagram for a machine learning paper.",
            "Use a restrained academic vector style: white background, crisp boxes, thin arrows, high contrast.",
            "Do not create fake numeric charts or unsupported values.",
            "Prefer 3 to 5 labeled blocks connected by arrows; keep labels large and readable.",
            f"Caption: {_clip(caption, 220)}",
            f"Objective: {_clip(fig.get('objective'), 260)}",
            f"Suggested labels: {_clip('; '.join(labels), 360)}",
        ]
    ).strip()


def _openai_image_base_url() -> str:
    return _normalize_openai_compatible_base_url(
        _env_first(
            "DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL",
            "DEEPGRAPH_PAPERBANANA_OPENAI_IMAGE_BASE_URL",
            "DEEPGRAPH_LLM_BASE_URL",
        )
    )


def _run_openai_compatible_image_generation(
    *,
    output_path: Path,
    prompt: str,
) -> int:
    api_key = _env_first(
        "DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY",
        "DEEPGRAPH_PAPERBANANA_OPENAI_IMAGE_API_KEY",
        "OPENAI_API_KEY",
    )
    base_url = _openai_image_base_url()
    model = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "IMAGE_GEN_MODEL_NAME") or "gpt-image-2"
    size = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_SIZE") or "1024x1024"
    if not api_key or not base_url:
        print("OpenAI-compatible image generation is missing API key or base URL.", file=sys.stderr)
        return 3

    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    url = f"{base_url}/images/generations"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    body: dict[str, Any] | None = None
    errors: list[str] = []
    client = (_env_first("DEEPGRAPH_PAPERBANANA_IMAGE_HTTP_CLIENT") or "").strip().lower()
    prefer_curl = client == "curl" or (client != "urllib" and os.name == "nt" and _curl_binary())
    attempts = _image_attempt_count()

    if prefer_curl:
        for attempt in range(1, attempts + 1):
            body, error = _post_openai_image_payload_with_curl(url=url, api_key=api_key, payload=payload)
            if body is not None:
                break
            errors.append(f"curl_attempt_{attempt}:{error}")
            if attempt < attempts:
                time.sleep(min(10, 2 * attempt))
    else:
        body, error = _post_openai_image_payload_with_urllib(url=url, api_key=api_key, payload=payload)
        if error:
            errors.append(f"urllib:{error}")
        if body is None:
            for attempt in range(1, attempts + 1):
                body, error = _post_openai_image_payload_with_curl(url=url, api_key=api_key, payload=payload)
                if body is not None:
                    break
                errors.append(f"curl_attempt_{attempt}:{error}")
                if attempt < attempts:
                    time.sleep(min(10, 2 * attempt))

    if body is None:
        print(f"OpenAI-compatible image generation failed: {'; '.join(errors)}", file=sys.stderr)
        return 4

    return _write_openai_image_response(output_path=output_path, body=body)


def _run_gemini_native_image_generation(
    *,
    output_path: Path,
    prompt: str,
    aspect_ratio: str,
) -> int:
    api_key = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY", "GEMINI_NATIVE_API_KEY")
    base_url = _normalize_gemini_native_base_url(
        _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL", "GEMINI_NATIVE_BASE_URL")
    )
    model = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "IMAGE_GEN_MODEL_NAME") or "gemini-2.5-flash-image"
    if not api_key or not base_url:
        print("Gemini-native image generation is missing API key or base URL.", file=sys.stderr)
        return 3

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio},
        },
    }
    url = f"{base_url}/v1beta/models/{model}:generateContent"
    request = urllib.request.Request(
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
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"Gemini-native image generation failed: {exc}", file=sys.stderr)
        return 4

    candidates = body.get("candidates") if isinstance(body, dict) else None
    for candidate in candidates or []:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts or []:
            if not isinstance(part, dict):
                continue
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline_data, dict):
                continue
            data = inline_data.get("data")
            if not data:
                continue
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(base64.b64decode(str(data)))
                return 0
            except Exception as exc:
                print(f"Gemini-native image base64 decode failed: {exc}", file=sys.stderr)
                return 4
    print("Gemini-native image generation response had no inline image data.", file=sys.stderr)
    return 4


def _image_attempt_count() -> int:
    raw = _env_first("DEEPGRAPH_PAPERBANANA_IMAGE_ATTEMPTS", "DEEPGRAPH_PAPERBANANA_IMAGE_RETRIES")
    try:
        return max(1, min(5, int(raw or "3")))
    except ValueError:
        return 3


def _post_openai_image_payload_with_urllib(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.0.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
            return body, ""
    except Exception as exc:
        return None, str(exc)


def _curl_binary() -> str | None:
    names = ("curl.exe", "curl") if os.name == "nt" else ("curl", "curl.exe")
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _curl_config_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _post_openai_image_payload_with_curl(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    curl = _curl_binary()
    if not curl:
        return None, "curl executable not found"

    payload_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            payload_path = Path(handle.name)

        config = "\n".join(
            [
                f'url = "{_curl_config_quote(url)}"',
                'request = "POST"',
                f'header = "Authorization: Bearer {_curl_config_quote(api_key)}"',
                'header = "Content-Type: application/json"',
                f'data-binary = "@{_curl_config_quote(payload_path.resolve().as_posix())}"',
                "silent",
                "show-error",
                "max-time = 600",
                "",
            ]
        )
        proc = subprocess.run(
            [curl, "-K", "-"],
            input=config,
            text=True,
            capture_output=True,
            timeout=660,
            check=False,
        )
        if proc.returncode != 0:
            return None, _clip(proc.stderr.strip() or f"exit {proc.returncode}", 400)
        try:
            return json.loads(proc.stdout), ""
        except json.JSONDecodeError as exc:
            preview = _clip(proc.stdout.strip(), 400)
            return None, f"invalid JSON response: {exc}; preview={preview}"
    except Exception as exc:
        return None, str(exc)
    finally:
        if payload_path:
            payload_path.unlink(missing_ok=True)


def _write_openai_image_response(*, output_path: Path, body: dict[str, Any]) -> int:
    rows = body.get("data") if isinstance(body, dict) else None
    item = rows[0] if isinstance(rows, list) and rows else {}
    if not isinstance(item, dict):
        print("Image generation returned no data rows.", file=sys.stderr)
        return 4
    b64_json = item.get("b64_json") or item.get("image_base64")
    if b64_json:
        try:
            output_path.write_bytes(base64.b64decode(str(b64_json)))
            return 0
        except Exception as exc:
            print(f"Image base64 decode failed: {exc}", file=sys.stderr)
            return 4
    url = item.get("url")
    if url:
        try:
            with urllib.request.urlopen(str(url), timeout=600) as response:
                output_path.write_bytes(response.read())
            return 0
        except Exception as exc:
            print(f"Image download failed: {exc}", file=sys.stderr)
            return 4
    print("Image generation response had neither b64_json nor url.", file=sys.stderr)
    return 4


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepGraph -> PaperBanana wrapper")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--spec", required=True, help="JSON spec from DeepGraph")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling PaperBanana")
    args = parser.parse_args()

    _ensure_paperbanana_env()
    _ensure_model_config()

    try:
        spec = json.loads(args.spec)
    except json.JSONDecodeError as exc:
        print(f"Invalid --spec JSON: {exc}", file=sys.stderr)
        return 2

    aspect_ratio = _nearest_supported_ratio((spec.get("figure") or {}).get("aspect_ratio"))
    caption = _build_caption(spec)
    content = _build_content(spec)
    ready, provider = _check_credentials()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "output": str(Path(args.out).resolve()),
                    "aspect_ratio": aspect_ratio,
                    "caption": caption,
                    "provider": provider,
                    "main_model": os.environ.get("MAIN_MODEL_NAME", ""),
                    "image_model": os.environ.get("IMAGE_GEN_MODEL_NAME", ""),
                    "gemini_native_base_url": os.environ.get("GEMINI_NATIVE_BASE_URL", ""),
                    "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
                    "openrouter_base_url": os.environ.get("OPENROUTER_BASE_URL", ""),
                    "content_preview": content[:600],
                    "paperbanana_python": str(PAPERBANANA_PYTHON),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not ready:
        print(
            "PaperBanana is installed, but no image-capable credential is configured. "
            "Set DEEPGRAPH_PAPERBANANA_IMAGE_PROTOCOL=gemini_native plus "
            "DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY and DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL "
            "(or OPENROUTER_API_KEY / GOOGLE_API_KEY) in /home/billion-token/Deepgraph/.env.",
            file=sys.stderr,
        )
        return 3

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not PAPERBANANA_PYTHON.exists() or not PAPERBANANA_ENTRY.exists():
        if provider == "gemini_native":
            return _run_gemini_native_image_generation(
                output_path=output_path,
                prompt=_image_prompt(spec, caption=caption, content=content),
                aspect_ratio=aspect_ratio,
            )
        if provider in {"openai", "openrouter", "openai_compatible_image"}:
            return _run_openai_compatible_image_generation(
                output_path=output_path,
                prompt=_image_prompt(spec, caption=caption, content=content),
            )
        print("PaperBanana is not installed at ~/PaperBanana.", file=sys.stderr)
        return 2

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as handle:
        handle.write(content)
        content_path = Path(handle.name)

    cmd = [
        str(PAPERBANANA_PYTHON),
        str(PAPERBANANA_ENTRY),
        "--content-file",
        str(content_path),
        "--caption",
        caption,
        "--task",
        "diagram",
        "--output",
        str(output_path),
        "--aspect-ratio",
        aspect_ratio,
        "--max-critic-rounds",
        "3",
        "--num-candidates",
        "1",
        "--retrieval-setting",
        "auto",
        "--exp-mode",
        "demo_full",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(PAPERBANANA_ROOT), check=False)
        return int(proc.returncode)
    finally:
        content_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
