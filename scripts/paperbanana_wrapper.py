#!/usr/bin/env python3
"""DeepGraph bridge for one-shot PaperBanana diagram generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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


def _check_credentials() -> tuple[bool, str]:
    if os.environ.get("GEMINI_NATIVE_API_KEY") and os.environ.get("GEMINI_NATIVE_BASE_URL"):
        return True, "gemini_native"
    if os.environ.get("OPENROUTER_API_KEY"):
        return True, "openrouter"
    if os.environ.get("GOOGLE_API_KEY"):
        return True, "google"
    if os.environ.get("OPENAI_API_KEY") and "gpt-image" in os.environ.get("IMAGE_GEN_MODEL_NAME", ""):
        return True, "openai"
    return False, "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepGraph -> PaperBanana wrapper")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--spec", required=True, help="JSON spec from DeepGraph")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling PaperBanana")
    args = parser.parse_args()

    if not PAPERBANANA_PYTHON.exists() or not PAPERBANANA_ENTRY.exists():
        print("PaperBanana is not installed at ~/PaperBanana.", file=sys.stderr)
        return 2

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
