#!/usr/bin/env python3
"""Write EvoScientist config from DeepGraph .env values."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402


def main() -> int:
    if not cfg.LLM_API_KEY or not cfg.LLM_BASE_URL:
        print("Missing DEEPGRAPH_LLM_API_KEY or DEEPGRAPH_LLM_BASE_URL; EvoScientist config not written.", flush=True)
        return 1

    config_dir = Path.home() / ".config" / "evoscientist"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    use_responses = "true" if cfg.LLM_PROTOCOL == "responses" else "false"
    lines = [
        'provider: "custom-openai"',
        f'model: {json.dumps(cfg.LLM_MODEL)}',
        f'custom_openai_api_key: {json.dumps(cfg.LLM_API_KEY)}',
        f'custom_openai_base_url: {json.dumps(cfg.LLM_BASE_URL.rstrip("/"))}',
        f'openai_api_key: {json.dumps(cfg.LLM_API_KEY)}',
        f'use_responses_api: {json.dumps(use_responses)}',
    ]
    if cfg.LLM_REASONING_EFFORT and use_responses == "true":
        lines.append(f'reasoning_effort: {json.dumps(cfg.LLM_REASONING_EFFORT)}')
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote EvoScientist config to {config_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
