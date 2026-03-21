"""Multi-provider LLM client with round-robin load balancing."""
import json
import threading
import time
import httpx
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_MAX_OUTPUT_TOKENS

# Provider pool: each entry is (base_url, api_key, model, name)
_providers = []
_provider_idx = 0
_provider_lock = threading.Lock()
_provider_stats = {}  # name -> {calls, tokens, errors, avg_latency}


def _init_providers():
    """Build provider pool from config + env vars."""
    global _providers
    if _providers:
        return

    # Provider 1: tabcode (from config)
    if LLM_API_KEY:
        _providers.append({
            "name": "tabcode",
            "base_url": LLM_BASE_URL,
            "api_key": LLM_API_KEY,
            "model": LLM_MODEL,
        })

    # Provider 2: Kimi (Chat Completions API, needs special User-Agent)
    import os
    kimi_key = os.environ.get("KIMI_API_KEY", "sk-kimi-A9HyGmdMtUiKL3oaSHsePPzlh28ckzOjlcWKPjszzcDtLiDau1LTluN2TgCG4Q6s")
    if kimi_key:
        _providers.append({
            "name": "kimi",
            "base_url": "https://api.kimi.com/coding/v1",
            "api_key": kimi_key,
            "model": "kimi-latest",
            "protocol": "chat_completions",  # uses /chat/completions not /responses
            "extra_headers": {"User-Agent": "claude-code/1.0"},
        })

    # Provider 3: MiniMax M2.7 (Chat Completions API)
    minimax_key = os.environ.get("MINIMAX_API_KEY", "sk-cp-gLnIPlTb9FtsyyldavYKa9T10i6dKbyTfN0lFh2F0jZc2hUEc9G5c4SjShTwEmofYlz938D9oUYFB_0nTj2bgtRR2UDFbetq3QzGv7KvSYPnB4LVs2NW9ys")
    if minimax_key:
        _providers.append({
            "name": "minimax",
            "base_url": "https://api.minimaxi.com/v1/text",
            "api_key": minimax_key,
            "model": "M2.7",
            "protocol": "chat_completions",
            "chat_endpoint": "/chatcompletion_v2",  # custom endpoint path
        })

    # Init stats
    for p in _providers:
        _provider_stats[p["name"]] = {"calls": 0, "tokens": 0, "errors": 0, "total_latency": 0}

    if not _providers:
        raise RuntimeError("No LLM providers configured. Set DEEPGRAPH_LLM_API_KEY or OPENAI_API_KEY.")


def _next_provider() -> dict:
    """Round-robin provider selection, skipping recently-errored ones."""
    global _provider_idx
    _init_providers()

    with _provider_lock:
        # Try each provider, prefer ones with fewer errors
        best = None
        for i in range(len(_providers)):
            idx = (_provider_idx + i) % len(_providers)
            p = _providers[idx]
            stats = _provider_stats[p["name"]]
            # Skip if error rate > 50% and has been called at least 3 times
            if stats["calls"] >= 3 and stats["errors"] / stats["calls"] > 0.5:
                continue
            best = p
            _provider_idx = (idx + 1) % len(_providers)
            break

        if best is None:
            # All providers erroring, just use first one
            best = _providers[0]
            _provider_idx = 1 % len(_providers)

        return best


def get_provider_stats() -> dict:
    """Return stats for all providers."""
    _init_providers()
    result = {}
    for p in _providers:
        name = p["name"]
        s = _provider_stats[name]
        result[name] = {
            "calls": s["calls"],
            "tokens": s["tokens"],
            "errors": s["errors"],
            "avg_latency": round(s["total_latency"] / max(s["calls"], 1), 1),
            "model": p["model"],
            "base_url": p["base_url"][:40],
        }
    return result


def _call_provider(provider: dict, system_prompt: str, user_prompt: str,
                   max_tokens: int) -> tuple[str, int]:
    """Call a specific provider. Routes to correct protocol."""
    protocol = provider.get("protocol", "responses")
    if protocol == "chat_completions":
        return _call_chat_completions(provider, system_prompt, user_prompt, max_tokens)
    return _call_responses_api(provider, system_prompt, user_prompt, max_tokens)


def _call_chat_completions(provider: dict, system_prompt: str, user_prompt: str,
                           max_tokens: int) -> tuple[str, int]:
    """Call via OpenAI Chat Completions API (for Kimi etc)."""
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(provider.get("extra_headers", {}))

    response_text = ""
    total_tokens = 0

    endpoint = provider.get("chat_endpoint", "/chat/completions")
    with httpx.Client(timeout=300) as client:
        with client.stream("POST", f"{provider['base_url']}{endpoint}",
                           json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        response_text += content
                usage = chunk.get("usage")
                if usage:
                    total_tokens = usage.get("total_tokens", 0)

    return response_text, total_tokens


def _call_responses_api(provider: dict, system_prompt: str, user_prompt: str,
                        max_tokens: int) -> tuple[str, int]:
    """Call via OpenAI Responses API (for tabcode etc)."""
    input_items = [
        {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]

    payload = {
        "model": provider["model"],
        "input": input_items,
        "stream": True,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": "high"},
    }

    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(provider.get("extra_headers", {}))

    response_text = ""
    total_tokens = 0

    with httpx.Client(timeout=300) as client:
        with client.stream("POST", f"{provider['base_url']}/responses",
                           json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
                if event_type == "response.output_text.delta":
                    response_text += event.get("delta", "")
                elif event_type == "response.completed":
                    usage = event.get("response", {}).get("usage", {})
                    total_tokens = usage.get("total_tokens", 0)

    return response_text, total_tokens


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             max_tokens: int = None) -> tuple[str, int]:
    """Call LLM with automatic provider selection and failover."""
    max_tokens = max_tokens or LLM_MAX_OUTPUT_TOKENS
    _init_providers()

    last_error = None
    tried = set()

    for attempt in range(len(_providers)):
        provider = _next_provider()
        if provider["name"] in tried:
            continue
        tried.add(provider["name"])

        start = time.time()
        try:
            text, tokens = _call_provider(provider, system_prompt, user_prompt, max_tokens)
            latency = time.time() - start

            stats = _provider_stats[provider["name"]]
            stats["calls"] += 1
            stats["tokens"] += tokens
            stats["total_latency"] += latency

            return text, tokens

        except Exception as e:
            latency = time.time() - start
            stats = _provider_stats[provider["name"]]
            stats["calls"] += 1
            stats["errors"] += 1
            stats["total_latency"] += latency
            last_error = e
            continue

    raise RuntimeError(f"All {len(_providers)} providers failed. Last error: {last_error}")


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> tuple[dict | list, int]:
    """Call LLM and parse response as JSON. Handles markdown blocks and extra text."""
    import re
    text, tokens = call_llm(system_prompt, user_prompt, temperature)
    text = text.strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[1:end]).strip()

    # Try direct parse first
    try:
        return json.loads(text), tokens
    except json.JSONDecodeError:
        pass

    # Try to find JSON object/array in the text
    for match in re.finditer(r'(\{[\s\S]*\}|\[[\s\S]*\])', text):
        try:
            return json.loads(match.group()), tokens
        except json.JSONDecodeError:
            continue

    # Last resort: if text is empty, return empty dict
    if not text:
        return {}, tokens

    raise json.JSONDecodeError(f"No valid JSON found in LLM response ({len(text)} chars)", text, 0)
