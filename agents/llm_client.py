"""Thin LLM client using OpenAI Responses API (tabcode compatible)."""
import json
import httpx
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_MAX_OUTPUT_TOKENS


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             max_tokens: int = None) -> tuple[str, int]:
    """Call LLM via Responses API. Returns (response_text, total_tokens)."""
    if not LLM_API_KEY:
        raise RuntimeError(
            "Missing LLM API key. Set DEEPGRAPH_LLM_API_KEY or OPENAI_API_KEY before running the pipeline."
        )
    max_tokens = max_tokens or LLM_MAX_OUTPUT_TOKENS

    input_items = [
        {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]

    payload = {
        "model": LLM_MODEL,
        "input": input_items,
        "stream": True,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": "high"},
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    response_text = ""
    total_tokens = 0

    with httpx.Client(timeout=300) as client:
        with client.stream("POST", f"{LLM_BASE_URL}/v1/responses",
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


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> tuple[dict | list, int]:
    """Call LLM and parse response as JSON."""
    text, tokens = call_llm(system_prompt, user_prompt, temperature)
    # Extract JSON from response (handle markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text), tokens
