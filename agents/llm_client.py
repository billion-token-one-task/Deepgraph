"""Multi-provider LLM client with load balancing and per-provider rate limiting."""
import json
import threading
import time
import httpx
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_CONNECT_TIMEOUT_SECONDS,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL,
    LLM_PROTOCOL,
    LLM_REASONING_EFFORT,
    LLM_RPM,
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_SECONDARY_API_KEY,
    LLM_SECONDARY_BASE_URL,
    LLM_SECONDARY_ENABLED,
    LLM_SECONDARY_MODEL,
    LLM_SECONDARY_PROTOCOL,
    LLM_SECONDARY_RPM,
    LLM_TRANSIENT_BACKOFF_SECONDS,
    LLM_TRANSIENT_COOLDOWN_SECONDS,
    LLM_TRANSIENT_RETRIES,
    LLM_USE_TABCODE,
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    MINIMAX_MODEL,
    MINIMAX_RPM,
)

_providers = []
_provider_idx = 0
_provider_lock = threading.Lock()
_provider_stats = {}

_rate_limiters = {}       # name -> _RateLimiter
_provider_cooldown = {}   # name -> resume_timestamp (epoch)


class LLMProviderUnavailableError(RuntimeError):
    """Raised when all configured providers are temporarily unavailable."""


class _RateLimiter:
    """Sliding-window rate limiter (thread-safe)."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm  # min seconds between calls
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        """Block until a call is allowed."""
        with self._lock:
            now = time.time()
            earliest = self._last_call + self.interval
            if now < earliest:
                time.sleep(earliest - now)
            self._last_call = time.time()


def _http_timeout() -> httpx.Timeout:
    return httpx.Timeout(float(LLM_REQUEST_TIMEOUT_SECONDS), connect=float(LLM_CONNECT_TIMEOUT_SECONDS))


def _init_providers():
    """Build provider pool from config + env vars."""
    global _providers
    if _providers:
        return

    # Provider: MiniMax (default primary — Chat Completions API)
    if MINIMAX_API_KEY:
        _providers.append({
            "name": "minimax",
            "base_url": MINIMAX_BASE_URL,
            "api_key": MINIMAX_API_KEY,
            "model": MINIMAX_MODEL,
            "protocol": "chat_completions",
            "rpm": MINIMAX_RPM,
            "stream_chat_completions": True,
        })

    def _append_openai_provider(
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        protocol: str,
        rpm: int = 0,
    ) -> None:
        if not api_key:
            return
        _providers.append({
            "name": name,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "protocol": protocol,
            # OpenAI-compatible proxies used here return usage cleanly on
            # non-stream chat completions, which keeps token accounting intact.
            "stream_chat_completions": False,
            "rpm": rpm,
        })

    # Optional: primary OpenAI-compatible proxy (Responses API or Chat Completions)
    if LLM_USE_TABCODE and LLM_API_KEY:
        _append_openai_provider(
            name="tabcode",
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            protocol=LLM_PROTOCOL,
            rpm=LLM_RPM,
        )

    # Optional: secondary OpenAI-compatible proxy for concurrent fan-out.
    if LLM_SECONDARY_ENABLED and LLM_SECONDARY_API_KEY:
        _append_openai_provider(
            name="secondary",
            base_url=LLM_SECONDARY_BASE_URL,
            api_key=LLM_SECONDARY_API_KEY,
            model=LLM_SECONDARY_MODEL,
            protocol=LLM_SECONDARY_PROTOCOL,
            rpm=LLM_SECONDARY_RPM,
        )

    # Init stats + rate limiters
    for p in _providers:
        _provider_stats[p["name"]] = {
            "calls": 0, "tokens": 0, "errors": 0, "total_latency": 0, "in_flight": 0,
            "cached_tokens": 0, "input_tokens": 0,
        }
        rpm = p.get("rpm", 0)
        if rpm > 0:
            _rate_limiters[p["name"]] = _RateLimiter(rpm)
            print(f"[LLM] Rate limiter for {p['name']}: {rpm} RPM ({60.0/rpm:.1f}s interval)", flush=True)

    if not _providers:
        raise RuntimeError(
            "No LLM providers configured. Set MINIMAX_API_KEY (default path), "
            "or set DEEPGRAPH_LLM_USE_TABCODE=1 plus DEEPGRAPH_LLM_API_KEY / OPENAI_API_KEY, "
            "or enable DEEPGRAPH_LLM_SECONDARY_ENABLED with secondary OpenAI-compatible credentials."
        )


def _next_provider() -> dict:
    """Atomically select a provider AND increment its in_flight counter.
    
    Strategy: find the fastest provider and send it most work.
    Slow providers only get 1 in-flight at a time (probe / trickle).
    Fast provider (<15s avg) gets up to 20 in-flight.
    """
    global _provider_idx
    _init_providers()

    FAST_THRESHOLD = 15.0
    FAST_MAX_INFLIGHT = 20
    SLOW_MAX_INFLIGHT = 1

    with _provider_lock:
        now = time.time()
        candidates = []
        for p in _providers:
            name = p["name"]
            # Skip providers in cooldown (quota exhausted)
            cooldown_until = _provider_cooldown.get(name, 0)
            if now < cooldown_until:
                remaining = int(cooldown_until - now)
                if remaining % 60 == 0:  # log once per minute
                    print(f"[LLM] {name} in cooldown, {remaining}s remaining", flush=True)
                continue
            elif cooldown_until > 0:
                print(f"[LLM] {name} cooldown expired, re-enabling", flush=True)
                _provider_cooldown[name] = 0

            stats = _provider_stats[name]
            if stats["calls"] >= 3 and stats["errors"] / stats["calls"] > 0.5:
                continue
            avg_lat = stats["total_latency"] / max(stats["calls"], 1)
            in_flight = stats.get("in_flight", 0)
            completed = stats["calls"]
            candidates.append((p, avg_lat, in_flight, completed))

        if not candidates:
            cooldown_remaining = []
            for p in _providers:
                remaining = _provider_cooldown.get(p["name"], 0) - now
                if remaining > 0:
                    cooldown_remaining.append(int(remaining))
            if cooldown_remaining:
                raise LLMProviderUnavailableError(
                    f"All LLM providers are cooling down; next retry in {min(cooldown_remaining)}s"
                )
            chosen = _providers[0]
            _provider_stats[chosen["name"]]["in_flight"] = _provider_stats[chosen["name"]].get("in_flight", 0) + 1
            return chosen

        # Classify providers
        fast = []
        slow = []
        unknown = []
        for c in candidates:
            p, avg_lat, in_flight, completed = c
            if completed == 0:
                unknown.append(c)
            elif avg_lat <= FAST_THRESHOLD:
                fast.append(c)
            else:
                slow.append(c)

        chosen = None

        # Priority 1: fast providers with room
        fast_avail = [c for c in fast if c[2] < FAST_MAX_INFLIGHT]
        if fast_avail:
            chosen = min(fast_avail, key=lambda c: c[2])[0]

        # Priority 2: unknown providers that need probing (1 at a time)
        if chosen is None:
            probe_avail = [c for c in unknown if c[2] == 0]
            if probe_avail:
                chosen = probe_avail[0][0]

        # Priority 3: slow providers with room (trickle)
        if chosen is None:
            slow_avail = [c for c in slow if c[2] < SLOW_MAX_INFLIGHT]
            if slow_avail:
                chosen = min(slow_avail, key=lambda c: c[1])[0]

        # Priority 4: everything full — pick least loaded overall
        if chosen is None:
            chosen = min(candidates, key=lambda c: c[2])[0]

        _provider_stats[chosen["name"]]["in_flight"] = _provider_stats[chosen["name"]].get("in_flight", 0) + 1
        return chosen


def _release_provider(name: str):
    """Decrement in_flight for a provider (thread-safe)."""
    with _provider_lock:
        stats = _provider_stats[name]
        stats["in_flight"] = max(0, stats.get("in_flight", 0) - 1)


def get_provider_stats() -> dict:
    """Return stats for all providers."""
    _init_providers()
    result = {}
    now = time.time()
    for p in _providers:
        name = p["name"]
        s = _provider_stats[name]
        total_input = s.get("input_tokens", 0)
        cached = s.get("cached_tokens", 0)
        cache_rate = round(cached / max(total_input, 1) * 100, 1)
        cooldown_until = _provider_cooldown.get(name, 0)
        cooldown_remaining = max(0, int(cooldown_until - now))
        result[name] = {
            "calls": s["calls"],
            "tokens": s["tokens"],
            "errors": s["errors"],
            "avg_latency": round(s["total_latency"] / max(s["calls"], 1), 1),
            "in_flight": s.get("in_flight", 0),
            "model": p["model"],
            "base_url": p["base_url"][:40],
            "cached_tokens": cached,
            "input_tokens": total_input,
            "cache_hit_rate": f"{cache_rate}%",
            "cooldown_remaining": f"{cooldown_remaining}s" if cooldown_remaining > 0 else "active",
        }
    return result


def _call_provider(provider: dict, system_prompt: str, user_prompt: str,
                   max_tokens: int) -> tuple[str, int, int, int]:
    """Call a specific provider. Returns (text, total_tokens, cached_tokens, input_tokens)."""
    protocol = provider.get("protocol", "responses")
    if protocol == "chat_completions":
        return _call_chat_completions(provider, system_prompt, user_prompt, max_tokens)
    return _call_responses_api(provider, system_prompt, user_prompt, max_tokens)


def _call_chat_completions(provider: dict, system_prompt: str, user_prompt: str,
                           max_tokens: int) -> tuple[str, int, int, int]:
    """Call via OpenAI Chat Completions API (for Kimi etc).
    Returns (text, total_tokens, cached_tokens, input_tokens)."""
    stream_chat = provider.get("stream_chat_completions", True)
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": stream_chat,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(provider.get("extra_headers", {}))

    response_text = ""
    total_tokens = 0
    cached_tokens = 0
    input_tokens = 0

    endpoint = provider.get("chat_endpoint", "/chat/completions")
    chunk_count = 0
    all_lines = []
    with httpx.Client(timeout=_http_timeout()) as client:
        if not stream_chat:
            resp = client.post(f"{provider['base_url']}{endpoint}", json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
            choices = body.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content") or ""
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("content") or ""
                            if text:
                                parts.append(text)
                    response_text = "".join(parts)
            usage = body.get("usage") or {}
            total_tokens = usage.get("total_tokens", 0)
            input_tokens = usage.get("prompt_tokens", 0)
            ptd = usage.get("prompt_tokens_details") or {}
            cached_tokens = ptd.get("cached_tokens", 0)
        else:
            with client.stream("POST", f"{provider['base_url']}{endpoint}",
                               json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                    elif line.startswith("data:"):
                        data_str = line[5:]
                    else:
                        all_lines.append(line)
                        continue
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    chunk_count += 1

                    # MiniMax embeds rate limit errors in SSE body (HTTP 200)
                    if chunk.get("type") == "error":
                        err_info = chunk.get("error", {})
                        err_type = err_info.get("type", "")
                        err_msg = err_info.get("message", "")
                        if "rate_limit" in err_type or "usage limit" in err_msg.lower():
                            print(f"[LLM] {provider['name']} SSE rate limit: {err_msg}", flush=True)
                            raise httpx.HTTPStatusError(
                                f"SSE rate limit: {err_msg}",
                                request=httpx.Request("POST", provider["base_url"]),
                                response=httpx.Response(429),
                            )
                        print(f"[LLM] {provider['name']} SSE error: {err_type}: {err_msg}", flush=True)
                        raise RuntimeError(f"{provider['name']} API error: {err_type}: {err_msg}")

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        # MiniMax (and similar) stream thinking in reasoning_content; final answer in content.
                        # Concatenate so JSON in either stream is preserved for downstream parsing.
                        reasoning = delta.get("reasoning_content") or ""
                        content = delta.get("content") or ""
                        piece = reasoning + content
                        if piece:
                            response_text += piece
                    usage = chunk.get("usage")
                    if usage:
                        total_tokens = usage.get("total_tokens", 0)
                        input_tokens = usage.get("prompt_tokens", 0)
                        # MiniMax cache info: usage.prompt_tokens_details.cached_tokens
                        ptd = usage.get("prompt_tokens_details") or {}
                        cached_tokens = ptd.get("cached_tokens", 0)

    if not response_text:
        non_data = [l for l in all_lines if l.strip()][:5]
        print(f"[LLM] WARNING: {provider['name']} empty after {chunk_count} chunks. "
              f"Non-data lines: {non_data}", flush=True)
        if chunk_count <= 2 and total_tokens > 0:
            print(f"[LLM] {provider['name']}: empty response despite {total_tokens} tokens reported", flush=True)

    return response_text, total_tokens, cached_tokens, input_tokens


def _call_responses_api(provider: dict, system_prompt: str, user_prompt: str,
                        max_tokens: int) -> tuple[str, int, int, int]:
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
        "reasoning": {"effort": LLM_REASONING_EFFORT},
    }

    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(provider.get("extra_headers", {}))

    response_text = ""
    total_tokens = 0
    cached_tokens = 0
    input_tokens = 0

    with httpx.Client(timeout=_http_timeout()) as client:
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
                    input_tokens = usage.get("input_tokens", 0)
                    # OpenAI cache: usage.input_tokens_details.cached_tokens
                    itd = usage.get("input_tokens_details") or {}
                    cached_tokens = itd.get("cached_tokens", 0)

    return response_text, total_tokens, cached_tokens, input_tokens


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is an HTTP 429 rate limit error."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    return False


def is_llm_auth_error(exc: Exception) -> bool:
    """Check if an exception points to bad credentials or authorization."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403):
        return True
    msg = str(exc).lower()
    auth_markers = (
        "401 unauthorized",
        "403 forbidden",
        "invalid api key",
        "authentication",
        "unauthorized",
        "forbidden",
    )
    return any(marker in msg for marker in auth_markers)


def is_llm_provider_unavailable_error(exc: Exception) -> bool:
    """Check if an exception means every provider is temporarily unavailable."""
    return isinstance(exc, LLMProviderUnavailableError)


def is_llm_transient_provider_error(exc: Exception) -> bool:
    """Check if a provider is temporarily unhealthy (timeout / 5xx gateway issues)."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (500, 502, 503, 504):
        return True
    msg = str(exc).lower()
    markers = (
        "connection refused",
        "connecterror",
        "all connection attempts failed",
        "temporary failure in name resolution",
        "timed out",
        "timeout",
        "gateway time-out",
        "bad gateway",
        "service unavailable",
        "server error '500",
        "server error '502",
        "server error '503",
        "server error '504",
    )
    return any(marker in msg for marker in markers)


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             max_tokens: int = None) -> tuple[str, int]:
    """Call LLM with automatic provider selection, rate limiting, and failover."""
    max_tokens = max_tokens or LLM_MAX_OUTPUT_TOKENS
    _init_providers()

    last_error = None
    tried = set()
    MAX_429_RETRIES = 3

    for attempt in range(len(_providers)):
        try:
            provider = _next_provider()
        except LLMProviderUnavailableError as e:
            last_error = e
            break
        if provider["name"] in tried:
            _release_provider(provider["name"])
            continue
        tried.add(provider["name"])

        stats = _provider_stats[provider["name"]]

        for retry in range(MAX_429_RETRIES + 1):
            limiter = _rate_limiters.get(provider["name"])
            if limiter:
                limiter.wait()

            start = time.time()
            try:
                text, tokens, cached_toks, input_toks = _call_provider(
                    provider, system_prompt, user_prompt, max_tokens)
                latency = time.time() - start

                if not text or len(text.strip()) < 10:
                    with _provider_lock:
                        stats["calls"] += 1
                        stats["errors"] += 1
                        stats["total_latency"] += latency
                    _release_provider(provider["name"])
                    print(f"[LLM] WARNING: {provider['name']} returned empty/short response, trying next provider", flush=True)
                    last_error = RuntimeError(f"{provider['name']} returned empty response")
                    break  # try next provider

                with _provider_lock:
                    stats["calls"] += 1
                    stats["tokens"] += tokens
                    stats["total_latency"] += latency
                    stats["cached_tokens"] += cached_toks
                    stats["input_tokens"] += input_toks
                _release_provider(provider["name"])
                return text, tokens

            except Exception as e:
                latency = time.time() - start
                if is_llm_auth_error(e):
                    cooldown_secs = 3600  # 1 hour cooldown for bad credentials / auth failures
                    status_code = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else "auth"
                    with _provider_lock:
                        _provider_cooldown[provider["name"]] = max(
                            _provider_cooldown.get(provider["name"], 0),
                            time.time() + cooldown_secs,
                        )
                        stats["calls"] += 1
                        stats["errors"] += 1
                        stats["total_latency"] += latency
                    _release_provider(provider["name"])
                    print(
                        f"[LLM] {provider['name']} auth failed ({status_code}), "
                        f"cooldown {cooldown_secs // 60}min",
                        flush=True,
                    )
                    last_error = e
                    break
                if _is_rate_limit_error(e):
                    err_msg = str(e)
                    is_quota = "usage limit" in err_msg.lower() or "2056" in err_msg
                    if is_quota:
                        cooldown_secs = 600  # 10 min cooldown for quota exhaustion
                        with _provider_lock:
                            _provider_cooldown[provider["name"]] = time.time() + cooldown_secs
                            stats["calls"] += 1
                            stats["errors"] += 1
                            stats["total_latency"] += latency
                        _release_provider(provider["name"])
                        print(f"[LLM] {provider['name']} quota exhausted (5h window), "
                              f"cooldown {cooldown_secs//60}min", flush=True)
                        last_error = e
                        break  # try next provider
                    elif retry < MAX_429_RETRIES:
                        backoff = (2 ** retry) * 5  # 5s, 10s, 20s
                        print(f"[LLM] 429 rate limit from {provider['name']}, "
                              f"retry {retry+1}/{MAX_429_RETRIES} after {backoff}s", flush=True)
                        with _provider_lock:
                            stats["errors"] += 1
                        time.sleep(backoff)
                        continue  # retry same provider

                if is_llm_transient_provider_error(e):
                    if retry < max(0, LLM_TRANSIENT_RETRIES):
                        backoff = max(1, LLM_TRANSIENT_BACKOFF_SECONDS) * (2 ** retry)
                        with _provider_lock:
                            stats["errors"] += 1
                        print(
                            f"[LLM] transient failure from {provider['name']}, "
                            f"retry {retry+1}/{LLM_TRANSIENT_RETRIES} after {backoff}s: {e}",
                            flush=True,
                        )
                        time.sleep(backoff)
                        continue
                    cooldown_secs = max(1, LLM_TRANSIENT_COOLDOWN_SECONDS)
                    with _provider_lock:
                        _provider_cooldown[provider["name"]] = max(
                            _provider_cooldown.get(provider["name"], 0),
                            time.time() + cooldown_secs,
                        )
                        stats["calls"] += 1
                        stats["errors"] += 1
                        stats["total_latency"] += latency
                    _release_provider(provider["name"])
                    print(
                        f"[LLM] {provider['name']} transient failure, cooldown {cooldown_secs}s: {e}",
                        flush=True,
                    )
                    last_error = e
                    break

                with _provider_lock:
                    stats["calls"] += 1
                    stats["errors"] += 1
                    stats["total_latency"] += latency
                _release_provider(provider["name"])
                last_error = e
                break  # try next provider

    if isinstance(last_error, LLMProviderUnavailableError):
        raise last_error
    raise RuntimeError(f"All {len(_providers)} providers failed. Last error: {last_error}")


def _first_balanced_json_slice(text: str, start: int) -> str | None:
    """Slice from start to matching top-level } or ]; respects JSON string rules."""
    if start < 0 or start >= len(text):
        return None
    op = text[start]
    if op not in "{[":
        return None
    cl = "}" if op == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == op:
            depth += 1
        elif c == cl:
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def _normalize_jsonish(s: str) -> str:
    """Fix common LLM JSON quirks before json.loads."""
    import re

    t = s.strip()
    if t.startswith("\ufeff"):
        t = t[1:].lstrip()
    # Smart quotes → ASCII (structural noise from Word/LaTeX copy-paste)
    t = t.translate(
        str.maketrans(
            {
                "\u201c": '"',
                "\u201d": '"',
                "\u2018": "'",
                "\u2019": "'",
            }
        )
    )
    # Python literals in pseudo-JSON
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null", t)
    return t


def _json_try_load(s: str) -> dict | list | None:
    """Try strict parse; then strip trailing commas LLMs often emit."""
    import re

    s = _normalize_jsonish(s)
    for candidate in (s,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    fixed = re.sub(r",(\s*})", r"\1", s)
    fixed = re.sub(r",(\s*])", r"\1", fixed)
    if fixed != s:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    return None


def parse_llm_json_text(text: str) -> tuple[dict | list, str]:
    """Extract JSON object/array from arbitrary LLM output. Returns (parsed, method_label)."""
    import re

    raw = text.strip()
    if not raw:
        return {}, "empty"

    # Strip thinking blocks (same pattern as paradigm_agent / validation_loop)
    t = re.sub(r"<thinking>[\s\S]*?</thinking>", "", raw, flags=re.I).strip()
    # Some providers stream "think" segments as ```think ... ```
    t = re.sub(r"```\s*think\s*[\s\S]*?```", "", t, flags=re.I).strip()

    # Explicit fenced blocks — try each ``` ... ``` body
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", t, re.I):
        body = m.group(1).strip()
        got = _json_try_load(body)
        if got is not None:
            return got, "markdown_fence"

    # Single opening fence without closer
    if t.startswith("```"):
        lines = t.split("\n")
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        t = "\n".join(lines[1:end]).strip()

    # Direct parse
    got = _json_try_load(t)
    if got is not None:
        return got, "direct"

    # Greedy {...} often overshoots; try every '{' position with brace matching
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        chunk = _first_balanced_json_slice(t, i)
        if not chunk:
            continue
        got = _json_try_load(chunk)
        if got is not None:
            return got, f"balanced_object@{i}"

    for i, ch in enumerate(t):
        if ch != "[":
            continue
        chunk = _first_balanced_json_slice(t, i)
        if not chunk:
            continue
        got = _json_try_load(chunk)
        if got is not None:
            return got, f"balanced_array@{i}"

    # Legacy greedy regex (last resort)
    for match in re.finditer(r"(\{[\s\S]*\}|\[[\s\S]*\])", t):
        got = _json_try_load(match.group(1))
        if got is not None:
            return got, "regex_greedy"

    raise json.JSONDecodeError(f"No valid JSON found ({len(raw)} chars)", raw, 0)


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> tuple[dict | list, int]:
    """Call LLM and parse response as JSON. Handles markdown blocks, thinking tags, and extra text."""
    text, tokens = call_llm(system_prompt, user_prompt, temperature)
    if not text or not str(text).strip():
        print(f"[LLM_JSON] WARNING: empty LLM response ({tokens} tokens)", flush=True)
        return {}, tokens
    try:
        parsed, how = parse_llm_json_text(text)
        if how not in ("direct", "empty"):
            print(f"[LLM_JSON] Parsed via {how} ({len(text)} chars)", flush=True)
        return parsed, tokens
    except json.JSONDecodeError as e:
        preview = str(text).replace("\n", " ")[:320]
        print(f"[LLM_JSON] Parse failed: {e}; preview: {preview}...", flush=True)
        raise
