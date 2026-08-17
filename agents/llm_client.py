"""Multi-provider LLM client with load balancing and per-provider rate limiting."""
import hashlib
import json
import os
import threading
import time
import httpx
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_CONNECT_TIMEOUT_SECONDS,
    LLM_EXTRA_PROVIDERS_JSON,
    LLM_PROVIDERS,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL,
    LLM_PROMPT_CACHE_ENABLED,
    LLM_PROMPT_CACHE_KEY,
    LLM_PROMPT_CACHE_RETENTION,
    LLM_PROTOCOL,
    LLM_REASONING_EFFORT,
    LLM_RPM,
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_ROLE_ROUTES,
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
_prompt_cache_unsupported = set()
_prompt_cache_lock = threading.Lock()

_rate_limiters = {}       # name -> _RateLimiter
_provider_cooldown = {}   # name -> resume_timestamp (epoch)

# A byte-level tokenizer cannot produce more prompt tokens than UTF-8 bytes.
# Explicitly bounded calls also reserve a deliberately oversized allowance for
# two chat-message wrappers and never ask a provider for more than 3000 output
# tokens. The tagged repair's measured prompt is about 2.8k bytes, leaving more
# than 5k tokens between prompt bytes and provider output before the 8k cap.
LLM_PROMPT_FRAMING_TOKEN_CEILING = 1500
LLM_EXPLICIT_PROVIDER_OUTPUT_TOKEN_CEILING = 3000


class LLMProviderUnavailableError(RuntimeError):
    """Raised when all configured providers are temporarily unavailable."""


def _bounded_role_token_caps(
    system_prompt: str,
    user_prompt: str,
    *,
    requested_output_cap: int,
    remaining_tokens: int,
    total_token_cap: int | None,
) -> tuple[int, int]:
    """Return (aggregate reservation, provider output) before any LLM call."""
    if requested_output_cap <= 0:
        raise ValueError("max_tokens must be a positive hard cap")
    if remaining_tokens <= 0:
        raise PermissionError("ResourceGrant token budget is exhausted")
    if total_token_cap is None:
        approx_prompt_tokens = (len(system_prompt) + len(user_prompt)) // 3
        aggregate_cap = min(
            requested_output_cap + approx_prompt_tokens,
            remaining_tokens,
        )
        return aggregate_cap, min(requested_output_cap, aggregate_cap)
    if int(total_token_cap) <= 0:
        raise ValueError("total_token_cap must be a positive hard cap")

    aggregate_cap = min(int(total_token_cap), remaining_tokens)
    prompt_token_ceiling = (
        len(system_prompt.encode("utf-8"))
        + len(user_prompt.encode("utf-8"))
        + LLM_PROMPT_FRAMING_TOKEN_CEILING
    )
    provider_output_cap = min(
        requested_output_cap,
        LLM_EXPLICIT_PROVIDER_OUTPUT_TOKEN_CEILING,
        aggregate_cap - prompt_token_ceiling,
    )
    if provider_output_cap <= 0:
        raise ValueError("prompt cannot fit inside total_token_cap")
    return aggregate_cap, provider_output_cap


def _resolve_route_reference(value: str) -> str:
    ref = str(value or "").strip()
    if not ref:
        return ""
    if ref.startswith("env:"):
        return str(os.environ.get(ref[4:], "") or "").strip()
    # Secret-manager references require an injected resolver. Treating their
    # labels as provider/model names would be a silent downgrade.
    if ":" in ref:
        return ""
    return ref


def configured_role_route_policy(role: str) -> dict[str, dict]:
    """Resolve non-secret provider/model policy without selecting a fallback."""
    if role not in {"proposer", "evaluator", "reviewer"}:
        raise ValueError("invalid LLM role")
    routes: dict[str, dict] = {}
    for item in LLM_ROLE_ROUTES.get(role, []):
        if not isinstance(item, dict):
            continue
        provider_name = _resolve_route_reference(item.get("provider_ref", ""))
        if not provider_name:
            continue
        if provider_name in routes:
            raise LLMProviderUnavailableError(
                f"duplicate configured {role} provider route:{provider_name}"
            )
        routes[provider_name] = {
            "model": _resolve_route_reference(item.get("model_ref", "")),
            "model_family": _resolve_route_reference(
                item.get("model_family_ref", "")
            )
            or str(item.get("model_family") or "").strip(),
            "prompt_version": str(item.get("prompt_version") or "").strip(),
        }
    if not routes:
        raise LLMProviderUnavailableError(
            f"no resolved {role} provider route; manual review required"
        )
    return routes


def configured_role_prompt_version(role: str) -> str:
    if role not in {"proposer", "evaluator", "reviewer"}:
        raise ValueError("invalid LLM role")
    # Prompt provenance is non-secret policy. Resolve it from the declared
    # route entries even when provider references are intentionally absent in
    # isolated policy tests; execution still fails closed in the route client.
    declared = LLM_ROLE_ROUTES.get(role, [])
    versions = {
        str(item.get("prompt_version") or "").strip()
        for item in declared
        if isinstance(item, dict)
    }
    if not versions or not (versions - {""}):
        versions = {
            str(policy.get("prompt_version") or "").strip()
            for policy in configured_role_route_policy(role).values()
        }
    versions.discard("")
    if len(versions) != 1:
        raise LLMProviderUnavailableError(
            f"{role} routes require one explicit prompt version"
        )
    return versions.pop()


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


def _safe_prompt_cache_piece(value: str, max_len: int) -> str:
    cleaned = []
    prev_dash = False
    for ch in str(value or "").strip().lower():
        if ch.isascii() and (ch.isalnum() or ch in "._:"):
            cleaned.append(ch)
            prev_dash = False
        elif not prev_dash:
            cleaned.append("-")
            prev_dash = True
        if len(cleaned) >= max_len:
            break
    return "".join(cleaned).strip("-")[:max_len]


def _prompt_cache_key(provider: dict, system_prompt: str) -> str:
    base = _safe_prompt_cache_piece(LLM_PROMPT_CACHE_KEY, 40)
    if not base:
        return ""
    model_piece = _safe_prompt_cache_piece(provider.get("model", ""), 18)
    prompt_digest = hashlib.sha256((system_prompt or "").encode("utf-8")).hexdigest()[:12]
    suffix = f"{model_piece}:{prompt_digest}" if model_piece else prompt_digest
    budget = max(1, 64 - len(suffix) - 1)
    return f"{base[:budget]}:{suffix}"[:64]


def _prompt_cache_disabled(provider: dict) -> bool:
    with _prompt_cache_lock:
        return provider.get("name") in _prompt_cache_unsupported


def _mark_prompt_cache_unsupported(provider: dict) -> None:
    with _prompt_cache_lock:
        name = provider.get("name")
        if not name or name in _prompt_cache_unsupported:
            return
        _prompt_cache_unsupported.add(name)
    print(f"[LLM] {name} does not accept prompt cache request fields; retrying without them", flush=True)


def _apply_prompt_cache_options(payload: dict, provider: dict, system_prompt: str) -> None:
    if not LLM_PROMPT_CACHE_ENABLED or _prompt_cache_disabled(provider):
        return
    cache_key = _prompt_cache_key(provider, system_prompt)
    if not cache_key:
        return
    payload["prompt_cache_key"] = cache_key
    retention = str(LLM_PROMPT_CACHE_RETENTION or "").strip()
    if retention.lower() not in {"", "0", "false", "none", "off", "disabled"}:
        payload["prompt_cache_retention"] = retention


def _without_prompt_cache_options(payload: dict) -> dict | None:
    if "prompt_cache_key" not in payload and "prompt_cache_retention" not in payload:
        return None
    stripped = dict(payload)
    stripped.pop("prompt_cache_key", None)
    stripped.pop("prompt_cache_retention", None)
    return stripped


def _is_http_400(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response is not None and exc.response.status_code == 400


def _extra_openai_providers() -> list[dict]:
    """Parse optional extra OpenAI-compatible provider routes from JSON config."""
    raw = (LLM_EXTRA_PROVIDERS_JSON or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[LLM] Ignoring invalid DEEPGRAPH_LLM_EXTRA_PROVIDERS_JSON: {exc}", flush=True)
        return []
    if isinstance(payload, dict):
        payload = payload.get("providers") or payload.get("routes") or []
    if not isinstance(payload, list):
        print("[LLM] Ignoring DEEPGRAPH_LLM_EXTRA_PROVIDERS_JSON: expected a JSON list", flush=True)
        return []

    def _as_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_bool(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    providers: list[dict] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict) or item.get("enabled") is False:
            continue
        api_key = str(item.get("api_key") or "").strip()
        base_url = str(item.get("base_url") or "").strip().rstrip("/")
        if not api_key or not base_url:
            continue
        providers.append(
            {
                "name": str(item.get("name") or f"extra_{idx}").strip() or f"extra_{idx}",
                "base_url": base_url,
                "api_key": api_key,
                "model": str(item.get("model") or LLM_MODEL).strip() or LLM_MODEL,
                "protocol": str(item.get("protocol") or LLM_PROTOCOL).strip().lower() or LLM_PROTOCOL,
                "rpm": _as_int(item.get("rpm"), 0),
                "stream_chat_completions": _as_bool(item.get("stream_chat_completions"), False),
                "chat_endpoint": str(item.get("chat_endpoint") or "/chat/completions"),
                "extra_headers": item.get("extra_headers") if isinstance(item.get("extra_headers"), dict) else {},
            }
        )
    return providers


class DeclaredProviderError(RuntimeError):
    """Raised when a declared provider entry is unsafe or unusable."""


def _declared_providers() -> list[dict]:
    """Providers declared in deepgraph.toml as ``[[llm.providers]]``.

    Only non-secret fields live in TOML. The API key is named by
    ``api_key_env`` and read from the environment, so adding a provider is one
    TOML block plus one line in the environment file, instead of a JSON blob.

    Two things deliberately fail rather than degrade:

    * a literal ``api_key`` in TOML is refused -- TOML is tracked in Git;
    * a referenced environment variable that is unset skips the provider with a
      named reason, so a missing key never silently falls back to another
      provider.
    """
    providers: list[dict] = []
    for index, entry in enumerate(LLM_PROVIDERS, start=1):
        if entry.get("enabled") is False:
            continue
        name = str(entry.get("name") or f"declared_{index}").strip()
        if entry.get("api_key"):
            raise DeclaredProviderError(
                f"provider {name} declares a literal api_key in TOML; "
                "use api_key_env with an environment variable name instead"
            )
        key_env = str(entry.get("api_key_env") or "").strip()
        # A private gateway hostname is site-specific even though it is not a
        # credential, and this file is in Git. base_url_env keeps the endpoint
        # in the operator's environment for exactly the reason api_key_env
        # keeps the key there; a literal base_url stays supported for public
        # provider endpoints that are safe to publish.
        base_url_env = str(entry.get("base_url_env") or "").strip()
        if base_url_env:
            base_url = str(os.environ.get(base_url_env, "") or "").strip().rstrip("/")
            if not base_url:
                print(
                    f"[LLM] Skipping declared provider {name}: "
                    f"{base_url_env} is not set",
                    flush=True,
                )
                continue
        else:
            base_url = str(entry.get("base_url") or "").strip().rstrip("/")
        model = str(entry.get("model") or "").strip()
        if not key_env or not base_url or not model:
            print(
                f"[LLM] Skipping declared provider {name}: "
                "api_key_env, base_url (or base_url_env) and model are all required",
                flush=True,
            )
            continue
        api_key = str(os.environ.get(key_env, "") or "").strip()
        if not api_key:
            print(
                f"[LLM] Skipping declared provider {name}: {key_env} is not set",
                flush=True,
            )
            continue
        providers.append(
            {
                "name": name,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "model_family": str(entry.get("model_family") or "").strip(),
                "protocol": str(entry.get("protocol") or LLM_PROTOCOL).strip().lower(),
                "rpm": int(entry.get("rpm") or 0),
                "stream_chat_completions": bool(
                    entry.get("stream_chat_completions", False)
                ),
                "chat_endpoint": str(entry.get("chat_endpoint") or "/chat/completions"),
                "extra_headers": (
                    dict(entry["extra_headers"])
                    if isinstance(entry.get("extra_headers"), dict)
                    else {}
                ),
            }
        )
    return providers


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

    for declared in _declared_providers():
        _providers.append(declared)

    for extra in _extra_openai_providers():
        _append_openai_provider(
            name=extra["name"],
            base_url=extra["base_url"],
            api_key=extra["api_key"],
            model=extra["model"],
            protocol=extra["protocol"],
            rpm=extra["rpm"],
        )
        _providers[-1]["stream_chat_completions"] = extra["stream_chat_completions"]
        _providers[-1]["chat_endpoint"] = extra["chat_endpoint"]
        _providers[-1]["extra_headers"] = extra["extra_headers"]

    seen_names: set[str] = set()
    for idx, provider in enumerate(_providers, start=1):
        base_name = str(provider.get("name") or f"provider_{idx}")
        name = base_name
        suffix = 2
        while name in seen_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        provider["name"] = name
        declared_family = str(provider.get("model_family") or "").strip()
        provider["model_family"] = declared_family or (
            str(provider.get("model") or "")
            .lower()
            .split("/", 1)[-1]
            .split("-", 1)[0]
        )
        seen_names.add(name)

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


def get_provider_models() -> list[dict]:
    """Return configured provider names and models for explicit reviewer routing."""
    _init_providers()
    return [
        {
            "name": p.get("name"),
            "model": p.get("model"),
            "base_url": str(p.get("base_url") or "")[:80],
            "protocol": p.get("protocol"),
            "model_family": p.get("model_family"),
        }
        for p in _providers
    ]


def _should_omit_token_limit(provider: dict) -> bool:
    """Return True for routes whose gateway rejects max token limit fields."""
    if provider.get("omit_token_limit"):
        return True
    model = str(provider.get("model") or "").strip().lower()
    model_name = model.rsplit("/", 1)[-1]
    return model_name == "gpt-5.5" or model_name.startswith("gpt-5.5-")


def _usage_cost_usd(usage: dict | None) -> float | None:
    """Read explicit provider billing data without estimating a price."""
    if not isinstance(usage, dict):
        return None
    candidates = (
        usage.get("cost_usd"),
        usage.get("total_cost_usd"),
        usage.get("total_cost"),
        usage.get("cost"),
        (usage.get("billing") or {}).get("cost_usd")
        if isinstance(usage.get("billing"), dict)
        else None,
    )
    for value in candidates:
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _call_provider(
    provider: dict,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    *,
    strict_single_request: bool = False,
) -> tuple[str, int, int, int, float | None]:
    """Return text, total/cached/input tokens, and explicit provider cost."""
    protocol = provider.get("protocol", "responses")
    if protocol == "chat_completions":
        return _call_chat_completions(
            provider,
            system_prompt,
            user_prompt,
            max_tokens,
            strict_single_request=strict_single_request,
        )
    return _call_responses_api(
        provider,
        system_prompt,
        user_prompt,
        max_tokens,
        strict_single_request=strict_single_request,
    )


def _call_chat_completions(
    provider: dict,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    *,
    strict_single_request: bool = False,
) -> tuple[str, int, int, int, float | None]:
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
    }
    if strict_single_request and (not max_tokens or _should_omit_token_limit(provider)):
        raise LLMProviderUnavailableError(
            "bounded route requires an enforceable provider output limit"
        )
    if max_tokens and not _should_omit_token_limit(provider):
        payload["max_tokens"] = max_tokens
    if not strict_single_request:
        _apply_prompt_cache_options(payload, provider, system_prompt)

    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(provider.get("extra_headers", {}))

    response_text = ""
    total_tokens = 0
    cached_tokens = 0
    input_tokens = 0
    cost_usd = None
    endpoint = provider.get("chat_endpoint", "/chat/completions")
    chunk_count = 0
    all_lines = []

    def _reset_response_state() -> None:
        nonlocal response_text, total_tokens, cached_tokens, input_tokens, cost_usd, chunk_count, all_lines
        response_text = ""
        total_tokens = 0
        cached_tokens = 0
        input_tokens = 0
        cost_usd = None
        chunk_count = 0
        all_lines = []

    def _consume_body(body: dict) -> None:
        nonlocal response_text, total_tokens, cached_tokens, input_tokens, cost_usd
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            reasoning = message.get("reasoning_content") or ""
            content = message.get("content")
            if isinstance(content, str):
                response_text = content.strip() or reasoning
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if text:
                            parts.append(text)
                joined = "".join(parts)
                response_text = joined.strip() or reasoning
            else:
                response_text = reasoning
        usage = body.get("usage") or {}
        total_tokens = usage.get("total_tokens", 0)
        input_tokens = usage.get("prompt_tokens", 0)
        ptd = usage.get("prompt_tokens_details") or {}
        cached_tokens = ptd.get("cached_tokens", 0)
        cost_usd = _usage_cost_usd(usage)

    def _consume_stream(resp) -> None:
        nonlocal response_text, total_tokens, cached_tokens, input_tokens, cost_usd, chunk_count, all_lines
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
                cost_usd = _usage_cost_usd(usage)

    def _send_once(request_payload: dict) -> None:
        _reset_response_state()
        with httpx.Client(timeout=_http_timeout()) as client:
            if not stream_chat:
                resp = client.post(f"{provider['base_url']}{endpoint}", json=request_payload, headers=headers)
                resp.raise_for_status()
                _consume_body(resp.json())
            else:
                with client.stream("POST", f"{provider['base_url']}{endpoint}",
                                   json=request_payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        resp.read()
                    resp.raise_for_status()
                    _consume_stream(resp)

    try:
        _send_once(payload)
    except httpx.HTTPStatusError as exc:
        if strict_single_request:
            raise
        fallback_payload = _without_prompt_cache_options(payload)
        if not (_is_http_400(exc) and fallback_payload is not None):
            raise
        _send_once(fallback_payload)
        _mark_prompt_cache_unsupported(provider)

    if not response_text:
        non_data = [l for l in all_lines if l.strip()][:5]
        print(f"[LLM] WARNING: {provider['name']} empty after {chunk_count} chunks. "
              f"Non-data lines: {non_data}", flush=True)
        if chunk_count <= 2 and total_tokens > 0:
            print(f"[LLM] {provider['name']}: empty response despite {total_tokens} tokens reported", flush=True)

    return response_text, total_tokens, cached_tokens, input_tokens, cost_usd


def _extract_responses_output_text(response: dict) -> str:
    """Best-effort text extraction from a Responses API final response object."""
    if not isinstance(response, dict):
        return ""
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    pieces: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_text = item.get("text")
        if isinstance(item_text, str) and item_text:
            pieces.append(item_text)
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text:
                pieces.append(text)
            nested = content.get("content")
            if isinstance(nested, str) and nested:
                pieces.append(nested)
    return "".join(pieces).strip()


def _call_responses_api(
    provider: dict,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    *,
    strict_single_request: bool = False,
) -> tuple[str, int, int, int, float | None]:
    """Call via OpenAI Responses API (for tabcode etc)."""
    input_items = [
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]

    payload = {
        "model": provider["model"],
        "instructions": system_prompt,
        "input": input_items,
        "stream": True,
        "reasoning": {"effort": LLM_REASONING_EFFORT},
    }
    if strict_single_request and (not max_tokens or _should_omit_token_limit(provider)):
        raise LLMProviderUnavailableError(
            "bounded route requires an enforceable provider output limit"
        )
    if max_tokens and not _should_omit_token_limit(provider):
        payload["max_output_tokens"] = max_tokens
    if not strict_single_request:
        _apply_prompt_cache_options(payload, provider, system_prompt)

    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(provider.get("extra_headers", {}))

    response_text = ""
    total_tokens = 0
    cached_tokens = 0
    input_tokens = 0
    cost_usd = None

    def _reset_response_state() -> None:
        nonlocal response_text, total_tokens, cached_tokens, input_tokens, cost_usd
        response_text = ""
        total_tokens = 0
        cached_tokens = 0
        input_tokens = 0
        cost_usd = None

    def _stream_response(request_payload: dict) -> None:
        nonlocal response_text, total_tokens, cached_tokens, input_tokens, cost_usd
        _reset_response_state()
        with httpx.Client(timeout=_http_timeout()) as client:
            with client.stream("POST", f"{provider['base_url']}/responses",
                               json=request_payload, headers=headers) as resp:
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
                        response_obj = event.get("response", {})
                        if not response_text:
                            response_text = _extract_responses_output_text(response_obj)
                        usage = response_obj.get("usage", {})
                        total_tokens = usage.get("total_tokens", 0)
                        input_tokens = usage.get("input_tokens", 0)
                        # OpenAI cache: usage.input_tokens_details.cached_tokens
                        itd = usage.get("input_tokens_details") or {}
                        cached_tokens = itd.get("cached_tokens", 0)
                        cost_usd = _usage_cost_usd(usage)

    def _add_fallback(candidates: list[tuple[str, dict]], label: str, candidate: dict | None) -> None:
        if not candidate or candidate == payload:
            return
        for _, existing in candidates:
            if existing == candidate:
                return
        candidates.append((label, candidate))

    try:
        _stream_response(payload)
    except httpx.HTTPStatusError as exc:
        if strict_single_request:
            raise
        if not _is_http_400(exc):
            raise
        candidates: list[tuple[str, dict]] = []
        _add_fallback(candidates, "prompt_cache", _without_prompt_cache_options(payload))

        compat_payload = dict(payload)
        compat_payload.pop("max_output_tokens", None)
        compat_payload.pop("reasoning", None)
        _add_fallback(candidates, "compat", compat_payload)
        _add_fallback(candidates, "compat_prompt_cache", _without_prompt_cache_options(compat_payload))

        last_exc = exc
        for label, candidate in candidates:
            try:
                _stream_response(candidate)
            except httpx.HTTPStatusError as retry_exc:
                if not _is_http_400(retry_exc):
                    raise
                last_exc = retry_exc
                continue
            if "prompt_cache" in label:
                _mark_prompt_cache_unsupported(provider)
            return response_text, total_tokens, cached_tokens, input_tokens, cost_usd
        raise last_exc

    return response_text, total_tokens, cached_tokens, input_tokens, cost_usd


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
                text, tokens, cached_toks, input_toks, _cost_usd = _call_provider(
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


def call_llm_with_provider(
    system_prompt: str,
    user_prompt: str,
    *,
    provider_name: str | None = None,
    provider_index: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = None,
) -> tuple[str, int, dict]:
    """Call one selected provider, used when reviewer roles should be model-routed.

    An explicitly requested route fails closed when unavailable. Temperature is
    accepted for API parity; provider routes currently use deterministic
    settings.
    """
    del temperature
    max_tokens = max_tokens or LLM_MAX_OUTPUT_TOKENS
    _init_providers()
    if not _providers:
        raise LLMProviderUnavailableError("No LLM providers configured.")

    provider = None
    explicit_route = provider_name is not None or provider_index is not None
    if provider_name:
        for candidate in _providers:
            if candidate.get("name") == provider_name:
                provider = candidate
                break
    if provider is None and provider_index is not None and 0 <= provider_index < len(_providers):
        provider = _providers[provider_index]
    if provider is None and explicit_route:
        raise LLMProviderUnavailableError(
            "requested provider route is unavailable; manual review required"
        )
    if provider is None:
        provider = _providers[0]

    name = provider["name"]
    limiter = _rate_limiters.get(name)
    if limiter:
        limiter.wait()
    with _provider_lock:
        _provider_stats[name]["in_flight"] = _provider_stats[name].get("in_flight", 0) + 1
    start = time.time()
    try:
        text, tokens, cached_toks, input_toks, _cost_usd = _call_provider(
            provider, system_prompt, user_prompt, max_tokens
        )
        latency = time.time() - start
        with _provider_lock:
            stats = _provider_stats[name]
            stats["calls"] += 1
            stats["tokens"] += tokens
            stats["total_latency"] += latency
            stats["cached_tokens"] += cached_toks
            stats["input_tokens"] += input_toks
        if not text or len(text.strip()) < 10:
            raise RuntimeError(f"{name} returned empty response")
        return text, tokens, {
            "name": provider.get("name"),
            "model": provider.get("model"),
            "protocol": provider.get("protocol"),
            "model_family": provider.get("model_family") or provider.get("model"),
        }
    except Exception:
        latency = time.time() - start
        with _provider_lock:
            stats = _provider_stats[name]
            stats["calls"] += 1
            stats["errors"] += 1
            stats["total_latency"] += latency
        raise
    finally:
        _release_provider(name)


def call_llm_for_role(
    system_prompt: str,
    user_prompt: str,
    *,
    agenda_id: int,
    idea_id: int,
    role: str,
    stage: str,
    resource_grant_id: int,
    operation: str,
    idempotency_key: str,
    prompt_version: str,
    allowed_provider_names: list[str] | None = None,
    proposer_route: dict | None = None,
    max_tokens: int | None = None,
    total_token_cap: int | None = None,
    max_route_attempts: int | None = None,
) -> tuple[str, int, dict]:
    """Resource-granted role route with provider/model/token observation.

    This is the meta-harness-v1 entry point. Legacy ``call_llm`` remains for
    pre-agenda ingestion only and must not be used by resource-consuming
    proposer/evaluator/reviewer jobs.
    """
    from contracts.meta_harness import ResourceGrant
    from meta_harness.grant_usage import GrantUsageLedger
    from meta_harness.llm_routing import (
        LLMExecutionFailure,
        LLMRouter,
        ProviderRoute,
        RouteRequest,
        RouteUsage,
    )
    from meta_harness.repository import MetaHarnessRepository
    from db import database as db

    requested_output_cap = int(max_tokens or LLM_MAX_OUTPUT_TOKENS)
    grant_row = db.fetchone(
        """
        SELECT * FROM resource_grants
        WHERE id=? AND agenda_id=? AND idea_id=?
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (resource_grant_id, agenda_id, idea_id),
    )
    if not grant_row:
        raise PermissionError("active scoped ResourceGrant is required for LLM role")
    # Trim the request to what this grant still has, rather than sending the
    # caller's own default and letting the sub-reservation refuse it. A small
    # short grant is the whole point of the bounded pilot path: without this a
    # single caller asking for the provider maximum makes any grant smaller
    # than that maximum unusable, no matter how little work it needs.
    remaining_tokens = GrantUsageLedger(resource_grant_id).remaining(
        agenda_id=agenda_id
    )
    # Explicit total caps separate the aggregate ledger reservation from the
    # provider output cap. The latter is reduced before the request is sent,
    # rather than discovering an input+output overrun after spend occurred.
    token_cap, provider_output_cap = _bounded_role_token_caps(
        system_prompt,
        user_prompt,
        requested_output_cap=requested_output_cap,
        remaining_tokens=remaining_tokens,
        total_token_cap=total_token_cap,
    )
    grant = ResourceGrant(
        agenda_id=int(grant_row["agenda_id"]),
        idea_id=int(grant_row["idea_id"]),
        decision_packet_id=int(grant_row["decision_packet_id"]),
        stage=str(grant_row["stage"]),
        token_cap=int(grant_row["token_cap"]),
        gpu_class=str(grant_row.get("gpu_class") or "none"),
        max_gpu_hours=float(grant_row.get("max_gpu_hours") or 0),
        backend_allowlist=json.loads(grant_row.get("backend_allowlist_json") or "[]"),
        artifact_requirements=json.loads(
            grant_row.get("artifact_requirements_json") or "[]"
        ),
        expires_at=str(grant_row["expires_at"]),
        grant_reason=str(grant_row["grant_reason"]),
        idempotency_key=str(grant_row["idempotency_key"]),
        status=str(grant_row["status"]),
        grant_id=int(grant_row["id"]),
        reservation_id=int(grant_row["reservation_id"]),
    )
    role_policy = configured_role_route_policy(role)
    configured_names = set(role_policy)
    allowed = (
        set(allowed_provider_names)
        if allowed_provider_names is not None
        else configured_names
    )
    if not allowed or not allowed.issubset(configured_names):
        raise LLMProviderUnavailableError(
            "requested provider set is outside configured role policy"
        )
    _init_providers()
    provider_map = {
        str(provider["name"]): provider
        for provider in _providers
        if provider.get("name") in allowed
    }
    if not allowed or set(provider_map) != allowed:
        raise LLMProviderUnavailableError(
            "one or more explicitly allowed provider routes are unavailable"
        )
    for name, provider in provider_map.items():
        expected_model = str(role_policy[name].get("model") or "")
        expected_family = str(role_policy[name].get("model_family") or "")
        expected_prompt = str(role_policy[name].get("prompt_version") or "")
        actual_family = str(
            provider.get("model_family") or provider.get("model") or ""
        )
        if expected_model and expected_model != str(provider.get("model") or ""):
            raise LLMProviderUnavailableError(
                f"configured {role} model does not match provider:{name}"
            )
        if expected_family and expected_family != actual_family:
            raise LLMProviderUnavailableError(
                f"configured {role} model family does not match provider:{name}"
            )
        if expected_prompt and expected_prompt != prompt_version:
            raise LLMProviderUnavailableError(
                f"configured {role} prompt version mismatch:{name}"
            )
    routes = [
        ProviderRoute(
            route_id=name,
            provider=name,
            model=str(provider["model"]),
            model_family=str(provider.get("model_family") or provider["model"]),
            prompt_version=prompt_version,
            timeout_seconds=int(LLM_REQUEST_TIMEOUT_SECONDS),
            transient_retries=int(LLM_TRANSIENT_RETRIES),
            transient_cooldown_seconds=int(LLM_TRANSIENT_COOLDOWN_SECONDS),
        )
        for name, provider in provider_map.items()
    ]
    proposer_contract = None
    if proposer_route:
        proposer_contract = ProviderRoute(
            route_id=str(proposer_route.get("route_id") or proposer_route.get("provider")),
            provider=str(proposer_route.get("provider") or ""),
            model=str(proposer_route.get("model") or ""),
            model_family=str(proposer_route.get("model_family") or ""),
            prompt_version=str(proposer_route.get("prompt_version") or ""),
            timeout_seconds=int(
                proposer_route.get("timeout_seconds") or LLM_REQUEST_TIMEOUT_SECONDS
            ),
        )
        proposer_contract.validate()
    repository = MetaHarnessRepository()
    router = LLMRouter(
        {name: list(routes) for name in ("proposer", "evaluator", "reviewer")},
        ledger=GrantUsageLedger(resource_grant_id),
        observation_sink=repository.save_route_observation,
        cooldown_store=repository,
    )
    request_contract = RouteRequest(
        agenda_id=agenda_id,
        idea_id=idea_id,
        role=role,
        stage=stage,
        resource_grant_id=resource_grant_id,
        token_cap=token_cap,
        operation=operation,
        idempotency_key=idempotency_key,
        proposer_route=proposer_contract,
        max_attempts=max_route_attempts,
    )

    def _execute(route, _request):
        provider = provider_map[route.route_id]
        limiter = _rate_limiters.get(route.route_id)
        if limiter:
            limiter.wait()
        start = time.time()
        try:
            text, tokens, cached_tokens, input_tokens, cost_usd = _call_provider(
                provider,
                system_prompt,
                user_prompt,
                provider_output_cap,
                strict_single_request=total_token_cap is not None,
            )
            output_tokens = max(0, int(tokens or 0) - int(input_tokens or 0))
            usage = RouteUsage(
                int(input_tokens or 0),
                output_tokens,
                cost_usd,
            )
            if not text or len(text.strip()) < 10:
                raise LLMExecutionFailure(
                    "provider returned an empty response",
                    category="provider_error",
                    usage=usage,
                )
            with _provider_lock:
                stats = _provider_stats[route.route_id]
                stats["calls"] += 1
                stats["tokens"] += int(tokens or 0)
                stats["input_tokens"] += int(input_tokens or 0)
                stats["cached_tokens"] += int(cached_tokens or 0)
                stats["total_latency"] += time.time() - start
            return text, usage
        except Exception as exc:
            category = (
                "auth"
                if is_llm_auth_error(exc)
                else "transient"
                if is_llm_transient_provider_error(exc)
                else "provider_error"
            )
            with _provider_lock:
                stats = _provider_stats[route.route_id]
                stats["calls"] += 1
                stats["errors"] += 1
                stats["total_latency"] += time.time() - start
            if isinstance(exc, LLMExecutionFailure):
                raise
            raise LLMExecutionFailure(
                str(exc),
                category=category,
            ) from exc

    result = router.invoke(request_contract, grant=grant, executor=_execute)
    return (
        str(result.output),
        result.usage.total_tokens,
        {
            "provider": result.route.provider,
            "model": result.route.model,
            "model_family": result.route.model_family,
            "prompt_version": result.route.prompt_version,
            "attempts": result.attempts,
        },
    )


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


def call_llm_json_for_role(
    system_prompt: str,
    user_prompt: str,
    **route_kwargs,
) -> tuple[dict | list, int, dict]:
    """Parse a resource-granted role-routed response as JSON."""
    text, tokens, route = call_llm_for_role(
        system_prompt,
        user_prompt,
        **route_kwargs,
    )
    parsed, how = parse_llm_json_text(text)
    if how not in ("direct", "empty"):
        print(
            f"[LLM_JSON] Role-routed response parsed via {how} "
            f"({len(text)} chars)",
            flush=True,
        )
    return parsed, tokens, route


def call_llm_json_with_provider(
    system_prompt: str,
    user_prompt: str,
    *,
    provider_name: str | None = None,
    provider_index: int | None = None,
    temperature: float = 0.0,
) -> tuple[dict | list, int, dict]:
    """Call a selected provider and parse the response as JSON."""
    text, tokens, provider = call_llm_with_provider(
        system_prompt,
        user_prompt,
        provider_name=provider_name,
        provider_index=provider_index,
        temperature=temperature,
    )
    if not text or not str(text).strip():
        print(f"[LLM_JSON] WARNING: empty provider-routed response ({tokens} tokens)", flush=True)
        return {}, tokens, provider
    try:
        parsed, how = parse_llm_json_text(text)
        if how not in ("direct", "empty"):
            print(f"[LLM_JSON] Parsed via {how} ({len(text)} chars)", flush=True)
        return parsed, tokens, provider
    except json.JSONDecodeError as e:
        preview = str(text).replace("\n", " ")[:320]
        print(f"[LLM_JSON] Provider-routed parse failed: {e}; preview: {preview}...", flush=True)
        raise
