"""Operator-editable, non-secret LLM provider configuration.

The public Flask app is deliberately not allowed to hold provider credentials:
``/api/runtime-config`` was disabled in this codebase precisely because runtime
configuration is an operator concern. This module keeps that boundary while
still letting an authorized operator manage routes from a page.

What is editable here is only the non-secret half of a provider: its name, base
URL, model, model family, protocol and rate limit. The API key is never
accepted, stored, or returned; an entry names the environment variable that
carries it, and the store only reports whether that variable is currently set.

Two further restrictions exist because a provider's ``base_url`` is where every
prompt is sent -- an attacker who can add a provider can exfiltrate everything
the harness reasons about:

* the URL must be HTTPS and its host must be in an operator-approved allowlist;
* the key variable must follow ``DEEPGRAPH_LLM_<NAME>_API_KEY``, so a config
  edit cannot repoint a provider at an unrelated secret already in the
  environment.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
KEY_ENV_PATTERN = re.compile(r"^DEEPGRAPH_LLM_[A-Z0-9_]{1,40}_API_KEY$")
PROTOCOLS = ("chat_completions", "responses")
STORE_VERSION = 1

EDITABLE_FIELDS = (
    "name",
    "base_url",
    "model",
    "model_family",
    "protocol",
    "api_key_env",
    "rpm",
    "enabled",
)


class ProviderConfigError(ValueError):
    """Raised when a submitted provider entry is unusable or unsafe."""


def _host_of(url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise ProviderConfigError("base_url must be an https URL with a host")
    return parsed.hostname.lower()


def validate_entry(
    payload: Mapping[str, Any],
    *,
    allowed_hosts: Iterable[str],
) -> dict[str, Any]:
    """Normalize one submitted entry or refuse it with a specific reason."""
    if not isinstance(payload, Mapping):
        raise ProviderConfigError("provider entry must be an object")
    for forbidden in ("api_key", "apikey", "key", "token", "secret"):
        if payload.get(forbidden):
            raise ProviderConfigError(
                "provider entries never carry a credential; use api_key_env"
            )
    unknown = sorted(set(payload) - set(EDITABLE_FIELDS))
    if unknown:
        raise ProviderConfigError("unsupported fields:" + ",".join(unknown))

    name = str(payload.get("name") or "").strip().lower()
    if not NAME_PATTERN.match(name):
        raise ProviderConfigError(
            "name must be lowercase letters, digits or underscore (2-31 chars)"
        )
    base_url = str(payload.get("base_url") or "").strip().rstrip("/")
    host = _host_of(base_url)
    allowed = {str(value).strip().lower() for value in allowed_hosts if str(value).strip()}
    if not allowed:
        raise ProviderConfigError(
            "no provider host allowlist is configured; an operator must set "
            "DEEPGRAPH_LLM_PROVIDER_HOST_ALLOWLIST before routes can be added"
        )
    if host not in allowed:
        raise ProviderConfigError(f"host is not in the operator allowlist:{host}")

    model = str(payload.get("model") or "").strip()
    if not model:
        raise ProviderConfigError("model is required")
    model_family = str(payload.get("model_family") or "").strip().lower()
    if not model_family:
        raise ProviderConfigError(
            "model_family is required; evaluator independence is judged on it"
        )
    protocol = str(payload.get("protocol") or "chat_completions").strip().lower()
    if protocol not in PROTOCOLS:
        raise ProviderConfigError("protocol must be one of " + ",".join(PROTOCOLS))
    api_key_env = str(payload.get("api_key_env") or "").strip()
    if not KEY_ENV_PATTERN.match(api_key_env):
        raise ProviderConfigError(
            "api_key_env must look like DEEPGRAPH_LLM_<NAME>_API_KEY"
        )
    try:
        rpm = int(payload.get("rpm") or 0)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigError("rpm must be an integer") from exc
    if rpm < 0:
        raise ProviderConfigError("rpm cannot be negative")

    return {
        "name": name,
        "base_url": base_url,
        "model": model,
        "model_family": model_family,
        "protocol": protocol,
        "api_key_env": api_key_env,
        "rpm": rpm,
        "enabled": bool(payload.get("enabled", True)),
    }


def load_store(path: str | Path) -> dict[str, Any]:
    """Read the managed store. A missing or unreadable file is an empty store."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": STORE_VERSION, "providers": []}
    if not isinstance(loaded, dict):
        return {"version": STORE_VERSION, "providers": []}
    providers = loaded.get("providers")
    if not isinstance(providers, list):
        providers = []
    return {
        "version": int(loaded.get("version") or STORE_VERSION),
        "providers": [entry for entry in providers if isinstance(entry, dict)],
        "updated_at": loaded.get("updated_at"),
        "updated_by": loaded.get("updated_by"),
    }


def save_store(
    path: str | Path,
    entries: list[dict[str, Any]],
    *,
    actor: str,
) -> dict[str, Any]:
    """Write the store atomically with restrictive permissions."""
    if not str(actor or "").strip():
        raise ProviderConfigError("an actor is required for an auditable change")
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ProviderConfigError("provider names must be unique")
    store = {
        "version": STORE_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": str(actor).strip()[:80],
        "providers": entries,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o640)
    temporary.replace(target)
    return store


def upsert(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    allowed_hosts: Iterable[str],
    actor: str,
) -> dict[str, Any]:
    entry = validate_entry(payload, allowed_hosts=allowed_hosts)
    entries = [
        item
        for item in load_store(path)["providers"]
        if str(item.get("name")) != entry["name"]
    ]
    entries.append(entry)
    entries.sort(key=lambda item: str(item.get("name")))
    return save_store(path, entries, actor=actor)


def remove(path: str | Path, name: str, *, actor: str) -> bool:
    target = str(name or "").strip().lower()
    entries = load_store(path)["providers"]
    remaining = [item for item in entries if str(item.get("name")) != target]
    if len(remaining) == len(entries):
        return False
    save_store(path, remaining, actor=actor)
    return True


def readiness(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe each entry for the UI. The key itself is never read out.

    ``key_present`` says whether the named variable currently holds a value in
    this process. Nothing else about the credential is exposed -- not its
    length, not a prefix, not a hash.
    """
    described = []
    for entry in entries:
        key_env = str(entry.get("api_key_env") or "")
        # A runtime entry is read back from the live provider pool, so its
        # credential demonstrably resolved at startup; there is no variable to
        # re-check and its absence would misreport a working route as broken.
        runtime = str(entry.get("source") or "") == "runtime"
        present = runtime or bool(str(os.environ.get(key_env, "") or "").strip())
        described.append(
            {
                "name": entry.get("name"),
                "base_url": entry.get("base_url"),
                "model": entry.get("model"),
                "model_family": entry.get("model_family"),
                "protocol": entry.get("protocol"),
                "rpm": entry.get("rpm", 0),
                "enabled": bool(entry.get("enabled", True)),
                "api_key_env": key_env,
                "key_present": present,
                "ready": bool(present and entry.get("enabled", True)),
                "source": entry.get("source", "managed"),
            }
        )
    return sorted(described, key=lambda item: str(item.get("name")))


def effective_pool() -> list[dict[str, Any]]:
    """The routes this process would actually use, with no credential material.

    The legacy provider slots (minimax, tabcode, secondary) are built from
    environment variables and never appear in the managed store, so a report
    based on the store alone would understate what is running -- and, worse,
    would tell an operator that no independent evaluator exists when one does.
    Reading the live pool back is the only answer that cannot drift.
    """
    try:
        from agents import llm_client

        llm_client._init_providers()  # noqa: SLF001 - reads config, no network
        pool = list(llm_client._providers)  # noqa: SLF001
    except Exception:
        return []
    return [
        {
            "name": str(entry.get("name") or ""),
            "base_url": str(entry.get("base_url") or ""),
            "model": str(entry.get("model") or ""),
            "model_family": str(entry.get("model_family") or ""),
            "protocol": str(entry.get("protocol") or ""),
            "rpm": int(entry.get("rpm") or 0),
            "enabled": True,
            "api_key_env": "",
            "source": "runtime",
        }
        for entry in pool
    ]


def independence_report(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Can a Frontier evaluator be independent of the proposer right now?

    Independence is judged on provider identity and model family, so the answer
    is simply whether two ready routes differ in family.
    """
    ready = [entry for entry in readiness(entries) if entry["ready"]]
    families = sorted({str(entry["model_family"]) for entry in ready})
    return {
        "ready_routes": len(ready),
        "distinct_model_families": families,
        "independent_evaluator_possible": len(families) >= 2,
    }
