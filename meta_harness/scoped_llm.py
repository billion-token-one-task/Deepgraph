"""Fail-closed helpers for ResourceGrant-backed LLM work."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


class ScopedLLMError(PermissionError):
    """Raised before a provider call when LLM authority is incomplete."""


def require_scope(
    llm_scope: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Validate the stable identity required before scoped work is started."""
    required = ("agenda_id", "idea_id", "resource_grant_id", "stage")
    if llm_scope is None:
        raise ScopedLLMError("ResourceGrant-backed LLM scope is required")
    missing = [key for key in required if not llm_scope.get(key)]
    if missing:
        raise ScopedLLMError(
            "ResourceGrant-backed LLM scope is incomplete: " + ",".join(missing)
        )
    for key in ("agenda_id", "idea_id", "resource_grant_id"):
        if int(llm_scope[key]) <= 0:
            raise ScopedLLMError(f"{key} must be positive")
    if not str(llm_scope["stage"]).strip():
        raise ScopedLLMError("stage is required")
    return llm_scope


def require_active_scope(
    llm_scope: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve and verify an active database-backed LLM grant before job work."""
    scope = dict(require_scope(llm_scope))
    from db import database as db

    row = db.fetchone(
        """
        SELECT agenda_id, idea_id, stage, token_cap, backend_allowlist_json
        FROM resource_grants
        WHERE id=? AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (int(scope["resource_grant_id"]),),
    )
    if not row:
        raise ScopedLLMError("active ResourceGrant is required")
    if int(row["agenda_id"]) != int(scope["agenda_id"]):
        raise ScopedLLMError("agenda scope does not match ResourceGrant")
    if int(row["idea_id"]) != int(scope["idea_id"]):
        raise ScopedLLMError("idea scope does not match ResourceGrant")
    if str(row["stage"]) != str(scope["stage"]):
        raise ScopedLLMError("stage scope does not match ResourceGrant")
    allowlist = json.loads(row.get("backend_allowlist_json") or "[]")
    if "llm" not in allowlist:
        raise ScopedLLMError("ResourceGrant does not allow LLM work")
    scope["token_cap"] = int(row["token_cap"])
    return scope


def proposer_json(
    system_prompt: str,
    user_prompt: str,
    *,
    llm_scope: Mapping[str, Any] | None,
    operation: str,
    token_cap: int = 32_000,
) -> tuple[dict | list, int, dict]:
    """Invoke a proposer route only within one active ResourceGrant scope."""
    llm_scope = require_scope(llm_scope)
    if not operation.strip():
        raise ScopedLLMError("LLM operation is required")
    granted_cap = int(llm_scope.get("token_cap") or token_cap)
    requested_cap = min(int(token_cap), granted_cap)
    if requested_cap <= 0:
        raise ScopedLLMError("LLM token cap must be positive")

    from agents.llm_client import (
        call_llm_json_for_role,
        configured_role_prompt_version,
    )

    digest = hashlib.sha256(
        "\n".join(
            (
                str(llm_scope["agenda_id"]),
                str(llm_scope["idea_id"]),
                str(llm_scope["resource_grant_id"]),
                str(llm_scope["stage"]),
                operation,
                system_prompt,
                user_prompt,
            )
        ).encode("utf-8")
    ).hexdigest()
    return call_llm_json_for_role(
        system_prompt,
        user_prompt,
        agenda_id=int(llm_scope["agenda_id"]),
        idea_id=int(llm_scope["idea_id"]),
        role="proposer",
        stage=str(llm_scope["stage"]),
        resource_grant_id=int(llm_scope["resource_grant_id"]),
        operation=operation,
        idempotency_key=f"{operation}:{digest}",
        prompt_version=configured_role_prompt_version("proposer"),
        max_tokens=requested_cap,
    )
