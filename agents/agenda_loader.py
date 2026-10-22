"""Parse and persist agenda definitions without a global active-agenda fallback."""

from __future__ import annotations

from typing import Any, Mapping

from agents.agenda_repository import AgendaRepository
from contracts.agenda import ResearchAgenda
from contracts.base import ensure_dict, ensure_string_list


def parse_agenda(
    payload: Mapping[str, Any],
    *,
    agenda_id: int | None = None,
) -> ResearchAgenda:
    if not isinstance(payload, Mapping):
        raise ValueError("agenda payload must be a mapping")
    agenda = ResearchAgenda(
        agenda_id=agenda_id,
        version=str(payload.get("version") or "v1"),
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or "").strip(),
        focus=ensure_string_list(payload.get("focus") or []),
        prefer=ensure_dict(payload.get("prefer") or {}),
        reject=ensure_dict(payload.get("reject") or {}),
        required_output=ensure_dict(payload.get("required_output") or {}),
        raw_config=dict(payload),
        is_active=bool(payload.get("is_active", True)),
        submitter=str(payload.get("submitter") or "").strip(),
        token_budget=payload.get("token_budget"),
        token_spent=int(payload.get("token_spent") or 0),
        token_reserved=int(payload.get("token_reserved") or 0),
        gpu_hours_budget=float(payload.get("gpu_hours_budget") or 0),
        gpu_hours_spent=float(payload.get("gpu_hours_spent") or 0),
        gpu_hours_reserved=float(payload.get("gpu_hours_reserved") or 0),
        max_concurrency=int(payload.get("max_concurrency") or 1),
        backend_allowlist=ensure_string_list(
            payload.get("backend_allowlist") or ["cpu", "llm"]
        ),
        backlog_policy=str(
            payload.get("backlog_policy") or "explicit_import_only"
        ),
        status=str(payload.get("status") or "active"),
    )
    agenda.validate()
    return agenda


def get_agenda(
    agenda_id: int,
    *,
    repository: AgendaRepository | None = None,
) -> ResearchAgenda | None:
    return (repository or AgendaRepository()).get(agenda_id)
