"""Deterministic user direction to hard-capped ResearchAgenda mapping."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from agents.agenda_loader import parse_agenda
from config import AGENDA_TOKEN_BUDGET_DEFAULT
from contracts.agenda import ResearchAgenda
from contracts.base import ensure_string_list


VALID_GOALS = {"idea_only", "experiment_plan", "verified_evidence"}


class DirectionParseError(ValueError):
    pass


def _tokens(text: str, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text):
        token = token.lower()
        if token not in seen:
            seen.add(token)
            result.append(token)
        if len(result) >= limit:
            break
    return result


def _slug(direction: str, keywords: list[str]) -> str:
    basis = keywords[:4] or _tokens(direction, 4)
    stem = "-".join(
        re.sub(r"[^a-z0-9]+", "-", item.lower()).strip("-") for item in basis
    )
    digest = hashlib.sha256(direction.encode("utf-8")).hexdigest()[:10]
    return f"direction-{stem[:40]}-{digest}" if stem else f"direction-{digest}"


def parse_direction_payload(payload: Mapping[str, Any]) -> ResearchAgenda:
    if not isinstance(payload, Mapping):
        raise DirectionParseError("direction submission must be a mapping")
    direction = str(payload.get("direction") or "").strip()
    submitter = str(payload.get("contact") or payload.get("submitter") or "").strip()
    if not direction:
        raise DirectionParseError("direction is required")
    if not submitter:
        raise DirectionParseError("contact or submitter is required")
    keywords = ensure_string_list(payload.get("keywords") or [])
    focus = keywords or _tokens(direction)
    if not focus:
        raise DirectionParseError("keywords are required for non-ASCII-only scope")
    goal = str(payload.get("goal") or "experiment_plan").strip()
    if goal not in VALID_GOALS:
        raise DirectionParseError(f"unsupported goal: {goal}")
    try:
        token_budget = int(
            payload.get("token_budget")
            if payload.get("token_budget") is not None
            else AGENDA_TOKEN_BUDGET_DEFAULT
        )
    except (TypeError, ValueError) as exc:
        raise DirectionParseError("token_budget must be an integer") from exc
    if token_budget <= 0:
        raise DirectionParseError("token_budget must be a positive hard cap")
    try:
        gpu_hours_budget = float(payload.get("gpu_hours_budget") or 0)
    except (TypeError, ValueError) as exc:
        raise DirectionParseError("gpu_hours_budget must be numeric") from exc
    constraints = payload.get("constraints")
    if constraints is not None and not isinstance(constraints, Mapping):
        raise DirectionParseError("constraints must be a mapping")
    backends = ensure_string_list(
        payload.get("backend_allowlist") or ["cpu", "llm"]
    )
    return parse_agenda(
        {
            "version": "v1",
            "name": _slug(direction, keywords),
            "description": direction,
            "focus": focus,
            "prefer": {"constraints": dict(constraints or {})},
            "required_output": {"goal": goal},
            "submitter": submitter,
            "token_budget": token_budget,
            "gpu_hours_budget": gpu_hours_budget,
            "backend_allowlist": backends,
            "max_concurrency": int(payload.get("max_concurrency") or 1),
            "backlog_policy": "explicit_import_only",
            "source": "direction_intake_v1",
        }
    )


def build_echo(agenda: ResearchAgenda) -> dict[str, Any]:
    return {
        "type": "direction_intake_echo",
        "name": agenda.name,
        "direction": agenda.description,
        "focus": list(agenda.focus),
        "submitter": agenda.submitter,
        "token_budget": agenda.token_budget,
        "gpu_hours_budget": agenda.gpu_hours_budget,
        "backend_allowlist": list(agenda.backend_allowlist),
        "backlog_policy": agenda.backlog_policy,
        "confirmation_required": True,
    }


def parse_direction_yaml(text: str) -> tuple[ResearchAgenda, dict[str, Any]]:
    if not str(text or "").strip():
        raise DirectionParseError("empty direction submission")
    import yaml  # type: ignore

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DirectionParseError(f"invalid YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DirectionParseError("direction submission must be a mapping")
    agenda = parse_direction_payload(payload)
    return agenda, build_echo(agenda)
