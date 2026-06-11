"""Agenda loader: parse YAML/JSON/dict input into ResearchAgenda + persistence.

Issue #9: configurable research agenda layer.

Public API:
- parse_agenda(payload, *, agenda_id=None) -> ResearchAgenda
- load_agenda_from_file(path) -> ResearchAgenda
- save_agenda(agenda) -> int                       # insert; returns agenda_id
- update_agenda(agenda_id, agenda) -> None
- get_agenda(agenda_id) -> ResearchAgenda | None
- get_active_agenda() -> ResearchAgenda | None     # newest active (several may be active)
- list_agendas(*, only_active=False) -> list[ResearchAgenda]
- set_active_agenda(agenda_id) -> None             # mark one agenda active

Multiple agendas may be active at the same time; callers that operate on a
specific agenda should pass agenda_id explicitly. get_active_agenda() is kept
as a convenience for single-agenda deployments and returns the newest active row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from contracts.agenda import ResearchAgenda
from contracts.base import ContractValidationError, ensure_dict, ensure_string_list
from db import database as db


def _row_to_agenda(row: Mapping[str, Any]) -> ResearchAgenda:
    raw_config = row.get("raw_config_json")
    if isinstance(raw_config, str):
        try:
            raw_config_obj = json.loads(raw_config) if raw_config else {}
        except json.JSONDecodeError:
            raw_config_obj = {}
    else:
        raw_config_obj = raw_config or {}

    def _decode(field_name: str, default: Any) -> Any:
        value = row.get(field_name)
        if isinstance(value, str):
            try:
                return json.loads(value) if value else default
            except json.JSONDecodeError:
                return default
        return value if value is not None else default

    agenda = ResearchAgenda(
        agenda_id=int(row["id"]) if row.get("id") is not None else None,
        version=str(row.get("version") or "v1"),
        name=str(row.get("name") or ""),
        description=str(row.get("description") or ""),
        focus=ensure_string_list(_decode("focus_json", [])),
        prefer=ensure_dict(_decode("prefer_json", {})),
        reject=ensure_dict(_decode("reject_json", {})),
        required_output=ensure_dict(_decode("required_output_json", {})),
        raw_config=ensure_dict(raw_config_obj),
        is_active=bool(row.get("is_active", 1)),
        submitter=str(row.get("submitter") or ""),
        token_budget=row.get("token_budget"),
        token_spent=int(row.get("token_spent") or 0),
        status=str(row.get("status") or "active"),
    )
    agenda.validate()
    return agenda


def parse_agenda(payload: Mapping[str, Any], *, agenda_id: int | None = None) -> ResearchAgenda:
    """Parse a YAML/JSON dict into a validated ResearchAgenda contract."""
    if not isinstance(payload, Mapping):
        raise ContractValidationError("agenda payload must be a mapping")

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
        status=str(payload.get("status") or "active"),
    )
    agenda.validate()
    return agenda


def load_agenda_from_file(path: str | Path) -> ResearchAgenda:
    """Load an agenda from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML agenda files; install pyyaml"
            ) from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text) if text.strip() else {}
    return parse_agenda(payload)


def save_agenda(agenda: ResearchAgenda) -> int:
    """Insert a new agenda. Returns the new agenda_id.

    Does not deactivate other agendas: several agendas may run concurrently,
    each isolated by agenda_id and its own token budget.
    """
    agenda.validate()
    new_id = db.insert_returning_id(
        """
        INSERT INTO research_agendas
            (version, name, description, focus_json, prefer_json, reject_json,
             required_output_json, raw_config_json, is_active,
             submitter, token_budget, token_spent, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            agenda.version,
            agenda.name,
            agenda.description,
            json.dumps(agenda.focus, ensure_ascii=False),
            json.dumps(agenda.prefer, ensure_ascii=False),
            json.dumps(agenda.reject, ensure_ascii=False),
            json.dumps(agenda.required_output, ensure_ascii=False),
            json.dumps(agenda.raw_config, ensure_ascii=False),
            1 if agenda.is_active else 0,
            agenda.submitter or None,
            agenda.token_budget,
            int(agenda.token_spent or 0),
            agenda.status,
        ),
    )
    db.commit()
    agenda.agenda_id = new_id
    return new_id


def update_agenda(agenda_id: int, agenda: ResearchAgenda) -> None:
    """Update an existing agenda in place."""
    agenda.validate()
    db.execute(
        """
        UPDATE research_agendas
        SET version=?, name=?, description=?, focus_json=?, prefer_json=?,
            reject_json=?, required_output_json=?, raw_config_json=?,
            is_active=?, submitter=?, token_budget=?, status=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            agenda.version,
            agenda.name,
            agenda.description,
            json.dumps(agenda.focus, ensure_ascii=False),
            json.dumps(agenda.prefer, ensure_ascii=False),
            json.dumps(agenda.reject, ensure_ascii=False),
            json.dumps(agenda.required_output, ensure_ascii=False),
            json.dumps(agenda.raw_config, ensure_ascii=False),
            1 if agenda.is_active else 0,
            agenda.submitter or None,
            agenda.token_budget,
            agenda.status,
            agenda_id,
        ),
    )
    db.commit()


def get_agenda(agenda_id: int) -> ResearchAgenda | None:
    row = db.fetchone(
        "SELECT * FROM research_agendas WHERE id=?",
        (agenda_id,),
    )
    if not row:
        return None
    return _row_to_agenda(row)


def get_active_agenda() -> ResearchAgenda | None:
    """Return the newest active agenda.

    Several agendas may be active at once; this helper exists for
    single-agenda deployments and callers without an explicit agenda_id.
    """
    row = db.fetchone(
        "SELECT * FROM research_agendas WHERE is_active=1 ORDER BY created_at DESC, id DESC LIMIT 1",
        (),
    )
    if not row:
        return None
    return _row_to_agenda(row)


def list_agendas(*, only_active: bool = False) -> list[ResearchAgenda]:
    if only_active:
        rows = db.fetchall(
            "SELECT * FROM research_agendas WHERE is_active=1 ORDER BY created_at DESC",
            (),
        )
    else:
        rows = db.fetchall(
            "SELECT * FROM research_agendas ORDER BY created_at DESC",
            (),
        )
    return [_row_to_agenda(r) for r in rows]


def set_active_agenda(agenda_id: int) -> None:
    """Mark an agenda active.

    Historically this cleared is_active on every other agenda (single-active
    model). Agendas are now isolated per agenda_id, so activating one no
    longer deactivates the rest.
    """
    db.execute(
        "UPDATE research_agendas SET is_active=1, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (agenda_id,),
    )
    db.commit()
