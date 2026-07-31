"""PostgreSQL-first persistence for agenda scope and hard reservations.

The module uses the existing DB adapter but does not initialize a database at
import time. Callers must run the reviewed migration before use.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from contracts.agenda import AgendaSelection, BudgetReservation, ResearchAgenda
from db import database as db


class AgendaNotFoundError(LookupError):
    pass


class AgendaScopeError(RuntimeError):
    pass


class BudgetReservationError(RuntimeError):
    pass


def _decode(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def row_to_agenda(row: Mapping[str, Any]) -> ResearchAgenda:
    agenda = ResearchAgenda(
        agenda_id=int(row["id"]),
        version=str(row.get("version") or "v1"),
        name=str(row.get("name") or ""),
        description=str(row.get("description") or ""),
        focus=_decode(row.get("focus_json"), []),
        prefer=_decode(row.get("prefer_json"), {}),
        reject=_decode(row.get("reject_json"), {}),
        required_output=_decode(row.get("required_output_json"), {}),
        raw_config=_decode(row.get("raw_config_json"), {}),
        is_active=bool(row.get("is_active")),
        submitter=str(row.get("submitter") or ""),
        token_budget=row.get("token_budget"),
        token_spent=int(row.get("token_spent") or 0),
        token_reserved=int(row.get("token_reserved") or 0),
        gpu_hours_budget=float(row.get("gpu_hours_budget") or 0),
        gpu_hours_spent=float(row.get("gpu_hours_spent") or 0),
        gpu_hours_reserved=float(row.get("gpu_hours_reserved") or 0),
        max_concurrency=int(row.get("max_concurrency") or 1),
        backend_allowlist=_decode(row.get("backend_allowlist_json"), ["cpu", "llm"]),
        backlog_policy=str(row.get("backlog_policy") or "explicit_import_only"),
        status=str(row.get("status") or "active"),
    )
    agenda.validate()
    return agenda


class AgendaRepository:
    def create(self, agenda: ResearchAgenda) -> int:
        agenda.validate()
        agenda_id = db.insert_returning_id(
            """
            INSERT INTO research_agendas
                (version, name, description, focus_json, prefer_json, reject_json,
                 required_output_json, raw_config_json, is_active, submitter,
                 token_budget, token_spent, token_reserved, gpu_hours_budget,
                 gpu_hours_spent, gpu_hours_reserved, max_concurrency,
                 backend_allowlist_json, backlog_policy, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                agenda.token_spent,
                agenda.token_reserved,
                agenda.gpu_hours_budget,
                agenda.gpu_hours_spent,
                agenda.gpu_hours_reserved,
                agenda.max_concurrency,
                json.dumps(agenda.backend_allowlist),
                agenda.backlog_policy,
                agenda.status,
            ),
        )
        db.commit()
        agenda.agenda_id = agenda_id
        return agenda_id

    def get(self, agenda_id: int, *, lock: bool = False) -> ResearchAgenda | None:
        suffix = " FOR UPDATE" if lock and db._use_pg() else ""  # noqa: SLF001
        row = db.fetchone(
            f"SELECT * FROM research_agendas WHERE id=?{suffix}",
            (int(agenda_id),),
        )
        return row_to_agenda(row) if row else None

    def list_active(self) -> list[ResearchAgenda]:
        rows = db.fetchall(
            """
            SELECT * FROM research_agendas
            WHERE is_active=1 AND status='active' AND token_budget > 0
            ORDER BY updated_at ASC, id ASC
            """
        )
        return [row_to_agenda(row) for row in rows]

    def set_status(self, agenda_id: int, status: str) -> None:
        cur = db.execute(
            """
            UPDATE research_agendas
            SET status=?,
                is_active=CASE WHEN ?='active' THEN 1 ELSE 0 END,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (status, status, int(agenda_id)),
        )
        if int(getattr(cur, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise AgendaNotFoundError(agenda_id)
        db.commit()

    def candidates(self, agenda_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return only records already bound to this agenda.

        There is deliberately no `agenda_id IS NULL` keyword fallback.
        """
        return db.fetchall(
            """
            SELECT * FROM deep_insights
            WHERE agenda_id=?
              AND COALESCE(status, 'candidate') NOT IN ('exists', 'archived')
              AND COALESCE(outcome, 'pending') NOT IN ('cleaned', 'archived')
              AND NOT EXISTS (
                  SELECT 1 FROM auto_research_jobs arj
                  WHERE arj.deep_insight_id=deep_insights.id
              )
            ORDER BY tier DESC, created_at ASC, id ASC
            LIMIT ?
            """,
            (int(agenda_id), max(1, min(int(limit), 1000))),
        )

    def save_selection(self, selection: AgendaSelection) -> int:
        selection.validate()
        selection_id = db.insert_returning_id(
            """
            INSERT INTO agenda_selections
                (agenda_id, selected_insight_id, score, rationale,
                 rejected_candidates_json, scoring_breakdown_json, status,
                 auto_research_job_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                selection.agenda_id,
                selection.selected_insight_id,
                selection.score,
                selection.rationale,
                json.dumps(selection.rejected_candidates, ensure_ascii=False),
                json.dumps(selection.scoring_breakdown, ensure_ascii=False),
                selection.status,
                selection.auto_research_job_id,
            ),
        )
        db.commit()
        selection.selection_id = selection_id
        return selection_id

    def queue_selected_insight(self, selection: AgendaSelection) -> int:
        selection.validate()
        if selection.selected_insight_id is None:
            raise AgendaScopeError("cannot queue a selection without an insight")
        row = db.fetchone(
            "SELECT agenda_id FROM deep_insights WHERE id=?",
            (selection.selected_insight_id,),
        )
        if not row or int(row.get("agenda_id") or 0) != selection.agenda_id:
            raise AgendaScopeError("selected insight is not bound to the agenda")
        existing_job = db.fetchone(
            "SELECT id, agenda_id FROM auto_research_jobs WHERE deep_insight_id=?",
            (selection.selected_insight_id,),
        )
        if existing_job:
            if int(existing_job.get("agenda_id") or 0) != selection.agenda_id:
                raise AgendaScopeError(
                    "existing backlog job is unscoped or belongs to another "
                    "agenda; explicitly import it before reuse"
                )
            job_id = int(existing_job["id"])
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='queued', stage='awaiting_portfolio_decision',
                    last_note=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=?
                """,
                (
                    f"agenda_selection:{selection.selection_id}",
                    job_id,
                    selection.agenda_id,
                ),
            )
        else:
            job_id = db.insert_returning_id(
                """
                INSERT INTO auto_research_jobs
                    (agenda_id, deep_insight_id, status, stage, last_note)
                VALUES (?, ?, 'queued', 'awaiting_portfolio_decision', ?)
                RETURNING id
                """,
                (
                    selection.agenda_id,
                    selection.selected_insight_id,
                    f"agenda_selection:{selection.selection_id}",
                ),
            )
        db.execute(
            """
            UPDATE agenda_selections
            SET auto_research_job_id=?, status='awaiting_portfolio_decision',
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (job_id, selection.selection_id, selection.agenda_id),
        )
        db.commit()
        selection.auto_research_job_id = job_id
        selection.status = "awaiting_portfolio_decision"
        return job_id

    def import_legacy_record(
        self,
        *,
        agenda_id: int,
        entity_type: str,
        entity_id: int,
        actor: str,
        reason: str,
        idempotency_key: str,
    ) -> int:
        if int(agenda_id) <= 0 or int(entity_id) <= 0:
            raise AgendaScopeError("legacy import ids must be positive")
        if not str(actor or "").strip():
            raise AgendaScopeError("legacy import actor is required")
        if not str(reason or "").strip():
            raise AgendaScopeError("legacy import reason is required")
        if not str(idempotency_key or "").strip():
            raise AgendaScopeError("legacy import idempotency_key is required")
        tables = {
            "deep_insight": "deep_insights",
            "research_problem": "research_problems",
            "auto_research_job": "auto_research_jobs",
        }
        table = tables.get(entity_type)
        if not table:
            raise AgendaScopeError("entity_type is not importable")
        if self.get(agenda_id) is None:
            raise AgendaNotFoundError(agenda_id)
        existing = db.fetchone(
            """
            SELECT id FROM legacy_scope_imports
            WHERE agenda_id=? AND idempotency_key=?
            """,
            (agenda_id, idempotency_key),
        )
        if existing:
            return int(existing["id"])
        row = db.fetchone(
            f"SELECT agenda_id FROM {table} WHERE id=?",
            (entity_id,),
        )
        if not row:
            raise AgendaScopeError("legacy record does not exist")
        bound = row.get("agenda_id")
        if bound not in (None, agenda_id):
            raise AgendaScopeError("legacy record belongs to another agenda")
        cur = db.execute(
            f"UPDATE {table} SET agenda_id=? WHERE id=? AND agenda_id IS NULL",
            (agenda_id, entity_id),
        )
        if bound is None and int(getattr(cur, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise AgendaScopeError("legacy record import raced with another writer")
        import_id = db.insert_returning_id(
            """
            INSERT INTO legacy_scope_imports
                (agenda_id, entity_type, entity_id, actor, reason, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (agenda_id, entity_type, entity_id, actor, reason, idempotency_key),
        )
        db.commit()
        return import_id

    def reserve(
        self,
        *,
        agenda_id: int,
        operation: str,
        idempotency_key: str,
        token_cap: int = 0,
        gpu_hours_cap: float = 0.0,
    ) -> BudgetReservation:
        if token_cap < 0 or gpu_hours_cap < 0 or (token_cap == 0 and gpu_hours_cap == 0):
            raise BudgetReservationError("reservation caps must be non-negative and non-empty")
        try:
            agenda = self.get(agenda_id, lock=True)
            if agenda is None:
                raise AgendaNotFoundError(agenda_id)
            existing = db.fetchone(
                """
                SELECT * FROM agenda_resource_ledger
                WHERE agenda_id=? AND idempotency_key=?
                """,
                (agenda_id, idempotency_key),
            )
            if existing:
                db.commit()
                return BudgetReservation(
                    reservation_id=int(existing["id"]),
                    agenda_id=agenda_id,
                    operation=str(existing["operation"]),
                    idempotency_key=str(existing["idempotency_key"]),
                    token_cap=int(existing.get("token_reserved") or 0),
                    gpu_hours_cap=float(existing.get("gpu_hours_reserved") or 0),
                    status=str(existing.get("status") or "reserved"),
                )
            if agenda.status != "active":
                raise BudgetReservationError(f"agenda is not active: {agenda.status}")
            if agenda.token_spent + agenda.token_reserved + token_cap > int(agenda.token_budget or 0):
                db.execute(
                    """
                    UPDATE research_agendas
                    SET status='paused_budget', updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (agenda_id,),
                )
                db.commit()
                raise BudgetReservationError("token hard cap would be exceeded")
            if (
                gpu_hours_cap > 0
                and agenda.gpu_hours_spent
                + agenda.gpu_hours_reserved
                + gpu_hours_cap
                > agenda.gpu_hours_budget
            ):
                db.execute(
                    """
                    UPDATE research_agendas
                    SET status='paused_budget', updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (agenda_id,),
                )
                db.commit()
                raise BudgetReservationError("GPU-hour hard cap would be exceeded")
            reservation_id = db.insert_returning_id(
                """
                INSERT INTO agenda_resource_ledger
                    (agenda_id, operation, idempotency_key, token_reserved,
                     gpu_hours_reserved, status)
                VALUES (?, ?, ?, ?, ?, 'reserved')
                RETURNING id
                """,
                (agenda_id, operation, idempotency_key, token_cap, gpu_hours_cap),
            )
            db.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved+?,
                    gpu_hours_reserved=gpu_hours_reserved+?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (token_cap, gpu_hours_cap, agenda_id),
            )
            db.commit()
            return BudgetReservation(
                reservation_id=reservation_id,
                agenda_id=agenda_id,
                operation=operation,
                idempotency_key=idempotency_key,
                token_cap=token_cap,
                gpu_hours_cap=gpu_hours_cap,
            )
        except Exception:
            db.rollback()
            raise

    def release(self, reservation_id: int, *, reason: str) -> None:
        if not str(reason or "").strip():
            raise BudgetReservationError("release reason is required")
        try:
            suffix = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            row = db.fetchone(
                f"SELECT * FROM agenda_resource_ledger WHERE id=?{suffix}",
                (reservation_id,),
            )
            if not row:
                raise BudgetReservationError("reservation not found")
            if row.get("status") == "released":
                db.commit()
                return
            if row.get("status") != "reserved":
                raise BudgetReservationError("reservation is not releasable")
            db.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved-?,
                    gpu_hours_reserved=gpu_hours_reserved-?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    int(row.get("token_reserved") or 0),
                    float(row.get("gpu_hours_reserved") or 0),
                    int(row["agenda_id"]),
                ),
            )
            db.execute(
                """
                UPDATE agenda_resource_ledger
                SET status='released', release_reason=?,
                    settled_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (reason, reservation_id),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

    def resume(self, agenda_id: int, *, token_budget: int) -> ResearchAgenda:
        if token_budget <= 0:
            raise BudgetReservationError("resume requires a positive token budget")
        try:
            agenda = self.get(agenda_id, lock=True)
            if agenda is None:
                raise AgendaNotFoundError(agenda_id)
            if token_budget <= agenda.token_spent + agenda.token_reserved:
                raise BudgetReservationError(
                    "new token budget must exceed spent plus reserved tokens"
                )
            db.execute(
                """
                UPDATE research_agendas
                SET token_budget=?, status='active', is_active=1,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (token_budget, agenda_id),
            )
            db.commit()
            updated = self.get(agenda_id)
            if updated is None:
                raise AgendaNotFoundError(agenda_id)
            return updated
        except Exception:
            db.rollback()
            raise

    def settle(
        self,
        reservation_id: int,
        *,
        tokens_used: int = 0,
        gpu_hours_used: float = 0.0,
        cost_usd: float | None = None,
    ) -> None:
        if tokens_used < 0 or gpu_hours_used < 0:
            raise BudgetReservationError("actual usage cannot be negative")
        try:
            suffix = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            row = db.fetchone(
                f"SELECT * FROM agenda_resource_ledger WHERE id=?{suffix}",
                (reservation_id,),
            )
            if not row:
                raise BudgetReservationError("reservation not found")
            if row.get("status") == "settled":
                db.commit()
                return
            if row.get("status") != "reserved":
                raise BudgetReservationError("reservation is not settleable")
            token_cap = int(row.get("token_reserved") or 0)
            gpu_cap = float(row.get("gpu_hours_reserved") or 0)
            if tokens_used > token_cap or gpu_hours_used > gpu_cap:
                raise BudgetReservationError("actual usage exceeds reserved hard cap")
            agenda_id = int(row["agenda_id"])
            db.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved-?,
                    gpu_hours_reserved=gpu_hours_reserved-?,
                    token_spent=token_spent+?,
                    gpu_hours_spent=gpu_hours_spent+?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (token_cap, gpu_cap, tokens_used, gpu_hours_used, agenda_id),
            )
            db.execute(
                """
                UPDATE agenda_resource_ledger
                SET tokens_used=?, gpu_hours_used=?, cost_usd=?,
                    status='settled', settled_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (tokens_used, gpu_hours_used, cost_usd, reservation_id),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
