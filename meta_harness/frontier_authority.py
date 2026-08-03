"""Admission and persistence for the Frontier-evaluator bootstrap authority.

The safety model correctly refuses an ungranted LLM, but a ResourceGrant needs
a portfolio decision, a portfolio decision needs a Frontier packet, and a
Frontier packet needs an evaluator. That is a deadlock, not a safety property.

A :class:`FrontierEvaluationAuthority` breaks it with the smallest possible
authority:

* one active Agenda and one persisted research problem;
* a hard token ceiling, a short TTL and an idempotency key;
* one pinned provider/model/prompt-version route;
* the ``frontier_assessment`` operation and nothing else;
* agenda budget reserved through the same ledger every other spend uses.

Everything it cannot do is enforced here rather than documented: no GPU, no
compute backend, no experiment, no proposal, no legacy import, no second
agenda, and no second run once consumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from contracts.meta_harness import FrontierEvaluationAuthority
from db import database as db


class FrontierAuthorityError(PermissionError):
    """Raised before any provider call when the authority is insufficient."""


class FrontierAuthorityPersistenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrontierEvaluationRequest:
    agenda_id: int
    research_problem_id: int
    operation: str
    token_cap: int
    backend: str = "llm"
    gpu_hours: float = 0.0
    # Independence: an evaluator that is the proposer is not an evaluator.
    proposer_provider: str | None = None
    proposer_model_family: str | None = None


def authorize(
    authority: FrontierEvaluationAuthority | None,
    request: FrontierEvaluationRequest,
    *,
    now: datetime | None = None,
) -> FrontierEvaluationAuthority:
    """Fail closed with every blocker named. Never partially authorizes."""
    if authority is None:
        raise FrontierAuthorityError("frontier_evaluation_authority_required")
    authority.validate()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = datetime.fromisoformat(
        authority.expires_at.replace("Z", "+00:00")
    ).astimezone(timezone.utc)

    blockers: list[str] = []
    if authority.status != "active":
        blockers.append(f"authority_{authority.status}")
    if expires <= current:
        blockers.append("authority_expired")
    if authority.agenda_id != int(request.agenda_id):
        blockers.append("agenda_scope_mismatch")
    if authority.research_problem_id != int(request.research_problem_id):
        blockers.append("research_problem_scope_mismatch")
    if request.operation not in authority.allowed_operations:
        blockers.append("operation_not_allowed")
    if request.backend not in authority.backend_allowlist:
        blockers.append("backend_not_allowed")
    if float(request.gpu_hours) > authority.max_gpu_hours:
        blockers.append("gpu_not_allowed")
    if int(request.token_cap) <= 0:
        blockers.append("token_cap_must_be_positive")
    if int(request.token_cap) > authority.token_cap:
        blockers.append("token_cap_exceeded")
    if request.proposer_provider or request.proposer_model_family:
        same_provider = authority.provider == (request.proposer_provider or "")
        same_family = authority.model_family == (request.proposer_model_family or "")
        if same_provider and same_family:
            blockers.append("evaluator_not_independent_of_proposer")
    if blockers:
        raise FrontierAuthorityError(",".join(sorted(blockers)))
    return authority


def _row_to_authority(row: dict) -> FrontierEvaluationAuthority:
    authority = FrontierEvaluationAuthority(
        agenda_id=int(row["agenda_id"]),
        research_problem_id=int(row["research_problem_id"]),
        token_cap=int(row["token_cap"]),
        issued_at=str(row["issued_at"]),
        expires_at=str(row["expires_at"]),
        idempotency_key=str(row["idempotency_key"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        model_family=str(row["model_family"]),
        prompt_version=str(row["prompt_version"]),
        evaluator=str(row["evaluator"]),
        issued_by=str(row["issued_by"]),
        issue_reason=str(row["issue_reason"]),
        status=str(row["status"]),
        authority_id=int(row["id"]),
        reservation_id=(
            int(row["reservation_id"]) if row.get("reservation_id") else None
        ),
    )
    authority.validate()
    return authority


class FrontierAuthorityRepository:
    """Persistence for authorities and their usage ledger."""

    def issue(self, authority: FrontierEvaluationAuthority) -> int:
        """Reserve agenda budget and persist one authority, idempotently."""
        authority.validate()
        try:
            existing = db.fetchone(
                """
                SELECT * FROM frontier_evaluation_authorities
                WHERE agenda_id=? AND idempotency_key=?
                """,
                (authority.agenda_id, authority.idempotency_key),
            )
            if existing:
                db.commit()
                authority.authority_id = int(existing["id"])
                authority.reservation_id = (
                    int(existing["reservation_id"])
                    if existing.get("reservation_id")
                    else None
                )
                return authority.authority_id
            agenda = db.fetchone(
                "SELECT id, status FROM research_agendas WHERE id=?",
                (authority.agenda_id,),
            )
            if not agenda or str(agenda.get("status")) != "active":
                raise FrontierAuthorityPersistenceError("agenda is not active")
            problem = db.fetchone(
                """
                SELECT id FROM research_problems WHERE id=? AND agenda_id=?
                """,
                (authority.research_problem_id, authority.agenda_id),
            )
            if not problem:
                raise FrontierAuthorityPersistenceError(
                    "research problem is not bound to this agenda"
                )

            from agents.agenda_repository import AgendaRepository

            reservation = AgendaRepository().reserve(
                agenda_id=authority.agenda_id,
                operation="frontier_bootstrap_evaluation",
                idempotency_key=f"frontier-authority:{authority.idempotency_key}",
                token_cap=authority.token_cap,
            )
            authority_id = db.insert_returning_id(
                """
                INSERT INTO frontier_evaluation_authorities
                    (agenda_id, research_problem_id, token_cap, issued_at,
                     expires_at, idempotency_key, provider, model, model_family,
                     prompt_version, evaluator, issued_by, issue_reason,
                     reservation_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                RETURNING id
                """,
                (
                    authority.agenda_id,
                    authority.research_problem_id,
                    authority.token_cap,
                    authority.issued_at,
                    authority.expires_at,
                    authority.idempotency_key,
                    authority.provider,
                    authority.model,
                    authority.model_family,
                    authority.prompt_version,
                    authority.evaluator,
                    authority.issued_by,
                    authority.issue_reason,
                    reservation.reservation_id,
                ),
            )
            db.commit()
            authority.authority_id = authority_id
            authority.reservation_id = reservation.reservation_id
            return authority_id
        except Exception:
            db.rollback()
            raise

    def load(
        self,
        authority_id: int,
        *,
        agenda_id: int,
        research_problem_id: int,
    ) -> FrontierEvaluationAuthority:
        row = db.fetchone(
            """
            SELECT * FROM frontier_evaluation_authorities
            WHERE id=? AND agenda_id=? AND research_problem_id=?
            """,
            (int(authority_id), int(agenda_id), int(research_problem_id)),
        )
        if not row:
            raise FrontierAuthorityError("scoped frontier authority not found")
        return _row_to_authority(row)

    def completed_packet_id(self, authority_id: int, *, agenda_id: int) -> int | None:
        """Idempotent replay: the packet a consumed authority already produced."""
        row = db.fetchone(
            """
            SELECT frontier_packet_id FROM frontier_authority_usage
            WHERE authority_id=? AND agenda_id=? AND status='succeeded'
              AND frontier_packet_id IS NOT NULL
            ORDER BY id DESC
            """,
            (int(authority_id), int(agenda_id)),
        )
        packet_id = int((row or {}).get("frontier_packet_id") or 0)
        return packet_id or None

    def record_usage(
        self,
        *,
        authority: FrontierEvaluationAuthority,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        status: str,
        failure_reason: str | None = None,
        frontier_packet_id: int | None = None,
        evidence_query_ref: str = "",
    ) -> int:
        """Append one auditable ledger row. Never overwrites a prior row."""
        if status not in {"succeeded", "failed"}:
            raise FrontierAuthorityPersistenceError("invalid authority usage status")
        try:
            usage_id = db.insert_returning_id(
                """
                INSERT INTO frontier_authority_usage
                    (authority_id, agenda_id, research_problem_id, operation,
                     provider, model, model_family, prompt_version,
                     input_tokens, output_tokens, cost_usd, status,
                     failure_reason, frontier_packet_id, evidence_query_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    int(authority.authority_id or 0),
                    authority.agenda_id,
                    authority.research_problem_id,
                    operation,
                    authority.provider,
                    authority.model,
                    authority.model_family,
                    authority.prompt_version,
                    max(0, int(input_tokens)),
                    max(0, int(output_tokens)),
                    cost_usd,
                    status,
                    failure_reason,
                    frontier_packet_id,
                    evidence_query_ref,
                ),
            )
            db.commit()
            return usage_id
        except Exception:
            db.rollback()
            raise

    def settle(
        self,
        authority: FrontierEvaluationAuthority,
        *,
        tokens_used: int,
        cost_usd: float | None,
        outcome: str,
    ) -> None:
        """Close the authority exactly once: consumed on success, revoked on failure.

        Either way the agenda budget is settled to actual usage, so a failed
        bootstrap cannot leave reserved tokens stranded.
        """
        if outcome not in {"consumed", "revoked"}:
            raise FrontierAuthorityPersistenceError("invalid authority outcome")
        from agents.agenda_repository import AgendaRepository

        repository = AgendaRepository()
        reservation_id = int(authority.reservation_id or 0)
        if reservation_id > 0:
            if int(tokens_used) > 0:
                repository.settle(
                    reservation_id,
                    tokens_used=min(int(tokens_used), authority.token_cap),
                    cost_usd=cost_usd,
                )
            else:
                repository.release(
                    reservation_id,
                    reason=f"frontier_bootstrap_{outcome}_without_usage",
                )
        try:
            db.execute(
                """
                UPDATE frontier_evaluation_authorities
                SET status=?, closed_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='active'
                """,
                (outcome, int(authority.authority_id or 0), authority.agenda_id),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

    def expire_stale(self, *, agenda_id: int) -> int:
        """Expire timed-out authorities and give their budget back.

        Marking an authority expired without releasing its agenda reservation
        would strand tokens: the Agenda would count them as reserved forever
        and eventually refuse work it can afford. Expiry is a withdrawal of
        authority, so the reservation is released, never settled as usage.

        Returns the number of authorities expired by this call. Scoped to one
        Agenda; there is deliberately no global sweep.
        """
        stale = db.fetchall(
            """
            SELECT id, reservation_id FROM frontier_evaluation_authorities
            WHERE agenda_id=? AND status='active'
              AND expires_at <= CURRENT_TIMESTAMP
            """,
            (int(agenda_id),),
        )
        if not stale:
            return 0

        from agents.agenda_repository import AgendaRepository

        repository = AgendaRepository()
        expired = 0
        for row in stale:
            reservation_id = int(row.get("reservation_id") or 0)
            if reservation_id > 0:
                repository.release(
                    reservation_id,
                    reason="frontier_authority_expired_unused",
                )
            try:
                db.execute(
                    """
                    UPDATE frontier_evaluation_authorities
                    SET status='expired', closed_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=? AND status='active'
                    """,
                    (int(row["id"]), int(agenda_id)),
                )
                db.commit()
                expired += 1
            except Exception:
                db.rollback()
                raise
        return expired

    def revoke_unused(
        self,
        authority_id: int,
        *,
        agenda_id: int,
        reason: str,
    ) -> bool:
        """Hand back an authority that was issued but never used.

        The operator path for "I issued this and then decided not to run it":
        it releases the reservation and closes the authority, and it refuses to
        touch one that already recorded usage.
        """
        if not str(reason).strip():
            raise FrontierAuthorityPersistenceError("revocation reason is required")
        used = db.fetchone(
            """
            SELECT COUNT(*) AS count FROM frontier_authority_usage
            WHERE authority_id=? AND agenda_id=?
            """,
            (int(authority_id), int(agenda_id)),
        )
        if int((used or {}).get("count") or 0) > 0:
            raise FrontierAuthorityPersistenceError(
                "authority already recorded usage; it cannot be revoked as unused"
            )
        row = db.fetchone(
            """
            SELECT id, reservation_id FROM frontier_evaluation_authorities
            WHERE id=? AND agenda_id=? AND status='active'
            """,
            (int(authority_id), int(agenda_id)),
        )
        if not row:
            return False

        from agents.agenda_repository import AgendaRepository

        reservation_id = int(row.get("reservation_id") or 0)
        if reservation_id > 0:
            AgendaRepository().release(
                reservation_id,
                reason=f"frontier_authority_revoked:{reason}"[:200],
            )
        try:
            db.execute(
                """
                UPDATE frontier_evaluation_authorities
                SET status='revoked', closed_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='active'
                """,
                (int(authority_id), int(agenda_id)),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        return True

    def audit_record(self, authority_id: int, *, agenda_id: int) -> dict:
        """Everything a reviewer needs to verify one bootstrap, no secrets."""
        authority = db.fetchone(
            """
            SELECT id, agenda_id, research_problem_id, token_cap, issued_at,
                   expires_at, provider, model, model_family, prompt_version,
                   evaluator, issued_by, issue_reason, status, closed_at
            FROM frontier_evaluation_authorities
            WHERE id=? AND agenda_id=?
            """,
            (int(authority_id), int(agenda_id)),
        )
        if not authority:
            raise FrontierAuthorityError("scoped frontier authority not found")
        usage = db.fetchall(
            """
            SELECT id, operation, provider, model, model_family, prompt_version,
                   input_tokens, output_tokens, cost_usd, status,
                   failure_reason, frontier_packet_id, evidence_query_ref,
                   created_at
            FROM frontier_authority_usage
            WHERE authority_id=? AND agenda_id=?
            ORDER BY id
            """,
            (int(authority_id), int(agenda_id)),
        )
        return {
            "authority": dict(authority),
            "usage": [dict(row) for row in usage],
            "totals": {
                "input_tokens": sum(int(row.get("input_tokens") or 0) for row in usage),
                "output_tokens": sum(
                    int(row.get("output_tokens") or 0) for row in usage
                ),
                "attempts": len(usage),
            },
        }


def assessment_schema() -> str:
    """The exact JSON an evaluator must return. Anything else fails closed."""
    return json.dumps(
        {
            "problem_status": "open|uncertain|duplicate|obsolete|solved",
            "contribution_delta": {"claim": "string", "versus": "string"},
            "why_not_obsolete": "string",
            "minimum_falsification_experiment": {
                "metric": "string",
                "baseline": "string",
                "decisive_comparison": "string",
            },
            "coverage_start": "YYYY-MM-DD",
            "coverage_end": "YYYY-MM-DD",
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
