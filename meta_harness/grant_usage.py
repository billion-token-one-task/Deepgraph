"""Sub-reservations inside one already budget-reserved ResourceGrant."""

from __future__ import annotations

from contracts.agenda import BudgetReservation
from db import database as db


class GrantUsageError(RuntimeError):
    pass


class GrantUsageLedger:
    """Adapter used by LLMRouter without double-reserving the agenda budget."""

    def __init__(self, resource_grant_id: int):
        self.resource_grant_id = int(resource_grant_id)
        if self.resource_grant_id <= 0:
            raise GrantUsageError("resource_grant_id must be positive")

    def reserve(
        self,
        *,
        agenda_id: int,
        operation: str,
        idempotency_key: str,
        token_cap: int,
        gpu_hours_cap: float = 0.0,
    ) -> BudgetReservation:
        if gpu_hours_cap:
            raise GrantUsageError("LLM sub-reservation cannot reserve GPU hours")
        if token_cap <= 0:
            raise GrantUsageError("LLM sub-reservation token cap must be positive")
        try:
            lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            grant = db.fetchone(
                f"""
                SELECT * FROM resource_grants
                WHERE id=? AND expires_at > CURRENT_TIMESTAMP{lock}
                """,
                (self.resource_grant_id,),
            )
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != int(agenda_id)
                or grant.get("status") != "active"
            ):
                raise GrantUsageError("ResourceGrant is not active in this agenda")
            active = db.fetchone(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN status='reserved' THEN token_reserved
                        WHEN status='settled' THEN COALESCE(tokens_used, 0)
                        ELSE 0
                    END
                ), 0) AS reserved
                FROM resource_grant_usage_reservations
                WHERE resource_grant_id=? AND status IN ('reserved', 'settled')
                """,
                (self.resource_grant_id,),
            )
            existing = db.fetchone(
                """
                SELECT * FROM resource_grant_usage_reservations
                WHERE resource_grant_id=? AND idempotency_key=?
                """,
                (self.resource_grant_id, idempotency_key),
            )
            if existing:
                raise GrantUsageError(
                    f"idempotency key already exists with status {existing['status']}"
                )
            if int((active or {}).get("reserved") or 0) + token_cap > int(
                grant.get("token_cap") or 0
            ):
                raise GrantUsageError("ResourceGrant token cap would be exceeded")
            reservation_id = db.insert_returning_id(
                """
                INSERT INTO resource_grant_usage_reservations
                    (agenda_id, resource_grant_id, operation, idempotency_key,
                     token_reserved, status)
                VALUES (?, ?, ?, ?, ?, 'reserved')
                RETURNING id
                """,
                (
                    agenda_id,
                    self.resource_grant_id,
                    operation,
                    idempotency_key,
                    token_cap,
                ),
            )
            db.commit()
            return BudgetReservation(
                reservation_id=reservation_id,
                agenda_id=agenda_id,
                operation=operation,
                idempotency_key=idempotency_key,
                token_cap=token_cap,
            )
        except Exception:
            db.rollback()
            raise

    def settle(
        self,
        reservation_id: int,
        *,
        tokens_used: int,
        gpu_hours_used: float = 0.0,
        cost_usd: float | None = None,
    ) -> None:
        if gpu_hours_used:
            raise GrantUsageError("LLM sub-reservation cannot settle GPU hours")
        try:
            lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            row = db.fetchone(
                f"""
                SELECT * FROM resource_grant_usage_reservations
                WHERE id=? AND resource_grant_id=?{lock}
                """,
                (reservation_id, self.resource_grant_id),
            )
            if not row:
                raise GrantUsageError("grant usage reservation not found")
            if row.get("status") == "settled":
                db.commit()
                return
            if row.get("status") != "reserved":
                raise GrantUsageError("grant usage reservation is not settleable")
            if tokens_used < 0 or tokens_used > int(row["token_reserved"]):
                raise GrantUsageError("actual tokens exceed grant sub-reservation")
            db.execute(
                """
                UPDATE resource_grant_usage_reservations
                SET tokens_used=?, cost_usd=?, status='settled',
                    settled_at=CURRENT_TIMESTAMP
                WHERE id=? AND resource_grant_id=?
                """,
                (tokens_used, cost_usd, reservation_id, self.resource_grant_id),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

    def release(self, reservation_id: int, *, reason: str) -> None:
        if not reason:
            raise GrantUsageError("release reason is required")
        try:
            db.execute(
                """
                UPDATE resource_grant_usage_reservations
                SET status='released', release_reason=?,
                    settled_at=CURRENT_TIMESTAMP
                WHERE id=? AND resource_grant_id=? AND status='reserved'
                """,
                (reason, reservation_id, self.resource_grant_id),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
