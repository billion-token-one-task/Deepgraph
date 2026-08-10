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

    def _committed_tokens(self) -> int:
        """Tokens the grant has already promised: open reservations + real spend."""
        row = db.fetchone(
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
        return int((row or {}).get("reserved") or 0)

    def remaining(self, *, agenda_id: int | None = None) -> int:
        """Tokens still available under this grant, never negative.

        Callers size a request against this instead of against their own
        default: a grant's cap is the point of the grant, and a caller asking
        for a provider's maximum output would otherwise be refused outright
        rather than trimmed to what was actually authorized.
        """
        grant = db.fetchone(
            """
            SELECT agenda_id, token_cap, status
            FROM resource_grants
            WHERE id=? AND expires_at > CURRENT_TIMESTAMP
            """,
            (self.resource_grant_id,),
        )
        if not grant or grant.get("status") != "active":
            return 0
        if agenda_id is not None and int(grant.get("agenda_id") or 0) != int(agenda_id):
            return 0
        return max(0, int(grant.get("token_cap") or 0) - self._committed_tokens())

    def next_attempt_key(self, base_key: str, *, max_attempts: int = 3) -> str:
        """Allocate the next attempt key for one logical operation.

        An idempotency key exists so the same *delivered* work is never charged
        twice. It was also, accidentally, the retry policy: a provider call that
        succeeded but returned nothing usable still settled its key, so the
        operation could never be attempted again and its grant and candidate
        were stranded for good. Agenda 11 lost both of its candidates that way,
        each one unusable provider response away from being permanently
        unrealizable.

        Attempt identity is therefore separate from operation identity. The
        forge ring already worked around this at its own call site; keeping the
        allocation here means a new call site cannot forget to, and gets a bound
        the forge's copy never had.

        The caller must ask for a further attempt only when the previous one
        delivered nothing - this cannot know that, and re-attempting delivered
        work would charge the agenda twice. ``reserve`` still refuses anything
        the grant can no longer afford, so the grant's own cap remains the hard
        ceiling on top of ``max_attempts``.
        """

        base = str(base_key or "").strip()
        if not base:
            raise GrantUsageError("attempt key requires a base operation key")
        if max_attempts <= 0:
            raise GrantUsageError("max_attempts must be positive")
        row = db.fetchone(
            """
            SELECT COUNT(*) AS c FROM resource_grant_usage_reservations
            WHERE resource_grant_id=?
              AND (idempotency_key=? OR idempotency_key LIKE ?)
            """,
            (self.resource_grant_id, base, f"{base}:t%"),
        )
        used = int((row or {}).get("c") or 0)
        if used >= max_attempts:
            raise GrantUsageError(
                f"operation {base} has used all {max_attempts} attempts "
                f"on this grant"
            )
        return f"{base}:t{used + 1}"

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
