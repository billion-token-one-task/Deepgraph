"""Context attribution and hard pre-call reservations for agenda resources.

`agenda_scope` attributes the work inside it to one agenda. Legacy `call_llm`
checks that agenda before contacting a provider and records the call on
`agenda_token_ledger` plus `token_spent`. Grant-backed role calls keep the
grant reservation as the `token_spent` clock — a parent settlement already
moves actual usage from reserved to spent — and instead append a ledger row
when the sub-reservation settles, so execution spend is visible before the
grant closes. A budget of NULL, 0, or negative stays accounting-only.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from agents.agenda_repository import AgendaRepository
from contracts.agenda import BudgetReservation


_scope: contextvars.ContextVar[tuple[int, str, str] | None] = contextvars.ContextVar(
    "agenda_resource_scope",
    default=None,
)


@contextmanager
def agenda_scope(
    agenda_id: int,
    operation: str,
    idempotency_key: str,
) -> Iterator[None]:
    if int(agenda_id) <= 0 or not operation or not idempotency_key:
        raise ValueError("agenda scope requires agenda_id, operation and idempotency_key")
    token = _scope.set((int(agenda_id), str(operation), str(idempotency_key)))
    try:
        yield
    finally:
        _scope.reset(token)


def current_scope() -> tuple[int, str, str] | None:
    return _scope.get()


def reserve(
    *,
    agenda_id: int,
    operation: str,
    idempotency_key: str,
    token_cap: int = 0,
    gpu_hours_cap: float = 0.0,
    repository: AgendaRepository | None = None,
) -> BudgetReservation:
    return (repository or AgendaRepository()).reserve(
        agenda_id=agenda_id,
        operation=operation,
        idempotency_key=idempotency_key,
        token_cap=token_cap,
        gpu_hours_cap=gpu_hours_cap,
    )


def settle(
    reservation_id: int,
    *,
    tokens_used: int = 0,
    gpu_hours_used: float = 0.0,
    cost_usd: float | None = None,
    repository: AgendaRepository | None = None,
) -> None:
    (repository or AgendaRepository()).settle(
        reservation_id,
        tokens_used=tokens_used,
        gpu_hours_used=gpu_hours_used,
        cost_usd=cost_usd,
    )


def release(
    reservation_id: int,
    *,
    reason: str,
    repository: AgendaRepository | None = None,
) -> None:
    (repository or AgendaRepository()).release(reservation_id, reason=reason)


def resume(
    agenda_id: int,
    *,
    token_budget: int,
    repository: AgendaRepository | None = None,
):
    return (repository or AgendaRepository()).resume(
        agenda_id,
        token_budget=token_budget,
    )


PAUSED_BUDGET = "paused_budget"
ACTIVE = "active"


class AgendaBudgetExceededError(RuntimeError):
    """Raised before an LLM call when the agenda's token budget is spent."""

    def __init__(self, agenda_id: int, token_spent: int, token_budget: int):
        self.agenda_id = int(agenda_id)
        self.token_spent = int(token_spent)
        self.token_budget = int(token_budget)
        super().__init__(
            f"agenda {agenda_id} token budget exhausted "
            f"({token_spent}/{token_budget} tokens); raise token_budget "
            f"above spent and reserved, then resume the agenda"
        )


def budget_cap(token_budget: object) -> int | None:
    """Return a positive cap, or None when the budget is accounting-only.

    NULL, 0, and negative budgets record usage and never pause the agenda.
    """
    if token_budget is None:
        return None
    try:
        value = int(token_budget)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _live_grant_tokens(agenda_id: int) -> int:
    """Settled LLM usage on grants that have not yet folded into token_spent.

    Parent grant settlement adds those tokens to token_spent and leaves the
    grant, so counting only still-active grants does not charge them twice.
    """
    from db import database as db

    if not db.table_exists("resource_grant_usage_reservations"):
        return 0
    if not db.table_exists("resource_grants"):
        return 0
    row = db.fetchone(
        """
        SELECT COALESCE(SUM(u.tokens_used), 0) AS spent
        FROM resource_grant_usage_reservations u
        JOIN resource_grants g ON g.id = u.resource_grant_id
        WHERE u.agenda_id=?
          AND u.status='settled'
          AND g.status='active'
          AND g.expires_at > CURRENT_TIMESTAMP
        """,
        (int(agenda_id),),
    )
    return int((row or {}).get("spent") or 0)


def _agenda_budget_row(agenda_id: int) -> dict | None:
    from db import database as db

    if not db.table_exists("research_agendas"):
        return None
    row = db.fetchone(
        """
        SELECT id, token_budget, token_spent, status
        FROM research_agendas
        WHERE id=?
        """,
        (int(agenda_id),),
    )
    return dict(row) if row else None


def charged_tokens(agenda_id: int, token_spent: int) -> int:
    """Tokens already spent plus LLM usage still inside an active grant."""
    return int(token_spent or 0) + _live_grant_tokens(agenda_id)


def _set_status(agenda_id: int, status: str, *, commit: bool) -> None:
    from db import database as db

    db.execute(
        """
        UPDATE research_agendas
        SET status=?, is_active=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (status, 1 if status == ACTIVE else 0, int(agenda_id)),
    )
    if commit:
        db.commit()


def _pause_if_exhausted(agenda_id: int, *, commit: bool) -> None:
    row = _agenda_budget_row(agenda_id)
    if not row:
        return
    cap = budget_cap(row.get("token_budget"))
    if cap is None:
        return
    charged = charged_tokens(agenda_id, int(row.get("token_spent") or 0))
    if charged >= cap and str(row.get("status") or "") != PAUSED_BUDGET:
        _set_status(agenda_id, PAUSED_BUDGET, commit=commit)


def check_budget(agenda_id: int) -> None:
    """Raise before a provider request when this agenda may not spend more.

    A NULL, 0, or negative budget disables the cap. A paused agenda whose
    budget was raised back above spent plus in-flight grant usage is
    reactivated, so increasing the budget is enough to continue.
    """
    row = _agenda_budget_row(agenda_id)
    if row is None:
        return
    cap = budget_cap(row.get("token_budget"))
    if cap is None:
        return
    charged = charged_tokens(agenda_id, int(row.get("token_spent") or 0))
    if charged >= cap:
        if str(row.get("status") or "") != PAUSED_BUDGET:
            _set_status(agenda_id, PAUSED_BUDGET, commit=True)
        raise AgendaBudgetExceededError(agenda_id, charged, cap)
    if str(row.get("status") or "") == PAUSED_BUDGET:
        _set_status(agenda_id, ACTIVE, commit=True)


def _insert_ledger_row(
    agenda_id: int,
    operation: str,
    tokens: int,
    cost_usd: float | None,
) -> bool:
    from db import database as db

    if not db.table_exists("agenda_token_ledger"):
        return False
    db.execute(
        """
        INSERT INTO agenda_token_ledger (agenda_id, operation, tokens, cost_usd)
        VALUES (?, ?, ?, ?)
        """,
        (int(agenda_id), str(operation or "llm_call"), int(tokens or 0), cost_usd),
    )
    return True


def record_usage(
    agenda_id: int,
    operation: str,
    tokens: int,
    cost_usd: float | None = None,
) -> None:
    """Record one legacy LLM call and bump token_spent.

    Grant-backed calls must not use this. Their parent reservation already
    adds actual tokens to token_spent when the grant settles.
    """
    from db import database as db

    tokens = int(tokens or 0)
    agenda_id = int(agenda_id)
    try:
        if not _insert_ledger_row(agenda_id, operation, tokens, cost_usd):
            return
        if db.table_exists("research_agendas"):
            db.execute(
                """
                UPDATE research_agendas
                SET token_spent = COALESCE(token_spent, 0) + ?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (tokens, agenda_id),
            )
        _pause_if_exhausted(agenda_id, commit=False)
        db.commit()
    except Exception:
        db.rollback()
        raise


def attribute_settled_grant_usage(
    agenda_id: int,
    operation: str,
    tokens: int,
    cost_usd: float | None = None,
) -> None:
    """Append a ledger row for one settled grant sub-reservation.

    Does not touch token_spent. The caller owns the transaction.
    """
    if not _insert_ledger_row(agenda_id, operation, int(tokens or 0), cost_usd):
        return
    _pause_if_exhausted(int(agenda_id), commit=False)
