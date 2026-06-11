"""Per-agenda LLM token accounting and budget enforcement.

Each research agenda carries a token budget (research_agendas.token_budget,
falling back to config.AGENDA_TOKEN_BUDGET_DEFAULT). A budget of 0, NULL
(with default 0) or a negative value means "no cap": check_budget always
passes and the agenda is never flipped to 'paused_budget', but record_usage
still writes the agenda_token_ledger row and bumps token_spent, so accounting
stays complete either way. Work performed on behalf of an agenda runs inside
`agenda_scope(agenda_id, operation)`; the LLM client then:

1. calls `check_budget(agenda_id)` BEFORE issuing the provider request — if the
   budget is exhausted the call fails with AgendaBudgetExceededError before any
   tokens are spent, so the caller can stop cleanly without partial writes;
2. calls `record_usage(...)` after a successful response — one ledger row per
   call plus an atomic increment of research_agendas.token_spent.

When token_spent crosses the budget the agenda is flipped to status
'paused_budget'. Raising the budget (token_spent < budget again) or calling
`resume_agenda` re-enables it.

Note: the check-then-record pair is not a distributed lock; concurrent calls
may overshoot the budget by roughly one call's worth of tokens. That is an
accepted tolerance for a soft cost cap.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

from config import AGENDA_TOKEN_BUDGET_DEFAULT
from db import database as db

AGENDA_STATUS_PAUSED_BUDGET = "paused_budget"
AGENDA_STATUS_ACTIVE = "active"

# (agenda_id, operation) for the work currently running in this context.
_scope_var: contextvars.ContextVar[tuple[int, str] | None] = contextvars.ContextVar(
    "agenda_budget_scope", default=None
)


class AgendaBudgetExceededError(RuntimeError):
    """Raised before an LLM call when the agenda's token budget is spent."""

    def __init__(self, agenda_id: int, token_spent: int, token_budget: int):
        self.agenda_id = int(agenda_id)
        self.token_spent = int(token_spent)
        self.token_budget = int(token_budget)
        super().__init__(
            f"agenda {agenda_id} token budget exhausted "
            f"({token_spent}/{token_budget} tokens); raise token_budget or call "
            f"POST /api/research_agenda/{agenda_id}/resume to continue"
        )


@contextmanager
def agenda_scope(agenda_id: int, operation: str = "llm_call") -> Iterator[None]:
    """Attribute all LLM calls inside the block to the given agenda."""
    token = _scope_var.set((int(agenda_id), str(operation)))
    try:
        yield
    finally:
        _scope_var.reset(token)


def current_scope() -> tuple[int, str] | None:
    """Return (agenda_id, operation) for the current context, if any."""
    return _scope_var.get()


def effective_budget(token_budget: Any) -> int:
    """Resolve a row's token_budget to the enforced value (NULL -> default).

    The resolved value may be <= 0, which means no cap is enforced (usage is
    still recorded by record_usage).
    """
    if token_budget is None:
        return int(AGENDA_TOKEN_BUDGET_DEFAULT)
    return int(token_budget)


def get_budget_state(agenda_id: int) -> dict[str, Any] | None:
    row = db.fetchone(
        "SELECT id, token_budget, token_spent, status FROM research_agendas WHERE id=?",
        (int(agenda_id),),
    )
    if not row:
        return None
    budget = effective_budget(row.get("token_budget"))
    spent = int(row.get("token_spent") or 0)
    return {
        "agenda_id": int(row["id"]),
        "token_budget": budget,
        "token_budget_raw": row.get("token_budget"),
        "token_spent": spent,
        "status": str(row.get("status") or AGENDA_STATUS_ACTIVE),
        "exhausted": budget > 0 and spent >= budget,
    }


def check_budget(agenda_id: int) -> None:
    """Raise AgendaBudgetExceededError if the agenda may not spend more tokens.

    A budget <= 0 (including the NULL -> default fallback) disables the cap:
    the check always passes and the agenda is never paused for budget reasons,
    while record_usage keeps writing the ledger and token_spent.

    A 'paused_budget' agenda whose budget was raised in the meantime
    (token_spent < budget again) is automatically reactivated, so increasing
    the budget alone is enough to continue.
    """
    state = get_budget_state(agenda_id)
    if state is None:
        # Unknown agenda: nothing to enforce. Scoped callers validate
        # existence elsewhere; do not block the call on accounting state.
        return
    budget = state["token_budget"]
    spent = state["token_spent"]
    if budget <= 0:  # explicit 0/negative budget disables the cap
        return
    if spent >= budget:
        if state["status"] != AGENDA_STATUS_PAUSED_BUDGET:
            _set_status(agenda_id, AGENDA_STATUS_PAUSED_BUDGET)
        raise AgendaBudgetExceededError(agenda_id, spent, budget)
    if state["status"] == AGENDA_STATUS_PAUSED_BUDGET:
        # Budget was raised above current spend: unpause and continue.
        _set_status(agenda_id, AGENDA_STATUS_ACTIVE)


def record_usage(
    agenda_id: int,
    operation: str,
    tokens: int,
    cost_usd: float | None = None,
) -> dict[str, Any] | None:
    """Append a ledger row and bump research_agendas.token_spent.

    If the new total crosses the budget, the agenda is set to 'paused_budget'
    so the next check_budget() stops further spending. Ledger insert, counter
    update and the status flip commit together.
    """
    tokens = int(tokens or 0)
    agenda_id = int(agenda_id)
    db.execute(
        """
        INSERT INTO agenda_token_ledger (agenda_id, operation, tokens, cost_usd)
        VALUES (?, ?, ?, ?)
        """,
        (agenda_id, str(operation or "llm_call"), tokens, cost_usd),
    )
    db.execute(
        "UPDATE research_agendas SET token_spent = COALESCE(token_spent, 0) + ?, "
        "updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (tokens, agenda_id),
    )
    state = get_budget_state(agenda_id)
    if state and state["exhausted"] and state["status"] != AGENDA_STATUS_PAUSED_BUDGET:
        db.execute(
            "UPDATE research_agendas SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (AGENDA_STATUS_PAUSED_BUDGET, agenda_id),
        )
        state["status"] = AGENDA_STATUS_PAUSED_BUDGET
    db.commit()
    return state


def resume_agenda(agenda_id: int, *, token_budget: int | None = None) -> dict[str, Any] | None:
    """Reactivate a budget-paused agenda, optionally raising its budget."""
    agenda_id = int(agenda_id)
    if token_budget is not None:
        db.execute(
            "UPDATE research_agendas SET token_budget=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (int(token_budget), agenda_id),
        )
    db.execute(
        "UPDATE research_agendas SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND status=?",
        (AGENDA_STATUS_ACTIVE, agenda_id, AGENDA_STATUS_PAUSED_BUDGET),
    )
    db.commit()
    return get_budget_state(agenda_id)


def _set_status(agenda_id: int, status: str) -> None:
    db.execute(
        "UPDATE research_agendas SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (status, int(agenda_id)),
    )
    db.commit()


def usage_summary() -> dict[str, Any]:
    """Aggregate the ledger per agenda + overall totals (local accounting view)."""
    per_agenda = db.fetchall(
        """
        SELECT ra.id AS agenda_id, ra.name, ra.status, ra.submitter,
               ra.token_budget, ra.token_spent,
               COALESCE(SUM(l.tokens), 0) AS ledger_tokens,
               SUM(l.cost_usd) AS ledger_cost_usd,
               COUNT(l.id) AS ledger_entries
        FROM research_agendas ra
        LEFT JOIN agenda_token_ledger l ON l.agenda_id = ra.id
        GROUP BY ra.id, ra.name, ra.status, ra.submitter, ra.token_budget, ra.token_spent
        ORDER BY ra.id
        """
    )
    rows = []
    total_tokens = 0
    total_cost = 0.0
    has_cost = False
    for r in per_agenda:
        spent = int(r.get("token_spent") or 0)
        budget = effective_budget(r.get("token_budget"))
        ledger_tokens = int(r.get("ledger_tokens") or 0)
        cost = r.get("ledger_cost_usd")
        if cost is not None:
            has_cost = True
            total_cost += float(cost)
        total_tokens += ledger_tokens
        rows.append(
            {
                "agenda_id": r.get("agenda_id"),
                "name": r.get("name"),
                "status": r.get("status") or AGENDA_STATUS_ACTIVE,
                "submitter": r.get("submitter"),
                "token_budget": budget,
                "token_spent": spent,
                "ledger_tokens": ledger_tokens,
                "ledger_cost_usd": cost,
                "ledger_entries": int(r.get("ledger_entries") or 0),
                "remaining": max(0, budget - spent) if budget > 0 else None,
            }
        )
    return {
        "agendas": rows,
        "totals": {
            "tokens": total_tokens,
            "cost_usd": total_cost if has_cost else None,
            "agenda_count": len(rows),
        },
    }
