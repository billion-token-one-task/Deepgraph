"""Context attribution and hard pre-call reservations for agenda resources."""

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
