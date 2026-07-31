"""Fair multi-agenda intake loop.

The loop may enqueue a scoped idea for portfolio review. It cannot call an LLM
or ComputeBackend and cannot issue a ResourceGrant.
"""

from __future__ import annotations

from typing import Any

from agents.agenda_repository import AgendaRepository
from agents.agenda_selector import select_next


def run_scoped_cycle(
    *,
    repository: AgendaRepository | None = None,
) -> dict[str, Any]:
    repo = repository or AgendaRepository()
    agendas = repo.list_active()
    if not agendas:
        return {"status": "idle", "reason": "no_active_agenda", "scheduled": []}
    for agenda in agendas:
        selection = select_next(int(agenda.agenda_id or 0), repository=repo)
        if selection is None:
            continue
        return {
            "status": "queued_for_portfolio",
            "agenda_id": selection.agenda_id,
            "selection_id": selection.selection_id,
            "insight_id": selection.selected_insight_id,
            "job_id": selection.auto_research_job_id,
            "scheduled": [selection.selected_insight_id],
        }
    return {
        "status": "idle",
        "reason": "no_scoped_candidate",
        "agenda_ids": [agenda.agenda_id for agenda in agendas],
        "scheduled": [],
    }
