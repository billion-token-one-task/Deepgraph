"""Read-only checks for downstream use of a positive scientific decision."""

from __future__ import annotations

from db import database as db


POSITIVE_DECISION_STATES = {"scientifically_decided", "manuscript_allowed"}


def positive_decision_authorized(
    *,
    agenda_id: int,
    run_id: int | None,
) -> bool:
    """Require both canonical state and its immutable supported transition."""
    try:
        agenda_id = int(agenda_id)
        run_id = int(run_id or 0)
    except (TypeError, ValueError):
        return False
    if agenda_id <= 0 or run_id <= 0:
        return False
    run = db.fetchone(
        """
        SELECT agenda_id, scientific_evidence_state
        FROM experiment_runs
        WHERE id=?
        """,
        (run_id,),
    )
    state = str((run or {}).get("scientific_evidence_state") or "")
    if (
        not run
        or int(run.get("agenda_id") or 0) != agenda_id
        or state not in POSITIVE_DECISION_STATES
    ):
        return False
    decision = db.fetchone(
        """
        SELECT verdict, verdict_hash, evidence_audit_record_id
        FROM scientific_decision_records
        WHERE agenda_id=? AND experiment_run_id=?
        """,
        (agenda_id, run_id),
    )
    return bool(
        decision
        and str(decision.get("verdict") or "").strip().lower() == "supported"
        and str(decision.get("verdict_hash") or "").strip()
        and int(decision.get("evidence_audit_record_id") or 0) > 0
    )
