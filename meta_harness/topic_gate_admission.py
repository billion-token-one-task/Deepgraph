"""Non-bypassable topic-gate admission for resource-consuming decisions.

The selector applies the gate when it ranks candidates, but ranking is advisory:
a candidate can also reach a portfolio decision through the legacy backlog, an
operator route, or a direct auto-research call. Those paths all converge on one
place before anything is spent -- a persisted ``promote``/``revisit`` decision
packet, which is the only thing a standard ResourceGrant can be issued against.

So the gate is re-evaluated here, deterministically, from the persisted rows:
no candidate can buy tokens or GPU hours without passing it, whatever route it
arrived by.
"""

from __future__ import annotations

from agents.topic_gate import (
    TopicGateDecision,
    TopicGatePolicy,
    policy_from_config,
    screen_candidate,
)
from db import database as db


class TopicGateAdmissionError(PermissionError):
    """Raised when a candidate cannot pay for the compute it is asking for."""


def evaluate(
    *,
    agenda_id: int,
    idea_id: int,
    policy: TopicGatePolicy | None = None,
) -> TopicGateDecision:
    """Re-run the gate from persisted, agenda-scoped rows."""
    if int(agenda_id) <= 0 or int(idea_id) <= 0:
        raise TopicGateAdmissionError("topic gate scope ids must be positive")
    candidate = db.fetchone(
        "SELECT * FROM deep_insights WHERE id=? AND agenda_id=?",
        (int(idea_id), int(agenda_id)),
    )
    if not candidate:
        raise TopicGateAdmissionError(
            "candidate is not bound to this agenda; explicit import is required"
        )
    agenda_row = db.fetchone(
        "SELECT * FROM research_agendas WHERE id=?",
        (int(agenda_id),),
    )
    if not agenda_row:
        raise TopicGateAdmissionError("agenda does not exist")

    from agents.agenda_repository import row_to_agenda

    return screen_candidate(
        dict(candidate),
        row_to_agenda(agenda_row),
        policy=policy or policy_from_config(),
    )


def require_pass(
    *,
    agenda_id: int,
    idea_id: int,
    policy: TopicGatePolicy | None = None,
) -> TopicGateDecision:
    """Return the decision, or refuse the candidate with auditable reasons."""
    decision = evaluate(agenda_id=agenda_id, idea_id=idea_id, policy=policy)
    if not decision.passed:
        raise TopicGateAdmissionError(
            "topic gate blocked this candidate:" + ",".join(decision.reason_codes)
        )
    return decision
