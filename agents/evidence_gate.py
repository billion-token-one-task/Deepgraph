"""Agenda-aware adapter to the single scientific evidence contract."""

from __future__ import annotations

import json
from typing import Any

from contracts.scientific_evidence import EvidenceDecisionInput, decide_evidence
from db import database as db


def evaluate_and_record(
    *,
    agenda_id: int,
    selection_id: int,
    experiment_run_id: int,
    evidence: EvidenceDecisionInput,
    evaluator_route_observation_id: int | None = None,
) -> dict[str, Any]:
    if agenda_id <= 0 or selection_id <= 0 or experiment_run_id <= 0:
        raise ValueError("agenda, selection and experiment run IDs are required")
    scope = db.fetchone(
        """
        SELECT s.agenda_id AS selection_agenda, r.agenda_id AS run_agenda
        FROM agenda_selections s
        JOIN experiment_runs r ON r.id=?
        WHERE s.id=?
        """,
        (experiment_run_id, selection_id),
    )
    if not scope or {
        int(scope.get("selection_agenda") or 0),
        int(scope.get("run_agenda") or 0),
    } != {agenda_id}:
        raise ValueError("cross-agenda evidence gate request")
    decision = decide_evidence(evidence)
    gate_id = db.insert_returning_id(
        """
        INSERT INTO agenda_evidence_gates
            (agenda_id, selection_id, experiment_run_id, status, blockers_json,
             metrics_summary_json, rule_set, evaluator_route_observation_id)
        VALUES (?, ?, ?, ?, ?, ?, 'meta_harness_v1', ?)
        RETURNING id
        """,
        (
            agenda_id,
            selection_id,
            experiment_run_id,
            "passed" if decision.confirmation_allowed else "blocked",
            json.dumps(decision.blockers),
            json.dumps(decision.to_dict()),
            evaluator_route_observation_id,
        ),
    )
    db.commit()
    return {"gate_id": gate_id, **decision.to_dict()}
