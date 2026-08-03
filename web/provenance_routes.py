"""Read-only provenance API: the evidence ladder, selection rationale, and
process timeline that the meta-harness already persists but the UI never read.

Every response passes through an explicit field allowlist. No SELECT * reaches
jsonify from this module, free-text fields are scrubbed of absolute filesystem
paths, and there are no mutating endpoints here by design: mutations belong to
the operator-authenticated /api/meta-harness/v1 blueprint.
"""
from __future__ import annotations

import re
from typing import Any

from flask import Blueprint, jsonify, request

from contracts.meta_harness import EVIDENCE_STATES
from db import database as db

blueprint = Blueprint("provenance_v1", __name__, url_prefix="/api/v1")

# Rank of each evidence state on the ladder; used to pick the furthest state an
# idea's runs have reached.
_STATE_RANK = {state: rank for rank, state in enumerate(EVIDENCE_STATES)}

_ABS_PATH = re.compile(r"/(?:home|tmp|var|root|mnt|opt|srv)(?:/[\w.@%+~-]+)+")

_TIMELINE_DEFAULT_LIMIT = 120
_TIMELINE_MAX_LIMIT = 500

# Keys copied from an evidence transition's context_json into the public
# timeline. Hashes and verdicts are public by design; anything else stays out.
_TRANSITION_CONTEXT_KEYS = (
    "verdict",
    "blockers",
    "p_value",
    "metric_value",
    "baseline_value",
    "pilot_only",
    "execution_succeeded",
)


def _scrub_text(value: str) -> str:
    return _ABS_PATH.sub("<path>", value)


def _scrub_deep(value: Any) -> Any:
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {key: _scrub_deep(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_deep(item) for item in value]
    return value


def _ts(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    return str(value)


def _json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        import json

        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _agenda_id_or_none() -> int | None:
    try:
        agenda_id = int(request.args.get("agenda_id", ""))
    except (TypeError, ValueError):
        return None
    return agenda_id if agenda_id > 0 else None


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    """fetchall that treats a missing table as an empty result.

    The meta-harness tables arrive with migration 0001; a database that has
    not run it yet should yield empty provenance, not a 500.
    """
    try:
        return [dict(row) for row in db.fetchall(sql, params)]
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        return []


# ── Agendas ──────────────────────────────────────────────────────────


@blueprint.get("/agendas")
def list_agendas():
    """Public agenda list. Used by the frontend to establish its agenda scope.

    Submitter contact details and raw agenda configuration are not public.
    """
    rows = _rows(
        """
        SELECT id, name, description, status, is_active, focus_json,
               token_budget, token_spent, created_at, updated_at
        FROM research_agendas
        ORDER BY is_active DESC, updated_at DESC
        LIMIT 100
        """
    )
    agendas = []
    for row in rows:
        token_budget = int(row.get("token_budget") or 0)
        token_spent = int(row.get("token_spent") or 0)
        agendas.append(
            {
                "id": row["id"],
                "name": _scrub_text(str(row.get("name") or "")),
                "description": _scrub_text(str(row.get("description") or "")),
                "status": row.get("status"),
                "is_active": bool(row.get("is_active")),
                "focus": _scrub_deep(_json_field(row.get("focus_json"), [])),
                "token_budget": token_budget,
                "token_spent": token_spent,
                "budget_pct": round(100.0 * token_spent / token_budget, 1)
                if token_budget
                else None,
                "created_at": _ts(row.get("created_at")),
                "updated_at": _ts(row.get("updated_at")),
            }
        )
    return jsonify({"agendas": agendas})


# ── Evidence states (for the two-register badges) ────────────────────


@blueprint.get("/evidence_states")
def evidence_states():
    """Latest evidence-ladder position per experiment run and per idea.

    The scientific register is intentionally separate from operational job
    status: a run that merely completed stays "not assessed" here until it has
    climbed the ladder, and only scientific_decision_records carry a verdict.
    """
    agenda_id = _agenda_id_or_none()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400

    transitions = _rows(
        """
        SELECT experiment_run_id, to_state, created_at
        FROM evidence_state_transitions
        WHERE agenda_id=?
        ORDER BY created_at ASC, id ASC
        """,
        (agenda_id,),
    )
    runs: dict[int, dict] = {}
    for row in transitions:
        run_id = row.get("experiment_run_id")
        if run_id is None:
            continue
        runs[int(run_id)] = {
            "state": row.get("to_state"),
            "verdict": None,
            "decided_at": None,
        }

    for row in _rows(
        """
        SELECT experiment_run_id, verdict, created_at
        FROM scientific_decision_records
        WHERE agenda_id=?
        ORDER BY created_at ASC, id ASC
        """,
        (agenda_id,),
    ):
        run_id = row.get("experiment_run_id")
        if run_id is None:
            continue
        entry = runs.setdefault(
            int(run_id), {"state": "scientifically_decided", "verdict": None, "decided_at": None}
        )
        entry["verdict"] = row.get("verdict")
        entry["decided_at"] = _ts(row.get("created_at"))

    # Roll runs up to their idea: an idea's badge is the furthest state any of
    # its runs has reached.
    ideas: dict[int, dict] = {}
    for row in _rows(
        "SELECT id, deep_insight_id FROM experiment_runs WHERE agenda_id=?",
        (agenda_id,),
    ):
        run_id = int(row["id"])
        insight_id = row.get("deep_insight_id")
        if insight_id is None or run_id not in runs:
            continue
        candidate = dict(runs[run_id])
        candidate["run_id"] = run_id
        current = ideas.get(int(insight_id))
        if current is None or _STATE_RANK.get(candidate["state"], -1) > _STATE_RANK.get(
            current["state"], -1
        ):
            ideas[int(insight_id)] = candidate

    return jsonify(
        {
            "agenda_id": agenda_id,
            "ladder": list(EVIDENCE_STATES),
            "runs": {str(run_id): entry for run_id, entry in runs.items()},
            "ideas": {str(idea_id): entry for idea_id, entry in ideas.items()},
        }
    )


# ── Selection rationale ──────────────────────────────────────────────


@blueprint.get("/agendas/<int:agenda_id>/selection")
def agenda_selection(agenda_id: int):
    """Why work was chosen: selections with rationale and rejected candidates,
    plus candidate decision packets with machine reason codes."""
    selections = []
    for row in _rows(
        """
        SELECT id, selected_insight_id, score, rationale,
               rejected_candidates_json, scoring_breakdown_json, status,
               created_at
        FROM agenda_selections
        WHERE agenda_id=?
        ORDER BY created_at DESC, id DESC
        LIMIT 20
        """,
        (agenda_id,),
    ):
        selections.append(
            {
                "id": row["id"],
                "selected_insight_id": row.get("selected_insight_id"),
                "score": row.get("score"),
                "status": row.get("status"),
                "rationale": _scrub_text(str(row.get("rationale") or "")),
                "rejected_candidates": _scrub_deep(
                    _json_field(row.get("rejected_candidates_json"), [])
                ),
                "scoring_breakdown": _scrub_deep(
                    _json_field(row.get("scoring_breakdown_json"), {})
                ),
                "created_at": _ts(row.get("created_at")),
            }
        )

    decisions = []
    for row in _rows(
        """
        SELECT id, idea_id, decision, reason_codes_json, candidate_family,
               revisit_after, decided_at
        FROM idea_decision_packets
        WHERE agenda_id=?
        ORDER BY decided_at DESC, id DESC
        LIMIT 50
        """,
        (agenda_id,),
    ):
        decisions.append(
            {
                "id": row["id"],
                "idea_id": row.get("idea_id"),
                "decision": row.get("decision"),
                "reason_codes": _json_field(row.get("reason_codes_json"), []),
                "candidate_family": row.get("candidate_family"),
                "revisit_after": _ts(row.get("revisit_after")),
                "decided_at": _ts(row.get("decided_at")),
            }
        )

    return jsonify(
        {"agenda_id": agenda_id, "selections": selections, "decisions": decisions}
    )


# ── Process timeline ─────────────────────────────────────────────────


def _transition_events(agenda_id: int) -> list[dict]:
    events = []
    for row in _rows(
        """
        SELECT experiment_run_id, from_state, to_state, actor, context_json,
               created_at
        FROM evidence_state_transitions
        WHERE agenda_id=?
        ORDER BY created_at DESC, id DESC
        LIMIT 200
        """,
        (agenda_id,),
    ):
        context = _json_field(row.get("context_json"), {})
        public_context = {
            key: _scrub_deep(context[key])
            for key in _TRANSITION_CONTEXT_KEYS
            if isinstance(context, dict) and context.get(key) not in (None, "", [])
        }
        events.append(
            {
                "kind": "evidence",
                "at": _ts(row.get("created_at")),
                "run_id": row.get("experiment_run_id"),
                "from_state": row.get("from_state"),
                "to_state": row.get("to_state"),
                "actor": row.get("actor"),
                "context": public_context,
            }
        )
    return events


def _decision_events(agenda_id: int) -> list[dict]:
    return [
        {
            "kind": "decision",
            "at": _ts(row.get("created_at")),
            "run_id": row.get("experiment_run_id"),
            "verdict": row.get("verdict"),
            "verdict_hash": row.get("verdict_hash"),
        }
        for row in _rows(
            """
            SELECT experiment_run_id, verdict, verdict_hash, created_at
            FROM scientific_decision_records
            WHERE agenda_id=?
            ORDER BY created_at DESC, id DESC
            LIMIT 100
            """,
            (agenda_id,),
        )
    ]


def _grant_events(agenda_id: int) -> list[dict]:
    return [
        {
            "kind": "authorization",
            "at": _ts(row.get("created_at")),
            "idea_id": row.get("idea_id"),
            "stage": row.get("stage"),
            "token_cap": row.get("token_cap"),
            "max_gpu_hours": row.get("max_gpu_hours"),
            "status": row.get("status"),
            "grant_reason": _scrub_text(str(row.get("grant_reason") or "")),
        }
        for row in _rows(
            """
            SELECT idea_id, stage, token_cap, max_gpu_hours, status,
                   grant_reason, created_at
            FROM resource_grants
            WHERE agenda_id=?
            ORDER BY created_at DESC, id DESC
            LIMIT 100
            """,
            (agenda_id,),
        )
    ]


def _candidate_events(agenda_id: int) -> list[dict]:
    return [
        {
            "kind": "candidate_decision",
            "at": _ts(row.get("decided_at")),
            "idea_id": row.get("idea_id"),
            "decision": row.get("decision"),
            "reason_codes": _json_field(row.get("reason_codes_json"), []),
        }
        for row in _rows(
            """
            SELECT idea_id, decision, reason_codes_json, decided_at
            FROM idea_decision_packets
            WHERE agenda_id=?
            ORDER BY decided_at DESC, id DESC
            LIMIT 100
            """,
            (agenda_id,),
        )
    ]


def _signal_events(agenda_id: int) -> list[dict]:
    return [
        {
            "kind": "signal",
            "at": _ts(row.get("created_at")),
            "research_problem_id": row.get("research_problem_id"),
            "gate_allowed": bool(row.get("gate_allowed")),
            "gate_reason_codes": _json_field(row.get("gate_reason_codes_json"), []),
        }
        for row in _rows(
            """
            SELECT research_problem_id, gate_allowed, gate_reason_codes_json,
                   created_at
            FROM frontier_packets
            WHERE agenda_id=?
            ORDER BY created_at DESC, id DESC
            LIMIT 100
            """,
            (agenda_id,),
        )
    ]


def _job_events(agenda_id: int) -> list[dict]:
    events = []
    for row in _rows(
        """
        SELECT id, idea_id, stage, backend_kind, status, failure_reason,
               created_at, updated_at
        FROM compute_jobs_v1
        WHERE agenda_id=?
        ORDER BY updated_at DESC, id DESC
        LIMIT 100
        """,
        (agenda_id,),
    ):
        failure = str(row.get("failure_reason") or "")
        events.append(
            {
                "kind": "job",
                "at": _ts(row.get("updated_at") or row.get("created_at")),
                "job_id": row.get("id"),
                "idea_id": row.get("idea_id"),
                "stage": row.get("stage"),
                "backend_kind": row.get("backend_kind"),
                "status": row.get("status"),
                "failure_reason": _scrub_text(failure[:300]) if failure else None,
            }
        )
    return events


def _outcome_events(agenda_id: int) -> list[dict]:
    return [
        {
            "kind": "outcome",
            "at": _ts(row.get("recorded_at")),
            "idea_id": row.get("idea_id"),
            "run_id": row.get("experiment_run_id"),
            "execution_result": row.get("execution_result"),
            "effect": row.get("effect"),
            "baseline": row.get("baseline"),
            "verdict": row.get("verdict"),
            "state_decision": row.get("state_decision"),
        }
        for row in _rows(
            """
            SELECT idea_id, experiment_run_id, execution_result, effect,
                   baseline, verdict, state_decision, recorded_at
            FROM outcome_records
            WHERE agenda_id=?
            ORDER BY recorded_at DESC, id DESC
            LIMIT 100
            """,
            (agenda_id,),
        )
    ]


@blueprint.get("/agendas/<int:agenda_id>/timeline")
def agenda_timeline(agenda_id: int):
    """Chronological provenance feed for one agenda.

    direction -> agenda -> signals -> candidate decisions -> authorization ->
    jobs -> evidence transitions -> decisions/outcomes. Failures and refused
    gate transitions appear alongside successes on purpose.
    """
    limit = min(
        request.args.get("limit", _TIMELINE_DEFAULT_LIMIT, type=int) or _TIMELINE_DEFAULT_LIMIT,
        _TIMELINE_MAX_LIMIT,
    )
    events: list[dict] = []
    events.extend(_signal_events(agenda_id))
    events.extend(_candidate_events(agenda_id))
    events.extend(_grant_events(agenda_id))
    events.extend(_job_events(agenda_id))
    events.extend(_transition_events(agenda_id))
    events.extend(_decision_events(agenda_id))
    events.extend(_outcome_events(agenda_id))
    events.sort(key=lambda item: item.get("at") or "", reverse=True)
    return jsonify({"agenda_id": agenda_id, "events": events[:limit]})


def register_provenance_routes(app) -> None:
    app.register_blueprint(blueprint)
