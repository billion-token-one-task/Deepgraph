"""Outcome enums, insight_events log, and helpers for the insight feedback loop."""

from __future__ import annotations

import json
import uuid
from typing import Any

from db import database as db

# Canonical outcome values (multi-stage lifecycle)
OUTCOME_PENDING = "pending"
OUTCOME_NOVELTY_REJECTED = "novelty_rejected"
OUTCOME_NOVELTY_PASSED = "novelty_passed"
OUTCOME_EXPERIMENT_QUEUED = "experiment_queued"
OUTCOME_EXPERIMENT_FAILED_SETUP = "experiment_failed_setup"
OUTCOME_EXPERIMENT_FAILED_RUN = "experiment_failed_run"
OUTCOME_EXPERIMENT_INCONCLUSIVE = "experiment_inconclusive"
OUTCOME_EXPERIMENT_SUCCEEDED = "experiment_succeeded"
OUTCOME_HUMAN_UPVOTED = "human_upvoted"
OUTCOME_HUMAN_DOWNVOTED = "human_downvoted"
OUTCOME_BECAME_MANUSCRIPT = "became_manuscript"
OUTCOME_SUBMITTED = "submitted"
OUTCOME_SCOOPED = "scooped"

ALL_OUTCOMES = frozenset(
    {
        OUTCOME_PENDING,
        OUTCOME_NOVELTY_REJECTED,
        OUTCOME_NOVELTY_PASSED,
        OUTCOME_EXPERIMENT_QUEUED,
        OUTCOME_EXPERIMENT_FAILED_SETUP,
        OUTCOME_EXPERIMENT_FAILED_RUN,
        OUTCOME_EXPERIMENT_INCONCLUSIVE,
        OUTCOME_EXPERIMENT_SUCCEEDED,
        OUTCOME_HUMAN_UPVOTED,
        OUTCOME_HUMAN_DOWNVOTED,
        OUTCOME_BECAME_MANUSCRIPT,
        OUTCOME_SUBMITTED,
        OUTCOME_SCOOPED,
    }
)


def _table_for_scope(scope: str) -> str:
    if scope == "insights":
        return "insights"
    if scope == "deep_insights":
        return "deep_insights"
    raise ValueError(f"invalid scope {scope!r}")


def append_event(
    scope: str,
    insight_id: int,
    to_outcome: str,
    *,
    from_outcome: str | None = None,
    reason: str | None = None,
    triggered_by: str = "system",
    meta: dict | None = None,
) -> int:
    """Append one row to insight_events (event-sourced history)."""
    eid = db.insert_returning_id(
        """INSERT INTO insight_events
           (scope, insight_id, from_outcome, to_outcome, reason, triggered_by, meta_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            scope,
            insight_id,
            from_outcome,
            to_outcome,
            reason,
            triggered_by,
            json.dumps(meta) if meta else None,
        ),
    )
    db.commit()
    return eid


def set_outcome(
    scope: str,
    insight_id: int,
    new_outcome: str,
    *,
    reason: str | None = None,
    triggered_by: str = "system",
    meta: dict | None = None,
) -> None:
    """Update current outcome snapshot + append insight_events row."""
    table = _table_for_scope(scope)
    row = db.fetchone(f"SELECT outcome FROM {table} WHERE id=?", (insight_id,))
    if not row:
        return
    old = row.get("outcome") or OUTCOME_PENDING
    if old == new_outcome:
        return
    if scope == "insights":
        db.execute(
            """UPDATE insights
               SET outcome=?, outcome_reason=?, outcome_updated_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (new_outcome, reason, insight_id),
        )
    else:
        db.execute(
            """UPDATE deep_insights
               SET outcome=?, outcome_reason=?, outcome_updated_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (new_outcome, reason, insight_id),
        )
    db.commit()
    append_event(
        scope,
        insight_id,
        new_outcome,
        from_outcome=old,
        reason=reason,
        triggered_by=triggered_by,
        meta=meta,
    )


def record_created(scope: str, insight_id: int, *, triggered_by: str = "pipeline") -> None:
    """First event after INSERT (outcome defaults to pending in schema)."""
    append_event(
        scope,
        insight_id,
        OUTCOME_PENDING,
        from_outcome=None,
        reason="insight_created",
        triggered_by=triggered_by,
        meta=None,
    )


def new_generation_run_id() -> str:
    return str(uuid.uuid4())


def record_harvester_run(
    pattern_name: str,
    candidate_count: int,
    execution_time_ms: int,
    *,
    query_hash: str | None = None,
    node_id: str | None = None,
    meta: dict | None = None,
) -> int:
    """Log one Signal Harvester pattern execution (yield analytics)."""
    hid = db.insert_returning_id(
        """INSERT INTO harvester_runs
           (pattern_name, query_hash, node_id, candidate_count, execution_time_ms, meta_json)
           VALUES (?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            pattern_name,
            query_hash,
            node_id,
            candidate_count,
            execution_time_ms,
            json.dumps(meta) if meta else None,
        ),
    )
    db.commit()
    return hid


def apply_novelty_verdict_to_deep_insight(
    insight_id: int,
    verdict: str,
    *,
    report_preview: str | None = None,
) -> None:
    """Map EvoScientist / verifier verdict string to outcome + event."""
    v = (verdict or "").strip().lower()
    if v in ("", "unchecked"):
        return
    if v == "novel":
        set_outcome(
            "deep_insights",
            insight_id,
            OUTCOME_NOVELTY_PASSED,
            reason=report_preview or "verifier: novel",
            triggered_by="verifier",
            meta={"novelty_verdict": verdict},
        )
    elif v in ("partially_exists", "partially exists"):
        set_outcome(
            "deep_insights",
            insight_id,
            OUTCOME_NOVELTY_PASSED,
            reason=report_preview or "partial overlap with prior work",
            triggered_by="verifier",
            meta={"novelty_verdict": verdict},
        )
    elif v == "exists":
        set_outcome(
            "deep_insights",
            insight_id,
            OUTCOME_NOVELTY_REJECTED,
            reason=report_preview or "verifier: exists",
            triggered_by="verifier",
            meta={"novelty_verdict": verdict},
        )


def apply_experiment_queued_deep(insight_id: int, *, note: str | None = None) -> None:
    set_outcome(
        "deep_insights",
        insight_id,
        OUTCOME_EXPERIMENT_QUEUED,
        reason=note or "experiment forged / queued",
        triggered_by="experiment",
    )


def apply_experiment_finished_deep(
    insight_id: int,
    *,
    verdict: str | None,
    success: bool,
    inconclusive: bool = False,
) -> None:
    if inconclusive or (verdict or "").lower() == "inconclusive":
        set_outcome(
            "deep_insights",
            insight_id,
            OUTCOME_EXPERIMENT_INCONCLUSIVE,
            reason=f"hypothesis_verdict={verdict}",
            triggered_by="experiment",
        )
    elif success:
        set_outcome(
            "deep_insights",
            insight_id,
            OUTCOME_EXPERIMENT_SUCCEEDED,
            reason=f"hypothesis_verdict={verdict}",
            triggered_by="experiment",
        )
    else:
        set_outcome(
            "deep_insights",
            insight_id,
            OUTCOME_EXPERIMENT_FAILED_RUN,
            reason=f"hypothesis_verdict={verdict}",
            triggered_by="experiment",
        )
