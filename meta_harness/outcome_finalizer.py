"""Idempotently close terminal experiment grants into trusted outcomes.

The execution workers deliberately do not manufacture caller supplied usage or
metrics.  This reconciler waits until the durable compute and canonical attempt
ledgers are terminal, then asks :class:`MetaHarnessRepository` to assemble the
only permitted OutcomeRecord from persisted facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from db import database as db
from db.insight_outcomes import apply_experiment_finished_deep
from meta_harness.attempt_gpu_usage import GrantGPUUsageControl
from meta_harness.repository import MetaHarnessRepository


@dataclass
class OutcomeFinalizationReport:
    attempted: int = 0
    finalized: list[int] = field(default_factory=list)
    already_finalized: list[int] = field(default_factory=list)
    deferred: dict[int, str] = field(default_factory=dict)
    recovery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "finalized": self.finalized,
            "already_finalized": self.already_finalized,
            "deferred": self.deferred,
            "recovery": self.recovery,
        }


def _recover_terminal_usage() -> dict[str, int]:
    """Finish commits that may have been interrupted by controller loss."""
    from orchestrator.meta_compute_runtime import (
        settle_colab_request,
        settle_legacy_job,
    )

    control = GrantGPUUsageControl()
    legacy_ids = control.reconcile_terminal_attempts()
    colab_ids = control.reconcile_terminal_colab_attempts()
    legacy_settled = 0
    colab_settled = 0
    for job_id in legacy_ids:
        settle_legacy_job(job_id)
        legacy_settled += 1
    for request_id in colab_ids:
        settle_colab_request(request_id)
        colab_settled += 1
    return {
        "terminal_attempts_reconciled": len(legacy_ids),
        "terminal_compute_jobs_settled": legacy_settled,
        "terminal_colab_attempts_reconciled": len(colab_ids),
        "terminal_colab_jobs_settled": colab_settled,
        "orphan_unstarted_attempts_released": control.release_orphaned_reservations(),
        "prelaunch_blocked_attempts_released": (
            control.release_prelaunch_blocked_reservations()
        ),
    }


def _candidate_rows(limit: int) -> list[dict[str, Any]]:
    rows = db.fetchall(
        """
        SELECT er.id AS experiment_run_id, er.agenda_id, er.deep_insight_id,
               er.resource_grant_id, er.status AS run_status,
               er.hypothesis_verdict, arj.id AS auto_job_id,
               arj.status AS auto_job_status, arj.stage AS auto_job_stage,
               existing.id AS outcome_record_id
        FROM experiment_runs er
        JOIN resource_grants rg
          ON rg.id=er.resource_grant_id AND rg.agenda_id=er.agenda_id
        LEFT JOIN auto_research_jobs arj
          ON arj.agenda_id=er.agenda_id
         AND arj.deep_insight_id=er.deep_insight_id
        LEFT JOIN outcome_records existing
          ON existing.resource_grant_id=er.resource_grant_id
        WHERE er.resource_grant_id IS NOT NULL
          AND er.status IN ('completed','failed','cancelled')
          AND rg.status IN ('active','consumed')
          AND rg.stage IN ('pilot','validation','full_benchmark')
          AND (
                er.status='completed'
                OR (
                    arj.status IN ('completed','bundle_ready','failed','blocked')
                    AND arj.resource_grant_id=er.resource_grant_id
                    AND arj.experiment_run_id=er.id
                )
              )
          AND COALESCE(arj.stage, '') NOT IN (
                'retry_failed_run', 'gpu_failed'
              )
        ORDER BY er.completed_at ASC NULLS LAST, er.id ASC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    db.commit()
    return [dict(row) for row in rows]


def _mark_closed(row: dict[str, Any], outcome_id: int, verdict: str) -> None:
    agenda_id = int(row["agenda_id"])
    idea_id = int(row["deep_insight_id"])
    run_id = int(row["experiment_run_id"])
    grant_id = int(row["resource_grant_id"])
    note = (
        f"Trusted outcome_record={outcome_id} assembled automatically from "
        f"metered usage and persisted artifacts; verdict={verdict}."
    )
    db.execute(
        """
        UPDATE auto_research_jobs
        SET status='completed', stage='outcome_recorded', assigned_worker=NULL,
            last_error=NULL, last_note=?, updated_at=CURRENT_TIMESTAMP,
            last_checked_at=CURRENT_TIMESTAMP
        WHERE agenda_id=? AND deep_insight_id=?
          AND experiment_run_id=?
          AND resource_grant_id=?
        """,
        (note, agenda_id, idea_id, run_id, grant_id),
    )
    db.commit()
    apply_experiment_finished_deep(
        idea_id,
        verdict=verdict,
        success=verdict == "supported",
        inconclusive=verdict == "inconclusive",
    )
    db.emit_pipeline_event(
        "outcome_recorded",
        {
            "agenda_id": agenda_id,
            "deep_insight_id": idea_id,
            "experiment_run_id": run_id,
            "resource_grant_id": grant_id,
            "outcome_record_id": int(outcome_id),
            "verdict": verdict,
        },
        entity_type="outcome_record",
        entity_id=str(outcome_id),
        dedupe_key=f"outcome_recorded:{outcome_id}",
    )


def finalize_terminal_outcomes(*, limit: int = 50) -> OutcomeFinalizationReport:
    """Finalize every currently eligible run without advancing live work.

    A failed run is intentionally ignored while its auto-research job is
    queued for recovery.  Completed runs may close immediately; unsupported
    positive claims are conservatively downgraded by trusted outcome assembly
    until an independent scientific decision exists.
    """
    report = OutcomeFinalizationReport()
    try:
        report.recovery = _recover_terminal_usage()
    except Exception as exc:  # recovery remains retryable on the next timer tick
        db.rollback()
        report.recovery = {"error": f"{type(exc).__name__}: {exc}"}

    repository = MetaHarnessRepository()
    for row in _candidate_rows(limit):
        grant_id = int(row["resource_grant_id"])
        report.attempted += 1
        if int(row.get("outcome_record_id") or 0) > 0:
            outcome_id = int(row["outcome_record_id"])
            verdict_row = db.fetchone(
                "SELECT verdict FROM outcome_records WHERE id=?", (outcome_id,)
            ) or {}
            _mark_closed(row, outcome_id, str(verdict_row.get("verdict") or "inconclusive"))
            report.already_finalized.append(outcome_id)
            continue
        try:
            outcome_id = repository.assemble_and_record_outcome(
                resource_grant_id=grant_id,
                experiment_run_id=int(row["experiment_run_id"]),
            )
            outcome = db.fetchone(
                "SELECT verdict FROM outcome_records WHERE id=?", (int(outcome_id),)
            ) or {}
            verdict = str(outcome.get("verdict") or "inconclusive")
            _mark_closed(row, int(outcome_id), verdict)
            report.finalized.append(int(outcome_id))
        except Exception as exc:
            db.rollback()
            report.deferred[grant_id] = f"{type(exc).__name__}: {exc}"
    return report
