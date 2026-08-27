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
    # What the signal layer was told about each finalised outcome. Reported
    # rather than silent, so a feedback loop that stops running is visible in
    # the same place the outcome is.
    signal_feedback: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "finalized": self.finalized,
            "already_finalized": self.already_finalized,
            "deferred": self.deferred,
            "recovery": self.recovery,
            "signal_feedback": self.signal_feedback,
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
        "stranded_token_reservations_released": (
            _release_stranded_token_reservations()
        ),
        "evidence_state_advanced": _advance_settled_evidence_state(),
    }


# How long a token reservation may sit unsettled before it is presumed dead.
# Longer than any single LLM turn the forge or validation loop makes, so a slow
# call is never reclaimed out from under itself.
from meta_harness.grant_usage import GrantUsageLedger

_TOKEN_RESERVATION_LEASE_SECONDS = 2 * 3600


def _state_is_behind(recorded: str, target: str) -> bool:
    """True when a recorded evidence state sits earlier than the one reached.

    Never move a record backward, and never touch a retraction: an outcome
    withdrawn as unmeasurable is a decision about the science, not a stale
    view of the run.
    """
    from contracts.meta_harness import EVIDENCE_STATES

    if recorded == "unmeasurable_retracted":
        return False
    try:
        return EVIDENCE_STATES.index(recorded) < EVIDENCE_STATES.index(target)
    except ValueError:
        return recorded != target


def evidence_advance_plan(
    *,
    state: str,
    verdict: str,
    artifacts_present: int,
    artifacts_missing: Any,
    artifacts_hash: str,
    resource_grant_id: int,
    grant_stage: str = "",
    contract_hash: str = "",
) -> tuple[str, "EvidenceTransitionContext"] | None:
    """Decide the next evidence rung from facts no backend owns.

    This is the whole rule, in one place. It says nothing about Colab, ssh or
    cpu because none of them change what makes a measurement admissible: the
    run reached a verdict and its artifacts are registered. Callers that
    already hold the run pass it in; the finalizer reads it from the ledger.
    Returns None when nothing may advance.
    """
    from meta_harness.repository import EvidenceTransitionContext

    if str(verdict or "").strip().lower() not in {
        "supported",
        "refuted",
        "inconclusive",
    }:
        return None
    if artifacts_present <= 0 or artifacts_missing:
        return None
    current = str(state or "planned")
    if current == "planned":
        return "sanity_passed", EvidenceTransitionContext(
            resource_grant_valid=True,
            resource_grant_id=int(resource_grant_id),
            execution_succeeded=True,
            pilot_only=True,
            raw_artifacts_present=True,
            raw_artifacts_hash=artifacts_hash,
        )
    if current == "sanity_passed":
        if str(grant_stage or "") != "full_benchmark" or not contract_hash:
            return None
        return "full_benchmark_complete", EvidenceTransitionContext(
            resource_grant_valid=True,
            resource_grant_id=int(resource_grant_id),
            execution_succeeded=True,
            pilot_only=False,
            full_benchmark_complete=True,
            raw_artifacts_present=True,
            raw_artifacts_hash=artifacts_hash,
            benchmark_contract_hash=contract_hash,
        )
    return None


def _advance_settled_evidence_state(
    limit: int = 50, run_id: int | None = None
) -> dict[str, int]:
    """Advance any run whose compute finished, whatever ran it.

    A GPU is a GPU. The evidence ladder has no business knowing whether the
    accelerator was reached over Colab's CLI or over ssh, yet the transition
    lived three times in two files with three different guard sets: colab in
    colab_worker (both rungs), legacy/ssh in meta_compute_runtime (first rung
    only), and cpu nowhere at all. So a pilot measured on ssh_gpu recorded a
    real two-arm result and sat at 'planned' forever -- every run in this
    repository that ever produced supported or refuted went through Colab --
    and each new rented accelerator would have needed a fourth copy.

    One implementation, keyed on what actually matters: the compute job
    succeeded and the runner's artifacts are all registered. Guards are the
    union of the three it replaces, so nothing that used to be refused is now
    admitted: one rung per pass, artifacts complete, and the upper rung
    additionally requires a full_benchmark-stage grant and the contract hash
    the preflight locked.
    """
    from orchestrator.bounded_execution import raw_artifacts_hash
    from meta_harness.repository import (
        EvidenceTransitionContext,
        MetaHarnessRepository,
    )

    counts = {"sanity_passed": 0, "full_benchmark_complete": 0, "refused": 0}
    # Deliberately says nothing about compute jobs: whether one exists, and
    # what its command_ref looks like, is itself a backend detail. Colab keys
    # them as colab-work-request:N, ssh as experiment-run:N, and a cpu run
    # creates none at all -- a first attempt at this query keyed on
    # command_ref and would have silently cut Colab out of its own ladder.
    # What the science needs is true of every backend: the run finished, it
    # reached a verdict, and its artifacts are registered.
    rows = db.fetchall(
        """
        SELECT er.id AS run_id, er.agenda_id,
               COALESCE(er.scientific_evidence_state, 'planned') AS state,
               er.resource_grant_id AS grant_id,
               er.hypothesis_verdict AS verdict
          FROM experiment_runs er
         WHERE er.status = 'completed'
           AND COALESCE(er.scientific_evidence_state, 'planned')
               IN ('planned', 'sanity_passed')
           AND LOWER(COALESCE(er.hypothesis_verdict, ''))
               IN ('supported', 'refuted', 'inconclusive')
           AND (CAST(? AS INTEGER) IS NULL OR er.id = ?)
         ORDER BY er.id ASC
         LIMIT ?
        """,
        (run_id, run_id, int(limit)),
    )
    for row in rows:
        record = dict(row)
        run_id = int(record["run_id"])
        agenda_id = int(record["agenda_id"])
        state = str(record["state"])
        try:
            digest, present, missing = raw_artifacts_hash(
                agenda_id=agenda_id, experiment_run_id=run_id
            )
        except Exception:  # noqa: BLE001 - a bookkeeping gap must not raise
            db.rollback()
            counts["refused"] += 1
            continue
        if present <= 0 or missing:
            counts["refused"] += 1
            continue
        grant_stage = ""
        contract_hash = ""
        if state == "sanity_passed":
            # Only the upper rung needs the grant, so only it reads one.
            grant = dict(
                db.fetchone(
                    "SELECT stage, preflight_result_id FROM resource_grants WHERE id=?",
                    (int(record["grant_id"] or 0),),
                )
                or {}
            )
            grant_stage = str(grant.get("stage") or "")
            contract_row = db.fetchone(
                """
                SELECT cer.requirements_hash
                  FROM candidate_preflight_results_v1 cpr
                  JOIN candidate_execution_requirements_v1 cer
                    ON cer.id = cpr.requirement_id
                 WHERE cpr.id = ?
                """,
                (int(grant.get("preflight_result_id") or 0),),
            )
            contract_hash = str(
                dict(contract_row or {}).get("requirements_hash") or ""
            )
        plan = evidence_advance_plan(
            state=state,
            verdict=str(record.get("verdict") or ""),
            artifacts_present=present,
            artifacts_missing=missing,
            artifacts_hash=digest,
            resource_grant_id=int(record["grant_id"] or 0),
            grant_stage=grant_stage,
            contract_hash=contract_hash,
        )
        if plan is None:
            counts["refused"] += 1
            continue
        target, context = plan
        try:
            MetaHarnessRepository().advance_experiment_state(
                agenda_id=agenda_id,
                experiment_run_id=run_id,
                target=target,
                context=context,
                actor="settled_compute_handoff_v1",
            )
            counts[target] += 1
            # An OutcomeRecord stamps state_decision from the run at assembly
            # time, assembly is idempotent per (grant, run), and
            # advance_to_full_benchmark requires the outcome and the run to
            # agree. On the colab path the transition ran before assembly so
            # they agreed by accident of ordering; advancing afterwards left
            # outcomes 200 and 202 reading 'planned' against runs that had
            # reached sanity_passed, and neither candidate could be funded for
            # the benchmark that would turn idea 241's measured -0.315 into a
            # directional verdict. state_decision is a derived view of the
            # run's evidence state, not a measurement, so bring it forward --
            # never backward, and never onto a retracted record.
            # Only the newest record, and only when it is behind: run 264
            # carries one outcome per rung (sanity_passed,
            # full_benchmark_complete, scientifically_decided) and rewriting
            # them all would flatten the ladder's own history.
            latest = db.fetchone(
                "SELECT id, state_decision FROM outcome_records"
                " WHERE experiment_run_id=? ORDER BY id DESC LIMIT 1",
                (run_id,),
            )
            latest_row = dict(latest or {})
            recorded = str(latest_row.get("state_decision") or "planned")
            if latest_row and _state_is_behind(recorded, target):
                db.execute(
                    "UPDATE outcome_records SET state_decision=? WHERE id=?",
                    (target, int(latest_row["id"])),
                )
                db.commit()
        except Exception:  # noqa: BLE001 - never undo a settled compute job
            db.rollback()
            counts["refused"] += 1
    return counts


def _release_stranded_token_reservations() -> int:
    """Return token budget held by reservations whose work already died.

    GrantUsage.release is only called on the normal completion path, and the
    orphan sweep next to it covers experiment_attempt_gpu_reservations_v1 --
    GPU hours, not tokens. So when a run dies mid-call the token reservation
    is never released: idea 241's validation_code_iteration was holding 17,275
    of a 40,000 token grant on 2026-08-26, reserved at 20:49:46 and orphaned
    twenty seconds later when the staleness sweep reclaimed run 266. The
    candidate then failed every forge with "ResourceGrant token budget is
    exhausted" while 43% of its budget was held by a dead call, and
    attempt_key refuses to retry an operation that still has an open
    reservation, so it could not even try again.

    Only reclaim when the lease has expired AND the grant has no live work:
    an unfinished compute job or a run that has not reached a terminal state
    means the call may still be in flight.
    """
    rows = db.fetchall(
        """
        SELECT u.id, u.resource_grant_id, u.operation, u.token_reserved
          FROM resource_grant_usage_reservations u
         WHERE u.status='reserved'
           AND u.created_at <= CURRENT_TIMESTAMP - CAST(? AS INTERVAL)
           AND NOT EXISTS (
                 SELECT 1 FROM compute_jobs_v1 cj
                  WHERE cj.resource_grant_id = u.resource_grant_id
                    AND cj.status NOT IN (
                          'succeeded', 'failed', 'cancelled', 'canceled',
                          'timed_out', 'usage_unknown', 'submission_unknown'
                    )
           )
           AND NOT EXISTS (
                 SELECT 1 FROM experiment_runs er
                  WHERE er.resource_grant_id = u.resource_grant_id
                    AND COALESCE(er.status, '') NOT IN ('completed', 'failed')
           )
        """,
        ("%d seconds" % _TOKEN_RESERVATION_LEASE_SECONDS,),
    )
    released = 0
    for row in rows:
        record = dict(row)
        try:
            GrantUsageLedger(int(record["resource_grant_id"])).release(
                int(record["id"]),
                reason="lease_expired_no_live_work:%s" % str(record.get("operation"))[:60],
            )
            released += 1
        except Exception:  # noqa: BLE001 - a stuck row must not stop the sweep
            db.rollback()
    return released


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
          -- A queued or running job on an active grant is pending work, not
          -- a terminal state to settle: the first full_benchmark grant in
          -- the repo's history (grant 78, 2026-08-18) was assembled into an
          -- outcome and closed before the completion mode could claim it.
          -- One narrow exception: a gpu_scheduler job still parked at
          -- queued_gpu after its run completed and its compute went terminal
          -- is not pending, it is frozen -- job 145 squatted the single
          -- execution slot this way and iced the whole candidate funnel
          -- (run 171 done, outcome unrecordable, slot never released).
          AND NOT (
                COALESCE(arj.status, '') IN (
                    'queued', 'running_gpu', 'running_cpu', 'review_pending',
                    'eligible', 'queued_gpu', 'harness_required'
                )
                AND NOT (
                    arj.status='queued_gpu'
                    AND arj.stage='gpu_scheduler'
                    AND er.status='completed'
                    AND NOT EXISTS (
                        SELECT 1 FROM colab_work_requests_v1 c
                        WHERE c.experiment_run_id=er.id
                          AND c.status IN ('queued', 'admitting', 'running')
                    )
                )
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



def _feed_signal_posterior(outcome_id: int, verdict: str) -> str:
    """Tell the signal layer what its lead produced.

    The evidence graph proposes research openings, and `agenda_signal_outcomes`
    is where the system learns which kinds of opening pay off -- an author's
    stated open question against a performance plateau against a claim-method
    gap. The writer for that table hangs off the older knowledge-loop path,
    which the meta-harness execution route never calls, so after 150 outcomes
    the table still held zero rows and every signal posterior was its prior.
    A search that cannot see which of its leads worked cannot improve, which is
    the whole premise of the harness.

    Failures here are reported, never raised: a bookkeeping gap must not undo a
    settled outcome.
    """
    if verdict not in ("supported", "refuted"):
        return "skipped_non_directional_verdict"
    row = db.fetchone(
        """
        SELECT o.agenda_id, o.idea_id, o.experiment_run_id, o.effect,
               di.source_signal_refs
          FROM outcome_records o
          LEFT JOIN deep_insights di ON di.id = o.idea_id
         WHERE o.id = ?
        """,
        (int(outcome_id),),
    )
    if not row:
        return "outcome_not_found"
    refs = dict(row).get("source_signal_refs")
    if not refs:
        return "no_signal_provenance"
    try:
        from agents.problem_first import update_signal_posterior

        updates = update_signal_posterior(
            refs,
            "confirmed" if verdict == "supported" else "refuted",
            agenda_id=int(dict(row)["agenda_id"]),
            run_id=dict(row).get("experiment_run_id"),
            experimental_claim_id=None,
            effect_size=dict(row).get("effect"),
            p_value=None,
            conditions={"source": "outcome_finalizer", "outcome_id": int(outcome_id)},
        )
        return "updated_%d_signals" % len(updates)
    except Exception as exc:                       # never undo a settled outcome
        return "%s: %s" % (type(exc).__name__, str(exc)[:120])


def finalize_terminal_outcomes(*, limit: int = 50) -> OutcomeFinalizationReport:
    """Finalize every currently eligible run without advancing live work.

    A failed run is intentionally ignored while its auto-research job is
    queued for recovery.  Completed runs may close immediately; unsupported
    positive claims are conservatively downgraded by trusted outcome assembly
    until an independent scientific decision exists.
    """
    report = OutcomeFinalizationReport()
    try:
        # A job left in an execution status whose grant already carries an
        # OutcomeRecord is bookkeeping debt, not live work: job 137 sat at
        # queued_gpu for eight hours after its transport died and its outcome
        # was recorded, occupying the launcher's single execution slot -- and
        # the pending-work guard above now protects exactly that zombie from
        # ever being closed. Only rows with no live colab request close here.
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='completed', stage='outcome_recorded',
                assigned_worker=NULL, updated_at=CURRENT_TIMESTAMP,
                last_note='Outcome already recorded for this grant; execution-state job closed by reconciliation.'
            WHERE status IN ('queued_gpu', 'running_gpu', 'running_cpu',
                             'running_experiment')
              AND resource_grant_id IN (
                  SELECT resource_grant_id FROM outcome_records
              )
              AND NOT EXISTS (
                  SELECT 1 FROM colab_work_requests_v1 c
                  WHERE c.resource_grant_id=auto_research_jobs.resource_grant_id
                    AND c.status IN ('queued', 'running')
              )
            """
        )
        # The complementary deadlock: an execution-state job whose transport
        # died WITHOUT an outcome could neither finalize (the pending-work
        # guard excludes execution statuses) nor close (no outcome exists).
        # Job 137 orbited that circle for nine hours. Surrender it to the
        # normal failed-run finalization path instead.
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='failed', stage='experiment_failed',
                assigned_worker=NULL, updated_at=CURRENT_TIMESTAMP,
                last_error='colab transport terminal with no outcome; surrendered for finalization',
                experiment_run_id=COALESCE(
                    experiment_run_id,
                    (SELECT MAX(er.id) FROM experiment_runs er
                     WHERE er.resource_grant_id=auto_research_jobs.resource_grant_id)
                )
            WHERE status IN ('queued_gpu', 'running_gpu', 'running_cpu',
                             'running_experiment')
              AND resource_grant_id IS NOT NULL
              AND resource_grant_id NOT IN (
                  SELECT resource_grant_id FROM outcome_records
              )
              AND EXISTS (
                  SELECT 1 FROM colab_work_requests_v1 c
                  WHERE c.resource_grant_id=auto_research_jobs.resource_grant_id
                    AND c.status IN ('failed', 'timed_out', 'cancelled')
              )
              AND NOT EXISTS (
                  SELECT 1 FROM colab_work_requests_v1 c2
                  WHERE c2.resource_grant_id=auto_research_jobs.resource_grant_id
                    AND c2.status IN ('queued', 'running')
              )
              -- A grant that produced a completed run has a PENDING outcome,
              -- not a missing one: an earlier failed request on the same
              -- grant must not condemn the successful retry that followed.
              -- Job 140 was surrendered this way while its run 180 held a
              -- verified measurement (2026-08-19).
              AND NOT EXISTS (
                  SELECT 1 FROM experiment_runs er2
                  WHERE er2.resource_grant_id=auto_research_jobs.resource_grant_id
                    AND er2.status='completed'
              )
            """
        )
        db.commit()
    except Exception:
        db.rollback()
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
            report.signal_feedback[int(outcome_id)] = _feed_signal_posterior(
                int(outcome_id), verdict)
            report.finalized.append(int(outcome_id))
        except Exception as exc:
            db.rollback()
            report.deferred[grant_id] = f"{type(exc).__name__}: {exc}"
    return report
