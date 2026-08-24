"""Execute exactly one portfolio-granted candidate, without global autonomy.

The meta-harness could issue a ResourceGrant and park the job at
``portfolio_granted``, but nothing in the codebase ever read that stage back:
the authorization was written and never consumed. Turning on global autonomy
did not fix that -- it only started every other loop -- so the recovery runbook
step "give the winning candidate one small, short CPU/LLM ResourceGrant and run
it" had no executor at all.

This module is that executor, and deliberately nothing more:

* it runs **one** candidate, named explicitly by (job, agenda, idea, grant) --
  there is no discovery query, no backlog, and no loop;
* it never reads or writes ``DEEPGRAPH_AUTO_RESEARCH_ENABLED`` /
  ``DEEPGRAPH_AUTO_PIPELINE_ENABLED``, so it cannot be a back door into global
  autonomy;
* it refuses any backend outside the grant's own allowlist and any allowlist
  wider than CPU/LLM, so a bounded pilot can never reach for GPUs;
* the experiment itself is built and run by the existing reviewed machinery
  (``forge_experiment`` then ``run_validation_loop``). Nothing here duplicates
  run creation, state authority, or budget accounting;
* every persisted run is sent through formal OutcomeRecord settlement; an
  unused grant is formally revoked, and any unresolved metering is exposed as
  ``pilot_settlement_required`` instead of being mistaken for completion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from contracts.meta_harness import EVIDENCE_STATES, ResourceGrant
from db import database as db
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.grants import ResourceRequest, authorize
from meta_harness.repository import MetaHarnessRepository


# A bounded pilot is a CPU/LLM errand. Anything wider needs its own grant and
# its own decision; widening it here would silently reintroduce GPU spend.
BOUNDED_BACKENDS = frozenset({"cpu", "llm"})
BOUNDED_STAGE = "pilot"
GRANTED_STAGE = "portfolio_granted"
RUNNING_STAGE = "pilot_running"
DONE_STAGE = "pilot_outcome_recorded"
FAILED_STAGE = "pilot_failed"
SETTLEMENT_REQUIRED_STAGE = "pilot_settlement_required"
WITHDRAWN_JOB_STAGES = frozenset({"resource_grant_revoked", "resource_grant_expired"})


class BoundedExecutionError(RuntimeError):
    """Raised when the single-candidate contract cannot be honoured."""


@dataclass(frozen=True)
class BoundedExecutionRequest:
    agenda_id: int
    idea_id: int
    resource_grant_id: int
    job_id: int = 0

    def validate(self) -> None:
        for name in ("agenda_id", "idea_id", "resource_grant_id", "job_id"):
            if int(getattr(self, name)) <= 0:
                raise BoundedExecutionError(f"{name} must be positive")


@dataclass
class BoundedExecutionResult:
    status: str
    agenda_id: int
    idea_id: int
    resource_grant_id: int
    job_id: int | None = None
    experiment_run_id: int | None = None
    outcome_record_id: int | None = None
    evidence_state: str | None = None
    verdict: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "agenda_id": self.agenda_id,
            "idea_id": self.idea_id,
            "resource_grant_id": self.resource_grant_id,
            "job_id": self.job_id,
            "experiment_run_id": self.experiment_run_id,
            "outcome_record_id": self.outcome_record_id,
            "evidence_state": self.evidence_state,
            "verdict": self.verdict,
            "reason": self.reason,
            "details": dict(self.details),
        }


def _load_list(value: Any) -> list[str]:
    try:
        loaded = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in loaded] if isinstance(loaded, list) else []


def _grant_from_row(row: dict[str, Any]) -> ResourceGrant:
    """Rebuild the contract object so the shared admission check can run."""
    return ResourceGrant(
        agenda_id=int(row.get("agenda_id") or 0),
        idea_id=int(row.get("idea_id") or 0),
        decision_packet_id=int(row.get("decision_packet_id") or 0),
        stage=str(row.get("stage") or ""),
        token_cap=int(row.get("token_cap") or 0),
        gpu_class=str(row.get("gpu_class") or "none"),
        max_gpu_hours=float(row.get("max_gpu_hours") or 0.0),
        backend_allowlist=_load_list(row.get("backend_allowlist_json")),
        artifact_requirements=_load_list(row.get("artifact_requirements_json")),
        expires_at=str(row.get("expires_at") or ""),
        grant_reason=str(row.get("grant_reason") or ""),
        idempotency_key=str(row.get("idempotency_key") or ""),
        status=str(row.get("status") or ""),
        grant_id=int(row.get("id") or 0),
        reservation_id=int(row.get("reservation_id") or 0),
        preflight_result_id=(
            int(row["preflight_result_id"])
            if row.get("preflight_result_id")
            else None
        ),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def raw_artifacts_hash(*, agenda_id: int, experiment_run_id: int) -> tuple[str, int, int]:
    """Hash what this run actually produced, rows and file bytes together.

    Deliberately independent of the retrospective reviewer's reconstruction
    hash: that one re-derives a digest for evidence recorded before the ladder
    existed, while this one is computed at the moment the artifacts are
    written. Two different provenance claims should not share one definition.

    Returns ``(digest, files_present, files_missing)``. A run whose artifact
    rows all point at absent files hashes to something stable but is reported
    as empty, so the caller can refuse to advance on it.
    """
    rows = db.fetchall(
        """
        SELECT id, artifact_type, path, metric_key, metric_value
        FROM experiment_artifacts
        WHERE agenda_id=? AND run_id=?
        ORDER BY id
        """,
        (int(agenda_id), int(experiment_run_id)),
    )
    digest = hashlib.sha256()
    digest.update(b"deepgraph:bounded-pilot:raw-artifacts:v1\n")
    present = missing = 0
    for row in rows:
        path = Path(str(row.get("path") or ""))
        digest.update(
            _canonical_json(
                {
                    "id": row.get("id"),
                    "type": row.get("artifact_type"),
                    "metric_key": row.get("metric_key"),
                    "metric_value": row.get("metric_value"),
                    "path_name": path.name,
                }
            ).encode("utf-8")
        )
        try:
            if path.is_file() and not path.is_symlink():
                digest.update(path.read_bytes())
                present += 1
            else:
                digest.update(b"<missing>")
                missing += 1
        except OSError:
            digest.update(b"<unreadable>")
            missing += 1
    return digest.hexdigest(), present, missing


def _authorize_bounded_grant(
    request: BoundedExecutionRequest,
) -> tuple[ResourceGrant, dict[str, Any]]:
    row = db.fetchone(
        "SELECT * FROM resource_grants WHERE id=?",
        (request.resource_grant_id,),
    )
    if not row:
        raise BoundedExecutionError("resource_grant_not_found")
    grant = _grant_from_row(dict(row))
    if grant.stage != BOUNDED_STAGE:
        raise BoundedExecutionError(
            f"bounded execution requires a '{BOUNDED_STAGE}' grant, "
            f"not '{grant.stage}'"
        )
    backends = {value.strip().lower() for value in grant.backend_allowlist}
    if not backends:
        raise BoundedExecutionError("grant_backend_allowlist_empty")
    if not backends.issubset(BOUNDED_BACKENDS):
        raise BoundedExecutionError(
            "bounded execution refuses backends outside cpu/llm: "
            + ",".join(sorted(backends - BOUNDED_BACKENDS))
        )
    if backends != BOUNDED_BACKENDS:
        raise BoundedExecutionError(
            "bounded execution grant must authorize exactly cpu/llm"
        )
    if grant.max_gpu_hours > 0:
        raise BoundedExecutionError("bounded execution refuses a GPU-hour grant")
    if str(grant.gpu_class or "none").strip().lower() not in {"none", "cpu"}:
        raise BoundedExecutionError("bounded execution refuses a GPU-class grant")
    requirements = {value.strip().lower() for value in grant.artifact_requirements}
    if "final_results" not in requirements:
        raise BoundedExecutionError(
            "bounded execution grant must require final_results"
        )
    # Check every backend the grant actually carries, so an expired or
    # out-of-scope grant is reported with the shared reason codes rather than a
    # bespoke message.
    for backend in sorted(backends):
        authorize(
            grant,
            ResourceRequest(
                agenda_id=request.agenda_id,
                idea_id=request.idea_id,
                stage=BOUNDED_STAGE,
                backend=backend,
                resource_grant_id=request.resource_grant_id,
                token_cap=grant.token_cap,
            ),
        )
    return grant, dict(row)


def _load_bounded_grant(
    request: BoundedExecutionRequest,
) -> tuple[ResourceGrant, dict[str, Any]]:
    """Scope-check a pilot grant without authorizing new execution.

    A replay after ``record_outcome`` must inspect a consumed grant so it can
    finish the job transition, but it must never be allowed to execute again.
    The normal authorization path remains the only path into forge/validation.
    """

    row = db.fetchone(
        "SELECT * FROM resource_grants WHERE id=?",
        (request.resource_grant_id,),
    )
    if not row:
        raise BoundedExecutionError("resource_grant_not_found")
    grant = _grant_from_row(dict(row))
    if grant.agenda_id != request.agenda_id or grant.idea_id != request.idea_id:
        raise BoundedExecutionError("resource_grant_scope_mismatch")
    if grant.stage != BOUNDED_STAGE:
        raise BoundedExecutionError(
            f"bounded execution requires a '{BOUNDED_STAGE}' grant, "
            f"not '{grant.stage}'"
        )
    backends = {value.strip().lower() for value in grant.backend_allowlist}
    if not backends:
        raise BoundedExecutionError("grant_backend_allowlist_empty")
    if not backends.issubset(BOUNDED_BACKENDS):
        raise BoundedExecutionError(
            "bounded execution refuses backends outside cpu/llm: "
            + ",".join(sorted(backends - BOUNDED_BACKENDS))
        )
    if backends != BOUNDED_BACKENDS:
        raise BoundedExecutionError(
            "bounded execution grant must authorize exactly cpu/llm"
        )
    if grant.max_gpu_hours > 0:
        raise BoundedExecutionError("bounded execution refuses a GPU-hour grant")
    if str(grant.gpu_class or "none").strip().lower() not in {"none", "cpu"}:
        raise BoundedExecutionError("bounded execution refuses a GPU-class grant")
    requirements = {value.strip().lower() for value in grant.artifact_requirements}
    if "final_results" not in requirements:
        raise BoundedExecutionError(
            "bounded execution grant must require final_results"
        )
    return grant, dict(row)


def _run_bound_to_grant(request: BoundedExecutionRequest) -> int:
    """The newest experiment run this grant produced, or 0.

    Asked of the database rather than of the forge's return value: the forge
    creates the run before the review stages that can reject it, so a failed
    forge can still leave a real run behind holding real metered spend.
    """
    row = db.fetchone(
        """
        SELECT id FROM experiment_runs
        WHERE agenda_id=? AND deep_insight_id=? AND resource_grant_id=?
        ORDER BY id DESC
        """,
        (request.agenda_id, request.idea_id, request.resource_grant_id),
    )
    return int((row or {}).get("id") or 0)


def _load_run(request: BoundedExecutionRequest, run_id: int) -> dict[str, Any]:
    row = db.fetchone(
        """
        SELECT id, agenda_id, deep_insight_id, status, resource_grant_id,
               scientific_evidence_state, resource_class, hypothesis_verdict
        FROM experiment_runs
        WHERE id=? AND agenda_id=? AND deep_insight_id=?
        """,
        (int(run_id), request.agenda_id, request.idea_id),
    )
    if not row or int(row.get("resource_grant_id") or 0) != request.resource_grant_id:
        raise BoundedExecutionError("run_not_bound_to_grant")
    return dict(row)


def _existing_outcome(
    request: BoundedExecutionRequest,
) -> dict[str, Any] | None:
    row = db.fetchone(
        """
        SELECT id, agenda_id, idea_id, resource_grant_id, experiment_run_id,
               execution_result, verdict, state_decision
        FROM outcome_records
        WHERE resource_grant_id=? AND agenda_id=? AND idea_id=?
        ORDER BY id DESC LIMIT 1
        """,
        (request.resource_grant_id, request.agenda_id, request.idea_id),
    )
    return dict(row) if row else None


def _load_job(request: BoundedExecutionRequest) -> dict[str, Any]:
    row = db.fetchone(
        """
        SELECT id, agenda_id, deep_insight_id, status, stage, resource_grant_id,
               experiment_run_id
        FROM auto_research_jobs
        WHERE id=? AND agenda_id=? AND deep_insight_id=? AND resource_grant_id=?
        """,
        (
            int(request.job_id),
            request.agenda_id,
            request.idea_id,
            request.resource_grant_id,
        ),
    )
    if not row:
        raise BoundedExecutionError("granted_job_not_found")
    return dict(row)


def _claim_job(request: BoundedExecutionRequest) -> dict[str, Any]:
    """Claim a new exact job, or recognize its bounded replay states."""

    job = _load_job(request)
    pair = (str(job.get("status") or ""), str(job.get("stage") or ""))
    replayable = {
        ("running_experiment", RUNNING_STAGE),
        ("blocked", SETTLEMENT_REQUIRED_STAGE),
        ("failed", FAILED_STAGE),
        ("completed", DONE_STAGE),
    } | {("blocked", stage) for stage in WITHDRAWN_JOB_STAGES}
    if pair in replayable:
        job["claim_mode"] = "replay"
        return job
    if pair != ("queued", GRANTED_STAGE):
        raise BoundedExecutionError(
            f"job is at non-replayable state status={pair[0]!r} stage={pair[1]!r}"
        )
    cursor = db.execute(
        """
        UPDATE auto_research_jobs
        SET status='running_experiment', stage=?, last_error=NULL,
            last_note=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=? AND deep_insight_id=?
          AND resource_grant_id=? AND stage=? AND status='queued'
        """,
        (
            RUNNING_STAGE,
            "bounded pilot claimed by operator-invoked execution path",
            int(job["id"]),
            request.agenda_id,
            request.idea_id,
            request.resource_grant_id,
            GRANTED_STAGE,
        ),
    )
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        db.rollback()
        raise BoundedExecutionError("granted_job_already_claimed")
    db.commit()
    job["claim_mode"] = "claimed"
    return dict(job)


def _mark_job_failed(
    *,
    request: BoundedExecutionRequest,
    reason: str,
    experiment_run_id: int | None = None,
    settlement_required: bool = False,
) -> None:
    """CAS one exact claimed job into a truthful non-success terminal state."""

    status = "blocked" if settlement_required else "failed"
    stage = SETTLEMENT_REQUIRED_STAGE if settlement_required else FAILED_STAGE
    current = _load_job(request)
    current_pair = (
        str(current.get("status") or ""),
        str(current.get("stage") or ""),
    )
    target_pair = (status, stage)
    current_run_id = int(current.get("experiment_run_id") or 0)
    expected_run_id = int(experiment_run_id or 0)
    if current_pair == target_pair:
        if current_run_id not in {0, expected_run_id}:
            raise BoundedExecutionError("exact job failure replay run mismatch")
        return
    if current_pair == ("completed", DONE_STAGE):
        raise BoundedExecutionError("refusing to downgrade a completed exact job")
    allowed = {
        ("running_experiment", RUNNING_STAGE),
        ("blocked", SETTLEMENT_REQUIRED_STAGE),
        ("failed", FAILED_STAGE),
    }
    if current_pair not in allowed:
        raise BoundedExecutionError(
            "exact job cannot enter failure state from "
            f"status={current_pair[0]!r} stage={current_pair[1]!r}"
        )
    cursor = db.execute(
        """
        UPDATE auto_research_jobs
        SET status=?, stage=?, last_error=?, experiment_run_id=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=? AND deep_insight_id=? AND resource_grant_id=?
          AND status=? AND stage=?
        """,
        (
            status,
            stage,
            reason[:1000],
            experiment_run_id,
            int(request.job_id or 0),
            request.agenda_id,
            request.idea_id,
            request.resource_grant_id,
            current_pair[0],
            current_pair[1],
        ),
    )
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        db.rollback()
        latest = _load_job(request)
        latest_pair = (
            str(latest.get("status") or ""),
            str(latest.get("stage") or ""),
        )
        latest_run_id = int(latest.get("experiment_run_id") or 0)
        if latest_pair == target_pair and latest_run_id in {0, expected_run_id}:
            return
        raise BoundedExecutionError("exact job failure transition did not match")
    db.commit()


def _settle_job(
    *,
    request: BoundedExecutionRequest,
    experiment_run_id: int,
    note: str,
) -> None:
    """Mark success only after a durable successful OutcomeRecord exists."""

    current = _load_job(request)
    current_pair = (
        str(current.get("status") or ""),
        str(current.get("stage") or ""),
    )
    current_run_id = int(current.get("experiment_run_id") or 0)
    if current_pair == ("completed", DONE_STAGE):
        if current_run_id != int(experiment_run_id):
            raise BoundedExecutionError("completed exact job run mismatch")
        return
    if current_pair not in {
        ("running_experiment", RUNNING_STAGE),
        ("blocked", SETTLEMENT_REQUIRED_STAGE),
    }:
        raise BoundedExecutionError(
            "exact job cannot complete from "
            f"status={current_pair[0]!r} stage={current_pair[1]!r}"
        )
    cursor = db.execute(
        """
        UPDATE auto_research_jobs
        SET status='completed', stage=?, experiment_run_id=?, last_error=NULL,
            last_note=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=? AND deep_insight_id=? AND resource_grant_id=?
          AND status=? AND stage=?
        """,
        (
            DONE_STAGE,
            int(experiment_run_id),
            note[:1000],
            int(request.job_id or 0),
            request.agenda_id,
            request.idea_id,
            request.resource_grant_id,
            current_pair[0],
            current_pair[1],
        ),
    )
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        db.rollback()
        latest = _load_job(request)
        if (
            str(latest.get("status") or "") == "completed"
            and str(latest.get("stage") or "") == DONE_STAGE
            and int(latest.get("experiment_run_id") or 0)
            == int(experiment_run_id)
        ):
            return
        raise BoundedExecutionError("exact job settlement transition did not match")
    db.commit()


def _attach_run_to_job(
    request: BoundedExecutionRequest,
    *,
    experiment_run_id: int,
) -> None:
    current = _load_job(request)
    current_pair = (
        str(current.get("status") or ""),
        str(current.get("stage") or ""),
    )
    bound_run_id = int(current.get("experiment_run_id") or 0)
    if bound_run_id:
        if bound_run_id != int(experiment_run_id):
            raise BoundedExecutionError("exact job is already bound to another run")
        return
    if current_pair != ("running_experiment", RUNNING_STAGE):
        raise BoundedExecutionError("exact job is not claimable for run attachment")
    cursor = db.execute(
        """
        UPDATE auto_research_jobs
        SET experiment_run_id=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=? AND deep_insight_id=? AND resource_grant_id=?
          AND status='running_experiment' AND stage=?
          AND experiment_run_id IS NULL
        """,
        (
            int(experiment_run_id),
            int(request.job_id or 0),
            request.agenda_id,
            request.idea_id,
            request.resource_grant_id,
            RUNNING_STAGE,
        ),
    )
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        db.rollback()
        latest = _load_job(request)
        if int(latest.get("experiment_run_id") or 0) == int(experiment_run_id):
            return
        raise BoundedExecutionError("exact job run attachment did not match")
    db.commit()


def _real_final_results_present(
    *, agenda_id: int, experiment_run_id: int
) -> bool:
    rows = db.fetchall(
        """
        SELECT path FROM experiment_artifacts
        WHERE agenda_id=? AND run_id=? AND artifact_type='final_results'
        ORDER BY id
        """,
        (int(agenda_id), int(experiment_run_id)),
    )
    for row in rows:
        path = Path(str(row.get("path") or ""))
        try:
            if path.is_file() and not path.is_symlink():
                return True
        except OSError:
            continue
    return False


def _durable_success(
    *,
    run: dict[str, Any],
    artifacts_present: int,
    final_results_present: bool,
    outcome_execution_result: str | None = None,
    outcome_verdict: str | None = None,
) -> bool:
    state = str(run.get("scientific_evidence_state") or "planned")
    try:
        evidence_ready = EVIDENCE_STATES.index(state) >= EVIDENCE_STATES.index(
            "sanity_passed"
        )
    except ValueError:
        evidence_ready = False
    verdict = str(
        outcome_verdict
        if outcome_verdict is not None
        else run.get("hypothesis_verdict") or ""
    ).strip().lower()
    return (
        str(run.get("status") or "") == "completed"
        and artifacts_present > 0
        and final_results_present
        and evidence_ready
        and str(run.get("resource_class") or "").strip().lower() == "cpu"
        and verdict in {"supported", "refuted", "inconclusive"}
        and (
            outcome_execution_result is None
            or str(outcome_execution_result).strip().lower() == "completed"
        )
    )


def _default_forge(idea_id: int, resource_grant_id: int) -> dict[str, Any]:
    from agents.experiment_forge import forge_experiment

    return forge_experiment(idea_id, resource_grant_id=resource_grant_id)


def _default_validate(run_id: int) -> dict[str, Any]:
    from agents.validation_loop import run_validation_loop

    return run_validation_loop(run_id)


def _mark_settlement_required(
    *,
    request: BoundedExecutionRequest,
    result: BoundedExecutionResult,
    reason: str,
    experiment_run_id: int | None,
) -> BoundedExecutionResult:
    result.status = "settlement_required"
    result.reason = reason
    try:
        current = _load_job(request)
        if (
            str(current.get("status") or "") == "blocked"
            and str(current.get("stage") or "") in WITHDRAWN_JOB_STAGES
        ):
            result.status = "failed"
            return result
        _mark_job_failed(
            request=request,
            reason=reason,
            experiment_run_id=experiment_run_id,
            settlement_required=True,
        )
    except Exception as transition_error:
        result.details["job_transition_error"] = (
            f"{type(transition_error).__name__}: {transition_error}"
        )
    return result


def _finish_existing_outcome(
    *,
    request: BoundedExecutionRequest,
    result: BoundedExecutionResult,
    outcome: dict[str, Any],
) -> BoundedExecutionResult:
    """Repair only the exact job transition after outcome commit.

    This is the critical crash boundary: a consumed grant cannot pass normal
    authorization, and replay must not call forge or validation again.
    """

    run_id = int(outcome.get("experiment_run_id") or 0)
    if run_id <= 0:
        return _mark_settlement_required(
            request=request,
            result=result,
            reason="existing outcome has no experiment run",
            experiment_run_id=None,
        )
    try:
        run = _load_run(request, run_id)
        digest, present, missing = raw_artifacts_hash(
            agenda_id=request.agenda_id,
            experiment_run_id=run_id,
        )
        del digest  # presence and persisted evidence state decide replay truth.
        final_results_present = _real_final_results_present(
            agenda_id=request.agenda_id,
            experiment_run_id=run_id,
        )
        result.experiment_run_id = run_id
        result.outcome_record_id = int(outcome["id"])
        result.evidence_state = str(
            run.get("scientific_evidence_state")
            or outcome.get("state_decision")
            or "planned"
        )
        result.verdict = str(outcome.get("verdict") or "") or None
        result.details["replay"] = "existing_outcome"
        result.details["artifacts"] = {
            "present": present,
            "missing": missing,
            "final_results_present": final_results_present,
        }
        if _durable_success(
            run=run,
            artifacts_present=present,
            final_results_present=final_results_present,
            outcome_execution_result=str(outcome.get("execution_result") or ""),
            outcome_verdict=str(outcome.get("verdict") or ""),
        ):
            _settle_job(
                request=request,
                experiment_run_id=run_id,
                note=(
                    f"bounded pilot replay completed: outcome_record={outcome['id']} "
                    f"state={result.evidence_state}"
                ),
            )
            result.status = "completed"
            return result
        _mark_job_failed(
            request=request,
            reason="bounded pilot outcome did not meet durable success criteria",
            experiment_run_id=run_id,
        )
        result.status = "settled_failed"
        result.reason = "durable_success_criteria_not_met"
        return result
    except Exception as exc:
        return _mark_settlement_required(
            request=request,
            result=result,
            reason=f"existing_outcome_replay_failed:{type(exc).__name__}: {exc}",
            experiment_run_id=run_id,
        )


def _finalize_persisted_run(
    *,
    request: BoundedExecutionRequest,
    actor: str,
    repository: MetaHarnessRepository,
    result: BoundedExecutionResult,
) -> BoundedExecutionResult:
    """Settle one persisted run without invoking compute or LLM work."""

    run_id = int(result.experiment_run_id or 0)
    try:
        run = _load_run(request, run_id)
        digest, present, missing = raw_artifacts_hash(
            agenda_id=request.agenda_id,
            experiment_run_id=run_id,
        )
        final_results_present = _real_final_results_present(
            agenda_id=request.agenda_id,
            experiment_run_id=run_id,
        )
        result.details["artifacts"] = {
            "present": present,
            "missing": missing,
            "final_results_present": final_results_present,
        }
        run_status = str(run.get("status") or "")
        resource_class = str(run.get("resource_class") or "").strip().lower()
        verdict = str(run.get("hypothesis_verdict") or "").strip().lower()
        if result.verdict is None:
            result.verdict = verdict or None
        valid_verdict = verdict in {"supported", "refuted", "inconclusive"}
        state = str(run.get("scientific_evidence_state") or "planned")
        try:
            already_sane = EVIDENCE_STATES.index(state) >= EVIDENCE_STATES.index(
                "sanity_passed"
            )
        except ValueError:
            already_sane = False
        if (
            run_status == "completed"
            and resource_class == "cpu"
            and present > 0
            and final_results_present
            and valid_verdict
            and not already_sane
        ):
            try:
                repository.advance_experiment_state(
                    agenda_id=request.agenda_id,
                    experiment_run_id=run_id,
                    target="sanity_passed",
                    context=EvidenceTransitionContext(
                        resource_grant_valid=True,
                        resource_grant_id=request.resource_grant_id,
                        execution_succeeded=True,
                        pilot_only=True,
                        raw_artifacts_present=True,
                        raw_artifacts_hash=digest,
                    ),
                    actor=actor,
                )
            except Exception as exc:
                # Evidence authority failure must prevent scientific success,
                # but it must not strand already-metered usage. Outcome
                # assembly below remains the formal settlement path.
                result.details["advance_error"] = f"{type(exc).__name__}: {exc}"
                result.details["not_advanced"] = "evidence_transition_failed"
            run = _load_run(request, run_id)
            state = str(run.get("scientific_evidence_state") or "planned")
        elif not already_sane:
            result.details["not_advanced"] = (
                "non_cpu_run"
                if resource_class != "cpu"
                else "invalid_or_missing_verdict"
                if not valid_verdict
                else "execution_incomplete"
                if run_status != "completed"
                else "no_final_results_file"
                if not final_results_present
                else "no_artifact_files"
            )
        result.evidence_state = str(run.get("scientific_evidence_state") or state)

        outcome_id = repository.assemble_and_record_outcome(
            resource_grant_id=request.resource_grant_id,
            experiment_run_id=run_id,
        )
        result.outcome_record_id = int(outcome_id)
        if _durable_success(
            run=run,
            artifacts_present=present,
            final_results_present=final_results_present,
        ):
            _settle_job(
                request=request,
                experiment_run_id=run_id,
                note=(
                    f"bounded pilot settled: outcome_record={outcome_id} "
                    f"state={result.evidence_state} "
                    f"verdict={result.verdict or 'unknown'}"
                ),
            )
            result.status = "completed"
        else:
            _mark_job_failed(
                request=request,
                reason="bounded pilot settled without durable scientific success",
                experiment_run_id=run_id,
            )
            result.status = "settled_failed"
            result.reason = "durable_success_criteria_not_met"
        return result
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return _mark_settlement_required(
            request=request,
            result=result,
            reason=f"run_settlement_failed:{type(exc).__name__}: {exc}",
            experiment_run_id=run_id,
        )


def _fail_without_run(
    *,
    request: BoundedExecutionRequest,
    repository: MetaHarnessRepository,
    result: BoundedExecutionResult,
    grant_status: str,
    reason: str,
) -> BoundedExecutionResult:
    """Refund an unused grant; fail closed if metered usage prevents it."""

    result.reason = reason
    if grant_status == "consumed":
        return _mark_settlement_required(
            request=request,
            result=result,
            reason=f"consumed_grant_without_outcome_or_run:{reason}",
            experiment_run_id=None,
        )
    try:
        if grant_status == "active":
            revoked = repository.revoke_grant(
                request.resource_grant_id,
                agenda_id=request.agenda_id,
                reason=f"bounded_pilot_failed:{reason}"[:500],
            )
            result.details["grant"] = (
                "revoked_and_refunded" if revoked else "already_withdrawn"
            )
            if not revoked:
                outcome = _existing_outcome(request)
                if outcome:
                    return _finish_existing_outcome(
                        request=request,
                        result=result,
                        outcome=outcome,
                    )
                _, refreshed = _load_bounded_grant(request)
                if str(refreshed.get("status") or "") == "consumed":
                    return _mark_settlement_required(
                        request=request,
                        result=result,
                        reason="grant_consumed_during_unused_release_without_outcome",
                        experiment_run_id=None,
                    )
        else:
            result.details["grant"] = f"already_{grant_status or 'non_active'}"
        current = _load_job(request)
        if (
            str(current.get("status") or "") == "blocked"
            and str(current.get("stage") or "") in WITHDRAWN_JOB_STAGES
        ):
            result.status = "failed"
            return result
        _mark_job_failed(
            request=request,
            reason=reason,
            experiment_run_id=None,
        )
        result.status = "failed"
        return result
    except Exception as exc:
        result.details["grant"] = f"settlement_required:{type(exc).__name__}: {exc}"
        return _mark_settlement_required(
            request=request,
            result=result,
            reason=f"unused_grant_release_failed:{type(exc).__name__}: {exc}",
            experiment_run_id=None,
        )


def execute_granted_candidate(
    request: BoundedExecutionRequest,
    *,
    actor: str,
    repository: MetaHarnessRepository | None = None,
    forge: Callable[[int, int], dict[str, Any]] | None = None,
    validate: Callable[[int], dict[str, Any]] | None = None,
) -> BoundedExecutionResult:
    """Run one already-authorized candidate through to an OutcomeRecord.

    ``forge`` and ``validate`` are injectable only so the wiring can be tested
    without the whole experiment stack; production always uses the reviewed
    implementations.
    """
    request.validate()
    if not str(actor or "").strip():
        raise BoundedExecutionError("actor is required")
    repo = repository or MetaHarnessRepository()
    run_forge = forge or _default_forge
    run_validate = validate or _default_validate

    grant, grant_row = _load_bounded_grant(request)
    job = _claim_job(request)
    job_id = int(job["id"])
    result = BoundedExecutionResult(
        status="failed",
        agenda_id=request.agenda_id,
        idea_id=request.idea_id,
        resource_grant_id=request.resource_grant_id,
        job_id=job_id,
    )
    outcome = _existing_outcome(request)
    if outcome:
        return _finish_existing_outcome(
            request=request,
            result=result,
            outcome=outcome,
        )

    # Durable bindings win over process-local return values. If a prior
    # invocation reached run creation, this replay settles that run and never
    # invokes forge or validation again.
    run_id = int(job.get("experiment_run_id") or 0) or _run_bound_to_grant(request)
    if run_id:
        result.experiment_run_id = run_id
        try:
            _attach_run_to_job(request, experiment_run_id=run_id)
        except BoundedExecutionError as exc:
            return _mark_settlement_required(
                request=request,
                result=result,
                reason=f"existing_run_attachment_failed:{exc}",
                experiment_run_id=run_id,
            )
        result.details["replay"] = "existing_run"
        return _finalize_persisted_run(
            request=request,
            actor=actor,
            repository=repo,
            result=result,
        )

    current_pair = (
        str(job.get("status") or ""),
        str(job.get("stage") or ""),
    )
    if current_pair in {
        ("blocked", SETTLEMENT_REQUIRED_STAGE),
        ("failed", FAILED_STAGE),
        ("completed", DONE_STAGE),
    } | {("blocked", stage) for stage in WITHDRAWN_JOB_STAGES}:
        return _fail_without_run(
            request=request,
            repository=repo,
            result=result,
            grant_status=str(grant_row.get("status") or ""),
            reason="terminal_job_has_no_outcome_or_run",
        )

    try:
        # Authorization is deliberately after the replay checks. A consumed
        # grant must be able to repair its exact job, but only an active grant
        # may enter forge/validation and incur new usage.
        grant, grant_row = _authorize_bounded_grant(request)
    except Exception as exc:
        return _fail_without_run(
            request=request,
            repository=repo,
            result=result,
            grant_status=str(grant_row.get("status") or ""),
            reason=f"grant_authorization_failed:{type(exc).__name__}: {exc}",
        )

    forge_error = ""
    forged: dict[str, Any] = {}
    try:
        candidate = run_forge(request.idea_id, request.resource_grant_id)
        if isinstance(candidate, dict):
            forged = candidate
            forge_error = str(candidate.get("error") or "")
        else:
            forge_error = "forge_returned_non_mapping"
    except Exception as exc:
        forge_error = f"{type(exc).__name__}: {exc}"

    run_id = _run_bound_to_grant(request) or int(forged.get("run_id") or 0)
    if not run_id:
        return _fail_without_run(
            request=request,
            repository=repo,
            result=result,
            grant_status=str(grant_row.get("status") or ""),
            reason=f"forge_failed:{forge_error or 'forge_returned_no_run'}",
        )

    result.experiment_run_id = run_id
    try:
        _load_run(request, run_id)
        _attach_run_to_job(request, experiment_run_id=run_id)
    except Exception as exc:
        return _mark_settlement_required(
            request=request,
            result=result,
            reason=f"new_run_binding_failed:{type(exc).__name__}: {exc}",
            experiment_run_id=run_id,
        )

    if forge_error:
        result.details["forge_error"] = forge_error
    else:
        run_before_validation = _load_run(request, run_id)
        if str(run_before_validation.get("resource_class") or "").lower() != "cpu":
            result.details["validation_skipped"] = "bounded_path_refuses_non_cpu_run"
        else:
            try:
                validated = run_validate(run_id)
                if not isinstance(validated, dict):
                    result.details["validation_error"] = "non_mapping_result"
                else:
                    result.verdict = str(validated.get("verdict") or "").lower() or None
                    result.details["validation"] = {
                        key: validated.get(key)
                        for key in (
                            "verdict",
                            "baseline",
                            "best_value",
                            "effect_pct",
                            "reason",
                            "error",
                        )
                    }
                    if validated.get("error"):
                        result.details["validation_error"] = str(validated["error"])
                    elif result.verdict == "blocked":
                        result.details["validation_error"] = (
                            "blocked:" + str(validated.get("reason") or "unknown")
                        )
            except Exception as exc:
                result.details["validation_error"] = f"{type(exc).__name__}: {exc}"

    # Whether validation succeeded, failed, or crashed, a persisted run may
    # carry metered usage. Settlement is mandatory; only the evidence/artifact
    # truth decides whether the job is successful.
    return _finalize_persisted_run(
        request=request,
        actor=actor,
        repository=repo,
        result=result,
    )
