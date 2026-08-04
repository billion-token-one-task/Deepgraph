"""Execute exactly one portfolio-granted candidate, without global autonomy.

The meta-harness could issue a ResourceGrant and park the job at
``portfolio_granted``, but nothing in the codebase ever read that stage back:
the authorization was written and never consumed. Turning on global autonomy
did not fix that -- it only started every other loop -- so the recovery runbook
step "give the winning candidate one small, short CPU/LLM ResourceGrant and run
it" had no executor at all.

This module is that executor, and deliberately nothing more:

* it runs **one** candidate, named explicitly by (agenda, idea, grant) -- there
  is no discovery query, no backlog, and no loop;
* it never reads or writes ``DEEPGRAPH_AUTO_RESEARCH_ENABLED`` /
  ``DEEPGRAPH_AUTO_PIPELINE_ENABLED``, so it cannot be a back door into global
  autonomy;
* it refuses any backend outside the grant's own allowlist and any allowlist
  wider than CPU/LLM, so a bounded pilot can never reach for GPUs;
* the experiment itself is built and run by the existing reviewed machinery
  (``forge_experiment`` then ``run_validation_loop``). Nothing here duplicates
  run creation, state authority, or budget accounting;
* every exit settles: an OutcomeRecord on success, a released grant on failure.
  A path that could strand the agenda's reservation would just be a slower way
  to wedge the budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from contracts.meta_harness import ResourceGrant
from db import database as db
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.grants import GrantDeniedError, ResourceRequest, authorize
from meta_harness.repository import MetaHarnessRepository
from orchestrator.pipeline import log_event


# A bounded pilot is a CPU/LLM errand. Anything wider needs its own grant and
# its own decision; widening it here would silently reintroduce GPU spend.
BOUNDED_BACKENDS = frozenset({"cpu", "llm"})
BOUNDED_STAGE = "pilot"
GRANTED_STAGE = "portfolio_granted"
RUNNING_STAGE = "pilot_running"
DONE_STAGE = "pilot_outcome_recorded"
FAILED_STAGE = "pilot_failed"


class BoundedExecutionError(RuntimeError):
    """Raised when the single-candidate contract cannot be honoured."""


@dataclass(frozen=True)
class BoundedExecutionRequest:
    agenda_id: int
    idea_id: int
    resource_grant_id: int

    def validate(self) -> None:
        for name in ("agenda_id", "idea_id", "resource_grant_id"):
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
    if grant.max_gpu_hours > 0:
        raise BoundedExecutionError("bounded execution refuses a GPU-hour grant")
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


def _claim_job(request: BoundedExecutionRequest) -> dict[str, Any]:
    """Take the granted job atomically; a second caller must find nothing."""
    job = db.fetchone(
        """
        SELECT id, agenda_id, deep_insight_id, status, stage, resource_grant_id,
               experiment_run_id
        FROM auto_research_jobs
        WHERE agenda_id=? AND deep_insight_id=? AND resource_grant_id=?
        """,
        (request.agenda_id, request.idea_id, request.resource_grant_id),
    )
    if not job:
        raise BoundedExecutionError("granted_job_not_found")
    if str(job.get("stage") or "") != GRANTED_STAGE:
        raise BoundedExecutionError(
            f"job is at stage '{job.get('stage')}', not '{GRANTED_STAGE}'"
        )
    cursor = db.execute(
        """
        UPDATE auto_research_jobs
        SET status='running_experiment', stage=?, last_error=NULL,
            last_note=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=? AND stage=? AND status='queued'
        """,
        (
            RUNNING_STAGE,
            "bounded pilot claimed by operator-invoked execution path",
            int(job["id"]),
            request.agenda_id,
            GRANTED_STAGE,
        ),
    )
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        db.rollback()
        raise BoundedExecutionError("granted_job_already_claimed")
    db.commit()
    return dict(job)


def _release_job(
    *,
    job_id: int,
    agenda_id: int,
    reason: str,
    experiment_run_id: int | None = None,
) -> None:
    db.execute(
        """
        UPDATE auto_research_jobs
        SET status='blocked', stage=?, last_error=?, experiment_run_id=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (FAILED_STAGE, reason[:1000], experiment_run_id, int(job_id), int(agenda_id)),
    )
    db.commit()


def _settle_job(
    *,
    job_id: int,
    agenda_id: int,
    experiment_run_id: int,
    note: str,
) -> None:
    db.execute(
        """
        UPDATE auto_research_jobs
        SET status='completed', stage=?, experiment_run_id=?, last_error=NULL,
            last_note=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (DONE_STAGE, int(experiment_run_id), note[:1000], int(job_id), int(agenda_id)),
    )
    db.commit()


def _default_forge(idea_id: int, resource_grant_id: int) -> dict[str, Any]:
    from agents.experiment_forge import forge_experiment

    return forge_experiment(idea_id, resource_grant_id=resource_grant_id)


def _default_validate(run_id: int) -> dict[str, Any]:
    from agents.validation_loop import run_validation_loop

    return run_validation_loop(run_id)


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

    grant, grant_row = _authorize_bounded_grant(request)
    job = _claim_job(request)
    job_id = int(job["id"])
    result = BoundedExecutionResult(
        status="failed",
        agenda_id=request.agenda_id,
        idea_id=request.idea_id,
        resource_grant_id=request.resource_grant_id,
        job_id=job_id,
    )
    log_event(
        "bounded_execution",
        {
            "step": "claimed",
            "agenda_id": request.agenda_id,
            "idea_id": request.idea_id,
            "resource_grant_id": request.resource_grant_id,
            "token_cap": grant.token_cap,
            "backends": sorted(grant.backend_allowlist),
        },
    )

    run_id: int | None = None
    try:
        forged = run_forge(request.idea_id, request.resource_grant_id)
        if not isinstance(forged, dict) or forged.get("error"):
            raise BoundedExecutionError(
                "forge_failed:" + str((forged or {}).get("error") or "unknown")
            )
        run_id = int(forged.get("run_id") or 0)
        if run_id <= 0:
            raise BoundedExecutionError("forge_returned_no_run")
        result.experiment_run_id = run_id

        validated = run_validate(run_id)
        if not isinstance(validated, dict) or validated.get("error"):
            raise BoundedExecutionError(
                "validation_failed:" + str((validated or {}).get("error") or "unknown")
            )
        verdict = str(validated.get("verdict") or "").strip().lower()
        result.verdict = verdict
        result.details["validation"] = {
            key: validated.get(key)
            for key in ("verdict", "baseline", "best_value", "effect_pct")
        }
        if verdict == "blocked":
            raise BoundedExecutionError(
                "validation_blocked:" + str(validated.get("reason") or "unknown")
            )

        run = db.fetchone(
            """
            SELECT id, agenda_id, deep_insight_id, status, resource_grant_id,
                   scientific_evidence_state
            FROM experiment_runs
            WHERE id=? AND agenda_id=?
            """,
            (run_id, request.agenda_id),
        )
        if not run or int(run.get("resource_grant_id") or 0) != request.resource_grant_id:
            raise BoundedExecutionError("run_not_bound_to_grant")
        execution_succeeded = str(run.get("status") or "") == "completed"

        digest, present, missing = raw_artifacts_hash(
            agenda_id=request.agenda_id,
            experiment_run_id=run_id,
        )
        result.details["artifacts"] = {"present": present, "missing": missing}
        if execution_succeeded and present > 0:
            # A pilot's ladder tops out at sanity_passed by construction:
            # pilot_only blocks full_benchmark_complete in the state machine.
            state = repo.advance_experiment_state(
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
            result.evidence_state = state
        else:
            # No artifacts, or an execution that did not complete, is a real
            # outcome. It still gets recorded -- an unsettled grant would leave
            # the agenda's reservation stranded, which is the failure mode this
            # path exists to avoid.
            result.evidence_state = str(run.get("scientific_evidence_state") or "planned")
            result.details["not_advanced"] = (
                "execution_incomplete" if not execution_succeeded else "no_artifact_files"
            )

        outcome_id = repo.assemble_and_record_outcome(
            resource_grant_id=request.resource_grant_id,
            experiment_run_id=run_id,
        )
        result.outcome_record_id = int(outcome_id)
        result.status = "completed"
        _settle_job(
            job_id=job_id,
            agenda_id=request.agenda_id,
            experiment_run_id=run_id,
            note=(
                f"bounded pilot settled: outcome_record={outcome_id} "
                f"state={result.evidence_state} verdict={verdict or 'unknown'}"
            ),
        )
        log_event(
            "bounded_execution",
            {"step": "settled", **result.to_dict()},
        )
        return result
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        result.reason = reason
        try:
            db.rollback()
        except Exception:
            pass
        _release_job(
            job_id=job_id,
            agenda_id=request.agenda_id,
            reason=reason,
            experiment_run_id=run_id,
        )
        # The grant must not stay reserved behind a failed pilot. Revocation
        # refunds the agenda; a grant that already metered usage cannot be
        # revoked, and is left for outcome assembly to settle explicitly.
        try:
            repo.revoke_grant(
                request.resource_grant_id,
                agenda_id=request.agenda_id,
                reason=f"bounded_pilot_failed:{reason}"[:500],
            )
            result.details["grant"] = "revoked_and_refunded"
        except Exception as revoke_error:
            result.details["grant"] = f"not_revoked:{revoke_error}"
        log_event("error", {"step": "bounded_execution", **result.to_dict()})
        if isinstance(exc, (BoundedExecutionError, GrantDeniedError)):
            return result
        raise
