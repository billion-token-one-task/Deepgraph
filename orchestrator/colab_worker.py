"""Single-process worker for durable Colab compute requests."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
from datetime import datetime, timedelta
from pathlib import Path

from config import COMPUTE_COLAB_POLL_SECONDS
from db import database as db
from meta_harness.backends.colab_durable import (
    ColabWorkRepository,
    execution_request_from_row,
    grant_from_row,
)
from meta_harness.compute import ComputeBackendError, ComputeJob
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.repository import MetaHarnessRepository
from meta_harness.runner_contract import validate_final_results, verify_metric_from_artifacts


_thread: threading.Thread | None = None
_lock = threading.Lock()
_stop = threading.Event()
_last_status: dict = {"status": "not_started"}


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:colab"


def _record_terminal_run_failure(row: dict, result, observed) -> None:
    """Make a terminal Colab result terminal for its owning experiment run.

    Durable compute state alone is not a scheduler consumer: an auto-research
    job remains ``queued_gpu`` until its experiment run changes state.  Without
    this handoff, a failed Colab request holds the only execution slot forever.
    The bounded Colab result payload remains the detailed diagnostic record.
    """

    status = str(getattr(observed, "status", "") or "")
    if status not in {"failed", "timed_out", "cancelled"}:
        return
    reason = str(
        getattr(result, "failure_reason", None)
        or getattr(observed, "failure_reason", None)
        or f"colab_returncode_{getattr(result, 'returncode', 'unknown')}"
    )
    db.execute(
        """
        UPDATE experiment_runs
        SET status='failed', phase='colab_compute_failed', error_message=?,
            completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
        WHERE id=? AND agenda_id=? AND deep_insight_id=?
          AND status NOT IN ('completed', 'bundle_ready', 'cancelled', 'superseded', 'archived')
        """,
        (
            f"colab_compute_{status}:{reason}"[:4000],
            int(row["experiment_run_id"]),
            int(row["agenda_id"]),
            int(row["idea_id"]),
        ),
    )
    db.commit()


def _record_terminal_run_success(row: dict, observed) -> None:
    """Promote a verified Colab result into the owning run's durable evidence.

    The Colab backend stores its own artifact manifest, but an outcome can only
    be assembled after the runner artifacts have been hash-verified and
    registered against the experiment run.  Keep this handoff here, adjacent to
    the failure handoff, so a terminal compute request can never strand a
    scheduler job in ``queued_gpu``.
    """

    if str(getattr(observed, "status", "") or "") != "succeeded":
        return
    run = db.fetchone(
        """
        SELECT id, agenda_id, deep_insight_id, resource_grant_id, workdir,
               scientific_evidence_state
        FROM experiment_runs
        WHERE id=? AND agenda_id=? AND deep_insight_id=?
        """,
        (int(row["experiment_run_id"]), int(row["agenda_id"]), int(row["idea_id"])),
    )
    if not run or int(run.get("resource_grant_id") or 0) != int(
        row["resource_grant_id"]
    ):
        return
    results_dir = Path(str(run.get("workdir") or "")) / "results"
    final_path = results_dir / "final_results.json"
    payload = validate_final_results(
        json.loads(final_path.read_text(encoding="utf-8"))
    )
    verification = verify_metric_from_artifacts(final_path)
    for artifact_type, reference in payload["artifacts"].items():
        relative_path = str((reference or {}).get("path") or "")
        artifact_path = (results_dir / relative_path).resolve()
        if (
            not relative_path
            or (artifact_path != results_dir and results_dir not in artifact_path.parents)
            or not artifact_path.is_file()
        ):
            raise ComputeBackendError(f"artifact_contract_violation:{artifact_type}")
        expected_hash = str(payload["artifact_hashes"].get(artifact_type) or "")
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ComputeBackendError(f"artifact_hash_mismatch:{artifact_type}")
        existing = db.fetchone(
            """
            SELECT id FROM experiment_artifacts
            WHERE agenda_id=? AND run_id=? AND artifact_type=? AND path=?
            LIMIT 1
            """,
            (int(run["agenda_id"]), int(run["id"]), artifact_type, str(artifact_path)),
        )
        if not existing:
            db.execute(
                """
                INSERT INTO experiment_artifacts
                    (agenda_id, run_id, artifact_type, path, metric_key,
                     metric_value, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(run["agenda_id"]),
                    int(run["id"]),
                    artifact_type,
                    str(artifact_path),
                    verification.metric_name,
                    verification.candidate_value if artifact_type == "final_results" else None,
                    json.dumps(
                        {
                            "contract_type": "RunnerArtifact",
                            "sha256": expected_hash or actual_hash,
                            "verified_by": "colab_terminal_handoff_v1",
                        },
                        sort_keys=True,
                    ),
                ),
            )
    effect = (
        verification.candidate_value - verification.baseline_value
        if verification.direction == "higher"
        else verification.baseline_value - verification.candidate_value
    )
    effect_pct = (
        (effect / abs(verification.baseline_value)) * 100.0
        if verification.baseline_value != 0
        else None
    )
    verdict = "refuted" if payload.get("scientific_negative_result") is True else "inconclusive"
    db.execute(
        """
        UPDATE experiment_runs
        SET status='completed', phase='colab_result_verified',
            baseline_metric_name=?, baseline_metric_value=?, best_metric_value=?,
            effect_size=?, effect_pct=?, hypothesis_verdict=?, error_message=NULL,
            completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
        WHERE id=? AND agenda_id=? AND deep_insight_id=?
          AND status NOT IN ('failed', 'cancelled', 'superseded', 'archived')
        """,
        (
            verification.metric_name,
            verification.baseline_value,
            verification.candidate_value,
            effect,
            effect_pct,
            verdict,
            int(run["id"]),
            int(run["agenda_id"]),
            int(run["deep_insight_id"]),
        ),
    )
    db.commit()

    # This is an acceptance-sized pilot, so it may only advance one evidence
    # rung.  In particular, it cannot become full-benchmark or manuscript
    # evidence merely because the infrastructure run succeeded.
    if str(run.get("scientific_evidence_state") or "planned") == "planned":
        from orchestrator.bounded_execution import raw_artifacts_hash

        digest, present, missing = raw_artifacts_hash(
            agenda_id=int(run["agenda_id"]), experiment_run_id=int(run["id"])
        )
        if present <= 0 or missing:
            raise ComputeBackendError("runner_artifact_registration_incomplete")
        MetaHarnessRepository().advance_experiment_state(
            agenda_id=int(run["agenda_id"]),
            experiment_run_id=int(run["id"]),
            target="sanity_passed",
            context=EvidenceTransitionContext(
                resource_grant_valid=True,
                resource_grant_id=int(row["resource_grant_id"]),
                execution_succeeded=True,
                pilot_only=True,
                raw_artifacts_present=True,
                raw_artifacts_hash=digest,
            ),
            actor="colab_terminal_handoff_v1",
        )


def _reconcile_succeeded_runs() -> int:
    """Recover success handoffs interrupted after durable Colab settlement."""

    rows = db.fetchall(
        """
        SELECT cwr.experiment_run_id, cwr.agenda_id, cwr.idea_id,
               cwr.resource_grant_id
        FROM colab_work_requests_v1 AS cwr
        JOIN compute_jobs_v1 AS cj ON cj.id=cwr.compute_job_id
        JOIN experiment_runs AS er ON er.id=cwr.experiment_run_id
        WHERE cwr.status='succeeded' AND cj.status='succeeded'
          AND er.status NOT IN ('failed', 'cancelled', 'superseded', 'archived')
          AND (
              er.status <> 'completed'
              OR COALESCE(er.scientific_evidence_state, 'planned') = 'planned'
          )
        ORDER BY cwr.completed_at ASC, cwr.id ASC
        LIMIT 20
        """
    )
    for row in rows:
        _record_terminal_run_success(
            dict(row), type("Observed", (), {"status": "succeeded"})()
        )
    return len(rows)


def recover_succeeded_run(*, experiment_run_id: int, resource_grant_id: int) -> bool:
    """Recover one explicitly named successful request without claiming work.

    This operator-safe entry point is intentionally narrower than ``run_one``:
    it cannot inspect, claim, submit, or execute any other queued Colab work.
    It only performs the verified terminal handoff for the supplied run/grant
    pair, which makes it suitable for recovering a controller interruption.
    """

    row = db.fetchone(
        """
        SELECT cwr.experiment_run_id, cwr.agenda_id, cwr.idea_id,
               cwr.resource_grant_id
        FROM colab_work_requests_v1 AS cwr
        JOIN compute_jobs_v1 AS cj ON cj.id=cwr.compute_job_id
        WHERE cwr.experiment_run_id=? AND cwr.resource_grant_id=?
          AND cwr.status='succeeded' AND cj.status='succeeded'
        ORDER BY cwr.completed_at DESC, cwr.id DESC
        LIMIT 1
        """,
        (int(experiment_run_id), int(resource_grant_id)),
    )
    if not row:
        return False
    _record_terminal_run_success(
        dict(row), type("Observed", (), {"status": "succeeded"})()
    )
    return True


def run_one() -> dict:
    """Claim and settle at most one request; safe for a scheduler loop or CI."""
    from meta_harness.attempt_gpu_usage import GrantGPUUsageControl
    from orchestrator.meta_compute_runtime import (
        build_scheduler,
        settle_colab_request,
    )

    for pending_request_id in (
        GrantGPUUsageControl().reconcile_terminal_colab_attempts()
    ):
        settle_colab_request(pending_request_id)
    reconciled_successes = _reconcile_succeeded_runs()

    repository = ColabWorkRepository()
    # A request the worker failed on its own defect never reached Colab and
    # cannot be recreated, because its idempotency key is derived from the run.
    # Give those back to the queue before claiming.
    repository.requeue_control_lost()
    row = repository.claim_next(worker_id=_worker_id())
    if not row:
        return {"status": "idle", "reconciled_succeeded_runs": reconciled_successes}
    worker_id = _worker_id()
    try:
        scheduler = build_scheduler()
        backend = scheduler.configured_backend("colab_gpu")
        grant_row = db.fetchone(
            """
            SELECT rg.*, rg.status AS grant_status,
                   rg.idempotency_key AS grant_idempotency_key,
                   -- grant_from_row reads the grant id under the name the
                   -- work-request rows use; resource_grants calls it "id", so
                   -- selecting rg.* alone left the mapper with a KeyError and
                   -- every claimed Colab request was quarantined as
                   -- colab_worker_control_lost before reaching the executor.
                   rg.id AS resource_grant_id
            FROM resource_grants AS rg
            WHERE rg.id=? AND rg.agenda_id=? AND rg.idea_id=?
              AND rg.status='active' AND rg.expires_at > CURRENT_TIMESTAMP
            """,
            (
                int(row["resource_grant_id"]),
                int(row["agenda_id"]),
                int(row["idea_id"]),
            ),
        )
        if not grant_row:
            raise ComputeBackendError(
                "Colab work ResourceGrant expired after claim"
            )
        transport = getattr(backend, "_transport", None)
        if transport is None or not hasattr(transport, "executor"):
            raise ComputeBackendError(
                "configured Colab backend has no durable executor"
            )
        result = transport.executor.run_request(
            execution_request_from_row(row),
            grant=grant_from_row(grant_row),
        )
        repository.save_result(int(row["id"]), result=result)
        persisted = db.fetchone(
            """
            SELECT cwr.completed_at, cwr.started_at, cj.gpu_attempt_reservation_id
            FROM colab_work_requests_v1 cwr
            JOIN compute_jobs_v1 cj ON cj.id=cwr.compute_job_id
            WHERE cwr.id=?
            """,
            (int(row["id"]),),
        ) or {}
        db.commit()
        reason_code = {
            "succeeded": "attempt_completed",
            "timed_out": "attempt_timed_out",
        }.get(result.status, "attempt_failed")
        started_at = persisted.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        completed_at = persisted.get("completed_at")
        if (
            isinstance(started_at, datetime)
            and float(result.wall_seconds or 0.0) > 0.0
        ):
            # The executor duration is the accelerator wall time.  The
            # request's completed_at is only when the controller persisted the
            # return value and can be milliseconds after claim.
            completed_at = started_at + timedelta(seconds=float(result.wall_seconds))
        GrantGPUUsageControl().settle_attempt(
            int(persisted.get("gpu_attempt_reservation_id") or 0),
            completed_at=completed_at,
            reason_code=reason_code,
        )
        observed = scheduler.refresh_and_settle(
            ComputeJob(
                backend_kind="colab_gpu",
                backend_job_id=str(row["backend_job_id"]),
                idempotency_key=str(row["idempotency_key"]),
                status="running",
                heartbeat_at=str(row.get("started_at") or "") or None,
            ),
            requirements=tuple(
                str(value)
                for value in __import__("json").loads(
                    grant_row.get("artifact_requirements_json") or "[]"
                )
            ),
        )
        _record_terminal_run_failure(row, result, observed)
        _record_terminal_run_success(row, observed)
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        persisted = db.fetchone(
            "SELECT status FROM colab_work_requests_v1 WHERE id=?",
            (int(row["id"]),),
        ) or {}
        if str(persisted.get("status") or "") == "running":
            repository.quarantine_claim(
                int(row["id"]),
                worker_id=worker_id,
                reason=f"colab_worker_control_lost:{type(exc).__name__}",
            )
        raise
    return {
        "status": observed.status,
        "colab_work_request_id": int(row["id"]),
        "compute_job_id": int(row["compute_job_id"]),
    }


def _loop() -> None:
    global _last_status
    while not _stop.is_set():
        try:
            _last_status = run_one()
        except Exception as exc:  # pragma: no cover - defensive worker guard
            try:
                db.rollback()
            except Exception:
                pass
            _last_status = {
                "status": "worker_error",
                "error": f"{type(exc).__name__}:{exc}",
            }
        _stop.wait(max(1, int(COMPUTE_COLAB_POLL_SECONDS)))


def start() -> dict:
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return {"status": "already_running", **_last_status}
        _stop.clear()
        _thread = threading.Thread(
            target=_loop,
            daemon=True,
            name="deepgraph-colab-worker",
        )
        _thread.start()
    return {"status": "started"}


def stop() -> dict:
    _stop.set()
    return {"status": "stopping"}


def get_status() -> dict:
    with _lock:
        running = bool(_thread and _thread.is_alive())
    return {"running": running, **_last_status}
