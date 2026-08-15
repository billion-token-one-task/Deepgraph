"""Single-process worker for durable Colab compute requests."""

from __future__ import annotations

import os
import socket
import threading

from config import COMPUTE_COLAB_POLL_SECONDS
from db import database as db
from meta_harness.backends.colab_durable import (
    ColabWorkRepository,
    execution_request_from_row,
    grant_from_row,
)
from meta_harness.compute import ComputeBackendError, ComputeJob


_thread: threading.Thread | None = None
_lock = threading.Lock()
_stop = threading.Event()
_last_status: dict = {"status": "not_started"}


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:colab"


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

    repository = ColabWorkRepository()
    # A request the worker failed on its own defect never reached Colab and
    # cannot be recreated, because its idempotency key is derived from the run.
    # Give those back to the queue before claiming.
    repository.requeue_control_lost()
    row = repository.claim_next(worker_id=_worker_id())
    if not row:
        return {"status": "idle"}
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
            SELECT cwr.completed_at, cj.gpu_attempt_reservation_id
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
        GrantGPUUsageControl().settle_attempt(
            int(persisted.get("gpu_attempt_reservation_id") or 0),
            completed_at=persisted.get("completed_at"),
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
