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
    from orchestrator.meta_compute_runtime import build_scheduler

    repository = ColabWorkRepository()
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
                   rg.idempotency_key AS grant_idempotency_key
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
