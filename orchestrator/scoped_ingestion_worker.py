"""Worker for explicitly queued, ResourceGrant-scoped paper enrichment."""

from __future__ import annotations

import json
import os
import socket
import threading

from config import (
    SCOPED_INGESTION_LEASE_SECONDS,
    SCOPED_INGESTION_POLL_SECONDS,
)
from db import database as db
from meta_harness.ingestion_queue import ScopedIngestionRepository
from orchestrator.pipeline import (
    _is_retryable_pipeline_error,
    process_single_paper,
)


_thread: threading.Thread | None = None
_lock = threading.Lock()
_stop = threading.Event()
_last_status: dict = {"status": "not_started"}


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:scoped-ingestion"


def run_one() -> dict:
    repository = ScopedIngestionRepository()
    worker_id = _worker_id()
    row = repository.claim_next(
        worker_id=worker_id,
        lease_seconds=SCOPED_INGESTION_LEASE_SECONDS,
    )
    if not row:
        return {"status": "idle"}
    paper_ids = json.loads(row.get("paper_ids_json") or "[]")
    scope = {
        "agenda_id": int(row["agenda_id"]),
        "idea_id": int(row["idea_id"]),
        "resource_grant_id": int(row["resource_grant_id"]),
        "stage": str(row["stage"]),
        "token_cap": int(row.get("token_cap") or 0),
    }
    results: list[dict] = []
    try:
        for paper_id in paper_ids:
            repository.renew_lease(
                int(row["id"]),
                agenda_id=int(row["agenda_id"]),
                worker_id=worker_id,
                lease_seconds=SCOPED_INGESTION_LEASE_SECONDS,
            )
            result = process_single_paper(str(paper_id), llm_scope=scope)
            results.append(dict(result))
            if result.get("error"):
                raise RuntimeError(str(result["error"]))
        repository.complete(
            int(row["id"]),
            agenda_id=int(row["agenda_id"]),
            worker_id=worker_id,
            results=results,
        )
        return {
            "status": "succeeded",
            "ingestion_job_id": int(row["id"]),
            "paper_count": len(results),
        }
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        target = repository.fail(
            int(row["id"]),
            agenda_id=int(row["agenda_id"]),
            worker_id=worker_id,
            reason=f"{type(exc).__name__}:{exc}",
            retryable=_is_retryable_pipeline_error(exc),
            partial_results=results,
        )
        return {
            "status": target,
            "ingestion_job_id": int(row["id"]),
            "paper_count": len(results),
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
        _stop.wait(max(1, int(SCOPED_INGESTION_POLL_SECONDS)))


def start() -> dict:
    global _thread
    repository = ScopedIngestionRepository()
    recovery = {"retryable": 0, "manual_reconciliation": 0}
    agendas = db.fetchall(
        """
        SELECT DISTINCT agenda_id
        FROM scoped_ingestion_jobs_v1
        WHERE status='running' AND lease_expires_at <= CURRENT_TIMESTAMP
        ORDER BY agenda_id
        """
    )
    for agenda in agendas:
        recovered = repository.recover_expired_leases(
            agenda_id=int(agenda["agenda_id"])
        )
        recovery["retryable"] += recovered["retryable"]
        recovery["manual_reconciliation"] += recovered[
            "manual_reconciliation"
        ]
    with _lock:
        if _thread and _thread.is_alive():
            return {"status": "already_running", "recovery": recovery}
        _stop.clear()
        _thread = threading.Thread(
            target=_loop,
            daemon=True,
            name="deepgraph-scoped-ingestion",
        )
        _thread.start()
    return {"status": "started", "recovery": recovery}


def stop() -> dict:
    _stop.set()
    return {"status": "stopping"}


def get_status() -> dict:
    with _lock:
        running = bool(_thread and _thread.is_alive())
    return {"running": running, **_last_status}
