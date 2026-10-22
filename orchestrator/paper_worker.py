"""Retired background paper ingestion worker.

The unscoped paper loop it used to run has no ResourceGrant scope, so start()
only reports it disabled; ingestion runs as agenda-scoped jobs instead
(scoped_ingestion_worker). get_status() and stop() remain for the dashboard and
shutdown paths.
"""

from __future__ import annotations

import threading

from compat.filelock import FileLock
from config import (
    AUTO_PIPELINE_BATCH_SIZE,
    AUTO_PIPELINE_INTERVAL_SECONDS,
)
from orchestrator.pipeline import log_event

_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()
_process_lock: FileLock | None = None
_last_status: dict = {"status": "not_started"}


def _release_process_lock() -> None:
    global _process_lock
    if _process_lock is None:
        return
    try:
        _process_lock.release()
    finally:
        _process_lock = None


def get_status() -> dict:
    with _worker_lock:
        running = bool(_worker_thread and _worker_thread.is_alive())
        last_status = dict(_last_status)
    return {
        "running": running,
        "batch_size": AUTO_PIPELINE_BATCH_SIZE,
        "interval_seconds": AUTO_PIPELINE_INTERVAL_SECONDS,
        **last_status,
    }


def start() -> dict:
    global _last_status
    status = {
        "status": "disabled_resource_grant_required",
        "reason": "agenda_scoped_ingestion_job_required",
    }
    with _worker_lock:
        _last_status = dict(status)
    return status


def stop() -> dict:
    _stop_event.set()
    _release_process_lock()
    log_event("paper_worker", {"step": "stopped"})
    return {"status": "stopping"}
