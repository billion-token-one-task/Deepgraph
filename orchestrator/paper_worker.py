"""Background paper ingestion/extraction worker.

This closes the gap between the read-only web process and the paper pipeline:
when enabled, the main process periodically calls ``run_continuous`` without
requiring the removed manual ``POST /api/start`` endpoint or an external shell
supervisor.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from compat.filelock import FileLock
from config import (
    AUTO_PIPELINE_BATCH_SIZE,
    AUTO_PIPELINE_INTERVAL_SECONDS,
    AUTO_PIPELINE_START_DELAY_SECONDS,
)
from db import database as db
from orchestrator.pipeline import log_event, run_continuous

_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()
_process_lock: FileLock | None = None
_last_status: dict = {"status": "not_started"}


def _lock_path() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("TEMP", str(Path.home() / ".cache"))) / "deepgraph-paper-worker.lock"
    return Path("/tmp/deepgraph-paper-worker.lock")


def _try_acquire_process_lock() -> bool:
    global _process_lock
    if _process_lock is not None:
        return True
    lock = FileLock(str(_lock_path()))
    if not lock.try_acquire():
        return False
    try:
        handle = getattr(lock, "_handle")
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        lock.release()
        return False
    _process_lock = lock
    return True


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
    return {
        "running": running,
        "batch_size": AUTO_PIPELINE_BATCH_SIZE,
        "interval_seconds": AUTO_PIPELINE_INTERVAL_SECONDS,
        **_last_status,
    }


def run_cycle() -> dict:
    db.init_db()
    batch_size = max(1, AUTO_PIPELINE_BATCH_SIZE)
    started = time.time()
    processed = run_continuous(batch_size)
    return {
        "status": "completed",
        "processed": processed,
        "batch_size": batch_size,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def _run_loop() -> None:
    global _last_status
    if AUTO_PIPELINE_START_DELAY_SECONDS > 0:
        _stop_event.wait(AUTO_PIPELINE_START_DELAY_SECONDS)
    while not _stop_event.is_set():
        try:
            _last_status = {"status": "running", "started_at": time.time()}
            log_event("paper_worker", {"step": "cycle_started", "batch_size": AUTO_PIPELINE_BATCH_SIZE})
            _last_status = run_cycle()
            log_event("paper_worker", {"step": "cycle_completed", **_last_status})
            sleep_s = max(5, AUTO_PIPELINE_INTERVAL_SECONDS)
        except Exception as exc:  # pragma: no cover - defensive background guard
            try:
                db.rollback()
            except Exception:
                pass
            _last_status = {"status": "failed", "error": str(exc)}
            log_event("error", {"step": "paper_worker_loop", "error": str(exc)})
            sleep_s = max(5, AUTO_PIPELINE_INTERVAL_SECONDS)
        _stop_event.wait(sleep_s)


def start() -> dict:
    global _worker_thread
    db.init_db()
    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return {"status": "already_running"}
        if not _try_acquire_process_lock():
            return {"status": "already_running_elsewhere"}
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_run_loop, daemon=True, name="deepgraph-paper-worker")
        _worker_thread.start()
    log_event("paper_worker", {"step": "started"})
    return {"status": "started"}


def stop() -> dict:
    _stop_event.set()
    _release_process_lock()
    log_event("paper_worker", {"step": "stopped"})
    return {"status": "stopping"}
