#!/usr/bin/env python3.12
"""DeepGraph - Hierarchical ML Research Knowledge Engine."""
import os
import sys
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from compat.filelock import FileLock

from config import (
    APP_NAME,
    AUTO_PIPELINE_ENABLED,
    AUTO_RESEARCH_ENABLED,
    BACKFILL_GRAPH_ON_START,
    COMPUTE_BACKENDS_ENABLED,
    IDEA_WORKSPACE_DIR,
    REFRESH_MERGE_CANDIDATES_ON_START,
    ROOT_NODE_ID,
    SCOPED_INGESTION_WORKER_ENABLED,
    WEB_HOST,
    WEB_PORT,
    WORKSPACE_DIR,
    PDF_CACHE_DIR,
)
from db.database import describe_backend, init_db
from db.evidence_graph import (
    backfill_entity_resolutions,
    backfill_graph_from_structured_data,
    refresh_merge_candidates,
)
from db.taxonomy import seed_taxonomy, backfill_result_taxonomy
from web.app import app

_PROCESS_LOCK = None
_PROCESS_LOCK_PATH = (
    Path(os.environ.get("TEMP", str(Path.home() / ".cache"))) / "deepgraph-main.lock"
    if os.name == "nt"
    else Path("/tmp/deepgraph-main.lock")
)


def _current_lock_owner() -> str | None:
    try:
        owner = _PROCESS_LOCK_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return owner or None


def _try_acquire_process_lock() -> bool:
    global _PROCESS_LOCK
    if _PROCESS_LOCK is not None:
        return True
    lock = FileLock(str(_PROCESS_LOCK_PATH))
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
    _PROCESS_LOCK = lock
    return True


def _release_process_lock() -> None:
    global _PROCESS_LOCK
    if _PROCESS_LOCK is None:
        return
    try:
        _PROCESS_LOCK.release()
    finally:
        _PROCESS_LOCK = None


def _serve_http() -> None:
    print(f"Starting {APP_NAME} at http://{WEB_HOST}:{WEB_PORT} (root node: {ROOT_NODE_ID})", flush=True)
    try:
        from waitress import serve
    except ImportError:
        print(
            "Waitress is not installed; falling back to Flask dev server. "
            "Install waitress for production deployments.",
            flush=True,
        )
        app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
        return
    serve(app, host=WEB_HOST, port=WEB_PORT, threads=8)


def _run_startup_maintenance(label: str, fn) -> bool:
    print(f"{label}...", flush=True)
    try:
        fn()
    except Exception as exc:
        if "database is locked" in str(exc).lower():
            print(f"{label} skipped: database is locked; continuing startup.", flush=True)
            return False
        raise
    print(f"{label} ready.", flush=True)
    return True


def main():
    if not _try_acquire_process_lock():
        owner = _current_lock_owner()
        if owner:
            print(f"DeepGraph main already running under pid {owner}; refusing duplicate startup.", flush=True)
        else:
            print("DeepGraph main already running; refusing duplicate startup.", flush=True)
        return

    # Ensure directories exist
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    IDEA_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize database
        print("Initializing database...", flush=True)
        init_db()
        backend = describe_backend()
        print("Database ready.", flush=True)
        print(f"Database target: {backend['target']} ({backend['backend']})", flush=True)

        # Startup maintenance is idempotent; do not let transient SQLite writers
        # prevent the controller, auto-research loop, and GPU scheduler from booting.
        _run_startup_maintenance("Seeding taxonomy tree", seed_taxonomy)
        _run_startup_maintenance("Backfilling result taxonomy links", backfill_result_taxonomy)
        _run_startup_maintenance("Backfilling entity resolution map", backfill_entity_resolutions)

        # Skip heavy backfills on startup for faster boot
        # These can run in the background via pipeline
        print("Skipping graph/merge backfill (run in pipeline instead).", flush=True)

        if AUTO_PIPELINE_ENABLED:
            from orchestrator.paper_worker import start as start_paper_worker
            print("Starting paper ingestion worker...", flush=True)
            paper_worker_status = start_paper_worker()
            if paper_worker_status.get("status") not in {
                "started",
                "already_running",
                "already_running_elsewhere",
            }:
                raise RuntimeError(
                    "Paper ingestion worker failed closed during startup: "
                    f"{paper_worker_status.get('status') or 'unknown'}"
                )
            print("Paper ingestion worker ready.", flush=True)

        if SCOPED_INGESTION_WORKER_ENABLED:
            from orchestrator.scoped_ingestion_worker import (
                start as start_scoped_ingestion_worker,
            )

            print("Starting scoped ingestion worker...", flush=True)
            scoped_ingestion_status = start_scoped_ingestion_worker()
            if scoped_ingestion_status.get("status") not in {
                "started",
                "already_running",
            }:
                raise RuntimeError(
                    "Scoped ingestion worker failed closed during startup: "
                    f"{scoped_ingestion_status.get('status') or 'unknown'}"
                )
            print("Scoped ingestion worker ready.", flush=True)

        configured_compute = {
            str(value).strip().lower()
            for value in (COMPUTE_BACKENDS_ENABLED or [])
        }
        if AUTO_RESEARCH_ENABLED or configured_compute.intersection(
            {"local_gpu", "ssh_gpu", "colab_gpu"}
        ):
            from orchestrator.gpu_scheduler import start as start_gpu_scheduler
            # Durable compute reconciliation must finish before the research
            # loop or a configured asynchronous backend worker starts.
            print("Starting compute scheduler and recovery...", flush=True)
            scheduler_status = start_gpu_scheduler()
            if scheduler_status.get("status") not in {
                "started",
                "already_running",
                "already_running_elsewhere",
            }:
                raise RuntimeError(
                    "Compute scheduler failed closed during startup: "
                    f"{scheduler_status.get('status') or 'unknown'}"
                )
            print("Compute scheduler and recovery ready.", flush=True)

        if AUTO_RESEARCH_ENABLED:
            from orchestrator.auto_research import start as start_auto_research

            print("Starting Auto Research worker...", flush=True)
            start_auto_research()
            print("Auto Research worker ready.", flush=True)

        # Warm the /api/stats cache in the background so the first browser paint
        # is served from a warm cache instead of a cold ~30-COUNT(*) query
        # (issue #34).
        from web.app import prewarm_stats_cache
        print("Prewarming stats cache in background...", flush=True)
        threading.Thread(target=prewarm_stats_cache, daemon=True).start()

        # Start web server
        _serve_http()
    finally:
        _release_process_lock()


if __name__ == "__main__":
    main()
