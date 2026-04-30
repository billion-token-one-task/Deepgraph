#!/usr/bin/env python3.12
"""DeepGraph - Hierarchical ML Research Knowledge Engine."""
import fcntl
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_NAME,
    AUTO_RESEARCH_ENABLED,
    BACKFILL_GRAPH_ON_START,
    IDEA_WORKSPACE_DIR,
    REFRESH_MERGE_CANDIDATES_ON_START,
    ROOT_NODE_ID,
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

_PROCESS_LOCK_HANDLE = None
_PROCESS_LOCK_PATH = Path("/tmp/deepgraph-main.lock")


def _current_lock_owner() -> str | None:
    try:
        owner = _PROCESS_LOCK_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return owner or None


def _try_acquire_process_lock() -> bool:
    global _PROCESS_LOCK_HANDLE
    if _PROCESS_LOCK_HANDLE is not None:
        return True
    _PROCESS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(_PROCESS_LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    _PROCESS_LOCK_HANDLE = handle
    return True


def _release_process_lock() -> None:
    global _PROCESS_LOCK_HANDLE
    if _PROCESS_LOCK_HANDLE is None:
        return
    try:
        fcntl.flock(_PROCESS_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        _PROCESS_LOCK_HANDLE.close()
    finally:
        _PROCESS_LOCK_HANDLE = None


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

        # Seed taxonomy tree
        print("Seeding taxonomy tree...", flush=True)
        seed_taxonomy()
        print("Taxonomy ready.", flush=True)

        print("Backfilling result taxonomy links...", flush=True)
        backfill_result_taxonomy()
        print("Result taxonomy ready.", flush=True)

        print("Backfilling entity resolution map...", flush=True)
        backfill_entity_resolutions()
        print("Entity resolutions ready.", flush=True)

        # Skip heavy backfills on startup for faster boot
        # These can run in the background via pipeline
        print("Skipping graph/merge backfill (run in pipeline instead).", flush=True)

        if AUTO_RESEARCH_ENABLED:
            from orchestrator.auto_research import start as start_auto_research
            from orchestrator.gpu_scheduler import start as start_gpu_scheduler
            print("Starting Auto Research worker...", flush=True)
            start_auto_research()
            print("Starting GPU scheduler...", flush=True)
            start_gpu_scheduler()
            print("Auto Research worker ready.", flush=True)

        # Start web server
        _serve_http()
    finally:
        _release_process_lock()


if __name__ == "__main__":
    main()
