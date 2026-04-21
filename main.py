#!/usr/bin/env python3.12
"""DeepGraph - Hierarchical ML Research Knowledge Engine."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_NAME,
    AUTO_RESEARCH_ENABLED,
    BACKFILL_GRAPH_ON_START,
    REFRESH_MERGE_CANDIDATES_ON_START,
    ROOT_NODE_ID,
    WEB_HOST,
    WEB_PORT,
    WORKSPACE_DIR,
    PDF_CACHE_DIR,
)
from db.database import init_db
from db.evidence_graph import (
    backfill_entity_resolutions,
    backfill_graph_from_structured_data,
    refresh_merge_candidates,
)
from db.taxonomy import seed_taxonomy, backfill_result_taxonomy
from web.app import app


def main():
    # Ensure directories exist
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    print("Initializing database...", flush=True)
    init_db()
    print("Database ready.", flush=True)

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
    print(f"Starting {APP_NAME} at http://{WEB_HOST}:{WEB_PORT} (root node: {ROOT_NODE_ID})", flush=True)
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
