#!/usr/bin/env python3.12
"""DeepGraph - Hierarchical ML Research Knowledge Engine."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_NAME, ROOT_NODE_ID, WEB_HOST, WEB_PORT, WORKSPACE_DIR, PDF_CACHE_DIR
from db.database import init_db
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

    # Start web server
    print(f"Starting {APP_NAME} at http://{WEB_HOST}:{WEB_PORT} (root node: {ROOT_NODE_ID})", flush=True)
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
