#!/usr/bin/env python3
"""Initialize DeepGraph and process one paper end-to-end."""

import argparse

from config import WORKSPACE_DIR, PDF_CACHE_DIR
from db.database import init_db
from db.evidence_graph import backfill_entity_resolutions
from db.taxonomy import seed_taxonomy, backfill_result_taxonomy
from orchestrator.pipeline import ingest_papers, process_single_paper
from db import database as db
from meta_harness.scoped_llm import require_active_scope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agenda-id", required=True, type=int)
    parser.add_argument("--idea-id", required=True, type=int)
    parser.add_argument("--resource-grant-id", required=True, type=int)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    init_db()
    seed_taxonomy()
    backfill_result_taxonomy()
    backfill_entity_resolutions()
    llm_scope = require_active_scope(
        {
            "agenda_id": args.agenda_id,
            "idea_id": args.idea_id,
            "resource_grant_id": args.resource_grant_id,
            "stage": args.stage,
        }
    )

    if db.fetchone("SELECT COUNT(*) as c FROM papers")["c"] == 0:
        ingest_papers(max_papers=1)

    paper = db.fetchone("SELECT id FROM papers ORDER BY published_date DESC LIMIT 1")
    if not paper:
        print("No papers available to process.", flush=True)
        return

    paper_id = paper["id"]
    print(f"processing {paper_id}", flush=True)
    print(process_single_paper(paper_id, llm_scope=llm_scope), flush=True)


if __name__ == "__main__":
    main()
