#!/usr/bin/env python3
"""Keep the corpus growing: fetch new arXiv papers in the focused categories.

Zero LLM cost by construction - this only talks to the arXiv API and inserts
rows at status='ingested'. Extraction (the part that costs tokens) is a
separate, grant-gated step and is deliberately not triggered here.

The corpus had been frozen since 2026-05-14 because the only caller of
ingest_papers was pipeline.run_continuous, which no live process runs.

Categories follow the owner's stated directions (2026-08-07): biological
evolution, ecosystems and communities, agents, harnesses, and language models.
Override with --category (repeatable) rather than editing the default.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db  # noqa: E402
from ingestion.arxiv_client import search_papers  # noqa: E402
from ingestion.arxiv_ids import arxiv_base_id  # noqa: E402

FOCUS_CATEGORIES = [
    "q-bio.PE",   # populations and evolution: evolution, ecology, communities
    "nlin.AO",    # adaptation and self-organizing systems: collective dynamics
    "cs.NE",      # neural and evolutionary computing: the bridge to ML
    "cs.MA",      # multiagent systems
    "cs.AI",      # agents, planning, tool use
    "cs.CL",      # language models
    "cs.LG",      # learning, benchmarks, harnesses
]


def harvest(categories: list[str], per_category: int, log_path: Path) -> dict:
    started = datetime.now(timezone.utc)
    totals = {"new": 0, "seen": 0, "by_category": {}}
    for category in categories:
        new_here = 0
        try:
            papers = search_papers(categories=[category], max_results=per_category)
        except Exception as exc:
            totals["by_category"][category] = f"error: {type(exc).__name__}: {exc}"
            continue
        for paper in papers:
            totals["seen"] += 1
            base = arxiv_base_id(paper["id"])
            existing = db.fetchone(
                "SELECT id FROM papers WHERE id=? OR arxiv_base_id=? LIMIT 1",
                (paper["id"], base),
            )
            # insert_paper is an upsert: it refreshes metadata on a paper we
            # already hold, which is why it runs on both branches.
            db.insert_paper(paper)
            if not existing:
                new_here += 1
        db.commit()
        totals["by_category"][category] = new_here
        totals["new"] += new_here
    record = {
        "at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "step": "harvest",
        **totals,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(f"[harvest] {line}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", dest="categories",
                        help="repeatable; defaults to the focused set")
    parser.add_argument("--per-category", type=int, default=60,
                        help="newest N per category per run")
    parser.add_argument("--log", default="/home/ec2-user/deepgraph-reports/harvest_log.jsonl")
    args = parser.parse_args()
    categories = args.categories or FOCUS_CATEGORIES
    try:
        harvest(categories, max(1, min(200, args.per_category)), Path(args.log))
    finally:
        try:
            db.rollback()  # leave no idle-in-transaction session behind
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
