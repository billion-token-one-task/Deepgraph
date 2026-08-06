#!/usr/bin/env python3
"""Bounded one-shot ideation seeding for one agenda (V1 glue).

Calls run_tier2_discovery directly with small explicit limits and never touches
harvest_signals: the harvesters pull whole-corpus joins into memory with no
LIMIT, which is what OOM-killed the previous seeding attempt on this 7GB host,
and the papers corpus has not changed since May so a re-harvest adds nothing.
run_tier2_discovery refreshes the agenda-scoped problem pool itself, so this is
the single entry point needed.

V1 operator glue; retired in V2 by the ideation job queue
(docs/upgrade-plan-v1-v2.md). Run it under a memory cap, e.g.:

  sudo systemd-run --scope -p MemoryMax=3G \
    /home/billion-token/Deepgraph/.venv/bin/python \
    /home/billion-token/Deepgraph/scripts/seed_ideation.py --agenda-id 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.discovery_scheduler import run_tier2_discovery  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agenda-id", type=int, required=True)
    parser.add_argument("--max-problems", type=int, default=4)
    parser.add_argument("--max-papers", type=int, default=3)
    args = parser.parse_args()
    if args.agenda_id <= 0:
        parser.error("--agenda-id must be positive")
    if not (1 <= args.max_problems <= 8) or not (1 <= args.max_papers <= 5):
        parser.error("limits out of the V1 seeding bounds (problems 1-8, papers 1-5)")

    started = datetime.now(timezone.utc).isoformat()
    print(f"[SEED] agenda {args.agenda_id} tier2 start {started} "
          f"(max_problems={args.max_problems}, max_papers={args.max_papers})", flush=True)
    stored = run_tier2_discovery(
        args.max_problems,
        args.max_papers,
        agenda_id=args.agenda_id,
        bulk=False,
    )
    summary = {
        "agenda_id": args.agenda_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "stored_insights": stored,
        "count": len(stored),
    }
    print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    # Zero stored ideas is a real (LLM-unavailable or empty-pool) outcome the
    # wrapper must see as failure, not silently treat as seeded.
    return 0 if stored else 2


if __name__ == "__main__":
    raise SystemExit(main())
