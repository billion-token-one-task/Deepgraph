#!/usr/bin/env python3
"""Work through the paper backlog using only the stages that cost nothing.

Two kinds of work qualify, and neither issues an LLM call:

  text     - papers with no usable full text: fetch and parse (network + CPU)
  graph    - papers whose extraction was already paid for and checkpointed but
             never written into the graph: replay the checkpoint

The paid stages (extraction, contradiction detection) are deliberately not
touched; they need a ResourceGrant and a budget decision.

This box has 2 cores, so "saturate" means a couple of workers plus a guard: the
loop pauses whenever load or memory pressure would start costing the research
system its own responsiveness. Progress is checkpointed per paper, so killing
it at any moment loses at most one paper's work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db  # noqa: E402
from orchestrator.pipeline import _prefetch_single_paper_text, process_single_paper  # noqa: E402


def _log(path: Path, step: str, **fields) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "step": step, **fields}
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(f"[backfill] {line}", flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _pressure(max_load: float, min_free_mb: int) -> str | None:
    load1 = os.getloadavg()[0]
    if load1 > max_load:
        return f"load {load1:.2f} > {max_load}"
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.strip().split()[0]) // 1024
        available = info.get("MemAvailable", 0)
        if available < min_free_mb:
            return f"available memory {available}MB < {min_free_mb}MB"
    except Exception:
        pass
    return None


def _graph_batch(limit: int) -> list[str]:
    return [
        r["id"]
        for r in db.fetchall(
            "SELECT p.id FROM papers p"
            "  JOIN paper_stage_checkpoints c ON c.paper_id=p.id AND c.stage='extracted'"
            " WHERE p.processing_stage='extracted' AND p.status IN ('ingested','error')"
            " ORDER BY p.published_date DESC NULLS LAST LIMIT ?",
            (limit,),
        )
    ]


def _text_batch(limit: int) -> list[str]:
    return [
        r["id"]
        for r in db.fetchall(
            "SELECT id FROM papers"
            " WHERE COALESCE(processing_stage,'ingested')='ingested'"
            "   AND status='ingested'"
            "   AND (full_text IS NULL OR length(full_text) < 100)"
            " ORDER BY published_date DESC NULLS LAST LIMIT ?",
            (limit,),
        )
    ]


def _do_graph(paper_id: str) -> tuple[str, bool, str]:
    try:
        out = process_single_paper(paper_id, stop_after="graph_written")
        if out.get("error"):
            return paper_id, False, str(out["error"])
        return paper_id, True, f"claims={out.get('claims', 0)} entities={out.get('graph_entities', 0)}"
    except Exception as exc:
        db.rollback()
        return paper_id, False, f"{type(exc).__name__}: {exc}"


def _do_text(paper_id: str) -> tuple[str, bool, str]:
    try:
        out = _prefetch_single_paper_text(paper_id)
        return paper_id, bool(out.get("ok")), str(out.get("error") or out.get("text_length", ""))
    except Exception as exc:
        db.rollback()
        return paper_id, False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["graph", "text", "both"], default="both")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--max-papers", type=int, default=0, help="0 = until the queue is empty")
    parser.add_argument("--max-load", type=float, default=1.6)
    parser.add_argument("--min-free-mb", type=int, default=900)
    parser.add_argument("--log", default="/home/ec2-user/deepgraph-reports/backfill_log.jsonl")
    args = parser.parse_args()

    log_path = Path(args.log)
    done = {"graph": 0, "text": 0, "failed": 0}
    started = time.time()
    _log(log_path, "start", mode=args.mode, workers=args.workers,
         pending_graph=len(_graph_batch(100000)), pending_text=len(_text_batch(100000)))

    modes = ["graph", "text"] if args.mode == "both" else [args.mode]
    for mode in modes:
        pick = _graph_batch if mode == "graph" else _text_batch
        work = _do_graph if mode == "graph" else _do_text
        while True:
            if args.max_papers and (done["graph"] + done["text"]) >= args.max_papers:
                break
            reason = _pressure(args.max_load, args.min_free_mb)
            if reason:
                _log(log_path, "paused", reason=reason)
                time.sleep(60)
                continue
            batch = pick(args.batch)
            if not batch:
                break
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(work, pid) for pid in batch]
                for future in as_completed(futures):
                    paper_id, ok, detail = future.result()
                    if ok:
                        done[mode] += 1
                    else:
                        done["failed"] += 1
                        _log(log_path, f"{mode}_failed", paper_id=paper_id, detail=detail[:200])
            _log(log_path, "progress", mode=mode, **done,
                 elapsed_min=round((time.time() - started) / 60, 1))
    _log(log_path, "done", **done, elapsed_min=round((time.time() - started) / 60, 1))
    try:
        db.rollback()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
