#!/usr/bin/env python3
"""V1 execution worker: run the granted candidates the advancer produced.

The system's execution machinery (forge -> experiment_run -> gpu_job ->
validation) lives in orchestrator.auto_research._launch_candidates_to_capacity,
and that function has no caller anywhere in the tree - the background
auto-research loop only runs the *selection* cycle. So grants pile up at
stage='portfolio_granted' and nothing ever executes them. This process is that
missing caller.

It must be a long-lived process, not a timer one-shot: the execution path
spawns daemon worker threads, which die with their parent. Entry into the
candidate pool is already gated by an active ResourceGrant, so this worker
inherits the grant discipline rather than adding its own.

The GPU half of the job is not done here - the scheduler thread inside the web
process claims queued gpu_jobs and dispatches them to the A100 workers.

V1 glue; retired in V2 by the auto-execution worker (docs/upgrade-plan-v1-v2.md).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db  # noqa: E402
from meta_harness.outcome_finalizer import finalize_terminal_outcomes  # noqa: E402
from orchestrator.auto_research import _launch_candidates_to_capacity  # noqa: E402

_stop = False


def _handle_stop(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True


def _log(path: Path, step: str, **fields) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "step": step, **fields}
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(f"[execute] {line}", flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _granted_jobs() -> dict[str, int]:
    """Job states that still hold an active grant, for the pass log."""
    rows = db.fetchall(
        """SELECT arj.status AS s, COUNT(*) AS c
             FROM auto_research_jobs arj
             JOIN resource_grants rg ON rg.id = arj.resource_grant_id
            WHERE rg.status='active' AND rg.expires_at > CURRENT_TIMESTAMP
            GROUP BY arj.status"""
    )
    return {str(dict(r)["s"]): int(dict(r)["c"]) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log", default="/home/ec2-user/deepgraph-reports/auto_execute_log.jsonl")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    log_path = Path(args.log)
    _log(log_path, "worker_start", interval=args.interval, once=args.once)

    while not _stop:
        try:
            # Always call it: the function's own preamble recovers stale runs
            # and repairs benchmark-harness jobs, and those states are not in
            # the queued set - gating on "queued" work would strand them.
            granted = _granted_jobs()
            before = finalize_terminal_outcomes().to_dict()
            result = _launch_candidates_to_capacity()
            after = finalize_terminal_outcomes().to_dict()
            scheduled = result.get("scheduled") if isinstance(result, dict) else None
            detail = {k: v for k, v in (result or {}).items()
                      if k != "scheduled" and v not in (0, {}, [], None)}
            _log(log_path, "launch_pass", granted_jobs=granted,
                 scheduled=scheduled, detail=detail,
                 outcomes_before=before, outcomes_after=after)
        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass
            _log(log_path, "launch_failed", reason=f"{type(exc).__name__}: {exc}")
        if args.once:
            break
        for _ in range(max(1, args.interval)):
            if _stop:
                break
            time.sleep(1)
    _log(log_path, "worker_stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
