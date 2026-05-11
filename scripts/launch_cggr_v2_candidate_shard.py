#!/usr/bin/env python3
"""Launch a preregistered CGGR-v2 candidate shard on an idle SSH GPU worker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from db import database as db  # noqa: E402
from stage_and_launch_cggr_top_venue_baseline_shard import (  # noqa: E402
    _idle_ssh_worker,
    _insert_running_job,
    _json_dumps,
    _mark_launch_failed,
    _write_and_start_remote_launcher,
)


DEFAULT_TIMEOUT_S = 172800


def _load_run(run_id: int) -> dict[str, Any]:
    row = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (int(run_id),))
    if not row:
        raise RuntimeError(f"experiment run not found: {run_id}")
    if not row.get("workdir"):
        raise RuntimeError(f"experiment run {run_id} has no workdir")
    return dict(row)


def _load_shard_config(workdir: Path) -> dict[str, Any]:
    path = workdir / "spec" / "shard_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid shard_config.json at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"shard_config.json is not an object: {path}")
    return payload


def _start_monitor(
    *,
    run_id: int,
    job_id: int,
    worker_id: str,
    remote_workdir: str,
    poll_seconds: int,
) -> dict[str, str]:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"monitor-run{int(run_id)}-cggr-v2.out.log"
    err_path = log_dir / f"monitor-run{int(run_id)}-cggr-v2.err.log"
    cmd = [
        sys.executable,
        "scripts/monitor_remote_manual_shard.py",
        "--run-id",
        str(int(run_id)),
        "--job-id",
        str(int(job_id)),
        "--worker-id",
        worker_id,
        "--remote-workdir",
        remote_workdir,
        "--poll-seconds",
        str(int(poll_seconds)),
    ]
    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    with out_path.open("ab") as out, err_path.open("ab") as err:
        subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    return {"stdout": str(out_path), "stderr": str(err_path), "command": " ".join(cmd)}


def launch_once(
    *,
    run_id: int,
    worker_id: str | None,
    timeout_s: int,
    priority: int,
    poll_seconds: int,
    run_remote_setup: bool,
    start_monitor: bool,
) -> dict[str, Any]:
    run = _load_run(run_id)
    workdir = Path(str(run["workdir"]))
    shard_config = _load_shard_config(workdir)
    if str(shard_config.get("launch_role") or "") != "paper_benchmark_v2_candidate_shard":
        raise RuntimeError(
            f"run {run_id} is not marked as a paper_benchmark_v2_candidate_shard; "
            "refusing to launch with the v2 candidate launcher"
        )
    worker = _idle_ssh_worker(worker_id)
    if not worker:
        return {"ok": False, "reason": "no idle SSH GPU worker", "run_id": int(run_id), "workdir": str(workdir)}

    selected_worker = str(worker["id"])
    insight_id = int(run.get("deep_insight_id") or 13)
    job_id: int | None = None
    try:
        job_id = _insert_running_job(
            insight_id=insight_id,
            run_id=int(run_id),
            worker_id=selected_worker,
            timeout_s=int(timeout_s),
            priority=int(priority),
        )
        remote = _write_and_start_remote_launcher(
            worker=worker,
            run_id=int(run_id),
            local_workdir=workdir,
            timeout_s=int(timeout_s),
            run_remote_setup=run_remote_setup,
        )
        note = {
            "manual_method_shard": True,
            "cggr_v2_candidate": True,
            "full_scale_candidate_shard": True,
            "method_subset": shard_config.get("method_subset"),
            "candidate_change": shard_config.get("candidate_change"),
            "remote_workdir": remote["remote_workdir"],
            "remote_pid": remote["remote_pid"],
            "worker_id": selected_worker,
        }
        db.execute(
            "UPDATE experiment_runs SET error_message=? WHERE id=?",
            (_json_dumps(note), int(run_id)),
        )
        db.commit()
        monitor = {}
        if start_monitor:
            monitor = _start_monitor(
                run_id=int(run_id),
                job_id=int(job_id),
                worker_id=selected_worker,
                remote_workdir=str(remote["remote_workdir"]),
                poll_seconds=int(poll_seconds),
            )
        return {
            "ok": True,
            "run_id": int(run_id),
            "job_id": int(job_id),
            "worker_id": selected_worker,
            **remote,
            "monitor": monitor,
        }
    except Exception as exc:
        _mark_launch_failed(int(run_id), job_id, selected_worker, str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--worker-id")
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--idle-poll-seconds", type=int, default=300)
    parser.add_argument("--skip-remote-setup", action="store_true")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args()

    db.init_db()
    while True:
        result = launch_once(
            run_id=int(args.run_id),
            worker_id=args.worker_id,
            timeout_s=int(args.timeout_s),
            priority=int(args.priority),
            poll_seconds=int(args.poll_seconds),
            run_remote_setup=not args.skip_remote_setup,
            start_monitor=not args.no_monitor,
        )
        print("[cggr-v2-launch] " + json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("ok") or not args.wait_for_idle:
            return
        time.sleep(max(30, int(args.idle_poll_seconds)))


if __name__ == "__main__":
    main()
