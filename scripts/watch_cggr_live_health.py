#!/usr/bin/env python3
"""Read-only live health watcher for active CGGR benchmark shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db
from orchestrator.ssh_gpu_backend import _run_ssh


def _worker(worker_id: str) -> dict:
    row = db.fetchone("SELECT * FROM gpu_workers WHERE id=?", (worker_id,))
    if not row:
        raise RuntimeError(f"GPU worker not found: {worker_id}")
    return dict(row)


def _db_status(run_ids: list[int]) -> dict:
    placeholders = ",".join("?" for _ in run_ids)
    runs = db.fetchall(
        f"SELECT id,status,phase,started_at,completed_at,error_message FROM experiment_runs WHERE id IN ({placeholders}) ORDER BY id",
        tuple(run_ids),
    )
    jobs = db.fetchall(
        f"SELECT id,experiment_run_id,status,assigned_worker,started_at,completed_at,error_message FROM gpu_jobs WHERE experiment_run_id IN ({placeholders}) ORDER BY experiment_run_id,id",
        tuple(run_ids),
    )
    return {
        "runs": [dict(row) for row in runs],
        "jobs": [dict(row) for row in jobs],
    }


def _remote_health(worker: dict, run_ids: list[int], remote_template: str) -> dict:
    script = f"""python3 - <<'PY'
import collections
import json
import pathlib
import re
import subprocess

run_ids = {json.dumps(run_ids)}
remote_template = {json.dumps(remote_template)}
error_pattern = re.compile(r"traceback|out of memory|cuda error|runtimeerror|exception|killed|oom|nan", re.I)

def count_lines(path):
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)

def parse_jsonl(path):
    rows = []
    bad = 0
    if not path.exists():
        return rows, bad
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad

def live_pids(root):
    found = []
    root_text = str(root.resolve())
    for cwd_link in pathlib.Path("/proc").glob("[0-9]*/cwd"):
        try:
            cwd = cwd_link.resolve()
        except Exception:
            continue
        cwd_text = str(cwd)
        if cwd_text == root_text or cwd_text.startswith(root_text + "/"):
            found.append(cwd_link.parent.name)
    return found

payload = {{"gpu": "", "runs": {{}}}}
try:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    payload["gpu"] = (proc.stdout or proc.stderr or "").strip()
except Exception as exc:
    payload["gpu"] = f"nvidia-smi failed: {{exc}}"

for run_id in run_ids:
    root = pathlib.Path(remote_template.format(run_id=run_id))
    results = root / "results"
    raw_path = results / "raw_predictions.jsonl"
    fail_path = results / "failure_cases.jsonl"
    log_path = root / "run.log"
    raw_rows, raw_bad = parse_jsonl(raw_path)
    fail_rows, fail_bad = parse_jsonl(fail_path)
    keys = collections.Counter((row.get("method"), row.get("dataset"), row.get("seed"), row.get("example_id")) for row in raw_rows)
    by_cell = collections.Counter((row.get("method"), row.get("dataset"), row.get("seed")) for row in raw_rows)
    fail_stages = collections.Counter(str(row.get("stage") or "<no_stage>") for row in fail_rows)
    fail_methods = collections.Counter(str(row.get("method") or "<no_method>") for row in fail_rows)
    fail_errors = collections.Counter(str(row.get("error_type") or row.get("error") or "<no_error>") for row in fail_rows)
    stage_tail = []
    error_tail = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if "BENCHMARK_STAGE:" in text:
                    stage_tail.append(text)
                    stage_tail = stage_tail[-8:]
                if error_pattern.search(text):
                    error_tail.append(text)
                    error_tail = error_tail[-5:]
    pids = live_pids(root)
    payload["runs"][str(run_id)] = {{
        "remote_root": str(root),
        "live_pid_count": len(pids),
        "live_pid_sample": pids[:5],
        "raw_lines": count_lines(raw_path),
        "raw_bad_json": raw_bad,
        "raw_duplicate_keys": sum(1 for count in keys.values() if count > 1),
        "raw_empty_prediction_like": sum(1 for row in raw_rows if not str(row.get("prediction") or row.get("parsed_answer") or row.get("final_answer") or "").strip()),
        "raw_zero_token_like": sum(1 for row in raw_rows if int(row.get("new_tokens") or row.get("completion_tokens") or 0) <= 0),
        "latest_cells": [
            {{"method": key[0], "dataset": key[1], "seed": key[2], "count": count}}
            for key, count in by_cell.most_common()[-8:]
        ],
        "failure_lines": count_lines(fail_path),
        "failure_bad_json": fail_bad,
        "failure_stages": dict(fail_stages),
        "failure_methods": fail_methods.most_common(8),
        "failure_errors": fail_errors.most_common(5),
        "stage_tail": stage_tail,
        "error_tail": error_tail,
    }}

print(json.dumps(payload, ensure_ascii=False))
PY"""
    proc = _run_ssh(worker, script, timeout=120)
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
        }
    try:
        return {"ok": True, **json.loads((proc.stdout or "{}").strip() or "{}")}
    except json.JSONDecodeError:
        return {"ok": False, "stdout": (proc.stdout or "")[-4000:], "stderr": (proc.stderr or "")[-4000:]}


def sample(worker_id: str, run_ids: list[int], remote_template: str) -> dict:
    db.init_db()
    worker = _worker(worker_id)
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "worker_id": worker_id,
        "db": _db_status(run_ids),
        "remote": _remote_health(worker, run_ids, remote_template),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--run-id", action="append", type=int, required=True)
    parser.add_argument("--remote-template", default="/root/deepgraph-remote-worker/runs/run_{run_id}")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        print("[live-health] " + json.dumps(sample(args.worker_id, args.run_id, args.remote_template), ensure_ascii=False, default=str), flush=True)
        if args.once:
            return
        time.sleep(max(60, args.poll_seconds))


if __name__ == "__main__":
    main()
