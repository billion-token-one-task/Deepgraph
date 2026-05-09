#!/usr/bin/env python3
"""Monitor a detached SSH benchmark shard and reconcile its artifacts.

This helper is for manually launched, non-canonical benchmark shards. It does
not launch or terminate experiments. It keeps the assigned worker busy while the
remote process is live, syncs artifacts when it exits, and records whether the
shard produced a benchmark summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db
from orchestrator import ssh_gpu_backend


def _worker(worker_id: str) -> dict:
    row = db.fetchone("SELECT * FROM gpu_workers WHERE id=?", (worker_id,))
    if not row:
        raise RuntimeError(f"GPU worker not found: {worker_id}")
    return {
        "id": row.get("id"),
        "hostname": row.get("hostname"),
        "gpu_index": row.get("gpu_index"),
        "gpu_model": row.get("gpu_model"),
        "total_mem_gb": row.get("total_mem_gb"),
        "metadata": row.get("metadata"),
    }


def _remote_live(worker: dict, remote_workdir: str) -> bool:
    script = "\n".join(
        [
            "set +e",
            f"remote_root=$(readlink -f {json.dumps(remote_workdir)} 2>/dev/null || printf '%s\\n' {json.dumps(remote_workdir)})",
            "for cwd_link in /proc/[0-9]*/cwd; do",
            "  pid=${cwd_link#/proc/}",
            "  pid=${pid%/cwd}",
            "  cwd=$(readlink -f \"$cwd_link\" 2>/dev/null || true)",
            "  case \"$cwd\" in",
            "    \"$remote_root\"|\"$remote_root\"/*) echo \"$pid\"; exit 0 ;;",
            "  esac",
            "done",
            "exit 1",
        ]
    )
    proc = ssh_gpu_backend._run_ssh(worker, script, timeout=60)
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def _remote_progress(worker: dict, remote_workdir: str) -> dict:
    script = f"""python - <<'PY'
import json
from pathlib import Path

root = Path({remote_workdir!r})
results = root / "results"
raw = results / "raw_predictions.jsonl"
run_log = root / "run.log"
payload = {{
    "raw_predictions_lines": 0,
    "raw_predictions_bytes": raw.stat().st_size if raw.exists() else 0,
    "result_files": [],
    "latest_stage": "",
}}
if raw.exists():
    with raw.open("rb") as handle:
        payload["raw_predictions_lines"] = sum(1 for _ in handle)
if results.exists():
    payload["result_files"] = sorted(p.name for p in results.iterdir() if p.is_file())
if run_log.exists():
    latest = ""
    with run_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "BENCHMARK_STAGE:" in line:
                latest = line.strip()
    payload["latest_stage"] = latest[-500:]
print(json.dumps(payload, ensure_ascii=False))
PY"""
    proc = ssh_gpu_backend._run_ssh(worker, script, timeout=60)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "").strip()}
    try:
        return json.loads((proc.stdout or "{}").strip() or "{}")
    except json.JSONDecodeError:
        return {"error": (proc.stdout or "").strip()}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _release_worker(worker_id: str, *, finished_job_id: int | None = None) -> None:
    params: tuple = (worker_id,)
    exclude_sql = ""
    if finished_job_id is not None:
        exclude_sql = " AND id<>?"
        params = (worker_id, finished_job_id)
    active = db.fetchone(
        f"""
        SELECT COUNT(*) AS c
        FROM gpu_jobs
        WHERE assigned_worker=?
          AND status='running'
          {exclude_sql}
        """,
        params,
    )
    status = "busy" if active and int(active.get("c") or 0) > 0 else "idle"
    db.execute("UPDATE gpu_workers SET status=?, heartbeat_at=CURRENT_TIMESTAMP WHERE id=?", (status, worker_id))


def reconcile(run_id: int, job_id: int, worker_id: str, remote_workdir: str, poll_seconds: int) -> dict:
    db.init_db()
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run or not run.get("workdir"):
        raise RuntimeError(f"Run {run_id} is missing or has no workdir")
    local_workdir = Path(run["workdir"])
    worker = _worker(worker_id)

    while _remote_live(worker, remote_workdir):
        progress = _remote_progress(worker, remote_workdir)
        db.execute("UPDATE gpu_workers SET status='busy', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?", (worker_id,))
        db.commit()
        print(
            f"[manual-shard-monitor] run_{run_id} active on {worker_id}; "
            f"raw_lines={progress.get('raw_predictions_lines', '?')} "
            f"raw_bytes={progress.get('raw_predictions_bytes', '?')} "
            f"latest_stage={progress.get('latest_stage') or progress.get('error') or '?'}; "
            f"sleeping {poll_seconds}s",
            flush=True,
        )
        time.sleep(max(10, poll_seconds))

    print(f"[manual-shard-monitor] run_{run_id} finished remotely; syncing artifacts", flush=True)
    ssh_gpu_backend.sync_workdir_from_remote(
        worker=worker,
        remote_workdir=remote_workdir,
        local_workdir=local_workdir,
    )

    results_dir = local_workdir / "results"
    summary = _load_json(results_dir / "benchmark_summary.json")
    manifest = _load_json(results_dir / "artifact_manifest.json")
    full_completed = bool(summary.get("full_benchmark_completed") or manifest.get("full_benchmark_completed"))
    raw_path = results_dir / "raw_predictions.jsonl"
    raw_lines = 0
    if raw_path.exists():
        with raw_path.open("rb") as handle:
            raw_lines = sum(1 for _ in handle)

    if not summary:
        error = "manual benchmark shard finished without benchmark_summary.json"
        db.execute(
            "UPDATE experiment_runs SET status='failed', phase='benchmark_method_shard', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (error, run_id),
        )
        db.execute(
            "UPDATE gpu_jobs SET status='failed', completed_at=CURRENT_TIMESTAMP, artifact_uri=?, error_message=? WHERE id=?",
            (str(local_workdir), error, job_id),
        )
        _release_worker(worker_id, finished_job_id=job_id)
        db.commit()
        return {"run_id": run_id, "status": "failed", "raw_lines": raw_lines, "error": error}

    note = {
        "manual_shard": True,
        "full_benchmark_completed": full_completed,
        "raw_predictions_lines": raw_lines,
        "methods": list((summary.get("per_method") or {}).keys()),
        "datasets": [row.get("name") for row in summary.get("datasets", []) if isinstance(row, dict)],
    }
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', phase='benchmark_method_shard',
               best_metric_value=?, error_message=?, completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (
            (summary.get("per_method") or {}).get("CGGR", {}).get("metric_value"),
            json.dumps(note, ensure_ascii=False),
            run_id,
        ),
    )
    db.execute(
        "UPDATE gpu_jobs SET status='completed', completed_at=CURRENT_TIMESTAMP, artifact_uri=?, error_message=NULL WHERE id=?",
        (str(local_workdir), job_id),
    )
    _release_worker(worker_id, finished_job_id=job_id)
    db.commit()
    return {"run_id": run_id, "status": "completed", **note}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--remote-workdir", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    print(
        json.dumps(
            reconcile(args.run_id, args.job_id, args.worker_id, args.remote_workdir, args.poll_seconds),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
