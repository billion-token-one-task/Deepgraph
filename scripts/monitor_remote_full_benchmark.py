#!/usr/bin/env python3
"""Monitor a detached SSH full-benchmark run and reconcile local artifacts.

This is a recovery helper for the rare case where the controller returns before
the remote benchmark process exits. It does not launch or kill experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.validation_loop import (
    _benchmark_scores,
    _determine_final_verdict,
    _parse_benchmark_summary_from_log,
    _read_success_criteria,
    _write_benchmark_artifact_manifest,
)
from config import EXPERIMENT_REFUTE_MIN_ITERS
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
    payload["result_files"] = sorted(
        [p.name for p in results.iterdir() if p.is_file()]
    )
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


def _release_worker(worker_id: str, job_id: int) -> None:
    active = db.fetchone(
        "SELECT COUNT(*) AS c FROM gpu_jobs WHERE assigned_worker=? AND status='running' AND id<>?",
        (worker_id, job_id),
    )
    status = "busy" if active and int(active.get("c") or 0) > 0 else "idle"
    db.execute("UPDATE gpu_workers SET status=?, heartbeat_at=CURRENT_TIMESTAMP WHERE id=?", (status, worker_id))


def reconcile(run_id: int, job_id: int, worker_id: str, poll_seconds: int) -> dict:
    db.init_db()
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run or not run.get("workdir"):
        raise RuntimeError(f"Run {run_id} is missing or has no workdir")
    workdir = Path(run["workdir"])
    worker = _worker(worker_id)
    remote_workdir, _ = ssh_gpu_backend._remote_paths(worker, run_id, workdir)

    while _remote_live(worker, remote_workdir):
        progress = _remote_progress(worker, remote_workdir)
        print(
            f"[monitor] run_{run_id} still active on {worker_id}; "
            f"raw_lines={progress.get('raw_predictions_lines', '?')} "
            f"raw_bytes={progress.get('raw_predictions_bytes', '?')} "
            f"latest_stage={progress.get('latest_stage') or progress.get('error') or '?'}; "
            f"sleeping {poll_seconds}s",
            flush=True,
        )
        time.sleep(max(10, poll_seconds))

    print(f"[monitor] run_{run_id} finished remotely; syncing artifacts", flush=True)
    ssh_gpu_backend.sync_workdir_from_remote(
        worker=worker,
        remote_workdir=remote_workdir,
        local_workdir=workdir,
    )

    results_dir = workdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "benchmark_summary.json"
    benchmark_summary = _load_json(summary_path)
    if not benchmark_summary:
        benchmark_summary = _parse_benchmark_summary_from_log(workdir / "run.log")
        if benchmark_summary:
            summary_path.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")

    if not benchmark_summary:
        error = "remote full benchmark finished without benchmark_summary.json or FINAL_RESULTS"
        db.execute(
            "UPDATE experiment_runs SET status='failed', phase='full_benchmark', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (error, run_id),
        )
        db.execute(
            "UPDATE gpu_jobs SET status='failed', completed_at=CURRENT_TIMESTAMP, error_message=? WHERE id=?",
            (error, job_id),
        )
        _release_worker(worker_id, job_id)
        db.commit()
        return {"run_id": run_id, "status": "failed", "error": error}

    criteria = _read_success_criteria(workdir, int(run.get("deep_insight_id") or 0) or None)
    metric_name = str(criteria.get("metric_name") or benchmark_summary.get("primary_metric") or "metric")
    direction = str(criteria.get("metric_direction") or "higher")
    _, _, candidate_value, baseline_value, _ = _benchmark_scores(benchmark_summary)
    baseline = float(baseline_value if baseline_value is not None else run.get("baseline_metric_value") or 0.0)
    best_value = float(candidate_value if candidate_value is not None else run.get("best_metric_value") or baseline)
    verdict = _determine_final_verdict(
        baseline=baseline,
        best_value=best_value,
        direction=direction,
        criteria=criteria,
        total_iters=0,
        total_kept=0,
        refute_min=EXPERIMENT_REFUTE_MIN_ITERS,
        benchmark_summary=benchmark_summary,
    )
    effect = best_value - baseline if direction == "higher" else baseline - best_value
    effect_pct = (effect / abs(baseline) * 100.0) if baseline else 0.0
    validation_summary_path = results_dir / "validation_summary.json"
    artifact_path, full_completed = _write_benchmark_artifact_manifest(
        workdir,
        run_id=run_id,
        metric_name=metric_name,
        benchmark_summary=benchmark_summary,
        criteria=criteria,
        verdict=verdict,
        validation_summary_path=validation_summary_path,
    )
    validation_summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "verdict": verdict,
                "baseline": baseline,
                "best_value": best_value,
                "effect_size": effect,
                "effect_pct": effect_pct,
                "benchmark_summary": benchmark_summary,
                "full_benchmark_completed": full_completed,
                "benchmark_artifact_manifest": str(artifact_path) if artifact_path else "",
                "reconciled_by": "scripts/monitor_remote_full_benchmark.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    db.execute(
        """UPDATE experiment_runs
           SET status='completed', phase='full_benchmark', hypothesis_verdict=?,
               baseline_metric_value=?, best_metric_value=?, effect_size=?, effect_pct=?,
               error_message=NULL, completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (verdict, baseline, best_value, effect, effect_pct, run_id),
    )
    db.execute(
        "UPDATE gpu_jobs SET status='completed', completed_at=CURRENT_TIMESTAMP, artifact_uri=?, error_message=NULL WHERE id=?",
        (str(workdir), job_id),
    )
    db.execute(
        """UPDATE auto_research_jobs
           SET status='completed', stage='full_benchmark_complete',
               last_note=?, assigned_worker=NULL, updated_at=CURRENT_TIMESTAMP,
               last_checked_at=CURRENT_TIMESTAMP
           WHERE experiment_run_id=?""",
        (f"Full benchmark reconciled: verdict={verdict}, full_benchmark_completed={full_completed}.", run_id),
    )
    _release_worker(worker_id, job_id)
    db.commit()
    return {
        "run_id": run_id,
        "status": "completed",
        "verdict": verdict,
        "full_benchmark_completed": full_completed,
        "artifact_manifest": str(artifact_path) if artifact_path else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.run_id, args.job_id, args.worker_id, args.poll_seconds), indent=2), flush=True)


if __name__ == "__main__":
    main()
