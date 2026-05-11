#!/usr/bin/env python3
"""Stage and optionally launch a CGGR top-venue baseline method shard.

This script is intentionally separate from the active run45/run46 -> run47
watcher. It prepares an additional method shard for strict top-venue claims and
only launches it when an SSH GPU worker is idle.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from db import database as db  # noqa: E402
from orchestrator import ssh_gpu_backend  # noqa: E402
from prepare_cggr_top_venue_baseline_shard import TOP_VENUE_METHODS, prepare  # noqa: E402


DEFAULT_SOURCE_RUN_ID = 45
DEFAULT_INSIGHT_ID = 13
DEFAULT_TIMEOUT_S = 172800


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _run_path_from_db(run_id: int) -> Path:
    row = db.fetchone("SELECT workdir FROM experiment_runs WHERE id=?", (int(run_id),))
    if not row or not row.get("workdir"):
        raise RuntimeError(f"source run {run_id} is missing or has no workdir")
    return Path(str(row["workdir"]))


def _source_insight_id(run_id: int, fallback: int) -> int:
    row = db.fetchone("SELECT deep_insight_id FROM experiment_runs WHERE id=?", (int(run_id),))
    if row and row.get("deep_insight_id") is not None:
        return int(row["deep_insight_id"])
    return int(fallback)


def _staged_run(run_id: int) -> dict[str, Any]:
    row = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (int(run_id),))
    if not row:
        raise RuntimeError(f"staged run {run_id} does not exist")
    workdir = str(row.get("workdir") or "").strip()
    if not workdir:
        raise RuntimeError(f"staged run {run_id} has no workdir")
    return {
        "run_id": int(run_id),
        "workdir": workdir,
        "prepare": {"ok": True, "reused_staged_run": True},
    }


def _insert_staged_run(*, insight_id: int, runs_root: Path, source_run: Path) -> tuple[int, Path]:
    note = {
        "manual_method_shard": True,
        "full_scale": True,
        "top_venue_baseline_shard": True,
        "source_run": str(source_run),
        "method_subset": TOP_VENUE_METHODS,
        "launch_policy": "launch only on an idle SSH GPU worker",
    }
    run_id = db.insert_returning_id(
        """
        INSERT INTO experiment_runs
        (deep_insight_id, experiment_suite, status, phase, workdir, codebase_url,
         codebase_ref, baseline_metric_name, resource_class, error_message)
        VALUES (?, 'main', 'pending', 'benchmark_method_shard_staged', '',
                'scratch', 'cggr_top_venue_baseline_shard', 'cost_adjusted_accuracy',
                'gpu_large', ?)
        RETURNING id
        """,
        (int(insight_id), _json_dumps(note)),
    )
    workdir = runs_root / f"run_{int(run_id)}"
    db.execute("UPDATE experiment_runs SET workdir=? WHERE id=?", (str(workdir), int(run_id)))
    db.commit()
    return int(run_id), workdir


def _load_worker(worker_id: str) -> dict[str, Any]:
    row = db.fetchone("SELECT * FROM gpu_workers WHERE id=?", (worker_id,))
    if not row:
        raise RuntimeError(f"GPU worker not found: {worker_id}")
    return dict(row)


def _idle_ssh_worker(worker_id: str | None = None) -> dict[str, Any] | None:
    params: list[Any] = ['%"backend": "ssh"%']
    worker_sql = ""
    if worker_id:
        worker_sql = " AND id=?"
        params.append(worker_id)
    rows = db.fetchall(
        f"""
        SELECT *
        FROM gpu_workers
        WHERE status='idle'
          AND metadata LIKE ?
          {worker_sql}
          AND NOT EXISTS (
              SELECT 1 FROM gpu_jobs
              WHERE assigned_worker=gpu_workers.id
                AND status='running'
          )
        ORDER BY gpu_index, id
        LIMIT 1
        """,
        tuple(params),
    )
    return dict(rows[0]) if rows else None


def _benchmark_env_from_shard(workdir: Path) -> dict[str, str]:
    shard_config = json.loads((workdir / "spec" / "shard_config.json").read_text(encoding="utf-8"))
    env = ssh_gpu_backend.benchmark_env_from_workdir(workdir)
    for key, value in (shard_config.get("env") or {}).items():
        if str(key).startswith("DEEPGRAPH_BENCHMARK_") and str(value).strip():
            env[str(key)] = str(value)
    return env


def _remote_python_tokens(worker: Mapping[str, Any]) -> list[str]:
    # The existing SSH backend maps the local Python command to the configured
    # remote interpreter. The template command is always "python train.py".
    return ssh_gpu_backend._remote_python(  # type: ignore[attr-defined]
        worker,
        ["python", "train.py"],
        sys.executable,
    )


def _write_and_start_remote_launcher(
    *,
    worker: Mapping[str, Any],
    run_id: int,
    local_workdir: Path,
    timeout_s: int,
    run_remote_setup: bool,
) -> dict[str, Any]:
    remote_workdir, remote_code_dir = ssh_gpu_backend._remote_paths(  # type: ignore[attr-defined]
        worker,
        int(run_id),
        local_workdir,
    )
    ssh_gpu_backend.cleanup_remote_run_processes(
        worker=worker,
        run_id=int(run_id),
        remote_workdir=remote_workdir,
    )
    ssh_gpu_backend.sync_workdir_to_remote(
        worker=worker,
        local_workdir=local_workdir,
        remote_workdir=remote_workdir,
    )
    command_tokens = _remote_python_tokens(worker)
    if run_remote_setup:
        ssh_gpu_backend._install_remote_repo_deps(  # type: ignore[attr-defined]
            worker=worker,
            remote_code_dir=remote_code_dir,
            remote_python=command_tokens[0],
            remote_workdir=remote_workdir,
        )

    metadata = ssh_gpu_backend._load_metadata(worker)  # type: ignore[attr-defined]
    visible_device = str(metadata.get("visible_device") or "0")
    launcher = ssh_gpu_backend._remote_launcher_script(  # type: ignore[attr-defined]
        run_id=int(run_id),
        remote_code_dir=remote_code_dir,
        remote_workdir=remote_workdir,
        visible_device=visible_device,
        command_tokens=command_tokens,
        benchmark_env=_benchmark_env_from_shard(local_workdir),
    )
    paths = ssh_gpu_backend._remote_run_paths(remote_workdir, int(run_id))  # type: ignore[attr-defined]
    delimiter = f"DEEPGRAPH_REMOTE_LAUNCHER_TOPVENUE_{int(run_id)}"
    while delimiter in launcher:
        delimiter += "_END"
    timeout_command = f"timeout --kill-after=30s {max(1, int(timeout_s) + 60)}s {shlex.quote(paths['launcher'])}"
    remote_script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(remote_workdir)}",
            f"cat > {shlex.quote(paths['launcher'])} <<'{delimiter}'",
            launcher,
            delimiter,
            f"chmod +x {shlex.quote(paths['launcher'])}",
            f"cd {shlex.quote(remote_workdir)}",
            f"nohup bash -lc {shlex.quote(timeout_command)} >/dev/null 2>&1 < /dev/null &",
            "echo $!",
        ]
    )
    proc = ssh_gpu_backend._run_ssh(worker, remote_script, timeout=120)  # type: ignore[attr-defined]
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "remote detached launch failed").strip())
    return {
        "remote_pid": (proc.stdout or "").strip().splitlines()[-1],
        "remote_workdir": remote_workdir,
        "remote_code_dir": remote_code_dir,
        "visible_device": visible_device,
    }


def _insert_running_job(
    *,
    insight_id: int,
    run_id: int,
    worker_id: str,
    timeout_s: int,
    priority: int,
) -> int:
    job_id = db.insert_returning_id(
        """
        INSERT INTO gpu_jobs
        (deep_insight_id, experiment_run_id, resource_class, gpu_count,
         vram_required_gb, timeout_s, priority, status, assigned_worker,
         started_at)
        VALUES (?, ?, 'gpu_large', 1, 40, ?, ?, 'running', ?, CURRENT_TIMESTAMP)
        RETURNING id
        """,
        (int(insight_id), int(run_id), int(timeout_s), int(priority), worker_id),
    )
    db.execute(
        """
        UPDATE experiment_runs
        SET status='running_gpu', phase='benchmark_method_shard',
            started_at=COALESCE(started_at, CURRENT_TIMESTAMP)
        WHERE id=?
        """,
        (int(run_id),),
    )
    db.execute("UPDATE gpu_workers SET status='busy', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?", (worker_id,))
    db.commit()
    return int(job_id)


def _mark_launch_failed(run_id: int, job_id: int | None, worker_id: str | None, error: str) -> None:
    db.execute(
        "UPDATE experiment_runs SET status='failed', phase='benchmark_method_shard_launch_failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
        (error, int(run_id)),
    )
    if job_id is not None:
        db.execute(
            "UPDATE gpu_jobs SET status='failed', completed_at=CURRENT_TIMESTAMP, error_message=? WHERE id=?",
            (error, int(job_id)),
        )
    if worker_id:
        db.execute(
            """
            UPDATE gpu_workers
            SET status=CASE WHEN EXISTS (
                SELECT 1 FROM gpu_jobs
                WHERE assigned_worker=? AND status='running' AND id<>COALESCE(?, -1)
            ) THEN 'busy' ELSE 'idle' END,
                heartbeat_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (worker_id, int(job_id or -1), worker_id),
        )
    db.commit()


def _start_monitor(
    *,
    run_id: int,
    job_id: int,
    worker_id: str,
    remote_workdir: str,
    poll_seconds: int,
    log_prefix: str,
) -> dict[str, str]:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{log_prefix}.out.log"
    err_path = log_dir / f"{log_prefix}.err.log"
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


def stage(source_run: Path, runs_root: Path, insight_id: int) -> dict[str, Any]:
    run_id, workdir = _insert_staged_run(
        insight_id=int(insight_id),
        runs_root=runs_root,
        source_run=source_run,
    )
    result = prepare(source_run, workdir, force=False)
    return {"run_id": run_id, "workdir": str(workdir), "prepare": result}


def launch_if_idle(
    *,
    run_id: int,
    workdir: Path,
    insight_id: int,
    worker_id: str | None,
    timeout_s: int,
    priority: int,
    poll_seconds: int,
    run_remote_setup: bool,
    start_monitor: bool,
) -> dict[str, Any]:
    worker = _idle_ssh_worker(worker_id)
    if not worker:
        return {
            "ok": False,
            "reason": "no idle SSH GPU worker",
            "run_id": int(run_id),
            "workdir": str(workdir),
        }

    selected_worker = str(worker["id"])
    job_id: int | None = None
    try:
        job_id = _insert_running_job(
            insight_id=int(insight_id),
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
            "full_scale": True,
            "top_venue_baseline_shard": True,
            "method_subset": TOP_VENUE_METHODS,
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
                log_prefix=f"monitor-run{int(run_id)}-topvenue-baseline",
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
    parser.add_argument("--source-run-id", type=int, default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--staged-run-id", type=int)
    parser.add_argument("--deep-insight-id", type=int)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--launch-if-idle", action="store_true")
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--worker-id")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--idle-poll-seconds", type=int, default=300)
    parser.add_argument("--skip-remote-setup", action="store_true")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args()

    db.init_db()
    if args.staged_run_id is not None:
        staged = _staged_run(args.staged_run_id)
        row = db.fetchone("SELECT deep_insight_id FROM experiment_runs WHERE id=?", (int(args.staged_run_id),))
        insight_id = int(args.deep_insight_id or (row.get("deep_insight_id") if row else DEFAULT_INSIGHT_ID))
        print("[topvenue-stage] " + json.dumps(staged, ensure_ascii=False), flush=True)
    else:
        source_run = (args.source_run or _run_path_from_db(args.source_run_id)).resolve()
        if not source_run.exists():
            raise SystemExit(f"source run does not exist: {source_run}")
        insight_id = int(args.deep_insight_id or _source_insight_id(args.source_run_id, DEFAULT_INSIGHT_ID))
        runs_root = (args.runs_root or source_run.parent).resolve()
        staged = stage(source_run, runs_root, insight_id)
        print("[topvenue-stage] " + json.dumps(staged, ensure_ascii=False), flush=True)

    if not args.launch_if_idle and not args.wait_for_idle:
        print(json.dumps({"ok": True, "staged": staged, "launched": False}, indent=2, ensure_ascii=False))
        return

    while True:
        launched = launch_if_idle(
            run_id=int(staged["run_id"]),
            workdir=Path(staged["workdir"]),
            insight_id=insight_id,
            worker_id=args.worker_id,
            timeout_s=int(args.timeout_s),
            priority=int(args.priority),
            poll_seconds=int(args.poll_seconds),
            run_remote_setup=not args.skip_remote_setup,
            start_monitor=not args.no_monitor,
        )
        print("[topvenue-launch] " + json.dumps(launched, ensure_ascii=False), flush=True)
        if launched.get("ok") or not args.wait_for_idle:
            print(json.dumps({"ok": bool(launched.get("ok")), "staged": staged, "launch": launched}, indent=2, ensure_ascii=False))
            return
        time.sleep(max(30, int(args.idle_poll_seconds)))


if __name__ == "__main__":
    main()
