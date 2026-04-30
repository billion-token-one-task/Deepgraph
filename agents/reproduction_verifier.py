from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, record_artifact
from db import database as db


DEFAULT_TIMEOUT_SECONDS = 120


def _safe_tail(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _default_commands() -> list[list[str]]:
    return [
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_safe_rl_cmdp_benchmark",
            "tests.test_benchmark_suite",
            "tests.test_statistical_reporter",
        ]
    ]


def run_reproduction_check(run_id: int, commands: list[list[str]] | None = None,
                           timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Run local reproduction smoke checks and write a structured artifact.

    The check is deliberately small: it verifies the benchmark capability, suite
    integration, and statistical reporter from the current checkout using the
    active Python interpreter. It does not claim container-level portability.
    """
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"status": "error", "reason": "run_not_found", "run_id": run_id}

    workdir = Path(run.get("workdir") or "")
    repo_root = Path.cwd()
    ensure_artifact_dirs(workdir)
    checks = []
    overall_status = "ok"
    for command in commands or _default_commands():
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            if exit_code != 0:
                overall_status = "failed"
            checks.append({
                "command": command,
                "exit_code": exit_code,
                "duration_seconds": time.time() - started,
                "stdout_tail": _safe_tail(completed.stdout),
                "stderr_tail": _safe_tail(completed.stderr),
            })
        except subprocess.TimeoutExpired as exc:
            overall_status = "failed"
            checks.append({
                "command": command,
                "exit_code": None,
                "duration_seconds": time.time() - started,
                "timeout_seconds": timeout_seconds,
                "stdout_tail": _safe_tail(exc.stdout or ""),
                "stderr_tail": _safe_tail(exc.stderr or ""),
                "error": "timeout",
            })

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "status": overall_status,
        "python_executable": sys.executable,
        "repo_root": str(repo_root),
        "scope": "local checkout smoke reproduction; not a clean container proof",
        "checks": checks,
    }
    path = artifact_path(workdir, "artifacts/results/reproduction_check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    record_artifact(workdir, run_id, "reproduction_check", path, {
        "schema_version": 1,
        "status": overall_status,
    })
    return {"status": overall_status, "run_id": run_id, "path": str(path), "checks": len(checks)}
