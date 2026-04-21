"""Codex CLI-backed repo editing for experiment iterations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CODEX_CLI_PATH, CODEX_EXEC_ENABLED, CODEX_MODEL, CODEX_TIMEOUT_SECONDS


def codex_binary() -> str | None:
    if not CODEX_EXEC_ENABLED:
        return None
    candidate = (CODEX_CLI_PATH or "").strip()
    if candidate and Path(candidate).exists():
        return candidate
    found = shutil.which("codex")
    return found


def codex_available() -> bool:
    return bool(codex_binary())


def _trim_json(obj: Any, max_chars: int = 2_500) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    return text[:max_chars]


def _parse_thread_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        raw = line.strip()
        if not raw.startswith("{"):
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if payload.get("type") == "thread.started" and payload.get("thread_id"):
            return str(payload["thread_id"])
    return None


def _load_session_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_session_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_last_message_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def _history_text(history: list[dict], limit: int = 8) -> str:
    if not history:
        return "No prior hypothesis-testing iterations yet."
    lines: list[str] = []
    for row in history[-limit:]:
        status = row.get("status", "?")
        metric = row.get("metric")
        description = str(row.get("description") or "").strip()
        lines.append(f"- iter {row.get('iteration', '?')}: status={status} metric={metric} change={description[:180]}")
    return "\n".join(lines)


def write_iteration_agents_md(
    *,
    code_dir: Path,
    method_desc: str,
    baseline: float | None,
    best_so_far: float | None,
    iteration: int,
    history: list[dict],
    proxy: dict[str, Any],
    success_criteria: dict[str, Any],
    experimental_plan: dict[str, Any],
    evidence_plan: dict[str, Any],
    supervisor_plan: dict[str, Any] | None = None,
) -> Path:
    """Write a focused AGENTS.md for the Codex worker inside the code repo."""
    metric_name = success_criteria.get("metric_name", "metric")
    metric_direction = success_criteria.get("metric_direction", "higher")
    main_train_file = proxy.get("main_train_file") or "auto-detect"
    baseline_command = proxy.get("baseline_command") or "auto-detect"

    content = f"""# DeepGraph Experiment Agent

You are executing one iteration of a DeepGraph experiment inside this repository.

## Goal
- Iteration: {iteration}
- Baseline metric ({metric_name}): {baseline}
- Best so far ({metric_name}): {best_so_far}
- Metric direction: {metric_direction}
- Main train file hint: {main_train_file}
- Baseline command hint: {baseline_command}

Make one focused repo-level change that improves the hypothesis under test.
If the previous attempt crashed, prioritize fixing the crash before trying a new idea.
If the previous attempt was discarded, try a meaningfully different approach.
If the previous attempt was kept, build on it conservatively.

## Method To Implement
{method_desc[:4000]}

## Success Criteria
{_trim_json(success_criteria)}

## Experimental Plan
{_trim_json(experimental_plan)}

## Evidence Plan
{_trim_json(evidence_plan)}

Honor the evidence plan. Do not invent ablations or visual analyses when they are disabled.

## Supervisor Plan
{_trim_json(supervisor_plan or {})}

## Iteration History
{_history_text(history)}

## Working Rules
- You may inspect and edit multiple files in this repo if needed.
- Keep changes minimal and hypothesis-directed.
- Prefer patching the existing training/evaluation pipeline rather than rewriting the repo.
- Do not delete unrelated files.
- Do not change dataset paths or external infrastructure unless required to make the experiment runnable.
- You may run local commands/tests to validate your change.

## Final Response
Return a concise JSON object with:
- summary
- files_changed
- commands_run
- validation_status
"""
    codex_dir = code_dir.parent / "codex"
    iter_dir = codex_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    archive_path = iter_dir / "AGENTS.md"
    archive_path.write_text(content, encoding="utf-8")
    repo_path = code_dir / "AGENTS.md"
    repo_path.write_text(content, encoding="utf-8")
    return archive_path


def run_codex_iteration(
    *,
    workdir: Path,
    code_dir: Path,
    iteration: int,
    method_desc: str,
    best_so_far: float | None,
    baseline: float | None,
    history: list[dict],
    proxy: dict[str, Any],
    success_criteria: dict[str, Any],
    experimental_plan: dict[str, Any],
    evidence_plan: dict[str, Any],
    supervisor_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Codex CLI against the experiment repo for one iteration."""
    binary = codex_binary()
    if not binary:
        return {"ok": False, "error": "codex_unavailable"}

    agents_path = write_iteration_agents_md(
        code_dir=code_dir,
        method_desc=method_desc,
        baseline=baseline,
        best_so_far=best_so_far,
        iteration=iteration,
        history=history,
        proxy=proxy,
        success_criteria=success_criteria,
        experimental_plan=experimental_plan,
        evidence_plan=evidence_plan,
        supervisor_plan=supervisor_plan,
    )

    out_dir = workdir / "codex" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"iter_{iteration:03d}_last_message.json"
    schema_path = out_dir / f"iter_{iteration:03d}_output_schema.json"
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "commands_run": {"type": "array", "items": {"type": "string"}},
            "validation_status": {"type": "string"},
        },
        "required": ["summary", "files_changed", "commands_run", "validation_status"],
        "additionalProperties": False,
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    session_state_path = workdir / "codex" / "session.json"
    session_state = _load_session_state(session_state_path)
    thread_id = str(session_state.get("thread_id") or "").strip()

    prompt = (
        "Read AGENTS.md in the repository root, inspect the codebase, follow the supervisor plan, "
        "make one focused change to improve the experiment, and return the required JSON summary."
    )

    env = os.environ.copy()
    result_path = out_dir / f"iter_{iteration:03d}_result.json"
    def _run_once(use_resume: bool) -> subprocess.CompletedProcess[str]:
        if use_resume and thread_id:
            cmd = [
                binary,
                "exec",
                "resume",
                "--json",
                "--full-auto",
                "--skip-git-repo-check",
                "-o",
                str(output_path),
            ]
            if CODEX_MODEL:
                cmd.extend(["-m", CODEX_MODEL])
            cmd.extend([thread_id, prompt])
        else:
            cmd = [
                binary,
                "exec",
                "--json",
                "--full-auto",
                "--skip-git-repo-check",
                "-C",
                str(code_dir),
                "-o",
                str(output_path),
                "--output-schema",
                str(schema_path),
                "--color",
                "never",
            ]
            if CODEX_MODEL:
                cmd.extend(["-m", CODEX_MODEL])
            cmd.append(prompt)
        return subprocess.run(
            cmd,
            cwd=str(code_dir),
            env=env,
            timeout=CODEX_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
        )

    proc: subprocess.CompletedProcess[str] | None = None
    proc_mode = "fresh"
    try:
        if thread_id:
            proc = _run_once(use_resume=True)
            proc_mode = "resume"
            if proc.returncode != 0:
                proc = _run_once(use_resume=False)
                proc_mode = "fresh_fallback"
        else:
            proc = _run_once(use_resume=False)
    except Exception as exc:
        result = {
            "ok": False,
            "error": str(exc),
            "artifact_paths": {"agents_md": str(agents_path), "codex_session_state": str(session_state_path)},
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    parsed_last_message = _load_last_message_json(output_path)
    new_thread_id = _parse_thread_id(proc.stdout or "") or thread_id
    if new_thread_id:
        _save_session_state(
            session_state_path,
            {
                "thread_id": new_thread_id,
                "last_iteration": iteration,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    artifact_paths = {
        "agents_md": str(agents_path),
        "codex_last_message": str(output_path),
        "codex_result": str(result_path),
        "codex_output_schema": str(schema_path),
        "codex_session_state": str(session_state_path),
    }
    result = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "session_mode": proc_mode,
        "thread_id": new_thread_id,
        "summary": (
            (parsed_last_message or {}).get("summary")
            or f"Codex exec returncode={proc.returncode}"
        ),
        "files_changed": (parsed_last_message or {}).get("files_changed", []),
        "commands_run": (parsed_last_message or {}).get("commands_run", []),
        "validation_status": (parsed_last_message or {}).get("validation_status", ""),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "artifact_paths": artifact_paths,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
