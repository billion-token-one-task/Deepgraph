"""Codex CLI-backed repo editing for experiment iterations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CODEX_CLI_PATH, CODEX_EXEC_ENABLED, CODEX_MODEL, CODEX_TIMEOUT_SECONDS
from agents.stage_prompts import prompt_block


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


def _history_text(history: list[dict], limit: int = 12) -> str:
    if not history:
        return "No prior hypothesis-testing iterations yet."
    lines: list[str] = []
    for row in history[-limit:]:
        status = row.get("status", "?")
        metric = row.get("metric")
        description = str(row.get("description") or "").strip()
        lines.append(f"- iter {row.get('iteration', '?')}: status={status} metric={metric} change={description[:180]}")
    return "\n".join(lines)


def _history_requires_fresh_codex_session(history: list[dict], limit: int = 3) -> bool:
    """Avoid resuming a Codex thread after contaminated or stale search context."""
    markers = (
        "benchmark_fairness_risk",
        "benchmark_semantic_risk",
        "upper_bound",
        "candidate-only",
        "candidate specific",
        "candidate-specific",
        "answer canonicalization",
        "answer canonicalizer",
    )
    for row in history[-limit:]:
        text = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in text for marker in markers):
            return True
    for row in history[-12:]:
        text = json.dumps(row, ensure_ascii=False).lower()
        if row.get("status") == "keep":
            has_budget_signal = any(marker in text for marker in ("token", "budget", "cap", "cost"))
            has_narrow_signal = any(marker in text for marker in ("yes/no", "zero-budget", "one-word", "1-token", "2-token", "3-token"))
            if has_budget_signal and has_narrow_signal:
                return True
        if row.get("status") == "discard" and any(
            marker in text
            for marker in (
                "shortest exact answer",
                "shortest final answer",
                "answer span",
                "answer phrase only",
                "context-preserving prompting",
                "context/passages/facts",
                "benchmark-provided context",
            )
        ):
            return True
    recent = history[-limit:]
    if len(recent) >= limit and all(
        str(row.get("status") or "").lower() == "discard"
        and (
            "no_gain" in json.dumps(row, ensure_ascii=False).lower()
            or "metric did not improve" in json.dumps(row, ensure_ascii=False).lower()
        )
        for row in recent
    ):
        return True
    return False


def _real_benchmark_guardrails_text() -> str:
    return """## Real Benchmark Guardrails
- Treat the benchmark contract as immutable: datasets, model targets, baselines, ablations, metrics, seeds, splits, and artifact names may not be weakened inside a method iteration.
- A sanity/bounded run may only prove infrastructure and metric parsing. It must not be described as paper evidence unless the full benchmark package passes with `full_benchmark_completed=true`.
- Preserve or produce paper-ready artifacts when working on a full benchmark path: `benchmark_summary.json`, `benchmark_artifact_manifest.json`, raw prediction JSONL, routing decisions, per-seed/per-dataset tables, ablation table, latency/token table, bootstrap CI, and failure cases.
- If this repository contains a generated real LLM benchmark runner, preserve the runner scaffolding while repairing or improving it.
- Keep local real-data fallback support such as `DEFAULT_LOCAL_JSONL`, `_read_jsonl_rows`, `_download_jsonl_rows`, and checksum validation.
- Keep real-model fallback support such as ModelScope snapshot loading for Qwen-family models and the `modelscope` dependency.
- Keep the benchmark output contract: `FINAL_RESULTS`, `per_method`, `candidate_method`, baseline-vs-candidate metrics, and peak VRAM reporting.
- If recent history says the candidate exceeds an `upper_bound` or `oracle_router`, treat that as a benchmark-semantics bug first: repair the comparator label/oracle calculation or explain why it is not an upper bound before making method-score tweaks.
- Do not change scoring, answer normalization, parsing, or post-processing only for the candidate method. Any evaluator-side normalization must apply to all methods, and any candidate-side post-processing must be an explicitly justified method component rather than a metric shortcut.
- Do not add dataset/example-specific lexical shortcuts, answer canonicalizers, or string rewrites that make the candidate easier to score without giving baselines the same evaluator.
- Do not reduce seeds, example counts, baseline coverage, or ablation coverage to make a formal run pass. If the current environment cannot afford the contract, return a blocker instead of downgrading evidence.
- Do not replace a concrete dataset id such as `openai/gsm8k` with display-only benchmark names such as `MuSiQue-Ans`.
- Do not convert formal benchmark runs to synthetic data, mocked examples, random tensors, CPU-only probes, or proxy-only experiments.
- If a dependency, dataset, or model load fails, patch the environment/config/fallback path around the real benchmark instead of replacing the benchmark.
"""


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

{prompt_block("method_worker")}

## Success Criteria
{_trim_json(success_criteria)}

## Experimental Plan
{_trim_json(experimental_plan)}

## Evidence Plan
{_trim_json(evidence_plan)}

Honor the evidence plan. Do not invent ablations or visual analyses when they are disabled.
For formal benchmark runs, keep the experiment on the real dataset/model named in the plan.
Do not replace failures with synthetic data, random tensors, mocked examples, or a CUDA-only probe.

{_real_benchmark_guardrails_text()}

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
    if _history_requires_fresh_codex_session(history):
        thread_id = ""

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
            encoding="utf-8",
            errors="replace",
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
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    parsed_thread_id = _parse_thread_id(stdout)
    new_thread_id = parsed_thread_id or (thread_id if proc_mode == "resume" else "")
    if new_thread_id:
        _save_session_state(
            session_state_path,
            {
                "thread_id": new_thread_id,
                "last_iteration": iteration,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    elif proc_mode in {"fresh", "fresh_fallback"}:
        _save_session_state(
            session_state_path,
            {
                "thread_id": "",
                "last_iteration": iteration,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "session_mode": proc_mode,
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
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "artifact_paths": artifact_paths,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_reproduction_repair_agents_md(
    *,
    code_dir: Path,
    workdir: Path,
    repair_round: int,
    baseline_command: str | None,
    metric_name: str,
    last_error: str,
    log_excerpt: str,
    environment_report: dict[str, Any],
) -> Path:
    """AGENTS.md for baseline reproduction repair (environment / imports / device), not hypothesis work."""
    log_path = workdir / "run.log"
    py_hint = os.environ.get("DEEPGRAPH_RUNTIME_PYTHON", sys.executable)
    content = f"""# DeepGraph - Baseline Reproduction Repair

## Context
- Repair round: {repair_round}
- Platform: {os.name}
- Preferred Python for runs: {py_hint}
- Success metric field name (for log parsing): {metric_name}

## What failed
Last error (truncated):
```
{last_error[:6000]}
```

## Log file
Read the full log at (absolute path):
`{log_path}`

Recent tail:
```
{log_excerpt[:8000]}
```

## Baseline command (validation loop will re-run this)
```
{baseline_command or "(auto: python <main train script>)"}
```

## Environment scout summary
{_trim_json(environment_report)}

{prompt_block("reproduction_repair")}

## Your job
1. Open the log, identify the **root cause** (missing dependency, wrong path, CUDA vs CPU, broken import, etc.).
2. Apply the **smallest** code or config edits so the baseline command completes with exit code 0.
3. Ensure stdout/stderr or `run.log` contains a parseable metric:
   - a line like `metric_value: <float>` OR `"{metric_name}": <float>`, OR
   - a `FINAL_RESULTS:` JSON line as documented for benchmarks.
4. If the job needs GPU/model/data and the environment is missing something, fix the dependency/configuration or report the blocker. Do not switch to CPU-only, synthetic, mocked, or toy data for formal benchmark runs.
5. Do **not** implement the research hypothesis yet - only make the baseline **runnable locally**.

{_real_benchmark_guardrails_text()}

## Rules
- Prefer editing existing entrypoints over adding heavy new dependencies.
- If you add dependencies, list them in a `requirements-experiment.txt` and mention them in the JSON summary.
- Keep runtime bounded by using a documented subset of the real benchmark when needed, while still loading the real dataset and real model.

## Final response
Return JSON with keys: summary, files_changed, commands_run, validation_status (same schema as experiment iterations).
"""
    codex_dir = workdir / "codex" / "repro_repairs"
    codex_dir.mkdir(parents=True, exist_ok=True)
    archive_path = codex_dir / f"repair_{repair_round:03d}_AGENTS.md"
    archive_path.write_text(content, encoding="utf-8")
    repo_path = code_dir / "AGENTS.md"
    repo_path.write_text(content, encoding="utf-8")
    return archive_path


def run_codex_reproduction_repair(
    *,
    workdir: Path,
    code_dir: Path,
    repair_round: int,
    baseline_command: str | None,
    metric_name: str,
    last_error: str,
    log_excerpt: str,
    environment_report: dict[str, Any],
) -> dict[str, Any]:
    """Run Codex to fix baseline execution (Phase 1). Uses a separate session from hypothesis iterations."""
    binary = codex_binary()
    if not binary:
        return {"ok": False, "error": "codex_unavailable"}

    agents_path = write_reproduction_repair_agents_md(
        code_dir=code_dir,
        workdir=workdir,
        repair_round=repair_round,
        baseline_command=baseline_command,
        metric_name=metric_name,
        last_error=last_error,
        log_excerpt=log_excerpt,
        environment_report=environment_report,
    )

    out_dir = workdir / "codex" / "repro_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"repair_{repair_round:03d}_last_message.json"
    schema_path = out_dir / f"repair_{repair_round:03d}_output_schema.json"
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
    session_state_path = workdir / "codex" / "session_repro.json"
    session_state = _load_session_state(session_state_path)
    thread_id = str(session_state.get("thread_id") or "").strip()

    prompt = (
        "Read AGENTS.md in the repository root. It contains baseline reproduction repair instructions. "
        "Fix the repo so the baseline runs successfully, then return the required JSON summary."
    )

    env = os.environ.copy()
    result_path = out_dir / f"repair_{repair_round:03d}_result.json"

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
            encoding="utf-8",
            errors="replace",
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
            "artifact_paths": {"agents_md": str(agents_path), "codex_session_repro": str(session_state_path)},
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    assert proc is not None
    parsed_last_message = _load_last_message_json(output_path)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    parsed_thread_id = _parse_thread_id(stdout)
    new_thread_id = parsed_thread_id or (thread_id if proc_mode == "resume" else "")
    if new_thread_id:
        _save_session_state(
            session_state_path,
            {
                "thread_id": new_thread_id,
                "last_repair_round": repair_round,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    elif proc_mode in {"fresh", "fresh_fallback"}:
        _save_session_state(
            session_state_path,
            {
                "thread_id": "",
                "last_repair_round": repair_round,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "session_mode": proc_mode,
            },
        )

    artifact_paths = {
        "agents_md": str(agents_path),
        "codex_last_message": str(output_path),
        "codex_result": str(result_path),
        "codex_output_schema": str(schema_path),
        "codex_session_repro": str(session_state_path),
    }
    result = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "session_mode": proc_mode,
        "thread_id": new_thread_id,
        "summary": (
            (parsed_last_message or {}).get("summary")
            or f"Codex repro repair returncode={proc.returncode}"
        ),
        "files_changed": (parsed_last_message or {}).get("files_changed", []),
        "commands_run": (parsed_last_message or {}).get("commands_run", []),
        "validation_status": (parsed_last_message or {}).get("validation_status", ""),
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "artifact_paths": artifact_paths,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
