"""Validation Loop: hypothesis-directed Karpathy-style experiment engine.

Two-phase loop:
  Phase 1 (Reproduction): run baseline as-is, record ground truth metric
  Phase 2 (Hypothesis Testing): implement proposed method, iterate, keep/discard

Key difference from autoresearch:
  - NOT open-ended optimization; directed by a specific method definition
  - Knows when to stop: hypothesis SUPPORTED, REFUTED, or TIMEOUT
  - Logs structured iteration data for the Result Interpreter
"""
import json
import os
import re
import sys
import subprocess
import textwrap
import time
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, record_artifact
from agents.benchmark_suite import run_benchmark_suite
from agents.evidence_gate import write_evidence_gate
from agents.statistical_reporter import write_statistical_report
from config import (
    EXPERIMENT_MAX_ITERATIONS,
    EXPERIMENT_REFUTE_MIN_ITERS,
    EXPERIMENT_REPRODUCTION_ITERS,
    EXPERIMENT_TIME_BUDGET,
)
from db import database as db


def _read_success_criteria(workdir: Path) -> dict:
    """Load success criteria from the workspace."""
    path = workdir / "success_criteria.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"metric_name": "metric", "metric_direction": "higher",
            "exciting": 0, "solid": 0, "disappointing": 0}


def _read_proxy_config(workdir: Path) -> dict:
    """Load proxy task configuration."""
    path = workdir / "proxy_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"time_budget_seconds": EXPERIMENT_TIME_BUDGET,
            "max_iterations": EXPERIMENT_MAX_ITERATIONS}


def _parse_metric_from_log(log_path: Path, metric_name: str) -> float | None:
    """Extract metric value from a run log or evaluate.py output."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    return _parse_metric_from_text(text, metric_name)


def _parse_metric_from_text(text: str, metric_name: str) -> float | None:
    """Extract a numeric metric from raw text or JSON output."""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and payload.get("metric_value") is not None:
            return float(payload["metric_value"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    patterns = [
        rf'{re.escape(metric_name)}[:\s]+([0-9]+\.?[0-9]*)',
        r'metric_value[:\s]+([0-9]+\.?[0-9]*)',
        r'val_bpb[:\s]+([0-9]+\.?[0-9]*)',
        r'accuracy[:\s]+([0-9]+\.?[0-9]*)',
        r'mAP[:\s]+([0-9]+\.?[0-9]*)',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def _parse_eval_metric(eval_stdout: str, log_metric: float | None) -> float | None:
    """Parse evaluator output without accepting legacy fake-zero fallbacks."""
    stripped = eval_stdout.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            if payload.get("valid") is False:
                return None
            if payload.get("metric_value") is not None:
                return float(payload["metric_value"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    metric = _parse_metric_from_text(stripped, metric_name="metric_value")
    if (
        metric == 0.0
        and log_metric is None
        and re.fullmatch(r"metric_value\s*:\s*0(?:\.0+)?", stripped, re.IGNORECASE)
    ):
        return None
    return metric


def _script_from_program(workdir: Path, code_dir: Path) -> Path | None:
    """Extract the Python script from the scaffold's documented run command."""
    program_path = workdir / "program.md"
    if not program_path.exists():
        return None
    try:
        text = program_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    matches = re.findall(r"`([^`]*\bpython(?:\.exe)?\s+([^`>\s]+)[^`]*)`", text, re.IGNORECASE)
    for _command, script in matches:
        script = script.strip().strip("\"'")
        if not script or script.startswith("-"):
            continue
        path = Path(script)
        if not path.is_absolute():
            path = code_dir / path
        try:
            resolved = path.resolve()
            resolved.relative_to(code_dir.resolve())
        except (OSError, ValueError):
            continue
        if resolved.exists() and resolved.suffix == ".py":
            return resolved
    return None


def _select_train_script(workdir: Path, code_dir: Path) -> Path:
    train_files = list(code_dir.glob("train*.py"))
    if train_files:
        return train_files[0]
    scripted = _script_from_program(workdir, code_dir)
    if scripted:
        return scripted
    return code_dir / "train.py"


def _write_json_artifact(workdir: Path, run_id: int, relative_path: str,
                         artifact_type: str, payload: dict | list,
                         metadata: dict | None = None):
    path = artifact_path(workdir, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    record_artifact(workdir, run_id, artifact_type, path, metadata or {})


def _copy_log_artifact(workdir: Path, run_id: int):
    source = workdir / "run.log"
    if not source.exists():
        return
    target = artifact_path(workdir, "artifacts/logs/run.log")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    record_artifact(workdir, run_id, "log", target, {"source": "validation_loop"})


def _run_experiment(workdir: Path, code_dir: Path, time_budget: int, metric_name: str = "") -> dict:
    """Run a single experiment iteration with time budget."""
    log_path = workdir / "run.log"
    eval_path = workdir / "evaluate.py"

    train_script = _select_train_script(workdir, code_dir)

    start = time.time()
    try:
        if not train_script.exists():
            return {"status": "crash", "duration": 0, "error": f"entrypoint not found: {train_script}"}
        proc = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(code_dir),
            timeout=time_budget + 60,
            capture_output=True,
            text=True,
        )
        duration = time.time() - start

        with open(log_path, "w") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(proc.stderr)

        if proc.returncode != 0:
            return {"status": "crash", "duration": duration,
                    "error": proc.stderr[-500:] if proc.stderr else "nonzero exit"}
    except subprocess.TimeoutExpired:
        return {"status": "crash", "duration": time.time() - start, "error": "timeout"}
    except Exception as e:
        return {"status": "crash", "duration": time.time() - start, "error": str(e)}

    log_metric = _parse_metric_from_log(log_path, metric_name)
    metric = log_metric
    if eval_path.exists():
        try:
            eval_result = subprocess.run(
                [sys.executable, str(eval_path), str(log_path)],
                cwd=str(workdir),
                timeout=60,
                capture_output=True,
                text=True,
            )
            if eval_result.stdout:
                eval_metric = _parse_eval_metric(eval_result.stdout, log_metric)
                if eval_metric is not None:
                    metric = eval_metric
        except Exception:
            pass

    if metric is None and not eval_path.exists():
        metric = _parse_metric_from_log(log_path, "")

    peak_mem = None
    mem_match = re.search(r'peak_vram_mb[:\s]+([0-9]+\.?[0-9]*)',
                          log_path.read_text(encoding="utf-8", errors="replace"))
    if mem_match:
        peak_mem = float(mem_match.group(1))

    return {
        "status": "ok",
        "metric": metric,
        "duration": duration,
        "peak_memory_mb": peak_mem,
    }


def _git_commit(code_dir: Path, message: str) -> str | None:
    """Commit changes in code_dir, return short hash."""
    try:
        _ensure_git_identity(code_dir)
        subprocess.run(["git", "add", "-A"], cwd=str(code_dir),
                       capture_output=True, timeout=10)
        subprocess.run(["git", "commit", "-m", message], cwd=str(code_dir),
                       capture_output=True, timeout=10)
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=str(code_dir), capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return None


def _ensure_git_identity(code_dir: Path):
    subprocess.run(["git", "config", "user.email", "sciforge@example.local"],
                   cwd=str(code_dir), capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "SciForge"],
                   cwd=str(code_dir), capture_output=True, timeout=10)


def _ensure_git_baseline(code_dir: Path) -> str:
    """Make the current scaffold state reset-safe and return its commit hash."""
    if not (code_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(code_dir), capture_output=True, timeout=10)
    _ensure_git_identity(code_dir)
    subprocess.run(["git", "add", "-A"], cwd=str(code_dir), capture_output=True, timeout=10)
    subprocess.run(["git", "commit", "-m", "scaffold baseline"],
                   cwd=str(code_dir), capture_output=True, timeout=10)
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=str(code_dir), capture_output=True, text=True, timeout=5)
    return result.stdout.strip()


def _git_reset(code_dir: Path, commit_hash: str):
    """Reset code_dir to a specific commit."""
    try:
        if not commit_hash:
            return
        subprocess.run(["git", "reset", "--hard", commit_hash],
                       cwd=str(code_dir), capture_output=True, timeout=10)
    except Exception:
        pass


def _git_diff(code_dir: Path) -> str:
    """Get current diff in code_dir."""
    try:
        result = subprocess.run(["git", "diff", "HEAD~1"],
                                cwd=str(code_dir), capture_output=True, text=True, timeout=10)
        return result.stdout[:2000]
    except Exception:
        return ""


def _is_better(new_val: float, old_val: float, direction: str) -> bool:
    """Check if new metric is better than old, given direction."""
    if direction == "lower":
        return new_val < old_val
    return new_val > old_val


def _meets_threshold(value: float, threshold: float, direction: str) -> bool:
    """Check if metric meets a success threshold."""
    if threshold == 0:
        return False
    if direction == "lower":
        return value <= threshold
    return value >= threshold


def _method_description_from_insight(insight: dict | None) -> str:
    """Build coding-agent context from both Tier 2 methods and Tier 1 insights."""
    if not insight:
        return ""

    try:
        method = json.loads(insight.get("proposed_method") or "{}")
    except (json.JSONDecodeError, TypeError):
        method = {}

    if isinstance(method, dict) and method:
        return (
            f"Name: {method.get('name', '?')}\n"
            f"Type: {method.get('type', '?')}\n"
            f"Summary: {method.get('one_line', '')}\n"
            f"Definition: {method.get('definition', '')[:800]}\n"
            f"Pseudocode: {method.get('pseudocode', '')[:500]}"
        )

    plan_text = ""
    raw_plan = insight.get("experimental_plan") or ""
    if isinstance(raw_plan, str) and raw_plan.strip():
        try:
            plan = json.loads(raw_plan)
            if isinstance(plan, dict):
                parts = []
                for key in ("procedure", "success_metric", "compute"):
                    if plan.get(key):
                        parts.append(f"{key}: {plan[key]}")
                for key in ("models", "datasets"):
                    if plan.get(key):
                        parts.append(f"{key}: {json.dumps(plan[key])[:700]}")
                plan_text = "\n".join(parts)
        except (json.JSONDecodeError, TypeError):
            plan_text = raw_plan[:1200]
    elif isinstance(raw_plan, dict):
        plan_text = json.dumps(raw_plan)[:1200]

    return "\n".join(
        part for part in (
            f"Name: {insight.get('title', '?')}",
            f"Formal structure: {(insight.get('formal_structure') or '')[:900]}",
            f"Transformation: {(insight.get('transformation') or '')[:700]}",
            f"Problem: {(insight.get('problem_statement') or '')[:500]}",
            f"Experimental plan:\n{plan_text[:1500]}" if plan_text else "",
        )
        if part and part.strip()
    )


def _launch_coding_agent(workdir: Path, code_dir: Path, iteration: int,
                         method_desc: str, best_so_far: float | None,
                         baseline: float | None, history: list[dict]) -> str:
    """Use LLM to generate the next code modification.

    Returns a description of what was tried (the actual code changes
    are written directly to files by the agent).
    """
    from agents.llm_client import call_llm

    recent_history = history[-10:] if history else []
    history_text = ""
    for h in recent_history:
        status_marker = "KEPT" if h.get("status") == "keep" else "DISCARDED"
        history_text += f"  Iter {h.get('iteration', '?')}: {h.get('description', '?')} -> {h.get('metric', '?')} [{status_marker}]\n"

    train_file = None
    for pattern in ["train*.py", "main*.py", "run*.py"]:
        matches = list(code_dir.glob(pattern))
        if matches:
            train_file = matches[0]
            break
    if not train_file:
        py_files = list(code_dir.glob("*.py"))
        train_file = py_files[0] if py_files else None

    current_code = ""
    if train_file and train_file.exists():
        try:
            current_code = train_file.read_text(encoding="utf-8")[:8000]
        except Exception:
            pass

    system = textwrap.dedent("""\
        You are an ML research engineer implementing a specific method modification.
        You will receive the current code, the method to implement, and experiment history.
        
        Output ONLY the modified code for the train file. No explanation, no markdown.
        Make ONE focused change per iteration. If the last change was discarded, try a different approach.
        If the last change was kept, build on it.""")

    prompt = textwrap.dedent(f"""\
        # Method to Implement
        {method_desc[:1500]}
        
        # Current State
        Baseline metric: {baseline}
        Best so far: {best_so_far}
        Iteration: {iteration}
        
        # Recent History
        {history_text if history_text else "No history yet - this is the first modification."}
        
        # Current Code ({train_file.name if train_file else 'train.py'})
        ```python
        {current_code}
        ```
        
        Output the COMPLETE modified file. Make one focused change to implement or improve the method.""")

    try:
        new_code, _ = call_llm(system, prompt, max_tokens=16000)
        new_code = new_code.strip()

        # Strip <think>...</think> blocks (reasoning models)
        new_code = re.sub(r'<think>[\s\S]*?</think>', '', new_code).strip()

        # Extract code from markdown code blocks (LLM often wraps in ```)
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', new_code, re.DOTALL)
        if code_blocks:
            # Use the longest code block (likely the full file)
            new_code = max(code_blocks, key=len).strip()
        elif new_code.startswith("```"):
            lines = new_code.split("\n")
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            new_code = "\n".join(lines[1:end])

        # Validate it looks like Python (must have def/import/class or assignment)
        has_python = bool(re.search(r'^(import |from |def |class |[a-zA-Z_]\w*\s*=)', new_code, re.MULTILINE))
        if not has_python:
            return f"LLM output not valid Python (iter {iteration})"

        if train_file and len(new_code) > 50:
            train_file.write_text(new_code, encoding="utf-8")
            return f"Modified {train_file.name} (iter {iteration})"
    except Exception as e:
        return f"LLM code generation failed: {e}"

    return f"No modification applied (iter {iteration})"


def _benchmark_metric_summary(workdir: Path) -> dict:
    path = workdir / "artifacts" / "results" / "statistical_report.json"
    if not path.exists():
        return {"baseline": None, "best": None, "effect": None, "effect_pct": None}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"baseline": None, "best": None, "effect": None, "effect_pct": None}

    baseline_method = report.get("baseline_method")
    best_method = report.get("best_method")
    summary = report.get("summary") or []

    def mean_for(method):
        values = [float(row["mean"]) for row in summary if row.get("method") == method and row.get("mean") is not None]
        return sum(values) / len(values) if values else None

    baseline = mean_for(baseline_method)
    best = mean_for(best_method)
    effect = None
    effect_pct = None
    if baseline is not None and best is not None:
        effect = best - baseline
        effect_pct = (effect / abs(baseline) * 100) if baseline != 0 else 0.0
    return {"baseline": baseline, "best": best, "effect": effect, "effect_pct": effect_pct}


def _benchmark_evidence_confirmed(gate: dict) -> bool:
    if gate.get("manuscript_status") == "paper_ready_candidate":
        return True
    blocking = set(gate.get("blocking_reasons") or [])
    if blocking != {"review_requires_revision"}:
        return False
    satisfied = set(gate.get("satisfied_requirements") or [])
    required = {
        "has_benchmark_results",
        "has_baseline_comparison",
        "has_statistical_report",
        "has_multi_seed",
        "has_multi_dataset",
        "has_ablation",
    }
    return required.issubset(satisfied)


def _run_benchmark_mode(run_id: int, workdir: Path) -> dict:
    db.execute(
        "UPDATE experiment_runs SET status='testing', phase='benchmark_suite', started_at=CURRENT_TIMESTAMP WHERE id=?",
        (run_id,),
    )
    db.commit()
    suite_result = run_benchmark_suite(run_id)
    if suite_result.get("status") != "complete":
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(suite_result), run_id),
        )
        db.commit()
        return {
            "run_id": run_id,
            "execution_mode": "benchmark_suite",
            "verdict": "failed",
            "reason": suite_result.get("reason", "benchmark_suite_failed"),
            "suite": suite_result,
        }

    stats_result = write_statistical_report(run_id)
    gate = write_evidence_gate(run_id)
    summary = _benchmark_metric_summary(workdir)
    verdict = "confirmed" if _benchmark_evidence_confirmed(gate) else "inconclusive"
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', phase='benchmark_suite', hypothesis_verdict=?,
               baseline_metric_value=?, best_metric_value=?,
               effect_size=?, effect_pct=?, completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (
            verdict,
            summary.get("baseline"),
            summary.get("best"),
            summary.get("effect"),
            summary.get("effect_pct"),
            run_id,
        ),
    )
    db.commit()
    return {
        "run_id": run_id,
        "execution_mode": "benchmark_suite",
        "verdict": verdict,
        "suite": suite_result,
        "statistics": stats_result,
        "evidence_gate": gate,
        **summary,
    }


def run_validation_loop(run_id: int) -> dict:
    """Execute the full two-phase validation loop for an experiment run.

    Returns the final verdict and statistics.
    """
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"error": f"Run {run_id} not found"}

    workdir = Path(run["workdir"])
    code_dir = workdir / "code"
    insight_id = run["deep_insight_id"]

    if not workdir.exists():
        return {"error": f"Workdir {workdir} does not exist"}

    if (workdir / "benchmark_config.json").exists():
        return _run_benchmark_mode(run_id, workdir)

    criteria = _read_success_criteria(workdir)
    proxy = _read_proxy_config(workdir)
    ensure_artifact_dirs(workdir)
    metric_name = criteria.get("metric_name", "metric")
    direction = criteria.get("metric_direction", "higher")
    time_budget = proxy.get("time_budget_seconds", EXPERIMENT_TIME_BUDGET)
    max_iters = proxy.get("max_iterations", EXPERIMENT_MAX_ITERATIONS)
    repro_iters = proxy.get("reproduction_iterations", EXPERIMENT_REPRODUCTION_ITERS)
    refute_min = proxy.get("refute_min_iterations", EXPERIMENT_REFUTE_MIN_ITERS)

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    method_desc = _method_description_from_insight(insight)

    _ensure_git_baseline(code_dir)

    db.execute("UPDATE experiment_runs SET status='reproducing', phase='reproduction', started_at=CURRENT_TIMESTAMP WHERE id=?", (run_id,))
    db.commit()

    # ── Phase 1: Reproduction ──
    print(f"[LOOP] Phase 1: Reproducing baseline ({repro_iters} iterations)...", flush=True)
    baseline_values = []
    iteration_records = []

    for i in range(repro_iters):
        result = _run_experiment(workdir, code_dir, time_budget, metric_name)
        metric = result.get("metric")

        db.execute(
            """INSERT INTO experiment_iterations
               (run_id, iteration_number, phase, metric_value, metric_name,
                peak_memory_mb, duration_seconds, status, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, i + 1, "reproduction", metric, metric_name,
             result.get("peak_memory_mb"), result.get("duration"),
             result.get("status", "ok"), f"baseline run {i+1}")
        )
        db.commit()
        iteration_records.append({
            "run_id": run_id,
            "iteration_number": i + 1,
            "phase": "reproduction",
            "metric_value": metric,
            "metric_name": metric_name,
            "status": result.get("status", "ok"),
            "description": f"baseline run {i+1}",
            "error": result.get("error"),
            "duration_seconds": result.get("duration"),
            "peak_memory_mb": result.get("peak_memory_mb"),
        })

        if metric is not None:
            baseline_values.append(metric)
            print(f"[LOOP] Reproduction {i+1}/{repro_iters}: {metric_name}={metric}", flush=True)
        else:
            print(f"[LOOP] Reproduction {i+1}/{repro_iters}: no metric (status={result.get('status')})", flush=True)

    if not baseline_values:
        last_error = next(
            (str(r.get("error")) for r in reversed(iteration_records) if r.get("error")),
            "no metric obtained",
        )
        error_message = f"reproduction failed: {last_error}"
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (error_message, run_id))
        db.commit()
        metrics_payload = {
            "run_id": run_id,
            "verdict": "failed",
            "reason": "reproduction_failure",
            "error": error_message,
            "metric_name": metric_name,
            "baseline": None,
            "best_value": None,
        }
        _write_json_artifact(workdir, run_id, "artifacts/results/metrics.json", "metrics", metrics_payload)
        _write_json_artifact(workdir, run_id, "artifacts/results/iterations.json", "iterations", iteration_records)
        _copy_log_artifact(workdir, run_id)
        print(f"[LOOP] Phase 1 FAILED: could not obtain baseline metric", flush=True)
        return {"verdict": "failed", "reason": "reproduction_failure"}

    baseline = sum(baseline_values) / len(baseline_values)
    baseline_commit = _ensure_git_baseline(code_dir)

    db.execute(
        "UPDATE experiment_runs SET baseline_metric_value=?, best_metric_value=?, phase='hypothesis_testing', status='testing' WHERE id=?",
        (baseline, baseline, run_id))
    db.commit()
    print(f"[LOOP] Baseline established: {metric_name}={baseline:.6f}", flush=True)

    # ── Phase 2: Hypothesis Testing ──
    print(f"[LOOP] Phase 2: Hypothesis testing (max {max_iters} iterations)...", flush=True)
    best_value = baseline
    best_commit = baseline_commit
    total_kept = 0
    iter_num = repro_iters
    effect_pct = 0.0
    history = []
    loop_start = time.time()
    exciting = criteria.get("exciting", 0)
    solid = criteria.get("solid", 0)
    disappointing = criteria.get("disappointing", 0)

    for i in range(max_iters):
        iter_num = repro_iters + i + 1

        desc = _launch_coding_agent(
            workdir, code_dir, i + 1, method_desc, best_value, baseline, history)

        commit_hash = _git_commit(code_dir, f"experiment iter {i+1}: {desc[:80]}")

        result = _run_experiment(workdir, code_dir, time_budget, metric_name)
        metric = result.get("metric")

        status = "crash"
        if result.get("status") == "ok" and metric is not None:
            if _is_better(metric, best_value, direction):
                status = "keep"
                best_value = metric
                best_commit = commit_hash
                total_kept += 1
            else:
                status = "discard"
                _git_reset(code_dir, best_commit)
        elif result.get("status") == "crash":
            _git_reset(code_dir, best_commit)

        diff = _git_diff(code_dir) if status == "keep" else ""

        db.execute(
            """INSERT INTO experiment_iterations
               (run_id, iteration_number, phase, code_diff, commit_hash,
                metric_value, metric_name, peak_memory_mb, duration_seconds,
                status, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, iter_num, "hypothesis_testing", diff, commit_hash,
             metric, metric_name, result.get("peak_memory_mb"),
             result.get("duration"), status, desc[:500])
        )

        effect = best_value - baseline if direction == "higher" else baseline - best_value
        effect_pct = (effect / abs(baseline) * 100) if baseline != 0 else 0
        db.execute(
            """UPDATE experiment_runs
               SET iterations_total=?, iterations_kept=?, best_metric_value=?,
                   effect_size=?, effect_pct=?
               WHERE id=?""",
            (iter_num, total_kept, best_value, effect, effect_pct, run_id)
        )
        db.commit()
        iteration_records.append({
            "run_id": run_id,
            "iteration_number": iter_num,
            "phase": "hypothesis_testing",
            "metric_value": metric,
            "metric_name": metric_name,
            "status": status,
            "description": desc[:500],
            "error": result.get("error"),
            "duration_seconds": result.get("duration"),
            "peak_memory_mb": result.get("peak_memory_mb"),
            "commit_hash": commit_hash,
        })

        history.append({
            "iteration": i + 1,
            "metric": metric,
            "status": status,
            "description": desc[:100],
        })

        if (i + 1) % 5 == 0:
            print(f"[LOOP] Iter {i+1}/{max_iters}: best={best_value:.6f} "
                  f"(baseline={baseline:.6f}, kept={total_kept})", flush=True)

        # Check termination conditions
        if exciting and _meets_threshold(best_value, exciting, direction):
            print(f"[LOOP] EXCITING result reached at iter {i+1}!", flush=True)
            break

        if solid and _meets_threshold(best_value, solid, direction) and i >= 10:
            print(f"[LOOP] Solid result reached at iter {i+1}, continuing for more.", flush=True)

        if i + 1 >= refute_min and not _is_better(best_value, baseline, direction):
            print(f"[LOOP] Hypothesis appears REFUTED after {i+1} iterations.", flush=True)
            break

    # ── Determine verdict ──
    effect = best_value - baseline if direction == "higher" else baseline - best_value
    is_improvement = effect > 0
    all_test_values = [
        r.get("metric_value")
        for r in iteration_records
        if r.get("phase") == "hypothesis_testing" and r.get("metric_value") is not None
    ]

    if not all_test_values:
        verdict = "inconclusive"
    elif is_improvement and exciting and _meets_threshold(best_value, exciting, direction):
        verdict = "confirmed"
    elif is_improvement and solid and _meets_threshold(best_value, solid, direction):
        verdict = "confirmed"
    elif is_improvement and effect_pct > 1.0:
        verdict = "confirmed"
    elif not is_improvement or effect_pct < 0.1:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    total_time = time.time() - loop_start
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', hypothesis_verdict=?,
               effect_size=?, effect_pct=?,
               completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (verdict, effect, effect_pct, run_id)
    )
    db.commit()

    print(f"[LOOP] Completed: verdict={verdict}, effect={effect:.6f} ({effect_pct:.2f}%), "
          f"iters={iter_num}, kept={total_kept}, time={total_time:.0f}s", flush=True)

    metrics_payload = {
        "run_id": run_id,
        "verdict": verdict,
        "metric_name": metric_name,
        "metric_direction": direction,
        "baseline": baseline,
        "best_value": best_value,
        "effect_size": effect,
        "effect_pct": effect_pct,
        "iterations_total": iter_num,
        "iterations_kept": total_kept,
        "total_seconds": total_time,
    }
    _write_json_artifact(workdir, run_id, "artifacts/results/metrics.json", "metrics", metrics_payload)
    _write_json_artifact(workdir, run_id, "artifacts/results/iterations.json", "iterations", iteration_records)
    _copy_log_artifact(workdir, run_id)

    return {
        "run_id": run_id,
        "verdict": verdict,
        "baseline": baseline,
        "best_value": best_value,
        "effect_size": effect,
        "effect_pct": effect_pct,
        "iterations_total": iter_num,
        "iterations_kept": total_kept,
        "total_seconds": total_time,
    }


