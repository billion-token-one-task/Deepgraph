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
import subprocess
import textwrap
import time
from pathlib import Path

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


def _run_experiment(workdir: Path, code_dir: Path, time_budget: int) -> dict:
    """Run a single experiment iteration with time budget."""
    log_path = workdir / "run.log"
    eval_path = workdir / "evaluate.py"

    train_files = list(code_dir.glob("train*.py"))
    train_script = str(train_files[0]) if train_files else "train.py"

    start = time.time()
    try:
        proc = subprocess.run(
            ["python3.12", train_script],
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

    metric = None
    if eval_path.exists():
        try:
            eval_result = subprocess.run(
                ["python3.12", str(eval_path), str(log_path)],
                cwd=str(workdir),
                timeout=60,
                capture_output=True,
                text=True,
            )
            metric = _parse_metric_from_log(
                Path("/dev/stdin"),  # dummy
                "metric_value"
            )
            if eval_result.stdout:
                match = re.search(r'metric_value[:\s]+([0-9]+\.?[0-9]*)', eval_result.stdout)
                if match:
                    metric = float(match.group(1))
        except Exception:
            pass

    if metric is None:
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
        subprocess.run(["git", "add", "-A"], cwd=str(code_dir),
                       capture_output=True, timeout=10)
        subprocess.run(["git", "commit", "-m", message], cwd=str(code_dir),
                       capture_output=True, timeout=10)
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=str(code_dir), capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return None


def _git_reset(code_dir: Path, commit_hash: str):
    """Reset code_dir to a specific commit."""
    try:
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

    criteria = _read_success_criteria(workdir)
    proxy = _read_proxy_config(workdir)
    metric_name = criteria.get("metric_name", "metric")
    direction = criteria.get("metric_direction", "higher")
    time_budget = proxy.get("time_budget_seconds", EXPERIMENT_TIME_BUDGET)
    max_iters = proxy.get("max_iterations", EXPERIMENT_MAX_ITERATIONS)
    repro_iters = proxy.get("reproduction_iterations", EXPERIMENT_REPRODUCTION_ITERS)
    refute_min = proxy.get("refute_min_iterations", EXPERIMENT_REFUTE_MIN_ITERS)

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    method_desc = ""
    if insight:
        try:
            method = json.loads(insight.get("proposed_method") or "{}")
            method_desc = (
                f"Name: {method.get('name', '?')}\n"
                f"Type: {method.get('type', '?')}\n"
                f"Summary: {method.get('one_line', '')}\n"
                f"Definition: {method.get('definition', '')[:800]}\n"
                f"Pseudocode: {method.get('pseudocode', '')[:500]}"
            )
        except (json.JSONDecodeError, TypeError):
            method_desc = insight.get("problem_statement", "") or insight.get("title", "")

    if not (code_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(code_dir), capture_output=True, timeout=10)
        subprocess.run(["git", "add", "-A"], cwd=str(code_dir), capture_output=True, timeout=10)
        subprocess.run(["git", "commit", "-m", "initial baseline"],
                       cwd=str(code_dir), capture_output=True, timeout=10)

    db.execute("UPDATE experiment_runs SET status='reproducing', phase='reproduction', started_at=CURRENT_TIMESTAMP WHERE id=?", (run_id,))
    db.commit()

    # ── Phase 1: Reproduction ──
    print(f"[LOOP] Phase 1: Reproducing baseline ({repro_iters} iterations)...", flush=True)
    baseline_values = []

    for i in range(repro_iters):
        result = _run_experiment(workdir, code_dir, time_budget)
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

        if metric is not None:
            baseline_values.append(metric)
            print(f"[LOOP] Reproduction {i+1}/{repro_iters}: {metric_name}={metric}", flush=True)
        else:
            print(f"[LOOP] Reproduction {i+1}/{repro_iters}: no metric (status={result.get('status')})", flush=True)

    if not baseline_values:
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message='reproduction failed: no metric obtained', completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (run_id,))
        db.commit()
        print(f"[LOOP] Phase 1 FAILED: could not obtain baseline metric", flush=True)
        return {"verdict": "failed", "reason": "reproduction_failure"}

    baseline = sum(baseline_values) / len(baseline_values)
    baseline_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(code_dir), capture_output=True, text=True, timeout=5
    ).stdout.strip()

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

    for i in range(max_iters):
        iter_num = repro_iters + i + 1

        desc = _launch_coding_agent(
            workdir, code_dir, i + 1, method_desc, best_value, baseline, history)

        commit_hash = _git_commit(code_dir, f"experiment iter {i+1}: {desc[:80]}")

        result = _run_experiment(workdir, code_dir, time_budget)
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
        exciting = criteria.get("exciting", 0)
        solid = criteria.get("solid", 0)
        disappointing = criteria.get("disappointing", 0)

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

    if exciting and _meets_threshold(best_value, exciting, direction):
        verdict = "confirmed"
    elif solid and _meets_threshold(best_value, solid, direction):
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


