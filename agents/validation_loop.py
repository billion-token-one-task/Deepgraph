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
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from agents import codex_executor
from agents import experiment_supervisor
from agents.workspace_layout import ensure_run_workspace, plan_file_path, promote_canonical_run, write_latest_status
from contracts import DeepInsightSpec, ExperimentIterationPacket, ExperimentSpec
from config import (
    EXPERIMENT_MAX_ITERATIONS,
    EXPERIMENT_REFUTE_MIN_ITERS,
    EXPERIMENT_REPRODUCTION_ITERS,
    EXPERIMENT_TIME_BUDGET,
    RUNTIME_PYTHON,
)
from db import database as db
from orchestrator import ssh_gpu_backend


def _git_binary() -> str | None:
    return shutil.which("git")


def _read_success_criteria(workdir: Path, insight_id: int | None = None) -> dict:
    """Load success criteria from the workspace."""
    candidates = []
    if insight_id is not None:
        candidates.append(plan_file_path(insight_id, "success_criteria.json"))
    candidates.extend((workdir / "spec" / "success_criteria.json", workdir / "success_criteria.json"))
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return {"metric_name": "metric", "metric_direction": "higher",
            "exciting": 0, "solid": 0, "disappointing": 0}


def _read_proxy_config(workdir: Path, insight_id: int | None = None) -> dict:
    """Load proxy task configuration."""
    candidates = []
    if insight_id is not None:
        candidates.append(plan_file_path(insight_id, "proxy_config.json"))
    candidates.extend((workdir / "spec" / "proxy_config.json", workdir / "proxy_config.json"))
    for path in candidates:
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
        rf'"?{re.escape(metric_name)}"?\s*[:=]\s*([0-9]+\.?[0-9]*)' if metric_name else None,
        r'"metric_value"\s*:\s*([0-9]+\.?[0-9]*)',
        r'metric_value[:\s]+([0-9]+\.?[0-9]*)',
        r'val_bpb[:\s]+([0-9]+\.?[0-9]*)',
        r'accuracy[:\s]+([0-9]+\.?[0-9]*)',
        r'mAP[:\s]+([0-9]+\.?[0-9]*)',
    ]
    for pat in patterns:
        if not pat:
            continue
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def _parse_benchmark_summary_from_log(log_path: Path) -> dict:
    """Parse structured benchmark output from a run log.

    Preferred format is a single line prefixed with ``FINAL_RESULTS:`` followed
    by JSON. As a fallback, accept a plain JSON line containing ``per_method``.
    """
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        payload = None
        if line.startswith("FINAL_RESULTS:"):
            _, _, text = line.partition(":")
            text = text.strip()
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                payload = None
        elif line.startswith("{"):
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                payload = None
        if isinstance(payload, dict) and (
            isinstance(payload.get("per_method"), dict)
            or isinstance(payload.get("seed_results"), list)
            or payload.get("best_method")
        ):
            return payload
    return {}


def _benchmark_scores(summary: dict) -> tuple[str, str | None, float | None, float | None, int]:
    """Return (metric_name, candidate_method, candidate_value, best_other_value, num_seeds)."""
    metric_name = str(summary.get("primary_metric") or summary.get("metric_name") or "metric")
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    candidate_method = str(summary.get("candidate_method") or ("cggr" if "cggr" in per_method else summary.get("best_method") or "")).strip() or None

    def _metric_for(method_name: str) -> float | None:
        row = per_method.get(method_name)
        if not isinstance(row, dict):
            return None
        raw = row.get(metric_name)
        if raw is None:
            raw = row.get("metric_value")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    candidate_value = _metric_for(candidate_method) if candidate_method else None
    best_other = None
    for method_name, row in per_method.items():
        if method_name == candidate_method or not isinstance(row, dict):
            continue
        try:
            value = float(row.get(metric_name, row.get("metric_value")))
        except (TypeError, ValueError):
            continue
        if best_other is None or value > best_other:
            best_other = value

    seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
    num_seeds = int(summary.get("num_seeds") or len(seed_results) or 0)
    return metric_name, candidate_method, candidate_value, best_other, num_seeds


def _normalize_command_tokens(command: str | None, python_bin: str) -> list[str]:
    if not command:
        return []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    if tokens and tokens[0] in {"python", "python3"}:
        tokens[0] = python_bin
    return tokens


def _run_experiment(
    workdir: Path,
    code_dir: Path,
    time_budget: int,
    *,
    baseline_command: str | None = None,
    run_id: int | None = None,
    execution_context: dict | None = None,
) -> dict:
    """Run a single experiment iteration with time budget."""
    log_path = workdir / "run.log"
    eval_candidates = []
    if run_id is not None:
        row = db.fetchone("SELECT deep_insight_id FROM experiment_runs WHERE id=?", (run_id,))
        if row and row.get("deep_insight_id") is not None:
            eval_candidates.append(plan_file_path(int(row["deep_insight_id"]), "evaluate.py"))
    eval_candidates.extend((workdir / "spec" / "evaluate.py", workdir / "evaluate.py"))
    eval_path = next((path for path in eval_candidates if path.exists()), workdir / "spec" / "evaluate.py")

    python_bin = RUNTIME_PYTHON or sys.executable
    command_tokens = _normalize_command_tokens(baseline_command, python_bin)
    if not command_tokens:
        train_file = _find_train_file(code_dir)
        train_script = str(train_file.relative_to(code_dir)) if train_file else "train.py"
        command_tokens = [python_bin, train_script]

    start = time.time()
    worker = (execution_context or {}).get("worker") if execution_context else None
    try:
        if run_id is not None and ssh_gpu_backend.is_ssh_worker(worker):
            remote = ssh_gpu_backend.run_remote_experiment(
                worker=worker,
                run_id=run_id,
                local_workdir=workdir,
                local_code_dir=code_dir,
                time_budget=time_budget,
                command_tokens=command_tokens,
                local_python=python_bin,
            )
            stdout = remote.get("stdout") or ""
            stderr = remote.get("stderr") or ""
            duration = time.time() - start
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(stdout)
                if stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(stderr)
            if int(remote.get("returncode") or 0) != 0:
                error = stderr[-500:] if stderr else stdout[-500:] if stdout else "nonzero exit"
                return {
                    "status": "crash",
                    "duration": duration,
                    "error": error,
                    "command_tokens": command_tokens,
                    "log_path": str(log_path),
                    "backend": "ssh",
                    "remote_host": remote.get("remote_host"),
                    "worker_id": remote.get("worker_id"),
                }
            execution_meta = {
                "backend": "ssh",
                "remote_host": remote.get("remote_host"),
                "worker_id": remote.get("worker_id"),
                "visible_device": remote.get("visible_device"),
            }
        else:
            proc = subprocess.run(
                command_tokens,
                cwd=str(code_dir),
                timeout=time_budget + 60,
                capture_output=True,
                text=True,
            )
            duration = time.time() - start

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(proc.stdout)
                if proc.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(proc.stderr)

            if proc.returncode != 0:
                return {
                    "status": "crash",
                    "duration": duration,
                    "error": proc.stderr[-500:] if proc.stderr else "nonzero exit",
                    "command_tokens": command_tokens,
                    "log_path": str(log_path),
                }
            execution_meta = {}
    except subprocess.TimeoutExpired:
        return {"status": "crash", "duration": time.time() - start, "error": "timeout"}
    except Exception as e:
        return {"status": "crash", "duration": time.time() - start, "error": str(e)}

    metric = None
    benchmark_summary = _parse_benchmark_summary_from_log(log_path)
    benchmark_metric_name, benchmark_candidate_method, benchmark_candidate_value, benchmark_baseline_value, benchmark_num_seeds = _benchmark_scores(benchmark_summary) if benchmark_summary else ("metric", None, None, None, 0)
    if benchmark_candidate_value is not None:
        metric = benchmark_candidate_value

    if eval_path.exists():
        try:
            eval_result = subprocess.run(
                [python_bin, str(eval_path), str(log_path)],
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
        "command_tokens": command_tokens,
        "log_path": str(log_path),
        "benchmark_summary": benchmark_summary,
        "benchmark_metric_name": benchmark_metric_name if benchmark_summary else None,
        "benchmark_candidate_method": benchmark_candidate_method,
        "benchmark_baseline_metric": benchmark_baseline_value,
        "benchmark_num_seeds": benchmark_num_seeds if benchmark_summary else 0,
        **execution_meta,
    }


def _git_commit(code_dir: Path, message: str) -> str | None:
    """Commit changes in code_dir, return short hash."""
    git_bin = _git_binary()
    if not git_bin:
        return None
    try:
        subprocess.run([git_bin, "add", "-A"], cwd=str(code_dir),
                       capture_output=True, timeout=10)
        subprocess.run([git_bin, "commit", "-m", message], cwd=str(code_dir),
                       capture_output=True, timeout=10)
        result = subprocess.run([git_bin, "rev-parse", "--short", "HEAD"],
                                cwd=str(code_dir), capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return None


def _git_reset(code_dir: Path, commit_hash: str):
    """Reset code_dir to a specific commit."""
    git_bin = _git_binary()
    if not git_bin or not commit_hash:
        return
    try:
        subprocess.run([git_bin, "reset", "--hard", commit_hash],
                       cwd=str(code_dir), capture_output=True, timeout=10)
    except Exception:
        pass


def _git_diff(code_dir: Path) -> str:
    """Get current diff in code_dir."""
    git_bin = _git_binary()
    if not git_bin:
        return ""
    try:
        result = subprocess.run([git_bin, "diff", "HEAD~1"],
                                cwd=str(code_dir), capture_output=True, text=True, timeout=10)
        return result.stdout[:2000]
    except Exception:
        return ""


def _snapshot_repo_tree(code_dir: Path, snapshot_dir: Path) -> None:
    """Store a full copy of the current repo tree for non-git rollback."""
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(code_dir, snapshot_dir, dirs_exist_ok=True)


def _restore_repo_tree(snapshot_dir: Path, code_dir: Path) -> None:
    """Restore a full repo tree snapshot when git is unavailable."""
    if not snapshot_dir.exists():
        return
    if code_dir.exists():
        shutil.rmtree(code_dir)
    shutil.copytree(snapshot_dir, code_dir, dirs_exist_ok=True)


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


def _determine_final_verdict(
    *,
    baseline: float,
    best_value: float,
    direction: str,
    criteria: dict,
    total_iters: int,
    total_kept: int,
    refute_min: int,
    benchmark_summary: dict | None = None,
) -> str:
    """Classify the overall run outcome.

    A reproduction-only run is useful as an execution checkpoint, but it is not
    scientific confirmation. Confirmation requires a positive improvement signal
    during hypothesis testing, while refutation requires exhausting at least the
    minimum refutation budget.
    """
    effect = best_value - baseline if direction == "higher" else baseline - best_value
    effect_pct = (effect / abs(baseline) * 100) if baseline != 0 else 0
    is_improvement = effect > 0
    exciting = criteria.get("exciting", 0)
    solid = criteria.get("solid", 0)

    if total_iters <= 0:
        summary = benchmark_summary or {}
        if summary:
            metric_name, candidate_method, candidate_value, best_other, num_seeds = _benchmark_scores(summary)
            if candidate_method and candidate_value is not None and best_other is not None and num_seeds >= 3:
                benchmark_effect = candidate_value - best_other if direction == "higher" else best_other - candidate_value
                benchmark_effect_pct = (benchmark_effect / abs(best_other) * 100) if best_other else 0
                best_method = str(summary.get("best_method") or "").strip().lower()
                if best_method and candidate_method.lower() != best_method:
                    return "inconclusive"
                if exciting and benchmark_effect > 0 and _meets_threshold(candidate_value, exciting, direction):
                    return "confirmed"
                if solid and benchmark_effect > 0 and _meets_threshold(candidate_value, solid, direction):
                    return "confirmed"
                if benchmark_effect_pct > 1.0:
                    return "confirmed"
                return "inconclusive"
        return "reproduced"
    if exciting and is_improvement and _meets_threshold(best_value, exciting, direction) and total_kept > 0:
        return "confirmed"
    if solid and is_improvement and _meets_threshold(best_value, solid, direction) and total_kept > 0:
        return "confirmed"
    if is_improvement and effect_pct > 1.0 and total_kept > 0:
        return "confirmed"
    if total_iters >= refute_min and not is_improvement:
        return "refuted"
    return "inconclusive"


def _find_train_file(code_dir: Path, preferred: str | None = None) -> Path | None:
    if preferred:
        preferred_path = code_dir / preferred
        if preferred_path.exists():
            return preferred_path
        preferred_name = Path(preferred).name
        for match in sorted(code_dir.rglob(preferred_name)):
            rel = match.relative_to(code_dir).as_posix()
            if rel.endswith(preferred.replace("\\", "/")):
                return match

    for pattern in ["train*.py", "main*.py", "run*.py", "inference.py"]:
        matches = sorted(code_dir.rglob(pattern))
        if matches:
            return matches[0]
    py_files = sorted(code_dir.rglob("*.py"))
    return py_files[0] if py_files else None


def _read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _record_artifact(
    run_id: int,
    artifact_type: str,
    path: Path,
    *,
    metric_key: str | None = None,
    metric_value: float | None = None,
    metadata: dict | None = None,
) -> None:
    db.execute(
        """
        INSERT INTO experiment_artifacts (run_id, artifact_type, path, metric_key, metric_value, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            artifact_type,
            str(path),
            metric_key,
            metric_value,
            json.dumps(metadata or {}),
        ),
    )


def _read_experiment_spec(
    run: dict,
    insight: dict,
    workdir: Path,
    *,
    criteria: dict,
    proxy: dict,
) -> ExperimentSpec:
    insight_spec = DeepInsightSpec.from_raw(insight)
    candidates = [
        plan_file_path(int(run["deep_insight_id"]), "experiment_spec.json"),
        workdir / "spec" / "experiment_spec.json",
        workdir / "experiment_spec.json",
    ]
    spec_path = next((path for path in candidates if path.exists()), candidates[0])
    payload = _read_json_file(spec_path, {})
    if isinstance(payload, dict):
        artifact_paths = payload.get("artifact_paths") if isinstance(payload.get("artifact_paths"), dict) else {}
    else:
        artifact_paths = {}
    spec = ExperimentSpec.from_run_row(run, insight_spec, success_criteria=criteria, proxy_config=proxy)
    spec.artifact_paths.update(artifact_paths)
    return spec


def _run_environment_scout(spec: ExperimentSpec, code_dir: Path) -> dict:
    train_file = _find_train_file(code_dir, spec.proxy_config.get("main_train_file"))
    code_files = list(code_dir.rglob("*.py")) if code_dir.exists() else []
    report = {
        "role": "EnvironmentScout",
        "formal_experiment": spec.formal_experiment,
        "smoke_test_only": spec.smoke_test_only,
        "resource_class": spec.resource_class or "cpu",
        "codebase_url": spec.codebase.get("url"),
        "baseline_command": spec.proxy_config.get("baseline_command"),
        "main_train_file": spec.proxy_config.get("main_train_file"),
        "resolved_train_file": train_file.relative_to(code_dir).as_posix() if train_file else None,
        "code_file_count": len(code_files),
        "git_available": bool(_git_binary()),
        "entrypoint_exists": train_file is not None or bool(spec.proxy_config.get("baseline_command")),
    }
    report["formal_ready"] = bool(report["entrypoint_exists"] and spec.formal_experiment)
    return report


def _judge_iteration_plan(
    spec: ExperimentSpec,
    *,
    iteration: int,
    history: list[dict],
    baseline: float | None,
    best_so_far: float | None,
) -> dict:
    last = history[-1] if history else {}
    crash_streak = 0
    for row in reversed(history):
        if row.get("status") == "crash":
            crash_streak += 1
            continue
        break
    if crash_streak >= 3:
        return {
            "role": "ExperimentJudge",
            "action": "stop",
            "continue": False,
            "reason": "Repeated execution crashes indicate the environment or baseline is broken.",
            "focus": "repair baseline before more hypothesis iterations",
        }
    if not history:
        focus = "establish first hypothesis-driven code modification"
    elif last.get("status") == "crash":
        focus = "repair the execution failure before exploring new hypotheses"
    elif last.get("status") == "discard":
        focus = "change approach because the last edit did not isolate the hypothesis"
    else:
        focus = "build on the last kept improvement while preserving baseline fairness"
    return {
        "role": "ExperimentJudge",
        "action": "continue",
        "continue": True,
        "reason": spec.judgement.summary or "Structured experiment review passed.",
        "focus": focus,
        "baseline": baseline,
        "best_so_far": best_so_far,
        "iteration": iteration,
    }


def _judge_iteration_result(
    *,
    result: dict,
    metric: float | None,
    best_before: float,
    baseline: float,
    direction: str,
    criteria: dict,
    iteration_index: int,
    refute_min: int,
) -> dict:
    exciting = criteria.get("exciting", 0)
    solid = criteria.get("solid", 0)
    disappointing = criteria.get("disappointing", 0)

    if result.get("status") == "crash" or metric is None:
        return {
            "role": "ResultJudge",
            "status": "crash",
            "summary": result.get("error") or "Experiment crashed or produced no metric.",
            "anomaly_type": "execution_failure",
            "continue": True,
            "terminate": False,
        }

    improved = _is_better(metric, best_before, direction)
    status = "keep" if improved else "discard"
    anomaly = "hypothesis_signal" if improved else "no_gain"
    summary = "Metric improved and was kept." if improved else "Metric did not improve; discard the change."
    terminate = False
    stop_reason = ""
    if exciting and _meets_threshold(metric, exciting, direction):
        terminate = True
        stop_reason = "Exciting threshold reached."
    elif solid and _meets_threshold(metric, solid, direction) and iteration_index >= 10:
        stop_reason = "Solid threshold reached; continue only if more evidence is needed."
    elif iteration_index >= refute_min and not _is_better(best_before, baseline, direction) and not improved:
        terminate = True
        stop_reason = "No improvement over baseline after the minimum refutation budget."
        anomaly = "hypothesis_refuted"
    elif disappointing and _meets_threshold(metric, disappointing, "lower" if direction == "higher" else "higher"):
        anomaly = "disappointing_result"

    return {
        "role": "ResultJudge",
        "status": status,
        "summary": summary,
        "anomaly_type": anomaly,
        "continue": not terminate,
        "terminate": terminate,
        "stop_reason": stop_reason,
        "metric": metric,
    }


def _write_iteration_packet(workdir: Path, packet: ExperimentIterationPacket, run_id: int) -> Path:
    packet_dir = workdir / "results" / "iteration_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    path = packet_dir / f"{packet.phase}_{packet.iteration_number:03d}.json"
    path.write_text(json.dumps(packet.to_dict(), indent=2), encoding="utf-8")
    _record_artifact(
        run_id,
        "source_data",
        path,
        metric_key=packet.metric_name,
        metric_value=packet.metric_value,
        metadata={"contract_type": "ExperimentIterationPacket", "phase": packet.phase, "status": packet.status},
    )
    return path


def _launch_coding_agent(workdir: Path, code_dir: Path, iteration: int,
                         method_desc: str, best_so_far: float | None,
                         baseline: float | None, history: list[dict],
                         spec: ExperimentSpec | None = None,
                         success_criteria: dict | None = None,
                         supervisor_plan: dict | None = None) -> dict:
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

    proxy = _read_proxy_config(workdir)
    success_criteria = success_criteria or {}
    if spec and codex_executor.codex_available():
        codex_result = codex_executor.run_codex_iteration(
            workdir=workdir,
            code_dir=code_dir,
            iteration=iteration,
            method_desc=method_desc,
            best_so_far=best_so_far,
            baseline=baseline,
            history=history,
            proxy=proxy,
            success_criteria=success_criteria,
            experimental_plan=spec.experimental_plan,
            evidence_plan=spec.evidence_plan,
            supervisor_plan=supervisor_plan,
        )
        if codex_result.get("ok"):
            summary = str(codex_result.get("summary") or f"Codex repo edit (iter {iteration})")
            return {
                "description": summary[:500],
                "artifact_paths": codex_result.get("artifact_paths", {}),
                "executor": "codex",
            }
        print(f"[LOOP] Codex iteration fallback at iter {iteration}: {codex_result.get('error') or codex_result.get('stderr') or codex_result.get('returncode')}", flush=True)

    train_file = _find_train_file(code_dir, proxy.get("main_train_file"))

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

    display_name = train_file.name if train_file else "train.py"
    if train_file:
        try:
            display_name = train_file.relative_to(code_dir).as_posix()
        except Exception:
            display_name = train_file.name

    prompt = textwrap.dedent(f"""\
        # Method to Implement
        {method_desc[:1500]}
        
        # Current State
        Baseline metric: {baseline}
        Best so far: {best_so_far}
        Iteration: {iteration}
        Supervisor Plan: {json.dumps(supervisor_plan or {}, ensure_ascii=False)[:1200]}
        
        # Recent History
        {history_text if history_text else "No history yet - this is the first modification."}
        
        # Current Code ({display_name})
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
            return {
                "description": f"LLM output not valid Python (iter {iteration})",
                "artifact_paths": {},
                "executor": "legacy_llm",
            }

        if train_file and len(new_code) > 50:
            train_file.write_text(new_code, encoding="utf-8")
            return {
                "description": f"Modified {train_file.name} (iter {iteration})",
                "artifact_paths": {},
                "executor": "legacy_llm",
            }
    except Exception as e:
        return {
            "description": f"LLM code generation failed: {e}",
            "artifact_paths": {},
            "executor": "legacy_llm",
        }

    return {
        "description": f"No modification applied (iter {iteration})",
        "artifact_paths": {},
        "executor": "legacy_llm",
    }


def run_validation_loop(run_id: int, execution_context: dict | None = None) -> dict:
    """Execute the full two-phase validation loop for an experiment run.

    Returns the final verdict and statistics.
    """
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"error": f"Run {run_id} not found"}

    insight_id = run["deep_insight_id"]
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return {"error": f"Insight {insight_id} not found"}
    run_layout = ensure_run_workspace(insight_id, run_id, insight=insight)
    workdir = Path(run["workdir"]) if run.get("workdir") else Path(run_layout["run_root"])
    if not workdir.exists() and Path(run_layout["run_root"]).exists():
        workdir = Path(run_layout["run_root"])
        db.execute("UPDATE experiment_runs SET workdir=? WHERE id=?", (str(workdir), run_id))
        db.commit()
    code_dir = workdir / "code"

    if not workdir.exists():
        return {"error": f"Workdir {workdir} does not exist"}

    criteria = _read_success_criteria(workdir, insight_id)
    proxy = _read_proxy_config(workdir, insight_id)

    spec = _read_experiment_spec(run, insight, workdir, criteria=criteria, proxy=proxy)
    metric_name = criteria.get("metric_name", "metric")
    direction = criteria.get("metric_direction", "higher")
    time_budget = proxy.get("time_budget_seconds", EXPERIMENT_TIME_BUDGET)
    baseline_command = proxy.get("baseline_command")
    max_iters = proxy.get("max_iterations", EXPERIMENT_MAX_ITERATIONS)
    repro_iters = proxy.get("reproduction_iterations", EXPERIMENT_REPRODUCTION_ITERS)
    refute_min = proxy.get("refute_min_iterations", EXPERIMENT_REFUTE_MIN_ITERS)

    if not spec.formal_experiment or spec.smoke_test_only:
        error = "Non-formal/smoke-only experiment cannot enter the validation loop."
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (error, run_id),
        )
        db.commit()
        write_latest_status(insight_id, {"stage": "validation_blocked", "status": "failed", "error": error}, run_id=run_id, insight=insight)
        return {"run_id": run_id, "verdict": "blocked", "reason": "non_formal_experiment"}

    method = spec.proposed_method
    method_desc = (
        f"Name: {method.get('name', '?')}\n"
        f"Type: {method.get('type', '?')}\n"
        f"Summary: {method.get('one_line', '')}\n"
        f"Definition: {method.get('definition', '')[:800]}\n"
        f"Pseudocode: {method.get('pseudocode', '')[:500]}"
    ).strip()
    if not method_desc:
        method_desc = insight.get("problem_statement", "") or insight.get("title", "")

    environment_report = _run_environment_scout(spec, code_dir)
    env_path = workdir / "results" / "environment_report.json"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps(environment_report, indent=2), encoding="utf-8")
    _record_artifact(run_id, "source_data", env_path, metadata={"contract_type": "EnvironmentScout"})
    if not environment_report.get("formal_ready"):
        error = "Formal validation blocked: environment scout could not locate a runnable baseline entrypoint."
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (error, run_id),
        )
        db.commit()
        write_latest_status(insight_id, {"stage": "environment_failed", "status": "failed", "error": error}, run_id=run_id, insight=insight)
        return {"run_id": run_id, "verdict": "failed", "reason": "environment_not_ready"}

    git_bin = _git_binary()
    if git_bin and not (code_dir / ".git").exists():
        subprocess.run([git_bin, "init"], cwd=str(code_dir), capture_output=True, timeout=10)
        subprocess.run([git_bin, "add", "-A"], cwd=str(code_dir), capture_output=True, timeout=10)
        subprocess.run([git_bin, "commit", "-m", "initial baseline"],
                       cwd=str(code_dir), capture_output=True, timeout=10)
    elif not git_bin:
        print("[LOOP] git not available; running without version-control checkpoints.", flush=True)
    snapshot_root = workdir / "results" / "repo_snapshots"
    best_repo_snapshot = snapshot_root / "best_state"
    if not git_bin:
        snapshot_root.mkdir(parents=True, exist_ok=True)
        _snapshot_repo_tree(code_dir, best_repo_snapshot)

    train_file = _find_train_file(code_dir, proxy.get("main_train_file"))
    best_train_snapshot = None
    if train_file and train_file.exists():
        best_train_snapshot = train_file.read_text(encoding="utf-8", errors="replace")

    db.execute("UPDATE experiment_runs SET status='reproducing', phase='reproduction', started_at=CURRENT_TIMESTAMP WHERE id=?", (run_id,))
    db.commit()
    promote_canonical_run(insight_id, run_id, insight=insight)
    write_latest_status(
        insight_id,
        {"stage": "reproduction", "status": "reproducing", "workdir": str(workdir), "metric_name": metric_name},
        run_id=run_id,
        insight=insight,
    )

    # ── Phase 1: Reproduction ──
    print(f"[LOOP] Phase 1: Reproducing baseline ({repro_iters} iterations)...", flush=True)
    baseline_values = []
    benchmark_baseline_values: list[float] = []
    benchmark_candidate_values: list[float] = []
    benchmark_summary: dict = {}

    for i in range(repro_iters):
        judge_plan = {
            "role": "ExperimentJudge",
            "phase": "reproduction",
            "focus": "establish baseline reproducibility before hypothesis edits",
            "continue": True,
        }
        result = _run_experiment(
            workdir,
            code_dir,
            time_budget,
            baseline_command=baseline_command,
            run_id=run_id,
            execution_context=execution_context,
        )
        metric = result.get("metric")
        packet = ExperimentIterationPacket(
            run_id=run_id,
            iteration_number=i + 1,
            phase="reproduction",
            status=result.get("status", "ok"),
            description=f"baseline run {i + 1}",
            metric_name=metric_name,
            metric_value=metric,
            baseline_value=None,
            best_value_before=None,
            best_value_after=metric,
            environment_report=environment_report,
            judge_report=judge_plan,
            execution_report=result,
            result_judgement={
                "role": "ResultJudge",
                "summary": "Reproduction run completed.",
                "status": result.get("status", "ok"),
            },
            artifact_paths={"log_path": result.get("log_path")},
        )
        _write_iteration_packet(workdir, packet, run_id)

        summary = result.get("benchmark_summary") if isinstance(result.get("benchmark_summary"), dict) else {}
        if summary:
            benchmark_summary = summary
            baseline_candidate = result.get("benchmark_baseline_metric")
            if baseline_candidate is not None:
                benchmark_baseline_values.append(float(baseline_candidate))
            if metric is not None:
                benchmark_candidate_values.append(float(metric))
            summary_path = workdir / "results" / "benchmark_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        db.execute(
            """INSERT INTO experiment_iterations
               (run_id, iteration_number, phase, metric_value, metric_name,
                peak_memory_mb, duration_seconds, status, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, i + 1, "reproduction", metric, metric_name,
             result.get("peak_memory_mb"), result.get("duration"),
             result.get("status", "ok"), json.dumps(packet.result_judgement)[:500])
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
        write_latest_status(
            insight_id,
            {"stage": "reproduction", "status": "failed", "error": "reproduction failed: no metric obtained"},
            run_id=run_id,
            insight=insight,
        )
        print(f"[LOOP] Phase 1 FAILED: could not obtain baseline metric", flush=True)
        return {"verdict": "failed", "reason": "reproduction_failure"}

    benchmark_mode = bool(benchmark_summary and benchmark_baseline_values and benchmark_candidate_values)
    if benchmark_mode:
        baseline = sum(benchmark_baseline_values) / len(benchmark_baseline_values)
        best_value = sum(benchmark_candidate_values) / len(benchmark_candidate_values)
    else:
        baseline = sum(baseline_values) / len(baseline_values)
        best_value = baseline
    baseline_commit = None
    if git_bin:
        baseline_commit = subprocess.run(
            [git_bin, "rev-parse", "--short", "HEAD"],
            cwd=str(code_dir), capture_output=True, text=True, timeout=5
        ).stdout.strip()

    db.execute(
        "UPDATE experiment_runs SET baseline_metric_value=?, best_metric_value=?, phase='hypothesis_testing', status='testing' WHERE id=?",
        (baseline, best_value, run_id))
    db.commit()
    write_latest_status(
        insight_id,
        {
            "stage": "hypothesis_testing",
            "status": "testing",
            "baseline_metric_value": baseline,
            "best_metric_value": best_value,
            "metric_name": metric_name,
        },
        run_id=run_id,
        insight=insight,
    )
    if benchmark_mode:
        print(
            f"[LOOP] Benchmark baseline established: best_non_target_{metric_name}={baseline:.6f}, "
            f"target={best_value:.6f}",
            flush=True,
        )
    else:
        print(f"[LOOP] Baseline established: {metric_name}={baseline:.6f}", flush=True)

    # ── Phase 2: Hypothesis Testing ──
    print(f"[LOOP] Phase 2: Hypothesis testing (max {max_iters} iterations)...", flush=True)
    if not benchmark_mode:
        best_value = baseline
    best_commit = baseline_commit
    total_kept = 0
    iter_num = repro_iters
    effect_pct = 0.0
    history = []
    loop_start = time.time()
    stop_reason = ""

    for i in range(max_iters):
        iter_num = repro_iters + i + 1
        judge_plan = _judge_iteration_plan(
            spec,
            iteration=i + 1,
            history=history,
            baseline=baseline,
            best_so_far=best_value,
        )
        if not judge_plan.get("continue"):
            stop_reason = judge_plan.get("reason", "")
            print(f"[LOOP] Judge requested stop before iter {i+1}: {stop_reason}", flush=True)
            break

        supervisor_plan = experiment_supervisor.build_supervisor_plan(
            spec=spec,
            environment_report=environment_report,
            baseline=baseline,
            best_so_far=best_value,
            history=history,
            iteration=i + 1,
            success_criteria=criteria,
        )
        supervisor_artifacts = experiment_supervisor.write_supervisor_artifacts(
            workdir,
            supervisor_plan,
        )

        coding_step = _launch_coding_agent(
            workdir,
            code_dir,
            i + 1,
            method_desc,
            best_value,
            baseline,
            history,
            spec=spec,
            success_criteria=criteria,
            supervisor_plan=supervisor_plan,
        )
        desc = coding_step["description"]

        commit_hash = _git_commit(code_dir, f"experiment iter {i+1}: {desc[:80]}")
        best_before = best_value

        result = _run_experiment(
            workdir,
            code_dir,
            time_budget,
            baseline_command=baseline_command,
            run_id=run_id,
            execution_context=execution_context,
        )
        metric = result.get("metric")

        result_judgement = _judge_iteration_result(
            result=result,
            metric=metric,
            best_before=best_before,
            baseline=baseline,
            direction=direction,
            criteria=criteria,
            iteration_index=i + 1,
            refute_min=refute_min,
        )
        status = result_judgement["status"]
        if status == "keep":
            best_value = metric if metric is not None else best_value
            best_commit = commit_hash
            total_kept += 1
            if train_file and train_file.exists():
                best_train_snapshot = train_file.read_text(encoding="utf-8", errors="replace")
            if not git_bin:
                _snapshot_repo_tree(code_dir, best_repo_snapshot)
        elif status == "discard":
            _git_reset(code_dir, best_commit)
            if not git_bin:
                _restore_repo_tree(best_repo_snapshot, code_dir)
                train_file = _find_train_file(code_dir, proxy.get("main_train_file"))
            elif (not best_commit) and train_file and best_train_snapshot is not None:
                train_file.write_text(best_train_snapshot, encoding="utf-8")
        elif result.get("status") == "crash":
            _git_reset(code_dir, best_commit)
            if not git_bin:
                _restore_repo_tree(best_repo_snapshot, code_dir)
                train_file = _find_train_file(code_dir, proxy.get("main_train_file"))
            elif (not best_commit) and train_file and best_train_snapshot is not None:
                train_file.write_text(best_train_snapshot, encoding="utf-8")

        diff = _git_diff(code_dir) if status == "keep" else ""
        packet = ExperimentIterationPacket(
            run_id=run_id,
            iteration_number=iter_num,
            phase="hypothesis_testing",
            status=status,
            description=desc[:500],
            metric_name=metric_name,
            metric_value=metric,
            baseline_value=baseline,
            best_value_before=best_before,
            best_value_after=best_value,
            environment_report=environment_report,
            judge_report=judge_plan,
            execution_report=result,
            result_judgement=result_judgement,
            artifact_paths={
                "log_path": result.get("log_path"),
                **supervisor_artifacts,
                **coding_step.get("artifact_paths", {}),
            },
            commit_hash=commit_hash or "",
            code_diff=diff,
        )
        _write_iteration_packet(workdir, packet, run_id)

        db.execute(
            """INSERT INTO experiment_iterations
               (run_id, iteration_number, phase, code_diff, commit_hash,
                metric_value, metric_name, peak_memory_mb, duration_seconds,
                status, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, iter_num, "hypothesis_testing", diff, commit_hash,
             metric, metric_name, result.get("peak_memory_mb"),
             result.get("duration"), status, json.dumps(result_judgement)[:500])
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
            "judge_report": judge_plan,
            "result_judgement": result_judgement,
        })

        if (i + 1) % 5 == 0:
            print(f"[LOOP] Iter {i+1}/{max_iters}: best={best_value:.6f} "
                  f"(baseline={baseline:.6f}, kept={total_kept})", flush=True)

        # Check termination conditions
        if result_judgement.get("stop_reason"):
            stop_reason = result_judgement["stop_reason"]
        if result_judgement.get("terminate"):
            print(f"[LOOP] Judge terminated loop at iter {i+1}: {stop_reason}", flush=True)
            break

    # ── Determine verdict ──
    verdict = _determine_final_verdict(
        baseline=baseline,
        best_value=best_value,
        direction=direction,
        criteria=criteria,
        total_iters=len(history),
        total_kept=total_kept,
        refute_min=refute_min,
        benchmark_summary=benchmark_summary if benchmark_mode else None,
    )
    effect = best_value - baseline if direction == "higher" else baseline - best_value
    effect_pct = (effect / abs(baseline) * 100) if baseline != 0 else 0

    total_time = time.time() - loop_start
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', hypothesis_verdict=?,
               effect_size=?, effect_pct=?, error_message=?,
               completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (verdict, effect, effect_pct, stop_reason or None, run_id)
    )
    db.commit()
    promote_canonical_run(insight_id, run_id, insight=insight)

    summary_path = workdir / "results" / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "verdict": verdict,
                "baseline": baseline,
                "best_value": best_value,
                "effect_size": effect,
                "effect_pct": effect_pct,
                "iterations_total": iter_num,
                "iterations_kept": total_kept,
                "environment_report": environment_report,
                "stop_reason": stop_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _record_artifact(
        run_id,
        "source_data",
        summary_path,
        metric_key=metric_name,
        metric_value=best_value,
        metadata={"contract_type": "ValidationSummary"},
    )
    db.commit()
    write_latest_status(
        insight_id,
        {
            "stage": "validation_complete",
            "status": "completed",
            "verdict": verdict,
            "baseline": baseline,
            "best_value": best_value,
            "effect_size": effect,
            "effect_pct": effect_pct,
            "iterations_total": iter_num,
            "iterations_kept": total_kept,
            "summary_path": str(summary_path),
        },
        run_id=run_id,
        insight=insight,
    )

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
        "environment_report": environment_report,
        "stop_reason": stop_reason,
    }
