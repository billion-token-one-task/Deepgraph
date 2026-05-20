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
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from agents.benchmark_audit import (
    benchmark_diagnostic_notes,
    benchmark_fairness_warnings_from_diff,
    benchmark_semantic_warnings,
)
from agents import codex_executor
from agents import experiment_supervisor
from agents.evosci_requirements import evosci_strict_gate_insight
from agents.workspace_layout import ensure_run_workspace, plan_file_path, promote_canonical_run, write_latest_status
from contracts import DeepInsightSpec, ExperimentIterationPacket, ExperimentSpec
from config import (
    ALLOW_SMOKE_EXPERIMENT_VALIDATION,
    EXPERIMENT_MAX_ITERATIONS,
    EXPERIMENT_PLATEAU_PATIENCE,
    EXPERIMENT_REFUTE_MIN_ITERS,
    EXPERIMENT_REPRODUCTION_ITERS,
    EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET,
    EXPERIMENT_TIME_BUDGET,
    REPRODUCTION_REPAIR_MAX_ROUNDS,
    RUNTIME_PYTHON,
)
from db import database as db
from agents.experiment_executor import (
    benchmark_env_for_execution,
    execution_diagnostics,
    find_train_file,
    normalize_command_tokens,
    command_entrypoint_exists,
    worker_visible_device,
)
from agents.metric_parser import (
    _FLOAT_RE,
    parse_benchmark_summary_from_log,
    parse_metric_from_log,
    benchmark_scores,
)


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
    return parse_metric_from_log(log_path, metric_name)


def _parse_benchmark_summary_from_log(log_path: Path) -> dict:
    return parse_benchmark_summary_from_log(log_path)


def _benchmark_scores(summary: dict) -> tuple[str, str | None, float | None, float | None, int]:
    return benchmark_scores(summary)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _execution_diagnostics(
    *,
    returncode: int | None,
    log_text: str,
    stderr: str = "",
    duration: float | None = None,
    time_budget: int | None = None,
    metric: float | None = None,
) -> dict:
    return execution_diagnostics(
        returncode=returncode,
        log_text=log_text,
        stderr=stderr,
        duration=duration,
        time_budget=time_budget,
        metric=metric,
    )


def _benchmark_package_complete(summary: dict, criteria: dict) -> bool:
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
    quality_gates = criteria.get("quality_gates") if isinstance(criteria.get("quality_gates"), dict) else {}
    contract = criteria.get("publication_evidence_contract") if isinstance(criteria.get("publication_evidence_contract"), dict) else {}
    try:
        minimum_seeds = int(contract.get("minimum_seeds") or quality_gates.get("minimum_seeds") or 3)
    except (TypeError, ValueError):
        minimum_seeds = 3
    try:
        num_seeds = int(summary.get("num_seeds") or len(seed_results) or 0)
    except (TypeError, ValueError):
        num_seeds = 0
    return bool(per_method and len(per_method) >= 2 and num_seeds >= minimum_seeds)


def _named_requirements(values) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values:
        if isinstance(value, dict):
            text = str(value.get("name") or value.get("dataset") or value.get("model") or "").strip()
        else:
            text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _canonical_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


_DATASET_ALIAS_GROUPS = (
    ("gsm8k", "openaigsm8k", "gradeschoolmath"),
    ("musique", "musiqueans", "multihopqa"),
    ("strategyqa", "strategyqa", "yesnoqa"),
    ("2wikimultihopqa", "twowikimultihopqa", "2wiki"),
    ("stresstestsplit", "simplevshard", "counterfactualpartition", "derivedstresssplit"),
)


_BASELINE_ALIAS_GROUPS = (
    ("vanilladirectanswering", "direct", "vanilla", "directanswering"),
    ("alwaysreasonchainofthought", "fixedcot", "cot", "chainofthought", "alwaysreasoncot"),
    ("selfconsistencyreasoning", "selfconsistency", "sc"),
    ("leasttomostprompting", "leasttomost", "ltm"),
    ("confidencegate", "adaptivegate", "budgetgate"),
    ("disagreementrouting", "disagreementgate", "disagreement", "selfconsistencygate"),
    ("randombudgetmatchedrouting", "randombudgetmatched", "randomrouting", "budgetmatchedrandom"),
    ("oracleroutingupperbound", "oracle", "oraclerouter", "upperbound", "cggroraclerouter"),
    ("cggr", "candidate", "proposedmethod"),
)


def _label_matches(required: str, observed: list[str], alias_groups: tuple[tuple[str, ...], ...]) -> bool:
    req = _canonical_label(required)
    if not req:
        return True
    observed_norms = {_canonical_label(item) for item in observed if _canonical_label(item)}
    if any(req in item or item in req for item in observed_norms):
        return True
    for group in alias_groups:
        group_norms = {_canonical_label(item) for item in group}
        if req in group_norms and observed_norms.intersection(group_norms):
            return True
    return False


def _observed_dataset_labels(summary: dict) -> list[str]:
    labels: list[str] = []
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    aliases = summary.get("dataset_aliases") if isinstance(summary.get("dataset_aliases"), list) else []
    for row in datasets + aliases:
        if isinstance(row, dict):
            for key in ("name", "id", "hf_dataset", "dataset", "source", "alias"):
                if row.get(key):
                    labels.append(str(row[key]))
            for alias in row.get("aliases") or []:
                labels.append(str(alias))
        elif row:
            labels.append(str(row))
    for key in ("id", "name", "hf_dataset", "dataset", "source"):
        if dataset.get(key):
            labels.append(str(dataset[key]))
    return _unique_ordered(labels)


def _observed_method_labels(summary: dict, per_method: dict) -> list[str]:
    labels: list[str] = [str(key) for key in per_method.keys()]
    aliases = summary.get("baseline_aliases")
    if isinstance(aliases, dict):
        for key, value in aliases.items():
            labels.append(str(key))
            if isinstance(value, list):
                labels.extend(str(item) for item in value)
            elif value:
                labels.append(str(value))
    elif isinstance(aliases, list):
        labels.extend(str(item) for item in aliases)
    method_aliases = summary.get("method_aliases")
    if isinstance(method_aliases, dict):
        for key, value in method_aliases.items():
            labels.append(str(key))
            if isinstance(value, list):
                labels.extend(str(item) for item in value)
            elif value:
                labels.append(str(value))
    return _unique_ordered(labels)


def _unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _benchmark_readiness_blockers(summary: dict, criteria: dict, verdict: str) -> list[str]:
    blockers: list[str] = []
    quality_gates = criteria.get("quality_gates") if isinstance(criteria.get("quality_gates"), dict) else {}
    contract = criteria.get("publication_evidence_contract") if isinstance(criteria.get("publication_evidence_contract"), dict) else {}
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
    try:
        minimum_seeds = int(contract.get("minimum_seeds") or quality_gates.get("minimum_seeds") or 3)
    except (TypeError, ValueError):
        minimum_seeds = 3
    try:
        num_seeds = int(summary.get("num_seeds") or len(seed_results) or 0)
    except (TypeError, ValueError):
        num_seeds = 0
    if verdict != "confirmed":
        blockers.append(f"verdict is {verdict!r}, not confirmed")
    if summary.get("full_benchmark_completed") is False:
        blockers.append("benchmark_summary.full_benchmark_completed is false")
    if summary.get("load_failures"):
        blockers.append("benchmark_summary.load_failures is non-empty")
    if not per_method or len(per_method) < 2:
        blockers.append("benchmark_summary.per_method must contain at least two methods")
    if num_seeds < minimum_seeds:
        blockers.append(f"num_seeds={num_seeds} is below minimum_seeds={minimum_seeds}")

    required_datasets = _named_requirements(
        contract.get("required_real_benchmarks") or contract.get("required_datasets") or []
    )
    observed_datasets = _observed_dataset_labels(summary)
    missing_datasets = [
        req
        for req in required_datasets
        if not _label_matches(req, observed_datasets, _DATASET_ALIAS_GROUPS)
    ]
    if required_datasets and missing_datasets:
        blockers.append(
            "required benchmark coverage missing: "
            + ", ".join(missing_datasets)
        )

    required_baselines = _named_requirements(contract.get("required_baselines") or [])
    observed_methods = _observed_method_labels(summary, per_method)
    missing_baselines = [
        name
        for name in required_baselines
        if not _label_matches(name, observed_methods, _BASELINE_ALIAS_GROUPS)
    ]
    if required_baselines and missing_baselines:
        blockers.append("required baselines missing: " + ", ".join(missing_baselines))

    required_ablations = _named_requirements(contract.get("required_ablations") or [])
    has_ablations = bool(
        summary.get("ablations")
        or summary.get("ablation_results")
        or summary.get("ablation_table")
    )
    if required_ablations and not has_ablations:
        blockers.append("required ablation table is missing")

    direction = str(criteria.get("metric_direction") or "higher")
    semantic_warnings = benchmark_semantic_warnings(
        summary,
        metric_name=str(summary.get("primary_metric") or summary.get("metric_name") or criteria.get("metric_name") or ""),
        candidate_method=str(summary.get("candidate_method") or ""),
        direction=direction,
    )
    blockers.extend(f"benchmark semantic warning: {warning}" for warning in semantic_warnings)

    return blockers


def _write_benchmark_artifact_manifest(
    workdir: Path,
    *,
    run_id: int,
    metric_name: str,
    benchmark_summary: dict,
    criteria: dict,
    verdict: str,
    validation_summary_path: Path,
) -> tuple[Path | None, bool]:
    if not benchmark_summary:
        return None, False
    results_dir = workdir / "results"
    summary_path = results_dir / "benchmark_summary.json"
    readiness_blockers = _benchmark_readiness_blockers(benchmark_summary, criteria, verdict)
    diagnostic_notes = benchmark_diagnostic_notes(
        benchmark_summary,
        metric_name=str(benchmark_summary.get("primary_metric") or benchmark_summary.get("metric_name") or metric_name),
        candidate_method=str(benchmark_summary.get("candidate_method") or ""),
        direction=str(criteria.get("metric_direction") or "higher"),
    )
    full_completed = bool(not readiness_blockers and _benchmark_package_complete(benchmark_summary, criteria))
    seed_results = benchmark_summary.get("seed_results") if isinstance(benchmark_summary.get("seed_results"), list) else []
    per_method = benchmark_summary.get("per_method") if isinstance(benchmark_summary.get("per_method"), dict) else {}
    manifest = {
        "run_id": run_id,
        "contract_type": "BenchmarkArtifactManifest",
        "full_benchmark_completed": full_completed,
        "verdict": verdict,
        "metric_name": metric_name,
        "num_seeds": benchmark_summary.get("num_seeds") or len(seed_results),
        "method_count": len(per_method),
        "primary_metric": benchmark_summary.get("primary_metric") or benchmark_summary.get("metric_name") or metric_name,
        "dataset": benchmark_summary.get("dataset") or {},
        "datasets": benchmark_summary.get("datasets") or [],
        "model": benchmark_summary.get("model") or {},
        "hardware": benchmark_summary.get("hardware")
        or ((benchmark_summary.get("model") or {}).get("hardware") if isinstance(benchmark_summary.get("model"), dict) else ""),
        "readiness_blockers": readiness_blockers,
        "diagnostic_notes": diagnostic_notes,
        "artifacts": {
            "benchmark_summary": str(summary_path),
            "validation_summary": str(validation_summary_path),
            "run_log": str(workdir / "run.log"),
            "iteration_packets": str(results_dir / "iteration_packets"),
        },
        "required_before_manuscript": [
            "benchmark_summary",
            "validation_summary",
            "run_log",
            "iteration_packets",
        ],
    }
    path = results_dir / "benchmark_artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, full_completed


def _normalize_command_tokens(command: str | None, python_bin: str) -> list[str]:
    return normalize_command_tokens(command, python_bin)


def _command_entrypoint_exists(tokens: list[str], code_dir: Path) -> bool:
    return command_entrypoint_exists(tokens, code_dir)


def _force_real_benchmark_command(proxy: dict, code_dir: Path, baseline_command: str | None) -> tuple[str | None, bool]:
    if not proxy.get("real_benchmark_required"):
        return baseline_command, False
    train_py = code_dir / "train.py"
    try:
        text = train_py.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return baseline_command, False
    required = ("final_results:", "full_benchmark_completed", "automodelforcausallm", "load_dataset")
    if not all(marker in text for marker in required):
        return baseline_command, False
    return "python train.py", baseline_command != "python train.py"


def _local_worker_visible_device(worker: dict | None) -> str | None:
    return worker_visible_device(worker)


def _workdir_uses_cggr_runner(workdir: Path) -> bool:
    train_py = workdir / "code" / "train.py"
    try:
        text = train_py.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
    except OSError:
        return False
    return '"cggr_mode": true' in text or "'cggr_mode': true" in text


def _benchmark_env_for_execution(workdir: Path, *, full_benchmark: bool = False) -> dict[str, str]:
    return benchmark_env_for_execution(workdir, full_benchmark=full_benchmark)


def _run_experiment(
    workdir: Path,
    code_dir: Path,
    time_budget: int,
    *,
    baseline_command: str | None = None,
    metric_name: str = "metric",
    run_id: int | None = None,
    execution_context: dict | None = None,
    full_benchmark: bool = False,
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
    if command_tokens and not _command_entrypoint_exists(command_tokens, code_dir):
        command_tokens = []
    if not command_tokens:
        train_file = _find_train_file(code_dir)
        train_script = str(train_file.relative_to(code_dir)) if train_file else "train.py"
        command_tokens = [python_bin, train_script]

    start = time.time()
    worker = (execution_context or {}).get("worker") if execution_context else None
    benchmark_env = _benchmark_env_for_execution(workdir, full_benchmark=full_benchmark)
    try:
        local_env = os.environ.copy()
        local_env.update(benchmark_env)
        visible_device = _local_worker_visible_device(worker)
        if visible_device is not None:
            local_env["CUDA_VISIBLE_DEVICES"] = visible_device
        proc = subprocess.run(
            command_tokens,
            cwd=str(code_dir),
            timeout=time_budget + 60,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=local_env,
        )
        duration = time.time() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stdout)
            if stderr:
                f.write("\n--- STDERR ---\n")
                f.write(stderr)

        if proc.returncode != 0:
            log_text = stdout + (("\n--- STDERR ---\n" + stderr) if stderr else "")
            diagnostics = _execution_diagnostics(
                returncode=proc.returncode,
                log_text=log_text,
                stderr=stderr,
                duration=duration,
                time_budget=time_budget,
            )
            return {
                "status": "crash",
                "duration": duration,
                "error": stderr[-500:] if stderr else "nonzero exit",
                **diagnostics,
                "command_tokens": command_tokens,
                "log_path": str(log_path),
                "backend": "local",
                "worker_id": worker.get("id") if worker else None,
                "visible_device": visible_device,
                "benchmark_env": benchmark_env,
            }
        execution_meta = {
            "backend": "local",
            "worker_id": worker.get("id") if worker else None,
            "visible_device": visible_device,
            "returncode": proc.returncode,
            "benchmark_env": benchmark_env,
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        try:
            log_path.write_text(
                f"Experiment timed out after {duration:.1f}s\nCommand: {command_tokens}\n{exc}\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return {
            "status": "crash",
            "duration": duration,
            "error": "timeout",
            "failure_type": "timeout",
            "final_results_present": False,
            "command_tokens": command_tokens,
            "log_path": str(log_path),
            "benchmark_env": benchmark_env,
        }
    except Exception as e:
        duration = time.time() - start
        try:
            log_path.write_text(
                f"Experiment launcher failed after {duration:.1f}s\nCommand: {command_tokens}\nError: {e}\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return {
            "status": "crash",
            "duration": duration,
            "error": str(e),
            "failure_type": "launcher_error",
            "final_results_present": False,
            "command_tokens": command_tokens,
            "log_path": str(log_path),
            "benchmark_env": benchmark_env,
        }

    metric = None
    benchmark_summary = _parse_benchmark_summary_from_log(log_path)
    benchmark_metric_name, benchmark_candidate_method, benchmark_candidate_value, benchmark_baseline_value, benchmark_num_seeds = _benchmark_scores(benchmark_summary) if benchmark_summary else ("metric", None, None, None, 0)
    if benchmark_candidate_value is not None:
        metric = benchmark_candidate_value
    if metric is None:
        metric = _parse_metric_from_log(log_path, metric_name)

    if metric is None and eval_path.exists():
        try:
            eval_result = subprocess.run(
                [python_bin, str(eval_path), str(log_path)],
                cwd=str(workdir),
                timeout=60,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if eval_result.stdout:
                for name in (metric_name, "metric_value"):
                    if not name:
                        continue
                    match = re.search(rf'{re.escape(name)}[:\s]+({_FLOAT_RE})', eval_result.stdout, re.IGNORECASE)
                    if match:
                        metric = float(match.group(1))
                        break
        except Exception:
            pass

    peak_mem = None
    log_text = _safe_read_text(log_path)
    mem_match = re.search(rf'peak_vram_mb[:\s]+({_FLOAT_RE})', log_text)
    if mem_match:
        peak_mem = float(mem_match.group(1))
    diagnostics = _execution_diagnostics(
        returncode=execution_meta.get("returncode"),
        log_text=log_text,
        duration=duration,
        time_budget=time_budget,
        metric=metric,
    )

    if metric is None:
        return {
            "status": "crash",
            "metric": None,
            "duration": duration,
            "peak_memory_mb": peak_mem,
            "error": diagnostics.get("failure_type") or "missing metric",
            **diagnostics,
            "command_tokens": command_tokens,
            "log_path": str(log_path),
            "benchmark_summary": benchmark_summary,
            "benchmark_metric_name": benchmark_metric_name if benchmark_summary else None,
            "benchmark_candidate_method": benchmark_candidate_method,
            "benchmark_baseline_metric": benchmark_baseline_value,
            "benchmark_num_seeds": benchmark_num_seeds if benchmark_summary else 0,
            **execution_meta,
        }

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
        **diagnostics,
        **execution_meta,
    }


def _git_commit(code_dir: Path, message: str) -> str | None:
    """Commit changes in code_dir, return short hash."""
    git_bin = _git_binary()
    if not git_bin:
        return None
    try:
        subprocess.run(
            [
                git_bin,
                "add",
                "-A",
                "--",
                ".",
                ":(exclude)AGENTS.md",
                ":(exclude)**/__pycache__/**",
                ":(exclude)**/*.pyc",
            ],
            cwd=str(code_dir),
            capture_output=True,
            timeout=10,
        )
        commit_result = subprocess.run(
            [git_bin, "commit", "-m", message],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if commit_result.returncode != 0:
            return None
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
    return find_train_file(code_dir, preferred)


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
    smoke_validation_allowed = bool(ALLOW_SMOKE_EXPERIMENT_VALIDATION and spec.smoke_test_only)
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
        "smoke_validation_allowed": smoke_validation_allowed,
    }
    report["formal_ready"] = bool(report["entrypoint_exists"] and (spec.formal_experiment or smoke_validation_allowed))
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
        direction = str(spec.success_criteria.get("metric_direction") or "higher").lower()
        if baseline is not None and best_so_far is not None and not _is_better(float(best_so_far), float(baseline), direction):
            focus = "close the remaining gap to baseline before treating the kept change as positive evidence"
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
    baseline_effect = metric - baseline if direction == "higher" else baseline - metric
    baseline_effect_pct = (baseline_effect / abs(baseline) * 100.0) if abs(baseline) > 1e-12 else None
    beats_baseline = baseline_effect > 1e-12
    ties_baseline = abs(baseline_effect) <= 1e-12
    status = "keep" if improved else "discard"
    summary_warnings = benchmark_semantic_warnings(
        result.get("benchmark_summary") if isinstance(result.get("benchmark_summary"), dict) else {},
        metric_name=str(result.get("benchmark_metric_name") or criteria.get("metric_name") or ""),
        candidate_method=str(result.get("benchmark_candidate_method") or ""),
        direction=direction,
    )

    if improved and beats_baseline:
        anomaly = "hypothesis_signal"
        summary = "Metric improved over best-so-far and beats the baseline."
    elif improved and ties_baseline:
        anomaly = "baseline_tie"
        summary = "Metric improved over best-so-far but only ties the baseline; keep as partial recovery."
    elif improved:
        anomaly = "partial_recovery"
        summary = "Metric improved over best-so-far but remains below baseline; keep only as partial recovery."
    else:
        anomaly = "no_gain"
        summary = "Metric did not improve; discard the change."
    terminate = False
    stop_reason = ""
    if exciting and beats_baseline and _meets_threshold(metric, exciting, direction):
        terminate = True
        stop_reason = "Exciting threshold reached."
    elif solid and beats_baseline and _meets_threshold(metric, solid, direction) and iteration_index >= 10:
        stop_reason = "Solid threshold reached; continue only if more evidence is needed."
    elif iteration_index >= refute_min and not _is_better(best_before, baseline, direction) and not improved:
        terminate = True
        stop_reason = "No improvement over baseline after the minimum refutation budget."
        anomaly = "hypothesis_refuted"
    elif disappointing and _meets_threshold(metric, disappointing, "lower" if direction == "higher" else "higher"):
        anomaly = "disappointing_result"

    if summary_warnings:
        status = "discard"
        anomaly = "benchmark_semantic_risk"
        terminate = False
        stop_reason = ""
        summary += " Benchmark semantic warning: " + summary_warnings[0]
        summary += " Discarding this iteration until the benchmark semantics are repaired."

    return {
        "role": "ResultJudge",
        "status": status,
        "summary": summary,
        "anomaly_type": anomaly,
        "continue": not terminate,
        "terminate": terminate,
        "stop_reason": stop_reason,
        "metric": metric,
        "baseline": baseline,
        "baseline_effect": baseline_effect,
        "baseline_effect_pct": baseline_effect_pct,
        "beats_baseline": beats_baseline,
        "benchmark_semantic_warnings": summary_warnings,
        "paper_evidence_warning": bool(summary_warnings),
    }


def _apply_benchmark_fairness_guard(*, status: str, result_judgement: dict, diff: str) -> tuple[str, list[str]]:
    """Reject iterations whose code diff changes candidate-only benchmark handling."""
    if status != "keep" or not diff:
        return status, []
    fairness_warnings = benchmark_fairness_warnings_from_diff(diff)
    if not fairness_warnings:
        return status, []

    existing_warnings = result_judgement.setdefault("benchmark_semantic_warnings", [])
    if not isinstance(existing_warnings, list):
        existing_warnings = []
        result_judgement["benchmark_semantic_warnings"] = existing_warnings
    existing_warnings.extend(w for w in fairness_warnings if w not in existing_warnings)
    result_judgement["status"] = "discard"
    result_judgement["anomaly_type"] = "benchmark_fairness_risk"
    result_judgement["paper_evidence_warning"] = True
    result_judgement["continue"] = True
    result_judgement["terminate"] = False
    result_judgement["stop_reason"] = ""
    result_judgement["summary"] = (
        str(result_judgement.get("summary") or "")
        + " Benchmark fairness warning: "
        + fairness_warnings[0]
    ).strip()
    return "discard", fairness_warnings


def _added_diff_text(diff: str) -> str:
    """Return only added diff lines, excluding diff headers."""
    return "\n".join(
        line[1:]
        for line in str(diff or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _blocked_pre_benchmark_diff_warnings(diff: str) -> list[str]:
    """Detect known-invalid candidate diffs before spending GPU time.

    These checks are intentionally narrow: they target intervention families
    that the supervisor has already declared off-limits after repeated failed
    attempts in this benchmark run.
    """
    if not diff:
        return []

    added = _added_diff_text(diff).lower()
    full = str(diff or "").lower()
    warnings: list[str] = []

    def add(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    if "_build_cggr_zero_budget_prompt" in full:
        add(
            "Pre-benchmark guard blocked a diff touching `_build_cggr_zero_budget_prompt`; "
            "recent zero-budget prompt-shortening attempts in this area lost utility."
        )
    if "_cggr_zero_budget_max_tokens" in full:
        add(
            "Pre-benchmark guard blocked a diff touching `_cggr_zero_budget_max_tokens`; "
            "recent token-cap microtuning is not admissible evidence."
        )

    answer_shape_markers = (
        "answer with only",
        "shortest exact answer",
        "shortest final answer",
        "shortest answer",
        "answer span",
        "answer phrase only",
        "phrase only",
        "short phrase",
        "do not explain",
        "no extra text",
    )
    if any(marker in added for marker in answer_shape_markers) and (
        "zero-budget" in full or "cggr" in full or "_build_cggr_zero_budget_prompt" in full
    ):
        add(
            "Pre-benchmark guard blocked a zero-budget answer-shape prompt change; "
            "shortest-answer, answer-span, or phrase-only rewrites are disallowed."
        )

    answer_shape_helper = re.search(
        r"def\s+_is_[a-z0-9_]*(?:goal|purpose|phrase|span|answer_shape)[a-z0-9_]*\s*\(",
        added,
    )
    if answer_shape_helper and ("question" in added or "zero-budget" in full or "cggr" in full):
        add(
            "Pre-benchmark guard blocked an answer-shape regex/helper change for zero-budget routing."
        )

    reasoning_budget_markers = (
        "zero-budget",
        "12-token",
        "token cap",
        "answer cap",
        "easy open",
        "goal/purpose",
        "phrase-only",
        "prompt",
    )
    if "reasoning_budget" in full and any(marker in added for marker in reasoning_budget_markers):
        add(
            "Pre-benchmark guard blocked reasoning-budget metadata/cap microtuning tied to zero-budget behavior."
        )

    context_helper_markers = (
        "_extract_context",
        "_context_to_text",
        "_question_with_context",
    )
    context_prompt_markers = (
        "benchmark-provided context",
        "context/passages/facts",
        "context fields",
        "context-preserving prompting",
    )
    if any(marker in added for marker in context_helper_markers):
        add(
            "Pre-benchmark guard blocked broad benchmark-context propagation helpers; "
            "recent all-method context prompt rewrites lowered utility."
        )
    elif any(marker in added for marker in context_prompt_markers) and (
        "prompt" in added or "question" in added or "all method" in added
    ):
        add(
            "Pre-benchmark guard blocked broad benchmark-context prompt propagation; "
            "recent all-method context prompt rewrites lowered utility."
        )

    return warnings


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


def _iteration_db_description(
    *,
    result_judgement: dict,
    coding_summary: str,
    executor: str | None = None,
) -> str:
    """Persist both the judge outcome and the actual attempted code change."""
    payload = dict(result_judgement)
    summary = str(coding_summary or "").strip()
    if summary:
        payload["coding_summary"] = summary[:500]
    if executor:
        payload["coding_executor"] = str(executor)
    return json.dumps(payload, ensure_ascii=False)


def _history_description_from_db(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            parts = []
            coding_summary = str(payload.get("coding_summary") or "").strip()
            if coding_summary:
                parts.append(coding_summary)
            anomaly = str(payload.get("anomaly_type") or "").strip()
            summary = str(payload.get("summary") or "").strip()
            if anomaly or summary:
                parts.append("judge=" + " ".join(part for part in (anomaly, summary) if part))
            warnings = payload.get("benchmark_semantic_warnings")
            if isinstance(warnings, list) and warnings:
                parts.append("warnings=" + "; ".join(str(item) for item in warnings[:2]))
            if parts:
                return " | ".join(parts)
    return text


def _resume_history_from_db(run_id: int, repro_iters: int) -> tuple[list[dict], int, int, str | None]:
    rows = db.fetchall(
        """
        SELECT iteration_number, status, metric_value, description, commit_hash
        FROM experiment_iterations
        WHERE run_id=? AND phase='hypothesis_testing'
        ORDER BY iteration_number ASC, id ASC
        """,
        (run_id,),
    )
    history: list[dict] = []
    max_iter_num = repro_iters
    total_kept = 0
    best_commit = None
    for row in rows:
        try:
            iteration_number = int(row.get("iteration_number") or 0)
        except (TypeError, ValueError):
            iteration_number = 0
        if iteration_number:
            max_iter_num = max(max_iter_num, iteration_number)
        status = row.get("status") or "unknown"
        if status == "keep":
            total_kept += 1
            if row.get("commit_hash"):
                best_commit = row.get("commit_hash")
        history.append(
            {
                "iteration": max(1, iteration_number - repro_iters) if iteration_number else len(history) + 1,
                "metric": row.get("metric_value"),
                "status": status,
                "description": _history_description_from_db(row.get("description"))[:500],
            }
        )
    return history, max_iter_num, total_kept, best_commit


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
        codex_failure = str(
            codex_result.get("error")
            or codex_result.get("stderr")
            or codex_result.get("returncode")
            or "unknown"
        )
        print(f"[LOOP] Codex iteration failed at iter {iteration}: {codex_failure[:500]}", flush=True)
        return {
            "description": f"Codex code generation failed: {codex_failure[:240]}",
            "artifact_paths": codex_result.get("artifact_paths", {}),
            "executor": "codex",
            "code_generation_failed": True,
        }

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


def _read_text_tail(path: Path, max_chars: int = 12_000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _safe_write_repo_file(code_dir: Path, rel_posix: str, content: str) -> bool:
    rel = Path(rel_posix)
    if rel.is_absolute() or any(p == ".." for p in rel.parts):
        return False
    target = (code_dir / rel).resolve()
    code_resolved = code_dir.resolve()
    if not str(target).startswith(str(code_resolved)):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError:
        return False
    return True


def _launch_reproduction_repair(
    *,
    workdir: Path,
    code_dir: Path,
    repair_round: int,
    baseline_command: str | None,
    metric_name: str,
    last_result: dict,
    environment_report: dict,
) -> dict:
    """Invoke Codex or LLM to fix baseline crashes / missing metrics before Phase 2."""
    log_path = Path(last_result.get("log_path") or workdir / "run.log")
    log_tail = _read_text_tail(log_path)
    err = str(last_result.get("error") or "") or str(last_result.get("status") or "crash")

    repair_log = workdir / "results" / f"repro_repair_{repair_round:02d}.json"

    if codex_executor.codex_available():
        print(f"[LOOP] Reproduction repair via Codex (round {repair_round})...", flush=True)
        out = codex_executor.run_codex_reproduction_repair(
            workdir=workdir,
            code_dir=code_dir,
            repair_round=repair_round,
            baseline_command=baseline_command,
            metric_name=metric_name,
            last_error=err,
            log_excerpt=log_tail,
            environment_report=environment_report,
        )
        summary = {
            "round": repair_round,
            "executor": "codex",
            "ok": bool(out.get("ok")),
            "summary": out.get("summary"),
            "files_changed": out.get("files_changed"),
            "artifact_paths": out.get("artifact_paths"),
        }
        repair_log.parent.mkdir(parents=True, exist_ok=True)
        repair_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    from agents.llm_client import call_llm_json

    print(f"[LOOP] Reproduction repair via LLM JSON patches (round {repair_round})...", flush=True)
    system = textwrap.dedent("""\
        You repair ML experiment code so a baseline command runs locally and prints a numeric metric.
        Return ONLY valid JSON with this shape:
        {"summary":"one line","files":[{"path":"relative/path.py","content":"FULL new file text"}]}
        Rules: at most 4 files; paths use forward slashes relative to repo root; no path segments ".." .
        Prefer fixing imports, device (CUDA->CPU), paths, and adding minimal smoke settings so a float metric is printed.
        The log must contain either a line like metric_value: 0.42 OR a FINAL_RESULTS: {...} JSON line.""")

    user = textwrap.dedent(f"""\
        metric_name preference: {metric_name}
        baseline_command: {baseline_command or "(python main train script)"}
        last status: {last_result.get("status")}
        error: {err[:4000]}
        log tail:
        {log_tail[:10000]}""")

    try:
        payload, _tokens = call_llm_json(system, user, temperature=0.0)
    except Exception as e:
        summary = {"round": repair_round, "executor": "llm_json", "ok": False, "error": str(e)}
        repair_log.parent.mkdir(parents=True, exist_ok=True)
        repair_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    files = payload.get("files") if isinstance(payload, dict) else None
    written: list[str] = []
    if isinstance(files, list):
        for item in files[:4]:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("path") or "").strip()
            content = item.get("content")
            if not rel or not isinstance(content, str) or len(content.strip()) < 5:
                continue
            if _safe_write_repo_file(code_dir, rel.replace("\\", "/"), content):
                written.append(rel)
    summary = {
        "round": repair_round,
        "executor": "llm_json",
        "ok": bool(written),
        "summary": (payload.get("summary") if isinstance(payload, dict) else None) or "llm patch",
        "files_written": written,
    }
    repair_log.parent.mkdir(parents=True, exist_ok=True)
    repair_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_full_benchmark_completion(run_id: int, execution_context: dict | None = None) -> dict:
    """Run the locked publication benchmark contract for an already forged run."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"run_id": run_id, "verdict": "blocked", "reason": "missing_run"}
    insight_id = run["deep_insight_id"]
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    run_layout = ensure_run_workspace(
        insight_id,
        run_id,
        insight=insight or {},
        suite=run.get("experiment_suite") or "main",
    )
    workdir = Path(run["workdir"]) if run.get("workdir") else Path(run_layout["run_root"])
    code_dir = workdir / "code"
    if not workdir.exists() or not code_dir.exists():
        error = f"Full benchmark blocked: workdir/code missing for run {run_id}."
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (error, run_id),
        )
        db.commit()
        return {"run_id": run_id, "verdict": "blocked", "reason": error}

    criteria = _read_success_criteria(workdir, insight_id)
    proxy = _read_proxy_config(workdir, insight_id)
    metric_name = criteria.get("metric_name", "metric")
    direction = criteria.get("metric_direction", "higher")
    baseline_command = proxy.get("baseline_command")
    baseline_command, forced_real_runner = _force_real_benchmark_command(proxy, code_dir, baseline_command)
    if forced_real_runner:
        proxy["baseline_command"] = baseline_command
        proxy["main_train_file"] = "train.py"
    try:
        time_budget = int(
            proxy.get("full_benchmark_time_budget_seconds")
            or proxy.get("full_benchmark_time_budget")
            or EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET
        )
    except (TypeError, ValueError):
        time_budget = EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET

    db.execute(
        "UPDATE experiment_runs SET status='running_gpu', phase='full_benchmark', started_at=COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id=?",
        (run_id,),
    )
    db.commit()
    result = _run_experiment(
        workdir,
        code_dir,
        time_budget,
        baseline_command=baseline_command,
        metric_name=metric_name,
        run_id=run_id,
        execution_context=execution_context,
        full_benchmark=True,
    )
    metric = result.get("metric")
    benchmark_summary = result.get("benchmark_summary") if isinstance(result.get("benchmark_summary"), dict) else {}
    results_dir = workdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if benchmark_summary:
        (results_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")

    next_iter = 1
    row = db.fetchone("SELECT MAX(iteration_number) AS n FROM experiment_iterations WHERE run_id=?", (run_id,))
    if row and row.get("n") is not None:
        try:
            next_iter = int(row["n"]) + 1
        except (TypeError, ValueError):
            next_iter = 1
    packet = ExperimentIterationPacket(
        run_id=run_id,
        iteration_number=next_iter,
        phase="full_benchmark",
        status=result.get("status", "ok"),
        description="publication full benchmark completion",
        metric_name=metric_name,
        metric_value=metric,
        baseline_value=run.get("baseline_metric_value"),
        best_value_before=run.get("best_metric_value"),
        best_value_after=metric if metric is not None else run.get("best_metric_value"),
        environment_report={
            "role": "FullBenchmarkCompletion",
            "formal_experiment": True,
            "full_benchmark": True,
            "benchmark_env": result.get("benchmark_env"),
        },
        judge_report={"role": "ExperimentJudge", "phase": "full_benchmark", "continue": True},
        execution_report=result,
        result_judgement={"role": "ResultJudge", "status": result.get("status", "ok")},
        artifact_paths={"log_path": result.get("log_path")},
    )
    _write_iteration_packet(workdir, packet, run_id)
    db.execute(
        """INSERT INTO experiment_iterations
           (run_id, iteration_number, phase, metric_value, metric_name,
            peak_memory_mb, duration_seconds, status, description)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            next_iter,
            "full_benchmark",
            metric,
            metric_name,
            result.get("peak_memory_mb"),
            result.get("duration"),
            result.get("status", "ok"),
            "publication full benchmark completion",
        ),
    )

    if result.get("status") != "ok" or not benchmark_summary:
        error = result.get("error") or result.get("failure_type") or "full benchmark did not produce benchmark_summary"
        db.execute(
            "UPDATE experiment_runs SET status='failed', phase='full_benchmark', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (str(error), run_id),
        )
        db.commit()
        return {"run_id": run_id, "verdict": "failed", "reason": str(error), "execution_report": result}

    _, _, candidate_value, baseline_value, _ = _benchmark_scores(benchmark_summary)
    baseline = float(baseline_value if baseline_value is not None else run.get("baseline_metric_value") or 0.0)
    best_value = float(candidate_value if candidate_value is not None else metric if metric is not None else run.get("best_metric_value") or baseline)
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
    effect_pct = (effect / abs(baseline) * 100) if baseline else 0.0
    summary_path = results_dir / "validation_summary.json"
    benchmark_artifact_path, full_benchmark_completed = _write_benchmark_artifact_manifest(
        workdir,
        run_id=run_id,
        metric_name=metric_name,
        benchmark_summary=benchmark_summary,
        criteria=criteria,
        verdict=verdict,
        validation_summary_path=summary_path,
    )
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "verdict": verdict,
                "baseline": baseline,
                "best_value": best_value,
                "effect_size": effect,
                "effect_pct": effect_pct,
                "benchmark_summary": benchmark_summary,
                "full_benchmark_completed": full_benchmark_completed,
                "benchmark_artifact_manifest": str(benchmark_artifact_path) if benchmark_artifact_path else "",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', phase='full_benchmark', hypothesis_verdict=?,
               baseline_metric_value=?, best_metric_value=?,
               effect_size=?, effect_pct=?, error_message=NULL,
               completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (verdict, baseline, best_value, effect, effect_pct, run_id),
    )
    db.commit()
    return {
        "run_id": run_id,
        "verdict": verdict,
        "baseline": baseline,
        "best_value": best_value,
        "effect_pct": effect_pct,
        "full_benchmark_completed": full_benchmark_completed,
        "benchmark_summary": benchmark_summary,
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

    gate = evosci_strict_gate_insight(dict(insight))
    if gate:
        err = gate.get("error", "EvoScientist strict gate blocked validation loop")
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (err, run_id),
        )
        db.commit()
        return {"run_id": run_id, "verdict": "blocked", "reason": err}

    run_layout = ensure_run_workspace(
        insight_id,
        run_id,
        insight=insight,
        suite=run.get("experiment_suite") or "main",
    )
    workdir = Path(run["workdir"]) if run.get("workdir") else Path(run_layout["run_root"])
    if not workdir.exists() and Path(run_layout["run_root"]).exists():
        workdir = Path(run_layout["run_root"])
        db.execute("UPDATE experiment_runs SET workdir=? WHERE id=?", (str(workdir), run_id))
        db.commit()
    code_dir = workdir / "code"

    if not workdir.exists():
        return {"error": f"Workdir {workdir} does not exist"}

    try:
        run_proxy = json.loads(run.get("proxy_config") or "{}")
    except (TypeError, json.JSONDecodeError):
        run_proxy = {}
    smoke_validation_allowed = bool(ALLOW_SMOKE_EXPERIMENT_VALIDATION and run_proxy.get("smoke_test_only"))
    if (run_proxy.get("formal_experiment") is False or run_proxy.get("smoke_test_only")) and not smoke_validation_allowed:
        error = "Non-formal/smoke-only experiment cannot enter the validation loop."
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (error, run_id),
        )
        db.commit()
        return {"run_id": run_id, "verdict": "blocked", "reason": "non_formal_experiment"}

    criteria = _read_success_criteria(workdir, insight_id)
    proxy = _read_proxy_config(workdir, insight_id)

    spec = _read_experiment_spec(run, insight, workdir, criteria=criteria, proxy=proxy)
    metric_name = criteria.get("metric_name", "metric")
    direction = criteria.get("metric_direction", "higher")
    try:
        time_budget = int(proxy.get("time_budget_seconds") or EXPERIMENT_TIME_BUDGET)
    except (TypeError, ValueError):
        time_budget = EXPERIMENT_TIME_BUDGET
    if proxy.get("real_benchmark_required") or proxy.get("benchmark_model"):
        try:
            real_budget = int(
                proxy.get("full_benchmark_time_budget_seconds")
                or proxy.get("full_benchmark_time_budget")
                or EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET
            )
        except (TypeError, ValueError):
            real_budget = EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET
        time_budget = max(time_budget, real_budget)
    baseline_command = proxy.get("baseline_command")
    baseline_command, forced_real_runner = _force_real_benchmark_command(proxy, code_dir, baseline_command)
    if forced_real_runner:
        proxy["baseline_command"] = baseline_command
        proxy["main_train_file"] = "train.py"
        try:
            (workdir / "spec" / "proxy_config.json").write_text(
                json.dumps(proxy, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        print("[LOOP] Real-benchmark runner detected; forcing baseline command to python train.py", flush=True)
    max_iters = proxy.get("max_iterations", EXPERIMENT_MAX_ITERATIONS)
    repro_iters = proxy.get("reproduction_iterations", EXPERIMENT_REPRODUCTION_ITERS)
    refute_min = proxy.get("refute_min_iterations", EXPERIMENT_REFUTE_MIN_ITERS)

    smoke_validation_allowed = bool(ALLOW_SMOKE_EXPERIMENT_VALIDATION and spec.smoke_test_only)
    if (not spec.formal_experiment or spec.smoke_test_only) and not smoke_validation_allowed:
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

    resume_validation = False
    resume_baseline: float | None = None
    resume_best: float | None = None
    resume_history: list[dict] = []
    resume_iter_num = repro_iters
    resume_total_kept = 0
    resume_best_commit: str | None = None
    if run.get("baseline_metric_value") is not None and (
        run.get("phase") == "hypothesis_testing" or run.get("status") in {"testing", "running_gpu"}
    ):
        try:
            resume_baseline = float(run.get("baseline_metric_value"))
            resume_best = float(run.get("best_metric_value")) if run.get("best_metric_value") is not None else resume_baseline
            resume_history, resume_iter_num, resume_total_kept, resume_best_commit = _resume_history_from_db(run_id, repro_iters)
            resume_validation = True
        except (TypeError, ValueError):
            resume_validation = False

    if resume_validation:
        db.execute(
            "UPDATE experiment_runs SET status='testing', phase='hypothesis_testing', started_at=COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id=?",
            (run_id,),
        )
        db.commit()
        promote_canonical_run(insight_id, run_id, insight=insight)
        write_latest_status(
            insight_id,
            {
                "stage": "hypothesis_testing",
                "status": "resuming",
                "workdir": str(workdir),
                "metric_name": metric_name,
                "baseline_metric_value": resume_baseline,
                "best_metric_value": resume_best,
                "iterations_total": resume_iter_num,
                "iterations_kept": resume_total_kept,
            },
            run_id=run_id,
            insight=insight,
        )
        print(
            f"[LOOP] Resuming hypothesis testing from iter {resume_iter_num} "
            f"(best={resume_best}, kept={resume_total_kept})...",
            flush=True,
        )
    else:
        db.execute("UPDATE experiment_runs SET status='reproducing', phase='reproduction', started_at=CURRENT_TIMESTAMP WHERE id=?", (run_id,))
        db.commit()
        promote_canonical_run(insight_id, run_id, insight=insight)
        write_latest_status(
            insight_id,
            {"stage": "reproduction", "status": "reproducing", "workdir": str(workdir), "metric_name": metric_name},
            run_id=run_id,
            insight=insight,
        )

    # ── Phase 1: Reproduction (with automatic repair rounds) ──
    print(f"[LOOP] Phase 1: Reproducing baseline ({repro_iters} attempts per round)...", flush=True)
    baseline_values = []
    benchmark_baseline_values: list[float] = []
    benchmark_candidate_values: list[float] = []
    benchmark_summary: dict = {}
    last_repro_result: dict = {"status": "pending", "error": "no attempt yet"}

    repair_round = 0
    while True:
        if resume_validation:
            baseline_values = [resume_baseline if resume_baseline is not None else 0.0]
            break
        baseline_values = []
        benchmark_baseline_values = []
        benchmark_candidate_values = []
        benchmark_summary = {}

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
                metric_name=metric_name,
                run_id=run_id,
                execution_context=execution_context,
            )
            last_repro_result = result
            metric = result.get("metric")
            iter_key = repair_round * max(repro_iters, 3) + i + 1
            packet = ExperimentIterationPacket(
                run_id=run_id,
                iteration_number=iter_key,
                phase="reproduction",
                status=result.get("status", "ok"),
                description=f"baseline run {i + 1} (repair_round={repair_round})",
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
                (run_id, iter_key, "reproduction", metric, metric_name,
                 result.get("peak_memory_mb"), result.get("duration"),
                 result.get("status", "ok"), json.dumps(packet.result_judgement)[:500])
            )
            db.commit()

            if metric is not None:
                baseline_values.append(metric)
                print(f"[LOOP] Reproduction {i+1}/{repro_iters}: {metric_name}={metric}", flush=True)
            else:
                print(f"[LOOP] Reproduction {i+1}/{repro_iters}: no metric (status={result.get('status')})", flush=True)

        if baseline_values:
            break

        if REPRODUCTION_REPAIR_MAX_ROUNDS <= 0:
            break
        if repair_round >= REPRODUCTION_REPAIR_MAX_ROUNDS:
            break

        _launch_reproduction_repair(
            workdir=workdir,
            code_dir=code_dir,
            repair_round=repair_round + 1,
            baseline_command=baseline_command,
            metric_name=metric_name,
            last_result=last_repro_result,
            environment_report=environment_report,
        )
        msg = f"repro auto-repair {repair_round + 1}"
        if _git_binary():
            _git_commit(code_dir, msg)
        repair_round += 1
        print(f"[LOOP] Phase 1: repair round {repair_round}/{REPRODUCTION_REPAIR_MAX_ROUNDS} applied; retrying baseline...", flush=True)

        environment_report = _run_environment_scout(spec, code_dir)
        env_path = workdir / "results" / "environment_report.json"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(json.dumps(environment_report, indent=2), encoding="utf-8")

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
    best_benchmark_summary = benchmark_summary if benchmark_mode else {}
    if resume_validation and resume_best is not None:
        best_value = resume_best
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
    if not benchmark_mode and not resume_validation:
        best_value = baseline
    best_commit = resume_best_commit or baseline_commit
    total_kept = resume_total_kept if resume_validation else 0
    iter_num = resume_iter_num if resume_validation else repro_iters
    effect_pct = 0.0
    history = list(resume_history) if resume_validation else []
    loop_start = time.time()
    stop_reason = ""
    completed_hypothesis_count = max(0, iter_num - repro_iters) if resume_validation else 0
    try:
        plateau_patience = int(proxy.get("plateau_patience_iterations", EXPERIMENT_PLATEAU_PATIENCE) or 0)
    except (TypeError, ValueError):
        plateau_patience = EXPERIMENT_PLATEAU_PATIENCE

    for i in range(completed_hypothesis_count, max_iters):
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
        diff = _git_diff(code_dir) if commit_hash else ""
        pre_benchmark_warnings = _blocked_pre_benchmark_diff_warnings(diff)
        if git_bin and not commit_hash:
            result = {
                "status": "blocked",
                "metric": None,
                "error": "No candidate code diff was produced.",
                "duration": 0.0,
                "peak_memory_mb": None,
            }
            metric = None
            iteration_benchmark_summary = {}
            result_judgement = {
                "role": "ResultJudge",
                "status": "discard",
                "summary": "No candidate code diff was produced, so the iteration was not benchmarked.",
                "anomaly_type": "no_candidate_diff",
                "continue": True,
                "terminate": False,
                "stop_reason": "",
                "metric": None,
                "baseline": baseline,
                "benchmark_semantic_warnings": [],
                "paper_evidence_warning": False,
            }
            status = "discard"
            fairness_warnings = []
        elif pre_benchmark_warnings:
            result = {
                "status": "blocked",
                "metric": None,
                "error": pre_benchmark_warnings[0],
                "duration": 0.0,
                "peak_memory_mb": None,
                "pre_benchmark_guard_warnings": pre_benchmark_warnings,
            }
            metric = None
            iteration_benchmark_summary = {}
            result_judgement = {
                "role": "ResultJudge",
                "status": "discard",
                "summary": (
                    "Pre-benchmark guard blocked this candidate before GPU evaluation: "
                    + pre_benchmark_warnings[0]
                ),
                "anomaly_type": "pre_benchmark_guard",
                "continue": True,
                "terminate": False,
                "stop_reason": "",
                "metric": None,
                "baseline": baseline,
                "benchmark_semantic_warnings": pre_benchmark_warnings,
                "paper_evidence_warning": True,
            }
            status = "discard"
            fairness_warnings = []
        else:
            result = _run_experiment(
                workdir,
                code_dir,
                time_budget,
                baseline_command=baseline_command,
                metric_name=metric_name,
                run_id=run_id,
                execution_context=execution_context,
            )
            metric = result.get("metric")
            iteration_benchmark_summary = result.get("benchmark_summary") if isinstance(result.get("benchmark_summary"), dict) else {}

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
            status, fairness_warnings = _apply_benchmark_fairness_guard(
                status=status,
                result_judgement=result_judgement,
                diff=diff,
            )
        if status == "keep":
            best_value = metric if metric is not None else best_value
            if iteration_benchmark_summary:
                best_benchmark_summary = iteration_benchmark_summary
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
             result.get("duration"), status, _iteration_db_description(
                 result_judgement=result_judgement,
                 coding_summary=desc,
                 executor=coding_step.get("executor"),
             ))
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
        write_latest_status(
            insight_id,
            {
                "stage": "hypothesis_testing",
                "status": "testing",
                "iteration": iter_num,
                "hypothesis_iteration": i + 1,
                "iterations_total": iter_num,
                "iterations_kept": total_kept,
                "last_iteration_status": status,
                "last_metric_value": metric,
                "baseline_metric_value": baseline,
                "best_metric_value": best_value,
                "effect_pct": effect_pct,
                "peak_memory_mb": result.get("peak_memory_mb"),
                "coding_executor": coding_step.get("executor"),
                "coding_summary": desc[:300],
                "supervisor_mode": (supervisor_plan or {}).get("mode"),
            },
            run_id=run_id,
            insight=insight,
        )

        if (i + 1) % 5 == 0:
            print(f"[LOOP] Iter {i+1}/{max_iters}: best={best_value:.6f} "
                  f"(baseline={baseline:.6f}, kept={total_kept})", flush=True)

        # Check termination conditions
        if result_judgement.get("stop_reason"):
            stop_reason = result_judgement["stop_reason"]
        if result_judgement.get("terminate"):
            print(f"[LOOP] Judge terminated loop at iter {i+1}: {stop_reason}", flush=True)
            break
        if plateau_patience > 0 and len(history) >= max(refute_min, plateau_patience):
            recent = history[-plateau_patience:]
            if recent and all(row.get("status") != "keep" for row in recent):
                stop_reason = f"No kept improvement in the last {plateau_patience} iterations."
                print(f"[LOOP] Plateau stop at iter {i+1}: {stop_reason}", flush=True)
                break

    # ── Determine verdict ──
    final_benchmark_summary = best_benchmark_summary or benchmark_summary
    verdict = _determine_final_verdict(
        baseline=baseline,
        best_value=best_value,
        direction=direction,
        criteria=criteria,
        total_iters=len(history),
        total_kept=total_kept,
        refute_min=refute_min,
        benchmark_summary=final_benchmark_summary if benchmark_mode else None,
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
    benchmark_artifact_path, full_benchmark_completed = _write_benchmark_artifact_manifest(
        workdir,
        run_id=run_id,
        metric_name=metric_name,
        benchmark_summary=final_benchmark_summary,
        criteria=criteria,
        verdict=verdict,
        validation_summary_path=summary_path,
    )
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
                "benchmark_summary": final_benchmark_summary,
                "full_benchmark_completed": full_benchmark_completed,
                "benchmark_artifact_manifest": str(benchmark_artifact_path) if benchmark_artifact_path else "",
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
    if benchmark_artifact_path:
        _record_artifact(
            run_id,
            "source_data",
            benchmark_artifact_path,
            metric_key=metric_name,
            metric_value=best_value,
            metadata={
                "contract_type": "BenchmarkArtifactManifest",
                "full_benchmark_completed": full_benchmark_completed,
            },
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
