"""Experiment execution: local subprocess runner with optional streaming output."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    EXPERIMENT_REAL_BENCHMARK_DATASET,
    EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG,
    EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
    EXPERIMENT_REAL_BENCHMARK_SEEDS,
    EXPERIMENT_REAL_LLM_MODEL,
    EXPERIMENT_VALIDATION_BENCHMARK_MAX_EXAMPLES,
    EXPERIMENT_VALIDATION_BENCHMARK_METHODS,
    EXPERIMENT_VALIDATION_BENCHMARK_SEEDS,
    RUNTIME_PYTHON,
)
from agents.metric_parser import (
    _FLOAT_RE,
    benchmark_scores,
    parse_benchmark_summary_from_log,
    parse_metric_from_log,
)


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run."""
    status: str  # "ok", "crash"
    metric: float | None = None
    duration: float = 0.0
    peak_memory_mb: float | None = None
    error: str | None = None
    failure_type: str | None = None
    returncode: int | None = None
    command_tokens: list[str] = field(default_factory=list)
    log_path: str = ""
    benchmark_summary: dict = field(default_factory=dict)
    benchmark_metric_name: str | None = None
    benchmark_candidate_method: str | None = None
    benchmark_baseline_metric: float | None = None
    benchmark_num_seeds: int = 0
    visible_device: str | None = None
    worker_id: str | None = None
    backend: str = "local"
    benchmark_env: dict = field(default_factory=dict)
    last_benchmark_stage: str | None = None
    benchmark_stage_trace: list[str] = field(default_factory=list)
    final_results_present: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None or k in ("metric", "error")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_text(*values: str | None) -> str:
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return ""


def _int_text(value, default: int = 0) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(default)


def _load_workdir_json(workdir: Path, filename: str) -> dict:
    for candidate in (workdir / "spec" / filename, workdir / filename):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int_text(value: str | None, cap: int) -> str:
    if cap <= 0:
        return str(value or "")
    try:
        current = int(str(value or "").strip())
    except (TypeError, ValueError):
        current = cap
    if current <= 0:
        current = cap
    return str(min(current, cap))


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _benchmark_stage_trace(text: str, *, limit: int = 16) -> list[str]:
    stages: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("BENCHMARK_STAGE:"):
            stages.append(line[:300])
    return stages[-limit:]


def _workdir_uses_cggr_runner(workdir: Path) -> bool:
    train_py = workdir / "code" / "train.py"
    try:
        text = train_py.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
    except OSError:
        return False
    return '"cggr_mode": true' in text or "'cggr_mode': true" in text


def _command_looks_placeholder(command: str | None) -> bool:
    text = str(command or "").strip().lower()
    if not text:
        return False
    if re.search(r"<[^>]+>", text):
        return True
    return any(
        marker in text
        for marker in (
            "hf_or_vllm_model",
            "your_model",
            "path_to_",
            "todo",
            "unknown",
        )
    )


# ---------------------------------------------------------------------------
# Public API: command / file helpers
# ---------------------------------------------------------------------------

def normalize_command_tokens(command: str | None, python_bin: str) -> list[str]:
    """Parse command string into tokens, replacing python/python3 with *python_bin*."""
    if not command:
        return []
    if _command_looks_placeholder(command):
        return []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    if tokens and tokens[0] in {"python", "python3"}:
        tokens[0] = python_bin
    return tokens


def command_entrypoint_exists(tokens: list[str], code_dir: Path) -> bool:
    """Check whether the entrypoint script exists in *code_dir*."""
    if not tokens:
        return False
    candidates = tokens[1:] if Path(tokens[0]).name.lower().startswith("python") else tokens[:1]
    for token in candidates:
        if token.startswith("-"):
            continue
        if token.endswith(".py"):
            return (code_dir / token).exists()
        break
    return True


def find_train_file(code_dir: Path, preferred: str | None = None) -> Path | None:
    """Find the main training script in *code_dir*."""
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


# ---------------------------------------------------------------------------
# Benchmark environment
# ---------------------------------------------------------------------------

def benchmark_env_from_workdir(local_workdir: Path) -> dict[str, str]:
    """Build benchmark environment vars from workdir proxy/spec contracts.

    Moved from the deleted ``ssh_gpu_backend`` module -- no SSH dependency.
    """
    env: dict[str, str] = {}
    if EXPERIMENT_REAL_LLM_MODEL:
        env["DEEPGRAPH_BENCHMARK_MODEL"] = EXPERIMENT_REAL_LLM_MODEL
    if EXPERIMENT_REAL_BENCHMARK_DATASET:
        env["DEEPGRAPH_BENCHMARK_DATASET"] = EXPERIMENT_REAL_BENCHMARK_DATASET
    if EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG:
        env["DEEPGRAPH_BENCHMARK_DATASET_CONFIG"] = EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG
    env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"] = _int_text(EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES, 64)
    env["DEEPGRAPH_BENCHMARK_SEEDS"] = _int_text(EXPERIMENT_REAL_BENCHMARK_SEEDS, 3)

    proxy = _load_workdir_json(local_workdir, "proxy_config.json")
    manifest = proxy.get("benchmark_manifest") if isinstance(proxy.get("benchmark_manifest"), dict) else {}
    full_stage = proxy.get("full_benchmark_stage") if isinstance(proxy.get("full_benchmark_stage"), dict) else {}
    sanity = proxy.get("sanity_stage") if isinstance(proxy.get("sanity_stage"), dict) else {}
    contract = proxy.get("publication_evidence_contract") if isinstance(proxy.get("publication_evidence_contract"), dict) else {}

    model = _first_text(
        manifest.get("model"),
        full_stage.get("model"),
        sanity.get("model"),
        contract.get("model"),
    )
    if model:
        env["DEEPGRAPH_BENCHMARK_MODEL"] = model
    dataset = _first_text(
        manifest.get("dataset"),
        full_stage.get("dataset"),
        sanity.get("dataset"),
        contract.get("dataset"),
    )
    if dataset:
        env["DEEPGRAPH_BENCHMARK_DATASET"] = dataset
    dataset_config = _first_text(
        manifest.get("dataset_config"),
        full_stage.get("dataset_config"),
        sanity.get("dataset_config"),
        contract.get("dataset_config"),
    )
    if dataset_config:
        env["DEEPGRAPH_BENCHMARK_DATASET_CONFIG"] = dataset_config
    max_examples = _first_text(
        str(manifest.get("max_examples") or ""),
        str(full_stage.get("max_examples") or ""),
        str(contract.get("max_examples") or ""),
    )
    if max_examples:
        env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"] = max_examples
    seeds = _first_text(
        str(manifest.get("seeds") or ""),
        str(full_stage.get("seeds") or ""),
        str(contract.get("minimum_seeds") or ""),
    )
    if seeds:
        env["DEEPGRAPH_BENCHMARK_SEEDS"] = seeds
    return env


def benchmark_env_for_execution(workdir: Path, *, full_benchmark: bool = False) -> dict[str, str]:
    """Benchmark env with validation caps applied."""
    env = benchmark_env_from_workdir(workdir)
    if full_benchmark or _env_truthy("DEEPGRAPH_BENCHMARK_FULL_RUN"):
        env["DEEPGRAPH_BENCHMARK_FULL_RUN"] = "1"
        env.setdefault("DEEPGRAPH_BENCHMARK_METHODS", "all")
        return env

    env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"] = _bounded_int_text(
        env.get("DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"),
        EXPERIMENT_VALIDATION_BENCHMARK_MAX_EXAMPLES,
    )
    env["DEEPGRAPH_BENCHMARK_SEEDS"] = _bounded_int_text(
        env.get("DEEPGRAPH_BENCHMARK_SEEDS"),
        EXPERIMENT_VALIDATION_BENCHMARK_SEEDS,
    )
    env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES_CAP"] = str(EXPERIMENT_VALIDATION_BENCHMARK_MAX_EXAMPLES)
    env["DEEPGRAPH_BENCHMARK_SEEDS_CAP"] = str(EXPERIMENT_VALIDATION_BENCHMARK_SEEDS)
    if EXPERIMENT_VALIDATION_BENCHMARK_METHODS and _workdir_uses_cggr_runner(workdir):
        env["DEEPGRAPH_BENCHMARK_METHODS"] = EXPERIMENT_VALIDATION_BENCHMARK_METHODS
    return env


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def worker_visible_device(worker: dict | None) -> str | None:
    """Extract visible_device from worker metadata."""
    if not worker:
        return None
    metadata: dict = {}
    raw_metadata = worker.get("metadata")
    if raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        if isinstance(parsed, dict):
            metadata = parsed
    visible_device = metadata.get("visible_device")
    if visible_device is None:
        visible_device = worker.get("gpu_index")
    if visible_device is None or str(visible_device).strip() == "":
        return None
    return str(visible_device)


# ---------------------------------------------------------------------------
# Execution diagnostics
# ---------------------------------------------------------------------------

def execution_diagnostics(
    *,
    returncode: int | None,
    log_text: str,
    stderr: str = "",
    duration: float | None = None,
    time_budget: int | None = None,
    metric: float | None = None,
) -> dict:
    """Categorize an experiment failure."""
    text = "\n".join(part for part in (log_text, stderr) if part)
    trace = _benchmark_stage_trace(text)
    last_stage = trace[-1] if trace else None
    lower = text.lower()
    failure_type = None
    if returncode == 124 or "timed out" in lower or (duration and time_budget and duration >= time_budget):
        failure_type = "timeout"
    elif "cuda out of memory" in lower or "cublas_status_alloc_failed" in lower:
        failure_type = "cuda_oom"
    elif returncode in (-9, 137) or re.search(r"\bkilled\b", lower):
        failure_type = "oom_or_sigkill"
    elif "no space left on device" in lower:
        failure_type = "disk_full"
    elif "modulenotfounderror" in lower or "no module named" in lower:
        failure_type = "missing_dependency"
    elif returncode not in (None, 0):
        if last_stage and "model_ready" in last_stage:
            failure_type = "post_model_execution_crash"
        else:
            failure_type = "nonzero_exit"
    elif metric is None and "FINAL_RESULTS:" not in text:
        failure_type = "missing_final_results"
    elif metric is None:
        failure_type = "missing_metric"

    out: dict = {
        "failure_type": failure_type,
        "last_benchmark_stage": last_stage,
        "benchmark_stage_trace": trace,
        "final_results_present": "FINAL_RESULTS:" in text,
    }
    if returncode is not None:
        out["returncode"] = returncode
    return out


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    workdir: Path,
    code_dir: Path,
    time_budget: int,
    *,
    baseline_command: str | None = None,
    metric_name: str = "metric",
    run_id: int | None = None,
    execution_context: dict | None = None,
    full_benchmark: bool = False,
    stream_output: bool = False,
) -> dict:
    """Run a single experiment iteration and return a result dict.

    When *stream_output* is True, use ``Popen`` to stream stdout to the
    console in real-time (useful for interactive debugging).
    """
    from agents.workspace_layout import plan_file_path
    from db import database as db

    log_path = workdir / "run.log"
    eval_candidates: list[Path] = []
    if run_id is not None:
        row = db.fetchone("SELECT deep_insight_id FROM experiment_runs WHERE id=?", (run_id,))
        if row and row.get("deep_insight_id") is not None:
            eval_candidates.append(plan_file_path(int(row["deep_insight_id"]), "evaluate.py"))
    eval_candidates.extend((workdir / "spec" / "evaluate.py", workdir / "evaluate.py"))
    eval_path = next((p for p in eval_candidates if p.exists()), workdir / "spec" / "evaluate.py")

    python_bin = RUNTIME_PYTHON or sys.executable
    command_tokens = normalize_command_tokens(baseline_command, python_bin)
    if command_tokens and not command_entrypoint_exists(command_tokens, code_dir):
        command_tokens = []
    if not command_tokens:
        train_file = find_train_file(code_dir)
        train_script = str(train_file.relative_to(code_dir)) if train_file else "train.py"
        command_tokens = [python_bin, train_script]

    start = time.time()
    worker = (execution_context or {}).get("worker") if execution_context else None
    benchmark_env = benchmark_env_for_execution(workdir, full_benchmark=full_benchmark)

    try:
        local_env = os.environ.copy()
        local_env.update(benchmark_env)
        visible_device = worker_visible_device(worker)
        if visible_device is not None:
            local_env["CUDA_VISIBLE_DEVICES"] = visible_device

        if stream_output:
            proc = subprocess.Popen(
                command_tokens,
                cwd=str(code_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=local_env,
                text=True,
            )
            with open(log_path, "w", encoding="utf-8") as log_file:
                for line in proc.stdout:  # type: ignore[union-attr]
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
            remaining = max(60, time_budget - int(time.time() - start))
            proc.wait(timeout=remaining)
            returncode = proc.returncode
            stdout = _safe_read_text(log_path)
            stderr_text = ""
        else:
            result = subprocess.run(
                command_tokens,
                cwd=str(code_dir),
                timeout=time_budget + 60,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=local_env,
            )
            returncode = result.returncode
            stdout = result.stdout or ""
            stderr_text = result.stderr or ""
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(stdout)
                if stderr_text:
                    f.write("\n--- STDERR ---\n")
                    f.write(stderr_text)

        duration = time.time() - start

        if returncode != 0:
            log_text = stdout + (("\n--- STDERR ---\n" + stderr_text) if stderr_text else "")
            diagnostics = execution_diagnostics(
                returncode=returncode,
                log_text=log_text,
                stderr=stderr_text,
                duration=duration,
                time_budget=time_budget,
            )
            return {
                "status": "crash",
                "duration": duration,
                "error": stderr_text[-500:] if stderr_text else "nonzero exit",
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
            "returncode": returncode,
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

    # --- Parse metrics ---
    metric = None
    benchmark_summary = parse_benchmark_summary_from_log(log_path)
    if benchmark_summary:
        benchmark_metric_name, benchmark_candidate_method, benchmark_candidate_value, benchmark_baseline_value, benchmark_num_seeds = benchmark_scores(benchmark_summary)
    else:
        benchmark_metric_name, benchmark_candidate_method, benchmark_candidate_value, benchmark_baseline_value, benchmark_num_seeds = "metric", None, None, None, 0
    if benchmark_candidate_value is not None:
        metric = benchmark_candidate_value
    if metric is None:
        metric = parse_metric_from_log(log_path, metric_name)

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
    diagnostics = execution_diagnostics(
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
