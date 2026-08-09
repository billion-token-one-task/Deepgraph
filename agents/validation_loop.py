"""Validation Loop: hypothesis-directed Karpathy-style experiment engine.

Two-phase loop:
  Phase 1 (Reproduction): run baseline as-is, record ground truth metric
  Phase 2 (Hypothesis Testing): implement proposed method, iterate, keep/discard

Key difference from autoresearch:
  - NOT open-ended optimization; directed by a specific method definition
  - Knows when to stop: hypothesis SUPPORTED, REFUTED, or TIMEOUT
  - Logs structured iteration data for the Result Interpreter
"""
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.benchmark_audit import (
    benchmark_diagnostic_notes,
    benchmark_fairness_warnings_from_diff,
    benchmark_semantic_warnings,
    full_benchmark_evidence_blockers,
)
from agents import codex_executor
from agents import experiment_feedback
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
    EXPERIMENT_VALIDATION_BENCHMARK_MAX_EXAMPLES,
    EXPERIMENT_VALIDATION_BENCHMARK_METHODS,
    EXPERIMENT_VALIDATION_BENCHMARK_SEEDS,
    REPRODUCTION_REPAIR_MAX_ROUNDS,
    RUNTIME_PYTHON,
)
from db import database as db
from meta_harness.runner_contract import (
    RunnerContractError,
    validate_final_results,
    verify_metric_from_artifacts,
)
from meta_harness.failure_policy import FailureContext, classify_failure, decide_recovery
from meta_harness.failure_repository import FailureRecoveryRepository
from orchestrator import ssh_gpu_backend


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TELEMETRY_RESULT_KEYS = {
    "peak_vram_mb",
    "peak_memory_mb",
    "reserved_vram_gb",
    "target_vram_gb",
    "cuda_device",
    "device",
    "method",
}
_AUTOMATION_FAILURE_ANOMALIES = {
    "no_candidate_diff",
    "pre_benchmark_guard",
    "benchmark_mismatch_or_redesign_required",
    "implementation_drift",
}
_REDESIGN_REQUIRED_FILENAMES = (
    "EXPERIMENT_REDESIGN_REQUIRED.json",
    "IMPLEMENTATION_REDESIGN_REQUIRED.json",
)
try:
    AUTOMATION_FAILURE_PATIENCE = max(1, int(os.environ.get("DEEPGRAPH_AUTOMATION_FAILURE_PATIENCE", "3")))
except ValueError:
    AUTOMATION_FAILURE_PATIENCE = 3


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

    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        payload = None
        if line.startswith("FINAL_RESULTS:"):
            _, _, json_text = line.partition(":")
            try:
                payload = json.loads(json_text.strip())
            except (json.JSONDecodeError, TypeError):
                payload = None
        elif line.startswith("{"):
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                payload = None
        if isinstance(payload, dict):
            for key in (metric_name, "metric_value"):
                if not key:
                    continue
                raw_value = payload.get(key)
                try:
                    return float(raw_value)
                except (TypeError, ValueError):
                    pass
            numeric_items = []
            for key, raw_value in payload.items():
                if str(key).lower() in _TELEMETRY_RESULT_KEYS:
                    continue
                try:
                    numeric_items.append(float(raw_value))
                except (TypeError, ValueError):
                    continue
            if len(numeric_items) == 1:
                return numeric_items[0]

    patterns = [
        rf'"?{re.escape(metric_name)}"?\s*[:=]\s*({_FLOAT_RE})' if metric_name else None,
        rf'"metric_value"\s*:\s*({_FLOAT_RE})',
        rf'metric_value[:\s]+({_FLOAT_RE})',
        rf'val_bpb[:\s]+({_FLOAT_RE})',
        rf'accuracy[:\s]+({_FLOAT_RE})',
        rf'mAP[:\s]+({_FLOAT_RE})',
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
    candidate_method = str(summary.get("candidate_method") or "").strip() or None

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


def _process_timeout_seconds(time_budget: int | None, *, full_benchmark: bool) -> int | None:
    """Return subprocess timeout seconds; 0/None means unlimited for full benchmarks."""

    try:
        budget = int(time_budget or 0)
    except (TypeError, ValueError):
        budget = 0
    if full_benchmark and budget <= 0:
        return None
    if budget <= 0:
        budget = EXPERIMENT_TIME_BUDGET
    return max(1, budget) + 60


def _full_benchmark_time_budget(proxy: dict) -> int:
    """Full benchmark completion defaults to unlimited unless explicitly capped."""

    raw = proxy.get("full_benchmark_time_budget_seconds")
    if raw in (None, ""):
        raw = proxy.get("full_benchmark_time_budget")
    if raw in (None, "", "none", "None", "unlimited", "Unlimited"):
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _execution_diagnostics(
    *,
    returncode: int | None,
    log_text: str,
    stderr: str = "",
    duration: float | None = None,
    time_budget: int | None = None,
    metric: float | None = None,
) -> dict:
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

    out = {
        "failure_type": failure_type,
        "last_benchmark_stage": last_stage,
        "benchmark_stage_trace": trace,
        "final_results_present": "FINAL_RESULTS:" in text,
    }
    if failure_type:
        out["reason_code"] = (
            "metric_missing"
            if failure_type in {"missing_final_results", "missing_metric"}
            else classify_failure(
                message=" ".join([failure_type, text]),
                returncode=returncode,
                final_results_present="FINAL_RESULTS:" in text,
            )
        )
    if returncode is not None:
        out["returncode"] = returncode
    return out


def _execution_recovery_decision(
    run_id: int,
    result: dict,
    *,
    retry_count: int,
):
    try:
        return FailureRecoveryRepository().decide_for_run(
            experiment_run_id=run_id,
            execution_result=result,
            retry_count=retry_count,
        )[0]
    except Exception:
        db.rollback()
        reason = str(result.get("reason_code") or "") or classify_failure(
            message=" ".join(
                [
                    str(result.get("error") or ""),
                    str(result.get("failure_type") or ""),
                ]
            ),
            returncode=result.get("returncode"),
            final_results_present=bool(result.get("final_results_present")),
        )
        remaining = _remaining_grant_gpu_seconds(run_id) or 0.0
        return decide_recovery(
            FailureContext(
                reason_code=reason,
                detail=str(result.get("error") or reason),
                code_hash="unavailable",
                environment_hash="unavailable",
                remaining_gpu_seconds=remaining,
                retry_count=retry_count,
            ),
            fingerprint_seen=False,
        )


def _apply_runner_recovery_adjustments(code_dir: Path, adjustments: dict) -> bool:
    config_path = code_dir / "execution_requirements.json"
    if not config_path.is_file() or not adjustments:
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    runtime = dict(config.get("runtime_adjustments") or {})
    runtime.update(adjustments)
    config["runtime_adjustments"] = runtime
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


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
    policy_blockers = full_benchmark_evidence_blockers(summary, criteria)
    return bool(per_method and len(per_method) >= 2 and num_seeds >= minimum_seeds and not policy_blockers)


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
    ("oracleroutingupperbound", "oracle", "oraclerouter", "upperbound"),
    ("candidate", "proposedmethod", "methodundertest"),
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
    if verdict != "supported":
        blockers.append(f"execution verdict is {verdict!r}, not supported")
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
    blockers.extend(f"full benchmark policy: {item}" for item in full_benchmark_evidence_blockers(summary, criteria))

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


def _command_entrypoint_exists(tokens: list[str], code_dir: Path) -> bool:
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


def _looks_like_generated_real_benchmark_runner(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    required = (
        "final_results:",
        "full_benchmark_completed",
        "automodelforcausallm",
        "load_dataset",
    )
    return all(marker in text for marker in required)


def _force_real_benchmark_command(proxy: dict, code_dir: Path, baseline_command: str | None) -> tuple[str | None, bool]:
    if not proxy.get("real_benchmark_required"):
        return baseline_command, False
    train_py = code_dir / "train.py"
    if not _looks_like_generated_real_benchmark_runner(train_py):
        return baseline_command, False
    return "python train.py", baseline_command != "python train.py"


def _local_worker_visible_device(worker: dict | None) -> str | None:
    if not worker or ssh_gpu_backend.is_ssh_worker(worker):
        return None
    metadata = {}
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


def _contract_context(workdir: Path) -> tuple[dict, dict, dict]:
    success = _read_success_criteria(workdir)
    proxy = _read_proxy_config(workdir)
    contract = success.get("publication_evidence_contract") if isinstance(success.get("publication_evidence_contract"), dict) else {}
    if not contract and isinstance(proxy.get("publication_evidence_contract"), dict):
        contract = proxy.get("publication_evidence_contract") or {}
    quality_gates = success.get("quality_gates") if isinstance(success.get("quality_gates"), dict) else {}
    if not quality_gates and isinstance(contract.get("quality_gates"), dict):
        quality_gates = contract.get("quality_gates") or {}
    return success, proxy, contract | {"quality_gates": quality_gates}


def _formal_benchmark_required(workdir: Path, *, full_benchmark: bool = False) -> bool:
    if full_benchmark:
        return True
    success, proxy, contract = _contract_context(workdir)
    quality_gates = contract.get("quality_gates") if isinstance(contract.get("quality_gates"), dict) else {}
    evidence_tier = str(contract.get("evidence_tier") or success.get("evidence_tier") or "")
    return bool(
        proxy.get("real_benchmark_required")
        or evidence_tier in {"benchmark_plan", "sanity_real_benchmark"}
        or quality_gates.get("requires_full_benchmark_package")
        or contract.get("required_real_benchmarks")
    )


def _command_script_path(command_tokens: list[str], code_dir: Path) -> Path:
    candidates = command_tokens[1:] if command_tokens and Path(command_tokens[0]).name.lower().startswith("python") else command_tokens[:1]
    for token in candidates:
        if token.startswith("-"):
            continue
        if token.endswith(".py"):
            return code_dir / token
        break
    return code_dir / "train.py"


def _runner_contract_violations(
    workdir: Path,
    code_dir: Path,
    command_tokens: list[str],
    *,
    full_benchmark: bool = False,
    execution_context: dict | None = None,
) -> list[str]:
    if not _formal_benchmark_required(workdir, full_benchmark=full_benchmark):
        return []
    script_path = _command_script_path(command_tokens, code_dir)
    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [f"formal benchmark entrypoint is missing: {script_path}"]
    lowered = text.lower()
    violations: list[str] = []

    literal_patterns = {
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*true\b": "Python code uses JSON boolean true in a keyword/assignment; use True.",
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*false\b": "Python code uses JSON boolean false in a keyword/assignment; use False.",
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*null\b": "Python code uses JSON null in an assignment; use None.",
        r"\breturn\s+null\b": "Python code uses JSON null; use None.",
        r"\bis\s+not\s+null\b": "Python code compares with null; use None.",
        r"\bis\s+null\b": "Python code compares with null; use None.",
        r":\s*null\b": "Python dict literal contains null; use None.",
    }
    for pattern, message in literal_patterns.items():
        if re.search(pattern, text):
            violations.append(message)

    proxy_markers = (
        "smoke baseline",
        "smoke-only",
        "synthetic",
        "simulated",
        "toy",
        "dummy",
        "random.randn",
        "torch.randn",
        "np.random",
        "cpu-only probe",
        "probe evidence",
    )
    real_markers = (
        "load_dataset",
        "from_pretrained",
        "default_local_jsonl",
        "raw_predictions",
        "benchmark_summary",
        "per_seed_results",
        "per_dataset_results",
    )
    if any(marker in lowered for marker in proxy_markers) and not any(marker in lowered for marker in real_markers):
        violations.append("formal benchmark runner appears to be a smoke/toy/synthetic proxy without real dataset/model loading.")

    success, proxy, contract = _contract_context(workdir)
    quality_gates = contract.get("quality_gates") if isinstance(contract.get("quality_gates"), dict) else {}
    benchmark_plan = bool(
        full_benchmark
        or str(contract.get("evidence_tier") or success.get("evidence_tier") or "") == "benchmark_plan"
        or quality_gates.get("requires_full_benchmark_package")
    )
    if benchmark_plan:
        required_output_markers = ("final_results:", "per_method", "candidate_method", "full_benchmark_completed")
        missing = [marker for marker in required_output_markers if marker not in lowered]
        if missing:
            violations.append("formal benchmark runner is missing required output contract markers: " + ", ".join(missing))

    job = (execution_context or {}).get("job") if execution_context else None
    worker = (execution_context or {}).get("worker") if execution_context else None
    resource_class = str((job or {}).get("resource_class") or "").lower()
    gpu_routed = bool(worker) or resource_class.startswith("gpu")
    real_benchmark_required = bool(
        proxy.get("real_benchmark_required")
        or proxy.get("benchmark_model")
        or proxy.get("benchmark_dataset")
        or contract.get("required_real_benchmarks")
    )
    if gpu_routed and real_benchmark_required:
        cpu_proxy_markers = (
            "sklearn.datasets",
            "load_digits",
            "load_breast_cancer",
            "load_iris",
            "logisticregression",
            "make_classification",
            "random.random",
            "random.seed",
            "local smoke",
            "smoke test",
            "proxy baseline",
        )
        if any(marker in lowered for marker in cpu_proxy_markers) and not any(marker in lowered for marker in real_markers):
            violations.append("gpu_large real benchmark runner is a CPU/proxy script; it must load the contracted model and dataset instead of sklearn/random/local smoke data.")

        dataset = str(proxy.get("benchmark_dataset") or "").strip()
        model = str(proxy.get("benchmark_model") or "").strip()
        if dataset:
            dataset_terms = {dataset.lower(), dataset.split("/")[-1].lower()}
            if not any(term and term in lowered for term in dataset_terms) and "load_dataset" not in lowered:
                violations.append(f"gpu_large real benchmark runner does not reference required benchmark dataset {dataset!r}.")
        if model:
            model_terms = {model.lower(), model.split("/")[-1].lower()}
            if not any(term and term in lowered for term in model_terms) and "from_pretrained" not in lowered:
                violations.append(f"gpu_large real benchmark runner does not reference required benchmark model {model!r}.")
        if "cuda" not in lowered and "torch.device" not in lowered:
            violations.append("gpu_large real benchmark runner has no CUDA/device handling, so it would not exercise the assigned GPU.")

    return violations



def _repair_generated_runner_json_literals(
    workdir: Path,
    code_dir: Path,
    command_tokens: list[str],
) -> list[str]:
    """Fix common LLM JSON-literal leaks in generated Python runners."""
    script_path = _command_script_path(command_tokens, code_dir)
    try:
        original = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    repaired = original
    replacements = (
        (r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*true\b", r"\1=True"),
        (r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*false\b", r"\1=False"),
        (r"->\s*null\b", "-> None"),
        (r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*null\b", r"\1=None"),
        (r"\breturn\s+null\b", "return None"),
        (r"\bis\s+not\s+null\b", "is not None"),
        (r"\bis\s+null\b", "is None"),
        (r":\s*null\b", ": None"),
        (r":\s*true\b", ": True"),
        (r":\s*false\b", ": False"),
    )
    applied: list[str] = []
    for pattern, replacement in replacements:
        repaired, count = re.subn(pattern, replacement, repaired)
        if count:
            applied.append(pattern)
    if repaired == original:
        return []

    try:
        compile(repaired, str(script_path), "exec")
    except SyntaxError:
        return []

    try:
        script_path.write_text(repaired, encoding="utf-8")
        repair_log = workdir / "results" / "generated_runner_literal_repair.json"
        repair_log.parent.mkdir(parents=True, exist_ok=True)
        repair_log.write_text(
            json.dumps(
                {
                    "status": "repaired",
                    "script": str(script_path),
                    "patterns": applied,
                    "reason": "Converted JSON-style true/false/null tokens to Python literals before execution.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        return []
    return applied


def _apply_worker_vram_env(
    benchmark_env: dict[str, str],
    execution_context: dict | None,
) -> dict[str, str]:
    env = dict(benchmark_env)
    worker = (execution_context or {}).get("worker") if execution_context else None
    job = (execution_context or {}).get("job") if execution_context else None
    try:
        worker_vram = float((worker or {}).get("total_mem_gb") or 0)
    except (TypeError, ValueError):
        worker_vram = 0.0
    try:
        requested_vram = float((job or {}).get("vram_required_gb") or 0)
    except (TypeError, ValueError):
        requested_vram = 0.0
    if worker_vram > 0:
        env.setdefault("DEEPGRAPH_GPU_WORKER_VRAM_GB", f"{worker_vram:.2f}")
        env.setdefault("DEEPGRAPH_BENCHMARK_TARGET_VRAM_GB", f"{max(1.0, min(worker_vram, requested_vram or worker_vram)):.2f}")
        env.setdefault("DEEPGRAPH_BENCHMARK_BATCH_SIZE", "1")
        env.setdefault("DEEPGRAPH_BENCHMARK_MICRO_BATCH_SIZE", "1")
    return env


def _benchmark_env_for_execution(workdir: Path, *, full_benchmark: bool = False) -> dict[str, str]:
    env = ssh_gpu_backend.benchmark_env_from_workdir(workdir)
    if full_benchmark or _env_truthy("DEEPGRAPH_BENCHMARK_FULL_RUN"):
        env["DEEPGRAPH_BENCHMARK_FULL_RUN"] = "1"
        env.setdefault("DEEPGRAPH_BENCHMARK_METHODS", "all")
        env.setdefault("DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES", "1")
        return env

    env.setdefault("DEEPGRAPH_BENCHMARK_MAX_EXAMPLES", str(EXPERIMENT_VALIDATION_BENCHMARK_MAX_EXAMPLES))
    env.setdefault("DEEPGRAPH_BENCHMARK_SEEDS", str(EXPERIMENT_VALIDATION_BENCHMARK_SEEDS))
    env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES_CAP"] = str(EXPERIMENT_VALIDATION_BENCHMARK_MAX_EXAMPLES)
    env["DEEPGRAPH_BENCHMARK_SEEDS_CAP"] = str(EXPERIMENT_VALIDATION_BENCHMARK_SEEDS)
    if EXPERIMENT_VALIDATION_BENCHMARK_METHODS:
        env["DEEPGRAPH_BENCHMARK_METHODS"] = EXPERIMENT_VALIDATION_BENCHMARK_METHODS
    return env




def _benchmark_model_list_from_env(env: dict[str, str]) -> list[str]:
    raw = str(env.get("DEEPGRAPH_BENCHMARK_MODELS") or "").strip()
    models: list[str] = []
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.split(",")]
        if isinstance(parsed, list):
            for item in parsed:
                text = str(item.get("hf_model") or item.get("model") or item.get("id") or item.get("name") if isinstance(item, dict) else item or "").strip()
                if text and text not in models:
                    models.append(text)
    fallback = str(env.get("DEEPGRAPH_BENCHMARK_MODEL") or "").strip()
    if fallback and fallback not in models:
        models.insert(0, fallback)
    return models


def _safe_model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_id or "model")).strip("_") or "model"


def _clean_benchmark_result_files(results_dir: Path) -> None:
    names = (
        "run_config.json",
        "raw_predictions.jsonl",
        "routing_decisions.jsonl",
        "per_seed_results.json",
        "per_dataset_results.json",
        "main_results_table.json",
        "cost_utility_tradeoff_table.json",
        "quality_cost_frontier.json",
        "route_rate_sweep_table.json",
        "ablation_table.json",
        "difficulty_breakdown_table.json",
        "routing_analysis.json",
        "latency_tokens_table.json",
        "simple_case_degradation.json",
        "calibration_reliability.json",
        "bootstrap_ci.json",
        "failure_cases.jsonl",
        "artifact_manifest.json",
        "environment_report.json",
        "benchmark_summary.json",
    )
    for name in names:
        path = results_dir / name
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            pass


def _metric_from_row(row: dict, metric_name: str | None = None) -> float | None:
    if not isinstance(row, dict):
        return None
    for key in (metric_name, "metric_value", "cost_adjusted_accuracy", "score"):
        if not key:
            continue
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            continue
    return None


def _upper_bound_name(name: str, row: dict | None = None) -> bool:
    label = str(name or "").replace("-", "_").replace(" ", "_").lower()
    return bool(isinstance(row, dict) and row.get("upper_bound")) or "oracle" in label or "upper_bound" in label


def _weighted_merge_method_rows(rows: list[dict]) -> dict:
    merged: dict = {}
    total_count = 0
    for row in rows:
        try:
            total_count += int(row.get("count") or 0)
        except (TypeError, ValueError):
            pass
    total_count = max(1, total_count)
    numeric_avg_keys = (
        "score",
        "exact",
        "f1",
        "avg_new_tokens",
        "avg_latency_seconds",
        "route_rate",
        "cost_adjusted_accuracy",
        "metric_value",
    )
    for key in numeric_avg_keys:
        value = 0.0
        seen = False
        for row in rows:
            try:
                count = int(row.get("count") or 0)
                value += float(row.get(key)) * max(1, count)
                seen = True
            except (TypeError, ValueError):
                continue
        if seen:
            merged[key] = float(value / total_count)
    merged["count"] = total_count
    if any(row.get("upper_bound") for row in rows if isinstance(row, dict)):
        merged["upper_bound"] = True
    return merged


def _paired_permutation_pvalue(candidate: list[float], baseline: list[float]) -> float | None:
    pairs = [(float(c), float(b)) for c, b in zip(candidate, baseline)]
    if not pairs:
        return None
    observed = abs(sum(c - b for c, b in pairs) / len(pairs))
    count = 0
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(pairs)):
        diff = abs(sum(sign * (c - b) for sign, (c, b) in zip(signs, pairs)) / len(pairs))
        count += 1
        if diff >= observed - 1e-12:
            extreme += 1
    return float(extreme / max(1, count))


def _bootstrap_ci(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    import random
    rng = random.Random(12345)
    means = []
    for _ in range(2000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / max(1, len(sample)))
    means.sort()
    return [float(means[int(0.025 * (len(means) - 1))]), float(means[int(0.975 * (len(means) - 1))])]


def _merge_model_benchmark_summaries(model_results: list[dict], criteria_metric_name: str) -> dict:
    summaries = [row.get("benchmark_summary") for row in model_results if isinstance(row.get("benchmark_summary"), dict)]
    if not summaries:
        return {}
    base = json.loads(json.dumps(summaries[0]))
    metric_name = str(base.get("primary_metric") or base.get("metric_name") or criteria_metric_name or "metric_value")
    candidate = str(base.get("candidate_method") or "").strip()
    all_methods = sorted({name for summary in summaries for name in (summary.get("per_method") or {}).keys()})
    per_method: dict = {}
    for method in all_methods:
        rows = [summary.get("per_method", {}).get(method) for summary in summaries]
        rows = [row for row in rows if isinstance(row, dict)]
        if rows:
            per_method[method] = _weighted_merge_method_rows(rows)
    seed_results = []
    for model_result, summary in zip(model_results, summaries):
        model = summary.get("model") if isinstance(summary.get("model"), dict) else {}
        model_id = model_result.get("model_id") or model.get("id") or ""
        for seed_row in summary.get("seed_results") or []:
            if isinstance(seed_row, dict):
                copied = json.loads(json.dumps(seed_row))
                copied["model_id"] = model_id
                seed_results.append(copied)
    datasets = []
    seen_datasets = set()
    for summary in summaries:
        for row in summary.get("datasets") or []:
            if not isinstance(row, dict):
                continue
            key = str(row.get("name") or row.get("id") or row).lower()
            if key and key not in seen_datasets:
                seen_datasets.add(key)
                datasets.append(row)
    models = []
    seen_models = set()
    for model_result, summary in zip(model_results, summaries):
        model = summary.get("model") if isinstance(summary.get("model"), dict) else {}
        model_id = str(model_result.get("model_id") or model.get("id") or "").strip()
        if model_id and model_id.lower() not in seen_models:
            seen_models.add(model_id.lower())
            models.append({**model, "id": model_id})
    strongest_name = ""
    strongest_value = None
    higher = True
    for name, row in per_method.items():
        if name == candidate or _upper_bound_name(name, row):
            continue
        value = _metric_from_row(row, metric_name)
        if value is None:
            continue
        if strongest_value is None or (value > strongest_value if higher else value < strongest_value):
            strongest_name, strongest_value = name, value
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    if candidate and strongest_name:
        for seed_row in seed_results:
            methods = seed_row.get("methods") if isinstance(seed_row, dict) else {}
            if not isinstance(methods, dict):
                continue
            cand = _metric_from_row(methods.get(candidate, {}), metric_name)
            base_value = _metric_from_row(methods.get(strongest_name, {}), metric_name)
            if cand is not None and base_value is not None:
                candidate_values.append(cand)
                baseline_values.append(base_value)
    p_value = _paired_permutation_pvalue(candidate_values, baseline_values)
    bootstrap = {
        "candidate_method": candidate,
        "baseline_method": strongest_name,
        "candidate_ci95": _bootstrap_ci(candidate_values),
        "baseline_ci95": _bootstrap_ci(baseline_values),
        "paired_permutation_p": p_value,
        "p_value": p_value,
    }
    for key in (
        "ablation_table",
        "ablation_results",
        "cost_utility_tradeoff_table",
        "quality_cost_frontier",
        "route_rate_sweep",
        "difficulty_breakdown_table",
        "latency_tokens_table",
        "calibration_reliability",
    ):
        rows = []
        for model_result, summary in zip(model_results, summaries):
            model = summary.get("model") if isinstance(summary.get("model"), dict) else {}
            model_id = model_result.get("model_id") or model.get("id") or ""
            for row in summary.get(key) or []:
                if isinstance(row, dict):
                    rows.append({"model_id": model_id, **row})
        if rows:
            base[key] = rows
    base["per_method"] = per_method
    base["seed_results"] = seed_results
    base["num_seeds"] = max(int(summary.get("num_seeds") or 0) for summary in summaries)
    base["datasets"] = datasets
    base["dataset"] = datasets[0] if datasets else {}
    base["models"] = models
    base["model"] = {"ids": [row.get("id") for row in models], "backend": "transformers", "cuda": True}
    base["load_failures"] = [failure for summary in summaries for failure in (summary.get("load_failures") or [])]
    base["full_benchmark_completed"] = bool(all(summary.get("full_benchmark_completed") for summary in summaries) and not base["load_failures"])
    base["duration_seconds"] = sum(float(summary.get("duration_seconds") or 0.0) for summary in summaries)
    base["peak_vram_mb"] = max(float(summary.get("peak_vram_mb") or 0.0) for summary in summaries)
    base["hardware"] = ", ".join(sorted({str(summary.get("hardware") or "") for summary in summaries if summary.get("hardware")}))
    base["bootstrap_ci"] = bootstrap
    if candidate and candidate in per_method:
        base[metric_name] = _metric_from_row(per_method[candidate], metric_name) or 0.0
    return base


def _run_experiment_model_matrix(
    workdir: Path,
    code_dir: Path,
    time_budget: int,
    *,
    baseline_command: str | None,
    metric_name: str,
    run_id: int | None,
    execution_context: dict | None,
    full_benchmark: bool,
    benchmark_env: dict[str, str],
    command_tokens: list[str],
) -> dict:
    models = _benchmark_model_list_from_env(benchmark_env)
    results_dir = workdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_model_results: list[dict] = []
    started = time.time()
    per_model_budget = time_budget if time_budget and time_budget > 0 else 0
    for model_id in models:
        model_env = dict(benchmark_env)
        model_env.pop("DEEPGRAPH_BENCHMARK_MODELS", None)
        model_env["DEEPGRAPH_BENCHMARK_MODEL"] = model_id
        _clean_benchmark_result_files(results_dir)
        result = _run_experiment(
            workdir,
            code_dir,
            per_model_budget,
            baseline_command=baseline_command,
            metric_name=metric_name,
            run_id=run_id,
            execution_context=execution_context,
            full_benchmark=full_benchmark,
            benchmark_env_override=model_env,
            _disable_model_matrix=True,
        )
        result["model_id"] = model_id
        safe = _safe_model_slug(model_id)
        try:
            if (workdir / "run.log").exists():
                shutil.copy2(workdir / "run.log", workdir / f"run.{safe}.log")
        except OSError:
            pass
        model_results_dir = results_dir / f"model_{safe}"
        try:
            if model_results_dir.exists():
                shutil.rmtree(model_results_dir)
            model_results_dir.mkdir(parents=True, exist_ok=True)
            for item in results_dir.iterdir():
                if item == model_results_dir or item.name.startswith("model_"):
                    continue
                if item.is_file():
                    shutil.copy2(item, model_results_dir / item.name)
        except OSError:
            pass
        per_model_results.append(result)
        if result.get("status") != "ok":
            merged_error = result.get("error") or result.get("failure_type") or f"model {model_id} failed"
            return {
                **result,
                "status": "crash",
                "duration": time.time() - started,
                "error": merged_error,
                "benchmark_env": benchmark_env,
                "per_model_results": per_model_results,
                "command_tokens": command_tokens,
            }
    merged = _merge_model_benchmark_summaries(per_model_results, metric_name)
    (results_dir / "benchmark_summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    metric = None
    candidate = str(merged.get("candidate_method") or "")
    if candidate:
        metric = _metric_from_row((merged.get("per_method") or {}).get(candidate, {}), str(merged.get("primary_metric") or merged.get("metric_name") or metric_name))
    log_path = workdir / "run.log"
    log_path.write_text(
        "MODEL_MATRIX_RESULTS: " + json.dumps({"models": models, "per_model_status": [r.get("status") for r in per_model_results]}, ensure_ascii=False) + "\n"
        + "FINAL_RESULTS: " + json.dumps(merged, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok" if metric is not None else "crash",
        "metric": metric,
        "duration": time.time() - started,
        "peak_memory_mb": max(float(r.get("peak_memory_mb") or 0.0) for r in per_model_results),
        "command_tokens": command_tokens,
        "log_path": str(log_path),
        "benchmark_summary": merged,
        "benchmark_metric_name": merged.get("primary_metric") or merged.get("metric_name"),
        "benchmark_candidate_method": merged.get("candidate_method"),
        "benchmark_baseline_metric": None,
        "benchmark_num_seeds": merged.get("num_seeds"),
        "benchmark_env": benchmark_env,
        "per_model_results": per_model_results,
        "backend": per_model_results[-1].get("backend") if per_model_results else None,
        "worker_id": per_model_results[-1].get("worker_id") if per_model_results else None,
        "visible_device": per_model_results[-1].get("visible_device") if per_model_results else None,
        "final_results_present": bool(merged),
    }

def _as_utc_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _remaining_grant_gpu_seconds(run_id: int) -> float | None:
    """Read this run's allowance from the canonical attempt control plane."""
    attempt = db.fetchone(
        """
        SELECT id FROM experiment_attempt_gpu_reservations_v1
        WHERE experiment_run_id=? AND status IN ('reserved','running')
        ORDER BY id DESC LIMIT 1
        """,
        (int(run_id),),
    )
    if not attempt:
        return None
    from meta_harness.attempt_gpu_usage import GrantGPUUsageControl

    return GrantGPUUsageControl().remaining_attempt_wall_seconds(
        int(attempt["id"])
    )


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
    benchmark_env_override: dict[str, str] | None = None,
    _disable_model_matrix: bool = False,
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
    benchmark_env = dict(benchmark_env_override) if benchmark_env_override is not None else _benchmark_env_for_execution(workdir, full_benchmark=full_benchmark)
    benchmark_env = _apply_worker_vram_env(benchmark_env, execution_context)
    literal_repairs = _repair_generated_runner_json_literals(workdir, code_dir, command_tokens)
    if literal_repairs:
        print(
            f"[LOOP] Repaired generated runner Python literals before execution: {len(literal_repairs)} pattern(s)",
            flush=True,
        )
    contract_violations = _runner_contract_violations(
        workdir,
        code_dir,
        command_tokens,
        full_benchmark=full_benchmark,
        execution_context=execution_context,
    )
    if contract_violations:
        duration = time.time() - start
        error = "; ".join(contract_violations)
        try:
            log_path.write_text(
                "Contract-preserving runner guard blocked execution.\n"
                f"Command: {command_tokens}\n"
                + "\n".join(f"- {item}" for item in contract_violations)
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return {
            "status": "crash",
            "duration": duration,
            "error": error,
            "failure_type": "contract_violation",
            "final_results_present": False,
            "command_tokens": command_tokens,
            "log_path": str(log_path),
            "benchmark_env": benchmark_env,
        }
    if full_benchmark and not _disable_model_matrix and len(_benchmark_model_list_from_env(benchmark_env)) > 1:
        return _run_experiment_model_matrix(
            workdir,
            code_dir,
            time_budget,
            baseline_command=baseline_command,
            metric_name=metric_name,
            run_id=run_id,
            execution_context=execution_context,
            full_benchmark=full_benchmark,
            benchmark_env=benchmark_env,
            command_tokens=command_tokens,
        )
    try:
        if run_id is not None and ssh_gpu_backend.is_ssh_worker(worker):
            remaining_gpu_seconds = _remaining_grant_gpu_seconds(run_id)
            if remaining_gpu_seconds is not None and remaining_gpu_seconds <= 0:
                return {
                    "status": "crash",
                    "duration": time.time() - start,
                    "error": "resource grant cumulative GPU-hour cap is exhausted",
                    "failure_type": "grant_gpu_hours_exhausted",
                    "final_results_present": False,
                    "command_tokens": command_tokens,
                    "log_path": str(log_path),
                    "backend": "ssh",
                    "benchmark_env": benchmark_env,
                }
            remote_time_budget = time_budget
            if remaining_gpu_seconds is not None:
                bounded_seconds = max(1, int(remaining_gpu_seconds))
                if remote_time_budget <= 0 or remote_time_budget > bounded_seconds:
                    remote_time_budget = bounded_seconds
            # _run_experiment performs metadata reads while preparing the remote
            # command.  With psycopg those reads open a transaction; leaving it
            # open across a model download/GPU run lets the session-side idle
            # transaction guard close the connection before settlement.  End
            # the read-only transaction before crossing the long remote boundary.
            db.commit()
            remote = ssh_gpu_backend.run_remote_experiment(
                worker=worker,
                run_id=run_id,
                local_workdir=workdir,
                local_code_dir=code_dir,
                time_budget=remote_time_budget,
                command_tokens=command_tokens,
                local_python=python_bin,
                benchmark_env=benchmark_env,
            )
            stdout = remote.get("stdout") or ""
            stderr = remote.get("stderr") or ""
            duration = time.time() - start
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(stdout)
                if stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(stderr)
            returncode = int(remote.get("returncode") or 0)
            if returncode != 0:
                log_text = stdout + (("\n--- STDERR ---\n" + stderr) if stderr else "")
                diagnostics = _execution_diagnostics(
                    returncode=returncode,
                    log_text=log_text,
                    stderr=stderr,
                    duration=duration,
                    time_budget=time_budget,
                )
                error = stderr[-500:] if stderr else stdout[-500:] if stdout else "nonzero exit"
                return {
                    "status": "crash",
                    "duration": duration,
                    "error": error,
                    **diagnostics,
                    "command_tokens": command_tokens,
                    "log_path": str(log_path),
                    "backend": "ssh",
                    "remote_host": remote.get("remote_host"),
                    "worker_id": remote.get("worker_id"),
                    "benchmark_env": benchmark_env,
                }
            execution_meta = {
                "backend": "ssh",
                "remote_host": remote.get("remote_host"),
                "worker_id": remote.get("worker_id"),
                "visible_device": remote.get("visible_device"),
                "returncode": returncode,
                "benchmark_env": benchmark_env,
            }
        else:
            local_env = os.environ.copy()
            local_env.update(benchmark_env)
            visible_device = _local_worker_visible_device(worker)
            if visible_device is not None:
                local_env["CUDA_VISIBLE_DEVICES"] = visible_device
            stderr = ""
            with open(log_path, "w", encoding="utf-8") as f:
                proc = subprocess.Popen(
                    command_tokens,
                    cwd=str(code_dir),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=local_env,
                    bufsize=1,
                )
                try:
                    proc.wait(timeout=_process_timeout_seconds(time_budget, full_benchmark=full_benchmark))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    raise
            duration = time.time() - start
            stdout = _safe_read_text(log_path)
            returncode = int(proc.returncode or 0)

            if returncode != 0:
                log_text = stdout
                diagnostics = _execution_diagnostics(
                    returncode=returncode,
                    log_text=log_text,
                    stderr=stderr,
                    duration=duration,
                    time_budget=time_budget,
                )
                return {
                    "status": "crash",
                    "duration": duration,
                    "error": stdout[-500:] if stdout else "nonzero exit",
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

    metric = None
    benchmark_summary = _parse_benchmark_summary_from_log(log_path)
    if benchmark_summary.get("schema_version") == "final_results_v1":
        try:
            _record_contract_artifacts(run_id, workdir, benchmark_summary)
        except RunnerContractError as exc:
            return {
                "status": "crash",
                "metric": None,
                "duration": duration,
                "error": str(exc),
                "failure_type": exc.reason_code,
                "final_results_present": True,
                "command_tokens": command_tokens,
                "log_path": str(log_path),
                "benchmark_summary": benchmark_summary,
                **execution_meta,
            }
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
            [
                git_bin,
                "-c",
                "user.name=DeepGraph Auto",
                "-c",
                "user.email=deepgraph-auto@local",
                "commit",
                "-m",
                message,
            ],
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


def _history_anomaly(row: dict) -> str:
    judgement = row.get("result_judgement") if isinstance(row.get("result_judgement"), dict) else {}
    return str(judgement.get("anomaly_type") or "").strip()


def _recent_automation_failure_streak(history: list[dict]) -> int:
    streak = 0
    for row in reversed(history):
        if row.get("metric") is not None or row.get("status") == "keep":
            break
        anomaly = _history_anomaly(row)
        if anomaly in _AUTOMATION_FAILURE_ANOMALIES:
            streak += 1
        else:
            break
    return streak


def _hypothesis_testing_automation_failed(history: list[dict]) -> bool:
    if not history:
        return False
    benchmarked = [row for row in history if row.get("metric") is not None]
    kept = [row for row in history if row.get("status") == "keep"]
    if benchmarked or kept:
        return False
    anomalies = []
    for row in history:
        anomaly = _history_anomaly(row)
        if anomaly:
            anomalies.append(anomaly)
    if not anomalies:
        return True
    return all(anomaly in _AUTOMATION_FAILURE_ANOMALIES for anomaly in anomalies)


def _write_automation_failure_artifact(
    workdir: Path,
    *,
    run_id: int,
    insight_id: int,
    history: list[dict],
    stop_reason: str,
    method_desc: str,
) -> Path:
    path = workdir / "results" / "automation_failure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    recent = history[-min(len(history), 12):]
    payload = {
        "contract_type": "HypothesisTestingAutomationFailure",
        "run_id": run_id,
        "deep_insight_id": insight_id,
        "failure_type": "no_benchmarked_candidate_method_change",
        "stop_reason": stop_reason,
        "method_excerpt": method_desc[:1200],
        "automation_anomalies": [_history_anomaly(row) for row in history if _history_anomaly(row)],
        "recent_history": recent,
        "not_scientific_verdict": True,
        "recommended_actions": [
            "Route the idea back to experiment repair/reforge instead of marking it refuted.",
            "Inspect why the proposed method was not operationalized into a source/config diff.",
            "Add or repair the benchmark harness hook that exposes the method path to the runner.",
            "If the method definition is underspecified, rewrite the idea/experimental plan before another validation loop.",
            "Do not draft a formal manuscript claim from this run until a full benchmarked candidate change exists.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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
    automation_failed: bool = False,
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

    if automation_failed:
        return "inconclusive"

    summary = benchmark_summary or {}
    metric_name, candidate_method, candidate_value, best_other, num_seeds = (
        _benchmark_scores(summary) if summary else ("", "", None, None, 0)
    )
    if summary and candidate_value is not None and best_other is not None:
        best_value = float(candidate_value)
        baseline = float(best_other)
        effect = best_value - baseline if direction == "higher" else baseline - best_value
        is_improvement = effect > 0

    if total_iters >= refute_min and not is_improvement:
        return "refuted"
    if not is_improvement:
        return "inconclusive"

    p_value = None
    for container in (
        summary,
        summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {},
        summary.get("significance") if isinstance(summary.get("significance"), dict) else {},
    ):
        for key in ("p_value", "paired_permutation_p"):
            try:
                if container.get(key) is not None:
                    p_value = float(container[key])
                    break
            except (TypeError, ValueError):
                continue
        if p_value is not None:
            break

    from contracts.scientific_evidence import EvidenceDecisionInput, decide_evidence

    evidence = decide_evidence(
        EvidenceDecisionInput(
            verdict="supported",
            p_value=p_value,
            metric_value=candidate_value if candidate_value is not None else best_value,
            baseline_value=best_other if best_other is not None else baseline,
            full_benchmark_complete=bool(
                summary.get("full_benchmark_completed") is True
                and candidate_method
                and num_seeds >= 3
            ),
            raw_artifacts_complete=bool(
                summary.get("raw_artifacts_complete")
                or summary.get("benchmark_artifact_manifest")
            ),
            claim_ledger_complete=bool(summary.get("claim_ledger_complete")),
            evaluator_id=str(summary.get("evaluator_id") or ""),
        )
    )
    # Execution may report support, but only the canonical evidence repository
    # can create a scientific decision record. Point estimates, pilot results,
    # missing p-values, zero baselines, and incomplete artifacts remain
    # inconclusive even if they clear a hand-written performance threshold.
    return "supported" if evidence.confirmation_allowed else "inconclusive"


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
    run = db.fetchone("SELECT agenda_id FROM experiment_runs WHERE id=?", (run_id,))
    agenda_id = int((run or {}).get("agenda_id") or 0)
    if agenda_id <= 0:
        raise RuntimeError("artifact write requires an agenda-scoped experiment run")
    existing = db.fetchone(
        """
        SELECT id FROM experiment_artifacts
        WHERE agenda_id=? AND run_id=? AND artifact_type=? AND path=?
        ORDER BY id LIMIT 1
        """,
        (agenda_id, run_id, artifact_type, str(path)),
    )
    if existing:
        return
    db.execute(
        """
        INSERT INTO experiment_artifacts
            (agenda_id, run_id, artifact_type, path, metric_key, metric_value, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agenda_id,
            run_id,
            artifact_type,
            str(path),
            metric_key,
            metric_value,
            json.dumps(metadata or {}),
        ),
    )


def _record_contract_artifacts(
    run_id: int | None,
    workdir: Path,
    log_summary: dict,
) -> None:
    """Verify raw evidence and register every logical runner artifact."""

    if run_id is None:
        raise RunnerContractError("runner_contract_violation", "run_id_missing")
    results_dir = (workdir / "results").resolve()
    final_path = results_dir / "final_results.json"
    if not final_path.is_file():
        raise RunnerContractError("artifact_contract_violation", "final_results")
    payload = validate_final_results(
        json.loads(final_path.read_text(encoding="utf-8"))
    )
    for key in (
        "dataset_revision",
        "model_revision",
        "baseline_method",
        "candidate_method",
        "metric_name",
        "metric_direction",
        "baseline_metric_value",
        "best_metric_value",
    ):
        if payload.get(key) != log_summary.get(key):
            raise RunnerContractError("runner_contract_violation", f"log_file_{key}")
    verify_metric_from_artifacts(final_path)
    artifacts = payload["artifacts"]
    hashes = payload["artifact_hashes"]
    for artifact_type, ref in artifacts.items():
        if not isinstance(ref, dict) or not str(ref.get("path") or "").strip():
            raise RunnerContractError(
                "artifact_contract_violation", str(artifact_type)
            )
        path = (results_dir / str(ref["path"])).resolve()
        if path != results_dir and results_dir not in path.parents:
            raise RunnerContractError("artifact_contract_violation", "path_escape")
        if not path.is_file():
            raise RunnerContractError(
                "artifact_contract_violation", str(artifact_type)
            )
        expected = str(hashes.get(artifact_type) or "")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RunnerContractError("artifact_hash_mismatch", str(artifact_type))
        _record_artifact(
            int(run_id),
            str(artifact_type),
            path,
            metric_key=str(payload["metric_name"]),
            metric_value=(
                float(payload["best_metric_value"])
                if artifact_type == "final_results"
                else None
            ),
            metadata={
                "contract_type": "RunnerArtifact",
                "sha256": expected or hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
    db.commit()


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
    """Reject candidate changes to evaluator, holdout, budget, or safety policy."""
    if not diff:
        return []

    added = _added_diff_text(diff).lower()
    full = str(diff or "").lower()
    warnings: list[str] = []

    protected_markers = (
        "held_out",
        "holdout",
        "evaluator",
        "resource_grant",
        "budget_policy",
        "safety_policy",
        "production_results",
    )
    if any(marker in full for marker in protected_markers):
        warnings.append(
            "Pre-benchmark guard blocked a candidate diff touching evaluator, "
            "holdout, budget, safety, grant, or production-result policy."
        )
    if "completed" in added and any(marker in added for marker in ("except", "error", "failed", "timeout")):
        warnings.append(
            "Pre-benchmark guard blocked a candidate diff that may convert "
            "failure or timeout paths into completed."
        )
    if any(
        marker in added
        for marker in (
            "benchmark-provided context",
            "benchmark provided context",
            "_context_to_text",
            ".get(\"context\"",
            ".get('context'",
        )
    ):
        warnings.append(
            "Pre-benchmark guard blocked broad-context propagation into the "
            "candidate prompt or scoring path."
        )
    if any(
        marker in added
        for marker in (
            "zero-budget",
            "zero_budget",
            "reasoning_budget",
            "answer with only",
            "phrase-only",
        )
    ):
        warnings.append(
            "Pre-benchmark guard blocked a zero-budget answer-shape or "
            "reasoning-budget change."
        )

    return warnings


def _read_redesign_required_artifact(code_dir: Path, workdir: Path) -> dict | None:
    """Return an explicit mechanism/benchmark mismatch artifact from the worker.

    A missing repo module is implementable and should not be marked unsupported.
    This artifact is for the narrower case where the locked benchmark/harness
    cannot exercise the proposed mechanism, so the run needs reforge or harness
    work rather than more hypothesis iterations.
    """
    for root in (code_dir, workdir):
        for filename in _REDESIGN_REQUIRED_FILENAMES:
            path = root / filename
            if not path.exists() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {"raw_text": _safe_read_text(path)[:4000]}
            if not isinstance(payload, dict):
                payload = {"payload": payload}
            payload.setdefault("artifact_path", str(path))
            payload.setdefault("not_scientific_verdict", True)
            payload.setdefault("recommended_route", "reforge_or_benchmark_harness")
            return payload
    return None


def _persist_redesign_required_artifact(workdir: Path, iteration: int, payload: dict) -> str:
    results_dir = workdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"redesign_required_iter_{iteration:03d}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


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


def _history_judgement_from_db(value: object) -> dict:
    text = str(value or "").strip()
    if not text or not text.startswith("{"):
        return {}
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
        result_judgement = _history_judgement_from_db(row.get("description"))
        history.append(
            {
                "iteration": max(1, iteration_number - repro_iters) if iteration_number else len(history) + 1,
                "metric": row.get("metric_value"),
                "status": status,
                "description": _history_description_from_db(row.get("description"))[:500],
                "result_judgement": result_judgement,
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
    from agents.llm_client import (
        call_llm_for_role,
        configured_role_prompt_version,
    )

    recent_history = history[-10:] if history else []
    history_text = ""
    for h in recent_history:
        status_marker = "KEPT" if h.get("status") == "keep" else "DISCARDED"
        history_text += f"  Iter {h.get('iteration', '?')}: {h.get('description', '?')} -> {h.get('metric', '?')} [{status_marker}]\n"

    proxy = _read_proxy_config(workdir)
    success_criteria = success_criteria or {}
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
        run_id = int(spec.run_id or 0) if spec is not None else 0
        run_scope = db.fetchone(
            """
            SELECT er.agenda_id, er.deep_insight_id, er.resource_grant_id,
                   rg.stage
            FROM experiment_runs AS er
            JOIN resource_grants AS rg
              ON rg.id=er.resource_grant_id
             AND rg.agenda_id=er.agenda_id
             AND rg.idea_id=er.deep_insight_id
             AND rg.status='active'
             AND rg.expires_at > CURRENT_TIMESTAMP
            WHERE er.id=?
            """,
            (run_id,),
        )
        if not run_scope:
            raise PermissionError(
                "role-routed coding requires an active scoped ResourceGrant"
            )
        idempotency_material = "|".join(
            (
                str(run_scope["agenda_id"]),
                str(run_scope["deep_insight_id"]),
                str(run_id),
                str(iteration),
                method_desc,
                current_code,
                history_text,
            )
        )
        new_code, _, route = call_llm_for_role(
            system,
            prompt,
            agenda_id=int(run_scope["agenda_id"]),
            idea_id=int(run_scope["deep_insight_id"]),
            role="proposer",
            stage=str(run_scope["stage"]),
            resource_grant_id=int(run_scope["resource_grant_id"]),
            operation="validation_code_iteration",
            idempotency_key=(
                "validation-code:"
                + hashlib.sha256(
                    idempotency_material.encode("utf-8")
                ).hexdigest()
            ),
            prompt_version=configured_role_prompt_version("proposer"),
            max_tokens=16000,
        )
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
                "artifact_paths": {"llm_route": route},
                "executor": "role_routed_llm",
            }

        if train_file and len(new_code) > 50:
            train_file.write_text(new_code, encoding="utf-8")
            return {
                "description": f"Modified {train_file.name} (iter {iteration})",
                "artifact_paths": {"llm_route": route},
                "executor": "role_routed_llm",
            }
    except Exception as e:
        return {
            "description": f"LLM code generation failed: {e}",
            "artifact_paths": {},
            "executor": "role_routed_llm",
            "code_generation_failed": True,
        }

    return {
        "description": f"No modification applied (iter {iteration})",
        "artifact_paths": {},
        "executor": "role_routed_llm",
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
    run_id: int,
    workdir: Path,
    code_dir: Path,
    repair_round: int,
    baseline_command: str | None,
    metric_name: str,
    last_result: dict,
    environment_report: dict,
) -> dict:
    """Use the resource-granted proposer route to repair reproduction failures."""
    log_path = Path(last_result.get("log_path") or workdir / "run.log")
    log_tail = _read_text_tail(log_path)
    err = str(last_result.get("error") or "") or str(last_result.get("status") or "crash")

    repair_log = workdir / "results" / f"repro_repair_{repair_round:02d}.json"
    success, proxy, contract = _contract_context(workdir)
    benchmark_contract = {
        "formal_benchmark_required": _formal_benchmark_required(workdir),
        "real_benchmark_required": bool(proxy.get("real_benchmark_required")),
        "benchmark_model": proxy.get("benchmark_model"),
        "benchmark_dataset": proxy.get("benchmark_dataset"),
        "benchmark_dataset_config": proxy.get("benchmark_dataset_config"),
        "required_real_benchmarks": contract.get("required_real_benchmarks"),
        "quality_gates": contract.get("quality_gates"),
    }

    from agents.llm_client import (
        call_llm_json_for_role,
        configured_role_prompt_version,
    )

    print(f"[LOOP] Reproduction repair via routed LLM JSON patches (round {repair_round})...", flush=True)
    system = textwrap.dedent("""\
        You repair ML experiment code so a baseline command runs locally and prints a numeric metric.
        Return ONLY valid JSON with this shape:
        {"summary":"one line","files":[{"path":"relative/path.py","content":"FULL new file text"}]}
        Rules: at most 4 files; paths use forward slashes relative to repo root; no path segments ".." .
        For ordinary local experiments, prefer fixing imports, paths, and minimal runtime settings.
        For formal real benchmarks, preserve the benchmark contract: use the contracted model/dataset, load real data, keep CUDA/device handling when a GPU worker is assigned, and do not replace the runner with sklearn/random/toy/smoke data.
        The log must contain either a line like metric_value: 0.42 OR a FINAL_RESULTS: {...} JSON line.""")

    user = textwrap.dedent(f"""\
        metric_name preference: {metric_name}
        baseline_command: {baseline_command or "(python main train script)"}
        benchmark_contract: {json.dumps(benchmark_contract, ensure_ascii=False)}
        environment_report: {json.dumps(environment_report, ensure_ascii=False)[:4000]}
        last status: {last_result.get("status")}
        error: {err[:4000]}
        log tail:
        {log_tail[:10000]}""")

    try:
        run_scope = db.fetchone(
            """
            SELECT er.agenda_id, er.deep_insight_id, er.resource_grant_id,
                   rg.stage
            FROM experiment_runs AS er
            JOIN resource_grants AS rg
              ON rg.id=er.resource_grant_id
             AND rg.agenda_id=er.agenda_id
             AND rg.idea_id=er.deep_insight_id
             AND rg.status='active'
             AND rg.expires_at > CURRENT_TIMESTAMP
            WHERE er.id=?
            """,
            (run_id,),
        )
        if not run_scope:
            raise PermissionError(
                "reproduction repair requires an active scoped ResourceGrant"
            )
        idempotency_material = "|".join(
            (
                str(run_scope["agenda_id"]),
                str(run_scope["deep_insight_id"]),
                str(run_id),
                str(repair_round),
                err,
                log_tail,
            )
        )
        payload, _tokens, route = call_llm_json_for_role(
            system,
            user,
            agenda_id=int(run_scope["agenda_id"]),
            idea_id=int(run_scope["deep_insight_id"]),
            role="proposer",
            stage=str(run_scope["stage"]),
            resource_grant_id=int(run_scope["resource_grant_id"]),
            operation="validation_reproduction_repair",
            idempotency_key=(
                "validation-reproduction-repair:"
                + hashlib.sha256(
                    idempotency_material.encode("utf-8")
                ).hexdigest()
            ),
            prompt_version=configured_role_prompt_version("proposer"),
            max_tokens=12000,
        )
    except Exception as e:
        summary = {
            "round": repair_round,
            "executor": "role_routed_llm",
            "ok": False,
            "error": str(e),
        }
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
        "executor": "role_routed_llm",
        "ok": bool(written),
        "summary": (payload.get("summary") if isinstance(payload, dict) else None) or "llm patch",
        "files_written": written,
        "llm_route": route,
    }
    repair_log.parent.mkdir(parents=True, exist_ok=True)
    repair_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_full_benchmark_completion(run_id: int, execution_context: dict | None = None) -> dict:
    """Run the locked publication benchmark contract for an already forged run."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"run_id": run_id, "verdict": "blocked", "reason": "missing_run"}
    grant = db.fetchone(
        """
        SELECT id FROM resource_grants
        WHERE id=? AND agenda_id=? AND stage='full_benchmark'
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (run.get("resource_grant_id"), run.get("agenda_id")),
    )
    if not grant:
        return {
            "run_id": run_id,
            "verdict": "blocked",
            "reason": "active full_benchmark ResourceGrant required",
        }
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
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (error, run_id, int(run["agenda_id"])),
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
    time_budget = _full_benchmark_time_budget(proxy)

    db.execute(
        "UPDATE experiment_runs SET status='running_gpu', phase='full_benchmark', started_at=COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id=? AND agenda_id=?",
        (run_id, int(run["agenda_id"])),
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
           (agenda_id, run_id, iteration_number, phase, metric_value, metric_name,
            peak_memory_mb, duration_seconds, status, description)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            int(run["agenda_id"]),
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
            "UPDATE experiment_runs SET status='failed', phase='full_benchmark', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (str(error), run_id, int(run["agenda_id"])),
        )
        db.commit()
        return {"run_id": run_id, "verdict": "failed", "reason": str(error), "execution_report": result}

    _, _, candidate_value, baseline_value, _ = _benchmark_scores(benchmark_summary)
    if baseline_value is None or candidate_value is None:
        error = "metric_missing:baseline_or_candidate_metric"
        db.execute(
            "UPDATE experiment_runs SET status='failed', phase='full_benchmark', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (error, run_id, int(run["agenda_id"])),
        )
        db.commit()
        return {
            "run_id": run_id,
            "verdict": "failed",
            "reason": error,
            "reason_code": "metric_missing",
            "execution_report": result,
        }
    baseline = float(baseline_value)
    best_value = float(candidate_value)
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
           WHERE id=? AND agenda_id=?""",
        (
            verdict,
            baseline,
            best_value,
            effect,
            effect_pct,
            run_id,
            int(run["agenda_id"]),
        ),
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
    grant = db.fetchone(
        """
        SELECT id FROM resource_grants
        WHERE id=? AND agenda_id=? AND stage IN ('pilot', 'validation')
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (run.get("resource_grant_id"), run.get("agenda_id")),
    )
    if not grant:
        return {
            "run_id": run_id,
            "verdict": "blocked",
            "reason": "active pilot/validation ResourceGrant required",
        }

    insight_id = run["deep_insight_id"]
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return {"error": f"Insight {insight_id} not found"}

    gate = evosci_strict_gate_insight(dict(insight))
    if gate:
        err = gate.get("error", "EvoScientist strict gate blocked validation loop")
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (err, run_id, int(run["agenda_id"])),
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
        db.execute(
            "UPDATE experiment_runs SET workdir=? WHERE id=? AND agenda_id=?",
            (str(workdir), run_id, int(run["agenda_id"])),
        )
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
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (error, run_id, int(run["agenda_id"])),
        )
        db.commit()
        return {"run_id": run_id, "verdict": "blocked", "reason": "non_formal_experiment"}

    criteria = _read_success_criteria(workdir, insight_id)
    proxy = _read_proxy_config(workdir, insight_id)

    spec = _read_experiment_spec(run, insight, workdir, criteria=criteria, proxy=proxy)
    metric_name = criteria.get("metric_name", "metric")
    direction = criteria.get("metric_direction", "higher")
    time_budget = proxy.get("time_budget_seconds", EXPERIMENT_TIME_BUDGET)
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
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (error, run_id, int(run["agenda_id"])),
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
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (error, run_id, int(run["agenda_id"])),
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
            "UPDATE experiment_runs SET status='testing', phase='hypothesis_testing', started_at=COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id=? AND agenda_id=?",
            (run_id, int(run["agenda_id"])),
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
        db.execute(
            "UPDATE experiment_runs SET status='reproducing', phase='reproduction', started_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (run_id, int(run["agenda_id"])),
        )
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
    last_recovery_decision = None

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
                   (agenda_id, run_id, iteration_number, phase, metric_value, metric_name,
                    peak_memory_mb, duration_seconds, status, description)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (int(run["agenda_id"]), run_id, iter_key, "reproduction", metric, metric_name,
                 result.get("peak_memory_mb"), result.get("duration"),
                 result.get("status", "ok"), json.dumps(packet.result_judgement)[:500])
            )
            db.commit()

            if metric is not None:
                baseline_values.append(metric)
                print(f"[LOOP] Reproduction {i+1}/{repro_iters}: {metric_name}={metric}", flush=True)
            else:
                print(f"[LOOP] Reproduction {i+1}/{repro_iters}: no metric (status={result.get('status')})", flush=True)
                last_recovery_decision = _execution_recovery_decision(
                    run_id,
                    result,
                    retry_count=repair_round,
                )
                result["reason_code"] = last_recovery_decision.reason_code
                result["recovery_action"] = last_recovery_decision.action
                if last_recovery_decision.action == "retry_adjusted":
                    if not _apply_runner_recovery_adjustments(
                        code_dir, dict(last_recovery_decision.adjustments)
                    ):
                        break
                    continue
                if last_recovery_decision.action == "retry_with_backoff":
                    time.sleep(max(0, int(last_recovery_decision.backoff_seconds)))
                    continue
                break

        if baseline_values:
            break

        if REPRODUCTION_REPAIR_MAX_ROUNDS <= 0:
            break
        if repair_round >= REPRODUCTION_REPAIR_MAX_ROUNDS:
            break

        if not last_recovery_decision or not last_recovery_decision.invoke_llm_repair:
            break

        repair_result = _launch_reproduction_repair(
            run_id=run_id,
            workdir=workdir,
            code_dir=code_dir,
            repair_round=repair_round + 1,
            baseline_command=baseline_command,
            metric_name=metric_name,
            last_result=last_repro_result,
            environment_report=environment_report,
        )
        # A refused/unavailable repair did not change the code. Retrying the
        # identical crashed command only burns GPU time and creates misleading
        # "repair applied" iterations. Leave the run failed so the bounded
        # outer autonomous recovery policy can act after an operator deploys a
        # real infrastructure fix.
        if not repair_result.get("ok"):
            print(
                "[LOOP] Phase 1: repair unavailable; refusing unchanged GPU retry "
                f"(error={str(repair_result.get('error') or 'no patch')[:200]})",
                flush=True,
            )
            break
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
        failure_type = str(last_repro_result.get("failure_type") or "missing_metric")
        last_error = str(last_repro_result.get("error") or "no metric obtained")
        recovery_action = (
            last_recovery_decision.action
            if last_recovery_decision is not None
            else "defer"
        )
        reason_code = (
            last_recovery_decision.reason_code
            if last_recovery_decision is not None
            else failure_type
        )
        error_message = (
            "reproduction failed: no metric obtained; "
            f"reason_code={reason_code}; recovery_action={recovery_action}; "
            f"last_failure_type={failure_type}; last_error={last_error[:500]}"
        )
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (error_message, run_id, int(run["agenda_id"])))
        db.commit()
        write_latest_status(
            insight_id,
            {"stage": "reproduction", "status": "failed", "error": error_message},
            run_id=run_id,
            insight=insight,
        )
        print(f"[LOOP] Phase 1 FAILED: could not obtain baseline metric", flush=True)
        return {
            "verdict": "failed",
            "reason": "reproduction_failure",
            "failure_type": failure_type,
            "code_repair_required": bool(
                last_recovery_decision and last_recovery_decision.invoke_llm_repair
            ),
            "reason_code": reason_code,
            "recovery_action": recovery_action,
        }

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
        "UPDATE experiment_runs SET baseline_metric_value=?, best_metric_value=?, phase='hypothesis_testing', status='testing' WHERE id=? AND agenda_id=?",
        (baseline, best_value, run_id, int(run["agenda_id"])))
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

        prior_method_feedback = experiment_feedback.load_latest_method_feedback(workdir)
        supervisor_plan = experiment_supervisor.build_supervisor_plan(
            spec=spec,
            environment_report=environment_report,
            baseline=baseline,
            best_so_far=best_value,
            history=history,
            iteration=i + 1,
            success_criteria=criteria,
            method_feedback=prior_method_feedback,
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
        redesign_payload = _read_redesign_required_artifact(code_dir, workdir)
        validation_status = str(coding_step.get("validation_status") or "").strip().lower()
        if validation_status in {"blocked_redesign_required", "redesign_required", "benchmark_mismatch"} and not redesign_payload:
            redesign_payload = {
                "reason": "Codex reported validation_status indicating redesign or benchmark mismatch.",
                "validation_status": validation_status,
                "mechanism_needed": method_desc[:800],
                "benchmark_gap": "No structured redesign artifact was written by the worker.",
                "why_not_scientific_failure": (
                    "The method was not tested against a suitable locked benchmark/harness, "
                    "so this is an operationalization issue rather than evidence against the idea."
                ),
                "not_scientific_verdict": True,
            }
        if redesign_payload:
            redesign_path = _persist_redesign_required_artifact(workdir, i + 1, redesign_payload)
            result = {
                "status": "blocked",
                "metric": None,
                "error": str(
                    redesign_payload.get("reason")
                    or redesign_payload.get("benchmark_gap")
                    or "Experiment redesign or benchmark harness work is required."
                ),
                "duration": 0.0,
                "peak_memory_mb": None,
                "redesign_required": redesign_payload,
                "redesign_required_path": redesign_path,
            }
            metric = None
            iteration_benchmark_summary = {}
            result_judgement = {
                "role": "ResultJudge",
                "status": "discard",
                "summary": (
                    "Worker reported a mechanism/benchmark mismatch before evaluation. "
                    "This iteration requires reforge or benchmark-harness work, not another "
                    "score-only method tweak."
                ),
                "anomaly_type": "benchmark_mismatch_or_redesign_required",
                "continue": True,
                "terminate": False,
                "stop_reason": "",
                "metric": None,
                "baseline": baseline,
                "benchmark_semantic_warnings": [result["error"]],
                "paper_evidence_warning": True,
                "redesign_required_path": redesign_path,
                "not_scientific_verdict": True,
            }
            status = "discard"
            fairness_warnings = []
        elif git_bin and not commit_hash:
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
        method_feedback_payload = experiment_feedback.build_method_feedback(
            workdir=workdir,
            run_id=run_id,
            iteration=i + 1,
            result=result,
            result_judgement=result_judgement,
            history=history,
            criteria=criteria,
            baseline=baseline,
            best_value=best_before,
        )
        method_feedback_path = experiment_feedback.write_method_feedback(workdir, method_feedback_payload)
        result_judgement["method_feedback"] = {
            "findings": method_feedback_payload.get("findings", [])[:5],
            "next_actions": method_feedback_payload.get("next_actions", [])[:5],
            "path": str(method_feedback_path),
        }

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
                "method_feedback": str(method_feedback_path),
                **(
                    {"redesign_required": result.get("redesign_required_path")}
                    if result.get("redesign_required_path")
                    else {}
                ),
                **supervisor_artifacts,
                **coding_step.get("artifact_paths", {}),
            },
            commit_hash=commit_hash or "",
            code_diff=diff,
        )
        _write_iteration_packet(workdir, packet, run_id)

        db.execute(
            """INSERT INTO experiment_iterations
               (agenda_id, run_id, iteration_number, phase, code_diff, commit_hash,
                metric_value, metric_name, peak_memory_mb, duration_seconds,
                status, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(run["agenda_id"]), run_id, iter_num, "hypothesis_testing", diff, commit_hash,
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
               WHERE id=? AND agenda_id=?""",
            (
                iter_num,
                total_kept,
                best_value,
                effect,
                effect_pct,
                run_id,
                int(run["agenda_id"]),
            )
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
        automation_streak = _recent_automation_failure_streak(history)
        if automation_streak >= AUTOMATION_FAILURE_PATIENCE:
            stop_reason = (
                "Automation failed: hypothesis testing produced no benchmarked candidate method "
                f"change for {automation_streak} consecutive iterations; code_repair_required; "
                "experiment_reforge_required."
            )
            print(f"[LOOP] Automation failure stop at iter {i+1}: {stop_reason}", flush=True)
            break
        if plateau_patience > 0 and len(history) >= max(refute_min, plateau_patience):
            recent = history[-plateau_patience:]
            if recent and all(row.get("status") != "keep" for row in recent):
                stop_reason = f"No kept improvement in the last {plateau_patience} iterations."
                print(f"[LOOP] Plateau stop at iter {i+1}: {stop_reason}", flush=True)
                break

    # ── Determine verdict ──
    final_benchmark_summary = best_benchmark_summary or benchmark_summary
    automation_failed = _hypothesis_testing_automation_failed(history)
    verdict = _determine_final_verdict(
        baseline=baseline,
        best_value=best_value,
        direction=direction,
        criteria=criteria,
        total_iters=len(history),
        total_kept=total_kept,
        refute_min=refute_min,
        benchmark_summary=final_benchmark_summary if benchmark_mode else None,
        automation_failed=automation_failed,
    )
    final_method_feedback_path = None
    if history:
        final_method_feedback_payload = experiment_feedback.build_method_feedback(
            workdir=workdir,
            run_id=run_id,
            iteration=None,
            result={"benchmark_summary": final_benchmark_summary or {}},
            result_judgement={
                "status": "discard" if automation_failed else verdict,
                "anomaly_type": "automation_failure" if automation_failed else "",
            },
            history=history,
            criteria=criteria,
            baseline=baseline,
            best_value=best_value,
        )
        final_method_feedback_path = experiment_feedback.write_method_feedback(workdir, final_method_feedback_payload)

    automation_failure_path = None
    if automation_failed:
        if not stop_reason or not stop_reason.startswith("Automation failed:"):
            stop_reason = (
                "Automation failed: hypothesis testing produced no benchmarked "
                "candidate method changes; refusing to label a no-op loop as refuted."
            )
        automation_failure_path = _write_automation_failure_artifact(
            workdir,
            run_id=run_id,
            insight_id=insight_id,
            history=history,
            stop_reason=stop_reason,
            method_desc=method_desc,
        )
    effect = best_value - baseline if direction == "higher" else baseline - best_value
    effect_pct = (effect / abs(baseline) * 100) if baseline != 0 else 0

    total_time = time.time() - loop_start
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', hypothesis_verdict=?,
               effect_size=?, effect_pct=?, error_message=?,
               completed_at=CURRENT_TIMESTAMP
           WHERE id=? AND agenda_id=?""",
        (
            verdict,
            effect,
            effect_pct,
            stop_reason or None,
            run_id,
            int(run["agenda_id"]),
        )
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
                "automation_failed": automation_failed,
                "automation_failure_path": str(automation_failure_path) if automation_failure_path else "",
                "method_feedback_path": str(final_method_feedback_path) if final_method_feedback_path else "",
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
    if automation_failure_path:
        _record_artifact(
            run_id,
            "source_data",
            automation_failure_path,
            metric_key=metric_name,
            metric_value=best_value,
            metadata={"contract_type": "HypothesisTestingAutomationFailure"},
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
            "automation_failed": automation_failed,
            "automation_failure_path": str(automation_failure_path) if automation_failure_path else "",
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
        "automation_failed": automation_failed,
        "automation_failure_path": str(automation_failure_path) if automation_failure_path else "",
    }
