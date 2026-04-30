"""Run configured benchmark suites and record result artifacts."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, record_artifact
from agents.capability_registry import get_capability
from db import database as db


def load_benchmark_config(workdir: Path) -> dict:
    """Load benchmark_config.json from a run workdir."""
    path = Path(workdir) / "benchmark_config.json"
    if not path.exists():
        raise FileNotFoundError("benchmark_config_not_found")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_capability(capability: dict, config: dict) -> dict:
    runner = capability.get("runner")
    if not runner:
        raise ValueError(f"capability has no runner: {capability.get('id')}")
    module = importlib.import_module(runner)
    if capability.get("id") == "fairness_classification":
        return module.run_fairness_benchmark(config)
    if capability.get("id") == "safe_rl_cmdp":
        return module.run_safe_rl_benchmark(config)
    raise ValueError(f"unsupported benchmark capability: {capability.get('id')}")


def _ablation_config(base_config: dict, ablation: dict) -> dict:
    config = dict(base_config)
    for key in (
        "datasets",
        "methods",
        "seeds",
        "primary_metric",
        "metric_direction",
        "timeout_seconds",
        "preference_penalties",
        "safety_penalty",
    ):
        if key in ablation:
            config[key] = ablation[key]
    config.pop("ablations", None)
    return config


def _run_with_ablations(capability: dict, config: dict) -> dict:
    payload = _run_capability(capability, config)
    rows = payload.setdefault("rows", [])
    for row in rows:
        row.setdefault("analysis_type", "main")

    for ablation in config.get("ablations") or []:
        if not isinstance(ablation, dict):
            continue
        name = str(ablation.get("name") or "unnamed_ablation")
        ablation_payload = _run_capability(capability, _ablation_config(config, ablation))
        label = name
        if "safety_penalty" in ablation:
            label = f"{name}:safety_penalty={float(ablation['safety_penalty']):.2f}"
        for row in ablation_payload.get("rows", []):
            row["analysis_type"] = "ablation"
            row["ablation"] = label
            rows.append(row)
    return payload


def run_benchmark_suite(run_id: int) -> dict:
    """Run the benchmark suite configured for an experiment run."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"status": "error", "reason": "run_not_found", "run_id": run_id}

    workdir = Path(run.get("workdir") or "")
    try:
        config = load_benchmark_config(workdir)
    except FileNotFoundError:
        return {"status": "error", "reason": "benchmark_config_not_found", "run_id": run_id}
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "error", "reason": "invalid_benchmark_config", "error": str(exc), "run_id": run_id}

    capability = get_capability(config.get("capability", ""))
    if not capability or not capability.get("implemented"):
        return {
            "status": "error",
            "reason": "capability_not_available",
            "capability": config.get("capability"),
            "run_id": run_id,
        }

    try:
        payload = _run_with_ablations(capability, config)
    except Exception as exc:
        return {
            "status": "error",
            "reason": "benchmark_runner_failed",
            "error": str(exc),
            "run_id": run_id,
        }

    payload["run_id"] = run_id
    payload["capability"] = capability["id"]
    ensure_artifact_dirs(workdir)
    path = artifact_path(workdir, "artifacts/results/benchmark_results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    record_artifact(
        workdir,
        run_id,
        "benchmark_results",
        path,
        {"capability": capability["id"]},
    )
    extra_artifacts = None
    if capability["id"] == "safe_rl_cmdp":
        from benchmarks.safe_rl_cmdp.artifacts import write_safe_rl_reproducibility_artifacts

        extra_artifacts = write_safe_rl_reproducibility_artifacts(workdir, run_id, config)
    return {
        "status": "complete",
        "run_id": run_id,
        "capability": capability["id"],
        "rows": len(payload.get("rows", [])),
        "path": str(path),
        "reproducibility_artifacts": extra_artifacts,
    }
