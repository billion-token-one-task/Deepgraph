"""Supervisor planning layer for experiment iterations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts import ExperimentSpec


def _history_summary(history: list[dict], limit: int = 8) -> list[dict]:
    out: list[dict] = []
    for row in history[-limit:]:
        out.append(
            {
                "iteration": row.get("iteration"),
                "status": row.get("status"),
                "metric": row.get("metric"),
                "description": str(row.get("description") or "")[:200],
            }
        )
    return out


def build_supervisor_plan(
    *,
    spec: ExperimentSpec,
    environment_report: dict[str, Any],
    baseline: float | None,
    best_so_far: float | None,
    history: list[dict],
    iteration: int,
    success_criteria: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic supervisor plan for the next worker turn."""
    last = history[-1] if history else {}
    crash_streak = 0
    for row in reversed(history):
        if row.get("status") == "crash":
            crash_streak += 1
        else:
            break

    if crash_streak >= 1:
        mode = "repair"
        diagnosis = "Recent iterations crashed; prioritize restoring a runnable baseline-compatible path."
    elif history and last.get("status") == "discard":
        mode = "redirect"
        diagnosis = "Last hypothesis did not improve metrics; try a materially different intervention."
    elif history and last.get("status") == "keep":
        mode = "refine"
        diagnosis = "Previous change improved the metric; refine carefully without destabilizing the run."
    else:
        mode = "bootstrap"
        diagnosis = "No successful hypothesis iteration yet; start with the smallest plausible implementation of the method."

    main_train_file = spec.proxy_config.get("main_train_file") or environment_report.get("resolved_train_file") or "auto-detect"
    metric_name = success_criteria.get("metric_name", "metric")
    metric_direction = success_criteria.get("metric_direction", "higher")

    if mode == "repair":
        next_actions = [
            "Locate the exact failing command, import, or file path that broke execution.",
            "Make the minimum repo change needed to restore a clean runnable experiment.",
            "Avoid broad method changes until the experiment executes again.",
        ]
    elif mode == "redirect":
        next_actions = [
            "Avoid repeating the last discarded intervention.",
            "Choose a different mechanism consistent with the proposed method definition.",
            "Keep the change narrow enough that its effect can be isolated in one run.",
        ]
    elif mode == "refine":
        next_actions = [
            "Build on the best-performing idea without changing unrelated components.",
            "Protect baseline fairness and evaluation compatibility.",
            "Prefer parameterization, scheduling, or integration refinements over wholesale rewrites.",
        ]
    else:
        next_actions = [
            "Implement the smallest end-to-end version of the proposed method.",
            "Preserve the current training/evaluation pipeline shape where possible.",
            "Keep the code change local and easy to reason about.",
        ]

    plan = {
        "role": "ExperimentSupervisor",
        "iteration": iteration,
        "mode": mode,
        "objective": f"Improve {metric_name} ({metric_direction}) over baseline {baseline} and best-so-far {best_so_far}.",
        "diagnosis": diagnosis,
        "main_train_file": main_train_file,
        "baseline_command": spec.proxy_config.get("baseline_command"),
        "environment": {
            "formal_ready": environment_report.get("formal_ready"),
            "entrypoint_exists": environment_report.get("entrypoint_exists"),
            "resource_class": environment_report.get("resource_class"),
            "resolved_train_file": environment_report.get("resolved_train_file"),
        },
        "repo_focus": [main_train_file],
        "guardrails": [
            "Honor the evidence plan and do not invent disabled analyses.",
            "Keep changes hypothesis-directed and minimal.",
            "Do not break the existing baseline execution path.",
        ],
        "next_actions": next_actions,
        "history": _history_summary(history),
        "success_criteria": success_criteria,
    }
    return plan


def write_supervisor_artifacts(workdir: Path, plan: dict[str, Any]) -> dict[str, str]:
    """Persist supervisor artifacts for the current iteration."""
    iter_num = int(plan.get("iteration") or 0)
    sup_dir = workdir / "codex" / "supervisor"
    sup_dir.mkdir(parents=True, exist_ok=True)
    json_path = sup_dir / f"iter_{iter_num:03d}_plan.json"
    md_path = sup_dir / f"iter_{iter_num:03d}_plan.md"

    json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    md_text = "\n".join(
        [
            "# Experiment Supervisor Plan",
            f"- Iteration: {plan.get('iteration')}",
            f"- Mode: {plan.get('mode')}",
            f"- Objective: {plan.get('objective')}",
            f"- Diagnosis: {plan.get('diagnosis')}",
            "",
            "## Next Actions",
            *[f"- {row}" for row in plan.get("next_actions", [])],
            "",
            "## Guardrails",
            *[f"- {row}" for row in plan.get("guardrails", [])],
        ]
    )
    md_path.write_text(md_text, encoding="utf-8")
    return {
        "supervisor_plan_json": str(json_path),
        "supervisor_plan_md": str(md_path),
    }
