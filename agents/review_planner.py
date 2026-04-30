"""Convert AI review feedback into follow-up experiment plan artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, record_artifact
from db import database as db


def _load_review(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("review"), dict):
        payload = payload["review"]
    return payload if isinstance(payload, dict) else None


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def _resolve_workdir(run_id: int, workdir: Path | None) -> tuple[Path | None, dict | None]:
    if workdir is not None:
        return Path(workdir), None
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return None, None
    raw_workdir = run.get("workdir")
    if not raw_workdir:
        return None, run
    return Path(raw_workdir), run


def plan_followup_experiments(run_id: int, workdir: Path | None = None) -> dict:
    """Write a follow-up experiment plan from `artifacts/reviews/review.json`."""
    resolved_workdir, run = _resolve_workdir(run_id, workdir)
    if resolved_workdir is None:
        reason = "run_not_found" if run is None and workdir is None else "missing_workdir"
        return {"status": "error", "reason": reason, "run_id": run_id}

    review_path = artifact_path(resolved_workdir, "artifacts/reviews/review.json")
    review = _load_review(review_path)
    if review is None:
        return {"status": "error", "reason": "review_not_found", "run_id": run_id}

    required = _as_list(review.get("required_experiments"))
    major_concerns = _as_list(review.get("major_concerns"))
    experiments = [
        {
            "source": "ai_review",
            "description": description,
            "suggested_artifact": "benchmark_config.json",
        }
        for description in required
    ]

    status = "needs_followup" if experiments else "no_followup_required"
    plan = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "review_recommendation": str(review.get("recommendation") or ""),
        "major_concerns": major_concerns,
        "experiments": experiments,
    }

    ensure_artifact_dirs(resolved_workdir)
    path = artifact_path(resolved_workdir, "artifacts/results/followup_experiment_plan.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    record_artifact(resolved_workdir, run_id, "followup_experiment_plan", path, {
        "status": status,
        "experiment_count": len(experiments),
    })
    return plan
