"""Agenda orchestrator: bridges agenda_selections to the existing closed loop.

Issue #9, step 4: once the selector picks an insight, this module either
- looks up the existing experiment_run / manuscript_run / submission_bundle for
  that insight (demo path, used when an insight already has artifacts), or
- enqueues the insight into the auto_research worker (fresh-run path).

The module deliberately does NOT spawn EvoScientist subprocesses directly —
that path (`agents.research_bridge.launch_evoscientist`) has heavy side effects
and depends on an LLM key. The auto_research worker handles dispatch.

Public API:
- link_existing_artifacts(selection_id) -> dict
- enqueue_for_auto_research(selection_id) -> dict
- dispatch_selection(selection_id, *, mode='auto') -> dict
"""

from __future__ import annotations

from typing import Any

from agents.agenda_selector import get_selection, update_selection_progress
from db import database as db


# ---------- artifact lookup helpers ----------


def _latest_experiment_run(insight_id: int) -> dict[str, Any] | None:
    return db.fetchone(
        """
        SELECT id, status, phase, hypothesis_verdict,
               baseline_metric_value, best_metric_value, effect_size,
               submission_bundle_id, workdir, started_at, completed_at
        FROM experiment_runs
        WHERE deep_insight_id=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (insight_id,),
    )


def _latest_manuscript_run(insight_id: int) -> dict[str, Any] | None:
    return db.fetchone(
        """
        SELECT id, status, workdir, created_at, updated_at
        FROM manuscript_runs
        WHERE deep_insight_id=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (insight_id,),
    )


def _bundle_for_run(manuscript_run_id: int | None) -> dict[str, Any] | None:
    if manuscript_run_id is None:
        return None
    return db.fetchone(
        """
        SELECT id, manuscript_run_id, bundle_format, status, bundle_path,
               manifest_path, created_at
        FROM submission_bundles
        WHERE manuscript_run_id=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (manuscript_run_id,),
    )


def _bundle_for_experiment(experiment_run_id: int | None) -> dict[str, Any] | None:
    if experiment_run_id is None:
        return None
    return db.fetchone(
        """
        SELECT id, manuscript_run_id, bundle_format, status, bundle_path,
               manifest_path, created_at
        FROM submission_bundles
        WHERE id = (
            SELECT submission_bundle_id FROM experiment_runs WHERE id=?
        )
        """,
        (experiment_run_id,),
    )


def _auto_research_job(insight_id: int) -> dict[str, Any] | None:
    return db.fetchone(
        """
        SELECT id, deep_insight_id, status, stage, last_note, last_error,
               research_workdir, last_checked_at, updated_at
        FROM auto_research_jobs
        WHERE deep_insight_id=?
        """,
        (insight_id,),
    )


# ---------- main entry points ----------


def link_existing_artifacts(selection_id: int) -> dict[str, Any]:
    """Look up existing artifacts for this selection's insight and record them.

    Returns a dict with the linked artifacts. Updates the selection row with
    experiment_run_id / manuscript_run_id / submission_bundle_id and sets
    status='completed' if a bundle exists, else 'launched'.
    """
    sel = get_selection(selection_id)
    if not sel:
        raise ValueError(f"selection {selection_id} not found")
    if not sel.get("selected_insight_id"):
        raise ValueError(f"selection {selection_id} has no selected_insight_id")

    insight_id = int(sel["selected_insight_id"])
    job = _auto_research_job(insight_id)
    exp = _latest_experiment_run(insight_id)
    manu = _latest_manuscript_run(insight_id)
    bundle = _bundle_for_run(manu["id"] if manu else None) or _bundle_for_experiment(
        exp["id"] if exp else None
    )

    progress: dict[str, Any] = {}
    if job:
        progress["auto_research_job_id"] = int(job["id"])
    if exp:
        progress["experiment_run_id"] = int(exp["id"])
    if manu:
        progress["manuscript_run_id"] = int(manu["id"])
    if bundle:
        progress["submission_bundle_id"] = int(bundle["id"])

    if bundle:
        progress["status"] = "completed"
    elif exp or manu or job:
        progress["status"] = "launched"
    else:
        progress["status"] = "blocked"
        progress["error_message"] = (
            f"No existing artifacts for insight {insight_id}; use enqueue mode for fresh run."
        )

    if progress:
        update_selection_progress(selection_id, **progress)

    return {
        "selection_id": selection_id,
        "insight_id": insight_id,
        "auto_research_job": job,
        "experiment_run": exp,
        "manuscript_run": manu,
        "submission_bundle": bundle,
        "applied_progress": progress,
    }


def enqueue_for_auto_research(selection_id: int) -> dict[str, Any]:
    """Hand the selected insight off to the existing auto_research worker.

    Uses `orchestrator.auto_research._upsert_job` so we plug into the same
    state machine that already exists. Sets selection.status='launched'
    and records the auto_research_job_id.
    """
    from orchestrator import auto_research

    sel = get_selection(selection_id)
    if not sel:
        raise ValueError(f"selection {selection_id} not found")
    if not sel.get("selected_insight_id"):
        raise ValueError(f"selection {selection_id} has no selected_insight_id")

    insight_id = int(sel["selected_insight_id"])
    auto_research._upsert_job(
        insight_id,
        status="queued",
        stage="queued",
        last_note=f"enqueued by agenda selection #{selection_id}",
    )
    job = _auto_research_job(insight_id)
    progress: dict[str, Any] = {"status": "launched"}
    if job:
        progress["auto_research_job_id"] = int(job["id"])
    update_selection_progress(selection_id, **progress)
    return {
        "selection_id": selection_id,
        "insight_id": insight_id,
        "auto_research_job": job,
        "applied_progress": progress,
    }


def dispatch_selection(selection_id: int, *, mode: str = "auto") -> dict[str, Any]:
    """Dispatch a selection.

    mode='link'    : only link existing artifacts (demo path)
    mode='enqueue' : enqueue into auto_research worker (fresh run)
    mode='auto'    : try link first; if nothing linked, fall back to enqueue
    """
    if mode == "link":
        return link_existing_artifacts(selection_id)
    if mode == "enqueue":
        return enqueue_for_auto_research(selection_id)
    if mode != "auto":
        raise ValueError(f"unknown dispatch mode: {mode}")

    linked = link_existing_artifacts(selection_id)
    progress = linked.get("applied_progress") or {}
    if progress.get("status") in ("completed", "launched"):
        return {"mode": "link", **linked}
    enqueued = enqueue_for_auto_research(selection_id)
    return {"mode": "enqueue", **enqueued}
