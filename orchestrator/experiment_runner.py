"""Async experiment execution lane (CPU fallback) mirroring gpu_scheduler dispatch."""

from __future__ import annotations

import threading

from agents.knowledge_loop import process_completed_run
from agents.manuscript_pipeline import generate_submission_bundle
from agents.validation_loop import run_validation_loop
from db import database as db
from orchestrator.benchmark_completion import schedule_benchmark_completion
from orchestrator.pipeline import log_event
from orchestrator.tracking import tracked_run

_active_lock = threading.Lock()
_active_run_ids: set[int] = {}


def active_execution_count() -> int:
    with _active_lock:
        return len(_active_run_ids)


def is_run_active(run_id: int) -> bool:
    with _active_lock:
        return int(run_id) in _active_run_ids


def _mark_active(run_id: int) -> None:
    with _active_lock:
        _active_run_ids.add(int(run_id))


def _mark_inactive(run_id: int) -> None:
    with _active_lock:
        _active_run_ids.discard(int(run_id))


def _append_error(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {exc}"


def _finish_run(insight_id: int, run_id: int, resource_class: str, result: dict, errors: list[str]) -> None:
    bundle: dict = {}
    try:
        bundle = generate_submission_bundle(run_id)
    except Exception as exc:
        bundle = {"error": str(exc)}
        errors.append(_append_error("submission_bundle_failed", exc))

    completion_queued = schedule_benchmark_completion(
        insight_id,
        run_id,
        bundle,
        source="experiment_runner_cpu",
        resource_class=resource_class,
    )
    verdict = (result or {}).get("verdict", "unknown")
    note = (
        f"Async validation finished: verdict={verdict}. "
        f"bundle={'ok' if 'error' not in bundle else 'failed'}."
    )
    if errors:
        note += f" warnings={len(errors)}"

    if completion_queued:
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='queued_gpu', stage=?, last_note=?, last_error=?, updated_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=?
            """,
            (
                "benchmark_completion",
                "Benchmark completion queued after async validation.",
                "\n".join(errors) if errors else None,
                insight_id,
            ),
        )
    else:
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status=?, stage=?, artifact_bundle_id=?, last_note=?, last_error=?, updated_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=?
            """,
            (
                "bundle_ready" if "error" not in bundle else "completed",
                "writing_submission" if "error" not in bundle else "closed_loop_complete",
                (bundle.get("bundle_ids") or [None])[-1],
                note,
                "\n".join(errors) if errors else None,
                insight_id,
            ),
        )
    db.commit()
    log_event(
        "experiment_runner",
        {"step": "validation_completed", "insight_id": insight_id, "run_id": run_id, "verdict": verdict},
    )


def _run_cpu_validation(insight_id: int, run_id: int, resource_class: str) -> None:
    _mark_active(run_id)
    errors: list[str] = []
    result: dict = {}
    try:
        db.execute(
            "UPDATE experiment_runs SET status='running_cpu', resource_class=? WHERE id=?",
            (resource_class, run_id),
        )
        db.commit()
        with tracked_run(
            f"deepgraph-cpu-run-{run_id}",
            tags={"insight_id": insight_id, "resource_class": resource_class, "lane": "async_cpu"},
        ):
            raw = run_validation_loop(run_id)
            result = raw if isinstance(raw, dict) else {"verdict": "failed", "error": f"unexpected {type(raw).__name__}"}
            try:
                process_completed_run(run_id)
            except Exception as exc:
                errors.append(_append_error("knowledge_loop_failed", exc))
    except Exception as exc:
        result = {"verdict": "failed", "error": str(exc)}
        errors.append(_append_error("validation_loop_failed", exc))
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=? WHERE id=?",
            (str(exc), run_id),
        )
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='failed', stage='experiment_failed', last_error=?, last_note=?
            WHERE deep_insight_id=?
            """,
            (str(exc), "Async validation loop failed.", insight_id),
        )
        db.commit()
        log_event("error", {"step": "experiment_runner_failed", "insight_id": insight_id, "run_id": run_id, "error": str(exc)})
    finally:
        _mark_inactive(run_id)

    if result.get("verdict") != "failed":
        try:
            _finish_run(insight_id, run_id, resource_class, result, errors)
        except Exception as exc:
            log_event("error", {"step": "experiment_runner_finish_failed", "run_id": run_id, "error": str(exc)})


def start_validation_loop_async(
    *,
    insight_id: int,
    run_id: int,
    resource_class: str = "cpu",
) -> None:
    """Dispatch validation_loop on a background thread; returns immediately."""
    thread = threading.Thread(
        target=_run_cpu_validation,
        args=(insight_id, run_id, resource_class),
        daemon=True,
        name=f"cpu-validation-{run_id}",
    )
    thread.start()
    log_event(
        "experiment_runner",
        {"step": "validation_dispatched", "insight_id": insight_id, "run_id": run_id, "resource_class": resource_class},
    )
