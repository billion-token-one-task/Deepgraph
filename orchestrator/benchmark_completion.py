"""Helpers for turning manuscript benchmark blockers into follow-up jobs."""

from __future__ import annotations

from db import database as db

BENCHMARK_COMPLETION_STAGE = "benchmark_completion_required"

_BENCHMARK_BLOCKER_MARKERS = (
    "full_benchmark",
    "full benchmark",
    "benchmark_summary",
    "benchmark artifact",
    "required benchmark",
    "benchmark coverage",
    "required baselines",
    "required baseline",
    "required ablation",
    "ablation table",
    "num_seeds",
    "seed",
)


def benchmark_completion_blockers(bundle: dict | None) -> list[str]:
    """Return submission blockers that should trigger another benchmark run."""
    if not isinstance(bundle, dict):
        return []
    raw_blockers = bundle.get("submission_blockers")
    blockers: list[str] = []
    if isinstance(raw_blockers, list):
        blockers.extend(str(item).strip() for item in raw_blockers if str(item or "").strip())
    error = str(bundle.get("error") or "").strip()
    if error:
        blockers.append(error)
    return [
        blocker
        for blocker in blockers
        if any(marker in blocker.lower() for marker in _BENCHMARK_BLOCKER_MARKERS)
    ]


def schedule_benchmark_completion(
    insight_id: int,
    run_id: int,
    bundle: dict | None,
    *,
    source: str,
    resource_class: str | None = None,
) -> bool:
    """Queue the insight for an automatic benchmark-completion refit if needed."""
    blockers = benchmark_completion_blockers(bundle)
    if not blockers:
        return False
    short_error = "; ".join(blockers[:6])
    note = (
        "Submission bundle is blocked by incomplete benchmark evidence; "
        "queued automatic real-benchmark completion."
    )
    existing = db.fetchone(
        "SELECT id, resource_class FROM auto_research_jobs WHERE deep_insight_id=?",
        (insight_id,),
    )
    chosen_resource = resource_class or (existing.get("resource_class") if existing else None) or "gpu_large"
    if existing:
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='queued',
                stage=?,
                experiment_run_id=?,
                resource_class=?,
                scheduler_priority=CASE
                    WHEN COALESCE(scheduler_priority, 0) < 2 THEN 2
                    ELSE scheduler_priority
                END,
                assigned_worker=NULL,
                last_note=?,
                last_error=?,
                last_checked_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=?
            """,
            (
                BENCHMARK_COMPLETION_STAGE,
                run_id,
                chosen_resource,
                note,
                short_error,
                insight_id,
            ),
        )
    else:
        db.execute(
            """
            INSERT INTO auto_research_jobs
              (deep_insight_id, status, stage, experiment_run_id, resource_class,
               scheduler_priority, last_note, last_error)
            VALUES (?, 'queued', ?, ?, ?, 2, ?, ?)
            """,
            (insight_id, BENCHMARK_COMPLETION_STAGE, run_id, chosen_resource, note, short_error),
        )
    db.commit()
    db.emit_pipeline_event(
        "benchmark_completion_required",
        {
            "deep_insight_id": insight_id,
            "experiment_run_id": run_id,
            "source": source,
            "blockers": blockers,
            "resource_class": chosen_resource,
        },
        entity_type="deep_insight",
        entity_id=str(insight_id),
        dedupe_key=f"benchmark_completion_required:{insight_id}:{run_id}",
    )
    return True
