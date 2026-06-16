"""Helpers for turning manuscript benchmark blockers into follow-up jobs."""

from __future__ import annotations

from db import database as db

BENCHMARK_COMPLETION_STAGE = "benchmark_completion_required"

_BENCHMARK_BLOCKER_MARKERS = (
    "benchmark artifact",
    "benchmark_artifact_manifest.json is missing",
    "missing or not linked",
    "benchmark summary is missing",
    "per_method must contain at least two",
    "must include at least two methods",
    "at least two methods/baselines",
    "no metric",
    "metric missing",
)

_SOFT_BENCHMARK_GAP_MARKERS = (
    "full_benchmark_completed=false",
    "full_benchmark_completed is false",
    "full_benchmark_completed",
    "full benchmark policy",
    "benchmark coverage",
    "required benchmark coverage missing",
    "required baselines missing",
    "required baseline missing",
    "required model coverage missing",
    "model coverage",
    "mip-nerf",
    "required ablation",
    "ablation table",
    "seed",
    "num_seeds",
    "seed(s) found",
    "minimum_seeds",
    "baseline",
    "load_failures",
    "code repair",
    "proof repair",
)


def _is_benchmark_completion_blocker(blocker: str) -> bool:
    text = str(blocker or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in (_BENCHMARK_BLOCKER_MARKERS + _SOFT_BENCHMARK_GAP_MARKERS))


def benchmark_completion_blockers(bundle: dict | None) -> list[str]:
    """Return blockers that need a benchmark/harness completion run.

    Incomplete full-benchmark coverage, insufficient seeds, and missing
    baselines/model families should go back to benchmark completion before
    another manuscript attempt.
    """
    if not isinstance(bundle, dict):
        return []
    raw_blockers = bundle.get("submission_blockers")
    blockers: list[str] = []
    if isinstance(raw_blockers, list):
        blockers.extend(str(item).strip() for item in raw_blockers if str(item or "").strip())
    error = str(bundle.get("error") or "").strip()
    if error:
        blockers.append(error)
    return [blocker for blocker in blockers if _is_benchmark_completion_blocker(blocker)]


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
