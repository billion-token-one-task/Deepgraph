"""Requeue incomplete or low-quality experiment jobs under hardened policy."""

from __future__ import annotations

import json
from pathlib import Path

from config import (
    AUTO_RESEARCH_REQUEUE_INCOMPLETE,
    EXPERIMENT_REQUIRE_FULL_BENCHMARK,
    IDEA_DEDUP_BENCHMARK_FINGERPRINT,
    IDEA_DEDUP_ENABLED,
    MANUSCRIPT_ALLOW_NEGATIVE_RESULTS,
)
from db import database as db
from orchestrator.benchmark_completion import BENCHMARK_COMPLETION_STAGE
from orchestrator.idea_dedup import check_insight_duplicate, _full_benchmark_completed
from orchestrator.pipeline import log_event


def _load_json(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _latest_run(insight_id: int) -> dict | None:
    row = db.fetchone(
        """
        SELECT * FROM experiment_runs
        WHERE deep_insight_id=?
          AND status NOT IN ('superseded', 'reset', 'archived', 'cancelled')
        ORDER BY id DESC LIMIT 1
        """,
        (insight_id,),
    )
    return dict(row) if row else None


def run_needs_hardened_followup(insight_id: int, run: dict | None) -> tuple[bool, str, str]:
    """Return (needs_action, stage, reason). stage is hardened_requeue or benchmark_completion_required."""
    if not run:
        return True, "hardened_requeue", "no experiment run on record"

    status = str(run.get("status") or "").strip().lower()
    if status == "failed":
        return True, "hardened_requeue", run.get("error_message") or "experiment run failed"

    if status not in {"completed"}:
        return False, "", ""

    proxy = _load_json(run.get("proxy_config"), {})
    verdict = str(run.get("hypothesis_verdict") or "").strip().lower()
    baseline = run.get("baseline_metric_value")
    best = run.get("best_metric_value")

    if baseline is None and best is None:
        return True, "hardened_requeue", "experiment completed without metrics"

    if bool(proxy.get("smoke_test_only")) or not bool(proxy.get("formal_experiment")):
        return True, "hardened_requeue", "run is smoke-only or non-formal"

    if verdict == "confirmed" and _full_benchmark_completed(run):
        return False, "", ""

    if EXPERIMENT_REQUIRE_FULL_BENCHMARK and not _full_benchmark_completed(run):
        return (
            True,
            BENCHMARK_COMPLETION_STAGE,
            "full benchmark package incomplete under hardened policy",
        )

    if not MANUSCRIPT_ALLOW_NEGATIVE_RESULTS and verdict in {"refuted", "inconclusive"}:
        # Legitimate negative result after real benchmark: keep closed, block manuscript elsewhere.
        if _full_benchmark_completed(run):
            return False, "", ""
        return True, "hardened_requeue", f"verdict={verdict} before full benchmark completion"

    if verdict not in {"confirmed", "supported"} and not _full_benchmark_completed(run):
        return True, "hardened_requeue", f"verdict={verdict or 'unknown'} with incomplete benchmark evidence"

    return False, "", ""


def requeue_incomplete_experiments(*, limit: int = 20) -> dict:
    if not AUTO_RESEARCH_REQUEUE_INCOMPLETE:
        return {"requeued": 0, "blocked": 0, "skipped": 0, "disabled": True}

    rows = db.fetchall(
        """
        SELECT di.id AS insight_id,
               di.title,
               arj.status AS auto_status,
               arj.stage AS auto_stage
        FROM deep_insights di
        JOIN auto_research_jobs arj ON arj.deep_insight_id = di.id
        WHERE COALESCE(di.status, 'candidate') NOT IN ('exists')
          AND arj.status IN ('completed', 'failed')
          AND arj.stage NOT IN ('abandoned', 'hardened_requeue', ?, 'benchmark_completion_required')
        ORDER BY di.tier DESC, arj.updated_at ASC
        LIMIT ?
        """,
        (BENCHMARK_COMPLETION_STAGE, max(limit * 4, 40)),
    )

    requeued = 0
    blocked = 0
    skipped = 0
    actions: list[dict] = []

    for row in rows:
        if requeued + blocked >= limit:
            break
        insight_id = int(row["insight_id"])
        run = _latest_run(insight_id)
        needs, stage, reason = run_needs_hardened_followup(insight_id, run)
        if not needs:
            skipped += 1
            continue

        if IDEA_DEDUP_ENABLED and stage == "hardened_requeue":
            dup = check_insight_duplicate(
                insight_id,
                run=run,
                text_dedup=True,
                benchmark_dedup=IDEA_DEDUP_BENCHMARK_FINGERPRINT,
            )
            if dup:
                db.execute(
                    """
                    UPDATE auto_research_jobs
                    SET status='blocked',
                        stage='duplicate_idea',
                        last_note=?,
                        last_error=?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE deep_insight_id=?
                    """,
                    (dup["reason"], dup["reason"], insight_id),
                )
                blocked += 1
                actions.append({"insight_id": insight_id, "action": "blocked", "reason": dup["reason"]})
                continue

        run_id = int(run["id"]) if run and stage == BENCHMARK_COMPLETION_STAGE else None
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='queued',
                stage=?,
                experiment_run_id=?,
                assigned_worker=NULL,
                scheduler_priority=CASE
                    WHEN COALESCE(scheduler_priority, 0) < 2 THEN 2
                    ELSE scheduler_priority
                END,
                last_note=?,
                last_error=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=?
            """,
            (stage, run_id, reason, reason if stage == "hardened_requeue" else None, insight_id),
        )
        requeued += 1
        actions.append({"insight_id": insight_id, "action": "requeued", "stage": stage, "reason": reason})

    if requeued or blocked:
        db.commit()
        log_event(
            "auto_research",
            {
                "step": "hardened_requeue",
                "requeued": requeued,
                "blocked": blocked,
                "skipped": skipped,
                "actions": actions[:20],
            },
        )

    return {"requeued": requeued, "blocked": blocked, "skipped": skipped, "actions": actions}
