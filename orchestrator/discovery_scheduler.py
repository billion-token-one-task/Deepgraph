"""Discovery Scheduler: background orchestration of Tier 1/2 insight generation.

Integrates with the main pipeline without blocking paper processing.
- Signal harvesting: after every pipeline batch (SQL only, fast)
- Tier 1 paradigm discovery: on-demand or daily schedule
- Tier 2 paper ideas: on-demand or after enough new papers
"""
import json
import threading
import time
import traceback
from datetime import datetime

from agents.llm_client import is_llm_auth_error, is_llm_provider_unavailable_error
from config import (
    DISCOVERY_AUTO_TRIGGER_PAPERS,
    DISCOVERY_MIN_TIER2_BACKLOG,
    DISCOVERY_BULK_TIER1_CANDIDATES,
    DISCOVERY_BULK_TIER1_OVERLAPS,
    DISCOVERY_BULK_TIER1_PATTERNS,
    DISCOVERY_BULK_TIER2_LIMIT_NODES,
    DISCOVERY_BULK_TIER2_PLATEAUS,
    DISCOVERY_BULK_TIER2_PROBLEMS,
    DISCOVERY_TIER1_CANDIDATES,
    DISCOVERY_TIER2_PAPERS,
    DISCOVERY_TIER2_PROBLEMS,
    PIPELINE_EVENT_POLL_SECONDS,
)
from db import database as db
from orchestrator.pipeline import log_event

_discovery_thread = None
_discovery_lock = threading.Lock()
_tier2_thread = None
_tier2_lock = threading.Lock()
_stop_event = threading.Event()
DISCOVERY_CONSUMER = "discovery_scheduler"
PARALLEL_TIER2_MIN_INTERVAL_SECONDS = 90
_last_parallel_tier2_at = 0.0


def _llm_temporarily_unavailable(exc: Exception) -> bool:
    return is_llm_auth_error(exc) or is_llm_provider_unavailable_error(exc)


def _init_schema_v2():
    """Ensure v2 tables exist."""
    db.init_db()


def harvest_signals() -> dict:
    """Run signal harvesting (SQL only, no LLM cost)."""
    _init_schema_v2()
    from agents.signal_harvester import harvest_all
    log_event("discovery", {"step": "signal_harvest_start"})
    stats = harvest_all()
    log_event("discovery", {"step": "signal_harvest_done", **stats})
    return stats


def run_tier1_discovery(
    max_candidates: int | None = None,
    *,
    bulk: bool = False,
) -> list[dict]:
    """Run Tier 1 (Paradigm) discovery. Returns stored insight IDs."""
    _init_schema_v2()
    from agents.paradigm_agent import discover_paradigm_insights, store_deep_insight

    if max_candidates is None:
        max_candidates = (
            DISCOVERY_BULK_TIER1_CANDIDATES if bulk else DISCOVERY_TIER1_CANDIDATES
        )
    top_ov = DISCOVERY_BULK_TIER1_OVERLAPS if bulk else 20
    top_pat = DISCOVERY_BULK_TIER1_PATTERNS if bulk else 15

    log_event(
        "discovery",
        {
            "step": "tier1_start",
            "max_candidates": max_candidates,
            "bulk": bulk,
            "signal_overlaps": top_ov,
            "signal_patterns": top_pat,
        },
    )
    print("[DISCOVERY] Starting Tier 1 (Paradigm) discovery...", flush=True)

    try:
        insights = discover_paradigm_insights(
            max_candidates=max_candidates,
            tier1_top_overlaps=top_ov,
            tier1_top_patterns=top_pat,
        )
        stored = []
        for ins in insights:
            insight_id = store_deep_insight(ins)
            if not insight_id:
                log_event(
                    "warning",
                    {
                        "step": "tier1_skip_incomplete",
                        "title": ins.get("title"),
                    },
                )
                continue
            stored.append({"id": insight_id, "title": ins["title"],
                           "adversarial_score": ins.get("adversarial_score", 0)})
            log_event("deep_insight", {
                "tier": 1,
                "id": insight_id,
                "title": ins["title"],
                "adversarial_score": ins.get("adversarial_score", 0),
            })
        log_event("discovery", {"step": "tier1_done", "count": len(stored)})
        print(f"[DISCOVERY] Tier 1 done: {len(stored)} paradigm insights stored", flush=True)
        return stored
    except Exception as e:
        if _llm_temporarily_unavailable(e):
            print(f"[DISCOVERY] Tier 1 skipped: LLM unavailable ({e})", flush=True)
            log_event("warning", {"step": "tier1_discovery", "error": str(e), "suppressed": True})
            return []
        print(f"[DISCOVERY] Tier 1 failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        log_event("error", {"step": "tier1_discovery", "error": str(e)})
        return []


def run_tier2_discovery(
    max_problems: int | None = None,
    max_papers: int | None = None,
    *,
    bulk: bool = False,
) -> list[dict]:
    """Run Tier 2 (Paper Ideas) discovery. Returns stored insight IDs.

    In bulk mode, expands every sharpened problem (max_papers follows max_problems).
    """
    _init_schema_v2()
    from agents.paradigm_agent import store_deep_insight
    from agents.paper_idea_agent import discover_paper_ideas

    if max_problems is None:
        max_problems = DISCOVERY_BULK_TIER2_PROBLEMS if bulk else DISCOVERY_TIER2_PROBLEMS
    if bulk:
        plateaus = DISCOVERY_BULK_TIER2_PLATEAUS
        lim_nodes = DISCOVERY_BULK_TIER2_LIMIT_NODES
        mpapers: int | None = None
    else:
        plateaus = 20
        lim_nodes = 15
        mpapers = max_papers if max_papers is not None else DISCOVERY_TIER2_PAPERS

    log_event(
        "discovery",
        {"step": "tier2_start", "max_problems": max_problems, "bulk": bulk},
    )
    print("[DISCOVERY] Starting Tier 2 (Paper Ideas) discovery...", flush=True)

    try:
        insights = discover_paper_ideas(
            max_problems=max_problems,
            max_papers=mpapers,
            tier2_plateau_limit=plateaus,
            tier2_limitation_nodes=lim_nodes,
        )
        stored = []
        for ins in insights:
            insight_id = store_deep_insight(ins)
            if not insight_id:
                log_event(
                    "warning",
                    {
                        "step": "tier2_skip_incomplete",
                        "title": ins.get("title"),
                    },
                )
                continue
            method = {}
            try:
                method = json.loads(ins.get("proposed_method", "{}"))
            except Exception:
                pass
            stored.append({"id": insight_id, "title": ins["title"],
                           "method_name": method.get("name", "")})
            log_event("deep_insight", {
                "tier": 2,
                "id": insight_id,
                "title": ins["title"],
                "method_name": method.get("name", ""),
            })
        log_event("discovery", {"step": "tier2_done", "count": len(stored)})
        print(f"[DISCOVERY] Tier 2 done: {len(stored)} paper ideas stored", flush=True)
        return stored
    except Exception as e:
        if _llm_temporarily_unavailable(e):
            print(f"[DISCOVERY] Tier 2 skipped: LLM unavailable ({e})", flush=True)
            log_event("warning", {"step": "tier2_discovery", "error": str(e), "suppressed": True})
            return []
        print(f"[DISCOVERY] Tier 2 failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        log_event("error", {"step": "tier2_discovery", "error": str(e)})
        return []


def run_full_discovery(
    tier1_candidates: int | None = None,
    tier2_problems: int | None = None,
    tier2_papers: int | None = None,
    *,
    bulk: bool = False,
) -> dict:
    """Run the full discovery pipeline: harvest → Tier 1 → Tier 2."""
    results = {"started_at": datetime.utcnow().isoformat(), "bulk": bulk}

    # Step 1: Harvest signals
    results["signals"] = harvest_signals()

    # Step 2: Tier 1
    results["tier1"] = run_tier1_discovery(max_candidates=tier1_candidates, bulk=bulk)

    # Step 3: Tier 2
    results["tier2"] = run_tier2_discovery(
        max_problems=tier2_problems,
        max_papers=tier2_papers,
        bulk=bulk,
    )

    results["completed_at"] = datetime.utcnow().isoformat()
    return results


def run_bulk_deep_insights() -> dict:
    """One-shot wide harvest + max Tier1 formalizations + Tier2 for all problems."""
    return run_full_discovery(bulk=True)


def _recent_node_insight_count(node_id: str, hours: int = 2) -> int:
    row = db.fetchone(
        f"SELECT COUNT(*) AS c FROM insights WHERE node_id=? AND {db.sql_created_after_hours(hours)}",
        (node_id,),
    )
    return int(row["c"]) if row else 0


def _eligible_tier2_backlog() -> int:
    row = db.fetchone(
        """
        SELECT COUNT(*) AS c
        FROM deep_insights di
        LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id = di.id
        WHERE di.tier = 2
          AND COALESCE(di.status, 'candidate') NOT IN ('exists')
          AND (
            arj.status IS NULL
            OR arj.status IN ('queued', 'eligible', 'failed', 'queued_cpu', 'queued_gpu')
            OR (arj.status='blocked' AND arj.cpu_eligible=1)
          )
        """
    )
    return int(row["c"]) if row else 0


def _warm_tier2_backlog() -> int:
    row = db.fetchone(
        """
        SELECT COUNT(*) AS c
        FROM deep_insights di
        LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id = di.id
        WHERE di.tier = 2
          AND COALESCE(di.status, 'candidate') NOT IN ('exists')
          AND (
            arj.status IS NULL
            OR arj.status IN (
                'queued',
                'eligible',
                'failed',
                'queued_cpu',
                'queued_gpu',
                'verifying',
                'researching',
                'review_pending',
                'running_experiment',
                'running_cpu',
                'running_gpu'
            )
            OR (
                arj.status='blocked'
                AND (
                    arj.cpu_eligible=1
                    OR arj.stage IN ('verification_input_missing', 'research_input_missing')
                )
            )
          )
        """
    )
    return int(row["c"]) if row else 0


def _reasoned_paper_count() -> int:
    row = db.fetchone("SELECT COUNT(*) AS c FROM papers WHERE status='reasoned'")
    return int(row["c"]) if row else 0


def _run_parallel_tier2_discovery() -> None:
    try:
        target_backlog = max(1, DISCOVERY_MIN_TIER2_BACKLOG)
        deficit = max(0, target_backlog - _warm_tier2_backlog())
        if deficit <= 0:
            return
        harvest_signals()
        stored = run_tier2_discovery(max_problems=deficit, max_papers=DISCOVERY_TIER2_PAPERS)
        log_event(
            "discovery",
            {
                "step": "parallel_tier2_done",
                "count": len(stored),
                "target_backlog": target_backlog,
                "requested_problems": deficit,
            },
        )
    except Exception as exc:
        print(f"[DISCOVERY] Parallel Tier 2 failed: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        log_event("error", {"step": "parallel_tier2", "error": str(exc)})


def _maybe_launch_parallel_tier2_discovery(trigger: str) -> dict:
    global _tier2_thread, _last_parallel_tier2_at
    warm_backlog = _warm_tier2_backlog()
    target_backlog = max(1, DISCOVERY_MIN_TIER2_BACKLOG)
    if warm_backlog >= target_backlog:
        return {"status": "backlog_ready", "warm_backlog": warm_backlog, "target_backlog": target_backlog}
    if _reasoned_paper_count() < max(5, DISCOVERY_TIER2_PAPERS):
        return {"status": "insufficient_reasoned_papers", "warm_backlog": warm_backlog, "target_backlog": target_backlog}
    now = time.time()
    with _tier2_lock:
        if _tier2_thread and _tier2_thread.is_alive():
            return {"status": "already_running", "warm_backlog": warm_backlog, "target_backlog": target_backlog}
        if now - _last_parallel_tier2_at < PARALLEL_TIER2_MIN_INTERVAL_SECONDS:
            return {"status": "cooldown", "warm_backlog": warm_backlog, "target_backlog": target_backlog}
        _last_parallel_tier2_at = now
        _tier2_thread = threading.Thread(
            target=_run_parallel_tier2_discovery,
            daemon=True,
            name="deepgraph-parallel-tier2",
        )
        _tier2_thread.start()
    log_event(
        "discovery",
        {
            "step": "parallel_tier2_started",
            "trigger": trigger,
            "warm_backlog": warm_backlog,
            "target_backlog": target_backlog,
        },
    )
    return {"status": "started", "warm_backlog": warm_backlog, "target_backlog": target_backlog}


def _refresh_node_outputs(node_id: str) -> dict:
    from agents.insight_agent import discover_insights, store_insight
    from db import evidence_graph as graph
    from db import opportunity_engine as opp
    from db import taxonomy as tax

    log_event("discovery", {"step": "node_refresh", "node_id": node_id})
    graph_summary = graph.ensure_node_graph_summary(node_id, force=True)
    opportunities = opp.ensure_node_opportunities(node_id, force=True)
    summary = tax.ensure_node_summary(node_id, force=True)

    stored_ids: list[int] = []
    if _recent_node_insight_count(node_id) < 2:
        insights, _tokens = discover_insights(node_id)
        for ins in insights:
            insight_id = store_insight(ins)
            stored_ids.append(insight_id)
            db.emit_pipeline_event(
                "deep_insight_created",
                {"insight_id": insight_id, "node_id": node_id, "title": ins.get("title")},
                entity_type="deep_insight",
                entity_id=str(insight_id),
            )
            log_event("deep_insight", {"tier": 0, "id": insight_id, "title": ins.get("title"), "node_id": node_id})
    return {
        "node_id": node_id,
        "graph_summary": bool(graph_summary),
        "opportunity_count": len(opportunities or []),
        "summary_ready": bool(summary),
        "insights_created": stored_ids,
    }


def consume_pipeline_events_once(limit: int = 100) -> dict:
    """Consume node/paper events and refresh only the touched areas."""
    _init_schema_v2()
    events = db.fetch_pipeline_events(
        DISCOVERY_CONSUMER,
        limit=limit,
        event_types=["node_touched", "paper_reasoned"],
    )
    if not events:
        return {"events": 0, "nodes_refreshed": 0}

    last_event_id = 0
    refreshed: dict[str, dict] = {}
    for event in events:
        last_event_id = int(event["id"])
        payload = db._load_json(event.get("payload"), {})
        if event.get("event_type") == "node_touched":
            node_id = str(payload.get("node_id") or "")
            if node_id and node_id not in refreshed:
                refreshed[node_id] = _refresh_node_outputs(node_id)
        elif event.get("event_type") == "paper_reasoned":
            log_event("discovery", {"step": "paper_reasoned_seen", "paper_id": payload.get("paper_id")})
    db.ack_pipeline_events(DISCOVERY_CONSUMER, last_event_id)
    tier2_status = _maybe_launch_parallel_tier2_discovery("pipeline_events")
    return {
        "events": len(events),
        "nodes_refreshed": len(refreshed),
        "details": list(refreshed.values()),
        "tier2": tier2_status,
    }


def _event_loop() -> None:
    while not _stop_event.is_set():
        try:
            stats = consume_pipeline_events_once(limit=100)
            if not stats.get("events"):
                _stop_event.wait(max(1, PIPELINE_EVENT_POLL_SECONDS))
        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass
            print(f"[DISCOVERY] Event loop failed: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            _stop_event.wait(max(1, PIPELINE_EVENT_POLL_SECONDS))


def schedule_discovery_if_ready():
    """Ensure the event-driven discovery consumer is running."""
    global _discovery_thread
    with _discovery_lock:
        if _discovery_thread and _discovery_thread.is_alive():
            return
        _stop_event.clear()
        _discovery_thread = threading.Thread(target=_event_loop, daemon=True, name="deepgraph-discovery-events")
        _discovery_thread.start()
