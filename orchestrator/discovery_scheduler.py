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

from config import (
    DISCOVERY_AUTO_TRIGGER_PAPERS,
    DISCOVERY_BULK_TIER1_CANDIDATES,
    DISCOVERY_BULK_TIER1_OVERLAPS,
    DISCOVERY_BULK_TIER1_PATTERNS,
    DISCOVERY_BULK_TIER2_LIMIT_NODES,
    DISCOVERY_BULK_TIER2_PLATEAUS,
    DISCOVERY_BULK_TIER2_PROBLEMS,
    DISCOVERY_TIER1_CANDIDATES,
    DISCOVERY_TIER2_PAPERS,
    DISCOVERY_TIER2_PROBLEMS,
)
from db import database as db
from orchestrator.pipeline import log_event

_discovery_thread = None
_discovery_lock = threading.Lock()


def _init_schema_v2():
    """Ensure v2 tables exist."""
    from pathlib import Path
    schema_path = Path(__file__).parent.parent / "db" / "schema_v2.sql"
    if schema_path.exists():
        conn = db.get_conn()
        conn.executescript(schema_path.read_text())
        conn.commit()


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


def schedule_discovery_if_ready():
    """Check if enough new data has accumulated to trigger discovery.

    Called after each pipeline batch. Non-blocking.
    """
    global _discovery_thread
    with _discovery_lock:
        if _discovery_thread and _discovery_thread.is_alive():
            return

    # Check if we have enough new papers since last discovery
    last_discovery = db.fetchone(
        "SELECT MAX(created_at) as last FROM deep_insights")
    last_ts = last_discovery["last"] if last_discovery and last_discovery["last"] else "2000-01-01"

    new_papers = db.fetchone(
        "SELECT COUNT(*) as c FROM papers WHERE status='reasoned' AND updated_at > ?",
        (last_ts,))

    if new_papers and new_papers["c"] >= DISCOVERY_AUTO_TRIGGER_PAPERS:
        print(f"[DISCOVERY] {new_papers['c']} new papers since last discovery (threshold={DISCOVERY_AUTO_TRIGGER_PAPERS}). Scheduling...", flush=True)

        def _run():
            try:
                run_full_discovery()
            except Exception as e:
                print(f"[DISCOVERY] Scheduled discovery failed: {e}", flush=True)
                print(traceback.format_exc(), flush=True)

        with _discovery_lock:
            _discovery_thread = threading.Thread(target=_run, daemon=True)
            _discovery_thread.start()
    else:
        count = new_papers["c"] if new_papers else 0
        print(f"[DISCOVERY] {count} new papers since last discovery (need 200). Skipping.", flush=True)


