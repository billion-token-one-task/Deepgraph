"""Orchestrator: continuous paper processing pipeline."""
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime

from config import PIPELINE_CONCURRENCY
from db import database as db
from db import evidence_graph as graph
from db import opportunity_engine as opp
from db import taxonomy as tax
from ingestion.arxiv_client import fetch_recent, search_papers
from ingestion.pdf_parser import get_paper_text
from agents.extraction_agent import extract_paper
from agents.reasoning_agent import detect_contradictions, detect_contradictions_batch, discover_matrix_gaps
from agents.taxonomy_expander import run_expansion, EXPANSION_THRESHOLD

logger = logging.getLogger(__name__)

# Global event log for SSE
_event_log: list[dict] = []
_event_lock = threading.Lock()


def get_events(since: int = 0) -> list[dict]:
    with _event_lock:
        return _event_log[since:]


def log_event(event_type: str, data: dict):
    with _event_lock:
        _event_log.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "seq": len(_event_log),
        })
        # Keep last 1000 events
        if len(_event_log) > 1000:
            _event_log.pop(0)


_insight_thread = None
_insight_lock = threading.Lock()

def _run_incremental_insights(papers_so_far: int):
    """Run insight discovery in a background thread so it doesn't block paper processing."""
    global _insight_thread
    with _insight_lock:
        if _insight_thread and _insight_thread.is_alive():
            print(f"[PIPELINE] Insight discovery still running, skipping this round", flush=True)
            return

    def _do_insights():
        from agents.insight_agent import discover_insights, store_insight
        nodes = db.fetchall(
            """SELECT t.id, t.name, COUNT(DISTINCT pt.paper_id) as pc
               FROM taxonomy_nodes t
               JOIN paper_taxonomy pt ON pt.node_id = t.id
               GROUP BY t.id
               HAVING pc >= 10
               ORDER BY pc DESC LIMIT 15"""
        )
        new_insights = 0
        for node in nodes:
            existing = db.fetchone(
                "SELECT COUNT(*) as c FROM insights WHERE node_id=? AND created_at > datetime('now', '-2 hours')",
                (node["id"],)
            )
            if existing and existing["c"] >= 2:
                continue
            try:
                print(f"[PIPELINE] Insight discovery: analyzing {node['id']} ({node['pc']}p)...", flush=True)
                log_event("step", {"step": "incremental_insight", "node_id": node["id"],
                                   "papers_so_far": papers_so_far})
                insights, tokens = discover_insights(node["id"])
                for ins in insights:
                    store_insight(ins)
                    new_insights += 1
                    print(f"[PIPELINE] New insight: [{ins['type']}] {ins['title'][:80]}", flush=True)
                    log_event("insight", {
                        "node_id": ins["node_id"],
                        "type": ins["type"],
                        "title": ins["title"],
                        "novelty": ins.get("novelty_score", 0),
                        "feasibility": ins.get("feasibility_score", 0),
                    })
            except Exception as e:
                logger.error("Insight discovery failed for %s: %s", node["id"], e)
        if new_insights:
            print(f"[PIPELINE] Incremental insights: {new_insights} new insights at {papers_so_far} papers", flush=True)
            log_event("incremental_insights_done", {"new_insights": new_insights, "papers_so_far": papers_so_far})
        else:
            print(f"[PIPELINE] Insight discovery: no new insights (all deduped or skipped)", flush=True)

    with _insight_lock:
        _insight_thread = threading.Thread(target=_do_insights, daemon=True)
        _insight_thread.start()


def ingest_papers(max_papers: int = 100) -> int:
    """Fetch and store papers from arXiv, paginating to find enough new ones."""
    log_event("ingest_start", {"max_papers": max_papers})
    count = 0
    total_fetched = 0
    start = 0
    batch_size = 200
    consecutive_empty = 0  # batches with zero new papers

    while count < max_papers:
        papers = search_papers(start=start, max_results=batch_size)
        if not papers:
            break
        total_fetched += len(papers)
        batch_new = 0
        for p in papers:
            existing = db.fetchone("SELECT id FROM papers WHERE id=?", (p["id"],))
            if not existing:
                db.insert_paper(p)
                count += 1
                batch_new += 1
                if count >= max_papers:
                    break
        start += len(papers)
        if batch_new == 0:
            consecutive_empty += 1
            # After 3 empty batches, jump ahead to skip the known region
            if consecutive_empty >= 3:
                start += batch_size * 5
                log_event("ingest_progress", {"new_so_far": count, "skipping_to": start})
            # Give up after jumping 5 times with no results
            if consecutive_empty >= 15:
                break
        else:
            consecutive_empty = 0

        log_event("ingest_progress", {"new_so_far": count, "fetched_so_far": total_fetched, "start": start})

    log_event("ingest_done", {"new_papers": count, "total_fetched": total_fetched})
    return count


def process_single_paper(paper_id: str) -> dict:
    """Run full pipeline on one paper: extract -> classify -> store results -> check contradictions."""
    paper = db.fetchone("SELECT * FROM papers WHERE id=?", (paper_id,))
    if not paper:
        return {"error": "Paper not found"}

    result = {"paper_id": paper_id, "claims": 0, "results": 0,
              "taxonomy_nodes": [], "graph_entities": 0, "graph_relations": 0,
              "contradictions": 0, "tokens": 0}

    try:
        # Step 1: Get full text
        log_event("step", {"paper_id": paper_id, "title": paper["title"], "step": "downloading"})
        text = paper["full_text"]
        if not text or len(text) < 100:
            text = get_paper_text(paper_id, paper.get("pdf_url", ""), paper.get("abstract", ""))
            db.update_paper_text(paper_id, text)

        if len(text) < 100:
            db.update_paper_status(paper_id, "error", "No text available")
            return result

        # Step 2: Extract claims, results, taxonomy classification
        log_event("step", {"paper_id": paper_id, "title": paper["title"], "step": "extracting"})
        extraction, tokens1 = extract_paper(paper_id, paper["title"], text)
        result["tokens"] += tokens1

        # Step 2a: Store taxonomy classification
        for tn in extraction.get("taxonomy_nodes", []):
            node_id = tn.get("node_id", "")
            confidence = tn.get("confidence", 1.0)
            # Verify node exists
            if tax.get_node(node_id):
                tax.assign_paper_to_node(paper_id, node_id, confidence)
                if node_id not in result["taxonomy_nodes"]:
                    result["taxonomy_nodes"].append(node_id)

        # Step 2a.5: Store plain-language paper overview for non-experts
        paper_overview = extraction.get("paper_overview")
        if isinstance(paper_overview, dict) and paper_overview:
            db.upsert_paper_insight(paper_id, paper_overview)

        # Step 2b: Store claims (backward compatible)
        claims = extraction.get("claims", [])
        for c in claims:
            c["paper_id"] = paper_id
            claim_id = db.insert_claim(c)
            c["_id"] = claim_id
        result["claims"] = len(claims)

        # Step 2c: Store methods
        for m in extraction.get("methods", []):
            m["first_paper_id"] = paper_id
            db.insert_method(m)

        # Step 2d: Store structured results (method, dataset, metric, value)
        node_ids = result["taxonomy_nodes"] or [None]
        for r in extraction.get("results", []):
            r["paper_id"] = paper_id
            r["node_id"] = node_ids[0] if node_ids[0] else None
            result_id = tax.insert_result(r)
            assigned_nodes = node_ids if node_ids[0] else []
            for node_id in assigned_nodes:
                tax.assign_result_to_node(result_id, node_id, commit=False)
        if extraction.get("results"):
            db.commit()
        result["results"] = len(extraction.get("results", []))

        # Step 2e: Store entity-relation-evidence graph (merge LLM graph with structured records)
        graph_payload = extraction.get("knowledge_graph") if isinstance(extraction.get("knowledge_graph"), dict) else {}
        structured_graph = graph.build_structured_graph_payload_from_records(
            extraction.get("methods", []),
            extraction.get("results", []),
            extraction.get("claims", []),
            extraction.get("paper_overview"),
        )
        merged_graph = graph.merge_graph_payloads(graph_payload, structured_graph)
        if merged_graph["entities"] or merged_graph["relations"]:
            graph_stats = graph.store_paper_graph(paper_id, result["taxonomy_nodes"], merged_graph)
            result["graph_entities"] = graph_stats["entities"]
            result["graph_relations"] = graph_stats["relations"]

        db.update_paper_status(paper_id, "extracted", token_cost=tokens1)

        # Step 3: Check for contradictions (batch — 1 LLM call per paper)
        log_event("step", {"paper_id": paper_id, "title": paper["title"], "step": "reasoning"})
        total_contradictions = 0
        tokens2 = 0
        contras, tokens2 = detect_contradictions_batch(claims)
        for contra in contras:
            new_claim = contra.pop("_new_claim", None)
            if new_claim:
                contra["claim_a_id"] = new_claim.get("_id")
            contra["claim_b_id"] = contra.get("existing_claim_id")
            db.insert_contradiction(contra)
            total_contradictions += 1
            log_event("contradiction", {
                "paper_id": paper_id,
                "description": contra.get("description", ""),
                "hypothesis": contra.get("hypothesis", ""),
            })

        result["contradictions"] = total_contradictions
        result["tokens"] += tokens2
        db.update_paper_status(paper_id, "reasoned", token_cost=tokens1 + tokens2)

        log_event("paper_done", {
            "paper_id": paper_id,
            "title": paper["title"],
            "claims": result["claims"],
            "results": result["results"],
            "taxonomy_nodes": result["taxonomy_nodes"],
            "graph_entities": result["graph_entities"],
            "graph_relations": result["graph_relations"],
            "contradictions": result["contradictions"],
            "tokens": result["tokens"],
        })

    except Exception as e:
        db.update_paper_status(paper_id, "error", str(e))
        log_event("error", {"paper_id": paper_id, "error": str(e),
                            "traceback": traceback.format_exc()})
        result["error"] = str(e)

    return result


def run_continuous(max_papers: int = 0):
    """Run the pipeline continuously."""
    print(f"[PIPELINE] Starting with max_papers={max_papers}", flush=True)
    log_event("pipeline_start", {"max_papers": max_papers})

    # Step 0: Recover any papers stuck in 'processing' (from crashed runs)
    stuck = db.fetchall("SELECT id FROM papers WHERE status='processing'")
    if stuck:
        for s in stuck:
            db.execute("UPDATE papers SET status='ingested' WHERE id=?", (s["id"],))
        db.commit()
        log_event("recovery", {"recovered_papers": len(stuck)})
        logger.info("Recovered %d stuck papers back to 'ingested'", len(stuck))

    # Step 1: Ingest (skip if we already have enough ingested papers)
    ingested_available = db.fetchone("SELECT COUNT(*) as c FROM papers WHERE status='ingested'")["c"]
    if ingested_available >= (max_papers or 100):
        print(f"[PIPELINE] Step 1: Skipping ingest ({ingested_available} papers already queued)", flush=True)
    else:
        try:
            print("[PIPELINE] Step 1: Ingesting papers...", flush=True)
            ingest_papers(max_papers=max_papers or 100)
            print("[PIPELINE] Step 1: Ingest done.", flush=True)
        except Exception as e:
            print(f"[PIPELINE] Ingest failed: {e}", flush=True)
            logger.error("Ingest failed: %s", e)
            log_event("error", {"step": "ingest", "error": str(e),
                                "traceback": traceback.format_exc()})

    # Step 2: Process papers in parallel
    ingested_count = db.fetchone("SELECT COUNT(*) as c FROM papers WHERE status='ingested'")
    print(f"[PIPELINE] Step 2: Processing papers ({ingested_count['c']} ingested available)...", flush=True)
    processed = 0
    workers = PIPELINE_CONCURRENCY
    summary_nodes: set[str] = set()
    _summary_lock = threading.Lock()

    # Continuous processing: submit papers as slots free up (no batch blocking)
    target = max_papers if max_papers else ingested_count['c']
    submitted = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict = {}

        def _submit_next(n: int):
            """Submit up to n papers to the pool."""
            nonlocal submitted
            room = n
            if max_papers:
                room = min(room, max_papers - submitted)
            if room <= 0:
                return
            papers = db.fetchall(
                "SELECT id FROM papers WHERE status='ingested' ORDER BY published_date DESC LIMIT ?",
                (room,)
            )
            for p in papers:
                db.execute("UPDATE papers SET status='processing' WHERE id=? AND status='ingested'", (p["id"],))
            db.commit()
            for p in papers:
                try:
                    f = pool.submit(process_single_paper, p["id"])
                    futures[f] = p["id"]
                    submitted += 1
                except Exception as e:
                    # Reset paper status if submit fails
                    db.execute("UPDATE papers SET status='ingested' WHERE id=?", (p["id"],))
                    db.commit()
                    print(f"[PIPELINE] Failed to submit {p['id']}: {e}", flush=True)

        # Start with a small warmup: submit 3 papers to probe providers,
        # then wait until at least 1 completes before filling up
        _submit_next(3)
        import time as _time
        print("[PIPELINE] Warmup: waiting for first paper to complete...", flush=True)
        warmup_timeout = 180  # max 3 min warmup wait
        warmup_start = _time.time()
        while _time.time() - warmup_start < warmup_timeout:
            for f in list(futures.keys()):
                if f.done():
                    pid = futures.pop(f)
                    try:
                        paper_result = f.result(timeout=1)
                        with _summary_lock:
                            for node_id in paper_result.get("taxonomy_nodes", []):
                                summary_nodes.update(tax.get_ancestor_ids(node_id))
                    except Exception as e:
                        print(f"[PIPELINE] Warmup error for {pid}: {e}", flush=True)
                    processed += 1
                    break
            if processed > 0:
                break
            _time.sleep(2)
        elapsed = _time.time() - warmup_start
        print(f"[PIPELINE] Warmup done in {elapsed:.0f}s, filling pool to {workers} workers", flush=True)
        _submit_next(workers - len(futures))

        while futures:
            done_futures = []
            for future in list(futures.keys()):
                if future.done():
                    done_futures.append(future)

            for future in done_futures:
                pid = futures.pop(future)
                try:
                    paper_result = future.result(timeout=1)
                    with _summary_lock:
                        for node_id in paper_result.get("taxonomy_nodes", []):
                            summary_nodes.update(tax.get_ancestor_ids(node_id))
                except Exception as e:
                    print(f"[PIPELINE] Error processing {pid}: {e}", flush=True)
                    log_event("error", {"paper_id": pid, "error": str(e)})
                processed += 1

            # Fill freed slots and check milestones
            if done_futures:
                old_processed = processed - len(done_futures)
                _submit_next(len(done_futures))
                if processed % 10 == 0:
                    print(f"[PIPELINE] Progress: {processed}/{target} processed, {len(futures)} active", flush=True)

                # Periodic insight discovery every ~100 papers
                if processed // 100 > old_processed // 100 and processed > 0:
                    try:
                        print(f"[PIPELINE] Running incremental insight discovery at {processed} papers...", flush=True)
                        _run_incremental_insights(processed)
                    except Exception as e:
                        logger.error("Incremental insight discovery failed: %s", e)
                        log_event("error", {"step": "incremental_insights", "error": str(e)})
            else:
                _time.sleep(1)

    # Step 2.5: Run abstraction to extract cross-domain patterns
    from agents.abstraction_agent import run_abstraction_for_nodes, run_bridge_discovery
    log_event("step", {"step": "pattern_abstraction"})
    try:
        num_patterns, abstraction_tokens = run_abstraction_for_nodes(min_claims=15)
        log_event("abstraction_done", {"patterns": num_patterns, "tokens": abstraction_tokens})

        if num_patterns >= 6:
            log_event("step", {"step": "bridge_discovery"})
            num_bridges, bridge_tokens = run_bridge_discovery()
            log_event("bridge_done", {"bridges": num_bridges, "tokens": bridge_tokens})
    except Exception as e:
        logger.error("Abstraction/bridge failed: %s", e)
        log_event("error", {"step": "abstraction", "error": str(e)})

    # Step 3: Run deep insight discovery (replaces old matrix gap finder)
    from agents.insight_agent import discover_insights, store_insight
    insight_nodes = db.fetchall(
        """SELECT t.id, t.name, COUNT(DISTINCT pt.paper_id) as pc
           FROM taxonomy_nodes t
           JOIN paper_taxonomy pt ON pt.node_id = t.id
           GROUP BY t.id
           HAVING pc >= 10
           ORDER BY pc DESC LIMIT 15"""
    )
    total_insight_tokens = 0
    for node in insight_nodes:
        # Skip if already has recent insights
        existing = db.fetchone(
            "SELECT COUNT(*) as c FROM insights WHERE node_id=? AND created_at > datetime('now', '-1 day')",
            (node["id"],)
        )
        if existing and existing["c"] >= 2:
            continue

        log_event("step", {"step": "insight_discovery", "node_id": node["id"]})
        insights, tokens = discover_insights(node["id"])
        total_insight_tokens += tokens
        for ins in insights:
            store_insight(ins)
            log_event("insight", {
                "node_id": ins["node_id"],
                "type": ins["type"],
                "title": ins["title"],
                "novelty": ins.get("novelty_score", 0),
                "feasibility": ins.get("feasibility_score", 0),
            })

    # Step 3b: Auto-expand taxonomy leaf nodes that have accumulated enough papers
    log_event("step", {"step": "taxonomy_expansion", "threshold": EXPANSION_THRESHOLD})
    try:
        expansion_results = run_expansion(min_papers=EXPANSION_THRESHOLD)
        total_expansion_tokens = 0
        for exp in expansion_results:
            total_expansion_tokens += exp.get("tokens", 0)
            if exp.get("new_children"):
                log_event("taxonomy_expanded", {
                    "node_id": exp["node_id"],
                    "new_children": exp["new_children"],
                    "papers_reassigned": exp["papers_reassigned"],
                    "tokens": exp.get("tokens", 0),
                })
                # Add the new children and their parent to summary_nodes
                with _summary_lock:
                    summary_nodes.add(exp["node_id"])
                    for child_id in exp["new_children"]:
                        summary_nodes.add(child_id)
            elif exp.get("error"):
                logger.info("Skipped expansion for %s: %s", exp["node_id"], exp["error"])
        if expansion_results:
            log_event("taxonomy_expansion_done", {
                "nodes_checked": len(expansion_results),
                "nodes_expanded": sum(1 for e in expansion_results if e.get("new_children")),
                "total_tokens": total_expansion_tokens,
            })
    except Exception as e:
        logger.error("Taxonomy expansion failed: %s", e)
        log_event("error", {"step": "taxonomy_expansion", "error": str(e)})

    # Step 4: Generate node-level graph summaries
    for node_id in sorted(summary_nodes):
        log_event("step", {"step": "graph_summary", "node_id": node_id})
        graph_summary = graph.ensure_node_graph_summary(node_id, force=True)
        if graph_summary:
            log_event("graph_summary", {
                "node_id": node_id,
                "entity_count": graph_summary.get("entity_count", 0),
                "relation_count": graph_summary.get("relation_count", 0),
            })

    # Step 5: Generate richer opportunity signals
    for node_id in sorted(summary_nodes):
        log_event("step", {"step": "opportunity_scoring", "node_id": node_id})
        opportunities = opp.ensure_node_opportunities(node_id, force=True)
        if opportunities:
            top = opportunities[0]
            log_event("opportunity", {
                "node_id": node_id,
                "title": top["title"],
                "value": top.get("value_score", 0),
            })

    # Step 6: Generate plain-language node summaries for exploration
    for node_id in sorted(summary_nodes):
        log_event("step", {"step": "node_summary", "node_id": node_id})
        summary = tax.ensure_node_summary(node_id, force=True)
        if summary:
            log_event("summary", {
                "node_id": node_id,
                "overview": (summary.get("overview") or "")[:180],
            })

    # Step 7: Signal harvesting + discovery scheduling (non-blocking)
    try:
        from orchestrator.discovery_scheduler import harvest_signals, schedule_discovery_if_ready
        log_event("step", {"step": "signal_harvest"})
        harvest_signals()
        schedule_discovery_if_ready()
    except Exception as e:
        logger.error("Signal harvest / discovery scheduling failed: %s", e)
        log_event("error", {"step": "signal_harvest", "error": str(e)})

    log_event("pipeline_done", {"papers_processed": processed, "stats": get_stats_dict()})
    return processed


def get_stats_dict() -> dict:
    """Get comprehensive stats."""
    base = db.get_stats()
    base["results_total"] = db.fetchone("SELECT COUNT(*) as c FROM results")["c"]
    base["taxonomy_assignments"] = db.fetchone("SELECT COUNT(*) as c FROM paper_taxonomy")["c"]
    try:
        base["matrix_gaps_total"] = db.fetchone("SELECT COUNT(*) as c FROM matrix_gaps")["c"]
    except Exception:
        base["matrix_gaps_total"] = 0
    base["taxonomy_nodes_total"] = db.fetchone("SELECT COUNT(*) as c FROM taxonomy_nodes")["c"]
    return base
