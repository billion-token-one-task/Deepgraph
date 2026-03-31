"""Flask web application for DeepGraph dashboard."""
import json
import threading
import time
import traceback
from flask import Flask, render_template, jsonify, Response, request
from config import APP_NAME, APP_SUBTITLE, PROFILE, ROOT_NODE_ID
from db import database as db
from db import evidence_graph as graph
from db import opportunity_engine as opp
from db import taxonomy as tax
from orchestrator.pipeline import get_events, run_continuous, log_event, get_stats_dict
from agents.taxonomy_expander import run_expansion

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

_pipeline_running = False
_pipeline_lock = threading.Lock()


@app.route("/")
def index():
    return render_template(
        "index.html",
        app_name=APP_NAME,
        subtitle=APP_SUBTITLE,
        root_node_id=ROOT_NODE_ID,
        profile=PROFILE,
    )


@app.route("/api/meta")
def api_meta():
    return jsonify({
        "app_name": APP_NAME,
        "subtitle": APP_SUBTITLE,
        "root_node_id": ROOT_NODE_ID,
        "profile": PROFILE,
    })


# ── Stats ──────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats_dict())


@app.route("/api/providers")
def api_providers():
    """Get LLM provider stats (round-robin load balancing)."""
    from agents.llm_client import get_provider_stats
    return jsonify(get_provider_stats())


@app.route("/api/processing")
def api_processing():
    """Get papers currently being processed + recently completed (last 15s)."""
    rows = db.fetchall(
        """SELECT id, title, status FROM papers
           WHERE status IN ('processing', 'extracted')
              OR (status IN ('reasoned', 'error') AND updated_at > datetime('now', '-15 seconds'))
           ORDER BY CASE status WHEN 'processing' THEN 0 WHEN 'extracted' THEN 1 ELSE 2 END, updated_at DESC
           LIMIT 30"""
    )
    processing_count = db.fetchone("SELECT COUNT(*) as c FROM papers WHERE status='processing'")["c"]
    with _pipeline_lock:
        is_running = _pipeline_running or processing_count > 0
    return jsonify({"papers": rows, "pipeline_running": is_running})


# ── Taxonomy Navigation ───────────────────────────────────────────

@app.route("/api/taxonomy")
def api_taxonomy():
    """Return the full taxonomy tree as a flat list."""
    return jsonify(tax.get_taxonomy_flat())


@app.route("/api/taxonomy/<node_id>")
def api_taxonomy_node(node_id):
    """Get a node, its children (with counts), breadcrumb path, papers, and matrix."""
    node = tax.get_node(node_id)
    if not node:
        return jsonify({"error": "Node not found"}), 404

    children = tax.get_children(node_id)
    breadcrumb = tax.get_breadcrumb(node_id)
    papers = tax.get_node_papers(node_id, limit=50)
    paper_clusters = tax.get_node_paper_clusters(node_id)
    is_leaf = len(children) == 0
    # Only load heavy data for leaf nodes
    intersections = tax.get_subfield_intersection_matrix(node_id) if not is_leaf else {}
    matrix = tax.get_method_dataset_matrix(node_id) if is_leaf else {"methods": [], "datasets": [], "metrics": [], "cells": {}}
    gaps = tax.get_node_gaps(node_id) if is_leaf else []
    # Only return cached data - never block on LLM generation during page load
    opportunities = opp.get_node_opportunities(node_id)
    triage_queue = opp.get_opportunity_triage(node_id, limit=50)
    summary = tax.get_node_summary(node_id)
    graph_summary = graph.get_node_graph_summary(node_id)

    return jsonify({
        "node": dict(node),
        "children": children,
        "breadcrumb": breadcrumb,
        "is_leaf": is_leaf,
        "papers": papers,
        "paper_clusters": paper_clusters,
        "intersections": intersections,
        "matrix": matrix,
        "gaps": gaps,
        "opportunities": opportunities,
        "triage_queue": triage_queue,
        "summary": summary,
        "graph_summary": graph_summary,
    })


@app.route("/api/taxonomy/<node_id>/children")
def api_taxonomy_children(node_id):
    """Get just the children of a node with counts."""
    children = tax.get_children(node_id)
    return jsonify(children)


@app.route("/api/taxonomy/<node_id>/matrix")
def api_taxonomy_matrix(node_id):
    """Get the method x dataset matrix for a node."""
    matrix = tax.get_method_dataset_matrix(node_id)
    return jsonify(matrix)


@app.route("/api/taxonomy/<node_id>/intersections")
def api_taxonomy_intersections(node_id):
    """Get the subfield intersection matrix for a node."""
    return jsonify(tax.get_subfield_intersection_matrix(node_id))


@app.route("/api/taxonomy/<node_id>/papers")
def api_taxonomy_papers(node_id):
    """Get papers for a node."""
    limit = request.args.get("limit", 50, type=int)
    papers = tax.get_node_papers(node_id, limit=limit)
    return jsonify(papers)


@app.route("/api/taxonomy/<node_id>/paper_clusters")
def api_taxonomy_paper_clusters(node_id):
    """Get paper clusters for a node."""
    return jsonify(tax.get_node_paper_clusters(node_id))


@app.route("/api/taxonomy/<node_id>/gaps")
def api_taxonomy_gaps(node_id):
    """Get matrix gaps for a node."""
    gaps = tax.get_node_gaps(node_id)
    return jsonify(gaps)


@app.route("/api/taxonomy/<node_id>/opportunities")
def api_taxonomy_opportunities(node_id):
    """Get richer deterministic opportunity themes for a node."""
    return jsonify(opp.get_node_opportunities(node_id))


@app.route("/api/opportunity_triage")
def api_opportunity_triage():
    """Get the prioritized opportunity queue."""
    node_id = request.args.get("node_id", "")
    band = request.args.get("band", "")
    limit = request.args.get("limit", 100, type=int)
    return jsonify(opp.get_opportunity_triage(node_id or None, band or None, limit=limit))


@app.route("/api/insights")
def api_insights():
    """Get deep research insights from the insight agent."""
    limit = request.args.get("limit", 50, type=int)
    node_id = request.args.get("node_id", "")
    insight_type = request.args.get("type", "")

    sql = "SELECT * FROM insights WHERE 1=1"
    params = []
    if node_id:
        sql += " AND (node_id=? OR node_id LIKE ? || '.%')"
        params.extend([node_id, node_id])
    if insight_type:
        sql += " AND insight_type=?"
        params.append(insight_type)
    sql += " ORDER BY (novelty_score + feasibility_score) DESC, created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return jsonify(rows)


@app.route("/api/patterns")
def api_patterns():
    """Get cross-domain abstract patterns."""
    limit = request.args.get("limit", 50, type=int)
    node_id = request.args.get("node_id", "")
    level = request.args.get("level", "")

    sql = "SELECT * FROM patterns WHERE 1=1"
    params = []
    if node_id:
        sql += " AND node_id=?"
        params.append(node_id)
    if level:
        sql += " AND abstraction_level=?"
        params.append(level)
    sql += " ORDER BY domain_count DESC, created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return jsonify(rows)


@app.route("/api/bridges")
def api_bridges():
    """Get cross-domain bridge insights."""
    limit = request.args.get("limit", 20, type=int)
    rows = db.fetchall(
        "SELECT * FROM insights WHERE insight_type='cross_domain_bridge' "
        "ORDER BY (novelty_score + feasibility_score) DESC LIMIT ?",
        (limit,)
    )
    return jsonify(rows)


@app.route("/api/taxonomy/<node_id>/graph")
def api_taxonomy_graph(node_id):
    """Get the entity-relation graph summary for a node."""
    return jsonify(graph.ensure_node_graph_summary(node_id) or {})


@app.route("/api/papers/<paper_id>/graph")
def api_paper_graph(paper_id):
    """Get entity-relation evidence for one paper."""
    return jsonify(graph.get_paper_graph(paper_id))


@app.route("/api/graph/merge_candidates")
def api_graph_merge_candidates():
    """List entity merge candidates."""
    status = request.args.get("status", "pending")
    entity_type = request.args.get("entity_type", "") or None
    limit = request.args.get("limit", 100, type=int)
    return jsonify(graph.list_merge_candidates_with_context(status=status, limit=limit, entity_type=entity_type))


@app.route("/api/graph/merge_candidates/<int:candidate_id>")
def api_graph_merge_candidate(candidate_id: int):
    """Get one merge candidate with supporting context."""
    row = graph.get_merge_candidate_context(candidate_id)
    if not row:
        return jsonify({"error": "Candidate not found"}), 404
    return jsonify(row)


@app.route("/api/graph/merge_candidates/refresh", methods=["POST"])
def api_graph_merge_candidates_refresh():
    """Refresh heuristic merge candidates."""
    entity_type = request.json.get("entity_type") if request.is_json else None
    min_score = request.json.get("min_score", 0.84) if request.is_json else 0.84
    max_entities_per_type = request.json.get("max_entities_per_type", 500) if request.is_json else 500

    def run_refresh():
        log_event("merge_candidates_refresh_start", {
            "entity_type": entity_type,
            "min_score": min_score,
            "max_entities_per_type": max_entities_per_type,
        })
        stats = graph.refresh_merge_candidates(
            entity_type=entity_type,
            min_score=min_score,
            max_entities_per_type=max_entities_per_type,
        )
        log_event("merge_candidates_refresh_done", stats)

    thread = threading.Thread(target=run_refresh, daemon=True)
    thread.start()
    return jsonify({"status": "started", "entity_type": entity_type, "min_score": min_score})


@app.route("/api/graph/merge_candidates/<int:candidate_id>/decision", methods=["POST"])
def api_graph_merge_candidate_decision(candidate_id: int):
    """Accept or reject a merge candidate."""
    decision = request.json.get("decision", "rejected") if request.is_json else "rejected"
    note = request.json.get("note", "") if request.is_json else ""
    row = graph.decide_merge_candidate(candidate_id, decision=decision, note=note)
    if not row:
        return jsonify({"error": "Candidate not found"}), 404
    log_event("merge_candidate_decision", {
        "candidate_id": candidate_id,
        "decision": decision,
    })
    return jsonify(row)


# ── Search ─────────────────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    """Search across papers, methods, gaps, and taxonomy nodes."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify({"papers": [], "methods": [], "gaps": [], "nodes": [], "opportunities": []})

    search_term = f"%{q}%"

    papers = db.fetchall(
        """SELECT p.id, p.title, p.status, p.published_date,
                  pi.plain_summary, pi.work_type
           FROM papers p
           LEFT JOIN paper_insights pi ON p.id = pi.paper_id
           WHERE p.title LIKE ? OR p.abstract LIKE ? OR pi.plain_summary LIKE ?
           ORDER BY p.published_date DESC
           LIMIT 15""",
        (search_term, search_term, search_term),
    )

    methods = db.fetchall(
        """SELECT DISTINCT method_name as name, COUNT(*) as result_count,
                  COUNT(DISTINCT paper_id) as paper_count
           FROM results
           WHERE method_name LIKE ?
           GROUP BY method_name
           ORDER BY paper_count DESC
           LIMIT 10""",
        (search_term,),
    )

    gaps = db.fetchall(
        """SELECT mg.*, tn.name as node_name
           FROM matrix_gaps mg
           JOIN taxonomy_nodes tn ON mg.node_id = tn.id
           WHERE mg.gap_description LIKE ? OR mg.method_name LIKE ?
              OR mg.dataset_name LIKE ? OR mg.research_proposal LIKE ?
           ORDER BY mg.value_score DESC
           LIMIT 10""",
        (search_term, search_term, search_term, search_term),
    )

    nodes = db.fetchall(
        """SELECT t.*,
                  (SELECT COUNT(DISTINCT pt.paper_id)
                   FROM paper_taxonomy pt
                   WHERE pt.node_id = t.id OR pt.node_id LIKE t.id || '.%') AS paper_count
           FROM taxonomy_nodes t
           WHERE t.name LIKE ? OR t.description LIKE ? OR t.id LIKE ?
           ORDER BY paper_count DESC
           LIMIT 10""",
        (search_term, search_term, search_term),
    )

    opportunities = db.fetchall(
        """SELECT no.*, tn.name as node_name
           FROM node_opportunities no
           JOIN taxonomy_nodes tn ON no.node_id = tn.id
           WHERE no.title LIKE ? OR no.description LIKE ?
           ORDER BY no.value_score DESC
           LIMIT 10""",
        (search_term, search_term),
    )

    return jsonify({
        "papers": papers,
        "methods": methods,
        "gaps": gaps,
        "nodes": nodes,
        "opportunities": opportunities,
    })


# ── Recently Discovered ───────────────────────────────────────────

@app.route("/api/recent_discoveries")
def api_recent_discoveries():
    """Get recently discovered gaps, contradictions, opportunities, and taxonomy expansions."""
    limit = request.args.get("limit", 10, type=int)

    recent_gaps = db.fetchall(
        """SELECT mg.*, tn.name as node_name
           FROM matrix_gaps mg
           JOIN taxonomy_nodes tn ON mg.node_id = tn.id
           ORDER BY mg.created_at DESC LIMIT ?""",
        (limit,),
    )

    recent_contradictions = db.fetchall(
        """SELECT c.*, ca.claim_text as claim_a_text, ca.paper_id as paper_a,
                  cb.claim_text as claim_b_text, cb.paper_id as paper_b
           FROM contradictions c
           LEFT JOIN claims ca ON c.claim_a_id = ca.id
           LEFT JOIN claims cb ON c.claim_b_id = cb.id
           ORDER BY c.created_at DESC LIMIT ?""",
        (limit,),
    )

    recent_opportunities = db.fetchall(
        """SELECT no.*, tn.name as node_name
           FROM node_opportunities no
           JOIN taxonomy_nodes tn ON no.node_id = tn.id
           ORDER BY no.created_at DESC LIMIT ?""",
        (limit,),
    )

    recent_papers = db.fetchall(
        """SELECT p.id, p.title, p.published_date, p.status,
                  pi.plain_summary, pi.work_type
           FROM papers p
           LEFT JOIN paper_insights pi ON p.id = pi.paper_id
           WHERE p.status IN ('extracted', 'reasoned')
           ORDER BY p.updated_at DESC LIMIT ?""",
        (limit,),
    )

    return jsonify({
        "gaps": recent_gaps,
        "contradictions": recent_contradictions,
        "opportunities": recent_opportunities,
        "papers": recent_papers,
    })


# ── Taxonomy Expansion Trigger ────────────────────────────────────

@app.route("/api/taxonomy/expand", methods=["POST"])
def api_taxonomy_expand():
    """Manually trigger taxonomy expansion."""
    min_papers = request.json.get("min_papers", 10) if request.is_json else 10

    def do_expand():
        log_event("taxonomy_expansion_start", {"min_papers": min_papers})
        results = run_expansion(min_papers=min_papers)
        for exp in results:
            if exp.get("new_children"):
                log_event("taxonomy_expanded", {
                    "node_id": exp["node_id"],
                    "new_children": exp["new_children"],
                    "papers_reassigned": exp["papers_reassigned"],
                })
        log_event("taxonomy_expansion_done", {
            "nodes_checked": len(results),
            "nodes_expanded": sum(1 for e in results if e.get("new_children")),
        })

    thread = threading.Thread(target=do_expand, daemon=True)
    thread.start()
    return jsonify({"status": "started", "min_papers": min_papers})


# ── Legacy Endpoints (kept for compatibility) ─────────────────────

@app.route("/api/papers")
def api_papers():
    limit = request.args.get("limit", 50, type=int)
    status = request.args.get("status", "")
    if status:
        papers = db.fetchall(
            "SELECT id, title, status, token_cost, created_at FROM papers WHERE status=? ORDER BY updated_at DESC LIMIT ?",
            (status, limit))
    else:
        papers = db.fetchall(
            "SELECT id, title, status, token_cost, created_at FROM papers ORDER BY updated_at DESC LIMIT ?",
            (limit,))
    return jsonify(papers)


@app.route("/api/claims")
def api_claims():
    limit = request.args.get("limit", 100, type=int)
    paper_id = request.args.get("paper_id", "")
    if paper_id:
        claims = db.fetchall("SELECT * FROM claims WHERE paper_id=?", (paper_id,))
    else:
        claims = db.fetchall("SELECT * FROM claims ORDER BY id DESC LIMIT ?", (limit,))
    return jsonify(claims)


@app.route("/api/results")
def api_results():
    """Get structured results, optionally filtered."""
    limit = request.args.get("limit", 100, type=int)
    paper_id = request.args.get("paper_id", "")
    node_id = request.args.get("node_id", "")
    method = request.args.get("method", "")

    sql = "SELECT DISTINCT r.*, p.title as paper_title FROM results r JOIN papers p ON r.paper_id = p.id"
    params = []
    if node_id:
        sql += " JOIN result_taxonomy rt ON rt.result_id = r.id"
    sql += " WHERE 1=1"
    if paper_id:
        sql += " AND r.paper_id=?"
        params.append(paper_id)
    if node_id:
        sql += " AND (rt.node_id=? OR rt.node_id LIKE ? || '.%')"
        params.extend([node_id, node_id])
    if method:
        sql += " AND r.method_name=?"
        params.append(method)
    sql += " ORDER BY r.id DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return jsonify(rows)


@app.route("/api/contradictions")
def api_contradictions():
    limit = request.args.get("limit", 50, type=int)
    rows = db.fetchall("""
        SELECT c.*, ca.claim_text as claim_a_text, ca.paper_id as paper_a,
               cb.claim_text as claim_b_text, cb.paper_id as paper_b
        FROM contradictions c
        LEFT JOIN claims ca ON c.claim_a_id = ca.id
        LEFT JOIN claims cb ON c.claim_b_id = cb.id
        ORDER BY c.id DESC LIMIT ?
    """, (limit,))
    return jsonify(rows)


@app.route("/api/matrix_gaps")
def api_matrix_gaps():
    """Get all matrix gaps, optionally filtered by node."""
    limit = request.args.get("limit", 50, type=int)
    node_id = request.args.get("node_id", "")
    if node_id:
        rows = db.fetchall(
            """SELECT mg.*, tn.name as node_name
               FROM matrix_gaps mg
               JOIN taxonomy_nodes tn ON mg.node_id = tn.id
               WHERE mg.node_id=? OR mg.node_id LIKE ? || '.%'
               ORDER BY mg.value_score DESC LIMIT ?""",
            (node_id, node_id, limit))
    else:
        rows = db.fetchall(
            """SELECT mg.*, tn.name as node_name
               FROM matrix_gaps mg
               JOIN taxonomy_nodes tn ON mg.node_id = tn.id
               ORDER BY mg.value_score DESC LIMIT ?""",
            (limit,))
    return jsonify(rows)


# ── Events (SSE) ──────────────────────────────────────────────────

@app.route("/api/events")
def api_events():
    """SSE endpoint for real-time updates."""
    def generate():
        # Start from near the end - only send last 20 events on connect
        all_events = get_events(0)
        last_seq = max(0, all_events[-1]["seq"] - 20) if all_events else 0
        while True:
            events = get_events(last_seq)
            for e in events:
                yield f"data: {json.dumps(e)}\n\n"
                last_seq = e["seq"] + 1
            time.sleep(2)  # slower polling = less browser load

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Pipeline Control ──────────────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def api_start():
    """Start the pipeline."""
    global _pipeline_running
    max_papers = request.json.get("max_papers", 20) if request.is_json else 20

    with _pipeline_lock:
        if _pipeline_running:
            return jsonify({"status": "already_running", "max_papers": max_papers})
        _pipeline_running = True

    def safe_run(n):
        global _pipeline_running
        import traceback as tb
        try:
            run_continuous(n)
        except Exception as e:
            print(f"[PIPELINE CRASH] {e}", flush=True)
            print(tb.format_exc(), flush=True)
            log_event("error", {"step": "pipeline_crash", "error": str(e),
                                "traceback": tb.format_exc()})
        finally:
            with _pipeline_lock:
                _pipeline_running = False

    thread = threading.Thread(target=safe_run, args=(max_papers,), daemon=True)
    thread.start()
    return jsonify({"status": "started", "max_papers": max_papers})


@app.route("/api/backfill_graph", methods=["POST"])
def api_backfill_graph():
    """Backfill graph evidence from existing structured records."""
    overwrite = request.json.get("overwrite", False) if request.is_json else False
    limit = request.json.get("limit") if request.is_json else None

    def run_backfill():
        log_event("backfill_start", {"overwrite": overwrite, "limit": limit})
        stats = graph.backfill_graph_from_structured_data(limit=limit, overwrite=overwrite)
        log_event("backfill_done", stats)

    thread = threading.Thread(target=run_backfill, daemon=True)
    thread.start()
    return jsonify({"status": "started", "overwrite": overwrite, "limit": limit})


# ── EvoScientist Bridge ──────────────────────────────────────────────

@app.route("/api/research/launch", methods=["POST"])
def api_research_launch():
    """Launch EvoScientist research from a DeepGraph insight."""
    from agents.research_bridge import launch_evoscientist
    insight_id = request.json.get("insight_id") if request.is_json else None
    if not insight_id:
        return jsonify({"error": "insight_id required"}), 400
    try:
        result = launch_evoscientist(int(insight_id))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research/status")
def api_research_status():
    """Check status of an EvoScientist research session."""
    from agents.research_bridge import get_research_status
    workdir = request.args.get("workdir", "")
    if not workdir:
        return jsonify({"error": "workdir required"}), 400
    return jsonify(get_research_status(workdir))


@app.route("/api/research/proposal/<int:insight_id>")
def api_research_proposal(insight_id):
    """Preview the research proposal that would be sent to EvoScientist."""
    from agents.research_bridge import gather_context, format_proposal
    try:
        ctx = gather_context(insight_id)
        if not ctx.get("insight"):
            return jsonify({"error": "Insight not found"}), 404
        proposal = format_proposal(ctx)
        return jsonify({
            "insight_id": insight_id,
            "title": ctx["insight"]["title"],
            "paper_count": len(ctx["papers"]),
            "claim_count": len(ctx["claims"]),
            "contradiction_count": len(ctx["contradictions"]),
            "proposal": proposal,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/insights/rank", methods=["POST"])
def api_rank_insights():
    """Rank all insights by paradigm-breaking potential."""
    from agents.insight_ranker import rank_insights_batch
    def run_ranking():
        log_event("ranking_start", {})
        stats = rank_insights_batch()
        log_event("ranking_done", stats)
    thread = threading.Thread(target=run_ranking, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


# ── Deep Insights (Tier 1 / Tier 2) ─────────────────────────────────

@app.route("/api/deep_insights")
def api_deep_insights():
    """List deep insights with optional tier/status filter."""
    tier = request.args.get("tier", "", type=str)
    status = request.args.get("status", "")
    limit = request.args.get("limit", 50, type=int)

    sql = "SELECT * FROM deep_insights WHERE 1=1"
    params = []
    if tier:
        sql += " AND tier=?"
        try:
            params.append(int(tier))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid tier value"}), 400
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY CASE WHEN adversarial_score IS NOT NULL THEN adversarial_score ELSE 0 END DESC, created_at DESC LIMIT ?"
    params.append(limit)

    try:
        rows = db.fetchall(sql, tuple(params))
        return jsonify(rows)
    except Exception:
        return jsonify([])


@app.route("/api/deep_insights/<int:insight_id>")
def api_deep_insight_detail(insight_id):
    """Get full detail for one deep insight."""
    row = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(dict(row))


@app.route("/api/deep_insights/generate", methods=["POST"])
def api_generate_deep_insights():
    """Trigger discovery pipeline (harvest + Tier 1 + Tier 2).

    JSON body (optional):
      tier: "1" | "2" | "both" (default both)
      bulk: true — use DISCOVERY_BULK_* wider signals + expand all Tier2 problems
    """
    from orchestrator.discovery_scheduler import run_full_discovery
    tier = request.json.get("tier", "both") if request.is_json else "both"
    bulk = bool(request.json.get("bulk")) if request.is_json else False

    def do_discovery():
        if tier == "1":
            from orchestrator.discovery_scheduler import harvest_signals, run_tier1_discovery
            harvest_signals()
            run_tier1_discovery(bulk=bulk)
        elif tier == "2":
            from orchestrator.discovery_scheduler import harvest_signals, run_tier2_discovery
            harvest_signals()
            run_tier2_discovery(bulk=bulk)
        else:
            run_full_discovery(bulk=bulk)

    t = threading.Thread(target=do_discovery, daemon=True)
    t.start()
    return jsonify({"status": "started", "tier": tier, "bulk": bulk})


@app.route("/api/deep_insights/<int:insight_id>/verify", methods=["POST"])
def api_verify_deep_insight(insight_id):
    """Launch novelty verification via EvoScientist."""
    from agents.novelty_verifier import launch_verification
    try:
        result = launch_verification(insight_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/deep_insights/<int:insight_id>/verify_status")
def api_verify_status(insight_id):
    """Check verification status."""
    from agents.novelty_verifier import check_verification_result
    return jsonify(check_verification_result(insight_id))


@app.route("/api/deep_insights/<int:insight_id>/research", methods=["POST"])
def api_deep_insight_research(insight_id):
    """Launch full EvoScientist research session."""
    from agents.novelty_verifier import launch_full_research
    try:
        result = launch_full_research(insight_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/deep_insights/signals")
def api_deep_insight_signals():
    """Get current signal harvester data."""
    try:
        overlaps = db.fetchall(
            "SELECT * FROM node_entity_overlap ORDER BY overlap_score DESC LIMIT 20")
        pattern_ms = db.fetchall(
            "SELECT * FROM pattern_matches ORDER BY similarity_score DESC LIMIT 15")
        clusters = db.fetchall(
            "SELECT * FROM contradiction_clusters ORDER BY cluster_size DESC")
        plateaus = db.fetchall(
            "SELECT * FROM performance_plateaus ORDER BY method_count DESC LIMIT 15")
        return jsonify({
            "entity_overlaps": overlaps,
            "pattern_matches": pattern_ms,
            "contradiction_clusters": clusters,
            "performance_plateaus": plateaus,
        })
    except Exception:
        return jsonify({"entity_overlaps": [], "pattern_matches": [],
                        "contradiction_clusters": [], "performance_plateaus": []})


# ── SciForge: Experiment Validation ──────────────────────────────────

@app.route("/api/experiments")
def api_experiments():
    """List experiment runs with optional status/insight filter."""
    status = request.args.get("status", "")
    insight_id = request.args.get("insight_id", "", type=str)
    limit = request.args.get("limit", 50, type=int)

    sql = """SELECT er.*, di.title as insight_title, di.tier as insight_tier
             FROM experiment_runs er
             JOIN deep_insights di ON er.deep_insight_id = di.id
             WHERE 1=1"""
    params = []
    if status:
        sql += " AND er.status=?"
        params.append(status)
    if insight_id:
        sql += " AND er.deep_insight_id=?"
        try:
            params.append(int(insight_id))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid insight_id value"}), 400
    sql += " ORDER BY er.created_at DESC LIMIT ?"
    params.append(limit)

    try:
        rows = db.fetchall(sql, tuple(params))
        return jsonify(rows)
    except Exception:
        return jsonify([])


@app.route("/api/experiments/<int:run_id>")
def api_experiment_detail(run_id):
    """Get full detail for one experiment run including iterations."""
    run = db.fetchone(
        """SELECT er.*, di.title as insight_title, di.tier as insight_tier
           FROM experiment_runs er
           JOIN deep_insights di ON er.deep_insight_id = di.id
           WHERE er.id=?""", (run_id,))
    if not run:
        return jsonify({"error": "Not found"}), 404

    iterations = db.fetchall(
        """SELECT * FROM experiment_iterations WHERE run_id=?
           ORDER BY iteration_number""", (run_id,))

    claims = db.fetchall(
        "SELECT * FROM experimental_claims WHERE run_id=?", (run_id,))

    return jsonify({
        "run": dict(run),
        "iterations": iterations,
        "claims": claims,
    })


@app.route("/api/experiments/forge", methods=["POST"])
def api_forge_experiment():
    """Forge an experiment from a deep insight (scaffold + codebase)."""
    from agents.experiment_forge import forge_experiment
    insight_id = request.json.get("insight_id") if request.is_json else None
    if not insight_id:
        return jsonify({"error": "insight_id required"}), 400

    def do_forge():
        log_event("sciforge", {"step": "forge_start", "insight_id": insight_id})
        result = forge_experiment(int(insight_id))
        log_event("sciforge", {"step": "forge_done", **{k: v for k, v in result.items() if k != "codebase"}})

    t = threading.Thread(target=do_forge, daemon=True)
    t.start()
    return jsonify({"status": "started", "insight_id": insight_id})


@app.route("/api/experiments/<int:run_id>/run", methods=["POST"])
def api_run_experiment(run_id):
    """Launch the validation loop for a forged experiment."""
    from agents.validation_loop import run_validation_loop
    from agents.knowledge_loop import process_completed_run

    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return jsonify({"error": "Run not found"}), 404

    def do_run():
        log_event("sciforge", {"step": "loop_start", "run_id": run_id})
        try:
            result = run_validation_loop(run_id)
            log_event("sciforge", {"step": "loop_done", "run_id": run_id,
                                   "verdict": result.get("verdict", "?")})
            process_completed_run(run_id)
            log_event("sciforge", {"step": "knowledge_loop_done", "run_id": run_id})
        except Exception as e:
            db.execute(
                "UPDATE experiment_runs SET status='failed', error_message=? WHERE id=?",
                (str(e), run_id))
            db.commit()
            log_event("error", {"step": "validation_loop", "run_id": run_id, "error": str(e)})
            print(f"[SCIFORGE] Validation loop failed: {e}\n{traceback.format_exc()}", flush=True)

    t = threading.Thread(target=do_run, daemon=True)
    t.start()
    return jsonify({"status": "started", "run_id": run_id})


@app.route("/api/experiments/run_full", methods=["POST"])
def api_run_full_experiment():
    """Full pipeline: forge + validation loop + knowledge loop for a deep insight."""
    from agents.experiment_forge import forge_experiment
    from agents.validation_loop import run_validation_loop
    from agents.knowledge_loop import process_completed_run

    insight_id = request.json.get("insight_id") if request.is_json else None
    if not insight_id:
        return jsonify({"error": "insight_id required"}), 400

    def do_full():
        log_event("sciforge", {"step": "full_start", "insight_id": insight_id})
        try:
            forge_result = forge_experiment(int(insight_id))
            if "error" in forge_result:
                log_event("error", {"step": "forge", "error": forge_result["error"]})
                return
            run_id = forge_result["run_id"]
            log_event("sciforge", {"step": "forge_done", "run_id": run_id})

            loop_result = run_validation_loop(run_id)
            log_event("sciforge", {"step": "loop_done", "run_id": run_id,
                                   "verdict": loop_result.get("verdict", "?")})

            process_completed_run(run_id)
            log_event("sciforge", {"step": "full_done", "run_id": run_id,
                                   "verdict": loop_result.get("verdict", "?")})
        except Exception as e:
            log_event("error", {"step": "full_experiment", "insight_id": insight_id, "error": str(e)})
            print(f"[SCIFORGE] Full experiment failed: {e}\n{traceback.format_exc()}", flush=True)

    t = threading.Thread(target=do_full, daemon=True)
    t.start()
    return jsonify({"status": "started", "insight_id": insight_id})


@app.route("/api/meta_report")
def api_meta_report():
    """Get the meta-learning report on hypothesis quality."""
    from agents.meta_learner import get_full_meta_report
    try:
        return jsonify(get_full_meta_report())
    except Exception:
        return jsonify({"status": "error", "total_experiments": 0})
