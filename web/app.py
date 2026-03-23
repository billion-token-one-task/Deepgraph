"""Flask web application for DeepGraph dashboard."""
import json
import threading
import time
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
    """Get papers currently being processed."""
    rows = db.fetchall(
        "SELECT id, title, status FROM papers WHERE status='processing' ORDER BY updated_at DESC LIMIT 10"
    )
    return jsonify(rows)


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
    max_papers = request.json.get("max_papers", 20) if request.is_json else 20
    thread = threading.Thread(target=run_continuous, args=(max_papers,), daemon=True)
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
    ctx = gather_context(insight_id)
    proposal = format_proposal(ctx)
    return jsonify({
        "insight_id": insight_id,
        "title": ctx["insight"]["title"],
        "paper_count": len(ctx["papers"]),
        "claim_count": len(ctx["claims"]),
        "contradiction_count": len(ctx["contradictions"]),
        "proposal": proposal,
    })


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
