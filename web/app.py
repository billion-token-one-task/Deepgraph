"""Flask web application for DeepGraph dashboard."""
import json
import threading
import time
from flask import Flask, render_template, jsonify, Response, request
from config import APP_NAME, APP_SUBTITLE, PROFILE, ROOT_NODE_ID
from db import database as db
from db import taxonomy as tax
from orchestrator.pipeline import get_events, run_continuous, log_event, get_stats_dict

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
    matrix = tax.get_method_dataset_matrix(node_id)
    gaps = tax.get_node_gaps(node_id)
    summary = tax.ensure_node_summary(node_id)

    # Check if this is a leaf node (no children in taxonomy)
    is_leaf = len(children) == 0

    return jsonify({
        "node": dict(node),
        "children": children,
        "breadcrumb": breadcrumb,
        "is_leaf": is_leaf,
        "papers": papers,
        "matrix": matrix,
        "gaps": gaps,
        "summary": summary,
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


@app.route("/api/taxonomy/<node_id>/papers")
def api_taxonomy_papers(node_id):
    """Get papers for a node."""
    limit = request.args.get("limit", 50, type=int)
    papers = tax.get_node_papers(node_id, limit=limit)
    return jsonify(papers)


@app.route("/api/taxonomy/<node_id>/gaps")
def api_taxonomy_gaps(node_id):
    """Get matrix gaps for a node."""
    gaps = tax.get_node_gaps(node_id)
    return jsonify(gaps)


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
        last_seq = 0
        while True:
            events = get_events(last_seq)
            for e in events:
                yield f"data: {json.dumps(e)}\n\n"
                last_seq = e["seq"] + 1
            time.sleep(1)

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
