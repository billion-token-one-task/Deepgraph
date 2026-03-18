"""Orchestrator: continuous paper processing pipeline."""
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import PIPELINE_CONCURRENCY
from db import database as db
from db import taxonomy as tax
from ingestion.arxiv_client import fetch_recent
from ingestion.pdf_parser import get_paper_text
from agents.extraction_agent import extract_paper
from agents.reasoning_agent import detect_contradictions, discover_matrix_gaps

# Global event log for SSE
_event_log: list[dict] = []


def get_events(since: int = 0) -> list[dict]:
    return _event_log[since:]


def log_event(event_type: str, data: dict):
    _event_log.append({
        "type": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        "seq": len(_event_log),
    })
    # Keep last 1000 events
    if len(_event_log) > 1000:
        _event_log.pop(0)


def ingest_papers(max_papers: int = 100) -> int:
    """Fetch and store papers from arXiv."""
    log_event("ingest_start", {"max_papers": max_papers})
    papers = fetch_recent(max_results=max_papers)
    count = 0
    for p in papers:
        existing = db.fetchone("SELECT id FROM papers WHERE id=?", (p["id"],))
        if not existing:
            db.insert_paper(p)
            count += 1
    log_event("ingest_done", {"new_papers": count, "total_fetched": len(papers)})
    return count


def process_single_paper(paper_id: str) -> dict:
    """Run full pipeline on one paper: extract -> classify -> store results -> check contradictions."""
    paper = db.fetchone("SELECT * FROM papers WHERE id=?", (paper_id,))
    if not paper:
        return {"error": "Paper not found"}

    result = {"paper_id": paper_id, "claims": 0, "results": 0,
              "taxonomy_nodes": [], "contradictions": 0, "tokens": 0}

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

        db.update_paper_status(paper_id, "extracted", token_cost=tokens1)

        # Step 3: Check for contradictions
        log_event("step", {"paper_id": paper_id, "title": paper["title"], "step": "reasoning"})
        total_contradictions = 0
        tokens2 = 0
        for c in claims:
            if c.get("claim_type") == "performance" and c.get("method_name"):
                contras, t = detect_contradictions(c, c["_id"])
                tokens2 += t
                for contra in contras:
                    contra["claim_a_id"] = c["_id"]
                    contra["claim_b_id"] = contra.get("existing_claim_id")
                    db.insert_contradiction(contra)
                    total_contradictions += 1
                    log_event("contradiction", {
                        "paper_id": paper_id,
                        "description": contra["description"],
                        "hypothesis": contra.get("hypothesis", ""),
                    })

        result["contradictions"] = total_contradictions
        result["tokens"] += tokens2
        db.update_paper_status(paper_id, "reasoned", token_cost=tokens2)

        log_event("paper_done", {
            "paper_id": paper_id,
            "title": paper["title"],
            "claims": result["claims"],
            "results": result["results"],
            "taxonomy_nodes": result["taxonomy_nodes"],
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
    log_event("pipeline_start", {"max_papers": max_papers})

    # Step 1: Ingest
    ingest_papers(max_papers=max_papers or 100)

    # Step 2: Process papers in parallel
    processed = 0
    workers = PIPELINE_CONCURRENCY
    summary_nodes: set[str] = set()

    while True:
        limit = min(workers, (max_papers - processed) if max_papers else workers)
        papers = db.fetchall(
            "SELECT id FROM papers WHERE status='ingested' ORDER BY published_date DESC LIMIT ?",
            (limit,)
        )
        if not papers:
            break

        # Mark as processing to avoid double-pick
        for p in papers:
            db.execute("UPDATE papers SET status='processing' WHERE id=? AND status='ingested'", (p["id"],))
        db.commit()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_single_paper, p["id"]): p["id"] for p in papers}
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    paper_result = future.result()
                    for node_id in paper_result.get("taxonomy_nodes", []):
                        summary_nodes.update(tax.get_ancestor_ids(node_id))
                except Exception as e:
                    log_event("error", {"paper_id": pid, "error": str(e)})
                processed += 1

        if max_papers and processed >= max_papers:
            break

    # Step 3: Run matrix gap discovery on nodes with enough data
    nodes_with_data = db.fetchall(
        """SELECT rt.node_id, COUNT(DISTINCT r.method_name) as mc,
                  COUNT(DISTINCT r.dataset_name) as dc
           FROM results r
           JOIN result_taxonomy rt ON rt.result_id = r.id
           GROUP BY rt.node_id
           HAVING mc >= 2 AND dc >= 2"""
    )
    total_gap_tokens = 0
    for node in nodes_with_data:
        log_event("step", {"step": "gap_discovery", "node_id": node["node_id"]})
        db.execute("DELETE FROM matrix_gaps WHERE node_id=?", (node["node_id"],))
        db.commit()
        gaps, tokens = discover_matrix_gaps(node["node_id"])
        total_gap_tokens += tokens
        for g in gaps:
            tax.insert_matrix_gap(g)
            log_event("gap", {
                "node_id": g["node_id"],
                "method": g["method_name"],
                "dataset": g["dataset_name"],
                "description": g["gap_description"],
                "value": g.get("value_score", 0),
            })

    # Step 4: Generate plain-language node summaries for exploration
    for node_id in sorted(summary_nodes):
        log_event("step", {"step": "node_summary", "node_id": node_id})
        summary = tax.ensure_node_summary(node_id, force=True)
        if summary:
            log_event("summary", {
                "node_id": node_id,
                "overview": (summary.get("overview") or "")[:180],
            })

    log_event("pipeline_done", {"papers_processed": processed, "stats": get_stats_dict()})
    return processed


def get_stats_dict() -> dict:
    """Get comprehensive stats."""
    base = db.get_stats()
    base["results_total"] = db.fetchone("SELECT COUNT(*) as c FROM results")["c"]
    base["taxonomy_assignments"] = db.fetchone("SELECT COUNT(*) as c FROM paper_taxonomy")["c"]
    base["matrix_gaps_total"] = db.fetchone("SELECT COUNT(*) as c FROM matrix_gaps")["c"]
    return base
