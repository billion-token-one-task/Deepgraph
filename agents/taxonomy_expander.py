"""Taxonomy Expander Agent: automatically sub-divides leaf nodes that accumulate many papers.

When a leaf taxonomy node accumulates EXPANSION_THRESHOLD or more papers, this agent:
1. Reads the titles and abstracts/summaries of the papers in that node
2. Asks the LLM to discover 3-6 natural sub-categories
3. Creates new child nodes in the taxonomy
4. Re-classifies existing papers into the new children
"""
import json
import logging
import re

from agents.llm_client import call_llm_json
from db import database as db

logger = logging.getLogger(__name__)

# A leaf node must have at least this many papers before we consider expanding it
EXPANSION_THRESHOLD = 10

# Maximum depth the taxonomy tree is allowed to grow to
MAX_TAXONOMY_DEPTH = 8

SYSTEM_PROMPT = """You are a taxonomy design agent for a scientific research knowledge base.

You will be given a taxonomy node and the papers classified under it. Your job is to discover
3-6 natural sub-categories that would meaningfully divide these papers.

Return a JSON object:
{
  "subcategories": [
    {
      "slug": "short_snake_case_id",
      "name": "Human Readable Name",
      "description": "One sentence describing what papers in this sub-area study"
    }
  ],
  "paper_assignments": {
    "<paper_id>": "slug_of_best_matching_subcategory"
  }
}

Rules:
- slug must be a short, lowercase, snake_case identifier (e.g. "anchor_free", "transformer_det")
- Create 3-6 subcategories. Prefer fewer, broader categories over many narrow ones.
- Every paper in the input MUST appear in paper_assignments.
- If a paper truly does not fit any subcategory, assign it to the most general one.
- Do NOT create a catch-all "other" category. Every category should have a clear identity.
- Subcategories should be roughly balanced in size when possible.
- Names should be concise (2-5 words).
- Descriptions should help a non-expert understand the sub-area.
- Return ONLY valid JSON, no markdown formatting."""


def get_expandable_leaves(min_papers: int = EXPANSION_THRESHOLD) -> list[dict]:
    """Find leaf taxonomy nodes that have enough papers to warrant expansion.

    Returns a list of dicts with node info and paper counts.
    """
    rows = db.fetchall(
        """SELECT t.id, t.name, t.parent_id, t.depth, t.description,
                  COUNT(DISTINCT pt.paper_id) AS paper_count
           FROM taxonomy_nodes t
           LEFT JOIN taxonomy_nodes c ON c.parent_id = t.id
           LEFT JOIN paper_taxonomy pt ON pt.node_id = t.id
           WHERE c.id IS NULL
           GROUP BY t.id
           HAVING paper_count >= ?
           ORDER BY paper_count DESC""",
        (min_papers,),
    )
    return rows


def _get_node_paper_details(node_id: str, limit: int = 40) -> list[dict]:
    """Get paper titles, abstracts, and summaries for papers in a specific node (direct, not subtree)."""
    rows = db.fetchall(
        """SELECT p.id, p.title, p.abstract,
                  pi.plain_summary, pi.work_type
           FROM papers p
           JOIN paper_taxonomy pt ON p.id = pt.paper_id
           LEFT JOIN paper_insights pi ON p.id = pi.paper_id
           WHERE pt.node_id = ?
           ORDER BY p.published_date DESC
           LIMIT ?""",
        (node_id, limit),
    )
    return rows


def _sanitize_slug(slug: str) -> str:
    """Ensure the slug is safe for use as a taxonomy node ID component."""
    slug = slug.strip().lower()
    slug = re.sub(r'[^a-z0-9_]', '_', slug)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug[:40]  # cap length


def expand_node(node_id: str) -> dict:
    """Expand a single leaf node by discovering sub-categories via LLM.

    Returns a dict with:
        - node_id: the expanded node
        - new_children: list of new child node IDs created
        - papers_reassigned: number of papers moved to children
        - tokens: LLM tokens used
    """
    node = db.fetchone("SELECT * FROM taxonomy_nodes WHERE id=?", (node_id,))
    if not node:
        return {"node_id": node_id, "error": "Node not found", "new_children": [], "papers_reassigned": 0, "tokens": 0}

    # Check if node already has children (not a leaf)
    children = db.fetchall("SELECT id FROM taxonomy_nodes WHERE parent_id=?", (node_id,))
    if children:
        return {"node_id": node_id, "error": "Node already has children", "new_children": [], "papers_reassigned": 0, "tokens": 0}

    # Check depth limit
    if node["depth"] >= MAX_TAXONOMY_DEPTH:
        return {"node_id": node_id, "error": "Max depth reached", "new_children": [], "papers_reassigned": 0, "tokens": 0}

    # Get papers in this node
    papers = _get_node_paper_details(node_id)
    if len(papers) < EXPANSION_THRESHOLD:
        return {"node_id": node_id, "error": "Not enough papers", "new_children": [], "papers_reassigned": 0, "tokens": 0}

    # Build LLM prompt
    paper_lines = []
    for p in papers:
        line = f"- [{p['id']}] {p['title']}"
        if p.get("plain_summary"):
            line += f" | {p['plain_summary']}"
        elif p.get("abstract"):
            abstract = p["abstract"]
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            line += f" | {abstract}"
        if p.get("work_type"):
            line += f" (type: {p['work_type']})"
        paper_lines.append(line)

    user_prompt = f"""Taxonomy node: {node['name']} (ID: {node_id})
Description: {node.get('description', '')}
Depth: {node['depth']}
Number of papers: {len(papers)}

Papers classified under this node:
{chr(10).join(paper_lines)}

Based on these papers, propose 3-6 natural sub-categories that would meaningfully divide this research area. Also assign each paper to its best-matching sub-category."""

    try:
        result, tokens = call_llm_json(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error("LLM call failed for taxonomy expansion of %s: %s", node_id, e)
        return {"node_id": node_id, "error": str(e), "new_children": [], "papers_reassigned": 0, "tokens": 0}

    subcategories = result.get("subcategories", [])
    paper_assignments = result.get("paper_assignments", {})

    if not subcategories or len(subcategories) < 2:
        return {"node_id": node_id, "error": "LLM returned too few subcategories", "new_children": [], "papers_reassigned": 0, "tokens": tokens}

    # Create new child nodes
    new_child_ids = []
    child_depth = node["depth"] + 1

    for sort_idx, subcat in enumerate(subcategories):
        slug = _sanitize_slug(subcat.get("slug") or subcat.get("name", "unnamed"))
        child_id = f"{node_id}.{slug}"
        child_name = subcat.get("name", slug)
        child_desc = subcat.get("description", "")

        # Ensure uniqueness: if this ID already exists, skip
        existing = db.fetchone("SELECT id FROM taxonomy_nodes WHERE id=?", (child_id,))
        if existing:
            logger.info("Taxonomy node %s already exists, skipping", child_id)
            new_child_ids.append(child_id)
            continue

        db.execute(
            """INSERT INTO taxonomy_nodes (id, name, parent_id, depth, description, sort_order)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (child_id, child_name, node_id, child_depth, child_desc, sort_idx),
        )
        new_child_ids.append(child_id)
        logger.info("Created taxonomy node: %s (%s)", child_id, child_name)

    db.commit()

    # Build a map of slug -> full child ID
    slug_to_id = {}
    for subcat, child_id in zip(subcategories, new_child_ids):
        slug = _sanitize_slug(subcat.get("slug", subcat["name"]))
        slug_to_id[slug] = child_id
        # Also map the original slug in case sanitization changed it
        slug_to_id[subcat.get("slug", "")] = child_id

    # Re-classify papers into children
    papers_reassigned = 0
    # Default to first child if assignment not found
    default_child = new_child_ids[0] if new_child_ids else None

    for paper in papers:
        paper_id = paper["id"]
        assigned_slug = paper_assignments.get(paper_id, "")
        sanitized = _sanitize_slug(assigned_slug) if assigned_slug else ""

        target_child = slug_to_id.get(assigned_slug) or slug_to_id.get(sanitized) or default_child

        if target_child:
            # Get the confidence from the original assignment
            orig = db.fetchone(
                "SELECT confidence FROM paper_taxonomy WHERE paper_id=? AND node_id=?",
                (paper_id, node_id),
            )
            confidence = orig["confidence"] if orig else 1.0

            # Add assignment to child node
            db.execute(
                """INSERT OR REPLACE INTO paper_taxonomy (paper_id, node_id, confidence)
                   VALUES (?, ?, ?)""",
                (paper_id, target_child, confidence),
            )

            # Also reassign any results that were linked to the parent node
            result_rows = db.fetchall(
                """SELECT rt.result_id FROM result_taxonomy rt
                   JOIN results r ON r.id = rt.result_id
                   WHERE rt.node_id = ? AND r.paper_id = ?""",
                (node_id, paper_id),
            )
            for rrow in result_rows:
                db.execute(
                    "INSERT OR IGNORE INTO result_taxonomy (result_id, node_id) VALUES (?, ?)",
                    (rrow["result_id"], target_child),
                )

            # Reassign entity mentions
            db.execute(
                """UPDATE paper_entity_mentions SET node_id = ?
                   WHERE paper_id = ? AND node_id = ?""",
                (target_child, paper_id, node_id),
            )

            # Reassign graph relations
            db.execute(
                """UPDATE graph_relations SET node_id = ?
                   WHERE paper_id = ? AND node_id = ?""",
                (target_child, paper_id, node_id),
            )

            papers_reassigned += 1

    db.commit()

    return {
        "node_id": node_id,
        "new_children": new_child_ids,
        "papers_reassigned": papers_reassigned,
        "tokens": tokens,
    }


def run_expansion(min_papers: int = EXPANSION_THRESHOLD) -> list[dict]:
    """Find all expandable leaves and expand them.

    Returns a list of expansion results (one per expanded node).
    """
    expandable = get_expandable_leaves(min_papers=min_papers)
    results = []

    for leaf in expandable:
        node_id = leaf["id"]
        logger.info(
            "Expanding taxonomy node %s (%s) with %d papers",
            node_id, leaf["name"], leaf["paper_count"],
        )
        result = expand_node(node_id)
        results.append(result)

        if result.get("new_children"):
            logger.info(
                "Expanded %s into %d children, reassigned %d papers",
                node_id, len(result["new_children"]), result["papers_reassigned"],
            )

    return results
