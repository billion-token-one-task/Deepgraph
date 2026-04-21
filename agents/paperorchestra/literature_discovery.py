"""Step 3 (PaperOrchestra §4): verified citation registry from graph evidence + S2 discovery."""

from __future__ import annotations

import json
from typing import Any

from db import database as db

from agents.paperorchestra.semantic_scholar import (
    arxiv_id_from_paper,
    paper_to_bibtex_entry,
    paper_to_bibtex_key,
    paper_year,
    search_papers,
)


def _extract_queries_from_outline(outline: dict) -> list[str]:
    """Flatten intro_related_work_plan (+ light section_plan citation_hints) into search strings."""
    qs: list[str] = []
    irw = outline.get("intro_related_work_plan") or {}
    intro = irw.get("introduction_strategy") or {}
    for s in intro.get("search_directions") or []:
        if isinstance(s, str) and s.strip():
            qs.append(s.strip())
    rw = irw.get("related_work_strategy") or {}
    for sub in rw.get("subsections") or []:
        if not isinstance(sub, dict):
            continue
        for k in ("limitation_search_queries",):
            for q in sub.get(k) or []:
                if isinstance(q, str) and q.strip():
                    qs.append(q.strip())
        for k in ("sota_investigation_mission", "methodology_cluster"):
            v = sub.get(k)
            if isinstance(v, str) and v.strip():
                qs.append(v.strip())
    # section_plan citation_hints as weak queries
    for sec in outline.get("section_plan") or []:
        if not isinstance(sec, dict):
            continue
        for sub in sec.get("subsections") or []:
            if not isinstance(sub, dict):
                continue
            for h in sub.get("citation_hints") or []:
                if isinstance(h, str) and len(h) > 10:
                    qs.append(h[:500])
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        loaded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return loaded if isinstance(loaded, list) else []


def _dedupe(items: list[Any]) -> list[Any]:
    seen = set()
    out: list[Any] = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _db_paper_to_registry_row(
    pid: str,
    *,
    source_claim_ids: list[str] | None = None,
    source_node_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    row = db.fetchone(
        """
        SELECT id, arxiv_base_id, title, abstract, authors, published_date
        FROM papers
        WHERE id=? OR arxiv_base_id=?
        ORDER BY CASE WHEN id=? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (pid, pid, pid),
    )
    if not row:
        return None
    arxiv_id = row.get("arxiv_base_id") or row.get("id") or pid
    key = str(arxiv_id).replace(".", "_").replace("/", "_")
    try:
        authors = json.loads(row["authors"]) if row.get("authors") else []
    except (json.JSONDecodeError, TypeError):
        authors = []
    return {
        "paperId": f"db:{row.get('id')}",
        "title": row.get("title"),
        "year": int(row["published_date"][:4]) if row.get("published_date") and len(row["published_date"]) >= 4 else None,
        "abstract": row.get("abstract"),
        "authors": [{"name": a} for a in authors if isinstance(a, str)],
        "externalIds": {"ArXiv": arxiv_id} if isinstance(arxiv_id, str) and arxiv_id else {},
        "_cite_key": key,
        "_source": "evidence_graph",
        "_db_paper_id": row.get("id"),
        "_source_claim_ids": source_claim_ids or [],
        "_source_node_ids": source_node_ids or [],
        "_matched_queries": [],
    }


def _merge_registry_row(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        merged = dict(incoming)
        merged["_sources"] = _dedupe([incoming.get("_source")])
        return merged
    merged = dict(existing)
    for key in ("title", "abstract", "paperId", "_db_paper_id"):
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming[key]
    if not merged.get("year") and incoming.get("year"):
        merged["year"] = incoming["year"]
    if not merged.get("authors") and incoming.get("authors"):
        merged["authors"] = incoming["authors"]
    ext = dict(merged.get("externalIds") or {})
    ext.update(incoming.get("externalIds") or {})
    merged["externalIds"] = ext
    merged["_source_claim_ids"] = _dedupe(list(merged.get("_source_claim_ids") or []) + list(incoming.get("_source_claim_ids") or []))
    merged["_source_node_ids"] = _dedupe(list(merged.get("_source_node_ids") or []) + list(incoming.get("_source_node_ids") or []))
    merged["_matched_queries"] = _dedupe(list(merged.get("_matched_queries") or []) + list(incoming.get("_matched_queries") or []))
    merged["_sources"] = _dedupe(list(merged.get("_sources") or []) + [incoming.get("_source")])
    if merged.get("_source") != "evidence_graph":
        merged["_source"] = incoming.get("_source") or merged.get("_source")
    return merged


def _paper_matches_id(paper: dict[str, Any], pid: str) -> bool:
    if not pid:
        return False
    if paper.get("_db_paper_id") == pid:
        return True
    arxiv_id = arxiv_id_from_paper(paper)
    return bool(arxiv_id and arxiv_id == pid)


def run_literature_discovery(
    outline: dict,
    evidence_paper_ids: list[str],
    *,
    claim_evidence: list[dict[str, Any]] | None = None,
    cutoff_year: int,
    api_key: str | None,
    max_queries: int = 24,
    per_query_limit: int = 5,
) -> dict[str, Any]:
    """
    Returns:
      collected_papers: list for literature_review_agent ``collected_papers``
      bibtex: merged .bib string
      bib_keys: ordered keys for \\cite
      registry: list of dicts (metadata)
    """
    by_key: dict[str, dict[str, Any]] = {}
    normalized_claims: list[dict[str, Any]] = []

    for idx, claim in enumerate(claim_evidence or []):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or f"claim_{idx + 1}")
        normalized = {
            "claim_id": claim_id,
            "claim_text": claim.get("claim_text") or "",
            "source_paper_ids": [str(x) for x in _as_list(claim.get("source_paper_ids")) if x],
            "source_node_ids": [str(x) for x in _as_list(claim.get("source_node_ids")) if x],
        }
        normalized_claims.append(normalized)
        for pid in normalized["source_paper_ids"]:
            row = _db_paper_to_registry_row(
                pid,
                source_claim_ids=[claim_id],
                source_node_ids=normalized["source_node_ids"],
            )
            if not row:
                continue
            by_key[row["_cite_key"]] = _merge_registry_row(by_key.get(row["_cite_key"]), row)

    for pid in evidence_paper_ids:
        row = _db_paper_to_registry_row(str(pid))
        if not row:
            continue
        k = row["_cite_key"]
        by_key[k] = _merge_registry_row(by_key.get(k), row)

    queries = _extract_queries_from_outline(outline)[:max_queries]
    for q in queries:
        try:
            hits = search_papers(q, limit=per_query_limit, api_key=api_key)
        except Exception:
            continue
        for p in hits:
            y = paper_year(p)
            if y is not None and y > cutoff_year:
                continue
            k = paper_to_bibtex_key(p)
            candidate = dict(p)
            candidate["_cite_key"] = k
            candidate["_source"] = "semantic_scholar"
            candidate["_source_claim_ids"] = []
            candidate["_source_node_ids"] = []
            candidate["_matched_queries"] = [q]
            by_key[k] = _merge_registry_row(by_key.get(k), candidate)

    registry = list(by_key.values())
    bib_chunks: list[str] = []
    bib_keys: list[str] = []
    collected: list[dict[str, Any]] = []
    claim_citation_map: dict[str, dict[str, Any]] = {}

    for p in registry:
        k = p["_cite_key"]
        bib_keys.append(k)
        bib_chunks.append(paper_to_bibtex_entry(p, k))
        collected.append(
            {
                "cite_key": k,
                "title": p.get("title"),
                "abstract": (p.get("abstract") or "")[:4000],
                "year": paper_year(p),
                "arxiv_id": arxiv_id_from_paper(p) or (p.get("paperId") or "").replace("db:", ""),
                "source": p.get("_source"),
                "sources": p.get("_sources") or [p.get("_source")],
                "source_claim_ids": p.get("_source_claim_ids") or [],
                "source_node_ids": p.get("_source_node_ids") or [],
                "matched_queries": p.get("_matched_queries") or [],
            }
        )

    for claim in normalized_claims:
        cite_keys = [
            row["_cite_key"]
            for row in registry
            if any(_paper_matches_id(row, pid) for pid in claim["source_paper_ids"])
        ]
        claim_citation_map[claim["claim_id"]] = {
            "claim_text": claim["claim_text"],
            "source_paper_ids": claim["source_paper_ids"],
            "source_node_ids": claim["source_node_ids"],
            "cite_keys": _dedupe(cite_keys),
        }

    return {
        "collected_papers": collected,
        "bibtex": "\n".join(bib_chunks),
        "bib_keys": bib_keys,
        "registry": registry,
        "claim_citation_map": claim_citation_map,
        "queries_used": queries,
    }
