"""Reference expansion gate for PaperOrchestra manuscripts.

The manager treats 50 references as a target, not a hard dependency on one
provider. Manuscript writing is allowed once a verified topic-relevant minimum is
met, while the search loop keeps trying multiple sources to approach the target.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from db import database as db

from agents.paperorchestra.literature_discovery import (
    _dedupe,
    _extract_queries_from_outline,
    _literature_relevance_score,
    _merge_registry_row,
)
from agents.paperorchestra.semantic_scholar import (
    arxiv_id_from_paper,
    paper_to_bibtex_entry,
    paper_to_bibtex_key,
    paper_year,
    search_papers,
)


REFERENCE_MANAGER_VERSION = "deepgraph_reference_manager_v1_2026_06_12"
DEFAULT_REFERENCE_TARGET = 50
DEFAULT_REFERENCE_MINIMUM = 30
STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "allocation",
    "answer",
    "benchmark",
    "benchmarks",
    "completed",
    "could",
    "large",
    "language",
    "models",
    "paper",
    "reasoning",
    "strongest",
    "system",
    "systems",
    "using",
    "with",
}

NON_PAPER_TITLES = {
    "abstract",
    "acknowledgment",
    "acknowledgments",
    "author index",
    "back matter",
    "bibliography",
    "contents",
    "copyright",
    "dedication",
    "editorial",
    "erratum",
    "foreword",
    "front matter",
    "index",
    "introduction",
    "preface",
    "references",
    "table of contents",
}


class ReferenceExpansionError(RuntimeError):
    """Raised when verified literature cannot reach the required minimum."""

    def __init__(self, report: dict[str, Any], expanded_literature: dict[str, Any] | None = None):
        self.report = report
        self.expanded_literature = expanded_literature or {}
        super().__init__(
            "Reference manager collected "
            f"{report.get('final_count', 0)}/{report.get('minimum_count', DEFAULT_REFERENCE_MINIMUM)} "
            f"verified references (target {report.get('target_count', DEFAULT_REFERENCE_TARGET)}); "
            "manuscript writing is blocked."
        )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())




def _has_named_author(row: dict[str, Any]) -> bool:
    authors = row.get("authors") if isinstance(row.get("authors"), list) else []
    for author in authors:
        name = ""
        if isinstance(author, dict):
            name = str(author.get("name") or "")
        elif isinstance(author, str):
            name = author
        cleaned = _clean_text(name).lower()
        if cleaned and cleaned not in {"unknown", "anonymous", "n/a", "na"}:
            return True
    return False


def _is_reference_candidate(row: dict[str, Any]) -> bool:
    """Reject provider metadata records that are real items but not citeable papers."""
    title = _clean_text(row.get("title"))
    if not title:
        return False
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    if normalized_title in NON_PAPER_TITLES:
        return False
    if not _has_named_author(row):
        return False
    if paper_year(row) is None:
        return False
    return True


def _state_queries(state: dict[str, Any], evidence_brief: dict[str, Any] | None) -> list[str]:
    method = _clean_text(state.get("method_name") or state.get("title"))
    title = _clean_text(state.get("title"))
    problem = _clean_text((state.get("problem_awareness") or {}).get("central_question"))
    intent = _clean_text((state.get("paper_intent") or {}).get("central_claim"))
    datasets = []
    experiment = (evidence_brief or {}).get("experiment") or {}
    for row in experiment.get("datasets") or []:
        if isinstance(row, dict) and row.get("name"):
            datasets.append(str(row["name"]).replace("-Controlled", ""))

    seeds = [
        f"{method} large language model multi-agent reasoning",
        f"{title} inference-time reasoning",
        f"{problem} large language models",
        f"{intent} inference-time compute allocation",
        "self-consistency chain-of-thought reasoning large language models",
        "Tree of Thoughts deliberate problem solving large language models",
        "multi-agent debate large language models reasoning",
        "large language model multi-agent systems survey reasoning",
        "LLM agents consensus verification reasoning",
        "verifier-guided reasoning large language models",
        "best-of-n sampling verifier large language model reasoning",
        "majority voting self-consistency large language models",
        "test-time compute allocation large language models reasoning",
        "adaptive inference budget large language models",
        "selective reasoning token budget large language models",
        "confidence calibration large language model reasoning",
        "uncertainty estimation large language model reasoning",
        "selective prediction abstention large language models",
        "model routing cost quality large language models",
        "RouteLLM routing large language models cost quality",
        "early exit large language model inference confidence",
        "answer aggregation large language models reasoning",
        "question answering reasoning benchmark large language models",
        "GSM8K chain-of-thought large language models",
        "StrategyQA large language model reasoning",
        "least-to-most prompting large language models reasoning",
        "program-of-thought prompting large language models",
        "LLM debate diversity reasoning",
        "multi-agent deliberation answer selection large language models",
        "LLM reasoning reliability calibration verification",
    ]
    for dataset in datasets:
        seeds.append(f"{dataset} large language model reasoning benchmark")
    return [q for q in seeds if q and len(q) > 12]


def _registry_from_lit(lit_out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in lit_out.get("registry") or []:
        if not isinstance(row, dict):
            continue
        key = row.get("_cite_key") or row.get("cite_key")
        if not key:
            try:
                key = paper_to_bibtex_key(row)
            except Exception:  # noqa: BLE001
                continue
        candidate = dict(row)
        candidate["_cite_key"] = str(key)
        candidate.setdefault("_source", row.get("source") or "literature_discovery")
        candidate.setdefault("_matched_queries", row.get("matched_queries") or [])
        candidate.setdefault("_source_claim_ids", row.get("source_claim_ids") or [])
        candidate.setdefault("_source_node_ids", row.get("source_node_ids") or [])
        by_key[str(key)] = _merge_registry_row(by_key.get(str(key)), candidate)
    return by_key


def _accepted_registry(by_key: dict[str, dict[str, Any]], queries: list[str]) -> list[dict[str, Any]]:
    registry = [
        row
        for row in by_key.values()
        if _is_reference_candidate(row) and _literature_relevance_score(row, queries) >= 1.0
    ]
    registry.sort(
        key=lambda row: (
            _literature_relevance_score(row, queries),
            int(row.get("citationCount") or 0),
            paper_year(row) or 0,
        ),
        reverse=True,
    )
    return registry


def _decode_openalex_abstract(index: dict[str, Any] | None) -> str:
    if not isinstance(index, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for pos in positions or []:
            try:
                words.append((int(pos), str(word)))
            except (TypeError, ValueError):
                continue
    return " ".join(word for _pos, word in sorted(words))


def _clean_doi(value: Any) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi


def _strip_markup(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return html.unescape(" ".join(text.split()))


def _keywords(query: str, limit: int = 7) -> list[str]:
    terms = []
    for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", query.lower()):
        if term in STOPWORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _local_db_search(query: str, *, limit: int) -> list[dict[str, Any]]:
    terms = _keywords(query)
    if not terms:
        return []
    where = " OR ".join(["lower(title) LIKE ? OR lower(abstract) LIKE ?" for _ in terms])
    params: list[Any] = []
    for term in terms:
        like = f"%{term}%"
        params.extend([like, like])
    params.append(limit)
    rows = db.fetchall(
        f"""
        SELECT id, arxiv_base_id, title, abstract, authors, published_date
        FROM papers
        WHERE {where}
        ORDER BY COALESCE(published_date, '') DESC
        LIMIT ?
        """,
        tuple(params),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            authors = json.loads(row.get("authors") or "[]")
        except (json.JSONDecodeError, TypeError):
            authors = []
        arxiv_id = row.get("arxiv_base_id") or row.get("id")
        out.append(
            {
                "paperId": f"db:{row.get('id')}",
                "title": row.get("title"),
                "year": int(row["published_date"][:4]) if row.get("published_date") else None,
                "abstract": row.get("abstract"),
                "authors": [{"name": a} for a in authors if isinstance(a, str)],
                "externalIds": {"ArXiv": arxiv_id} if arxiv_id else {},
                "venue": "local paper database",
                "citationCount": 0,
            }
        )
    return out


def _openalex_search(query: str, *, limit: int, timeout: float = 20.0) -> list[dict[str, Any]]:
    import httpx

    params = {"search": query, "per-page": min(limit, 50)}
    with httpx.Client(timeout=timeout) as client:
        response = client.get("https://api.openalex.org/works", params=params)
        response.raise_for_status()
        payload = response.json()
    out: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        authors = []
        for auth in item.get("authorships") or []:
            author = auth.get("author") if isinstance(auth, dict) else {}
            name = author.get("display_name") if isinstance(author, dict) else ""
            if name:
                authors.append({"name": name})
        doi = _clean_doi(item.get("doi"))
        source = ((item.get("primary_location") or {}).get("source") or {}).get("display_name")
        out.append(
            {
                "paperId": f"openalex:{item.get('id')}",
                "title": item.get("display_name"),
                "year": item.get("publication_year"),
                "publicationDate": item.get("publication_date"),
                "abstract": _decode_openalex_abstract(item.get("abstract_inverted_index")),
                "authors": authors,
                "externalIds": {k: v for k, v in {"DOI": doi, "OpenAlex": item.get("id")}.items() if v},
                "venue": source or "OpenAlex",
                "citationCount": item.get("cited_by_count") or 0,
            }
        )
    time.sleep(0.15)
    return out


def _crossref_search(query: str, *, limit: int, timeout: float = 20.0) -> list[dict[str, Any]]:
    import httpx

    params = {"query.bibliographic": query, "rows": min(limit, 20)}
    with httpx.Client(timeout=timeout) as client:
        response = client.get("https://api.crossref.org/works", params=params)
        response.raise_for_status()
        payload = response.json()
    out: list[dict[str, Any]] = []
    for item in (payload.get("message") or {}).get("items") or []:
        title = " ".join(item.get("title") or [])
        if not title:
            continue
        year = None
        parts = ((item.get("issued") or {}).get("date-parts") or [[]])[0]
        if parts:
            try:
                year = int(parts[0])
            except (TypeError, ValueError):
                year = None
        authors = []
        for author in item.get("author") or []:
            given = author.get("given") or ""
            family = author.get("family") or ""
            name = " ".join(x for x in [given, family] if x).strip()
            if name:
                authors.append({"name": name})
        doi = _clean_doi(item.get("DOI"))
        venue = " ".join(item.get("container-title") or [])
        out.append(
            {
                "paperId": f"crossref:{doi or title[:80]}",
                "title": title,
                "year": year,
                "abstract": _strip_markup(item.get("abstract")),
                "authors": authors,
                "externalIds": {"DOI": doi} if doi else {},
                "venue": venue or "Crossref",
                "citationCount": item.get("is-referenced-by-count") or 0,
            }
        )
    time.sleep(0.15)
    return out


def _arxiv_search(query: str, *, limit: int, timeout: float = 20.0) -> list[dict[str, Any]]:
    import httpx

    compact = " ".join(_keywords(query, limit=5)) or query[:120]
    params = urllib.parse.urlencode(
        {"search_query": f'all:"{compact}"', "start": 0, "max_results": min(limit, 20)}
    )
    url = "https://export.arxiv.org/api/query?" + params
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        xml_text = response.text
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    out: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns)
        arxiv_url = entry.findtext("atom:id", default="", namespaces=ns)
        arxiv_id = arxiv_url.rstrip("/").split("/")[-1].split("v")[0]
        authors = []
        for author in entry.findall("atom:author", ns):
            name = _clean_text(author.findtext("atom:name", default="", namespaces=ns))
            if name:
                authors.append({"name": name})
        if not title:
            continue
        out.append(
            {
                "paperId": f"arxiv:{arxiv_id}",
                "title": title,
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "publicationDate": published[:10] if published else "",
                "abstract": abstract,
                "authors": authors,
                "externalIds": {"ArXiv": arxiv_id} if arxiv_id else {},
                "venue": "arXiv",
                "citationCount": 0,
            }
        )
    time.sleep(0.35)
    return out


def _search_source(source: str, query: str, *, limit: int, api_key: str | None) -> list[dict[str, Any]]:
    if source == "local_paper_db":
        return _local_db_search(query, limit=limit)
    if source == "openalex":
        return _openalex_search(query, limit=limit)
    if source == "crossref":
        return _crossref_search(query, limit=limit)
    if source == "arxiv":
        return _arxiv_search(query, limit=limit)
    if source == "semantic_scholar":
        return search_papers(query, limit=limit, api_key=api_key)
    raise ValueError(f"unknown reference source: {source}")


def _materialize_literature(
    registry: list[dict[str, Any]],
    claim_citation_map: dict[str, Any],
    queries_used: list[str],
    report: dict[str, Any],
) -> dict[str, Any]:
    bib_chunks: list[str] = []
    bib_keys: list[str] = []
    collected: list[dict[str, Any]] = []
    for p in registry:
        key = str(p.get("_cite_key") or paper_to_bibtex_key(p))
        p["_cite_key"] = key
        bib_keys.append(key)
        bib_chunks.append(paper_to_bibtex_entry(p, key))
        collected.append(
            {
                "cite_key": key,
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
    return {
        "collected_papers": collected,
        "bibtex": "\n".join(bib_chunks),
        "bib_keys": bib_keys,
        "registry": registry,
        "claim_citation_map": claim_citation_map,
        "queries_used": queries_used,
        "reference_manager": report,
    }


def expand_references_or_raise(
    lit_out: dict[str, Any],
    outline: dict[str, Any],
    state: dict[str, Any],
    evidence_brief: dict[str, Any] | None,
    *,
    cutoff_year: int,
    api_key: str | None,
    target_count: int = DEFAULT_REFERENCE_TARGET,
    minimum_count: int = DEFAULT_REFERENCE_MINIMUM,
    per_query_limit: int = 20,
    max_queries: int = 80,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Expand verified references using multiple providers.

    ``target_count`` is aspirational; ``minimum_count`` is the hard floor for
    manuscript writing. Each provider failure is recorded but does not stop the
    expansion loop while other providers remain available.
    """
    minimum_count = max(1, min(int(minimum_count), int(target_count)))
    initial_registry = _accepted_registry(
        _registry_from_lit(lit_out),
        list(lit_out.get("queries_used") or []),
    )
    by_key = {str(row.get("_cite_key")): row for row in initial_registry if row.get("_cite_key")}
    initial_count = len(initial_registry)
    query_pool = _dedupe(
        list(lit_out.get("queries_used") or [])
        + _extract_queries_from_outline(outline)
        + _state_queries(state, evidence_brief)
    )[:max_queries]
    source_order = sources or ["local_paper_db", "openalex", "crossref", "arxiv", "semantic_scholar"]
    query_attempts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    registry = _accepted_registry(by_key, query_pool)
    for query in query_pool:
        if len(registry) >= target_count:
            break
        source_attempts: list[dict[str, Any]] = []
        for source in source_order:
            if len(registry) >= target_count:
                break
            try:
                hits = _search_source(source, query, limit=per_query_limit, api_key=api_key)
            except Exception as exc:  # noqa: BLE001
                errors.append({"query": query, "source": source, "error": str(exc)[:500]})
                source_attempts.append({"source": source, "status": "error", "error": str(exc)[:300]})
                continue

            accepted = 0
            for paper in hits:
                if not _is_reference_candidate(paper):
                    continue
                year = paper_year(paper)
                if year is not None and year > cutoff_year:
                    continue
                try:
                    key = paper_to_bibtex_key(paper)
                except Exception:  # noqa: BLE001
                    continue
                candidate = dict(paper)
                candidate["_cite_key"] = key
                candidate["_source"] = source
                candidate["_source_claim_ids"] = []
                candidate["_source_node_ids"] = []
                candidate["_matched_queries"] = [query]
                merged_preview = _merge_registry_row(by_key.get(key), candidate)
                if _literature_relevance_score(merged_preview, query_pool) < 1.0:
                    continue
                before = key in by_key
                by_key[key] = merged_preview
                if not before:
                    accepted += 1
            registry = _accepted_registry(by_key, query_pool)
            source_attempts.append(
                {
                    "source": source,
                    "status": "ok",
                    "hit_count": len(hits),
                    "accepted_new_count": accepted,
                    "running_count": len(registry),
                }
            )
        query_attempts.append({"query": query, "sources": source_attempts, "running_count": len(registry)})

    final_count = len(registry)
    status = "ok" if final_count >= target_count else "ok_minimum_met" if final_count >= minimum_count else "insufficient_references"
    report = {
        "schema_version": REFERENCE_MANAGER_VERSION,
        "target_count": target_count,
        "minimum_count": minimum_count,
        "initial_count": initial_count,
        "final_count": final_count,
        "status": status,
        "sources": source_order,
        "queries_attempted": query_attempts,
        "errors": errors[:30],
        "blockers": [],
    }
    expanded = _materialize_literature(
        registry,
        lit_out.get("claim_citation_map") or {},
        query_pool,
        report,
    )
    if final_count < minimum_count:
        report["blockers"] = [
            f"Reference manager collected {final_count}/{minimum_count} verified references (target {target_count}).",
            "Manuscript writing is blocked until literature discovery reaches the minimum reference floor.",
        ]
        expanded["reference_manager"] = report
        raise ReferenceExpansionError(report, expanded_literature=expanded)
    return expanded
