"""Hard reference expansion gate for PaperOrchestra manuscripts."""

from __future__ import annotations

import json
from typing import Any

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


REFERENCE_MANAGER_VERSION = "deepgraph_reference_manager_v1_2026_06_11"
DEFAULT_REFERENCE_TARGET = 50


class ReferenceExpansionError(RuntimeError):
    """Raised when verified literature cannot be expanded to the required floor."""

    def __init__(self, report: dict[str, Any], expanded_literature: dict[str, Any] | None = None):
        self.report = report
        self.expanded_literature = expanded_literature or {}
        super().__init__(
            "Reference manager collected "
            f"{report.get('final_count', 0)}/{report.get('target_count', DEFAULT_REFERENCE_TARGET)} "
            "verified references; manuscript writing is blocked."
        )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


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


def _accepted_registry(
    by_key: dict[str, dict[str, Any]],
    queries: list[str],
) -> list[dict[str, Any]]:
    registry = [
        row
        for row in by_key.values()
        if row.get("title") and _literature_relevance_score(row, queries) >= 1.0
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
    per_query_limit: int = 20,
    max_queries: int = 80,
) -> dict[str, Any]:
    """Expand Semantic Scholar references until ``target_count`` is met, or block writing.

    The input/output schema matches ``run_literature_discovery`` so downstream
    writing agents receive only verified cite keys that exist in ``references.bib``.
    """
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
    query_attempts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    registry = _accepted_registry(by_key, query_pool)
    for query in query_pool:
        if len(registry) >= target_count:
            break
        try:
            hits = search_papers(query, limit=per_query_limit, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            errors.append({"query": query, "error": str(exc)[:500]})
            query_attempts.append({"query": query, "status": "error", "error": str(exc)[:300]})
            continue

        accepted = 0
        for paper in hits:
            year = paper_year(paper)
            if year is not None and year > cutoff_year:
                continue
            key = paper_to_bibtex_key(paper)
            candidate = dict(paper)
            candidate["_cite_key"] = key
            candidate["_source"] = "reference_manager"
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
        query_attempts.append(
            {
                "query": query,
                "status": "ok",
                "hit_count": len(hits),
                "accepted_new_count": accepted,
                "running_count": len(registry),
            }
        )

    final_count = len(registry)
    report = {
        "schema_version": REFERENCE_MANAGER_VERSION,
        "target_count": target_count,
        "initial_count": initial_count,
        "final_count": final_count,
        "status": "ok" if final_count >= target_count else "insufficient_references",
        "queries_attempted": query_attempts,
        "errors": errors[:20],
        "blockers": [],
    }
    expanded = _materialize_literature(
        registry,
        lit_out.get("claim_citation_map") or {},
        query_pool,
        report,
    )
    if final_count < target_count:
        report["blockers"] = [
            f"Reference manager collected {final_count}/{target_count} verified references.",
            "Manuscript writing is blocked until literature discovery reaches the required reference floor.",
        ]
        expanded["reference_manager"] = report
        raise ReferenceExpansionError(report, expanded_literature=expanded)
    return expanded
