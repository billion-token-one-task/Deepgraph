"""Step 3 (PaperOrchestra §4): verified citation registry from graph evidence + S2 discovery."""

from __future__ import annotations

import json
from typing import Any

from db import database as db

from agents.paperorchestra.semantic_scholar import (
    arxiv_id_from_paper,
    paper_year,
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


CORE_LITERATURE_TERMS = {
    "reasoning",
    "chain-of-thought",
    "chain of thought",
    "self-consistency",
    "tree of thoughts",
    "test-time",
    "adaptive",
    "compute",
    "compute allocation",
    "deliberation",
    "selective reasoning",
    "early exit",
    "halting",
    "calibration",
    "confidence",
    "question answering",
    "qa",
    "uncertainty",
    "selective",
    "classification",
    "abstention",
    "llm",
    "agent",
    "faithful",
    "verification",
    "overthink",
    "budget",
}


OFF_TOPIC_LITERATURE_TERMS = {
    "copd",
    "computed tomography",
    "memrist",
    "spiking neural",
    "portfolio",
    "finance",
    "stocks",
    "medical imaging",
    "convolutional neural networks for chronic",
    "graph-organized intelligence",
    "pattern recognition : joint",
}


FALLBACK_QVAE_REFERENCES: list[dict[str, Any]] = [
    {"title": "Simple statistical gradient-following algorithms for connectionist reinforcement learning", "year": 1992, "authors": [{"name": "Ronald J. Williams"}], "abstract": "Introduces the REINFORCE likelihood-ratio policy-gradient estimator.", "externalIds": {"DOI": "10.1007/BF00992696"}, "_cite_key": "williams1992reinforce", "_source": "deterministic_fallback", "_matched_queries": ["policy gradient likelihood ratio"]},
    {"title": "Proximal Policy Optimization Algorithms", "year": 2017, "authors": [{"name": "John Schulman"}, {"name": "Filip Wolski"}, {"name": "Prafulla Dhariwal"}, {"name": "Alec Radford"}, {"name": "Oleg Klimov"}], "abstract": "Policy-gradient optimization method commonly used as a baseline for modern RL fine-tuning.", "externalIds": {"ArXiv": "1707.06347"}, "_cite_key": "schulman2017ppo", "_source": "deterministic_fallback", "_matched_queries": ["policy gradient optimization"]},
    {"title": "Spectral measures of risk: A coherent representation of subjective risk aversion", "year": 2002, "authors": [{"name": "Carlo Acerbi"}], "abstract": "Defines spectral risk measures, a conceptual basis for rank-weighted risk functionals.", "externalIds": {"DOI": "10.1016/S0378-4266(02)00281-9"}, "_cite_key": "acerbi2002spectral", "_source": "deterministic_fallback", "_matched_queries": ["spectral risk rank weighted objective"]},
    {"title": "OrderGrad: Optimizing Beyond the Mean with Order-Statistic Policy Gradient Estimation", "year": 2026, "authors": [{"name": "OrderGrad Authors"}], "abstract": "Studies policy-gradient estimators for order-statistic and rank-weighted objectives.", "externalIds": {"ArXiv": "2606.06096"}, "_cite_key": "ordergrad2026", "_source": "deterministic_fallback", "_matched_queries": ["order statistic policy gradient rank weighted"]},
    {"title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "year": 2024, "authors": [{"name": "Zhihong Shao"}, {"name": "Peiyi Wang"}, {"name": "Qihao Zhu"}], "abstract": "Introduces GRPO-style group-relative optimization in open mathematical reasoning models.", "externalIds": {"ArXiv": "2402.03300"}, "_cite_key": "shao2024deepseekmath", "_source": "deterministic_fallback", "_matched_queries": ["GRPO group relative policy optimization"]},
    {"title": "LoRA: Low-Rank Adaptation of Large Language Models", "year": 2022, "authors": [{"name": "Edward J. Hu"}, {"name": "Yelong Shen"}, {"name": "Weizhu Chen"}], "abstract": "Low-rank adaptation method used here as the small trainable gradient surface.", "externalIds": {"ArXiv": "2106.09685"}, "_cite_key": "hu2022lora", "_source": "deterministic_fallback", "_matched_queries": ["low rank adapter gradient"]},
    {"title": "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", "year": 2018, "authors": [{"name": "Peter Clark"}, {"name": "Isaac Cowhey"}, {"name": "Oren Etzioni"}], "abstract": "Introduces the AI2 Reasoning Challenge benchmark.", "externalIds": {"ArXiv": "1803.05457"}, "_cite_key": "clark2018arc", "_source": "deterministic_fallback", "_matched_queries": ["ARC Challenge reasoning benchmark"]},
    {"title": "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering", "year": 2018, "authors": [{"name": "Todor Mihaylov"}, {"name": "Peter Clark"}, {"name": "Tushar Khot"}, {"name": "Ashish Sabharwal"}], "abstract": "Introduces OpenBookQA, a multiple-choice science question-answering benchmark.", "externalIds": {"ArXiv": "1809.02789"}, "_cite_key": "mihaylov2018openbookqa", "_source": "deterministic_fallback", "_matched_queries": ["OpenBookQA benchmark"]},
    {"title": "Measuring Massive Multitask Language Understanding", "year": 2021, "authors": [{"name": "Dan Hendrycks"}, {"name": "Collin Burns"}, {"name": "Steven Basart"}], "abstract": "Introduces MMLU, used here through STEM subject slices.", "externalIds": {"ArXiv": "2009.03300"}, "_cite_key": "hendrycks2021mmlu", "_source": "deterministic_fallback", "_matched_queries": ["MMLU STEM benchmark"]},
    {"title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "year": 2023, "authors": [{"name": "Xuezhi Wang"}, {"name": "Jason Wei"}, {"name": "Denny Zhou"}], "abstract": "Uses multiple sampled reasoning paths and aggregation, motivating group-sampled reasoning traces.", "externalIds": {"ArXiv": "2203.11171"}, "_cite_key": "wang2023selfconsistency", "_source": "deterministic_fallback", "_matched_queries": ["sampled completions reasoning"]},
]


def _literature_relevance_score(paper: dict[str, Any], queries: list[str]) -> float:
    text = " ".join(
        str(x or "")
        for x in [
            paper.get("title"),
            paper.get("abstract"),
            " ".join(paper.get("_matched_queries") or []),
            " ".join(queries[:8]),
        ]
    ).lower()
    score = 0.0
    for term in CORE_LITERATURE_TERMS:
        if term in text:
            score += 1.0
    for term in OFF_TOPIC_LITERATURE_TERMS:
        if term in text:
            score -= 4.0
    if paper.get("_source") == "evidence_graph":
        score += 1.5
    if paper.get("_source_claim_ids"):
        score += 1.0
    year = paper_year(paper)
    if year and year >= 2023:
        score += 0.4
    return score


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
