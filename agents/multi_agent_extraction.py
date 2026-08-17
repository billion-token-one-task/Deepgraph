"""Multi-role paper extraction.

Each role reads the same compact paper text but is asked to optimize for a
different evidence surface. The merger preserves the legacy extraction schema
so downstream storage and graph-writing code do not need to change.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from meta_harness.scoped_llm import proposer_json


TAXONOMY_OVERVIEW_SYSTEM = """You are the Taxonomy and Paper-Overview Reader.

Your role is narrow and strict:
1. Classify the paper into 1-3 provided leaf taxonomy nodes.
2. Write a grounded plain-language overview.
3. Extract stated key findings, stated limitations, and stated open questions.

Do NOT extract numeric result tables. Do NOT invent missing limitations.

Return one JSON object:
{
  "taxonomy_nodes": [{"node_id": "leaf.node.id", "confidence": 0.0-1.0}],
  "paper_overview": {
    "plain_summary": "...",
    "problem_statement": "...",
    "approach_summary": "...",
    "work_type": "model|training_method|benchmark|dataset|application|system|analysis|theory",
    "key_findings": ["..."],
    "limitations": ["..."],
    "open_questions": ["..."]
  },
  "key_findings": ["..."],
  "limitations_stated": ["..."]
}

Rules:
- Use ONLY taxonomy leaf node IDs listed by the user.
- Prefer specific nodes over broad nodes.
- If evidence is weak, lower confidence instead of hallucinating."""


EMPIRICAL_RESULTS_SYSTEM = """You are the Empirical Results Reader.

Your role is to extract every concrete quantitative result tuple:
method, dataset, metric, value, unit, SOTA flag, evidence location, and exact source quote.

Ignore high-level summaries unless they contain a numeric result. Do NOT infer
numbers from plots. Do NOT copy unsupported table fragments.

Return one JSON object:
{
  "results": [
    {
      "method_name": "...",
      "dataset_name": "...",
      "metric_name": "...",
      "metric_value": 0.0,
      "metric_unit": "%",
      "is_sota": 0,
      "evidence_location": "Table/Figure/Section",
      "source_quote": "verbatim contiguous text, at least 12 chars"
    }
  ]
}

Rules:
- Extract all rows you can ground, not only the best result.
- metric_value must be numeric. If unavailable, omit the row.
- source_quote must appear verbatim in the provided paper text."""


CLAIMS_METHODS_SYSTEM = """You are the Claims and Methods Reader.

Your role is to identify the paper's methods and scientifically meaningful claims.
Focus on claims that can later support contradictions, method transfers, or
claim-method gap signals.

Return one JSON object:
{
  "methods": [
    {
      "name": "...",
      "category": "...",
      "description": "...",
      "key_innovation": "...",
      "builds_on": ["..."]
    }
  ],
  "claims": [
    {
      "claim_text": "...",
      "claim_type": "performance|method|finding|limitation",
      "method_name": "...",
      "dataset_name": "...",
      "metric_name": "...",
      "metric_value": 0.0,
      "evidence_location": "...",
      "source_quote": "verbatim contiguous text, at least 12 chars",
      "conditions": {}
    }
  ]
}

Rules:
- Claims must be grounded by source_quote.
- Include non-numeric mechanism/finding claims when they are central.
- Omit claims whose evidence is ambiguous."""


GRAPH_CONTEXT_SYSTEM = """You are the Evidence-Graph Reader.

Your role is to extract canonical reusable entities and typed relations for the
DeepGraph evidence graph. Prefer concepts, methods, datasets, metrics, tasks,
artifacts, and theories that other papers may mention again.

Return one JSON object:
{
  "knowledge_graph": {
    "entities": [
      {
        "name": "...",
        "entity_type": "concept|method|task|dataset|metric|artifact|material|gene|protein|disease|organism|theory",
        "description": "...",
        "aliases": ["..."],
        "mention_role": "proposed|used|evaluated|baseline|limitation|theory",
        "confidence": 0.0-1.0,
        "evidence_location": "...",
        "source_text": "short grounding snippet"
      }
    ],
    "relations": [
      {
        "subject": "...",
        "subject_type": "method",
        "predicate": "uses|builds_on|evaluated_on|measured_by|compares_with|applied_to|improves_over|part_of|studies|predicts|treats|interacts_with|derived_from|related_to",
        "object": "...",
        "object_type": "dataset",
        "confidence": 0.0-1.0,
        "evidence_location": "...",
        "source_text": "short grounding snippet"
      }
    ]
  }
}

Rules:
- Keep names canonical and reusable.
- Do not create entities for trivial ablations unless the paper names them.
- Relations must be explicitly supported by the text."""


RESEARCH_FACETS_SYSTEM = """You are the Research-Facets Reader.

Your role is to extract reusable research reasoning units that can later create
cross-paper discovery signals. Focus on why the paper exists and what mechanism
or design pattern it contributes, not just benchmark outcomes.

Return one JSON object:
{
  "research_facets": [
    {
      "facet_type": "problem_frame|motivation_rationale|methodology_unit|design_decision|protocol_mechanism|boundary_condition",
      "facet_name": "short canonical name",
      "summary": "specific grounded statement",
      "evidence_location": "Section/Table/Figure",
      "source_quote": "verbatim or near-verbatim short supporting quote",
      "metadata": {
        "target_setting": "...",
        "why_existing_methods_fail": "...",
        "alternative_rejected": "...",
        "expected_effect": "...",
        "mechanism_tags": ["..."]
      }
    }
  ]
}

Facet definitions:
- problem_frame: the concrete problem, bottleneck, or missing capability.
- motivation_rationale: why this direction should work; analogy, hypothesis, or why-now.
- methodology_unit: reusable method object, interface, objective, adapter, decoder, or policy.
- design_decision: a deliberate choice plus reason and expected effect.
- protocol_mechanism: evaluation/training protocol, metric adapter, decoding rule, exclusion rule, data-mixture rule.
- boundary_condition: assumptions, limitations, failure modes, where it may not work.

Rules:
- Extract 8-18 facets when the paper supports them.
- Prefer specific mechanisms over generic phrases like "better performance".
- Each summary should be reusable across papers for matching.
- Do not invent limitations; boundary_condition may be empty if the paper gives none.
- Keep facet_name short and canonical, e.g. "RGB output decoding" or "low-ratio task-data mixture"."""


def _paper_user_prompt(
    paper_id: str,
    title: str,
    taxonomy_hint: str,
    compact_text: str,
    *,
    include_taxonomy: bool = True,
) -> str:
    """Build the shared paper prompt.

    Only the taxonomy reader classifies, so only it is charged for the taxonomy
    listing.  The other roles used to receive the same multi-hundred-thousand
    character list they never referenced, once each.

    The taxonomy leads the prompt when present: it is identical for every paper
    in a branch, so keeping it in front of the per-paper text lets prompt
    caching reuse it instead of re-sending it per paper.
    """
    paper_block = f"""Paper ID: {paper_id}
Title: {title}

Full text:
{compact_text}"""
    if include_taxonomy and taxonomy_hint:
        return f"{taxonomy_hint}\n\n{paper_block}"
    return paper_block


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _dedupe_by_key(items: list[dict], key_fn) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = key_fn(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _merge_taxonomy(*payloads: dict) -> list[dict]:
    by_node: dict[str, dict] = {}
    for payload in payloads:
        for row in _list(payload.get("taxonomy_nodes")):
            node_id = str(row.get("node_id") or "").strip()
            if not node_id:
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            current = by_node.get(node_id)
            if current is None or confidence > float(current.get("confidence") or 0):
                by_node[node_id] = {"node_id": node_id, "confidence": round(confidence, 4)}
    return sorted(by_node.values(), key=lambda row: row["confidence"], reverse=True)[:3]


def _merge_overview(overview_payload: dict, claims_payload: dict) -> dict:
    overview = dict(_dict(overview_payload.get("paper_overview")))
    overview.setdefault("plain_summary", "")
    overview.setdefault("problem_statement", "")
    overview.setdefault("approach_summary", "")
    overview.setdefault("work_type", "analysis")
    overview["key_findings"] = _list(overview.get("key_findings")) or _list(overview_payload.get("key_findings"))
    overview["limitations"] = _list(overview.get("limitations")) or _list(overview_payload.get("limitations_stated"))
    overview["open_questions"] = _list(overview.get("open_questions"))
    if not overview.get("approach_summary"):
        methods = _list(claims_payload.get("methods"))
        if methods:
            overview["approach_summary"] = "; ".join(str(m.get("description") or m.get("name") or "") for m in methods[:3])
    return overview


def _merge_research_facets(*payloads: dict) -> list[dict]:
    facets = []
    for payload in payloads:
        facets.extend(_list(payload.get("research_facets")))
    return _dedupe_by_key(
        facets,
        lambda row: "|".join(
            [
                str(row.get("facet_type") or "").strip().lower(),
                str(row.get("facet_name") or row.get("name") or "").strip().lower(),
                str(row.get("summary") or row.get("description") or "").strip().lower(),
            ]
        ),
    )


def merge_role_extractions(role_outputs: dict[str, dict]) -> dict:
    overview = _dict(role_outputs.get("taxonomy_overview"))
    empirical = _dict(role_outputs.get("empirical_results"))
    claims_methods = _dict(role_outputs.get("claims_methods"))
    graph_context = _dict(role_outputs.get("graph_context"))
    research_facets = _dict(role_outputs.get("research_facets"))

    methods = _dedupe_by_key(
        _list(claims_methods.get("methods")),
        lambda row: str(row.get("name") or "").strip().lower(),
    )
    claims = _dedupe_by_key(
        _list(claims_methods.get("claims")),
        lambda row: "|".join(
            [
                str(row.get("claim_text") or "").strip().lower(),
                str(row.get("method_name") or "").strip().lower(),
                str(row.get("dataset_name") or "").strip().lower(),
            ]
        ),
    )
    results = _dedupe_by_key(
        _list(empirical.get("results")),
        lambda row: "|".join(
            [
                str(row.get("method_name") or "").strip().lower(),
                str(row.get("dataset_name") or "").strip().lower(),
                str(row.get("metric_name") or "").strip().lower(),
                str(row.get("metric_value") or "").strip(),
            ]
        ),
    )

    graph_payload = _dict(graph_context.get("knowledge_graph"))
    graph_payload["entities"] = _dedupe_by_key(
        _list(graph_payload.get("entities")),
        lambda row: f"{str(row.get('entity_type') or '').lower()}:{str(row.get('name') or '').lower()}",
    )
    graph_payload["relations"] = _dedupe_by_key(
        _list(graph_payload.get("relations")),
        lambda row: "|".join(
            [
                str(row.get("subject") or "").strip().lower(),
                str(row.get("predicate") or "").strip().lower(),
                str(row.get("object") or "").strip().lower(),
            ]
        ),
    )

    paper_overview = _merge_overview(overview, claims_methods)
    return {
        "taxonomy_nodes": _merge_taxonomy(overview),
        "paper_overview": paper_overview,
        "knowledge_graph": graph_payload,
        "results": results,
        "methods": methods,
        "claims": claims,
        "research_facets": _merge_research_facets(research_facets),
        "key_findings": paper_overview.get("key_findings", []),
        "limitations_stated": paper_overview.get("limitations", []),
        "multi_agent_extraction": {
            "roles": sorted(role_outputs),
            "role_count": len(role_outputs),
        },
    }


def extract_paper_multi_agent(
    paper_id: str,
    title: str,
    taxonomy_hint: str,
    compact_text: str,
    *,
    llm_scope: Mapping[str, Any] | None = None,
) -> tuple[dict, int]:
    """Run role-specialized extraction and merge into legacy schema."""
    # Only the taxonomy reader is told to classify into leaf nodes, so it is the
    # only role that needs the leaf listing in its prompt.
    roles = [
        ("taxonomy_overview", TAXONOMY_OVERVIEW_SYSTEM, True),
        ("empirical_results", EMPIRICAL_RESULTS_SYSTEM, False),
        ("claims_methods", CLAIMS_METHODS_SYSTEM, False),
        ("graph_context", GRAPH_CONTEXT_SYSTEM, False),
        ("research_facets", RESEARCH_FACETS_SYSTEM, False),
    ]
    outputs: dict[str, dict] = {}
    total_tokens = 0
    errors: dict[str, str] = {}
    for role_name, system_prompt, needs_taxonomy in roles:
        user_prompt = _paper_user_prompt(
            paper_id,
            title,
            taxonomy_hint,
            compact_text,
            include_taxonomy=needs_taxonomy,
        )
        try:
            payload, tokens, _route = proposer_json(
                system_prompt,
                user_prompt,
                llm_scope=llm_scope,
                operation=f"paper_extraction:{paper_id}:{role_name}",
            )
            total_tokens += tokens
            outputs[role_name] = payload if isinstance(payload, dict) else {}
        except Exception as exc:
            errors[role_name] = str(exc)

    if errors:
        raise RuntimeError(
            "multi-agent extraction failed closed: "
            + "; ".join(f"{name}={error}" for name, error in sorted(errors.items()))
        )
    merged = merge_role_extractions(outputs)
    return merged, total_tokens
