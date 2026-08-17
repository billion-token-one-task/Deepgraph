"""Extraction Agent: scientific paper -> taxonomy classification, evidence rows, and graph evidence."""
from collections.abc import Mapping, Sequence
from typing import Any

from db.taxonomy import get_all_leaf_ids, get_leaf_ids_under, get_nodes_to_depth
from config import (
    EXTRACTION_MAX_PROMPT_CHARS,
    EXTRACTION_ROUTING_HEAD_CHARS,
    EXTRACTION_TAXONOMY_LEAF_BUDGET_CHARS,
    EXTRACTION_TAXONOMY_ROUTING_DEPTH,
    EXTRACTION_TAXONOMY_ROUTING_ENABLED,
    MULTI_AGENT_EXTRACTION_ENABLED,
)
from meta_harness.scoped_llm import proposer_json

SYSTEM_PROMPT = """You are a scientific paper analysis agent. Extract structured, concrete information from research papers.

You will be given a paper and a taxonomy of scientific research areas (leaf node IDs). Your job:

1. CLASSIFY the paper into 1-3 leaf taxonomy nodes (pick the most specific match)
2. EXTRACT every (method, dataset, metric, value) result tuple from the paper
3. EXTRACT methods proposed or used
4. NOTE key findings and limitations

Return a JSON object:
{
  "taxonomy_nodes": [
    {"node_id": "ml.dl.cv.detection", "confidence": 0.95}
  ],
  "paper_overview": {
    "plain_summary": "In simple language, what this paper is trying to do and why",
    "problem_statement": "What concrete problem or bottleneck the paper addresses",
    "approach_summary": "How the paper approaches the problem in plain language",
    "work_type": "model|training_method|benchmark|dataset|application|system|analysis|theory",
    "key_findings": ["Finding 1", "Finding 2"],
    "limitations": ["Limitation 1", "Limitation 2"],
    "open_questions": ["Open question 1"]
  },
  "knowledge_graph": {
    "entities": [
      {
        "name": "Transformer",
        "entity_type": "method",
        "description": "Sequence model based on attention",
        "aliases": ["attention-only model"],
        "mention_role": "used",
        "confidence": 0.95,
        "evidence_location": "Section 2",
        "source_text": "We use a Transformer backbone..."
      }
    ],
    "relations": [
      {
        "subject": "Transformer",
        "subject_type": "method",
        "predicate": "evaluated_on",
        "object": "WMT14 En-De",
        "object_type": "dataset",
        "confidence": 0.94,
        "evidence_location": "Table 1",
        "source_text": "Transformer is evaluated on WMT14 En-De."
      }
    ]
  },
  "results": [
    {
      "method_name": "YOLOv9",
      "dataset_name": "COCO val2017",
      "metric_name": "mAP@0.5:0.95",
      "metric_value": 55.6,
      "metric_unit": "%",
      "is_sota": 1,
      "evidence_location": "Table 2",
      "source_quote": "Verbatim sentence or table caption line from the Full text below that states this number (copy character-for-character, at least 12 characters)."
    }
  ],
  "methods": [
    {
      "name": "YOLOv9",
      "category": "object detection",
      "description": "Real-time object detector with PGI and GELAN",
      "key_innovation": "Programmable Gradient Information for better training",
      "builds_on": ["YOLOv7", "ELAN"]
    }
  ],
  "claims": [
    {
      "claim_text": "YOLOv9 achieves 55.6% mAP on COCO with 50% fewer parameters than YOLOv8",
      "claim_type": "performance",
      "method_name": "YOLOv9",
      "dataset_name": "COCO val2017",
      "metric_name": "mAP@0.5:0.95",
      "metric_value": 55.6,
      "evidence_location": "Table 2",
      "source_quote": "A contiguous verbatim substring from the Full text below that supports this claim (same wording as in the paper, at least 12 characters). Do not paraphrase.",
      "conditions": {"model_size": "large", "input_resolution": "640x640"}
    }
  ],
  "key_findings": ["Finding 1", "Finding 2"],
  "limitations_stated": ["Limitation 1"]
}

Rules:
- For taxonomy_nodes: use ONLY the provided leaf node IDs. Pick 1-3 most relevant.
- For paper_overview: write for a technically curious non-expert. Avoid unexplained jargon.
- work_type must be one of: model, training_method, benchmark, dataset, application, system, analysis, theory
- For knowledge_graph entities, use one of:
  concept, method, task, dataset, metric, artifact, material, gene, protein, disease, organism, theory
- For knowledge_graph predicates, prefer:
  uses, builds_on, evaluated_on, measured_by, compares_with, applied_to,
  improves_over, part_of, studies, predicts, treats, interacts_with, derived_from, related_to
- Keep entities canonical and reusable across papers. Do not create separate entities for trivial ablations unless the paper treats them as named systems.
- Add source_text only as a short grounding snippet, not a long quote.
- For results: extract EVERY quantitative result from tables and text. Each row in a results table = one result entry.
- method_name should be the exact name used in the paper (e.g. "GPT-4o", "LLaMA-3-70B", "ResNet-50")
- dataset_name should be the standard benchmark name (e.g. "ImageNet-1K", "COCO val2017", "MMLU")
- metric_name should be standard (e.g. "accuracy", "mAP@0.5", "BLEU", "F1", "perplexity")
- metric_value should be numeric. If a percentage, store the number (95.2 not 0.952)
- is_sota=1 only if the paper explicitly claims state-of-the-art
- Extract ALL results, not just the best ones
- Limitations and open questions should be grounded in the paper, not invented.
- GROUNDING (required for every item in "claims" and "results"):
  - Include "source_quote" for each claim and each result row: copy a contiguous substring EXACTLY as it appears in the Full text below (character-for-character). Minimum length 12 characters. The quote must appear in that Full text (it will be verified automatically).
  - Choose a quote that directly supports the numeric result or the claim; table cells may be given as a single line from the table or its caption.
  - If you cannot find supporting text, omit that claim or result row rather than inventing a quote.
- Return ONLY valid JSON, no markdown formatting"""

TAXONOMY_ROUTING_SYSTEM_PROMPT = """You route a paper to the research areas it belongs to.

You are given a coarse slice of a research taxonomy (area IDs and names) and the
opening of a paper. Pick the 1-3 areas whose subtrees are most likely to contain
the paper's precise topic. Prefer the most specific area you are confident in; a
broader area is better than a wrong narrow one.

Return ONLY valid JSON:
{"areas": ["ml.dl.nlp", "ml.dl.foundation"]}

Rules:
- Use ONLY area IDs listed by the user, copied exactly.
- 1 area when the paper is clearly single-topic, up to 3 when it spans areas.
- No prose, no markdown."""

MAX_PROMPT_CHARS = EXTRACTION_MAX_PROMPT_CHARS
PRIORITY_SECTION_KEYWORDS = (
    "abstract",
    "introduction",
    "method",
    "approach",
    "architecture",
    "experiment",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "limitation",
)


def _compact_paper_text(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Trim long PDFs into a high-signal slice for extraction."""
    clean = (text or "").strip()
    if len(clean) <= max_chars:
        return clean

    lines = [line.strip() for line in clean.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return clean[:max_chars]

    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        lower = line.lower()
        looks_like_heading = (
            len(line) <= 80
            and sum(ch.isalpha() for ch in line) >= 3
            and (line == line.upper() or any(word in lower for word in PRIORITY_SECTION_KEYWORDS))
        )
        if looks_like_heading and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    # Always keep an opening window for title/abstract/introduction context.
    kept: list[str] = [clean[: min(6000, max_chars // 3)]]
    seen = {kept[0]}
    budget = len(kept[0])

    def try_add(block: str) -> None:
        nonlocal budget
        block = block.strip()
        if not block or block in seen or budget >= max_chars:
            return
        remaining = max_chars - budget
        clipped = block[:remaining]
        kept.append(clipped)
        seen.add(block)
        budget += len(clipped)

    for section in sections:
        lower = section.lower()
        if any(keyword in lower for keyword in PRIORITY_SECTION_KEYWORDS):
            try_add(section)

    # Preserve the tail as many papers place ablations / limitations near the end.
    if budget < max_chars:
        try_add(clean[-min(5000, max_chars - budget):])

    return "\n\n".join(kept)[:max_chars]


def format_taxonomy_hint(leaf_ids: Sequence[str]) -> tuple[str, bool]:
    """Render a leaf-ID hint inside the character budget.

    Truncation is returned rather than swallowed: a hint that silently dropped
    half the taxonomy looks exactly like one that had room for all of it, and
    this pipeline has already been bitten once by a cap nobody could see.
    """
    unique = sorted({str(leaf_id).strip() for leaf_id in leaf_ids if str(leaf_id).strip()})
    budget = max(int(EXTRACTION_TAXONOMY_LEAF_BUDGET_CHARS), 0)
    kept: list[str] = []
    used = 0
    # Shortest IDs first: those are the more general nodes, so overrunning the
    # budget costs precision instead of dropping an entire branch.
    for leaf_id in sorted(unique, key=lambda value: (len(value), value)):
        cost = len(leaf_id) + 3
        if budget and used + cost > budget and kept:
            break
        kept.append(leaf_id)
        used += cost
    hint = "Available taxonomy leaf nodes:\n" + "\n".join(
        f"  {nid}" for nid in sorted(kept)
    )
    return hint, len(kept) < len(unique)


def _route_to_leaf_ids(
    paper_id: str,
    title: str,
    compact_text: str,
    llm_scope: Mapping[str, Any] | None,
) -> tuple[list[str], dict]:
    """Narrow ~4k taxonomy leaves to the branch this paper actually belongs to.

    The full leaf list costs several times more than the paper being classified,
    and only the taxonomy reader ever needs it.  Routing on the paper's opening
    is enough to pick a branch, so the expensive list is never sent whole.
    """
    nodes = get_nodes_to_depth(EXTRACTION_TAXONOMY_ROUTING_DEPTH)
    if not nodes:
        return get_all_leaf_ids(), {"taxonomy_routing": "unavailable"}

    listing = "\n".join(f"  {node['id']} | {node['name']}" for node in nodes)
    head = (compact_text or "").strip()[: max(int(EXTRACTION_ROUTING_HEAD_CHARS), 0)]
    user_prompt = (
        f"Candidate research areas:\n{listing}\n\n"
        f"Paper ID: {paper_id}\n"
        f"Title: {title}\n\n"
        f"Paper opening:\n{head}"
    )
    routed, tokens, _route = proposer_json(
        TAXONOMY_ROUTING_SYSTEM_PROMPT,
        user_prompt,
        llm_scope=llm_scope,
        operation=f"paper_extraction:{paper_id}:taxonomy_routing",
    )
    known = {str(node["id"]) for node in nodes}
    areas = (routed or {}).get("areas") if isinstance(routed, dict) else None
    picked = [
        str(area).strip()
        for area in (areas or [])
        if str(area).strip() in known
    ]
    leaf_ids = get_leaf_ids_under(picked) if picked else []
    if not leaf_ids:
        return get_all_leaf_ids(), {
            "taxonomy_routing": "fallback_full",
            "taxonomy_routing_tokens": tokens,
        }
    return leaf_ids, {
        "taxonomy_routing": "routed",
        "taxonomy_areas": picked,
        "taxonomy_routing_tokens": tokens,
    }


def resolve_taxonomy_hint(
    paper_id: str,
    title: str,
    compact_text: str,
    llm_scope: Mapping[str, Any] | None,
) -> tuple[str, dict]:
    """Return the taxonomy hint for this paper plus the routing audit trail."""
    if not EXTRACTION_TAXONOMY_ROUTING_ENABLED:
        leaf_ids, meta = get_all_leaf_ids(), {"taxonomy_routing": "disabled"}
    else:
        try:
            leaf_ids, meta = _route_to_leaf_ids(paper_id, title, compact_text, llm_scope)
        except Exception as exc:
            # Routing is an optimisation.  Losing it must cost tokens, not the
            # paper: fall back to the full list, which the budget still caps.
            leaf_ids = get_all_leaf_ids()
            meta = {
                "taxonomy_routing": "error",
                "taxonomy_routing_error": str(exc)[:200],
            }
    hint, truncated = format_taxonomy_hint(leaf_ids)
    meta["taxonomy_leaf_count"] = len(leaf_ids)
    meta["taxonomy_hint_truncated"] = truncated
    return hint, meta


def extract_paper(
    paper_id: str,
    title: str,
    text: str,
    *,
    llm_scope: Mapping[str, Any] | None = None,
) -> tuple[dict, int]:
    """Extract structured info from a paper. Returns (extraction_dict, tokens_used)."""
    compact_text = _compact_paper_text(text)
    taxonomy_hint, routing_meta = resolve_taxonomy_hint(
        paper_id, title, compact_text, llm_scope
    )
    routing_tokens = int(routing_meta.get("taxonomy_routing_tokens") or 0)

    if MULTI_AGENT_EXTRACTION_ENABLED:
        from agents.multi_agent_extraction import extract_paper_multi_agent

        result, tokens = extract_paper_multi_agent(
            paper_id,
            title,
            taxonomy_hint,
            compact_text,
            llm_scope=llm_scope,
        )
        result.setdefault("extraction_mode", "multi_agent")
        result.setdefault("prompt_routing", routing_meta)
        return result, tokens + routing_tokens

    user_prompt = f"""{taxonomy_hint}

Paper ID: {paper_id}
Title: {title}

Full text:
{compact_text}"""

    result, tokens, _route = proposer_json(
        SYSTEM_PROMPT,
        user_prompt,
        llm_scope=llm_scope,
        operation=f"paper_extraction:{paper_id}:monolithic",
    )
    if isinstance(result, dict):
        result.setdefault("extraction_mode", "monolithic")
        result.setdefault("prompt_routing", routing_meta)
    return result, tokens + routing_tokens
