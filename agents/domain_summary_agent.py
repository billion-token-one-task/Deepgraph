"""Generate plain-language domain summaries and research opportunities."""

from collections.abc import Mapping
from typing import Any

from meta_harness.scoped_llm import proposer_json

SYSTEM_PROMPT = """You explain research landscapes to non-specialists.

Goal: help a curious outsider understand:
1. what this area is about,
2. what kinds of work people are doing,
3. where the meaningful gaps are.

Return JSON:
{
  "overview": "2-3 sentence plain-language overview",
  "why_it_matters": "1-2 sentence explanation of why this area matters",
  "what_people_are_building": [
    {
      "label": "short theme name",
      "description": "plain-language description of the workstream",
      "paper_count": 0
    }
  ],
  "common_patterns": ["short pattern", "short pattern"],
  "common_methods": ["method or system names"],
  "common_datasets": ["benchmark or dataset names"],
  "current_gaps": [
    {
      "title": "short gap title",
      "description": "what is missing in plain language",
      "why_now": "why this is worth doing now"
    }
  ],
  "starter_questions": ["concrete question", "concrete question"]
}

Rules:
- Write for a technical reader who is not a specialist.
- Prefer themes over paper-by-paper summaries.
- Keep jargon low; if jargon is necessary, explain it in simple words.
- Ground gaps in the evidence provided. Do not invent speculative gaps with no support.
- Keep each field concise and readable.
- Return ONLY valid JSON."""


def generate_domain_summary(
    node: dict,
    snapshot: dict,
    *,
    llm_scope: Mapping[str, Any] | None = None,
) -> tuple[dict, int]:
    """Generate a node summary from structured evidence."""
    papers = snapshot.get("papers", [])
    paper_lines = []
    for idx, paper in enumerate(papers[:12], start=1):
        line = f"{idx}. {paper['title']}"
        if paper.get("plain_summary"):
            line += f" | summary: {paper['plain_summary']}"
        if paper.get("problem_statement"):
            line += f" | problem: {paper['problem_statement']}"
        if paper.get("approach_summary"):
            line += f" | approach: {paper['approach_summary']}"
        if paper.get("work_type"):
            line += f" | type: {paper['work_type']}"
        if paper.get("limitations") and isinstance(paper["limitations"], list):
            line += f" | limitations: {'; '.join(str(l) for l in paper['limitations'][:2])}"
        if paper.get("open_questions") and isinstance(paper["open_questions"], list):
            line += f" | open questions: {'; '.join(str(q) for q in paper['open_questions'][:2])}"
        paper_lines.append(line)

    child_lines = [
        f"- {child['name']}: {child.get('paper_count', 0)} papers, {child.get('method_count', 0)} methods"
        for child in snapshot.get("children", [])[:12]
    ]
    work_type_lines = [
        f"- {item['work_type']}: {item['count']} papers"
        for item in snapshot.get("work_types", [])[:10]
    ]
    method_lines = [
        f"- {item['name']}: {item['paper_count']} papers"
        for item in snapshot.get("methods", [])[:12]
    ]
    dataset_lines = [
        f"- {item['name']}: {item['paper_count']} papers"
        for item in snapshot.get("datasets", [])[:12]
    ]
    graph_summary = snapshot.get("graph_summary", {})
    entity_lines = [
        f"- {item['name']} ({item['entity_type']}): {item['paper_count']} papers"
        for item in graph_summary.get("top_entities", [])[:12]
    ]
    relation_lines = [
        f"- {item['subject']} --{item['predicate']}--> {item['object']} ({item['paper_count']} papers)"
        for item in graph_summary.get("top_relations", [])[:12]
    ]
    cluster_lines = [
        f"- {item['label']}: {item['paper_count']} papers; shared entities: {', '.join(item.get('shared_entities', [])[:3])}"
        for item in snapshot.get("paper_clusters", [])[:8]
    ]
    limitation_lines = [f"- {text}" for text in snapshot.get("limitations", [])[:12]]
    question_lines = [f"- {text}" for text in snapshot.get("open_questions", [])[:12]]
    matrix_gap_lines = [
        f"- {gap['method_name']} on {gap['dataset_name']}: {gap['gap_description']}"
        for gap in snapshot.get("matrix_gaps", [])[:8]
    ]
    opportunity_lines = [
        f"- {item['title']}: {item['description']} [score={item.get('value_score', 0)}]"
        for item in snapshot.get("opportunities", [])[:8]
    ]

    user_prompt = f"""Node: {node['name']} ({node['id']})
Description: {node.get('description', '')}
Paper count in this node: {snapshot.get('paper_count', 0)}
Result count in this node: {snapshot.get('result_count', 0)}

Child areas:
{chr(10).join(child_lines) or "- none"}

Work type distribution:
{chr(10).join(work_type_lines) or "- unknown"}

Representative papers:
{chr(10).join(paper_lines) or "- none"}

Common methods/systems:
{chr(10).join(method_lines) or "- none"}

Common datasets/benchmarks:
{chr(10).join(dataset_lines) or "- none"}

Core entities in this area:
{chr(10).join(entity_lines) or "- none"}

Frequent graph relations:
{chr(10).join(relation_lines) or "- none"}

Paper clusters:
{chr(10).join(cluster_lines) or "- none"}

Recurring limitations:
{chr(10).join(limitation_lines) or "- none"}

Open questions stated in papers:
{chr(10).join(question_lines) or "- none"}

Existing matrix-style gaps:
{chr(10).join(matrix_gap_lines) or "- none"}

Deterministic opportunity signals:
{chr(10).join(opportunity_lines) or "- none"}

Write a plain-language landscape summary for this area."""

    result, tokens, _route = proposer_json(
        SYSTEM_PROMPT,
        user_prompt,
        llm_scope=llm_scope,
        operation=f"domain_summary:{node['id']}",
    )
    return result, tokens
