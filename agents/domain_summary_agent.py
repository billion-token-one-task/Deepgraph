"""Generate plain-language domain summaries and research opportunities."""
from agents.llm_client import call_llm_json


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


def generate_domain_summary(node: dict, snapshot: dict) -> tuple[dict, int]:
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
        if paper.get("limitations"):
            line += f" | limitations: {'; '.join(paper['limitations'][:2])}"
        if paper.get("open_questions"):
            line += f" | open questions: {'; '.join(paper['open_questions'][:2])}"
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
    limitation_lines = [f"- {text}" for text in snapshot.get("limitations", [])[:12]]
    question_lines = [f"- {text}" for text in snapshot.get("open_questions", [])[:12]]
    matrix_gap_lines = [
        f"- {gap['method_name']} on {gap['dataset_name']}: {gap['gap_description']}"
        for gap in snapshot.get("matrix_gaps", [])[:8]
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

Recurring limitations:
{chr(10).join(limitation_lines) or "- none"}

Open questions stated in papers:
{chr(10).join(question_lines) or "- none"}

Existing matrix-style gaps:
{chr(10).join(matrix_gap_lines) or "- none"}

Write a plain-language landscape summary for this area."""

    return call_llm_json(SYSTEM_PROMPT, user_prompt)


def fallback_domain_summary(node: dict, snapshot: dict) -> dict:
    """Fallback summary when the LLM is unavailable."""
    children = snapshot.get("children", [])
    work_types = snapshot.get("work_types", [])
    methods = [item["name"] for item in snapshot.get("methods", [])[:5]]
    datasets = [item["name"] for item in snapshot.get("datasets", [])[:5]]
    limitations = snapshot.get("limitations", [])
    open_questions = snapshot.get("open_questions", [])

    overview_parts = []
    if node.get("description"):
        overview_parts.append(node["description"])
    if children:
        child_names = ", ".join(child["name"] for child in children[:4])
        overview_parts.append(f"The main sub-areas currently visible are {child_names}.")
    if snapshot.get("paper_count", 0):
        overview_parts.append(
            f"The system currently has evidence from {snapshot['paper_count']} papers in this area."
        )
    overview = " ".join(overview_parts) or f"{node['name']} is an active machine learning area."

    workstreams = []
    for item in work_types[:3]:
        workstreams.append({
            "label": item["work_type"].replace("_", " ").title(),
            "description": f"A recurring kind of work in this area is {item['work_type'].replace('_', ' ')}.",
            "paper_count": item["count"],
        })
    if not workstreams and children:
        for child in children[:3]:
            workstreams.append({
                "label": child["name"],
                "description": child.get("description") or f"Researchers are active in {child['name']}.",
                "paper_count": child.get("paper_count", 0),
            })

    gaps = []
    for text in limitations[:2]:
        gaps.append({
            "title": "Known limitation",
            "description": text,
            "why_now": "Multiple papers in this area are already reporting this as a bottleneck.",
        })
    for text in open_questions[:2]:
        if len(gaps) >= 3:
            break
        gaps.append({
            "title": "Open question",
            "description": text,
            "why_now": "This is explicitly named by authors as unresolved work.",
        })

    if not gaps and datasets:
        gaps.append({
            "title": "Broader evaluation coverage",
            "description": f"Current work appears concentrated on a small set of benchmarks such as {', '.join(datasets[:3])}.",
            "why_now": "Broader testing is often the fastest way to reveal what really generalizes.",
        })

    return {
        "overview": overview,
        "why_it_matters": f"{node['name']} matters because it changes what machine learning systems can do in practice.",
        "what_people_are_building": workstreams,
        "common_patterns": [item["work_type"].replace("_", " ") for item in work_types[:4]],
        "common_methods": methods,
        "common_datasets": datasets,
        "current_gaps": gaps,
        "starter_questions": open_questions[:3] or [
            "Which settings are still under-tested?",
            "What fails outside the main benchmark?",
        ],
    }
