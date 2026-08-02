"""Read old claims as evidence without importing legacy research backlog.

The legacy corpus may inform a new ResearchAgenda, but claims never acquire an
agenda_id and are never mutated.  This module selects a small, reproducible
paper set and creates one *new* agenda-scoped research problem whose evidence
links remain explicit.
"""

from __future__ import annotations

from typing import Any

from agents.agenda_repository import AgendaRepository
from agents.problem_first import problem_quality_score, upsert_research_problem
from db import database as db


HARNESS_EVIDENCE_TERMS = (
    "harness",
    "large language",
    "language model",
    "llm",
    "agentic",
    "multi-agent",
    "prompt",
    "tool use",
    "tool-use",
    "context window",
    "in-context",
    "reasoning model",
)
SELECTION_VERSION = "legacy_claim_triage_v1"


def _patterns() -> list[str]:
    return [f"%{term}%" for term in HARNESS_EVIDENCE_TERMS]


def select_harness_evidence(*, max_claims: int = 96, min_grounding: float = 0.7) -> list[dict[str, Any]]:
    """Return only IDs and quality metadata for relevant legacy evidence.

    The query deliberately does not return claim text, quotes, or paper text:
    downstream persistence needs only stable paper references.  Existing claim
    rows stay unscoped and unchanged.
    """
    limit = max(1, min(int(max_claims), 500))
    query = """
        SELECT c.id, c.paper_id, COALESCE(c.grounding_score, 0) AS grounding_score,
               c.grounding_status, p.published_date
        FROM claims AS c
        JOIN papers AS p ON p.id=c.paper_id
        WHERE lower(concat_ws(' ', COALESCE(c.claim_text, ''),
                                     COALESCE(c.method_name, ''),
                                     COALESCE(c.dataset_name, ''),
                                     COALESCE(c.conditions, ''))) LIKE ANY(?)
          AND (
              c.grounding_status IN ('grounded', 'verified')
              OR COALESCE(c.grounding_score, 0) >= ?
          )
        ORDER BY COALESCE(c.grounding_score, 0) DESC,
                 p.published_date DESC, c.id DESC
        LIMIT ?
    """
    return [dict(row) for row in db.fetchall(query, (_patterns(), min_grounding, limit))]


def build_initial_harness_problem(
    rows: list[dict[str, Any]],
    *,
    agenda_id: int,
    max_papers: int = 24,
) -> dict[str, Any]:
    """Build a bounded, evidence-linked problem without asserting a result."""
    paper_ids: list[str] = []
    claim_ids: list[int] = []
    seen_papers: set[str] = set()
    for row in rows:
        paper_id = str(row.get("paper_id") or "").strip()
        claim_id = row.get("id")
        if not paper_id or claim_id is None:
            continue
        claim_ids.append(int(claim_id))
        if paper_id not in seen_papers and len(paper_ids) < max(1, int(max_papers)):
            seen_papers.add(paper_id)
            paper_ids.append(paper_id)
    if not paper_ids:
        raise ValueError("no grounded harness evidence is available for initial triage")
    problem = {
        "problem_statement": (
            "Map and falsify recurring failure modes and resource-quality tradeoffs "
            "in self-improving LLM harnesses under verifier-grounded evaluation."
        ),
        "source_signal_ref": {
            "table": "claims",
            "selection": "agenda5_initial_harness_evidence",
            "selection_version": SELECTION_VERSION,
            "agenda_id": int(agenda_id),
            "evidence_claim_ids": claim_ids,
        },
        "node_ids": [],
        "paper_ids": paper_ids,
        "support_count": len(paper_ids),
        "status": "open",
        "attempts_count": 0,
        "ruled_out_approaches": [],
    }
    problem["problem_quality_score"] = problem_quality_score(problem)
    return problem


def seed_initial_harness_problem(
    *,
    agenda_id: int,
    max_claims: int = 96,
    max_papers: int = 24,
    repository: AgendaRepository | None = None,
) -> tuple[int, dict[str, Any]]:
    """Persist one new scoped problem; never import or edit legacy claims."""
    repo = repository or AgendaRepository()
    agenda = repo.get(int(agenda_id))
    if agenda is None or agenda.status != "active" or not agenda.is_active:
        raise ValueError("initial harness triage requires an active agenda")
    rows = select_harness_evidence(max_claims=max_claims)
    problem = build_initial_harness_problem(
        rows,
        agenda_id=int(agenda_id),
        max_papers=max_papers,
    )
    problem_id = upsert_research_problem(problem, agenda_id=int(agenda_id))
    problem["id"] = problem_id
    return problem_id, problem
