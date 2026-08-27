"""Pure prompt and deterministic scope guard for agenda-bound candidates."""

from __future__ import annotations

import re
from typing import Any

from contracts.agenda import ResearchAgenda


_GENERIC = {
    "research",
    "model",
    "method",
    "data",
    "learning",
    "paper",
    "task",
}


def agenda_scope_terms(agenda: ResearchAgenda) -> list[str]:
    terms: list[str] = []
    for raw in list(agenda.focus) + list((agenda.prefer or {}).get("keywords") or []):
        text = str(raw or "").strip().lower()
        if text and text not in _GENERIC and text not in terms:
            terms.append(text)
    return terms


def agenda_constraint_block(agenda: ResearchAgenda) -> str:
    agenda.validate()
    return "\n".join(
        [
            "# RESEARCH AGENDA (hard scope)",
            f"agenda_id: {agenda.agenda_id}",
            f"direction: {agenda.description or agenda.name}",
            f"scope terms: {', '.join(agenda_scope_terms(agenda))}",
            "Generate only objects inside this agenda.",
            "Return fewer candidates rather than drifting out of scope.",
            "Do not consume or relabel unscoped legacy backlog.",
        ]
    )


# What the candidate asserts it will do. The reject list is a statement about
# method -- "no weight updates here" -- so it must be matched against the
# candidate's own claim, not against every sentence in the file.
_CLAIM_FIELDS = (
    "title",
    "problem_statement",
    "formal_structure",
    "transformation",
    "proposed_method",
)


def candidate_claim_text(candidate: dict[str, Any]) -> str:
    """The fields where a candidate says what it proposes to do.

    ``candidate_scope_text`` deliberately includes the whole experimental and
    evidence plan, because an older candidate often names its agenda's terms
    only there. That breadth is right for asking "is this in scope"; it is
    wrong for asking "does this do a forbidden thing", because those plans
    carry risk registers and expected-result prose that discuss the forbidden
    thing in order to disclaim it. On 2026-08-27 five candidates were refused
    642 times in one day, every one of them for a sentence like "without
    requiring weight updates or fine-tuning" -- three of the five matched
    inside a risks entry. The matcher sees a phrase, not a negation, so the
    only fix that does not need it to read English is to stop showing it prose
    that was never a claim.
    """
    return " ".join(
        str(candidate.get(field) or "") for field in _CLAIM_FIELDS
    ).lower()


def candidate_scope_text(candidate: dict[str, Any]) -> str:
    fields = (
        "title",
        "problem_statement",
        "formal_structure",
        "transformation",
        "proposed_method",
        # Older candidates often put agenda-specific terms in the declared
        # experiment rather than repeating them in the method prose.
        "experimental_plan",
        "evidence_plan",
    )
    return " ".join(
        str(candidate.get(field) or "") for field in fields
    ).lower()


def insight_in_scope(
    candidate: dict[str, Any],
    agenda: ResearchAgenda,
    *,
    minimum_hits: int = 1,
) -> bool:
    if int(candidate.get("agenda_id") or 0) != int(agenda.agenda_id or 0):
        return False
    terms = agenda_scope_terms(agenda)
    if not terms:
        return False
    text = candidate_scope_text(candidate)
    hits = sum(1 for term in terms if re.search(re.escape(term), text))
    return hits >= max(1, int(minimum_hits))
