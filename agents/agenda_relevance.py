"""Agenda direction guardrails: prompt constraint block + deterministic scope gate.

PR #41 scoped what the Tier 1 / Tier 2 agents *read* (signal queries circled to
the agenda's taxonomy subgraph), but nothing constrained what they *write*: the
generation prompts never mentioned the agenda, so off-topic candidates passed
through whenever the taxonomy match was loose or fell back to the global scan.

This module adds the missing two pieces, both rule-based (no extra LLM calls):

1. ``agenda_constraint_block(agenda)`` — a prompt section appended to every
   generation prompt when an agenda is present, stating the user's direction
   verbatim plus the scope keywords, and instructing the model to stay inside.
2. ``insight_in_scope(insight, agenda)`` — a post-generation keyword gate.
   Generated insights whose text matches none of the agenda's scope terms are
   dropped before storage and reported as ``dropped_out_of_scope``.

The gate is intentionally lenient: by default one term hit is enough to keep
an insight (configurable via AGENDA_SCOPE_MIN_TERM_HITS). Its job is to catch
clearly unrelated output, not to rank borderline cases — prompt steering does
the fine-grained work, the gate is the deterministic backstop.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from config import AGENDA_SCOPE_MIN_TERM_HITS

# Tokens that appear in almost any ML research direction. Useless as scope
# evidence when auto-extracted from free text, so they are skipped during
# tokenization. Explicit keyword phrases (focus / prefer.keywords) are always
# kept verbatim regardless of this list.
_GENERIC_TOKENS = {
    "and", "are", "based", "between", "data", "deep", "for", "from", "into",
    "learning", "machine", "method", "methods", "model", "models", "new",
    "novel", "over", "paper", "papers", "research", "task", "tasks", "that",
    "the", "this", "toward", "towards", "under", "use", "using", "via",
    "with",
}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")

# Fields whose text represents what a generated insight is about. Covers both
# tiers: Tier 2 ideas carry title/problem_statement/proposed_method, Tier 1
# paradigm insights carry title/formal_structure/transformation.
_SCOPE_TEXT_FIELDS = (
    "title",
    "problem_statement",
    "proposed_method",
    "formal_structure",
    "transformation",
)


def _tokens(text: Any) -> list[str]:
    return [
        tok
        for tok in _TOKEN_RE.findall(str(text or "").lower())
        if tok not in _GENERIC_TOKENS
    ]


def agenda_match_terms(agenda) -> list[str]:
    """Lowercased scope terms for the relevance gate.

    Combines, deduplicated and in order:
    - focus + prefer.keywords phrases, verbatim (the user named these);
    - the individual tokens of those phrases (so "outlier rejection" also
      matches text that only says "outlier");
    - tokens extracted from the direction description (catches terms the
      user wrote in the free-text direction but not in the keyword list).
    """
    from agents.agenda_selector import agenda_scope_keywords

    seen: set[str] = set()
    terms: list[str] = []

    def _add(term: str) -> None:
        term = term.strip().lower()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)

    for phrase in agenda_scope_keywords(agenda):
        _add(phrase)
        for tok in _tokens(phrase):
            _add(tok)
    for tok in _tokens(getattr(agenda, "description", "")):
        _add(tok)
    return terms


def agenda_constraint_block(agenda) -> str:
    """Prompt section that pins generation to the agenda's direction.

    Appended to the user prompt of every generation call when an agenda is
    present; without an agenda the prompts are untouched.
    """
    from agents.agenda_selector import agenda_scope_keywords

    direction = str(getattr(agenda, "description", "") or "").strip()
    if not direction:
        direction = str(getattr(agenda, "name", "") or "").strip()
    lines = [
        "",
        "# RESEARCH DIRECTION CONSTRAINT (hard requirement)",
        "",
        "All output must stay inside this user-defined research direction:",
        "",
        f"Direction: {direction}",
    ]
    keywords = agenda_scope_keywords(agenda)
    if keywords:
        lines.append(f"Scope keywords: {', '.join(keywords)}")
    lines.extend(
        [
            "",
            "Rules:",
            "- Only propose problems, insights, and methods that fall inside this direction.",
            "- Ignore signals and evidence unrelated to the direction, even if they look promising.",
            "- If little of the evidence fits the direction, return fewer items rather than drifting off-topic.",
        ]
    )
    return "\n".join(lines)


def insight_scope_text(insight: dict) -> str:
    """Lowercased text of the fields that describe what an insight is about."""
    parts = []
    for field in _SCOPE_TEXT_FIELDS:
        value = insight.get(field)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def count_term_hits(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def insight_in_scope(insight: dict, agenda, *, min_hits: int | None = None) -> bool:
    """Deterministic check that a generated insight matches the agenda's scope.

    Lenient by design (default: one term hit keeps the insight). Disabled —
    everything passes — when there is no agenda, when the threshold is set to
    zero, or when the agenda yields no ASCII-matchable term (e.g. a direction
    written entirely in Chinese cannot be matched against English insight
    text, and dropping everything would be worse than dropping nothing).
    """
    if agenda is None:
        return True
    if min_hits is None:
        min_hits = AGENDA_SCOPE_MIN_TERM_HITS
    if min_hits <= 0:
        return True
    terms = agenda_match_terms(agenda)
    if not any(re.search(r"[a-z0-9]", term) for term in terms):
        return True
    return count_term_hits(insight_scope_text(insight), terms) >= min_hits
