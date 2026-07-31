"""Agenda-isolated candidate ranking.

This selector produces a feature-backed selection only. It cannot allocate
tokens, GPUs, or a ComputeBackend.
"""

from __future__ import annotations

import json
from typing import Any

from agents.agenda_relevance import agenda_scope_terms, candidate_scope_text, insight_in_scope
from agents.agenda_repository import AgendaRepository
from contracts.agenda import AgendaSelection, ResearchAgenda


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def score_candidate(
    candidate: dict[str, Any],
    agenda: ResearchAgenda,
) -> tuple[float, dict[str, Any], list[str]]:
    if not insight_in_scope(candidate, agenda):
        return float("-inf"), {"scope": 0.0}, ["agenda_scope_mismatch"]
    text = candidate_scope_text(candidate)
    terms = agenda_scope_terms(agenda)
    term_hits = [term for term in terms if term in text]
    score = min(1.0, len(term_hits) / max(1, len(terms))) * 0.45
    breakdown: dict[str, Any] = {
        "scope": score,
        "matched_terms": term_hits,
    }
    resource = str(candidate.get("resource_class") or "cpu")
    preferred_resources = list((agenda.prefer or {}).get("resource_class") or [])
    if preferred_resources:
        resource_score = 0.2 if resource in preferred_resources else -0.4
        score += resource_score
        breakdown["resource_preference"] = resource_score
    novelty = _json_obj(candidate.get("novelty_report"))
    novelty_score = float(novelty.get("score") or candidate.get("adversarial_score") or 0)
    novelty_feature = max(0.0, min(novelty_score, 10.0)) / 10.0 * 0.2
    score += novelty_feature
    breakdown["novelty_feature"] = novelty_feature
    experimentability = str(candidate.get("experimentability") or "")
    if experimentability in {"easy", "medium"}:
        score += 0.15
        breakdown["feedback_speed"] = 0.15
    rejected: list[str] = []
    for phrase in list((agenda.reject or {}).get("keywords") or []):
        if str(phrase).lower() in text:
            rejected.append(f"reject_keyword:{phrase}")
    if rejected:
        return float("-inf"), breakdown, rejected
    return score, breakdown, []


def select_next(
    agenda_id: int,
    *,
    repository: AgendaRepository | None = None,
    limit: int = 100,
) -> AgendaSelection | None:
    repo = repository or AgendaRepository()
    agenda = repo.get(agenda_id)
    if agenda is None or agenda.status != "active":
        return None
    ranked: list[tuple[float, dict[str, Any], dict[str, Any], list[str]]] = []
    for candidate in repo.candidates(agenda_id, limit=limit):
        score, breakdown, blockers = score_candidate(candidate, agenda)
        ranked.append((score, candidate, breakdown, blockers))
    ranked.sort(key=lambda item: (-item[0], int(item[1].get("id") or 0)))
    accepted = next((item for item in ranked if item[0] != float("-inf")), None)
    if accepted is None:
        return None
    score, candidate, breakdown, _ = accepted
    rejected = [
        {
            "insight_id": int(row.get("id") or 0),
            "score": None if item_score == float("-inf") else item_score,
            "reason_codes": blockers,
        }
        for item_score, row, _parts, blockers in ranked
        if int(row.get("id") or 0) != int(candidate.get("id") or 0)
    ]
    selection = AgendaSelection(
        agenda_id=agenda_id,
        selected_insight_id=int(candidate["id"]),
        score=score,
        rationale="agenda_scope_and_feature_score",
        rejected_candidates=rejected,
        scoring_breakdown=breakdown,
    )
    repo.save_selection(selection)
    repo.queue_selected_insight(selection)
    return selection
