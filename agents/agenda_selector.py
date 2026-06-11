"""Agenda selector: pick the best deep_insight for a given ResearchAgenda.

Issue #9: agenda-driven autonomous research loop, selection step.

Public API:
- score_insight(agenda, insight) -> dict          # full scoring breakdown
- evaluate_candidates(agenda, insights) -> dict   # selected + rejected
- select_and_persist(agenda, *, limit=200) -> AgendaSelection
- update_selection_progress(selection_id, **fields) -> None
- get_selection(selection_id) -> dict | None
- get_latest_selection(agenda_id=None) -> dict | None

Scoring model (all weights tunable as constants):
- Hard reject filters: reject.statuses / reject.novelty_status / reject.keywords
- Score gates: prefer.paradigm_score_min (uses adversarial_score/10),
               prefer.feasibility_score_min (uses experimentability map)
- Positive signals: focus keywords in title/problem_statement/formal_structure,
                    prefer.keywords match, prefer.tiers match, prefer.resource_class,
                    adversarial_score, experimentability
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from contracts.agenda import VALID_SELECTION_STATUS, AgendaSelection, ResearchAgenda
from contracts.base import ContractValidationError, ensure_dict, ensure_list, ensure_string_list
from db import database as db


# ---------- scoring weights ----------

W_FOCUS_KEYWORD = 0.15
W_FOCUS_MAX = 0.45
W_PREFER_KEYWORD = 0.10
W_PREFER_KEYWORD_MAX = 0.30
W_PREFER_TIER = 0.15
W_PREFER_RESOURCE = 0.10
W_PARADIGM = 0.20  # multiplied by adversarial_score/10
W_FEASIBILITY = 0.10  # multiplied by feasibility map value

EXPERIMENTABILITY_MAP = {
    "easy": 1.0,
    "medium": 0.6,
    "hard": 0.3,
    "": 0.5,
}


# ---------- helpers ----------


def _text_blob(insight: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "problem_statement", "formal_structure", "existing_weakness", "transformation"):
        value = insight.get(key)
        if value:
            parts.append(str(value))
    # proposed_method may be JSON string
    pm = insight.get("proposed_method")
    if isinstance(pm, str) and pm.strip():
        parts.append(pm)
    elif isinstance(pm, dict):
        parts.append(json.dumps(pm, ensure_ascii=False))
    return " ".join(parts).lower()


def _tier_label(tier: Any) -> str:
    try:
        return f"tier_{int(tier)}"
    except (TypeError, ValueError):
        return "tier_unknown"


def _paradigm_score(insight: Mapping[str, Any]) -> float:
    raw = insight.get("adversarial_score")
    try:
        v = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(v / 10.0, 1.0))


def _feasibility_score(insight: Mapping[str, Any]) -> float:
    label = str(insight.get("experimentability") or "").strip().lower()
    return EXPERIMENTABILITY_MAP.get(label, EXPERIMENTABILITY_MAP[""])


# ---------- scoring ----------


def score_insight(agenda: ResearchAgenda, insight: Mapping[str, Any]) -> dict[str, Any]:
    """Score one insight against an agenda. Returns full breakdown.

    Output keys:
      - score (float)
      - blocked (bool)
      - block_reasons (list[str])
      - components (dict of named contributions)
      - matched_focus (list[str])
      - matched_prefer_keywords (list[str])
      - paradigm_score (float, 0..1)
      - feasibility_score (float, 0..1)
    """
    text = _text_blob(insight)
    block_reasons: list[str] = []

    # Hard rejects
    reject = agenda.reject or {}
    reject_statuses = {s.lower() for s in ensure_string_list(reject.get("statuses"))}
    if reject_statuses and str(insight.get("status") or "").lower() in reject_statuses:
        block_reasons.append(f"status={insight.get('status')} in reject.statuses")
    reject_novelty = {s.lower() for s in ensure_string_list(reject.get("novelty_status"))}
    if reject_novelty and str(insight.get("novelty_status") or "").lower() in reject_novelty:
        block_reasons.append(f"novelty_status={insight.get('novelty_status')} in reject.novelty_status")
    reject_keywords = [k.lower() for k in ensure_string_list(reject.get("keywords"))]
    for kw in reject_keywords:
        if kw and kw in text:
            block_reasons.append(f"reject keyword hit: {kw}")
            break

    # Positive components
    components: dict[str, float] = {}
    matched_focus: list[str] = []
    for kw in agenda.focus:
        if kw.lower() in text:
            matched_focus.append(kw)
    focus_score = min(len(matched_focus) * W_FOCUS_KEYWORD, W_FOCUS_MAX)
    components["focus"] = focus_score

    prefer = agenda.prefer or {}
    matched_prefer_keywords: list[str] = []
    for kw in ensure_string_list(prefer.get("keywords")):
        if kw.lower() in text:
            matched_prefer_keywords.append(kw)
    prefer_kw_score = min(len(matched_prefer_keywords) * W_PREFER_KEYWORD, W_PREFER_KEYWORD_MAX)
    components["prefer_keywords"] = prefer_kw_score

    tier_score = 0.0
    prefer_tiers = {str(t).lower() for t in ensure_string_list(prefer.get("tiers"))}
    if prefer_tiers and _tier_label(insight.get("tier")) in prefer_tiers:
        tier_score = W_PREFER_TIER
    components["prefer_tier"] = tier_score

    resource_score = 0.0
    prefer_resources = {str(r).lower() for r in ensure_string_list(prefer.get("resource_class"))}
    insight_resource = str(insight.get("resource_class") or "").lower()
    if prefer_resources and insight_resource in prefer_resources:
        resource_score = W_PREFER_RESOURCE
    components["prefer_resource_class"] = resource_score

    paradigm = _paradigm_score(insight)
    feasibility = _feasibility_score(insight)
    components["paradigm"] = paradigm * W_PARADIGM
    components["feasibility"] = feasibility * W_FEASIBILITY

    # Score gates (after components computed, but report independently)
    paradigm_min = prefer.get("paradigm_score_min")
    if paradigm_min is not None:
        try:
            if paradigm < float(paradigm_min):
                block_reasons.append(
                    f"paradigm_score {paradigm:.2f} < min {float(paradigm_min):.2f}"
                )
        except (TypeError, ValueError):
            pass
    feasibility_min = prefer.get("feasibility_score_min")
    if feasibility_min is not None:
        try:
            if feasibility < float(feasibility_min):
                block_reasons.append(
                    f"feasibility_score {feasibility:.2f} < min {float(feasibility_min):.2f}"
                )
        except (TypeError, ValueError):
            pass

    total = sum(components.values())
    return {
        "score": round(total, 4),
        "blocked": bool(block_reasons),
        "block_reasons": block_reasons,
        "components": {k: round(v, 4) for k, v in components.items()},
        "matched_focus": matched_focus,
        "matched_prefer_keywords": matched_prefer_keywords,
        "paradigm_score": round(paradigm, 4),
        "feasibility_score": round(feasibility, 4),
    }


def evaluate_candidates(
    agenda: ResearchAgenda,
    insights: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score all candidates, returning {selected, rejected, all_scored}.

    `selected` is the highest-scoring non-blocked insight (None if all blocked).
    `rejected` is the list of {insight_id, score, reason} for blocked / lower ranked.
    """
    scored: list[dict[str, Any]] = []
    for insight in insights:
        breakdown = score_insight(agenda, insight)
        scored.append({"insight": insight, "breakdown": breakdown})

    eligible = [s for s in scored if not s["breakdown"]["blocked"]]
    eligible.sort(key=lambda s: s["breakdown"]["score"], reverse=True)
    blocked = [s for s in scored if s["breakdown"]["blocked"]]

    selected = eligible[0] if eligible else None
    rejected: list[dict[str, Any]] = []
    for s in eligible[1:]:
        rejected.append(
            {
                "insight_id": s["insight"].get("id"),
                "title": s["insight"].get("title"),
                "score": s["breakdown"]["score"],
                "reason": "lower_score_than_selected",
            }
        )
    for s in blocked:
        rejected.append(
            {
                "insight_id": s["insight"].get("id"),
                "title": s["insight"].get("title"),
                "score": s["breakdown"]["score"],
                "reason": "; ".join(s["breakdown"]["block_reasons"]) or "blocked",
            }
        )
    return {
        "selected": selected,
        "rejected": rejected,
        "all_scored": scored,
    }


# ---------- persistence + DB-facing API ----------


def agenda_scope_keywords(agenda: ResearchAgenda) -> list[str]:
    """Keywords that define the agenda's topical scope: focus + prefer.keywords."""
    seen: set[str] = set()
    out: list[str] = []
    for kw in list(agenda.focus or []) + ensure_string_list((agenda.prefer or {}).get("keywords")):
        k = str(kw).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_POOL_SELECT = """
        SELECT id, tier, status, title, problem_statement, formal_structure,
               existing_weakness, transformation, proposed_method,
               adversarial_score, novelty_status, resource_class, experimentability,
               submission_status, outcome
        FROM deep_insights
"""


def _fetch_insight_pool(
    limit: int = 200,
    agenda: ResearchAgenda | None = None,
) -> list[dict[str, Any]]:
    """Fetch candidate insights, optionally scoped to one agenda.

    Without an agenda this is the original whole-table query (backward
    compatible). With an agenda the pool is restricted to:
      - insights tagged with this agenda_id (produced for this agenda), plus
      - untagged insights (agenda_id IS NULL) whose text matches the agenda's
        scope keywords (focus + prefer.keywords).
    Insights tagged with a different agenda_id are always excluded.
    """
    status_filter = "(status IS NULL OR status NOT IN ('rejected', 'archived'))"
    if agenda is None:
        rows = db.fetchall(
            f"{_POOL_SELECT} WHERE {status_filter} ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        return rows

    keywords = agenda_scope_keywords(agenda)
    clauses: list[str] = []
    params: list[Any] = []
    if agenda.agenda_id:
        clauses.append("agenda_id = ?")
        params.append(int(agenda.agenda_id))
    if keywords:
        likes = []
        for kw in keywords:
            likes.append(
                "LOWER(COALESCE(title, '') || ' ' || COALESCE(problem_statement, '') "
                "|| ' ' || COALESCE(formal_structure, '')) LIKE ?"
            )
            params.append(f"%{kw}%")
        clauses.append(f"(agenda_id IS NULL AND ({' OR '.join(likes)}))")
    else:
        # No scope keywords: keep untagged insights visible (legacy data).
        clauses.append("agenda_id IS NULL")
    rows = db.fetchall(
        f"{_POOL_SELECT} WHERE {status_filter} AND ({' OR '.join(clauses)}) "
        "ORDER BY id DESC LIMIT ?",
        (*params, int(limit)),
    )
    return rows


def _format_rationale(agenda: ResearchAgenda, selected: dict[str, Any]) -> str:
    insight = selected["insight"]
    bd = selected["breakdown"]
    pieces: list[str] = []
    pieces.append(
        f"Selected insight #{insight.get('id')} ({insight.get('title') or 'untitled'}) "
        f"for agenda '{agenda.name}' (score={bd['score']:.2f})."
    )
    if bd["matched_focus"]:
        pieces.append(f"Matched focus: {', '.join(bd['matched_focus'])}.")
    if bd["matched_prefer_keywords"]:
        pieces.append(f"Matched prefer keywords: {', '.join(bd['matched_prefer_keywords'])}.")
    pieces.append(
        f"Paradigm score: {bd['paradigm_score']:.2f}; feasibility: {bd['feasibility_score']:.2f}."
    )
    return " ".join(pieces)


def select_and_persist(
    agenda: ResearchAgenda,
    *,
    limit: int = 200,
    pool: list[Mapping[str, Any]] | None = None,
    scope_to_agenda: bool = False,
) -> AgendaSelection:
    """Run selection over the deep_insights table, persist result, return contract.

    With scope_to_agenda=True the candidate pool is pre-filtered to the
    agenda's own insights + keyword-matching untagged ones (see
    _fetch_insight_pool); default False keeps the historical whole-pool
    behavior where scoring alone decides.
    """
    agenda.validate()
    if not agenda.agenda_id:
        raise ContractValidationError(
            "agenda must be persisted (have agenda_id) before running selection"
        )
    if pool is not None:
        insight_pool = list(pool)
    else:
        insight_pool = _fetch_insight_pool(
            limit=limit, agenda=agenda if scope_to_agenda else None
        )
    if not insight_pool:
        # persist an empty selection so the UI can see "no candidates"
        sel = AgendaSelection(
            agenda_id=agenda.agenda_id,
            selected_insight_id=None,
            score=None,
            rationale=f"No deep_insights available for agenda '{agenda.name}'.",
            rejected_candidates=[],
            scoring_breakdown={"reason": "empty_pool"},
            status="blocked",
        )
        sel_id = _insert_selection(sel)
        sel.selection_id = sel_id
        return sel

    result = evaluate_candidates(agenda, insight_pool)
    selected = result["selected"]
    rejected = result["rejected"]

    if selected is None:
        sel = AgendaSelection(
            agenda_id=agenda.agenda_id,
            selected_insight_id=None,
            score=None,
            rationale=(
                f"All {len(insight_pool)} candidates blocked by agenda '{agenda.name}'."
            ),
            rejected_candidates=rejected,
            scoring_breakdown={"reason": "all_blocked", "pool_size": len(insight_pool)},
            status="blocked",
        )
    else:
        sel = AgendaSelection(
            agenda_id=agenda.agenda_id,
            selected_insight_id=int(selected["insight"]["id"]),
            score=float(selected["breakdown"]["score"]),
            rationale=_format_rationale(agenda, selected),
            rejected_candidates=rejected,
            scoring_breakdown=selected["breakdown"],
            status="pending",
        )
    sel_id = _insert_selection(sel)
    sel.selection_id = sel_id
    return sel


def _insert_selection(sel: AgendaSelection) -> int:
    sel.validate()
    new_id = db.insert_returning_id(
        """
        INSERT INTO agenda_selections
            (agenda_id, selected_insight_id, score, rationale,
             rejected_candidates_json, scoring_breakdown_json, status,
             auto_research_job_id, experiment_run_id, manuscript_run_id,
             submission_bundle_id, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            sel.agenda_id,
            sel.selected_insight_id,
            sel.score,
            sel.rationale,
            json.dumps(sel.rejected_candidates, ensure_ascii=False),
            json.dumps(sel.scoring_breakdown, ensure_ascii=False),
            sel.status,
            sel.auto_research_job_id,
            sel.experiment_run_id,
            sel.manuscript_run_id,
            sel.submission_bundle_id,
            sel.error_message,
        ),
    )
    db.commit()
    return new_id


# Subset of fields the selection record exposes for incremental progress updates
_PROGRESS_FIELDS = {
    "status",
    "auto_research_job_id",
    "experiment_run_id",
    "manuscript_run_id",
    "submission_bundle_id",
    "error_message",
}


def update_selection_progress(selection_id: int, **fields: Any) -> None:
    unknown = set(fields) - _PROGRESS_FIELDS
    if unknown:
        raise ValueError(f"unknown progress fields: {sorted(unknown)}")
    if "status" in fields and fields["status"] is not None:
        if fields["status"] not in VALID_SELECTION_STATUS:
            raise ValueError(
                f"invalid selection status {fields['status']!r}; "
                f"must be one of {sorted(VALID_SELECTION_STATUS)}"
            )
    if not fields:
        return
    set_clauses = ", ".join(f"{k}=?" for k in fields)
    params = tuple(fields[k] for k in fields) + (selection_id,)
    db.execute(
        f"UPDATE agenda_selections SET {set_clauses}, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        params,
    )
    db.commit()


def _row_to_selection_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    def _decode(field_name: str, default: Any) -> Any:
        value = row.get(field_name)
        if isinstance(value, str):
            try:
                return json.loads(value) if value else default
            except json.JSONDecodeError:
                return default
        return value if value is not None else default

    return {
        "id": row.get("id"),
        "agenda_id": row.get("agenda_id"),
        "selected_insight_id": row.get("selected_insight_id"),
        "score": row.get("score"),
        "rationale": row.get("rationale"),
        "rejected_candidates": _decode("rejected_candidates_json", []),
        "scoring_breakdown": _decode("scoring_breakdown_json", {}),
        "status": row.get("status"),
        "auto_research_job_id": row.get("auto_research_job_id"),
        "experiment_run_id": row.get("experiment_run_id"),
        "manuscript_run_id": row.get("manuscript_run_id"),
        "submission_bundle_id": row.get("submission_bundle_id"),
        "error_message": row.get("error_message"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def get_selection(selection_id: int) -> dict[str, Any] | None:
    row = db.fetchone("SELECT * FROM agenda_selections WHERE id=?", (selection_id,))
    if not row:
        return None
    return _row_to_selection_dict(row)


def get_latest_selection(agenda_id: int | None = None) -> dict[str, Any] | None:
    if agenda_id is None:
        row = db.fetchone(
            "SELECT * FROM agenda_selections ORDER BY created_at DESC, id DESC LIMIT 1",
            (),
        )
    else:
        row = db.fetchone(
            "SELECT * FROM agenda_selections WHERE agenda_id=? ORDER BY created_at DESC, id DESC LIMIT 1",
            (agenda_id,),
        )
    if not row:
        return None
    return _row_to_selection_dict(row)
