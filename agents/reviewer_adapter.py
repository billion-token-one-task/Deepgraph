"""Reviewer adapter for the agenda loop (issue #9 step 5).

Provides an `internal_evidence_gate` reviewer that aggregates existing
experiment evidence (hypothesis_verdict, effect_size, experimental_claims,
evidence_plan) into an AgendaReview record. Issue #9 explicitly says external
reviewers (paperreview.ai) are NOT required — a thin internal reviewer based
on existing evidence is enough.

The adapter is intentionally extensible: register more reviewers via
`register_reviewer(name, fn)` and route by name in `run_review`.

Public API:
- run_review(selection_id, *, reviewer='internal_evidence_gate') -> AgendaReview
- register_reviewer(name, fn) -> None
- list_reviewers() -> list[str]
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from contracts.agenda import AgendaReview
from contracts.base import ContractValidationError, ensure_dict, ensure_list
from db import database as db


ReviewerFn = Callable[[dict[str, Any]], AgendaReview]


_REVIEWERS: dict[str, ReviewerFn] = {}


def register_reviewer(name: str, fn: ReviewerFn) -> None:
    name = name.strip()
    if not name:
        raise ValueError("reviewer name cannot be empty")
    _REVIEWERS[name] = fn


def list_reviewers() -> list[str]:
    return sorted(_REVIEWERS.keys())


# ---------- data assembly ----------


def _selection_row(selection_id: int) -> dict[str, Any]:
    row = db.fetchone("SELECT * FROM agenda_selections WHERE id=?", (selection_id,))
    if not row:
        raise ValueError(f"selection {selection_id} not found")
    return row


def _experiment_run(run_id: int | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    return db.fetchone(
        """
        SELECT id, deep_insight_id, status, phase, hypothesis_verdict,
               baseline_metric_name, baseline_metric_value, best_metric_value,
               effect_size, effect_pct, error_message, submission_bundle_id
        FROM experiment_runs WHERE id=?
        """,
        (run_id,),
    )


def _experimental_claims(run_id: int | None) -> list[dict[str, Any]]:
    if not run_id:
        return []
    return db.fetchall(
        """
        SELECT id, claim_text, claim_type, verdict, effect_size, confidence,
               p_value, supporting_data
        FROM experimental_claims WHERE run_id=?
        ORDER BY id
        """,
        (run_id,),
    )


def _insight_evidence_plan(insight_id: int | None) -> dict[str, Any]:
    if not insight_id:
        return {}
    row = db.fetchone(
        "SELECT evidence_plan, predictions, falsification FROM deep_insights WHERE id=?",
        (insight_id,),
    )
    if not row:
        return {}
    plan = row.get("evidence_plan")
    if isinstance(plan, str):
        try:
            plan = json.loads(plan) if plan else {}
        except json.JSONDecodeError:
            plan = {}
    out = ensure_dict(plan)
    out.setdefault("predictions", row.get("predictions"))
    out.setdefault("falsification", row.get("falsification"))
    return out


def _bundle_row(bundle_id: int | None) -> dict[str, Any] | None:
    if not bundle_id:
        return None
    return db.fetchone(
        """
        SELECT id, manuscript_run_id, bundle_format, status, bundle_path,
               manifest_path
        FROM submission_bundles WHERE id=?
        """,
        (bundle_id,),
    )


# ---------- internal reviewer ----------


def _summarize_claims(claims: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"confirmed": 0, "refuted": 0, "inconclusive": 0, "other": 0}
    strengths: list[str] = []
    weaknesses: list[str] = []
    for c in claims:
        verdict = str(c.get("verdict") or "").lower()
        counts[verdict if verdict in counts else "other"] += 1
        text = (c.get("claim_text") or "").strip()
        if not text:
            continue
        if verdict == "confirmed":
            strengths.append(text[:200])
        elif verdict == "refuted":
            weaknesses.append(f"refuted: {text[:200]}")
        elif verdict == "inconclusive":
            weaknesses.append(f"inconclusive: {text[:200]}")
    return {"counts": counts, "strengths": strengths, "weaknesses": weaknesses}


def _required_evidence_blockers(
    plan: Mapping[str, Any],
    claims: list[dict[str, Any]],
    bundle: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Identify required evidence items that are missing or not satisfied."""
    blockers: list[dict[str, Any]] = []
    if not plan:
        return blockers

    # evidence_plan may contain entries like:
    #   {"main_table": {"enabled": true, "priority": "required"}, ...}
    for key, value in plan.items():
        if not isinstance(value, dict):
            continue
        priority = str(value.get("priority") or "").lower()
        enabled = value.get("enabled", True)
        if priority == "required" and enabled:
            # Heuristic: if no claim mentions the key and no bundle, treat as missing.
            mentioned = any(key.replace("_", " ") in (c.get("claim_text") or "").lower() for c in claims)
            if not mentioned and not bundle:
                blockers.append({"requirement": key, "reason": "not_demonstrated"})

    predictions = ensure_list(plan.get("predictions"))
    for pred in predictions:
        if isinstance(pred, dict) and pred.get("required"):
            text = str(pred.get("statement") or pred.get("text") or "").strip()
            if text and not any(text.lower()[:40] in (c.get("claim_text") or "").lower() for c in claims):
                blockers.append({"requirement": "prediction", "statement": text, "reason": "not_tested"})
    return blockers


def _internal_evidence_gate(ctx: dict[str, Any]) -> AgendaReview:
    selection = ctx["selection"]
    exp = ctx.get("experiment_run") or {}
    claims = ctx.get("experimental_claims") or []
    bundle = ctx.get("submission_bundle")
    plan = ctx.get("evidence_plan") or {}

    claim_summary = _summarize_claims(claims)
    blockers = _required_evidence_blockers(plan, claims, bundle)

    verdict = str(exp.get("hypothesis_verdict") or "").lower()
    effect_size = exp.get("effect_size")
    try:
        effect = float(effect_size) if effect_size is not None else None
    except (TypeError, ValueError):
        effect = None

    strengths = list(claim_summary["strengths"])
    weaknesses = list(claim_summary["weaknesses"])
    required_revisions: list[str] = []

    if effect is not None and effect > 0 and verdict == "confirmed":
        strengths.append(
            f"Hypothesis confirmed with positive effect_size={effect:.4f}"
            + (f" ({exp.get('effect_pct'):.2f}%)" if exp.get("effect_pct") else "")
        )

    if not exp:
        weaknesses.append("No experiment_run linked to this selection.")
        required_revisions.append("Launch an experiment_run for this insight.")

    if exp and verdict not in ("confirmed", "refuted", "inconclusive"):
        weaknesses.append(f"experiment_run verdict missing or unknown ({verdict or 'NULL'}).")
        required_revisions.append("Re-run hypothesis testing phase to obtain a verdict.")

    if not bundle and verdict == "confirmed":
        weaknesses.append("Confirmed result but no submission_bundle produced.")
        required_revisions.append("Generate a manuscript bundle via PaperOrchestra.")

    for b in blockers:
        required_revisions.append(
            f"Provide required evidence: {b.get('requirement')} ({b.get('reason')})."
        )

    # Recommendation logic
    if verdict == "refuted":
        recommendation = "reject"
        confidence = 0.85
    elif verdict == "inconclusive" or not exp:
        recommendation = "major_revision"
        confidence = 0.55
    elif blockers or required_revisions:
        recommendation = "minor_revision"
        confidence = 0.7
    elif verdict == "confirmed":
        recommendation = "accept"
        confidence = 0.9
    else:
        recommendation = "major_revision"
        confidence = 0.5

    review = AgendaReview(
        selection_id=int(selection["id"]),
        submission_bundle_id=int(bundle["id"]) if bundle else None,
        manuscript_run_id=selection.get("manuscript_run_id"),
        reviewer="internal_evidence_gate",
        recommendation=recommendation,
        confidence=confidence,
        strengths=strengths,
        weaknesses=weaknesses,
        required_revisions=required_revisions,
        evidence_blockers=blockers,
        raw_review={
            "claim_counts": claim_summary["counts"],
            "verdict": verdict,
            "effect_size": effect,
            "bundle_present": bool(bundle),
        },
    )
    review.validate()
    return review


register_reviewer("internal_evidence_gate", _internal_evidence_gate)


# ---------- public entry ----------


def _build_review_context(selection_id: int) -> dict[str, Any]:
    sel = _selection_row(selection_id)
    exp = _experiment_run(sel.get("experiment_run_id"))
    claims = _experimental_claims(sel.get("experiment_run_id"))
    bundle = _bundle_row(sel.get("submission_bundle_id"))
    plan = _insight_evidence_plan(sel.get("selected_insight_id"))
    return {
        "selection": sel,
        "experiment_run": exp,
        "experimental_claims": claims,
        "submission_bundle": bundle,
        "evidence_plan": plan,
    }


def _persist_review(review: AgendaReview) -> int:
    review.validate()
    return db.insert_returning_id(
        """
        INSERT INTO agenda_reviews
            (selection_id, submission_bundle_id, manuscript_run_id, reviewer,
             recommendation, confidence, strengths_json, weaknesses_json,
             required_revisions_json, evidence_blockers_json, raw_review_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            review.selection_id,
            review.submission_bundle_id,
            review.manuscript_run_id,
            review.reviewer,
            review.recommendation,
            review.confidence,
            json.dumps(review.strengths, ensure_ascii=False),
            json.dumps(review.weaknesses, ensure_ascii=False),
            json.dumps(review.required_revisions, ensure_ascii=False),
            json.dumps(review.evidence_blockers, ensure_ascii=False),
            json.dumps(review.raw_review, ensure_ascii=False),
        ),
    )


def run_review(selection_id: int, *, reviewer: str = "internal_evidence_gate") -> AgendaReview:
    fn = _REVIEWERS.get(reviewer)
    if fn is None:
        raise ContractValidationError(
            f"unknown reviewer '{reviewer}'; registered: {list_reviewers()}"
        )
    ctx = _build_review_context(selection_id)
    review = fn(ctx)
    review_id = _persist_review(review)
    review.review_id = review_id
    db.commit()
    return review


def get_review(review_id: int) -> dict[str, Any] | None:
    row = db.fetchone("SELECT * FROM agenda_reviews WHERE id=?", (review_id,))
    if not row:
        return None
    return _row_to_review_dict(row)


def get_latest_review(selection_id: int) -> dict[str, Any] | None:
    row = db.fetchone(
        "SELECT * FROM agenda_reviews WHERE selection_id=? ORDER BY created_at DESC, id DESC LIMIT 1",
        (selection_id,),
    )
    if not row:
        return None
    return _row_to_review_dict(row)


def _row_to_review_dict(row: Mapping[str, Any]) -> dict[str, Any]:
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
        "selection_id": row.get("selection_id"),
        "submission_bundle_id": row.get("submission_bundle_id"),
        "manuscript_run_id": row.get("manuscript_run_id"),
        "reviewer": row.get("reviewer"),
        "recommendation": row.get("recommendation"),
        "confidence": row.get("confidence"),
        "strengths": _decode("strengths_json", []),
        "weaknesses": _decode("weaknesses_json", []),
        "required_revisions": _decode("required_revisions_json", []),
        "evidence_blockers": _decode("evidence_blockers_json", []),
        "raw_review": _decode("raw_review_json", {}),
        "created_at": row.get("created_at"),
    }
