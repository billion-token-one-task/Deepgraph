"""Revision planner: convert a reviewer's findings into next-step experiments.

Issue #9 step 5b: given an AgendaReview, produce an AgendaRevisionPlan with
structured `next_experiments` entries.

The planner is deterministic and rule-based (no LLM call required for the
demo). Each `next_experiments` entry has the shape:

    {
      "name": "<short identifier>",
      "rationale": "<why this experiment is needed>",
      "kind": "ablation|baseline|robustness|evidence_gap|prediction_test",
      "priority": "high|medium|low",
      "source": "review.required_revisions[<idx>]" | "review.evidence_blockers[<idx>]"
    }

Public API:
- build_revision_plan(review_id) -> AgendaRevisionPlan
- get_revision_plan(plan_id) -> dict | None
- get_latest_plan_for_selection(selection_id) -> dict | None
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agents.reviewer_adapter import get_review
from contracts.agenda import AgendaRevisionPlan
from contracts.base import ContractValidationError, ensure_list, ensure_string_list
from db import database as db


# ---------- rule-based mapping ----------


_KIND_KEYWORDS = (
    ("ablation", ("ablation", "ablate", "component")),
    ("baseline", ("baseline", "compare against", "comparison")),
    ("robustness", ("robustness", "noise", "perturb", "stress", "out-of-distribution", "ood")),
    ("prediction_test", ("prediction", "predicted", "falsification")),
    ("evidence_gap", ("evidence", "table", "figure", "manuscript", "bundle")),
)


def _classify(text: str) -> str:
    lower = text.lower()
    for kind, keywords in _KIND_KEYWORDS:
        if any(k in lower for k in keywords):
            return kind
    return "evidence_gap"


def _priority(recommendation: str, kind: str) -> str:
    rec = recommendation.lower()
    if rec == "reject":
        return "high"
    if rec == "major_revision":
        return "high" if kind in ("baseline", "ablation", "prediction_test") else "medium"
    if rec == "minor_revision":
        return "medium"
    return "low"


def _slugify(text: str, idx: int) -> str:
    tokens = []
    for ch in text.lower():
        if ch.isalnum():
            tokens.append(ch)
        elif tokens and tokens[-1] != "_":
            tokens.append("_")
    slug = "".join(tokens).strip("_")[:48] or f"experiment_{idx}"
    return f"{slug}_{idx}"


def _experiments_from_review(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    recommendation = str(review.get("recommendation") or "")
    items: list[dict[str, Any]] = []

    required = ensure_string_list(review.get("required_revisions"))
    for idx, text in enumerate(required):
        kind = _classify(text)
        items.append(
            {
                "name": _slugify(text, idx),
                "rationale": text,
                "kind": kind,
                "priority": _priority(recommendation, kind),
                "source": f"review.required_revisions[{idx}]",
            }
        )

    blockers = ensure_list(review.get("evidence_blockers"))
    for idx, blocker in enumerate(blockers):
        if not isinstance(blocker, dict):
            continue
        requirement = str(blocker.get("requirement") or "unspecified")
        rationale = blocker.get("statement") or blocker.get("reason") or requirement
        kind = "evidence_gap" if "prediction" not in requirement else "prediction_test"
        items.append(
            {
                "name": _slugify(f"blocker_{requirement}", idx),
                "rationale": f"Resolve evidence blocker: {rationale}",
                "kind": kind,
                "priority": "high" if recommendation in ("reject", "major_revision") else "medium",
                "source": f"review.evidence_blockers[{idx}]",
            }
        )

    # If recommendation is reject but no concrete experiments, add a default refactor item.
    if recommendation.lower() == "reject" and not items:
        items.append(
            {
                "name": "reframe_hypothesis_0",
                "rationale": "Hypothesis was refuted; reformulate the claim before further runs.",
                "kind": "prediction_test",
                "priority": "high",
                "source": "review.recommendation",
            }
        )

    return items


# ---------- public API ----------


def build_revision_plan(review_id: int) -> AgendaRevisionPlan:
    review = get_review(review_id)
    if not review:
        raise ContractValidationError(f"review {review_id} not found")

    experiments = _experiments_from_review(review)
    recommendation = str(review.get("recommendation") or "")
    rationale_parts: list[str] = [
        f"Reviewer '{review.get('reviewer')}' recommended '{recommendation}'."
    ]
    if review.get("confidence") is not None:
        rationale_parts.append(f"Confidence={float(review['confidence']):.2f}.")
    weaknesses = ensure_string_list(review.get("weaknesses"))
    if weaknesses:
        rationale_parts.append(
            "Weaknesses to address: " + "; ".join(weaknesses[:3])
        )

    status = "proposed"
    if recommendation.lower() == "accept" and not experiments:
        status = "noop"
        experiments = experiments or []
        rationale_parts.append("No revisions required; recommending direct submission.")
        # AgendaRevisionPlan requires either next_experiments or rationale — rationale covers it.

    plan = AgendaRevisionPlan(
        selection_id=int(review["selection_id"]),
        review_id=int(review["id"]),
        rationale=" ".join(rationale_parts),
        next_experiments=experiments,
        status=status,
    )
    plan.validate()

    plan_id = db.insert_returning_id(
        """
        INSERT INTO agenda_revision_plans
            (selection_id, review_id, rationale, next_experiments_json, status)
        VALUES (?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            plan.selection_id,
            plan.review_id,
            plan.rationale,
            json.dumps(plan.next_experiments, ensure_ascii=False),
            plan.status,
        ),
    )
    plan.plan_id = plan_id
    db.commit()
    return plan


def _row_to_plan_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("next_experiments_json")
    if isinstance(raw, str):
        try:
            experiments = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            experiments = []
    else:
        experiments = raw or []
    return {
        "id": row.get("id"),
        "selection_id": row.get("selection_id"),
        "review_id": row.get("review_id"),
        "rationale": row.get("rationale"),
        "next_experiments": experiments,
        "status": row.get("status"),
        "created_at": row.get("created_at"),
    }


def get_revision_plan(plan_id: int) -> dict[str, Any] | None:
    row = db.fetchone("SELECT * FROM agenda_revision_plans WHERE id=?", (plan_id,))
    if not row:
        return None
    return _row_to_plan_dict(row)


def get_latest_plan_for_selection(selection_id: int) -> dict[str, Any] | None:
    row = db.fetchone(
        "SELECT * FROM agenda_revision_plans WHERE selection_id=? "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (selection_id,),
    )
    if not row:
        return None
    return _row_to_plan_dict(row)
