"""Realize exactly one proposal-granted research job.

The periodic discovery path scans and refreshes an agenda-wide problem pool.
That is appropriate for autonomy, but not for a controlled recovery canary.
This module accepts the complete persisted identity (job, agenda, idea and
grant), validates it without changing anything, and then realizes only that
proposal shell through the existing proposal agent and repository settlement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from db import database as db
from meta_harness.repository import MetaHarnessRepository
from orchestrator.pipeline import log_event


class BoundedProposalError(RuntimeError):
    """The named job cannot be realized without widening its authority."""


@dataclass(frozen=True)
class BoundedProposalRequest:
    job_id: int
    agenda_id: int
    idea_id: int
    resource_grant_id: int

    def validate(self) -> None:
        for field in ("job_id", "agenda_id", "idea_id", "resource_grant_id"):
            if int(getattr(self, field)) <= 0:
                raise BoundedProposalError(f"{field} must be positive")


@dataclass(frozen=True)
class BoundedProposalResult:
    status: str
    job_id: int
    agenda_id: int
    idea_id: int
    resource_grant_id: int
    tokens_used: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "status": self.status,
            "job_id": self.job_id,
            "agenda_id": self.agenda_id,
            "idea_id": self.idea_id,
            "resource_grant_id": self.resource_grant_id,
            "tokens_used": self.tokens_used,
        }


def _load_json_list(value: Any) -> set[str]:
    try:
        loaded = json.loads(value or "[]")
    except (TypeError, ValueError):
        return set()
    return {str(item).strip().lower() for item in loaded} if isinstance(loaded, list) else set()


def authorize_bounded_proposal(
    request: BoundedProposalRequest,
) -> tuple[dict[str, Any], bool]:
    """Return the exact scope and whether it is already fully settled."""

    request.validate()
    row = db.fetchone(
        """
        SELECT arj.id AS job_id, arj.agenda_id, arj.deep_insight_id,
               arj.status AS job_status, arj.stage AS job_stage,
               arj.resource_grant_id AS job_resource_grant_id,
               di.status AS insight_status,
               di.research_problem_id, rg.stage AS grant_stage,
               rg.id AS grant_id, rg.status AS grant_status, rg.expires_at,
               rg.token_cap, rg.max_gpu_hours, rg.backend_allowlist_json,
               COALESCE((
                   SELECT SUM(u.tokens_used)
                   FROM resource_grant_usage_reservations u
                   WHERE u.resource_grant_id=rg.id AND u.status='settled'
               ), 0) AS grant_tokens_used,
               CASE WHEN rg.expires_at > CURRENT_TIMESTAMP THEN 1 ELSE 0 END
                   AS grant_live
        FROM auto_research_jobs arj
        JOIN deep_insights di
          ON di.id=arj.deep_insight_id AND di.agenda_id=arj.agenda_id
        JOIN resource_grants rg
          ON rg.id=? AND rg.agenda_id=arj.agenda_id
         AND rg.idea_id=arj.deep_insight_id
        WHERE arj.id=? AND arj.agenda_id=? AND arj.deep_insight_id=?
        """,
        (
            request.resource_grant_id,
            request.job_id,
            request.agenda_id,
            request.idea_id,
        ),
    )
    if not row:
        raise BoundedProposalError("exact proposal job/grant scope was not found")
    scope = dict(row)
    if str(scope.get("grant_stage") or "") != "proposal":
        raise BoundedProposalError("bounded proposal requires a proposal grant")
    if float(scope.get("max_gpu_hours") or 0.0) != 0.0:
        raise BoundedProposalError("bounded proposal refuses GPU authority")
    backends = _load_json_list(scope.get("backend_allowlist_json"))
    if backends != {"llm"}:
        raise BoundedProposalError("bounded proposal requires an llm-only grant")

    already_completed = (
        str(scope.get("grant_status") or "") == "consumed"
        and str(scope.get("job_status") or "") == "queued"
        and str(scope.get("job_stage") or "") == "awaiting_portfolio_decision"
        and str(scope.get("insight_status") or "") == "candidate"
        and scope.get("job_resource_grant_id") is None
    )
    if already_completed:
        return scope, True
    if (
        str(scope.get("grant_status") or "") != "active"
        or int(scope.get("grant_live") or 0) != 1
        or int(scope.get("job_resource_grant_id") or 0)
        != request.resource_grant_id
        or str(scope.get("job_status") or "") != "deferred"
        or str(scope.get("job_stage") or "") != "proposal_generation_granted"
        or str(scope.get("insight_status") or "") not in {"proposal_pending", "candidate"}
        or int(scope.get("research_problem_id") or 0) <= 0
    ):
        raise BoundedProposalError("proposal job is not in an executable exact-target state")
    return scope, False


def _default_discover(request: BoundedProposalRequest) -> list[dict]:
    from agents.paper_idea_agent import discover_paper_ideas

    return discover_paper_ideas(
        max_problems=1,
        max_papers=1,
        agenda_id=request.agenda_id,
        proposal_job_id=request.job_id,
        proposal_candidate_id=request.idea_id,
        proposal_grant_id=request.resource_grant_id,
    )


def _default_store(candidate: dict) -> int:
    from agents.paradigm_agent import store_deep_insight

    return int(store_deep_insight(candidate))


def execute_bounded_proposal(
    request: BoundedProposalRequest,
    *,
    actor: str,
    repository: MetaHarnessRepository | None = None,
    discover: Callable[[BoundedProposalRequest], list[dict]] | None = None,
    store: Callable[[dict], int] | None = None,
) -> BoundedProposalResult:
    """Realize and settle one proposal; replay closes post-store crash windows."""

    if not str(actor or "").strip():
        raise BoundedProposalError("actor is required")
    scope, already_completed = authorize_bounded_proposal(request)
    if already_completed:
        return BoundedProposalResult(
            status="already_completed",
            job_id=request.job_id,
            agenda_id=request.agenda_id,
            idea_id=request.idea_id,
            resource_grant_id=request.resource_grant_id,
            tokens_used=int(scope.get("grant_tokens_used") or 0),
        )

    repo = repository or MetaHarnessRepository()
    if str(scope.get("insight_status") or "") == "proposal_pending":
        candidates = (discover or _default_discover)(request)
        if len(candidates) != 1 or not isinstance(candidates[0], dict):
            raise BoundedProposalError("exact proposal did not produce one candidate")
        candidate = dict(candidates[0])
        if (
            int(candidate.get("proposal_candidate_id") or 0) != request.idea_id
            or int(candidate.get("resource_grant_id") or 0)
            != request.resource_grant_id
            or int(candidate.get("agenda_id") or 0) != request.agenda_id
        ):
            raise BoundedProposalError("realized proposal escaped the requested scope")
        stored_id = int((store or _default_store)(candidate))
        if stored_id != request.idea_id:
            raise BoundedProposalError("proposal storage returned a different idea")

    # If the process previously died after store_deep_insight committed, the
    # exact shell is already `candidate` while the grant/job remain active and
    # deferred.  Skipping generation above and calling the idempotent
    # repository settlement is the formal recovery for that crash window.
    tokens_used = int(
        repo.complete_proposal_generation(
            grant_id=request.resource_grant_id,
            agenda_id=request.agenda_id,
            idea_id=request.idea_id,
            target_job_id=request.job_id,
        )
    )
    log_event(
        "bounded_proposal",
        {
            "step": "completed",
            "actor": str(actor),
            **request.__dict__,
            "tokens_used": tokens_used,
        },
    )
    return BoundedProposalResult(
        status="completed",
        job_id=request.job_id,
        agenda_id=request.agenda_id,
        idea_id=request.idea_id,
        resource_grant_id=request.resource_grant_id,
        tokens_used=tokens_used,
    )
