#!/usr/bin/env python3
"""V1 chain advancer: move agenda candidates from ideation to a granted job.

One pass per invocation (a systemd timer supplies the cadence). Each pass:

  1. reconcile expired grants and requeue withdrawn candidates (housekeeping
     that has no other periodic caller);
  2. per target agenda:
     a. mechanically draft topic-gate pre-registrations for insights that lack
        one (the gate scores unregistered candidates -inf, so nothing queues
        without this step); refused records are logged, never forced;
     b. run agenda_selector.select_next to queue gate-passing candidates;
     c. ensure one gate-allowed frontier packet exists (bootstrap evaluation,
        real LLM spend, ~14k tokens - cached in the state file);
     d. for queued jobs at awaiting_portfolio_decision: build an honestly
        attributed heuristic IdeaDecisionPacket, decide_portfolio,
        save_decision (re-runs the topic gate), issue a pilot ResourceGrant.

  Execution is deliberately NOT triggered here: the auto_research loop's
  candidate pool already claims queued jobs that carry an active grant, and
  it is the only executor with a GPU path. This script stops at the grant.

Everything goes through reviewed module entry points; the only raw SQL is
read-only SELECTs. Every action and refusal is appended as one JSON line to
--log. An optional token-spend guard refuses to issue new authorities or
grants past --spend-limit (delta over the baseline captured for that process).

V1 glue; retired in V2 by the meta-harness workers (docs/upgrade-plan-v1-v2.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agenda_selector import select_next  # noqa: E402
from contracts.meta_harness import (  # noqa: E402
    Estimate,
    FrontierEvaluationAuthority,
    IdeaDecisionPacket,
)
from db import database as db  # noqa: E402
from db.insight_outcomes import (  # noqa: E402
    OUTCOME_PROPOSAL_UNREALIZED,
    set_outcome,
)
from meta_harness.frontier_authority import FrontierAuthorityRepository  # noqa: E402
from meta_harness.frontier_bootstrap import run_bootstrap_evaluation  # noqa: E402
from meta_harness.job_states import RECYCLABLE  # noqa: E402
from meta_harness.outcome_finalizer import finalize_terminal_outcomes  # noqa: E402
from meta_harness.portfolio import decide_portfolio, issue_resource_grant  # noqa: E402
from meta_harness.preflight_repository import CandidatePreflightRepository  # noqa: E402
from meta_harness.repository import MetaHarnessRepository  # noqa: E402
from meta_harness.topic_gate_record import record_prediction  # noqa: E402

ACTOR = "ops:auto-advance-v1"
ARTIFACT_REQUIREMENTS = [
    "final_results",
    "raw_predictions",
    "environment_manifest",
    "dataset_manifest",
    "model_manifest",
]
# Bumped after discovering that the old recycler incremented its counter before
# checking packet/preflight/grant eligibility. Three preflight deferrals could
# therefore manufacture ``recycle_exhausted`` without a single requeue. Counts
# now mean successful requeues only; one epoch reset discards the polluted
# operational counters without editing any business row.
RECYCLE_EPOCH = "successful-requeue-counting-2026-08-12"

# Frontier rationing. The ration per problem is unchanged; what changes is that
# the pool no longer stops at the top 3, so spending a problem's ration retires
# that problem instead of the whole agenda. Attempts per pass are capped below
# the previous effective 3 so widening the pool cannot raise the burn rate:
# each attempt is one real evaluator call (~14k tokens under a 20k authority).
FRONTIER_PROBLEM_POOL = 24
FRONTIER_MAX_TRIES = 4
FRONTIER_ATTEMPTS_PER_PASS = 2


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Journal:
    def __init__(self, path: Path):
        self.path = path

    def log(self, step: str, **fields) -> None:
        record = {"at": _now().isoformat(), "step": step, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        print(f"[advance] {line}", flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in db.fetchall(sql, params)]


def active_agenda_ids() -> list[int]:
    return [
        int(row["id"])
        for row in _rows(
            """
            SELECT id FROM research_agendas
            WHERE is_active=1 AND status='active'
              AND token_budget > token_spent + token_reserved
            ORDER BY updated_at ASC, id ASC
            """
        )
    ]


def _next_discovery_agenda(agenda_ids: list[int], state: dict) -> int | None:
    """Choose one agenda per pass, rotating across the active ordered set.

    Tier-2 discovery reads a global signal bundle.  Running it once for every
    agenda multiplied the same large-relation reads by the active agenda
    count.  Advancement still visits every agenda; only candidate discovery
    is rate-limited and rotated here.  An explicit single ``--agenda`` keeps
    the targeted-pass behavior unchanged.
    """
    ordered = [int(agenda_id) for agenda_id in agenda_ids]
    if not ordered:
        return None
    previous = state.get("discovery_agenda_last")
    try:
        index = (ordered.index(int(previous)) + 1) % len(ordered)
    except (TypeError, ValueError):
        index = 0
    selected = ordered[index]
    state["discovery_agenda_last"] = selected
    return selected


def _load_state(path: Path) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {"spend_baseline": {}, "frontier_packets": {}}
    # Retry counts are operational state, not business history.  A deployed
    # repair must get one fresh autonomous attempt without an operator editing
    # the state file or any business table.
    if state.get("recycle_epoch") != RECYCLE_EPOCH:
        state["recycles"] = {}
        state["recycle_epoch"] = RECYCLE_EPOCH
    return state


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _agenda_committed_spend(agenda_id: int) -> int | None:
    """Return spent plus metered or live-reserved tokens for one agenda."""
    row = db.fetchone(
        "SELECT COALESCE(token_spent,0) AS s FROM research_agendas WHERE id=?",
        (agenda_id,),
    )
    if not row:
        return None
    spent = int(dict(row)["s"])
    metered = db.fetchone(
        "SELECT COALESCE(SUM(u.tokens_used),0) AS s"
        "  FROM resource_grant_usage_reservations u"
        "  JOIN resource_grants g ON g.id = u.resource_grant_id"
        " WHERE u.agenda_id=? AND u.status='settled'"
        "   AND g.status <> 'consumed'"
        # A live grant's full cap is counted below. Its settled calls are
        # already inside that cap and must not be added a second time.
        "   AND NOT (g.status='active' AND g.expires_at > CURRENT_TIMESTAMP)",
        (agenda_id,),
    )
    spent += int(dict(metered or {}).get("s") or 0)
    live_reservations = db.fetchone(
        "SELECT COALESCE(SUM(token_cap),0) AS s"
        "  FROM resource_grants"
        " WHERE agenda_id=? AND status='active'"
        "   AND expires_at > CURRENT_TIMESTAMP",
        (agenda_id,),
    )
    return spent + int(dict(live_reservations or {}).get("s") or 0)


def _capture_spend_baseline(agenda_ids: list[int]) -> dict[str, int]:
    """Capture a per-process baseline for an explicitly bounded invocation."""
    baseline: dict[str, int] = {}
    for agenda_id in agenda_ids:
        spent = _agenda_committed_spend(agenda_id)
        if spent is not None:
            baseline[str(agenda_id)] = spent
    return baseline


def _spent_delta(
    state: dict,
    agenda_ids: list[int],
    process_baseline: dict[str, int] | None = None,
) -> int:
    """Tokens consumed or irrevocably reserved since this driver started.

    research_agendas.token_spent alone undercounts badly: it only moves when a
    whole grant settles, so every call made inside a still-active grant is
    invisible. Expired grants also retain settled sub-reservations even though
    their agenda reservation is released and no OutcomeRecord is created.
    Count those metered calls and the full cap of live grants so the outer
    pilot guard cannot issue a second grant after reserving the remaining
    budget.
    """
    total = 0
    durable_baseline = state.setdefault("spend_baseline", {})
    for aid in agenda_ids:
        spent = _agenda_committed_spend(aid)
        if spent is None:
            continue
        if process_baseline is None:
            baseline = durable_baseline.setdefault(str(aid), spent)
        else:
            baseline = process_baseline.get(str(aid), spent)
        total += max(0, spent - int(baseline))
    return total


def _guard_spent_delta(state: dict, args) -> int:
    return _spent_delta(
        state,
        args.agenda,
        getattr(args, "process_spend_baseline", None),
    )


def _grant_key(agenda_id: int, idea_id: int, suffix: str) -> str:
    """A key that has not been used before.

    issue_grant is idempotent on (agenda_id, idempotency_key) and *returns the
    existing grant* on replay - so a fixed key would silently hand back
    yesterday's expired grant and strand the job forever.
    """
    row = db.fetchone(
        "SELECT COUNT(*) AS c FROM resource_grants"
        " WHERE agenda_id=? AND idea_id=? AND idempotency_key LIKE ?",
        (agenda_id, idea_id, "auto-advance-v1:%"),
    )
    return f"auto-advance-v1:idea{idea_id}:{suffix}:a{int(dict(row)['c']) + 1}"


def _clip(text: str | None, limit: int = 600) -> str:
    return (text or "").strip()[:limit]


def draft_preregistration(
    insight: dict,
    *,
    token_cap: int,
    gpu_hours: float,
) -> dict:
    """Mechanical pre-registration from the insight's own fields.

    No LLM call happens here (pre-idea LLM spend would be ungranted). The
    provenance says exactly what this is; the topic gate stays the judge -
    a record it rejects is not persisted.
    """
    method = {}
    try:
        method = json.loads(insight.get("proposed_method") or "{}")
    except Exception:
        pass
    predicted = _clip(insight.get("falsification")) or _clip(insight.get("predictions")) or (
        f"The intervention proposed in '{_clip(insight.get('title'), 160)}' outperforms the"
        " unmodified baseline on the primary metric of its experimental plan."
    )
    metric = _clip(str(method.get("metric") or "")) or (
        "primary metric named in the insight's experimental plan: "
        + _clip(insight.get("experimental_plan"), 200)
    )
    baseline = _clip(str(method.get("baseline") or "")) or (
        "the frozen base model / unmodified pipeline without the proposed change"
    )
    comparison = _clip(insight.get("experimental_plan"), 400) or (
        "single controlled comparison: proposed change vs baseline under identical budget"
    )
    return {
        "prediction": {
            "predicted_outcome": predicted,
            "confidence": 0.55,
            "action_if_confirmed": (
                "Escalate to a validation-stage grant and pre-register a full benchmark run."
            ),
            "action_if_refuted": (
                "Record a negative OutcomeRecord, archive the candidate, and feed the failure"
                " back into agenda selection."
            ),
        },
        "minimum_falsification_experiment": {
            "metric": metric,
            "baseline": baseline,
            "decisive_comparison": comparison,
            "estimated_cost": {
                "tokens": max(1, int(token_cap)),
                "gpu_hours": max(0.0, float(gpu_hours)),
                "wall_hours": 4,
            },
        },
        "provenance": {
            "drafted_by": f"{ACTOR}(mechanical-from-insight-fields)",
            "authorized_by": "operator:v1-standing-order-2026-08-06",
            "review_status": "auto_drafted_unreviewed",
        },
    }


def _estimate(value: float, lower: float, upper: float, sources: list[str]) -> Estimate:
    return Estimate(
        value=value,
        lower=lower,
        upper=upper,
        evaluator=f"{ACTOR}-heuristic",
        provider="heuristic",
        model="auto-advance-v1",
        evidence_sources=sources,
    )


def build_packet(agenda_id: int, idea_id: int, frontier_packet_id: int) -> IdeaDecisionPacket:
    src = [f"deep_insight:{idea_id}", f"frontier_packet:{frontier_packet_id}"]
    return IdeaDecisionPacket(
        agenda_id=agenda_id,
        idea_id=idea_id,
        frontier_packet_id=frontier_packet_id,
        expected_impact=_estimate(0.50, 0.30, 0.70, src),
        success_probability=_estimate(0.45, 0.25, 0.65, src),
        novelty=_estimate(0.60, 0.40, 0.80, src),
        obsolescence_probability=_estimate(0.30, 0.10, 0.50, src),
        falsification_value=_estimate(0.75, 0.60, 0.90, src),
        reuse_value=_estimate(0.50, 0.30, 0.70, src),
        expected_token_cost=_estimate(5000.0, 3000.0, 8000.0, src),
        expected_gpu_cost=_estimate(0.0, 0.0, 2.0, src),
        time_to_feedback=_estimate(4.0, 2.0, 12.0, src),
        execution_risk=_estimate(0.60, 0.40, 0.80, src),
        information_value=_estimate(0.80, 0.60, 0.95, src),
        candidate_family=f"agenda{agenda_id}-v1",
        correlation_keys=[f"agenda:{agenda_id}", f"insight:{idea_id}"],
        reason_codes=["auto_advance_v1_candidate"],
        # Input packets default to decision="park", which the contract only
        # accepts with a revisit condition; decide_portfolio overwrites both.
        revisit_condition={"on": ["awaiting_initial_portfolio_decision"]},
    )


def ensure_frontier_packet(
    agenda_id: int, state: dict, journal: Journal, args
) -> int | None:
    cached = state["frontier_packets"].get(str(agenda_id))
    if cached:
        row = db.fetchone(
            "SELECT id FROM frontier_packets WHERE id=? AND agenda_id=? AND gate_allowed=1",
            (int(cached), agenda_id),
        )
        if row:
            return int(cached)
        journal.log("frontier_cache_stale", agenda_id=agenda_id, cached=cached)
    # Ration attempts per problem, but keep rotating: an agenda can hold
    # dozens of open problems, and capping the pool at the top 3 meant that
    # once those three were spent the agenda could never obtain a packet again
    # no matter how many untried problems it still had.
    problems = _rows(
        "SELECT id, problem_statement FROM research_problems"
        " WHERE agenda_id=? AND COALESCE(status,'open')='open'"
        " ORDER BY problem_quality_score DESC NULLS LAST, id ASC LIMIT ?",
        (agenda_id, FRONTIER_PROBLEM_POOL),
    )
    if not problems:
        journal.log("frontier_no_problems", agenda_id=agenda_id)
        return None
    repo = FrontierAuthorityRepository()
    attempts = state.setdefault("frontier_attempts", {})
    # Authority idempotency keys are burned on failure, so key uniqueness needs
    # its own monotonic counter - it must keep advancing even on attempts that
    # deliberately do not consume the per-problem ration.
    issues = state.setdefault("frontier_issues", {})
    considered = 0
    for problem in problems:
        akey = f"{agenda_id}:{problem['id']}"
        tries = int(attempts.get(akey, 0))
        if tries >= FRONTIER_MAX_TRIES:
            journal.log("frontier_attempts_exhausted", agenda_id=agenda_id,
                        problem_id=problem["id"], tries=tries)
            continue
        considered += 1
        if considered > FRONTIER_ATTEMPTS_PER_PASS:
            journal.log("frontier_pass_budget_reached", agenda_id=agenda_id,
                        attempted=FRONTIER_ATTEMPTS_PER_PASS)
            return None
        attempts[akey] = tries + 1
        serial = int(issues.get(akey, 0)) + 1
        issues[akey] = serial
        # A failed authority is settled+revoked and its idempotency key is
        # burned forever, so every attempt needs a fresh key.
        key = f"auto-advance-v1:agenda{agenda_id}:problem{problem['id']}:t{serial}"
        issued = _now()
        try:
            authority_id = repo.issue(
                FrontierEvaluationAuthority(
                    agenda_id=agenda_id,
                    research_problem_id=int(problem["id"]),
                    token_cap=args.authority_token_cap,
                    issued_at=issued.isoformat(),
                    expires_at=(issued + timedelta(minutes=60)).isoformat(),
                    idempotency_key=key,
                    provider=args.evaluator_provider,
                    model=args.evaluator_model,
                    model_family=args.evaluator_family,
                    prompt_version="evaluator_v1",
                    evaluator=f"{args.evaluator_provider}/{args.evaluator_model}",
                    issued_by=ACTOR,
                    issue_reason="V1 chain advance: frontier packet required before portfolio",
                )
            )
            result = run_bootstrap_evaluation(
                authority_id=authority_id,
                agenda_id=agenda_id,
                research_problem_id=int(problem["id"]),
                proposer_provider=args.proposer_provider,
                proposer_model_family=args.proposer_family,
            )
        except Exception as exc:
            db.rollback()
            # An unreachable evaluator says nothing about this problem. Refund
            # the ration and stop the pass: hammering the remaining problems
            # would burn their rations against the same dead route.
            if getattr(exc, "transient", False):
                attempts[akey] = tries
                journal.log(
                    "frontier_bootstrap_transient",
                    agenda_id=agenda_id,
                    problem_id=problem["id"],
                    tries_preserved=tries,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                return None
            journal.log(
                "frontier_bootstrap_refused",
                agenda_id=agenda_id,
                problem_id=problem["id"],
                reason=f"{type(exc).__name__}: {exc}",
            )
            continue
        packet_id = result.get("frontier_packet_id")
        journal.log(
            "frontier_bootstrap_done",
            agenda_id=agenda_id,
            problem_id=problem["id"],
            packet_id=packet_id,
            gate_allowed=result.get("gate_allowed"),
            reason_codes=result.get("gate_reason_codes"),
            tokens_used=result.get("tokens_used"),
        )
        if packet_id and not result.get("gate_allowed"):
            # An "already_completed" replay omits gate_allowed; trust the row.
            row = db.fetchone(
                "SELECT gate_allowed FROM frontier_packets WHERE id=? AND agenda_id=?",
                (int(packet_id), agenda_id),
            )
            if row and int(dict(row)["gate_allowed"] or 0) == 1:
                result["gate_allowed"] = True
        if packet_id and result.get("gate_allowed"):
            state["frontier_packets"][str(agenda_id)] = int(packet_id)
            return int(packet_id)
    return None


_STARVED = "provider_usage_exceeded_reserved_cap"

# Declared in meta_harness.job_states alongside the claim predicate, so the
# recycler and the candidate pool cannot drift apart. Kept as a module name
# because the recycler and its tests read it.
DEAD_END = RECYCLABLE
MAX_RECYCLES = 3


def _grant_gpu_usage(resource_grant_id: int):
    from meta_harness.attempt_gpu_usage import GrantGPUUsageControl

    control = GrantGPUUsageControl()
    usage = control.grant_usage(int(resource_grant_id))
    if usage.exhausted and usage.grant_status == "active":
        usage = control.reconcile_exhausted_grant(int(resource_grant_id))
    return usage


def recycle_stranded(agenda_id: int, state: dict, journal: Journal, args) -> None:
    """Give a stranded candidate a live grant and a state someone reads.

    Two independent failure modes put a job somewhere nothing claims it: a
    pilot cap the forge cannot live on (the benchmark designer meters first and
    the next agent dies on provider_usage_exceeded_reserved_cap), and a repair
    that parks the job in a stage outside the candidate pool's accepted set. An
    exhausted grant cannot be revoked - its usage is settled - so it is left to
    expire and a fresh one is issued. Bounded per idea, and it never touches a
    job that is genuinely in flight.
    """
    counts = state.setdefault("recycles", {})
    rows = _rows(
        "SELECT arj.id, arj.deep_insight_id, arj.status, arj.stage, arj.resource_grant_id,"
        "       arj.last_error, rg.token_cap, rg.max_gpu_hours, rg.status AS grant_status,"
        "       (rg.expires_at > CURRENT_TIMESTAMP) AS grant_live"
        "  FROM auto_research_jobs arj"
        "  LEFT JOIN resource_grants rg ON rg.id = arj.resource_grant_id"
        " WHERE arj.agenda_id=? ORDER BY arj.id",
        (agenda_id,),
    )
    for job in rows:
        idea_id = int(job["deep_insight_id"])
        stranded = (str(job["status"]), str(job["stage"])) in DEAD_END
        starved = _STARVED in str(job["last_error"] or "")
        if not (stranded or starved):
            continue
        used = int(counts.get(str(idea_id), 0))
        if used >= MAX_RECYCLES:
            journal.log("recycle_exhausted", agenda_id=agenda_id, idea_id=idea_id,
                        status=job["status"], stage=job["stage"], recycles=used)
            continue
        required_token_cap = int(args.grant_token_cap)
        grant_ok = (
            str(job["grant_status"] or "") == "active"
            and bool(job["grant_live"])
            and int(job["token_cap"] or 0) >= required_token_cap
        )
        if grant_ok and float(job.get("max_gpu_hours") or 0.0) > 0:
            gpu_usage = _grant_gpu_usage(int(job["resource_grant_id"]))
            if gpu_usage.remaining_gpu_seconds <= 0:
                journal.log(
                    (
                        "gpu_budget_exhausted"
                        if gpu_usage.exhausted
                        else "gpu_budget_fully_reserved"
                    ),
                    agenda_id=agenda_id,
                    idea_id=idea_id,
                    resource_grant_id=int(job["resource_grant_id"]),
                    settled_gpu_hours=gpu_usage.settled_gpu_seconds / 3600.0,
                    active_reserved_gpu_hours=(
                        gpu_usage.active_reserved_gpu_seconds / 3600.0
                    ),
                    max_gpu_hours=gpu_usage.cap_gpu_seconds / 3600.0,
                    reason=(
                        "canonical grant GPU remainder is exhausted"
                        if gpu_usage.exhausted
                        else "canonical grant GPU remainder is held by an active attempt"
                    ),
                )
                continue
        if (
            not grant_ok
            and args.spend_limit > 0
            and _guard_spent_delta(state, args) + required_token_cap > args.spend_limit
        ):
            journal.log(
                "spend_limit_reached",
                agenda_id=agenda_id,
                idea_id=idea_id,
                limit=args.spend_limit,
                reason="fresh grant would exceed the process pilot guard",
            )
            continue
        if grant_ok:
            grant_id = int(job["resource_grant_id"])
            journal.log("recycle_keeps_grant", agenda_id=agenda_id, idea_id=idea_id,
                        resource_grant_id=grant_id)
            _requeue_for_consumer(
                agenda_id, idea_id, grant_id, journal, args, used + 1,
                token_cap=int(job["token_cap"] or 0),
            )
            counts[str(idea_id)] = used + 1
            continue
        packet_row = db.fetchone(
            "SELECT id FROM idea_decision_packets WHERE agenda_id=? AND idea_id=?"
            "   AND decision IN ('promote','revisit') ORDER BY id DESC LIMIT 1",
            (agenda_id, idea_id),
        )
        if not packet_row:
            journal.log("regrant_no_packet", agenda_id=agenda_id, idea_id=idea_id)
            continue
        packet = _rebuild_decision(agenda_id, idea_id, int(dict(packet_row)["id"]))
        backends = json.loads(dict(db.fetchone(
            "SELECT backend_allowlist_json FROM research_agendas WHERE id=?", (agenda_id,)
        ))["backend_allowlist_json"])
        preflight = CandidatePreflightRepository().run_candidate(
            agenda_id=agenda_id,
            idea_id=idea_id,
        )
        if not preflight.passed or preflight.selected_backend not in backends:
            journal.log(
                "regrant_preflight_deferred",
                agenda_id=agenda_id,
                idea_id=idea_id,
                status=preflight.status,
                reason_codes=preflight.reason_codes,
            )
            continue
        allowed_backends = [preflight.selected_backend]
        if "llm" in backends:
            allowed_backends.append("llm")
        try:
            grant = issue_resource_grant(
                packet,
                stage="pilot",
                token_cap=required_token_cap,
                gpu_class=(
                    args.gpu_class
                    if preflight.selected_backend != "cpu"
                    else "none"
                ),
                max_gpu_hours=(
                    args.grant_gpu_hours
                    if preflight.selected_backend != "cpu"
                    else 0.0
                ),
                backend_allowlist=allowed_backends,
                artifact_requirements=ARTIFACT_REQUIREMENTS,
                expires_at=(_now() + timedelta(hours=12)).isoformat(),
                idempotency_key=_grant_key(agenda_id, idea_id, "regrant"),
                preflight_result_id=preflight.preflight_result_id,
            )
            grant_id = MetaHarnessRepository().issue_grant(grant)
        except Exception as exc:
            db.rollback()
            journal.log("regrant_refused", agenda_id=agenda_id, idea_id=idea_id,
                        reason=f"{type(exc).__name__}: {exc}")
            continue
        journal.log("regranted", agenda_id=agenda_id, idea_id=idea_id,
                    old_grant=job["resource_grant_id"], new_grant=grant_id,
                    token_cap=required_token_cap,
                    was=f"{job['status']}/{job['stage']}")
        _requeue_for_consumer(
            agenda_id, idea_id, grant_id, journal, args, used + 1,
            token_cap=required_token_cap,
        )
        counts[str(idea_id)] = used + 1


def _requeue_for_consumer(agenda_id: int, idea_id: int, grant_id: int,
                          journal: Journal, args, recycle: int, *,
                          token_cap: int | None = None) -> None:
    """Move a job into a state some consumer actually claims.

    The candidate pool takes status='queued' unconditionally; the harness
    consumer takes harness_required/harness_required. Clearing last_error and
    rewriting last_note also resets the repair-attempt counters, which are
    parsed out of that prose rather than stored as columns.
    """
    from orchestrator.auto_research import _upsert_job

    harness_open = db.fetchone(
        "SELECT 1 AS x FROM benchmark_harness_jobs"
        " WHERE agenda_id=? AND deep_insight_id=? AND status='harness_required'",
        (agenda_id, idea_id),
    )
    status = "harness_required" if harness_open else "queued"
    stage = "benchmark_harness_required" if harness_open else "retry_failed_run"
    effective_token_cap = args.grant_token_cap if token_cap is None else int(token_cap)
    _upsert_job(
        idea_id,
        status=status,
        stage=stage,
        resource_grant_id=grant_id,
        assigned_worker=None,
        last_error=None,
        last_note=(f"auto-advance-v1 recycle {recycle}/{MAX_RECYCLES}: requeued on grant"
                   f" {grant_id} (cap {effective_token_cap})."),
    )
    journal.log("requeued_for_consumer", agenda_id=agenda_id, idea_id=idea_id,
                resource_grant_id=grant_id, status=status, stage=stage, recycle=recycle)


def _rebuild_decision(agenda_id: int, idea_id: int, packet_id: int) -> IdeaDecisionPacket:
    """A promoted packet carrying the persisted decision_packet_id."""
    row = dict(db.fetchone(
        "SELECT frontier_packet_id FROM idea_decision_packets WHERE id=?", (packet_id,)
    ))
    packet = build_packet(agenda_id, idea_id, int(row["frontier_packet_id"]))
    packet.decision = "promote"
    packet.reason_codes = ["portfolio_score_selected", "regrant_after_stranded_or_starved"]
    packet.revisit_condition = {}
    packet.revisit_after = None
    packet.decision_packet_id = packet_id
    return packet


def advance_agenda(agenda_id: int, state: dict, journal: Journal, args) -> None:
    # a. pre-registrations for unregistered, still-live insights
    unregistered = _rows(
        "SELECT * FROM deep_insights WHERE agenda_id=?"
        " AND (topic_gate_json IS NULL OR topic_gate_json='')"
        " AND COALESCE(status,'candidate') NOT IN ('exists','archived')"
        " ORDER BY id ASC LIMIT 10",
        (agenda_id,),
    )
    for insight in unregistered:
        try:
            proposal_pending = str(insight.get("status") or "") == "proposal_pending"
            outcome = record_prediction(
                agenda_id=agenda_id,
                idea_id=int(insight["id"]),
                record=draft_preregistration(
                    insight,
                    token_cap=(
                        args.proposal_token_cap
                        if proposal_pending
                        else args.grant_token_cap
                    ),
                    gpu_hours=0.0 if proposal_pending else args.grant_gpu_hours,
                ),
                actor=ACTOR,
            )
            # record_prediction's dict already carries agenda_id/idea_id.
            journal.log("preregistration", **outcome)
        except Exception as exc:
            db.rollback()
            journal.log(
                "preregistration_refused",
                agenda_id=agenda_id,
                idea_id=insight["id"],
                reason=f"{type(exc).__name__}: {exc}",
            )

    # b. queue gate-passing candidates (select_next persists the selection itself)
    for _ in range(2):
        try:
            selection = select_next(agenda_id)
        except Exception as exc:
            db.rollback()
            journal.log("select_next_failed", agenda_id=agenda_id,
                        reason=f"{type(exc).__name__}: {exc}")
            break
        if selection is None:
            journal.log("select_next_empty", agenda_id=agenda_id)
            break
        journal.log("selected", agenda_id=agenda_id,
                    idea_id=selection.selected_insight_id, score=selection.score)

    # c/d. decide + grant for waiting jobs
    waiting = _rows(
        "SELECT arj.id, arj.deep_insight_id, di.status AS insight_status"
        " FROM auto_research_jobs arj"
        " JOIN deep_insights di ON di.id=arj.deep_insight_id"
        " WHERE arj.agenda_id=? AND arj.status='queued'"
        " AND arj.stage='awaiting_portfolio_decision'"
        " ORDER BY arj.updated_at ASC, arj.id ASC LIMIT ?",
        (agenda_id, args.max_new_grants),
    )
    if not waiting:
        journal.log("no_waiting_jobs", agenda_id=agenda_id)
        return
    packet_id = ensure_frontier_packet(agenda_id, state, journal, args)
    if not packet_id:
        journal.log("no_frontier_packet", agenda_id=agenda_id)
        return
    repo = MetaHarnessRepository()
    try:
        portfolio = decide_portfolio(
            [
                build_packet(
                    agenda_id,
                    int(job["deep_insight_id"]),
                    packet_id,
                )
                for job in waiting
            ]
        )
    except Exception as exc:
        db.rollback()
        journal.log(
            "portfolio_refused",
            agenda_id=agenda_id,
            candidate_ids=[int(job["deep_insight_id"]) for job in waiting],
            reason=f"{type(exc).__name__}: {exc}",
        )
        return
    decision_by_idea = {int(item.idea_id): item for item in portfolio}
    for job in waiting:
        proposal_pending = str(job.get("insight_status") or "") == "proposal_pending"
        requested_token_cap = (
            int(args.proposal_token_cap)
            if proposal_pending
            else int(args.grant_token_cap)
        )
        if (
            args.spend_limit > 0
            and _guard_spent_delta(state, args) + requested_token_cap > args.spend_limit
        ):
            journal.log("spend_limit_reached", agenda_id=agenda_id, limit=args.spend_limit)
            return
        idea_id = int(job["deep_insight_id"])
        decision = decision_by_idea.get(idea_id)
        if decision is None:
            journal.log(
                "decision_missing",
                agenda_id=agenda_id,
                idea_id=idea_id,
            )
            continue
        try:
            repo.save_decision(decision)
        except Exception as exc:
            db.rollback()
            journal.log("decision_refused", agenda_id=agenda_id, idea_id=idea_id,
                        reason=f"{type(exc).__name__}: {exc}")
            continue
        journal.log("decided", agenda_id=agenda_id, idea_id=idea_id,
                    decision=decision.decision, reason_codes=decision.reason_codes,
                    decision_packet_id=decision.decision_packet_id)
        if decision.decision not in {"promote", "revisit"}:
            continue
        agenda_backends = json.loads(dict(db.fetchone(
            "SELECT backend_allowlist_json FROM research_agendas WHERE id=?", (agenda_id,)
        ))["backend_allowlist_json"])
        if proposal_pending:
            if "llm" not in agenda_backends:
                journal.log(
                    "proposal_grant_deferred",
                    agenda_id=agenda_id,
                    idea_id=idea_id,
                    reason="agenda_llm_backend_unavailable",
                )
                continue
            try:
                grant = issue_resource_grant(
                    decision,
                    stage="proposal",
                    token_cap=requested_token_cap,
                    gpu_class="none",
                    max_gpu_hours=0.0,
                    backend_allowlist=["llm"],
                    artifact_requirements=["candidate_design"],
                    expires_at=(_now() + timedelta(hours=4)).isoformat(),
                    idempotency_key=_grant_key(agenda_id, idea_id, "proposal"),
                )
                grant_id = repo.issue_grant(grant)
            except Exception as exc:
                db.rollback()
                journal.log(
                    "proposal_grant_refused",
                    agenda_id=agenda_id,
                    idea_id=idea_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                # Refusing to re-fund is only half the fix. Left queued, the
                # candidate is reselected every pass and refused every pass,
                # and it holds one of the agenda's portfolio slots forever.
                # Retire it through the reviewed outcome writer so the agenda
                # moves on to a problem that can still produce something.
                if "without delivering a candidate" in str(exc):
                    try:
                        set_outcome(
                            "deep_insights",
                            idea_id,
                            OUTCOME_PROPOSAL_UNREALIZED,
                            reason=str(exc)[:500],
                            triggered_by=ACTOR,
                        )
                        # The outcome alone did not retire anything. Selection
                        # keys on status, so idea 125 was promoted, refused and
                        # "retired" once per pass on 2026-08-10 at 21:10, 21:22
                        # and 21:51; and because the row stayed
                        # status='proposal_pending' it kept owning
                        # idx_deep_insights_pending_proposal, so research
                        # problem 49 could never be worked again. 'archived' is
                        # the status the live candidate queries already exclude
                        # (paper_idea_agent.py:974, agenda_repository.py:151,
                        # this script's pre-registration sweep), so it both
                        # stops the churn and releases the problem.
                        db.execute(
                            "UPDATE deep_insights SET status='archived',"
                            " updated_at=CURRENT_TIMESTAMP"
                            " WHERE id=? AND agenda_id=?"
                            " AND status='proposal_pending'",
                            (idea_id, agenda_id),
                        )
                        db.commit()
                        journal.log(
                            "proposal_candidate_retired",
                            agenda_id=agenda_id,
                            idea_id=idea_id,
                            outcome=OUTCOME_PROPOSAL_UNREALIZED,
                            status="archived",
                        )
                    except Exception as retire_exc:
                        db.rollback()
                        journal.log(
                            "proposal_retire_failed",
                            agenda_id=agenda_id,
                            idea_id=idea_id,
                            reason=f"{type(retire_exc).__name__}: {retire_exc}",
                        )
                continue
            journal.log(
                "proposal_granted",
                agenda_id=agenda_id,
                idea_id=idea_id,
                resource_grant_id=grant_id,
                token_cap=requested_token_cap,
            )
            continue
        preflight = CandidatePreflightRepository().run_candidate(
            agenda_id=agenda_id,
            idea_id=idea_id,
        )
        journal.log(
            "preflight",
            agenda_id=agenda_id,
            idea_id=idea_id,
            status=preflight.status,
            reason_codes=preflight.reason_codes,
            adapter_id=preflight.adapter_id,
            selected_backend=preflight.selected_backend,
            preflight_result_id=preflight.preflight_result_id,
        )
        if not preflight.passed:
            from orchestrator.auto_research import _upsert_job

            _upsert_job(
                idea_id,
                status="deferred",
                stage="capability_preflight_deferred",
                assigned_worker=None,
                last_error="preflight:" + ",".join(preflight.reason_codes),
                last_note=(
                    "Candidate deferred before grant; capability or metadata "
                    "requirements are not currently satisfied."
                ),
            )
            continue
        if preflight.selected_backend not in agenda_backends:
            journal.log(
                "preflight_backend_not_in_agenda",
                agenda_id=agenda_id,
                idea_id=idea_id,
                selected_backend=preflight.selected_backend,
            )
            continue
        compute_backend = str(preflight.selected_backend)
        selected_backends = [compute_backend]
        if "llm" in agenda_backends:
            selected_backends.append("llm")
        gpu_selected = compute_backend != "cpu"
        attempts = [{
            "backends": selected_backends,
            "gpu_hours": args.grant_gpu_hours if gpu_selected else 0.0,
            "gpu_class": args.gpu_class if gpu_selected else "none",
            "ttl_hours": 12 if gpu_selected else 24,
            "suffix": "preflight",
        }]
        for attempt in attempts:
            try:
                grant = issue_resource_grant(
                    decision,
                    stage="pilot",
                    token_cap=args.grant_token_cap,
                    gpu_class=attempt["gpu_class"],
                    max_gpu_hours=attempt["gpu_hours"],
                    backend_allowlist=attempt["backends"],
                    artifact_requirements=ARTIFACT_REQUIREMENTS,
                    expires_at=(_now() + timedelta(hours=attempt["ttl_hours"])).isoformat(),
                    idempotency_key=_grant_key(agenda_id, idea_id, attempt["suffix"]),
                    preflight_result_id=preflight.preflight_result_id,
                )
                grant_id = repo.issue_grant(grant)
            except Exception as exc:
                db.rollback()
                journal.log("grant_refused", agenda_id=agenda_id, idea_id=idea_id,
                            variant=attempt["suffix"],
                            reason=f"{type(exc).__name__}: {exc}")
                continue
            journal.log("granted", agenda_id=agenda_id, idea_id=idea_id,
                        resource_grant_id=grant_id, variant=attempt["suffix"],
                        backends=attempt["backends"], gpu_hours=attempt["gpu_hours"],
                        token_cap=args.grant_token_cap)
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agenda", type=int, action="append",
                        help="repeatable; default: every active budgeted agenda")
    parser.add_argument("--state", default="/home/ec2-user/deepgraph-reports/auto_advance_state.json")
    parser.add_argument("--log", default="/home/ec2-user/deepgraph-reports/auto_advance_log.jsonl")
    parser.add_argument("--spend-limit", type=int, default=0,
                        help="optional extra process cap; 0 relies on each agenda ledger")
    parser.add_argument("--max-new-grants", type=int, default=2, help="per agenda per pass")
    # One forge pass measured 2026-08-07: repair 12626 + code scout 1976 +
    # benchmark design ~8400 = ~23000. A 15000 cap left 398 for the design
    # call, so the pass died one step from a runnable plan.
    parser.add_argument("--grant-token-cap", type=int, default=40000)
    parser.add_argument("--proposal-token-cap", type=int, default=32000)
    parser.add_argument("--grant-gpu-hours", type=float, default=2.0)
    parser.add_argument("--gpu-class", default="NVIDIA A100-PCIE-40GB")
    # Contract max is 20000; cycle-1's real evaluator consumed 13717 and a
    # 15000 cap was exceeded in practice, charging the agenda for nothing.
    parser.add_argument("--authority-token-cap", type=int, default=20000)
    parser.add_argument("--evaluator-provider", default="sora2_claude")
    parser.add_argument("--evaluator-model", default="claude-opus-4-6-thinking")
    parser.add_argument("--evaluator-family", default="claude")
    parser.add_argument("--proposer-provider", default="sora2_gemini")
    parser.add_argument("--proposer-family", default="gemini")
    args = parser.parse_args()
    args.agenda = args.agenda or active_agenda_ids()

    journal = Journal(Path(args.log))
    state_path = Path(args.state)
    state = _load_state(state_path)
    args.process_spend_baseline = (
        _capture_spend_baseline(args.agenda) if args.spend_limit > 0 else None
    )
    journal.log("pass_start", agendas=args.agenda, backend=db.describe_backend())

    try:
        repo = MetaHarnessRepository()
        finalized = finalize_terminal_outcomes().to_dict()
        if finalized.get("attempted") or finalized.get("finalized"):
            journal.log("outcome_finalization", **finalized)
        reconciled = repo.reconcile_expired_grants()
        if reconciled:
            journal.log("reconciled_expired_grants", count=reconciled)
        discovery_agenda_id = _next_discovery_agenda(args.agenda, state)
        if discovery_agenda_id is not None:
            try:
                from orchestrator.discovery_scheduler import run_tier2_discovery

                generated = run_tier2_discovery(
                    max_problems=2,
                    max_papers=2,
                    agenda_id=discovery_agenda_id,
                )
                journal.log(
                    "candidate_generation_pass",
                    agenda_id=discovery_agenda_id,
                    generated=generated,
                )
            except Exception as exc:
                db.rollback()
                journal.log(
                    "candidate_generation_failed",
                    agenda_id=discovery_agenda_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
        for agenda_id in args.agenda:
            for job in _rows(
                "SELECT deep_insight_id, stage FROM auto_research_jobs WHERE agenda_id=?"
                " AND stage IN ('resource_grant_expired','resource_grant_revoked')",
                (agenda_id,),
            ):
                try:
                    moved = repo.requeue_withdrawn_candidate(
                        agenda_id=agenda_id,
                        idea_id=int(job["deep_insight_id"]),
                        reason="auto_advance_v1_recycle",
                    )
                    journal.log("requeued", agenda_id=agenda_id,
                                idea_id=job["deep_insight_id"], moved=moved)
                except Exception as exc:
                    db.rollback()
                    journal.log("requeue_refused", agenda_id=agenda_id,
                                idea_id=job["deep_insight_id"],
                                reason=f"{type(exc).__name__}: {exc}")
            spent = _guard_spent_delta(state, args)
            if args.spend_limit > 0 and spent >= args.spend_limit:
                journal.log("spend_limit_reached", spent=spent, limit=args.spend_limit)
                break
            recycle_stranded(agenda_id, state, journal, args)
            advance_agenda(agenda_id, state, journal, args)
    finally:
        _save_state(state_path, state)
        try:
            db.rollback()  # never leave an idle-in-transaction session behind
        except Exception:
            pass
    journal.log("pass_end", spent_delta=_guard_spent_delta(state, args))
    _save_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
