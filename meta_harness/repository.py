"""PostgreSQL persistence for frontier, decisions, grants, and outcomes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from contracts.meta_harness import (
    FrontierPacket,
    IdeaDecisionPacket,
    OutcomeRecord,
    ResourceGrant,
)
from contracts.scientific_evidence import EvidenceDecisionInput, decide_evidence
from db import database as db
from meta_harness.llm_routing import ProviderRoute, RouteObservation
from meta_harness.evidence_state import EvidenceTransitionContext, advance
from meta_harness.frontier import evaluate_frontier
from meta_harness.reviewer_approval import (
    ReviewerApproval,
    ReviewerApprovalVerifier,
    scientific_manuscript_subject,
)


class MetaHarnessPersistenceError(RuntimeError):
    pass


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _canonical_hash(value: str) -> str:
    text = str(value or "").strip().lower()
    return text.removeprefix("sha256:")


def _expect_one(cursor: Any, *, operation: str) -> None:
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        raise MetaHarnessPersistenceError(f"concurrent persistence race:{operation}")


def _estimate_payload(packet: IdeaDecisionPacket) -> dict[str, Any]:
    return {
        name: getattr(packet, name).to_dict()
        for name in (
            "expected_impact",
            "success_probability",
            "novelty",
            "obsolescence_probability",
            "falsification_value",
            "reuse_value",
            "expected_token_cost",
            "expected_gpu_cost",
            "time_to_feedback",
            "execution_risk",
            "information_value",
        )
    }


class MetaHarnessRepository:
    def load_active_cooldowns(
        self,
        route_ids: list[str],
        *,
        now: datetime,
    ) -> dict[str, datetime]:
        """Load provider cooldowns so a process restart cannot bypass them."""
        normalized = sorted(
            {
                str(route_id or "").strip()
                for route_id in route_ids
                if str(route_id or "").strip()
            }
        )
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        rows = db.fetchall(
            f"""
            SELECT route_id, cooldown_until
            FROM llm_provider_cooldowns
            WHERE route_id IN ({placeholders})
              AND cooldown_until > ?
            """,
            (*normalized, now.astimezone(timezone.utc).isoformat()),
        )
        active: dict[str, datetime] = {}
        for row in rows:
            raw = row.get("cooldown_until")
            if isinstance(raw, datetime):
                parsed = raw
            else:
                parsed = datetime.fromisoformat(
                    str(raw or "").replace("Z", "+00:00")
                )
            active[str(row["route_id"])] = parsed.astimezone(timezone.utc)
        return active

    def save_cooldown(
        self,
        route: ProviderRoute,
        *,
        until: datetime,
        failure_category: str,
    ) -> None:
        """Persist the longest cooldown observed for an explicit route."""
        route.validate()
        category = str(failure_category or "").strip()
        if category not in {"auth", "transient", "provider_error"}:
            raise MetaHarnessPersistenceError("invalid provider cooldown category")
        try:
            db.execute(
                """
                INSERT INTO llm_provider_cooldowns
                    (route_id, provider, model, failure_category, cooldown_until,
                     updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (route_id) DO UPDATE SET
                    provider=EXCLUDED.provider,
                    model=EXCLUDED.model,
                    failure_category=EXCLUDED.failure_category,
                    cooldown_until=GREATEST(
                        llm_provider_cooldowns.cooldown_until,
                        EXCLUDED.cooldown_until
                    ),
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    route.route_id,
                    route.provider,
                    route.model,
                    category,
                    until.astimezone(timezone.utc).isoformat(),
                ),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

    def attach_grant_to_run(
        self,
        *,
        agenda_id: int,
        idea_id: int,
        experiment_run_id: int,
        resource_grant_id: int,
    ) -> None:
        """Attach a later-stage grant without changing run or idea scope."""
        try:
            lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            run = db.fetchone(
                f"""
                SELECT agenda_id, deep_insight_id
                FROM experiment_runs
                WHERE id=?{lock}
                """,
                (experiment_run_id,),
            )
            grant = db.fetchone(
                """
                SELECT agenda_id, idea_id, stage
                FROM resource_grants
                WHERE id=? AND status='active'
                  AND expires_at > CURRENT_TIMESTAMP
                """,
                (resource_grant_id,),
            )
            if (
                not run
                or not grant
                or int(run.get("agenda_id") or 0) != agenda_id
                or int(run.get("deep_insight_id") or 0) != idea_id
                or int(grant.get("agenda_id") or 0) != agenda_id
                or int(grant.get("idea_id") or 0) != idea_id
            ):
                raise MetaHarnessPersistenceError("run/grant scope mismatch")
            if grant.get("stage") not in {
                "validation",
                "full_benchmark",
                "evidence_audit",
            }:
                raise MetaHarnessPersistenceError(
                    "grant stage cannot be attached to an existing run"
                )
            db.execute(
                """
                UPDATE experiment_runs
                SET resource_grant_id=?
                WHERE id=? AND agenda_id=? AND deep_insight_id=?
                """,
                (resource_grant_id, experiment_run_id, agenda_id, idea_id),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET resource_grant_id=?, updated_at=CURRENT_TIMESTAMP
                WHERE agenda_id=? AND deep_insight_id=?
                """,
                (resource_grant_id, agenda_id, idea_id),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

    def advance_experiment_state(
        self,
        *,
        agenda_id: int,
        experiment_run_id: int,
        target: str,
        context: EvidenceTransitionContext,
        actor: str,
    ) -> str:
        """Advance exactly one state and append an immutable audit transition."""
        if not actor.strip():
            raise MetaHarnessPersistenceError("state transition actor is required")
        try:
            lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            row = db.fetchone(
                f"""
                SELECT agenda_id, deep_insight_id, resource_grant_id,
                       scientific_evidence_state
                FROM experiment_runs
                WHERE id=?{lock}
                """,
                (experiment_run_id,),
            )
            if not row or int(row.get("agenda_id") or 0) != int(agenda_id):
                raise MetaHarnessPersistenceError("experiment run scope mismatch")
            grant = db.fetchone(
                f"""
                SELECT agenda_id, idea_id, stage, status, expires_at
                FROM resource_grants
                WHERE id=?
                  AND (
                    status='consumed'
                    OR (status='active' AND expires_at > CURRENT_TIMESTAMP)
                  )
                {lock}
                """,
                (int(context.resource_grant_id or 0),),
            )
            target_grant_stages = {
                "sanity_passed": {"pilot", "validation"},
                "full_benchmark_complete": {"full_benchmark"},
                "evidence_audited": {"evidence_audit"},
                "scientifically_decided": {"evidence_audit"},
                "manuscript_allowed": {"evidence_audit", "manuscript"},
            }
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != int(agenda_id)
                or int(grant.get("idea_id") or 0)
                != int(row.get("deep_insight_id") or 0)
                or int(row.get("resource_grant_id") or 0)
                != int(context.resource_grant_id or 0)
                or str(grant.get("stage") or "")
                not in target_grant_stages.get(target, set())
            ):
                raise MetaHarnessPersistenceError(
                    "persisted ResourceGrant does not authorize state transition"
                )
            current = str(row.get("scientific_evidence_state") or "planned")
            reviewer_approval: ReviewerApproval | None = None
            if target == "manuscript_allowed":
                subject = scientific_manuscript_subject(
                    agenda_id=int(agenda_id),
                    experiment_run_id=int(experiment_run_id),
                    verdict_hash=context.verdict_hash,
                )
                reviewer_approval = ReviewerApprovalVerifier.from_environment().verify(
                    context.reviewer_approval,
                    purpose="scientific_manuscript",
                    subject=subject,
                )
                if reviewer_approval.reviewer_id != actor:
                    raise MetaHarnessPersistenceError(
                        "transition actor does not match signed reviewer"
                    )
                context = replace(context, reviewer_approved=True)
            next_state = advance(current, target, context)
            audit_record_id: int | None = None
            evidence_decision_payload: dict[str, Any] | None = None
            if next_state == "evidence_audited":
                audit_record_id = db.insert_returning_id(
                    """
                    INSERT INTO evidence_audit_records
                        (agenda_id, experiment_run_id, raw_artifacts_hash,
                         claim_ledger_hash, benchmark_contract_hash,
                         evaluator_ref, evaluator_hash, holdout_ref,
                         holdout_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                    """,
                    (
                        agenda_id,
                        experiment_run_id,
                        _canonical_hash(context.raw_artifacts_hash),
                        _canonical_hash(context.claim_ledger_hash),
                        _canonical_hash(context.benchmark_contract_hash),
                        context.evaluator_ref,
                        _canonical_hash(context.evaluator_hash),
                        context.holdout_ref,
                        _canonical_hash(context.holdout_hash),
                    ),
                )
            elif next_state == "scientifically_decided":
                audit = db.fetchone(
                    f"""
                    SELECT * FROM evidence_audit_records
                    WHERE agenda_id=? AND experiment_run_id=?{lock}
                    """,
                    (agenda_id, experiment_run_id),
                )
                if not audit:
                    raise MetaHarnessPersistenceError(
                        "scientific decision requires persisted evidence audit"
                    )
                expected_hashes = {
                    "raw_artifacts_hash": context.raw_artifacts_hash,
                    "claim_ledger_hash": context.claim_ledger_hash,
                    "benchmark_contract_hash": context.benchmark_contract_hash,
                    "evaluator_hash": context.evaluator_hash,
                    "holdout_hash": context.holdout_hash,
                }
                mismatches = [
                    name
                    for name, value in expected_hashes.items()
                    if _canonical_hash(str(audit.get(name) or ""))
                    != _canonical_hash(value)
                ]
                for name, value in (
                    ("evaluator_ref", context.evaluator_ref),
                    ("holdout_ref", context.holdout_ref),
                ):
                    if str(audit.get(name) or "") != str(value or ""):
                        mismatches.append(name)
                if mismatches:
                    raise MetaHarnessPersistenceError(
                        "scientific decision evidence hash mismatch:"
                        + ",".join(sorted(mismatches))
                    )
                decision_input = EvidenceDecisionInput(
                    verdict=context.verdict,
                    p_value=context.p_value,
                    alpha=context.alpha,
                    metric_value=context.metric_value,
                    baseline_value=context.baseline_value,
                    full_benchmark_complete=True,
                    raw_artifacts_complete=True,
                    claim_ledger_complete=True,
                    evaluator_id=str(audit.get("evaluator_ref") or ""),
                )
                decision = decide_evidence(decision_input)
                if context.verdict == "supported" and (
                    not context.evidence_decision_passed
                    or not decision.confirmation_allowed
                ):
                    raise MetaHarnessPersistenceError(
                        "positive scientific decision failed integrity contract:"
                        + ",".join(decision.blockers)
                    )
                audit_record_id = int(audit["id"])
                evidence_decision_payload = {
                    "input": decision_input.to_dict(),
                    "decision": decision.to_dict(),
                    "holdout_ref": audit.get("holdout_ref"),
                    "holdout_hash": audit.get("holdout_hash"),
                }
            elif next_state == "manuscript_allowed":
                decision_row = db.fetchone(
                    f"""
                    SELECT verdict, verdict_hash
                    FROM scientific_decision_records
                    WHERE agenda_id=? AND experiment_run_id=?{lock}
                    """,
                    (agenda_id, experiment_run_id),
                )
                if (
                    not decision_row
                    or decision_row.get("verdict") != "supported"
                    or _canonical_hash(str(decision_row.get("verdict_hash") or ""))
                    != _canonical_hash(context.verdict_hash)
                ):
                    raise MetaHarnessPersistenceError(
                        "manuscript approval does not match a supported decision"
                    )
            reviewer = (
                reviewer_approval.reviewer_id
                if reviewer_approval is not None
                else None
            )
            cursor = db.execute(
                """
                UPDATE experiment_runs
                SET scientific_evidence_state=?,
                    scientific_reviewer_approved_by=COALESCE(?, scientific_reviewer_approved_by),
                    scientific_reviewer_approved_at=CASE
                        WHEN CAST(? AS TEXT) IS NOT NULL THEN CURRENT_TIMESTAMP
                        ELSE scientific_reviewer_approved_at
                    END
                WHERE id=? AND agenda_id=? AND scientific_evidence_state=?
                """,
                (
                    next_state,
                    reviewer,
                    reviewer,
                    experiment_run_id,
                    agenda_id,
                    current,
                ),
            )
            _expect_one(cursor, operation="advance_experiment_state")
            if reviewer_approval is not None:
                approval_record = reviewer_approval.public_record()
                db.execute(
                    """
                    INSERT INTO reviewer_approval_records
                        (agenda_id, purpose, subject, reviewer_id, key_id,
                         issued_at, signature_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agenda_id,
                        approval_record["purpose"],
                        approval_record["subject"],
                        approval_record["reviewer_id"],
                        approval_record["key_id"],
                        approval_record["issued_at"],
                        approval_record["signature_hash"],
                    ),
                )
            transition_context = dict(context.__dict__)
            if reviewer_approval is not None:
                transition_context["reviewer_approval"] = (
                    reviewer_approval.public_record()
                )
            db.execute(
                """
                INSERT INTO evidence_state_transitions
                    (agenda_id, experiment_run_id, from_state, to_state, actor,
                     context_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    agenda_id,
                    experiment_run_id,
                    current,
                    next_state,
                    actor,
                    _dump(transition_context),
                ),
            )
            if next_state == "scientifically_decided":
                db.execute(
                    """
                    INSERT INTO scientific_decision_records
                        (agenda_id, experiment_run_id, evidence_audit_record_id,
                         verdict, verdict_hash, evidence_decision_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agenda_id,
                        experiment_run_id,
                        audit_record_id,
                        context.verdict,
                        _canonical_hash(context.verdict_hash),
                        _dump(evidence_decision_payload),
                    ),
                )
            db.commit()
            return next_state
        except Exception:
            db.rollback()
            raise

    def save_frontier(self, packet: FrontierPacket) -> int:
        packet.validate()
        gate = evaluate_frontier(packet)
        payload = packet.to_json()
        content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        existing = db.fetchone(
            "SELECT id FROM frontier_packets WHERE content_hash=?",
            (content_hash,),
        )
        if existing:
            packet.frontier_packet_id = int(existing["id"])
            return packet.frontier_packet_id
        packet_id = db.insert_returning_id(
            """
            INSERT INTO frontier_packets
                (agenda_id, research_problem_id, retrieved_at, coverage_json,
                 problem_status, strongest_recent_work_json,
                 latest_benchmarks_json, nearest_prior_art_json,
                 contribution_delta_json, obsolete_evidence_json,
                 counterevidence_json, why_not_obsolete,
                 minimum_falsification_experiment_json, gate_allowed,
                 gate_reason_codes_json, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                packet.agenda_id,
                packet.research_problem_id,
                packet.retrieved_at,
                _dump(packet.coverage),
                packet.problem_status,
                _dump(packet.strongest_recent_work),
                _dump(packet.latest_benchmarks),
                _dump(packet.nearest_prior_art),
                _dump(packet.contribution_delta),
                _dump(packet.obsolete_or_duplicate_evidence),
                _dump(packet.counterevidence_and_negative_results),
                packet.why_not_obsolete,
                _dump(packet.minimum_falsification_experiment),
                1 if gate.allowed else 0,
                _dump(gate.reason_codes),
                content_hash,
            ),
        )
        db.commit()
        packet.frontier_packet_id = packet_id
        return packet_id

    def save_decision(self, packet: IdeaDecisionPacket) -> int:
        packet.validate()
        frontier = db.fetchone(
            "SELECT agenda_id, gate_allowed FROM frontier_packets WHERE id=?",
            (packet.frontier_packet_id,),
        )
        if not frontier or int(frontier.get("agenda_id") or 0) != packet.agenda_id:
            raise MetaHarnessPersistenceError("frontier packet scope mismatch")
        if int(frontier.get("gate_allowed") or 0) != 1:
            raise MetaHarnessPersistenceError(
                "portfolio decision cannot bypass a rejected Frontier Gate"
            )
        packet_id = db.insert_returning_id(
            """
            INSERT INTO idea_decision_packets
                (agenda_id, idea_id, frontier_packet_id, decision,
                 estimates_json, candidate_family, correlation_keys_json,
                 reason_codes_json, revisit_condition_json, revisit_after,
                 policy_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                packet.agenda_id,
                packet.idea_id,
                packet.frontier_packet_id,
                packet.decision,
                _dump(_estimate_payload(packet)),
                packet.candidate_family,
                _dump(packet.correlation_keys),
                _dump(packet.reason_codes),
                _dump(packet.revisit_condition) if packet.revisit_condition else None,
                packet.revisit_after,
                packet.policy_version,
            ),
        )
        db.commit()
        packet.decision_packet_id = packet_id
        return packet_id

    def issue_grant(self, grant: ResourceGrant) -> int:
        """Reserve agenda resources and persist a grant in one transaction."""
        grant.validate()
        try:
            lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            agenda = db.fetchone(
                f"SELECT * FROM research_agendas WHERE id=?{lock}",
                (grant.agenda_id,),
            )
            if not agenda or agenda.get("status") != "active":
                raise MetaHarnessPersistenceError("agenda is not active")
            decision = db.fetchone(
                """
                SELECT agenda_id, idea_id, decision
                FROM idea_decision_packets
                WHERE id=?
                """,
                (grant.decision_packet_id,),
            )
            if (
                not decision
                or int(decision.get("agenda_id") or 0) != grant.agenda_id
                or int(decision.get("idea_id") or 0) != grant.idea_id
                or decision.get("decision") not in {"promote", "revisit"}
            ):
                raise MetaHarnessPersistenceError(
                    "ResourceGrant requires a scoped promote/revisit decision"
                )
            existing = db.fetchone(
                """
                SELECT id, reservation_id FROM resource_grants
                WHERE agenda_id=? AND idempotency_key=?
                """,
                (grant.agenda_id, grant.idempotency_key),
            )
            if existing:
                db.commit()
                grant.grant_id = int(existing["id"])
                grant.reservation_id = int(existing["reservation_id"])
                return grant.grant_id
            active_grants = db.fetchone(
                """
                SELECT COUNT(*) AS count
                FROM resource_grants
                WHERE agenda_id=? AND status='active'
                  AND expires_at > CURRENT_TIMESTAMP
                """,
                (grant.agenda_id,),
            )
            if int((active_grants or {}).get("count") or 0) >= int(
                agenda.get("max_concurrency") or 1
            ):
                raise MetaHarnessPersistenceError(
                    "agenda max_concurrency would be exceeded"
                )
            token_budget = int(agenda.get("token_budget") or 0)
            token_total = (
                int(agenda.get("token_spent") or 0)
                + int(agenda.get("token_reserved") or 0)
                + grant.token_cap
            )
            gpu_budget = float(agenda.get("gpu_hours_budget") or 0)
            gpu_total = (
                float(agenda.get("gpu_hours_spent") or 0)
                + float(agenda.get("gpu_hours_reserved") or 0)
                + grant.max_gpu_hours
            )
            if token_budget <= 0 or token_total > token_budget:
                raise MetaHarnessPersistenceError("agenda token hard cap exceeded")
            if grant.max_gpu_hours > 0 and gpu_total > gpu_budget:
                raise MetaHarnessPersistenceError("agenda GPU-hour hard cap exceeded")
            agenda_backends = set(
                json.loads(agenda.get("backend_allowlist_json") or "[]")
            )
            if not set(grant.backend_allowlist).issubset(agenda_backends):
                raise MetaHarnessPersistenceError(
                    "ResourceGrant backend exceeds agenda allowlist"
                )
            reservation_id = db.insert_returning_id(
                """
                INSERT INTO agenda_resource_ledger
                    (agenda_id, operation, idempotency_key, token_reserved,
                     gpu_hours_reserved, status)
                VALUES (?, 'resource_grant', ?, ?, ?, 'reserved')
                RETURNING id
                """,
                (
                    grant.agenda_id,
                    f"grant:{grant.idempotency_key}",
                    grant.token_cap,
                    grant.max_gpu_hours,
                ),
            )
            db.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved+?,
                    gpu_hours_reserved=gpu_hours_reserved+?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (grant.token_cap, grant.max_gpu_hours, grant.agenda_id),
            )
            grant_id = db.insert_returning_id(
                """
                INSERT INTO resource_grants
                    (agenda_id, idea_id, decision_packet_id, stage, token_cap,
                     gpu_class, max_gpu_hours, backend_allowlist_json,
                     artifact_requirements_json, expires_at, grant_reason,
                     reservation_id, status, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    grant.agenda_id,
                    grant.idea_id,
                    grant.decision_packet_id,
                    grant.stage,
                    grant.token_cap,
                    grant.gpu_class,
                    grant.max_gpu_hours,
                    _dump(grant.backend_allowlist),
                    _dump(grant.artifact_requirements),
                    grant.expires_at,
                    grant.grant_reason,
                    reservation_id,
                    grant.status,
                    grant.idempotency_key,
                ),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET resource_grant_id=?, status='queued',
                    stage='portfolio_granted', updated_at=CURRENT_TIMESTAMP
                WHERE agenda_id=? AND deep_insight_id=?
                  AND stage='awaiting_portfolio_decision'
                """,
                (grant_id, grant.agenda_id, grant.idea_id),
            )
            db.commit()
            grant.grant_id = grant_id
            grant.reservation_id = reservation_id
            return grant_id
        except Exception:
            db.rollback()
            raise

    def reconcile_expired_grants(self, *, agenda_id: int | None = None) -> int:
        """Release expired reservations after restart without completing work.

        Expiry is operational failure/withdrawal of authority. It never marks a
        task completed and never creates an OutcomeRecord.
        """
        params: tuple[Any, ...] = ()
        scope = ""
        if agenda_id is not None:
            if int(agenda_id) <= 0:
                raise MetaHarnessPersistenceError("agenda_id must be positive")
            scope = " AND rg.agenda_id=?"
            params = (int(agenda_id),)
        try:
            lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            rows = db.fetchall(
                f"""
                SELECT rg.id, rg.agenda_id, rg.reservation_id,
                       arl.token_reserved, arl.gpu_hours_reserved,
                       arl.status AS reservation_status
                FROM resource_grants rg
                JOIN agenda_resource_ledger arl ON arl.id=rg.reservation_id
                WHERE rg.status='active'
                  AND rg.expires_at <= CURRENT_TIMESTAMP{scope}{lock}
                """,
                params,
            )
            reconciled = 0
            for row in rows:
                grant_id = int(row["id"])
                if row.get("reservation_status") == "reserved":
                    db.execute(
                        """
                        UPDATE research_agendas
                        SET token_reserved=token_reserved-?,
                            gpu_hours_reserved=gpu_hours_reserved-?,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                        """,
                        (
                            int(row.get("token_reserved") or 0),
                            float(row.get("gpu_hours_reserved") or 0),
                            int(row["agenda_id"]),
                        ),
                    )
                    db.execute(
                        """
                        UPDATE agenda_resource_ledger
                        SET status='released', release_reason='grant_expired',
                            settled_at=CURRENT_TIMESTAMP
                        WHERE id=? AND status='reserved'
                        """,
                        (int(row["reservation_id"]),),
                    )
                db.execute(
                    """
                    UPDATE resource_grant_usage_reservations
                    SET status='released', release_reason='grant_expired',
                        settled_at=CURRENT_TIMESTAMP
                    WHERE resource_grant_id=? AND status='reserved'
                    """,
                    (grant_id,),
                )
                db.execute(
                    """
                    UPDATE resource_grants
                    SET status='expired'
                    WHERE id=? AND agenda_id=? AND status='active'
                    """,
                    (grant_id, int(row["agenda_id"])),
                )
                db.execute(
                    """
                    UPDATE auto_research_jobs
                    SET status='blocked', stage='resource_grant_expired',
                        last_error='ResourceGrant expired before OutcomeRecord',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE resource_grant_id=? AND agenda_id=?
                      AND status NOT IN ('completed', 'failed', 'blocked')
                    """,
                    (grant_id, int(row["agenda_id"])),
                )
                reconciled += 1
            db.commit()
            return reconciled
        except Exception:
            db.rollback()
            raise

    def assemble_and_record_outcome(
        self,
        *,
        resource_grant_id: int,
        experiment_run_id: int,
    ) -> int:
        """Assemble an OutcomeRecord only from persisted metering and evidence.

        The operator API deliberately exposes this method instead of accepting
        caller-supplied token/GPU/effect/verdict values.
        """
        resource_grant_id = int(resource_grant_id)
        experiment_run_id = int(experiment_run_id)
        if min(resource_grant_id, experiment_run_id) <= 0:
            raise MetaHarnessPersistenceError(
                "resource_grant_id and experiment_run_id must be positive"
            )
        grant = db.fetchone(
            """
            SELECT rg.*, idp.estimates_json
            FROM resource_grants AS rg
            JOIN idea_decision_packets AS idp
              ON idp.id=rg.decision_packet_id
             AND idp.agenda_id=rg.agenda_id
             AND idp.idea_id=rg.idea_id
            WHERE rg.id=?
            """,
            (resource_grant_id,),
        )
        if not grant:
            raise MetaHarnessPersistenceError(
                "ResourceGrant and decision packet were not found"
            )
        run = db.fetchone(
            """
            SELECT id, agenda_id, deep_insight_id, status,
                   scientific_evidence_state, baseline_metric_value,
                   effect_size, hypothesis_verdict
            FROM experiment_runs
            WHERE id=? AND resource_grant_id=?
            """,
            (experiment_run_id, resource_grant_id),
        )
        if (
            not run
            or int(run.get("agenda_id") or 0)
            != int(grant.get("agenda_id") or 0)
            or int(run.get("deep_insight_id") or 0)
            != int(grant.get("idea_id") or 0)
        ):
            raise MetaHarnessPersistenceError(
                "experiment run does not match ResourceGrant scope"
            )
        llm_usage = db.fetchone(
            """
            SELECT
                COALESCE(SUM(CASE WHEN status='settled' THEN tokens_used ELSE 0 END), 0)
                    AS tokens_used,
                COALESCE(SUM(CASE WHEN status='reserved' THEN 1 ELSE 0 END), 0)
                    AS open_reservations
            FROM resource_grant_usage_reservations
            WHERE resource_grant_id=?
            """,
            (resource_grant_id,),
        ) or {}
        if int(llm_usage.get("open_reservations") or 0):
            raise MetaHarnessPersistenceError(
                "trusted outcome assembly found open LLM reservations"
            )
        compute_rows = db.fetchall(
            """
            SELECT id, status, backend_kind, artifact_manifest_json,
                   usage_json, failure_reason
            FROM compute_jobs_v1
            WHERE resource_grant_id=?
            ORDER BY id
            """,
            (resource_grant_id,),
        )
        active_compute = [
            str(row.get("status") or "")
            for row in compute_rows
            if str(row.get("status") or "")
            not in {"succeeded", "failed", "cancelled", "timed_out"}
        ]
        if active_compute:
            raise MetaHarnessPersistenceError(
                "trusted outcome assembly found non-terminal compute jobs"
            )
        actual_gpu_hours = 0.0
        wall_seconds = 0.0
        compute_artifacts = []
        for row in compute_rows:
            usage = _load_mapping(row.get("usage_json"))
            if not usage:
                raise MetaHarnessPersistenceError(
                    "compute outcome is missing durable usage accounting"
                )
            actual_gpu_hours += float(usage.get("gpu_hours") or 0)
            wall_seconds += float(usage.get("wall_seconds") or 0)
            compute_artifacts.append(
                {
                    "compute_job_id": int(row["id"]),
                    "backend_kind": str(row.get("backend_kind") or ""),
                    "status": str(row.get("status") or ""),
                    "artifact_manifest": _load_mapping(
                        row.get("artifact_manifest_json")
                    ),
                    "failure_reason": row.get("failure_reason"),
                }
            )
        decision = db.fetchone(
            """
            SELECT id, verdict, evidence_decision_json
            FROM scientific_decision_records
            WHERE agenda_id=? AND experiment_run_id=?
            """,
            (int(grant["agenda_id"]), experiment_run_id),
        )
        operational_verdict = str(
            run.get("hypothesis_verdict") or "inconclusive"
        ).strip().lower()
        verdict = str((decision or {}).get("verdict") or "").strip().lower()
        if verdict not in {"supported", "refuted", "inconclusive", "invalid"}:
            verdict = (
                operational_verdict
                if operational_verdict in {"refuted", "inconclusive", "invalid"}
                else "inconclusive"
            )
        experiment_artifacts = db.fetchall(
            """
            SELECT id, artifact_type, path, metric_key, metric_value, metadata
            FROM experiment_artifacts
            WHERE agenda_id=? AND run_id=?
            ORDER BY id
            """,
            (int(grant["agenda_id"]), experiment_run_id),
        )
        route_observations = db.fetchall(
            """
            SELECT lro.id, lro.role, lro.provider, lro.model,
                   lro.prompt_version, lro.status
            FROM llm_route_observations AS lro
            JOIN resource_grant_usage_reservations AS rgu
              ON rgu.id=lro.grant_usage_reservation_id
            WHERE rgu.resource_grant_id=?
            ORDER BY lro.id
            """,
            (resource_grant_id,),
        )
        estimates = _load_mapping(grant.get("estimates_json"))

        def estimate(name: str) -> float | None:
            row = estimates.get(name)
            if not isinstance(row, dict) or row.get("value") is None:
                return None
            try:
                return float(row["value"])
            except (TypeError, ValueError):
                return None

        actual_tokens = int(llm_usage.get("tokens_used") or 0)
        observed_success = 1.0 if verdict == "supported" else 0.0
        prediction_error = {
            "success_probability": (
                observed_success - estimate("success_probability")
                if estimate("success_probability") is not None
                else None
            ),
            "token_cost": (
                actual_tokens - estimate("expected_token_cost")
                if estimate("expected_token_cost") is not None
                else None
            ),
            "gpu_cost": (
                actual_gpu_hours - estimate("expected_gpu_cost")
                if estimate("expected_gpu_cost") is not None
                else None
            ),
            "impact": (
                float(run.get("effect_size")) - estimate("expected_impact")
                if run.get("effect_size") is not None
                and estimate("expected_impact") is not None
                else None
            ),
        }
        decision_payload = _load_mapping((decision or {}).get("evidence_decision_json"))
        decision_reason_codes = _load_list(
            (decision or {}).get("reason_codes_json")
        ) or _load_list(decision_payload.get("reason_codes"))
        outcome = OutcomeRecord(
            agenda_id=int(grant["agenda_id"]),
            idea_id=int(grant["idea_id"]),
            resource_grant_id=resource_grant_id,
            experiment_run_id=experiment_run_id,
            actual_tokens=actual_tokens,
            actual_gpu_hours=actual_gpu_hours,
            wall_seconds=wall_seconds,
            execution_result=str(run.get("status") or "unknown"),
            effect=(
                float(run["effect_size"])
                if run.get("effect_size") is not None
                else None
            ),
            baseline=(
                float(run["baseline_metric_value"])
                if run.get("baseline_metric_value") is not None
                else None
            ),
            verdict=verdict,
            new_information={
                "scientific_decision_record_id": (
                    int(decision["id"]) if decision else None
                ),
                "decision_reason_codes": decision_reason_codes,
                "compute_job_statuses": [
                    str(row.get("status") or "") for row in compute_rows
                ],
            },
            state_decision=str(
                run.get("scientific_evidence_state") or "planned"
            ),
            prediction_error=prediction_error,
            artifact_manifest={
                "source": "trusted_persistence_v1",
                "compute": compute_artifacts,
                "experiment_artifacts": [dict(row) for row in experiment_artifacts],
                "llm_route_observations": [
                    dict(row) for row in route_observations
                ],
            },
        )
        return self.record_outcome(outcome)

    def record_outcome(self, outcome: OutcomeRecord) -> int:
        """Persist actual usage and consume the reserved grant atomically."""
        outcome.validate()
        try:
            lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            grant = db.fetchone(
                f"SELECT * FROM resource_grants WHERE id=?{lock}",
                (outcome.resource_grant_id,),
            )
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != outcome.agenda_id
                or int(grant.get("idea_id") or 0) != outcome.idea_id
            ):
                raise MetaHarnessPersistenceError("OutcomeRecord grant scope mismatch")
            if grant.get("status") == "consumed":
                existing = db.fetchone(
                    "SELECT id FROM outcome_records WHERE resource_grant_id=?",
                    (outcome.resource_grant_id,),
                )
                if existing:
                    db.commit()
                    outcome.outcome_record_id = int(existing["id"])
                    return outcome.outcome_record_id
            if grant.get("status") != "active":
                raise MetaHarnessPersistenceError("grant is not active")
            if outcome.actual_tokens > int(grant.get("token_cap") or 0):
                raise MetaHarnessPersistenceError("actual tokens exceed ResourceGrant")
            if outcome.actual_gpu_hours > float(grant.get("max_gpu_hours") or 0):
                raise MetaHarnessPersistenceError("actual GPU hours exceed ResourceGrant")
            usage = db.fetchone(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN status='settled' THEN tokens_used ELSE 0 END), 0)
                        AS tokens_used,
                    COALESCE(SUM(CASE WHEN status='reserved' THEN 1 ELSE 0 END), 0)
                        AS open_reservations
                FROM resource_grant_usage_reservations
                WHERE resource_grant_id=?
                """,
                (outcome.resource_grant_id,),
            )
            if int((usage or {}).get("open_reservations") or 0):
                raise MetaHarnessPersistenceError(
                    "OutcomeRecord cannot close a grant with open LLM reservations"
                )
            metered_tokens = int((usage or {}).get("tokens_used") or 0)
            if metered_tokens != outcome.actual_tokens:
                raise MetaHarnessPersistenceError(
                    "OutcomeRecord token usage does not match metered grant usage"
                )
            reservation_id = int(grant["reservation_id"])
            ledger = db.fetchone(
                f"SELECT * FROM agenda_resource_ledger WHERE id=?{lock}",
                (reservation_id,),
            )
            if not ledger or ledger.get("status") != "reserved":
                raise MetaHarnessPersistenceError("grant reservation is not active")
            outcome_id = db.insert_returning_id(
                """
                INSERT INTO outcome_records
                    (agenda_id, idea_id, resource_grant_id, experiment_run_id,
                     actual_tokens,
                     actual_gpu_hours, wall_seconds, execution_result, effect,
                     baseline, verdict, new_information_json, state_decision,
                     prediction_error_json, artifact_manifest_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    outcome.agenda_id,
                    outcome.idea_id,
                    outcome.resource_grant_id,
                    outcome.experiment_run_id,
                    outcome.actual_tokens,
                    outcome.actual_gpu_hours,
                    outcome.wall_seconds,
                    outcome.execution_result,
                    outcome.effect,
                    outcome.baseline,
                    outcome.verdict,
                    _dump(outcome.new_information),
                    outcome.state_decision,
                    _dump(outcome.prediction_error),
                    _dump(outcome.artifact_manifest),
                ),
            )
            token_reserved = int(ledger.get("token_reserved") or 0)
            gpu_reserved = float(ledger.get("gpu_hours_reserved") or 0)
            db.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved-?,
                    gpu_hours_reserved=gpu_hours_reserved-?,
                    token_spent=token_spent+?,
                    gpu_hours_spent=gpu_hours_spent+?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    token_reserved,
                    gpu_reserved,
                    outcome.actual_tokens,
                    outcome.actual_gpu_hours,
                    outcome.agenda_id,
                ),
            )
            db.execute(
                """
                UPDATE agenda_resource_ledger
                SET tokens_used=?, gpu_hours_used=?, status='settled',
                    settled_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (outcome.actual_tokens, outcome.actual_gpu_hours, reservation_id),
            )
            db.execute(
                "UPDATE resource_grants SET status='consumed' WHERE id=? AND agenda_id=?",
                (outcome.resource_grant_id, outcome.agenda_id),
            )
            db.commit()
            outcome.outcome_record_id = outcome_id
            return outcome_id
        except Exception:
            db.rollback()
            raise

    def save_route_observation(self, observation: RouteObservation) -> int:
        observation_id = db.insert_returning_id(
            """
            INSERT INTO llm_route_observations
                (agenda_id, idea_id, role, provider, model, model_family, prompt_version,
                 input_tokens, output_tokens, cost_usd, status, failure_reason,
                 grant_usage_reservation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                observation.agenda_id,
                observation.idea_id,
                observation.role,
                observation.provider,
                observation.model,
                observation.model_family,
                observation.prompt_version,
                observation.input_tokens,
                observation.output_tokens,
                observation.cost_usd,
                observation.status,
                observation.failure_reason,
                observation.reservation_id,
            ),
        )
        db.commit()
        return observation_id
