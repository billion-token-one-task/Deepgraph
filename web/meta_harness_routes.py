"""Minimal operator-authenticated API for meta-harness-v1.

The blueprint has no startup actions. Mutations fail closed unless an operator
token is configured in the environment and supplied in the request header.
"""

from __future__ import annotations

import hmac
import os
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, render_template, request

from config import (
    HARNESS_CANDIDATE_ROOT,
    LLM_PROVIDERS,
    LLM_PROVIDER_HOST_ALLOWLIST,
    LLM_PROVIDER_STORE,
    HARNESS_DATABASE_NAMESPACE_PREFIX,
    HARNESS_EVALUATOR_ARTIFACT_ROOT,
    HARNESS_EVALUATOR_ISOLATION_BINARY,
    HARNESS_EVALUATOR_ROOT,
    HARNESS_HOLDOUT_ROOT,
    HARNESS_MAX_CHANGED_LINES,
    HARNESS_MAX_MODULES,
    HARNESS_POLICY_VERSION,
    HARNESS_PRODUCTION_DATABASE_NAMESPACE,
    HARNESS_PRODUCTION_PATH,
)
from agents.agenda_loader import parse_agenda
from agents.agenda_repository import AgendaRepository
from agents.direction_intake import parse_direction_payload
from contracts.meta_harness import (
    FrontierEvaluationAuthority,
    FrontierPacket,
    IdeaDecisionPacket,
    ResourceGrant,
)
from db import database as db
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.frontier import evaluate_frontier
from meta_harness.frontier_authority import FrontierAuthorityRepository
from meta_harness.frontier_bootstrap import run_bootstrap_evaluation
from meta_harness.frontier_source import (
    EvidenceGraphFrontierSource,
    FrontierAssessment,
)
from meta_harness.portfolio import decide_portfolio
from meta_harness.repository import MetaHarnessRepository
from meta_harness.evaluator_runner import (
    EvaluatorSuiteSpec,
    IsolatedEvaluatorRunner,
)
from meta_harness.harness_evolution import HarnessCandidate, HarnessPolicy
from meta_harness.harness_repository import HarnessRepository
from meta_harness.backends.colab_durable import ColabWorkSpec
from meta_harness.ingestion_queue import (
    ScopedIngestionRepository,
    ScopedIngestionRequest,
)
from orchestrator.meta_compute_runtime import submit_colab_work
from web import provider_config


blueprint = Blueprint(
    "meta_harness_v1",
    __name__,
    url_prefix="/api/meta-harness/v1",
    # A dedicated template folder keeps this page out of the shared frontend
    # templates, so operator tooling and the public UI never collide.
    template_folder="templates",
)


def _payload() -> dict:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("JSON object body required")
    return value


def _require_operator() -> None:
    expected = os.environ.get("DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN", "")
    supplied = request.headers.get("X-DeepGraph-Operator-Token", "")
    if not expected:
        raise PermissionError("operator mutation API is disabled")
    if not supplied or not hmac.compare_digest(expected, supplied):
        raise PermissionError("operator authorization failed")


def _error(exc: Exception, status: int = 400):
    try:
        db.rollback()
    except Exception:
        pass
    if isinstance(exc, PermissionError):
        status = 403
    return jsonify({"status": "error", "error": str(exc)}), status


@blueprint.get("/status")
def status():
    """Count-only operational view; it does not return business row contents."""
    try:
        counts = {}
        for label, table in (
            ("agendas", "research_agendas"),
            ("frontier_packets", "frontier_packets"),
            ("decision_packets", "idea_decision_packets"),
            ("active_grants", "resource_grants"),
            ("outcomes", "outcome_records"),
            ("compute_jobs", "compute_jobs_v1"),
            ("scoped_ingestion_jobs", "scoped_ingestion_jobs_v1"),
            ("colab_work_requests", "colab_work_requests_v1"),
            ("harness_candidates", "harness_candidates"),
        ):
            where = " WHERE status='active'" if label == "active_grants" else ""
            row = db.fetchone(f"SELECT COUNT(*) AS count FROM {table}{where}")
            counts[label] = int((row or {}).get("count") or 0)
        return jsonify(
            {
                "schema_version": "meta-harness-v1",
                "counts": counts,
                "mutation_api_enabled": bool(
                    os.environ.get("DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN")
                ),
            }
        )
    except Exception as exc:
        return _error(exc, 503)


@blueprint.post("/agendas")
def create_agenda():
    try:
        _require_operator()
        payload = _payload()
        if payload.get("confirmed") is not True:
            raise ValueError("confirmed=true is required after reviewing the agenda echo")
        definition = payload.get("agenda")
        if not isinstance(definition, dict):
            raise ValueError("agenda mapping is required")
        agenda = (
            parse_direction_payload(definition)
            if "direction" in definition
            else parse_agenda(definition)
        )
        agenda_id = AgendaRepository().create(agenda)
        return jsonify(
            {
                "status": "created",
                "agenda_id": agenda_id,
                "backlog_policy": agenda.backlog_policy,
                "token_budget": agenda.token_budget,
                "gpu_hours_budget": agenda.gpu_hours_budget,
            }
        ), 201
    except Exception as exc:
        return _error(exc)


@blueprint.post("/legacy-import")
def import_legacy():
    try:
        _require_operator()
        payload = _payload()
        import_id = AgendaRepository().import_legacy_record(
            agenda_id=int(payload["agenda_id"]),
            entity_type=str(payload["entity_type"]),
            entity_id=int(payload["entity_id"]),
            actor=str(payload["actor"]),
            reason=str(payload["reason"]),
            idempotency_key=str(payload["idempotency_key"]),
        )
        return jsonify({"status": "imported", "legacy_scope_import_id": import_id})
    except Exception as exc:
        return _error(exc)


@blueprint.post("/frontier")
def save_frontier():
    try:
        _require_operator()
        packet = FrontierPacket.from_partial_dict(_payload())
        gate = evaluate_frontier(packet)
        packet_id = MetaHarnessRepository().save_frontier(packet)
        return jsonify(
            {
                "frontier_packet_id": packet_id,
                "allowed": gate.allowed,
                "reason_codes": gate.reason_codes,
            }
        ), (201 if gate.allowed else 409)
    except Exception as exc:
        return _error(exc)


@blueprint.post("/frontier/from-evidence-graph")
def save_frontier_from_evidence_graph():
    """Build evidence arrays from the scoped graph; accept judgments only."""
    try:
        _require_operator()
        payload = _payload()
        assessment_payload = payload.get("assessment")
        if not isinstance(assessment_payload, dict):
            raise ValueError("assessment must be an object")
        assessment = FrontierAssessment(
            problem_status=str(assessment_payload.get("problem_status") or ""),
            contribution_delta=dict(
                assessment_payload.get("contribution_delta") or {}
            ),
            why_not_obsolete=str(
                assessment_payload.get("why_not_obsolete") or ""
            ),
            minimum_falsification_experiment=dict(
                assessment_payload.get("minimum_falsification_experiment") or {}
            ),
            evaluator=str(assessment_payload.get("evaluator") or ""),
            provider=str(assessment_payload.get("provider") or ""),
            model=str(assessment_payload.get("model") or ""),
            prompt_version=str(assessment_payload.get("prompt_version") or ""),
            coverage_start=str(assessment_payload.get("coverage_start") or ""),
            coverage_end=str(assessment_payload.get("coverage_end") or ""),
        )
        packet = EvidenceGraphFrontierSource().build(
            agenda_id=int(payload["agenda_id"]),
            research_problem_id=int(payload["research_problem_id"]),
            assessment=assessment,
        )
        gate = evaluate_frontier(packet)
        packet_id = MetaHarnessRepository().save_frontier(packet)
        return jsonify(
            {
                "frontier_packet_id": packet_id,
                "allowed": gate.allowed,
                "reason_codes": gate.reason_codes,
                "coverage": packet.coverage,
            }
        ), (201 if gate.allowed else 409)
    except Exception as exc:
        return _error(exc)


@blueprint.get("/llm-providers/admin")
def llm_provider_admin_page():
    """Operator page for the non-secret half of provider configuration.

    The page itself carries no data and no credential: every call it makes is
    an operator-token request, and the token is typed in by the operator rather
    than embedded here.
    """
    return render_template("meta_harness/llm_providers.html")


@blueprint.get("/llm-providers")
def list_llm_providers():
    try:
        _require_operator()
        store = provider_config.load_store(LLM_PROVIDER_STORE)
        managed = store["providers"]
        declared = [
            entry for entry in LLM_PROVIDERS if entry.get("source") == "toml"
        ]
        # Independence is judged on what the process would actually route to,
        # which includes the legacy env-configured slots, not just the store.
        runtime = provider_config.effective_pool()
        return jsonify(
            {
                "providers": provider_config.readiness(managed),
                "declared_in_toml": provider_config.readiness(declared),
                "runtime_pool": provider_config.readiness(runtime),
                "independence": provider_config.independence_report(runtime),
                "host_allowlist": list(LLM_PROVIDER_HOST_ALLOWLIST),
                "store_path": str(LLM_PROVIDER_STORE),
                "updated_at": store.get("updated_at"),
                "updated_by": store.get("updated_by"),
                "restart_required_to_apply": True,
            }
        )
    except Exception as exc:
        return _error(exc)


@blueprint.post("/llm-providers")
def upsert_llm_provider():
    """Create or replace one provider entry. Credentials are never accepted."""
    try:
        _require_operator()
        payload = _payload()
        actor = str(payload.pop("actor", "") or "").strip()
        if not actor:
            raise ValueError("actor is required so the change is auditable")
        store = provider_config.upsert(
            LLM_PROVIDER_STORE,
            payload,
            allowed_hosts=LLM_PROVIDER_HOST_ALLOWLIST,
            actor=actor,
        )
        return jsonify(
            {
                "status": "saved",
                "providers": provider_config.readiness(store["providers"]),
                "independence": provider_config.independence_report(
                    store["providers"]
                ),
                "restart_required_to_apply": True,
                "note": "restart deepgraph-web.service to load this route",
            }
        ), 201
    except Exception as exc:
        return _error(exc)


@blueprint.delete("/llm-providers/<name>")
def delete_llm_provider(name: str):
    try:
        _require_operator()
        actor = str(request.args.get("actor") or "").strip()
        if not actor:
            raise ValueError("actor is required so the change is auditable")
        removed = provider_config.remove(LLM_PROVIDER_STORE, name, actor=actor)
        return jsonify(
            {
                "status": "removed" if removed else "not_found",
                "restart_required_to_apply": bool(removed),
            }
        ), (200 if removed else 404)
    except Exception as exc:
        return _error(exc)


@blueprint.post("/frontier/authority")
def issue_frontier_authority():
    """Issue one bounded, single-use Frontier-evaluator bootstrap authority.

    This is the operator's explicit act. It is not a ResourceGrant: it cannot
    reach GPU, an experiment, a proposal, or a second agenda.
    """
    try:
        _require_operator()
        payload = _payload()
        issued_at = datetime.now(timezone.utc)
        ttl_minutes = int(payload.get("ttl_minutes") or 30)
        authority = FrontierEvaluationAuthority(
            agenda_id=int(payload["agenda_id"]),
            research_problem_id=int(payload["research_problem_id"]),
            token_cap=int(payload["token_cap"]),
            issued_at=issued_at.isoformat(),
            expires_at=(issued_at + timedelta(minutes=ttl_minutes)).isoformat(),
            idempotency_key=str(payload["idempotency_key"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            model_family=str(payload["model_family"]),
            prompt_version=str(payload["prompt_version"]),
            evaluator=str(payload["evaluator"]),
            issued_by=str(payload["issued_by"]),
            issue_reason=str(payload["issue_reason"]),
        )
        authority_id = FrontierAuthorityRepository().issue(authority)
        return jsonify(
            {
                "frontier_evaluation_authority_id": authority_id,
                "expires_at": authority.expires_at,
                "token_cap": authority.token_cap,
                "backend_allowlist": list(authority.backend_allowlist),
                "allowed_operations": list(authority.allowed_operations),
            }
        ), 201
    except Exception as exc:
        return _error(exc)


@blueprint.post("/frontier/bootstrap")
def run_frontier_bootstrap():
    """Run the one authorized evaluation and persist its Frontier packet."""
    try:
        _require_operator()
        payload = _payload()
        result = run_bootstrap_evaluation(
            authority_id=int(payload["frontier_evaluation_authority_id"]),
            agenda_id=int(payload["agenda_id"]),
            research_problem_id=int(payload["research_problem_id"]),
            proposer_provider=payload.get("proposer_provider"),
            proposer_model_family=payload.get("proposer_model_family"),
        )
        return jsonify(result), (200 if result.get("gate_allowed") else 202)
    except Exception as exc:
        return _error(exc)


@blueprint.get("/frontier/authority/<int:authority_id>/audit")
def frontier_authority_audit(authority_id: int):
    """Independently verifiable record of one bootstrap. No secrets."""
    try:
        _require_operator()
        agenda_id = int(request.args.get("agenda_id") or 0)
        return jsonify(
            FrontierAuthorityRepository().audit_record(
                authority_id, agenda_id=agenda_id
            )
        )
    except Exception as exc:
        return _error(exc)


@blueprint.post("/portfolio/decide")
def portfolio_decide():
    try:
        _require_operator()
        payload = _payload()
        raw_packets = payload.get("packets")
        if not isinstance(raw_packets, list) or not raw_packets:
            raise ValueError("packets must be a non-empty list")
        packets = [
            IdeaDecisionPacket.from_partial_dict(item)
            for item in raw_packets
            if isinstance(item, dict)
        ]
        decisions = decide_portfolio(
            packets,
            killed_signatures=set(payload.get("killed_signatures") or []),
        )
        repository = MetaHarnessRepository()
        ids = [repository.save_decision(decision) for decision in decisions]
        return jsonify(
            {
                "status": "decided",
                "decision_packet_ids": ids,
                "decisions": [
                    {
                        "idea_id": decision.idea_id,
                        "decision": decision.decision,
                        "reason_codes": decision.reason_codes,
                    }
                    for decision in decisions
                ],
            }
        ), 201
    except Exception as exc:
        return _error(exc)


@blueprint.post("/grants")
def issue_grant():
    try:
        _require_operator()
        grant = ResourceGrant.from_partial_dict(_payload())
        grant_id = MetaHarnessRepository().issue_grant(grant)
        return jsonify(
            {
                "status": "active",
                "resource_grant_id": grant_id,
                "reservation_id": grant.reservation_id,
            }
        ), 201
    except Exception as exc:
        return _error(exc)


@blueprint.post("/runs/<int:run_id>/attach-grant")
def attach_grant(run_id: int):
    try:
        _require_operator()
        payload = _payload()
        MetaHarnessRepository().attach_grant_to_run(
            agenda_id=int(payload["agenda_id"]),
            idea_id=int(payload["idea_id"]),
            experiment_run_id=run_id,
            resource_grant_id=int(payload["resource_grant_id"]),
        )
        return jsonify({"status": "attached", "experiment_run_id": run_id})
    except Exception as exc:
        return _error(exc)


@blueprint.post("/runs/<int:run_id>/evidence-state")
def transition_evidence(run_id: int):
    try:
        _require_operator()
        payload = _payload()
        context = EvidenceTransitionContext(**dict(payload.get("context") or {}))
        state = MetaHarnessRepository().advance_experiment_state(
            agenda_id=int(payload["agenda_id"]),
            experiment_run_id=run_id,
            target=str(payload["target"]),
            context=context,
            actor=str(payload["actor"]),
        )
        return jsonify({"status": "advanced", "scientific_evidence_state": state})
    except Exception as exc:
        return _error(exc)


@blueprint.post("/outcomes")
def record_outcome():
    try:
        _require_operator()
        payload = _payload()
        outcome_id = MetaHarnessRepository().assemble_and_record_outcome(
            resource_grant_id=int(payload["resource_grant_id"]),
            experiment_run_id=int(payload["experiment_run_id"]),
        )
        return jsonify({"status": "recorded", "outcome_record_id": outcome_id}), 201
    except Exception as exc:
        return _error(exc)


@blueprint.post("/ingestion/jobs")
def enqueue_scoped_ingestion():
    try:
        _require_operator()
        payload = _payload()
        paper_ids = payload.get("paper_ids")
        if not isinstance(paper_ids, list):
            raise ValueError("paper_ids must be an array")
        job_id = ScopedIngestionRepository().enqueue(
            ScopedIngestionRequest(
                agenda_id=int(payload["agenda_id"]),
                idea_id=int(payload["idea_id"]),
                resource_grant_id=int(payload["resource_grant_id"]),
                stage=str(payload["stage"]),
                idempotency_key=str(payload["idempotency_key"]),
                paper_ids=tuple(str(value) for value in paper_ids),
                max_attempts=int(payload.get("max_attempts") or 3),
            )
        )
        return jsonify(
            {"status": "queued", "scoped_ingestion_job_id": job_id}
        ), 202
    except Exception as exc:
        return _error(exc)


@blueprint.post("/compute/colab/jobs")
def enqueue_colab_job():
    try:
        _require_operator()
        payload = _payload()
        command_tokens = payload.get("command_tokens")
        artifact_map = payload.get("artifact_map")
        environment = payload.get("environment") or {}
        if not isinstance(command_tokens, list):
            raise ValueError("command_tokens must be an array")
        if not isinstance(artifact_map, dict):
            raise ValueError("artifact_map must be an object")
        if not isinstance(environment, dict):
            raise ValueError("environment must be an object")
        job = submit_colab_work(
            ColabWorkSpec(
                agenda_id=int(payload["agenda_id"]),
                idea_id=int(payload["idea_id"]),
                experiment_run_id=int(payload["experiment_run_id"]),
                resource_grant_id=int(payload["resource_grant_id"]),
                stage=str(payload["stage"]),
                idempotency_key=str(payload["idempotency_key"]),
                code_dir=str(payload["code_dir"]),
                command_tokens=tuple(str(value) for value in command_tokens),
                environment={
                    str(key): str(value)
                    for key, value in environment.items()
                },
                artifact_map={
                    str(key): str(value)
                    for key, value in artifact_map.items()
                },
                artifact_output_dir=str(payload["artifact_output_dir"]),
                timeout_seconds=int(payload["timeout_seconds"]),
            )
        )
        return jsonify(
            {
                "status": job.status,
                "backend_kind": job.backend_kind,
                "backend_job_id": job.backend_job_id,
                "idempotency_key": job.idempotency_key,
            }
        ), 202
    except Exception as exc:
        return _error(exc)


@blueprint.post("/harness/candidates/<int:candidate_id>/evaluate")
def run_isolated_harness_evaluation(candidate_id: int):
    try:
        _require_operator()
        if (
            not str(HARNESS_PRODUCTION_PATH).strip()
            or not str(HARNESS_PRODUCTION_DATABASE_NAMESPACE).strip()
        ):
            raise PermissionError(
                "isolated evaluator production boundary is not configured"
            )
        payload = _payload()
        patch_id = int(payload["patch_id"])
        arguments = payload.get("arguments") or []
        if not isinstance(arguments, list):
            raise ValueError("arguments must be an array")
        row = db.fetchone(
            """
            SELECT hc.*, hp.id AS patch_id, hp.patch_hash,
                   hp.agenda_id AS patch_agenda_id
            FROM harness_candidates AS hc
            JOIN harness_patches AS hp ON hp.candidate_id=hc.id
            WHERE hc.id=? AND hp.id=?
            """,
            (candidate_id, patch_id),
        )
        if (
            not row
            or int(row.get("agenda_id") or 0)
            != int(row.get("patch_agenda_id") or 0)
        ):
            raise ValueError("candidate/patch scope mismatch")
        policy = HarnessPolicy(
            version=HARNESS_POLICY_VERSION,
            max_modules=HARNESS_MAX_MODULES,
            max_changed_lines=HARNESS_MAX_CHANGED_LINES,
            candidate_root=str(HARNESS_CANDIDATE_ROOT),
            namespace_prefix=HARNESS_DATABASE_NAMESPACE_PREFIX,
        )
        candidate = HarnessCandidate(
            agenda_id=int(row["agenda_id"]),
            candidate_ref=str(row["candidate_ref"]),
            base_commit=str(row["base_commit"]),
            worktree_path=str(row["worktree_path"]),
            database_namespace=str(row["database_namespace"]),
            artifact_namespace=str(row["artifact_namespace"]),
        )
        evaluation = IsolatedEvaluatorRunner(
            policy=policy,
            production_path=HARNESS_PRODUCTION_PATH,
            production_database_namespace=(
                HARNESS_PRODUCTION_DATABASE_NAMESPACE
            ),
            evaluator_root=str(HARNESS_EVALUATOR_ROOT),
            holdout_root=str(HARNESS_HOLDOUT_ROOT),
            artifact_root=str(HARNESS_EVALUATOR_ARTIFACT_ROOT),
            isolation_binary=HARNESS_EVALUATOR_ISOLATION_BINARY,
        ).run(
            candidate=candidate,
            spec=EvaluatorSuiteSpec(
                suite=str(payload["suite"]),
                evaluator_root=str(payload["evaluator_root"]),
                evaluator_entrypoint=str(payload["evaluator_entrypoint"]),
                evaluator_hash=str(payload["evaluator_hash"]),
                suite_root=str(payload["suite_root"]),
                suite_hash=str(payload["suite_hash"]),
                output_dir=str(payload["output_dir"]),
                timeout_seconds=int(payload.get("timeout_seconds") or 1800),
                arguments=tuple(
                    str(value) for value in arguments
                ),
            ),
        )
        evaluation_id = HarnessRepository().save_evaluation(
            evaluation,
            candidate_id=candidate_id,
            patch_id=patch_id,
        )
        return jsonify(
            {
                "status": evaluation.status,
                "harness_evaluation_run_id": evaluation_id,
                "evaluator_hash": evaluation.evaluator_hash,
                "failure_reason": evaluation.failure_reason,
            }
        ), (201 if evaluation.status == "passed" else 409)
    except Exception as exc:
        return _error(exc)
