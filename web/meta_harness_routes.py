"""Minimal operator-authenticated API for meta-harness-v1.

The blueprint has no startup actions. Mutations fail closed unless an operator
token is configured in the environment and supplied in the request header.
"""

from __future__ import annotations

import hmac
import os

from flask import Blueprint, jsonify, request

from agents.agenda_loader import parse_agenda
from agents.agenda_repository import AgendaRepository
from agents.direction_intake import parse_direction_payload
from contracts.meta_harness import (
    FrontierPacket,
    IdeaDecisionPacket,
    ResourceGrant,
)
from db import database as db
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.frontier import evaluate_frontier
from meta_harness.frontier_source import (
    EvidenceGraphFrontierSource,
    FrontierAssessment,
)
from meta_harness.portfolio import decide_portfolio
from meta_harness.repository import MetaHarnessRepository


blueprint = Blueprint("meta_harness_v1", __name__, url_prefix="/api/meta-harness/v1")


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
