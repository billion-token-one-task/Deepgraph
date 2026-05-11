"""Flask Blueprint for the agenda-driven research loop (issue #9 step 6).

Endpoints (mounted under /api/research_agenda):
    POST   /api/research_agenda                     upload an agenda (YAML/JSON)
    GET    /api/research_agenda/current             active agenda
    GET    /api/research_agenda                     list all agendas
    POST   /api/research_agenda/select              run selector + dispatch
    GET    /api/research_agenda/selection/latest    latest selection (active agenda)
    GET    /api/research_agenda/selection/<id>      a single selection
    POST   /api/research_agenda/selection/<id>/review     run reviewer
    POST   /api/research_agenda/selection/<id>/plan       build revision plan
    GET    /api/research_agenda/loop/<selection_id>       full end-to-end snapshot
"""

from __future__ import annotations

import json

import yaml  # type: ignore
from flask import Blueprint, jsonify, request

from agents import agenda_loader, agenda_orchestrator, agenda_selector, reviewer_adapter, revision_planner
from contracts.agenda import LoopInspectionSnapshot
from contracts.base import ContractValidationError
from db import database as db


bp = Blueprint("research_agenda", __name__, url_prefix="/api/research_agenda")


def _agenda_to_dict(agenda):
    if not agenda:
        return None
    return {
        "id": agenda.agenda_id,
        "version": agenda.version,
        "name": agenda.name,
        "description": agenda.description,
        "focus": agenda.focus,
        "prefer": agenda.prefer,
        "reject": agenda.reject,
        "required_output": agenda.required_output,
        "is_active": agenda.is_active,
        "raw_config": agenda.raw_config,
    }


def _parse_payload_from_request():
    """Accept JSON body, raw YAML body, or multipart 'file' upload."""
    if request.files and "file" in request.files:
        text = request.files["file"].read().decode("utf-8")
        return yaml.safe_load(text) or {}
    if request.is_json:
        return request.get_json(silent=True) or {}
    raw = request.get_data(as_text=True) or ""
    raw = raw.strip()
    if not raw:
        return {}
    # try JSON, then YAML
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    return yaml.safe_load(raw) or {}


# ---------- agenda CRUD ----------


@bp.route("", methods=["POST"])
def upload_agenda():
    try:
        payload = _parse_payload_from_request()
        agenda = agenda_loader.parse_agenda(payload)
        agenda_id = agenda_loader.save_agenda(agenda)
    except ContractValidationError as e:
        return jsonify({"error": "invalid_agenda", "message": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "parse_failed", "message": str(e)}), 400
    agenda.agenda_id = agenda_id
    return jsonify({"agenda": _agenda_to_dict(agenda)}), 201


@bp.route("", methods=["GET"])
def list_agendas():
    only_active = request.args.get("active", "").lower() in ("1", "true", "yes")
    rows = agenda_loader.list_agendas(only_active=only_active)
    return jsonify({"agendas": [_agenda_to_dict(a) for a in rows]})


@bp.route("/current", methods=["GET"])
def current_agenda():
    agenda = agenda_loader.get_active_agenda()
    if agenda is None:
        return jsonify({"agenda": None}), 404
    return jsonify({"agenda": _agenda_to_dict(agenda)})


# ---------- selection ----------


@bp.route("/select", methods=["POST"])
def trigger_selection():
    body = request.get_json(silent=True) or {}
    agenda_id = body.get("agenda_id")
    dispatch_mode = body.get("dispatch_mode", "auto")
    if dispatch_mode not in ("auto", "link", "enqueue", "none"):
        return (
            jsonify({"error": "invalid_dispatch_mode", "message": "must be auto|link|enqueue|none"}),
            400,
        )
    if agenda_id:
        agenda = agenda_loader.get_agenda(int(agenda_id))
        if agenda is None:
            return jsonify({"error": "agenda_not_found", "agenda_id": agenda_id}), 404
    else:
        agenda = agenda_loader.get_active_agenda()
        if agenda is None:
            return jsonify({"error": "no_active_agenda"}), 404

    selection = agenda_selector.select_and_persist(agenda)
    dispatch_result = None
    dispatch_succeeded = None
    if dispatch_mode != "none" and selection.selected_insight_id:
        try:
            dispatch_result = agenda_orchestrator.dispatch_selection(
                selection.selection_id, mode=dispatch_mode
            )
            dispatch_succeeded = True
        except Exception as e:  # noqa: BLE001
            dispatch_result = {"error": str(e), "error_type": type(e).__name__}
            dispatch_succeeded = False
    sel_row = agenda_selector.get_selection(selection.selection_id)
    return jsonify({
        "selection": sel_row,
        "dispatch": dispatch_result,
        "dispatch_succeeded": dispatch_succeeded,
        "agenda": _agenda_to_dict(agenda),
    }), 201


@bp.route("/selection/latest", methods=["GET"])
def latest_selection():
    agenda_id = request.args.get("agenda_id", type=int)
    row = agenda_selector.get_latest_selection(agenda_id)
    if row is None:
        return jsonify({"selection": None}), 404
    return jsonify({"selection": row})


@bp.route("/selection/<int:selection_id>", methods=["GET"])
def get_selection_endpoint(selection_id: int):
    row = agenda_selector.get_selection(selection_id)
    if row is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"selection": row})


# ---------- review + revision plan ----------


@bp.route("/selection/<int:selection_id>/review", methods=["POST"])
def run_review_endpoint(selection_id: int):
    body = request.get_json(silent=True) or {}
    reviewer = body.get("reviewer", "internal_evidence_gate")
    try:
        review = reviewer_adapter.run_review(selection_id, reviewer=reviewer)
    except ContractValidationError as e:
        return jsonify({"error": "invalid_request", "message": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": "not_found", "message": str(e)}), 404
    return jsonify({"review": reviewer_adapter.get_review(review.review_id)}), 201


@bp.route("/selection/<int:selection_id>/plan", methods=["POST"])
def build_plan_endpoint(selection_id: int):
    body = request.get_json(silent=True) or {}
    review_id = body.get("review_id")
    if not review_id:
        latest = reviewer_adapter.get_latest_review(selection_id)
        if not latest:
            return jsonify({"error": "no_review", "message": "Run review first."}), 400
        review_id = latest["id"]
    try:
        plan = revision_planner.build_revision_plan(int(review_id))
    except ContractValidationError as e:
        return jsonify({"error": "invalid_request", "message": str(e)}), 400
    return jsonify({"plan": revision_planner.get_revision_plan(plan.plan_id)}), 201


# ---------- loop inspection (audit view) ----------


def _experiment_run_row(run_id):
    if not run_id:
        return None
    return db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))


def _experimental_claims_summary(run_id):
    if not run_id:
        return {"counts": {}, "claims": []}
    rows = db.fetchall(
        "SELECT id, claim_text, verdict, effect_size, confidence, p_value "
        "FROM experimental_claims WHERE run_id=? ORDER BY id",
        (run_id,),
    )
    counts: dict[str, int] = {}
    for c in rows:
        v = (c.get("verdict") or "unknown").lower()
        counts[v] = counts.get(v, 0) + 1
    return {"counts": counts, "claims": rows}


def _manuscript_run_row(run_id):
    if not run_id:
        return None
    return db.fetchone("SELECT * FROM manuscript_runs WHERE id=?", (run_id,))


def _submission_bundle_row(bundle_id):
    if not bundle_id:
        return None
    return db.fetchone("SELECT * FROM submission_bundles WHERE id=?", (bundle_id,))


def _insight_row(insight_id):
    if not insight_id:
        return None
    return db.fetchone(
        "SELECT id, tier, status, title, problem_statement, formal_structure, "
        "adversarial_score, novelty_status, resource_class, experimentability "
        "FROM deep_insights WHERE id=?",
        (insight_id,),
    )


def _auto_job_row(insight_id):
    if not insight_id:
        return None
    return db.fetchone(
        "SELECT id, deep_insight_id, status, stage, last_note, last_error, "
        "research_workdir, updated_at FROM auto_research_jobs WHERE deep_insight_id=?",
        (insight_id,),
    )


@bp.route("/loop/<int:selection_id>", methods=["GET"])
def loop_inspection(selection_id: int):
    selection = agenda_selector.get_selection(selection_id)
    if selection is None:
        return jsonify({"error": "not_found"}), 404
    agenda = agenda_loader.get_agenda(selection["agenda_id"])
    insight = _insight_row(selection.get("selected_insight_id"))
    auto_job = _auto_job_row(selection.get("selected_insight_id"))
    exp = _experiment_run_row(selection.get("experiment_run_id"))
    evidence = _experimental_claims_summary(selection.get("experiment_run_id"))
    manu = _manuscript_run_row(selection.get("manuscript_run_id"))
    bundle = _submission_bundle_row(selection.get("submission_bundle_id"))
    review = reviewer_adapter.get_latest_review(selection_id)
    plan = revision_planner.get_latest_plan_for_selection(selection_id)

    snapshot = LoopInspectionSnapshot(
        selection=selection,
        agenda=_agenda_to_dict(agenda) or {},
        insight=insight or {},
        auto_research_job=auto_job or {},
        experiment_run=exp or {},
        evidence_gate=evidence,
        manuscript_run=manu or {},
        submission_bundle=bundle or {},
        review=review or {},
        revision_plan=plan or {},
    )
    return jsonify({"loop": snapshot.to_dict()})


def register(app):
    app.register_blueprint(bp)
