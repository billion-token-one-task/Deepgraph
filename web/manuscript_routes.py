"""Flask Blueprint for manuscript routing + format linting (issue #15 / D4).

Endpoints mounted under ``/api/manuscript``:

    GET    /api/manuscript/venues                       list registered adapters
    POST   /api/manuscript/route                        score venues for a state dict (preview, no persist)
    POST   /api/manuscript/route/<selection_id>         route_and_persist for an agenda selection
    GET    /api/manuscript/route/<selection_id>         latest persisted routing for a selection
    POST   /api/manuscript/lint                         lint a source against a template_id (preview)
    POST   /api/manuscript/lint/<selection_id>          lint + persist tied to a selection
    GET    /api/manuscript/lint_run/<run_id>            fetch a persisted lint run

This is the API surface the Manuscript Routing dashboard card consumes; it
glues together D1's TemplateAdapter, D2's 4 venue adapters, and D3's
FormatLinter + LLM tiebreaker into one HTTP-level contract.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from agents import format_linter, venue_router
from agents.manuscript_templates import get_adapter, list_adapters


bp = Blueprint("manuscript_routes", __name__, url_prefix="/api/manuscript")


# ---------- venues ----------


@bp.route("/venues", methods=["GET"])
def get_venues():
    """Return the full registry of adapters with venue contract metadata."""
    adapters = []
    for tid in list_adapters():
        a = get_adapter(tid)
        adapters.append({
            "template_id": a.template_id,
            "venue_label": a.venue_label,
            "column_layout": a.column_layout,
            "bibstyle_name": a.bibstyle_name,
            "max_pages": a.max_pages,
        })
    return jsonify({"venues": adapters})


# ---------- routing ----------


def _coerce_state(body):
    """Extract a state dict from POST body; tolerate flat-or-nested shape."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    if "state" in body and isinstance(body["state"], dict):
        return body["state"]
    return body


@bp.route("/route", methods=["POST"])
def preview_route():
    """Score venues for a caller-supplied state. No DB writes.

    Optional ``include_tiebreak`` flag in the body runs ``needs_tiebreak`` and
    ``tiebreak_with_llm`` (deterministic fallback unless the caller wires its
    own LLM, which the HTTP surface does not allow today).
    """
    body = request.get_json(silent=True) or {}
    try:
        state = _coerce_state(body)
    except ValueError as e:
        return jsonify({"error": "invalid_state", "message": str(e)}), 400
    include_tiebreak = bool(body.get("include_tiebreak", False))
    try:
        venues = venue_router.load_venue_config()
        result = venue_router.evaluate_venues(state, venues)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "route_failed", "message": str(e)}), 500
    selected = result["selected"]
    payload = {
        "selected": (
            {
                "template_id": selected["venue"].template_id,
                "score": selected["breakdown"]["score"],
                "matched_keywords": selected["breakdown"].get("matched_keywords") or [],
            }
            if selected
            else None
        ),
        "rejected": result["rejected"],
        "all_scored": [
            {
                "template_id": s["venue"].template_id,
                "score": s["breakdown"]["score"],
                "blocked": s["breakdown"]["blocked"],
                "matched_keywords": s["breakdown"].get("matched_keywords") or [],
            }
            for s in result["all_scored"]
        ],
    }
    if include_tiebreak:
        payload["needs_tiebreak"] = venue_router.needs_tiebreak(result["all_scored"])
        payload["tiebreak"] = venue_router.tiebreak_with_llm(state, result["all_scored"])
    return jsonify(payload)


@bp.route("/route/<int:selection_id>", methods=["POST"])
def persist_route(selection_id: int):
    """Persist a routing decision tied to an existing agenda selection_id."""
    body = request.get_json(silent=True) or {}
    try:
        state = _coerce_state(body)
    except ValueError as e:
        return jsonify({"error": "invalid_state", "message": str(e)}), 400
    try:
        out = venue_router.route_and_persist(selection_id, state)
    except RuntimeError as e:
        return jsonify({"error": "route_failed", "message": str(e)}), 500
    return jsonify({"routing": out}), 201


@bp.route("/route/<int:selection_id>", methods=["GET"])
def get_route(selection_id: int):
    row = venue_router.get_routing(selection_id)
    if row is None:
        return jsonify({"routing": None}), 404
    return jsonify({"routing": row})


# ---------- lint ----------


def _resolve_adapter_or_400(template_id):
    if not template_id:
        return None, (jsonify({"error": "missing_template_id"}), 400)
    if template_id not in list_adapters():
        return None, (
            jsonify({
                "error": "unknown_template_id",
                "template_id": template_id,
                "registered": sorted(list_adapters()),
            }),
            404,
        )
    return get_adapter(template_id), None


@bp.route("/lint", methods=["POST"])
def preview_lint():
    """Run FormatLinter on a caller-supplied source. No DB writes.

    Request body::

        {
            "template_id": "iclr2026",
            "source": "<raw LaTeX>",
            "page_count": 8,           # optional
            "normalize": true          # optional; default true
        }
    """
    body = request.get_json(silent=True) or {}
    template_id = body.get("template_id")
    source = body.get("source") or ""
    page_count = body.get("page_count")
    normalize = body.get("normalize", True)
    adapter, err = _resolve_adapter_or_400(template_id)
    if err is not None:
        return err
    if not isinstance(source, str) or not source.strip():
        return jsonify({"error": "empty_source"}), 400
    if normalize:
        source = adapter.normalize_source(source)
    try:
        result = format_linter.lint_manuscript(source, adapter, page_count=page_count)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "lint_failed", "message": str(e)}), 500
    return jsonify({"lint": result})


@bp.route("/lint/<int:selection_id>", methods=["POST"])
def persist_lint(selection_id: int):
    """Run + persist a lint run tied to an agenda selection_id."""
    body = request.get_json(silent=True) or {}
    template_id = body.get("template_id")
    source = body.get("source") or ""
    page_count = body.get("page_count")
    normalize = body.get("normalize", True)
    adapter, err = _resolve_adapter_or_400(template_id)
    if err is not None:
        return err
    if not isinstance(source, str) or not source.strip():
        return jsonify({"error": "empty_source"}), 400
    if normalize:
        source = adapter.normalize_source(source)
    result = format_linter.lint_manuscript(source, adapter, page_count=page_count)
    run_id = format_linter.persist_lint_run(
        selection_id=selection_id, adapter=adapter, lint_result=result
    )
    return jsonify({"lint": result, "run_id": run_id}), 201


@bp.route("/lint_run/<int:run_id>", methods=["GET"])
def get_lint_run_endpoint(run_id: int):
    row = format_linter.get_lint_run(run_id)
    if row is None:
        return jsonify({"lint_run": None}), 404
    return jsonify({"lint_run": row})


def register(app):
    app.register_blueprint(bp)
