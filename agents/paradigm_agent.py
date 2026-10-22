"""Tier 1 Paradigm Agent: discover hidden unifying structures across distant ML subfields.

Not "apply method A to domain B" — instead: "A and B are the same problem in disguise,
and here's the mathematical structure that proves it."

3-stage LLM pipeline:
  Call 1: Structure Detection (gpt-5.4) — find isomorphisms across fields
  Call 2: Formalization (gpt-5.4) — produce precise mathematical claims + predictions
  Call 3: Adversarial Challenge — skeptic tries to destroy the insight
"""
import json
from agents.discovery_metadata import enrich_deep_insight
from agents.idea_taste import (
    attach_graph_taste_to_insight,
    graph_novelty_gate,
)
from agents.insight_validation import get_evosci_input_issue
from agents.llm_client import (
    is_llm_auth_error,
    is_llm_provider_unavailable_error,
)
from contracts import DeepInsightSpec, normalize_deep_insight_storage
from config import LLM_MODEL, PROMPT_VERSION
from db import database as db
from db.insight_outcomes import new_generation_run_id, record_created


def _llm_temporarily_unavailable(exc: Exception) -> bool:
    return is_llm_auth_error(exc) or is_llm_provider_unavailable_error(exc)


def _jsonify(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def store_deep_insight(insight: dict) -> int:
    """Store a deep insight in the database."""
    try:
        agenda_id = int(insight.get("agenda_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("agenda_id is required when storing a deep insight") from exc
    if agenda_id <= 0:
        raise ValueError("agenda_id must be a positive integer")

    input_issue = get_evosci_input_issue(insight, mode="verification")
    if input_issue:
        missing = ", ".join(input_issue.get("missing_fields") or [])
        print(
            f"[DISCOVERY] Skipping incomplete insight '{insight.get('title', '')[:80]}' before insert (missing: {missing})",
            flush=True,
        )
        return 0

    gate = graph_novelty_gate(insight)
    if gate:
        print(
            f"[DISCOVERY] Skipping low-graph-novelty insight '{insight.get('title', '')[:80]}'",
            flush=True,
        )
        return 0

    normalized = enrich_deep_insight(attach_graph_taste_to_insight(insight))
    spec = DeepInsightSpec.from_raw(normalized)
    insight = {**normalized, **normalize_deep_insight_storage(spec)}
    insight.setdefault("generation_run_id", new_generation_run_id())
    insight.setdefault("prompt_version", insight.get("prompt_version") or PROMPT_VERSION)
    insight.setdefault("model_version", insight.get("model_version") or LLM_MODEL)
    insight.setdefault("outcome", "pending")
    proposal_candidate_id = int(insight.get("proposal_candidate_id") or 0)
    if proposal_candidate_id > 0:
        cursor = db.execute(
            """
            UPDATE deep_insights
            SET tier=?, status=?, title=?, problem_statement=?,
                existing_weakness=?, proposed_method=?, experimental_plan=?,
                related_work_positioning=?, supporting_papers=?,
                source_node_ids=?, evidence_summary=?, signal_mix=?,
                mechanism_type=?, evidence_packet=?, evidence_plan=?,
                experimentability=?, resource_class=?, submission_status=?,
                novelty_status=?, novelty_report=?, generation_tokens=?,
                llm_calls=?, generation_run_id=?, source_signal_ids=?,
                source_signal_refs=?, source_paper_ids=?, prompt_version=?,
                model_version=?, exemplars_used=?, token_cost_usd=?,
                wall_clock_seconds=?, outcome=?, research_problem_id=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=? AND status='proposal_pending'
            """,
            (
                insight.get("tier", 2),
                insight.get("status", "candidate"),
                insight["title"],
                insight.get("problem_statement"),
                insight.get("existing_weakness"),
                _jsonify(insight.get("proposed_method")),
                _jsonify(insight.get("experimental_plan")),
                _jsonify(insight.get("related_work_positioning")),
                _jsonify(insight.get("supporting_papers", [])),
                _jsonify(insight.get("source_node_ids", [])),
                insight.get("evidence_summary"),
                _jsonify(insight.get("signal_mix")),
                insight.get("mechanism_type"),
                _jsonify(insight.get("evidence_packet")),
                _jsonify(insight.get("evidence_plan")),
                insight.get("experimentability"),
                insight.get("resource_class"),
                insight.get("submission_status"),
                insight.get("novelty_status", "unchecked"),
                _jsonify(insight.get("novelty_report")),
                insight.get("generation_tokens", 0),
                insight.get("llm_calls", 0),
                insight.get("generation_run_id"),
                _jsonify(insight.get("source_signal_ids")),
                _jsonify(insight.get("source_signal_refs")),
                _jsonify(insight.get("source_paper_ids")),
                insight.get("prompt_version"),
                insight.get("model_version"),
                _jsonify(insight.get("exemplars_used")),
                insight.get("token_cost_usd"),
                insight.get("wall_clock_seconds"),
                insight.get("outcome", "pending"),
                insight.get("research_problem_id"),
                proposal_candidate_id,
                agenda_id,
            ),
        )
        if int(getattr(cursor, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise RuntimeError(
                "proposal candidate promotion scope/status mismatch"
            )
        rid = proposal_candidate_id
        db.commit()
        db.emit_pipeline_event(
            "deep_insight_promoted",
            {
                "insight_id": rid,
                "agenda_id": agenda_id,
                "research_problem_id": insight.get("research_problem_id"),
                "resource_grant_id": insight.get("resource_grant_id"),
            },
            entity_type="deep_insight",
            entity_id=str(rid),
            dedupe_key=f"deep_insight_promoted:{agenda_id}:{rid}",
        )
        return rid
    rid = db.insert_returning_id(
        """INSERT INTO deep_insights
           (agenda_id, tier, status, title, formal_structure, field_a, field_b,
            transformation, predictions, falsification,
            adversarial_score, adversarial_critique,
            problem_statement, existing_weakness, proposed_method,
            experimental_plan, related_work_positioning,
            supporting_papers, source_node_ids, evidence_summary,
            signal_mix, mechanism_type, evidence_packet, evidence_plan, experimentability,
            resource_class, submission_status,
            novelty_status, novelty_report,
            generation_tokens, llm_calls,
            generation_run_id, source_signal_ids, source_signal_refs, source_paper_ids,
            prompt_version, model_version, exemplars_used,
            token_cost_usd, wall_clock_seconds, outcome, research_problem_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            agenda_id,
            insight.get("tier", 1),
            insight.get("status", "candidate"),
            insight["title"],
            insight.get("formal_structure"),
            _jsonify(insight.get("field_a")),
            _jsonify(insight.get("field_b")),
            insight.get("transformation"),
            _jsonify(insight.get("predictions")),
            _jsonify(insight.get("falsification")),
            insight.get("adversarial_score"),
            _jsonify(insight.get("adversarial_critique")),
            insight.get("problem_statement"),
            insight.get("existing_weakness"),
            _jsonify(insight.get("proposed_method")),
            _jsonify(insight.get("experimental_plan")),
            _jsonify(insight.get("related_work_positioning")),
            _jsonify(insight.get("supporting_papers", [])),
            _jsonify(insight.get("source_node_ids", [])),
            insight.get("evidence_summary"),
            _jsonify(insight.get("signal_mix")),
            insight.get("mechanism_type"),
            _jsonify(insight.get("evidence_packet")),
            _jsonify(insight.get("evidence_plan")),
            insight.get("experimentability"),
            insight.get("resource_class"),
            insight.get("submission_status"),
            insight.get("novelty_status", "unchecked"),
            _jsonify(insight.get("novelty_report")),
            insight.get("generation_tokens", 0),
            insight.get("llm_calls", 0),
            insight.get("generation_run_id"),
            _jsonify(insight.get("source_signal_ids")),
            _jsonify(insight.get("source_signal_refs")),
            _jsonify(insight.get("source_paper_ids")),
            insight.get("prompt_version"),
            insight.get("model_version"),
            _jsonify(insight.get("exemplars_used")),
            insight.get("token_cost_usd"),
            insight.get("wall_clock_seconds"),
            insight.get("outcome", "pending"),
            insight.get("research_problem_id"),
        ),
    )
    db.commit()
    if rid and rid > 0:
        record_created("deep_insights", rid)
        db.emit_pipeline_event(
            "deep_insight_created",
            {
                "insight_id": rid,
                "agenda_id": agenda_id,
                "tier": insight.get("tier", 1),
                "title": insight.get("title"),
                "deep_insight_spec": {**spec.to_dict(), "insight_id": rid},
            },
            entity_type="deep_insight",
            entity_id=str(rid),
        )
    return rid
