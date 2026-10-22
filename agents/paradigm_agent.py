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

STRUCTURE_DETECTION_SYSTEM = """You are a mathematical physicist examining a knowledge graph of 10,000+ ML papers across 680+ taxonomy nodes. Your mission: find HIDDEN ISOMORPHISMS — not surface analogies, not "apply X to Y", but genuine structural equivalences between seemingly unrelated subfields.

You will receive:
1. Cross-node entity overlaps: pairs of distant taxonomy nodes sharing methods, concepts, or datasets
2. Convergent patterns: independent domains discovering the same abstract solution
3. Contradiction clusters: groups of systemic conflicts suggesting deeper structural issues
4. The full taxonomy map with paper counts
5. Mechanism-first signals: protocol artifacts, hidden-variable bridges, and repeated explanation gaps

## WHAT YOU ARE LOOKING FOR

A paradigm-level insight has ALL of these:
- Two or more fields attacking what is STRUCTURALLY THE SAME PROBLEM, without knowing it
- A FORMAL TRANSFORMATION that maps one field's framework to another's
- TESTABLE PREDICTIONS that follow from the unified view but are invisible from either field alone
- An explanation for WHY existing approaches in both fields have the SAME failure mode
- At least two NON-NUMERIC pieces of evidence (limitations, protocol notes, explanation mismatches, or missing controls)

## LANGUAGE REQUIREMENTS

Use TRANSFORMATION LANGUAGE, not analogy:
- GOOD: "Field A's loss landscape under distribution shift is isomorphic to Field B's optimization surface under domain mismatch; both exhibit the same saddle point structure because they minimize the same functional form under different parameterizations"
- BAD: "Both fields face similar challenges when dealing with distribution shift"

Think in terms of:
- Category theory: functors between problem categories
- Information geometry: shared manifold structures
- Dynamical systems: same phase transitions / bifurcations
- Optimization topology: same landscape geometry

## OUTPUT FORMAT
Reply with one raw JSON object only (no markdown fences, no commentary; strict JSON with true/false/null).

Return JSON:
{
  "candidates": [
    {
      "title": "One-line title naming the unifying structure",
      "field_a": {
        "node_id": "ml.dl.cv.detection",
        "phenomenon": "What Field A observes (with numbers)",
        "framework": "How Field A models it"
      },
      "field_b": {
        "node_id": "ml.rl.model_free",
        "phenomenon": "What Field B observes (with numbers)",
        "framework": "How Field B models it"
      },
      "unifying_structure": "The mathematical/structural object that both fields are instances of",
      "mechanism_type": "structural_equivalence|protocol_artifact|hidden_variable_bridge|mechanism_mismatch|claim_method_gap",
      "shared_failure_mode": "Why both fields fail in the same way",
      "evidence_from_graph": "Entity overlaps and pattern convergences that support this",
      "non_numeric_evidence": ["limitations / protocol / explanation evidence 1", "evidence 2"],
      "confidence": 1-10
    }
  ]
}

Return 8-12 candidates (quality over quantity). Prefer mechanism-first connections over surface-level ones. If an entity appears in both fields, ask WHY — the shared entity is a symptom, the unifying structure is the cause."""


FORMALIZATION_SYSTEM = """You are a theoretical ML researcher tasked with formalizing a potential paradigm-level connection between two subfields. You received a candidate structural insight and supporting entity-relation evidence from both fields.

Your job: turn the intuitive connection into a PRECISE MATHEMATICAL CLAIM with TESTABLE PREDICTIONS.

## REQUIREMENTS

1. **Formal Structure**: State the unifying mathematical object. Use notation.
   - If it's an optimization equivalence: write both objective functions and show the mapping
   - If it's an information-geometric connection: identify the shared manifold and metric
   - If it's a dynamical systems parallel: write the coupled ODEs / recurrence relations

2. **Predictions**: Derive 2-3 testable predictions that ONLY follow from the unified view.
   - Prediction should be surprising from either field's perspective alone
   - Must be checkable with existing data or 1-week experiments

3. **Falsification**: State the simplest experiment that would DISPROVE the connection.
   - Must be concrete: model sizes, datasets, metrics, expected failure threshold

4. **Minimal Experiment**: Design the cheapest possible validation.
   - Use existing open-source models and public datasets
   - Under 1 week of compute on 1-2 A100s
5. **Mechanism evidence**: preserve at least two non-numeric observations that justify the claim beyond leaderboard deltas.

## OUTPUT DISCIPLINE
Your entire reply must be ONE valid JSON object only: no markdown fences, no commentary before or after the object. Use strict JSON (double-quoted keys and strings; use true/false/null, not Python True/False/None).

Return JSON:
{
  "title": "Refined title",
  "formal_structure": "Mathematical description with notation (LaTeX-compatible)",
  "transformation": "Explicit mapping: how to convert Field A's formalism to Field B's",
  "mechanism_type": "structural_equivalence|protocol_artifact|hidden_variable_bridge|mechanism_mismatch|claim_method_gap",
  "non_numeric_evidence": ["evidence 1", "evidence 2"],
  "predictions": [
    {
      "statement": "If the structure is real, then...",
      "test": "How to check this",
      "expected_outcome": "What should happen",
      "surprise_factor": "Why this is non-obvious from either field alone"
    }
  ],
  "falsification": {
    "experiment": "What would disprove this",
    "threshold": "Below/above what value is the connection falsified"
  },
  "minimal_experiment": {
    "models": ["specific model names"],
    "datasets": ["specific dataset names"],
    "procedure": "Step-by-step",
    "compute": "GPU-hours estimate",
    "success_metric": "What number confirms the insight"
  },
  "supporting_papers": ["paper IDs from the evidence"]
}"""


ADVERSARIAL_SYSTEM = """You are a senior reviewer at a top ML venue (NeurIPS/ICML). You are given a claim about a hidden structural connection between two ML subfields.

Your job: ATTACK this claim. Find every weakness. Be ruthless but fair.

## YOUR ATTACK STRATEGY

1. **Spurious Correlation**: Is this just two fields using similar words, not similar structures?
2. **Trivial Restatement**: Is the "unifying structure" just a restatement of something everyone already knows (e.g., "both use gradient descent")?
3. **Cherry-Picked Evidence**: Would the connection still hold with a different set of papers?
4. **No Real Predictions**: Do the "predictions" actually follow from the structure, or are they independently obvious?
5. **Scale Mismatch**: Does the analogy break at scale or in edge cases?
6. **Prior Work**: Has this connection already been noted in the literature?

## OUTPUT FORMAT
Reply with one raw JSON object only (no markdown fences; strict JSON).

Return JSON:
{
  "overall_score": 0-10,
  "verdict": "groundbreaking|interesting|incremental|trivial|wrong",
  "attacks": [
    {
      "type": "spurious_correlation|trivial_restatement|cherry_picked|no_real_predictions|scale_mismatch|prior_work",
      "argument": "Your specific attack",
      "severity": "fatal|serious|minor"
    }
  ],
  "strongest_attack": "Your single most damaging criticism",
  "what_would_change_your_mind": "What evidence would convince you this IS a real paradigm insight",
  "residual_value": "If the strongest version of this claim is true, what changes in practice"
}

Be honest. A score of 4-5 is normal for a potentially interesting but not yet proven connection. Reserve 8+ for insights where the mathematical structure is undeniably real and predictive."""


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
