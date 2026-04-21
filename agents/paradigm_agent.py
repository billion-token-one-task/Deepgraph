"""Tier 1 Paradigm Agent: discover hidden unifying structures across distant ML subfields.

Not "apply method A to domain B" — instead: "A and B are the same problem in disguise,
and here's the mathematical structure that proves it."

3-stage LLM pipeline:
  Call 1: Structure Detection (gpt-5.4) — find isomorphisms across fields
  Call 2: Formalization (gpt-5.4) — produce precise mathematical claims + predictions
  Call 3: Adversarial Challenge (MiniMax) — skeptic tries to destroy the insight
"""
import json
import time
from agents.discovery_metadata import build_evidence_packet, enrich_deep_insight
from agents.insight_validation import get_evosci_input_issue
from agents.llm_client import (
    call_llm_json,
    call_llm,
    _init_providers,
    _providers,
    is_llm_auth_error,
    is_llm_provider_unavailable_error,
)
from contracts import DeepInsightSpec, normalize_deep_insight_storage
from agents.signal_harvester import get_tier1_signals
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


def _guess_mechanism_type(candidate: dict, formalized: dict) -> str:
    corpus = " ".join(
        str(part or "")
        for part in [
            candidate.get("mechanism_type"),
            candidate.get("shared_failure_mode"),
            candidate.get("evidence_from_graph"),
            formalized.get("mechanism_type"),
            formalized.get("formal_structure"),
        ]
    ).lower()
    if "protocol" in corpus or "annotation" in corpus or "metric" in corpus:
        return "protocol_artifact"
    if "hidden" in corpus or "latent" in corpus:
        return "hidden_variable_bridge"
    if "mismatch" in corpus or "explanation" in corpus:
        return "mechanism_mismatch"
    if "claim" in corpus and "evidence" in corpus:
        return "claim_method_gap"
    return "structural_equivalence"


def _build_structure_prompt(signals: dict) -> str:
    """Build the evidence prompt for Call 1 (Structure Detection)."""
    sections = []

    sections.append("# CROSS-FIELD EVIDENCE FROM 10,000+ ML PAPERS\n")

    # Entity overlaps
    if signals["entity_overlaps"]:
        sections.append("## CROSS-NODE ENTITY OVERLAPS")
        sections.append("(Distant taxonomy nodes sharing specific methods/concepts/datasets)\n")
        for ov in signals["entity_overlaps"][:20]:
            shared = json.loads(ov["shared_entity_ids"]) if ov["shared_entity_ids"] else []
            types = json.loads(ov["shared_entity_types"]) if ov["shared_entity_types"] else {}
            entity_names = [e["name"] for e in shared[:8]]
            sections.append(
                f"- **{ov['node_a_id']}** ↔ **{ov['node_b_id']}** "
                f"(distance={ov['taxonomic_distance']}, shared={ov['shared_entity_count']}, "
                f"score={ov['overlap_score']:.3f})")
            sections.append(f"  Types: {types}")
            sections.append(f"  Shared: {', '.join(entity_names)}")
            sections.append("")

    # Convergent patterns
    if signals["pattern_matches"]:
        sections.append("\n## CONVERGENT PATTERNS")
        sections.append("(Independent domains discovering the same abstract solution)\n")
        for pm in signals["pattern_matches"][:15]:
            sections.append(f"- Similarity={pm['similarity_score']:.3f}")
            sections.append(f"  Pattern A ({pm.get('node_a_id', '?')}): {pm.get('text_a', '')[:150]}")
            sections.append(f"  Pattern B ({pm.get('node_b_id', '?')}): {pm.get('text_b', '')[:150]}")
            sections.append("")

    # Contradiction clusters
    if signals["contradiction_clusters"]:
        sections.append("\n## CONTRADICTION CLUSTERS")
        sections.append("(Groups of related conflicts suggesting systemic issues)\n")
        for cl in signals["contradiction_clusters"]:
            entities = json.loads(cl["shared_entities"]) if cl["shared_entities"] else []
            nodes = json.loads(cl["node_ids"]) if cl["node_ids"] else []
            sections.append(f"- Theme: {cl['theme']} (size={cl['cluster_size']})")
            sections.append(f"  Affected nodes: {', '.join(nodes[:5])}")
            sections.append(f"  Entities: {', '.join(entities[:8])}")
            sections.append("")

    # Taxonomy overview
    if signals["taxonomy_map"]:
        sections.append("\n## TAXONOMY MAP (with paper counts)")
        depth2_nodes = [t for t in signals["taxonomy_map"] if t["depth"] <= 2]
        for t in depth2_nodes:
            indent = "  " * t["depth"]
            sections.append(f"{indent}{t['name']} ({t['id']}) — {t['paper_count']}p")

    for key, title in [
        ("protocol_artifacts", "PROTOCOL ARTIFACTS"),
        ("hidden_variable_bridges", "HIDDEN VARIABLE BRIDGES"),
        ("claim_method_gaps", "CLAIM-METHOD GAPS"),
    ]:
        rows = signals.get(key) or []
        if not rows:
            continue
        sections.append(f"\n## {title}")
        for row in rows[:10]:
            sections.append(f"- {json.dumps(row, ensure_ascii=True)[:220]}")

    return "\n".join(sections)


def _build_formalization_prompt(candidate: dict, signals: dict) -> str:
    """Build the evidence prompt for Call 2 (Formalization)."""
    sections = [f"# CANDIDATE PARADIGM INSIGHT\n"]
    sections.append(f"## Title: {candidate['title']}\n")

    fa = candidate.get("field_a", {})
    fb = candidate.get("field_b", {})

    sections.append(f"### Field A: {fa.get('node_id', '?')}")
    sections.append(f"Phenomenon: {fa.get('phenomenon', '?')}")
    sections.append(f"Framework: {fa.get('framework', '?')}\n")

    sections.append(f"### Field B: {fb.get('node_id', '?')}")
    sections.append(f"Phenomenon: {fb.get('phenomenon', '?')}")
    sections.append(f"Framework: {fb.get('framework', '?')}\n")

    sections.append(f"### Proposed Unifying Structure")
    sections.append(candidate.get("unifying_structure", ""))
    sections.append(f"\n### Shared Failure Mode")
    sections.append(candidate.get("shared_failure_mode", ""))
    sections.append(f"\n### Evidence from Knowledge Graph")
    sections.append(candidate.get("evidence_from_graph", ""))

    # Add entity-relation context from both fields
    for node_id_key, label in [("field_a", "Field A"), ("field_b", "Field B")]:
        field = candidate.get(node_id_key, {})
        nid = field.get("node_id", "")
        if not nid:
            continue
        sections.append(f"\n## Entity-Relation Context for {label} ({nid})")

        graph_summary = db.fetchone(
            "SELECT top_entities, top_relations FROM node_graph_summaries WHERE node_id=?",
            (nid,))
        if graph_summary:
            if graph_summary["top_entities"]:
                sections.append(f"Top entities: {graph_summary['top_entities'][:500]}")
            if graph_summary["top_relations"]:
                sections.append(f"Top relations: {graph_summary['top_relations'][:500]}")

        results = db.fetchall("""
            SELECT method_name, dataset_name, metric_name, metric_value
            FROM results WHERE node_id = ? OR node_id LIKE ? || '.%'
            ORDER BY id DESC LIMIT 15
        """, (nid, nid))
        if results:
            sections.append("Results:")
            for r in results:
                sections.append(f"  {r['method_name']} on {r['dataset_name']} [{r['metric_name']}] = {r['metric_value']}")

    return "\n".join(sections)


def _call_with_provider(system: str, user: str, provider_name: str = None) -> tuple:
    """Call LLM, optionally targeting a specific provider for adversarial diversity."""
    if provider_name:
        _init_providers()
        for p in _providers:
            if p["name"] == provider_name:
                from agents.llm_client import _call_provider, _rate_limiters
                limiter = _rate_limiters.get(p["name"])
                if limiter:
                    limiter.wait()
                text, tokens, _, _ = _call_provider(p, system, user, 16_000)
                import re
                text = text.strip()
                text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    end = len(lines)
                    for i in range(len(lines) - 1, 0, -1):
                        if lines[i].strip() == "```":
                            end = i
                            break
                    text = "\n".join(lines[1:end]).strip()
                try:
                    return json.loads(text), tokens
                except json.JSONDecodeError:
                    for match in re.finditer(r'(\{[\s\S]*\}|\[[\s\S]*\])', text):
                        try:
                            return json.loads(match.group()), tokens
                        except json.JSONDecodeError:
                            continue
                    print(f"[PARADIGM] {provider_name} returned non-JSON ({len(text)} chars), falling back", flush=True)
        # fallback
    return call_llm_json(system, user)


def _llm_temporarily_unavailable(exc: Exception) -> bool:
    return is_llm_auth_error(exc) or is_llm_provider_unavailable_error(exc)


def discover_paradigm_insights(
    max_candidates: int = 5,
    *,
    tier1_top_overlaps: int = 20,
    tier1_top_patterns: int = 15,
) -> list[dict]:
    """Run the 3-stage paradigm discovery pipeline.

    Returns list of deep_insight dicts ready for storage.
    """
    print(f"[PARADIGM] Starting Tier 1 discovery (max {max_candidates} candidates)...", flush=True)
    total_tokens = 0
    total_calls = 0

    # Stage 0: Gather signals
    signals = get_tier1_signals(
        top_overlaps=tier1_top_overlaps, top_patterns=tier1_top_patterns
    )
    if not signals["entity_overlaps"] and not signals["pattern_matches"]:
        print("[PARADIGM] No signals available. Run signal_harvester first.", flush=True)
        return []

    # Stage 1: Structure Detection
    print("[PARADIGM] Call 1/3: Structure Detection...", flush=True)
    structure_prompt = _build_structure_prompt(signals)
    try:
        result1, tokens1 = call_llm_json(STRUCTURE_DETECTION_SYSTEM, structure_prompt)
        total_tokens += tokens1
        total_calls += 1
    except Exception as e:
        if _llm_temporarily_unavailable(e):
            print(f"[PARADIGM] Structure detection skipped: LLM unavailable ({e})", flush=True)
            return []
        print(f"[PARADIGM] Structure detection failed: {e}", flush=True)
        return []

    candidates = result1.get("candidates", [])
    if not candidates:
        print("[PARADIGM] No candidates from structure detection", flush=True)
        return []

    candidates.sort(key=lambda c: c.get("confidence", 0), reverse=True)
    candidate_budget = min(len(candidates), max_candidates + max(2, max_candidates // 2))
    candidates = candidates[:candidate_budget]
    print(
        f"[PARADIGM] {len(candidates)} candidates queued to produce up to {max_candidates} accepted insights",
        flush=True,
    )

    # Stage 2 + 3: Formalize and Challenge each candidate
    deep_insights = []
    for i, candidate in enumerate(candidates):
        if len(deep_insights) >= max_candidates:
            break
        title = candidate.get("title", f"Candidate {i+1}")
        print(f"[PARADIGM] Processing candidate {i+1}/{len(candidates)}: {title[:80]}", flush=True)

        # Stage 2: Formalization
        print(f"[PARADIGM] Call 2/3: Formalizing '{title[:50]}'...", flush=True)
        formal_prompt = _build_formalization_prompt(candidate, signals)
        try:
            result2, tokens2 = call_llm_json(FORMALIZATION_SYSTEM, formal_prompt)
            total_tokens += tokens2
            total_calls += 1
        except Exception as e:
            if _llm_temporarily_unavailable(e):
                print(f"[PARADIGM] Formalization paused: LLM unavailable ({e})", flush=True)
                break
            print(f"[PARADIGM] Formalization failed for '{title[:50]}': {e}", flush=True)
            continue

        # Stage 3: Adversarial Challenge (use MiniMax for diversity)
        print(f"[PARADIGM] Call 3/3: Adversarial challenge for '{title[:50]}'...", flush=True)
        adversarial_prompt = json.dumps({
            "title": result2.get("title", title),
            "formal_structure": result2.get("formal_structure", ""),
            "transformation": result2.get("transformation", ""),
            "predictions": result2.get("predictions", []),
            "field_a": candidate.get("field_a", {}),
            "field_b": candidate.get("field_b", {}),
        }, indent=2)

        try:
            result3, tokens3 = _call_with_provider(
                ADVERSARIAL_SYSTEM, adversarial_prompt, provider_name="minimax")
            total_tokens += tokens3
            total_calls += 1
        except Exception as e:
            if _llm_temporarily_unavailable(e):
                print(f"[PARADIGM] Adversarial challenge skipped: LLM unavailable ({e})", flush=True)
                result3 = {"overall_score": 5, "verdict": "unknown",
                           "attacks": [], "strongest_attack": "LLM unavailable during challenge"}
            else:
                print(f"[PARADIGM] Adversarial challenge failed: {e}. Using next LLM provider in pool.", flush=True)
                try:
                    result3, tokens3 = call_llm_json(ADVERSARIAL_SYSTEM, adversarial_prompt)
                    total_tokens += tokens3
                    total_calls += 1
                except Exception as e2:
                    if _llm_temporarily_unavailable(e2):
                        print(f"[PARADIGM] Adversarial fallback skipped: LLM unavailable ({e2})", flush=True)
                        result3 = {"overall_score": 5, "verdict": "unknown",
                                   "attacks": [], "strongest_attack": "LLM unavailable during challenge"}
                    else:
                        print(f"[PARADIGM] Adversarial fallback also failed: {e2}", flush=True)
                        result3 = {"overall_score": 5, "verdict": "unknown",
                                   "attacks": [], "strongest_attack": "Could not evaluate"}

        score = result3.get("overall_score", 0)
        verdict = result3.get("verdict", "unknown")
        print(f"[PARADIGM] Adversarial score: {score}/10 ({verdict})", flush=True)

        if score < 4:
            print(f"[PARADIGM] Rejected (score {score} < 4): {title[:60]}", flush=True)
            continue

        deep_insight = {
            "tier": 1,
            "status": "candidate",
            "title": result2.get("title", title),
            "formal_structure": result2.get("formal_structure", ""),
            "field_a": json.dumps(candidate.get("field_a", {})),
            "field_b": json.dumps(candidate.get("field_b", {})),
            "transformation": result2.get("transformation", ""),
            "predictions": json.dumps(result2.get("predictions", [])),
            "falsification": json.dumps(result2.get("falsification", {})),
            "adversarial_score": score,
            "adversarial_critique": json.dumps({
                "verdict": verdict,
                "attacks": result3.get("attacks", []),
                "strongest_attack": result3.get("strongest_attack", ""),
                "what_would_change_mind": result3.get("what_would_change_your_mind", ""),
                "residual_value": result3.get("residual_value", ""),
            }),
            "supporting_papers": json.dumps(result2.get("supporting_papers", [])),
            "source_node_ids": json.dumps([
                candidate.get("field_a", {}).get("node_id", ""),
                candidate.get("field_b", {}).get("node_id", ""),
            ]),
            "evidence_summary": candidate.get("evidence_from_graph", ""),
            "mechanism_type": _guess_mechanism_type(candidate, result2),
            "signal_mix": json.dumps(
                ["entity_overlap", "pattern_match", _guess_mechanism_type(candidate, result2)]
            ),
            "evidence_packet": build_evidence_packet(
                signal_mix=["entity_overlap", "pattern_match", _guess_mechanism_type(candidate, result2)],
                evidence_summary=candidate.get("evidence_from_graph", ""),
                falsification=json.dumps(result2.get("falsification", {})),
                structural_evidence=[candidate.get("evidence_from_graph", "")],
                non_numeric_evidence=result2.get("non_numeric_evidence") or candidate.get("non_numeric_evidence") or [],
            ),
            "novelty_status": "unchecked",
            "generation_tokens": total_tokens,
            "llm_calls": total_calls,
        }

        # Also store minimal experiment info in experimental_plan
        if result2.get("minimal_experiment"):
            deep_insight["experimental_plan"] = json.dumps(result2["minimal_experiment"])

        input_issue = get_evosci_input_issue(deep_insight, mode="verification")
        if input_issue:
            missing = ", ".join(input_issue.get("missing_fields") or [])
            print(
                f"[PARADIGM] Skipped underspecified insight '{title[:60]}' (missing: {missing})",
                flush=True,
            )
            continue

        deep_insights.append(enrich_deep_insight(deep_insight))
        print(f"[PARADIGM] Accepted: {title[:80]} (score={score})", flush=True)

    print(f"[PARADIGM] Done: {len(deep_insights)} insights from {len(candidates)} candidates. "
          f"Tokens: {total_tokens}, LLM calls: {total_calls}", flush=True)
    return deep_insights


def _jsonify(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def store_deep_insight(insight: dict) -> int:
    """Store a deep insight in the database."""
    input_issue = get_evosci_input_issue(insight, mode="verification")
    if input_issue:
        missing = ", ".join(input_issue.get("missing_fields") or [])
        print(
            f"[DISCOVERY] Skipping incomplete insight '{insight.get('title', '')[:80]}' before insert (missing: {missing})",
            flush=True,
        )
        return 0

    normalized = enrich_deep_insight(insight)
    spec = DeepInsightSpec.from_raw(normalized)
    insight = {**normalized, **normalize_deep_insight_storage(spec)}
    insight.setdefault("generation_run_id", new_generation_run_id())
    insight.setdefault("prompt_version", insight.get("prompt_version") or PROMPT_VERSION)
    insight.setdefault("model_version", insight.get("model_version") or LLM_MODEL)
    insight.setdefault("outcome", "pending")
    rid = db.insert_returning_id(
        """INSERT INTO deep_insights
           (tier, status, title, formal_structure, field_a, field_b,
            transformation, predictions, falsification,
            adversarial_score, adversarial_critique,
            problem_statement, existing_weakness, proposed_method,
            experimental_plan, related_work_positioning,
            supporting_papers, source_node_ids, evidence_summary,
            signal_mix, mechanism_type, evidence_packet, evidence_plan, experimentability,
            resource_class, submission_status,
            novelty_status, novelty_report,
            generation_tokens, llm_calls,
            generation_run_id, source_signal_ids, source_paper_ids,
            prompt_version, model_version, exemplars_used,
            token_cost_usd, wall_clock_seconds, outcome)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            insight.get("tier", 1),
            insight.get("status", "candidate"),
            insight["title"],
            insight.get("formal_structure"),
            insight.get("field_a"),
            insight.get("field_b"),
            insight.get("transformation"),
            insight.get("predictions"),
            insight.get("falsification"),
            insight.get("adversarial_score"),
            insight.get("adversarial_critique"),
            insight.get("problem_statement"),
            insight.get("existing_weakness"),
            insight.get("proposed_method"),
            insight.get("experimental_plan"),
            insight.get("related_work_positioning"),
            insight.get("supporting_papers", "[]"),
            insight.get("source_node_ids", "[]"),
            insight.get("evidence_summary"),
            insight.get("signal_mix"),
            insight.get("mechanism_type"),
            insight.get("evidence_packet"),
            insight.get("evidence_plan"),
            insight.get("experimentability"),
            insight.get("resource_class"),
            insight.get("submission_status"),
            insight.get("novelty_status", "unchecked"),
            insight.get("novelty_report"),
            insight.get("generation_tokens", 0),
            insight.get("llm_calls", 0),
            insight.get("generation_run_id"),
            _jsonify(insight.get("source_signal_ids")),
            _jsonify(insight.get("source_paper_ids")),
            insight.get("prompt_version"),
            insight.get("model_version"),
            _jsonify(insight.get("exemplars_used")),
            insight.get("token_cost_usd"),
            insight.get("wall_clock_seconds"),
            insight.get("outcome", "pending"),
        ),
    )
    db.commit()
    if rid and rid > 0:
        record_created("deep_insights", rid)
        db.emit_pipeline_event(
            "deep_insight_created",
            {
                "insight_id": rid,
                "tier": insight.get("tier", 1),
                "title": insight.get("title"),
                "deep_insight_spec": {**spec.to_dict(), "insight_id": rid},
            },
            entity_type="deep_insight",
            entity_id=str(rid),
        )
    return rid
