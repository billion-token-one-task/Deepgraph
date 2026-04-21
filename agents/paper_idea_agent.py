"""Tier 2 Paper Idea Agent: generate directly executable top-venue paper ideas.

Not brainstorming — concrete paper-ready research with genuine technical novelty.
The bar: a senior researcher reads it and says "this is a real paper, let me implement it."

3-stage LLM pipeline:
  Call 1: Problem Sharpening — formal problem definition + identify what causes failure
  Call 2: Method Invention — design a NEW algorithm/loss/architecture (not "apply A to B")
  Call 3: Experimental Design — complete plan with baselines, datasets, ablations
"""
import json
from agents.discovery_metadata import build_evidence_packet, enrich_deep_insight
from agents.insight_validation import get_evosci_input_issue
from agents.llm_client import call_llm_json, is_llm_auth_error, is_llm_provider_unavailable_error
from agents.signal_harvester import get_tier2_signals
from db import database as db


PROBLEM_SHARPENING_SYSTEM = """You are a senior ML researcher identifying SHARP, FORMAL research problems from evidence of contradictions, performance plateaus, recurring limitations, protocol artifacts, and explanation gaps across thousands of papers.

You will receive:
1. Contradiction clusters (groups of papers disagreeing on comparable setups)
2. Performance plateaus (subfields where top methods have converged within ~1-3%)
3. Recurring limitation clusters (3+ papers in the same node sharing the same limitation)
4. High-scoring insights from prior analysis that lack concrete methods
5. Mechanism-first signals such as protocol artifacts, hidden-variable bridges, and claim-method gaps

## YOUR JOB

For each signal source, extract a FORMAL problem statement:
- State the problem as an optimization / learning problem
- Identify WHAT PROPERTY of current methods causes the failure
- Name the DESIDERATUM: what would a solution need to guarantee

## WHAT MAKES A GOOD PROBLEM

- SPECIFIC: "Cross-domain feature alignment fails because marginal matching ignores conditional structure" not "transfer learning is hard"
- FORMAL: Can be written as minimize/maximize/guarantee over defined quantities
- GROUNDED: Tied to specific numbers from specific papers
- ACTIONABLE: Clear what a solution would look like (even if you don't design it here)
- NOT PURELY NUMERIC: every accepted problem must cite at least two non-numeric observations

Output: one raw JSON object only (no markdown fences; strict JSON).

Return JSON:
{
  "problems": [
    {
      "title": "Problem title with key numbers",
      "source_type": "contradiction|plateau|limitation|insight",
      "source_evidence": "Specific numbers and paper IDs",
      "formal_statement": "Minimize/maximize formulation or formal desideratum",
      "current_failure_mode": "What property of current methods causes this (be mechanistic)",
      "desideratum": "What a solution must guarantee",
      "mechanism_type": "protocol_artifact|mechanism_mismatch|negative_space_gap|hidden_variable_bridge|claim_method_gap|plateau",
      "non_numeric_evidence": ["limitations / protocol / explanation evidence 1", "evidence 2"],
      "difficulty": "hard|medium",
      "impact_scope": "How many papers/methods this affects",
      "related_node_ids": ["ml.dl.cv.detection", ...]
    }
  ]
}

Return 6-12 problems. Quality over quantity. A problem without specific numbers is NOT a problem, and a problem with only numbers but no mechanism evidence is also NOT a problem."""


METHOD_INVENTION_SYSTEM = """You are a methods researcher. Given a formal problem statement with specific failure modes, you must design a GENUINELY NEW method. Not "apply existing method X" — invent something new.

## CRITICAL RULES

1. DO NOT suggest "applying [known technique] to [domain]". That is incremental.
2. Your method must have a NAME (be creative but clear)
3. Your method must have a MATHEMATICAL DEFINITION
4. Your method must address the SPECIFIC failure mode identified in the problem
5. State explicitly what mechanism the method repairs and what falsification result would kill the idea

## METHOD TYPES (choose one or combine):

### NEW LOSS FUNCTION
- Define L(θ; x, y) mathematically
- State gradient properties (smooth? convex in what regime? bounded?)
- Show how it differs from standard losses for this problem
- Key hyperparameters and their effect

### NEW ARCHITECTURE COMPONENT
- Define the computation graph (input → transformations → output)
- State complexity: O(?) time, O(?) memory
- Show the inductive bias it introduces and why it helps
- How it composes with existing architectures

### NEW TRAINING PROCEDURE
- Pseudocode (numbered steps, clear loop structure)
- Convergence properties or training stability argument
- Interaction with existing optimizers (SGD, Adam)
- When to use it vs. standard training

### NEW THEORETICAL FRAMEWORK
- Define the mathematical formalism (spaces, mappings, measures)
- State the key theorem or proposition (even if unproven, state the conjecture)
- Show what it explains that current frameworks cannot
- Practical implications

## OUTPUT FORMAT
Reply with one raw JSON object only (no markdown code fences, no prose outside JSON; strict JSON with true/false/null).

Return JSON:
{
  "method": {
    "name": "Creative but descriptive name",
    "type": "loss_function|architecture|training_procedure|framework|hybrid",
    "one_line": "One sentence: what it does and why it works",
    "definition": "Full mathematical definition (use LaTeX-compatible notation)",
    "pseudocode": "If applicable, numbered steps",
    "complexity": {"time": "O(?)", "memory": "O(?)"},
    "key_properties": [
      "Property 1: why this addresses the failure mode",
      "Property 2: what guarantee it provides"
    ],
    "hyperparameters": [
      {"name": "param_name", "role": "what it controls", "default": "suggested value", "sensitivity": "low|medium|high"}
    ],
    "why_novel": "How this differs from the 3 closest existing methods",
    "limitations": "Honest assessment of where this might fail",
    "mechanism_repair": "What hidden failure mode or protocol defect this method directly fixes",
    "falsification_hook": "The cleanest result that would directly undermine the method"
  }
}

Be bold but rigorous. A novel loss function that provably addresses the failure mode is better than a complex system that might work."""


EXPERIMENT_DESIGN_SYSTEM = """You are designing a COMPLETE experimental plan for a proposed ML method. The plan must be detailed enough that a PhD student can execute it in 4-6 weeks.

You will receive the problem statement and proposed method.

## REQUIREMENTS

1. **Baselines**: Use SPECIFIC model names with sizes and checkpoints
   - At least 3 baselines: (a) vanilla baseline, (b) strongest existing approach, (c) ablation of your method
   - Include paper IDs where these baselines were reported

2. **Datasets**: Use SPECIFIC dataset names with splits
   - At least 2 datasets: one standard benchmark, one stress test
   - Specify train/val/test splits and any preprocessing

3. **Metrics**: Use STANDARD metrics for the field
   - Primary metric (what you optimize for)
   - Secondary metrics (what you also report)
   - Significance testing: paired bootstrap or Wilcoxon

4. **Ablations**: At least 3 ablation experiments
   - Each ablation removes ONE component to isolate its contribution
   - Name each ablation clearly

5. **Expected Results**: Be quantitative
   - Estimate improvement range over strongest baseline
   - State what result would be DISAPPOINTING vs EXCITING

6. **Compute Budget**: Be realistic
   - GPU type and count
   - Training time per experiment
   - Total GPU-hours for all experiments including ablations

7. **Risk Analysis**: What could go wrong
   - Technical risks and mitigation
   - What's plan B if the primary method doesn't work

Output: one raw JSON object only (no markdown fences; strict JSON).

Return JSON:
{
  "paper_title": "Suggested paper title (concise, descriptive)",
  "target_venue": "NeurIPS|ICML|ICLR|ACL|CVPR|specific workshop",
  "baselines": [
    {
      "name": "Method name",
      "model": "Specific model (e.g., Llama-3-8B, ViT-L/14)",
      "source_paper": "paper ID if known",
      "expected_performance": "Estimated metric value"
    }
  ],
  "datasets": [
    {
      "name": "Dataset name",
      "split": "train/val/test sizes",
      "why": "Why this dataset tests the hypothesis"
    }
  ],
  "metrics": {
    "primary": "metric name and why",
    "secondary": ["other metrics"],
    "significance": "testing method"
  },
  "ablations": [
    {
      "name": "Ablation name",
      "removes": "What component is removed",
      "expected_effect": "What should happen and why"
    }
  ],
  "expected_results": {
    "exciting": "What result would be a strong contribution",
    "solid": "What result would be a clear accept",
    "disappointing": "What result would mean the idea doesn't work"
  },
  "compute_budget": {
    "gpu_type": "A100-80GB",
    "experiments": "Number of runs",
    "hours_per_run": "Estimate",
    "total_gpu_hours": "Estimate",
    "estimated_cost": "$X at cloud rates"
  },
  "risks": [
    {
      "risk": "What could go wrong",
      "likelihood": "low|medium|high",
      "mitigation": "What to do about it"
    }
  ],
  "paper_outline": {
    "abstract_sketch": "2-3 sentence abstract draft",
    "contributions": ["Contribution 1", "Contribution 2", "Contribution 3"],
    "related_work_sections": ["Section 1 title", "Section 2 title"]
  },
  "submission_keywords": ["keyword 1", "keyword 2"]
}"""


def _build_problem_prompt(signals: dict) -> str:
    """Build evidence prompt for Call 1 (Problem Sharpening)."""
    sections = ["# EVIDENCE FROM 10,000+ ML PAPERS\n"]

    # Contradiction clusters
    if signals["contradiction_clusters"]:
        sections.append("## CONTRADICTION CLUSTERS")
        sections.append("(Groups of papers disagreeing on comparable setups)\n")
        for cl in signals["contradiction_clusters"]:
            entities = json.loads(cl["shared_entities"]) if cl["shared_entities"] else []
            nodes = json.loads(cl["node_ids"]) if cl["node_ids"] else []
            contra_ids = json.loads(cl["contradiction_ids"]) if cl["contradiction_ids"] else []

            sections.append(f"### Cluster: {cl['theme']} ({cl['cluster_size']} contradictions)")
            sections.append(f"Nodes: {', '.join(nodes[:5])}")
            sections.append(f"Entities: {', '.join(entities[:8])}")

            for cid in contra_ids[:3]:
                contra = db.fetchone("""
                    SELECT c.description, c.hypothesis,
                           ca.method_name, ca.metric_name, ca.metric_value, ca.paper_id as pa,
                           cb.method_name as method_b, cb.metric_value as value_b, cb.paper_id as pb
                    FROM contradictions c
                    JOIN claims ca ON c.claim_a_id = ca.id
                    JOIN claims cb ON c.claim_b_id = cb.id
                    WHERE c.id = ?
                """, (cid,))
                if contra:
                    sections.append(f"  - {contra['description'][:200]}")
                    if contra["method_name"] and contra["metric_value"]:
                        sections.append(
                            f"    {contra['pa']}: {contra['method_name']} = {contra['metric_value']} "
                            f"vs {contra['pb']}: {contra.get('method_b', '?')} = {contra.get('value_b', '?')}")
            sections.append("")

    # Performance plateaus
    if signals["performance_plateaus"]:
        sections.append("\n## PERFORMANCE PLATEAUS")
        sections.append("(Subfields where top methods have converged)\n")
        for pl in signals["performance_plateaus"]:
            top = json.loads(pl["top_methods"]) if pl["top_methods"] else []
            sections.append(
                f"- **{pl['node_id']}** on {pl['dataset_name']} [{pl['metric_name']}]: "
                f"spread={pl['spread_pct']:.2f}% across {pl['method_count']} methods")
            for m in top[:4]:
                sections.append(f"    {m['method']}: {m['value']}")
            sections.append("")

    # Limitation clusters
    if signals["limitation_clusters"]:
        sections.append("\n## RECURRING LIMITATIONS")
        sections.append("(Same limitation appears across 3+ papers in a node)\n")
        for lc in signals["limitation_clusters"]:
            paper_ids = lc["paper_ids"].split(",")[:5] if lc.get("paper_ids") else []
            sections.append(f"- **{lc['node_id']}** ({lc['lim_count']} papers with limitations)")
            for pid in paper_ids[:3]:
                pi = db.fetchone(
                    "SELECT limitations FROM paper_insights WHERE paper_id=?", (pid.strip(),))
                if pi and pi["limitations"]:
                    try:
                        lims = json.loads(pi["limitations"])
                        for lim in lims[:2]:
                            if isinstance(lim, str) and len(lim) > 15:
                                sections.append(f"    [{pid.strip()}] {lim[:150]}")
                    except (json.JSONDecodeError, TypeError):
                        pass
            sections.append("")

    # High-potential existing insights
    if signals["high_potential_insights"]:
        sections.append("\n## HIGH-SCORING PRIOR INSIGHTS (need method innovation)")
        for ins in signals["high_potential_insights"][:5]:
            label = ins.get("insight_type") or ins.get("mechanism_type") or "insight"
            sections.append(f"- [{label}] {ins['title']}")
            hypothesis = ins.get("hypothesis") or ins.get("evidence_summary") or ""
            sections.append(f"  Hypothesis: {hypothesis[:200]}")
            score = ins.get("paradigm_score", ins.get("adversarial_score", 0))
            sections.append(f"  Prior score: {score}")
            sections.append("")

    for key, title in [
        ("mechanism_mismatches", "MECHANISM MISMATCHES"),
        ("protocol_artifacts", "PROTOCOL ARTIFACTS"),
        ("negative_space_gaps", "NEGATIVE SPACE GAPS"),
        ("hidden_variable_bridges", "HIDDEN VARIABLE BRIDGES"),
        ("claim_method_gaps", "CLAIM-METHOD GAPS"),
    ]:
        rows = signals.get(key) or []
        if not rows:
            continue
        sections.append(f"\n## {title}")
        for row in rows[:6]:
            sections.append(f"- {json.dumps(row, ensure_ascii=True, default=str)[:260]}")

    return "\n".join(sections)


def _build_method_prompt(problem: dict) -> str:
    """Build prompt for Call 2 (Method Invention)."""
    return f"""# RESEARCH PROBLEM

## Title: {problem['title']}

## Source: {problem['source_type']}
{problem['source_evidence']}

## Formal Statement
{problem['formal_statement']}

## Current Failure Mode
{problem['current_failure_mode']}

## Desideratum
{problem['desideratum']}

## Impact Scope
{problem['impact_scope']}

## Related Areas: {', '.join(problem.get('related_node_ids', []))}

Design a NEW method that addresses this specific failure mode.
The method must be technically novel — not "apply [existing technique] to [this domain]"."""


def _build_experiment_prompt(problem: dict, method: dict) -> str:
    """Build prompt for Call 3 (Experimental Design)."""
    return f"""# PROPOSED RESEARCH

## Problem
Title: {problem['title']}
Formal Statement: {problem['formal_statement']}
Failure Mode: {problem['current_failure_mode']}

## Proposed Method: {method.get('name', 'Unnamed')}
Type: {method.get('type', '?')}
Summary: {method.get('one_line', '')}
Definition: {method.get('definition', '')[:500]}
Properties: {json.dumps(method.get('key_properties', []))}
Limitations: {method.get('limitations', '')}

## Related Areas: {', '.join(problem.get('related_node_ids', []))}

Design a complete experimental plan for validating this method.
Be specific: exact model names, dataset names, metric names, compute estimates."""


def _llm_temporarily_unavailable(exc: Exception) -> bool:
    return is_llm_auth_error(exc) or is_llm_provider_unavailable_error(exc)


def discover_paper_ideas(
    max_problems: int = 8,
    max_papers: int | None = None,
    *,
    tier2_plateau_limit: int = 20,
    tier2_limitation_nodes: int = 15,
) -> list[dict]:
    """Run the 3-stage paper idea discovery pipeline.

    Returns list of deep_insight dicts ready for storage.
    If max_papers is None, every sharpened problem (up to max_problems) is expanded.
    """
    if max_papers is None:
        max_papers = max_problems

    print(f"[PAPER_IDEA] Starting Tier 2 discovery...", flush=True)
    total_tokens = 0
    total_calls = 0

    # Stage 0: Gather signals
    signals = get_tier2_signals(
        plateau_limit=tier2_plateau_limit,
        limitation_node_limit=tier2_limitation_nodes,
    )
    has_signals = (
        signals["contradiction_clusters"]
        or signals["performance_plateaus"]
        or signals["limitation_clusters"]
        or signals["high_potential_insights"]
        or signals["mechanism_mismatches"]
        or signals["protocol_artifacts"]
        or signals["negative_space_gaps"]
        or signals["hidden_variable_bridges"]
        or signals["claim_method_gaps"]
    )
    if not has_signals:
        print("[PAPER_IDEA] No signals available. Run signal_harvester first.", flush=True)
        return []

    # Stage 1: Problem Sharpening
    print("[PAPER_IDEA] Call 1/3: Problem Sharpening...", flush=True)
    problem_prompt = _build_problem_prompt(signals)
    try:
        result1, tokens1 = call_llm_json(PROBLEM_SHARPENING_SYSTEM, problem_prompt)
        total_tokens += tokens1
        total_calls += 1
    except Exception as e:
        if _llm_temporarily_unavailable(e):
            print(f"[PAPER_IDEA] Problem sharpening skipped: LLM unavailable ({e})", flush=True)
            return []
        print(f"[PAPER_IDEA] Problem sharpening failed: {e}", flush=True)
        return []

    problems = result1.get("problems", [])
    if not problems:
        print("[PAPER_IDEA] No problems extracted", flush=True)
        return []

    problem_budget = min(len(problems), max_problems + max(2, max_papers // 2))
    problems = problems[:problem_budget]
    print(
        f"[PAPER_IDEA] {len(problems)} problems queued to produce up to {max_papers} accepted ideas",
        flush=True,
    )

    # Stage 2 + 3: Method Invention + Experiment Design for top problems
    deep_insights = []
    for i, problem in enumerate(problems):
        if len(deep_insights) >= max_papers:
            break

        title = problem.get("title", f"Problem {i+1}")
        print(f"[PAPER_IDEA] Processing problem {i+1}/{len(problems)}: {title[:80]}", flush=True)

        # Stage 2: Method Invention
        print(f"[PAPER_IDEA] Call 2/3: Inventing method for '{title[:50]}'...", flush=True)
        method_prompt = _build_method_prompt(problem)
        try:
            result2, tokens2 = call_llm_json(METHOD_INVENTION_SYSTEM, method_prompt)
            total_tokens += tokens2
            total_calls += 1
        except Exception as e:
            if _llm_temporarily_unavailable(e):
                print(f"[PAPER_IDEA] Method invention paused: LLM unavailable ({e})", flush=True)
                break
            print(f"[PAPER_IDEA] Method invention failed for '{title[:50]}': {e}", flush=True)
            continue

        method = result2.get("method", {})
        if not method.get("name"):
            print(f"[PAPER_IDEA] No method produced for '{title[:50]}'", flush=True)
            continue

        # Reject "apply X to Y" non-novel methods
        why_novel = method.get("why_novel", "").lower()
        if not why_novel or len(why_novel) < 30:
            print(f"[PAPER_IDEA] Rejected (no novelty argument): {method['name']}", flush=True)
            continue

        # Stage 3: Experimental Design
        print(f"[PAPER_IDEA] Call 3/3: Designing experiments for '{method['name']}'...", flush=True)
        exp_prompt = _build_experiment_prompt(problem, method)
        try:
            result3, tokens3 = call_llm_json(EXPERIMENT_DESIGN_SYSTEM, exp_prompt)
            total_tokens += tokens3
            total_calls += 1
        except Exception as e:
            if _llm_temporarily_unavailable(e):
                print(f"[PAPER_IDEA] Experiment design skipped: LLM unavailable ({e})", flush=True)
                result3 = {}
            else:
                print(f"[PAPER_IDEA] Experiment design failed: {e}", flush=True)
                result3 = {}

        deep_insight = {
            "tier": 2,
            "status": "candidate",
            "title": result3.get("paper_title", f"{method['name']}: {title}"),
            "problem_statement": problem.get("formal_statement", ""),
            "existing_weakness": problem.get("current_failure_mode", ""),
            "proposed_method": json.dumps(method),
            "experimental_plan": json.dumps({
                "baselines": result3.get("baselines", []),
                "datasets": result3.get("datasets", []),
                "metrics": result3.get("metrics", {}),
                "ablations": result3.get("ablations", []),
                "expected_results": result3.get("expected_results", {}),
                "compute_budget": result3.get("compute_budget", {}),
                "risks": result3.get("risks", []),
            }),
            "related_work_positioning": json.dumps(result3.get("paper_outline", {})),
            "source_node_ids": json.dumps(problem.get("related_node_ids", [])),
            "evidence_summary": problem.get("source_evidence", ""),
            "mechanism_type": problem.get("mechanism_type", "mechanism_mismatch"),
            "signal_mix": json.dumps(
                sorted(
                    {
                        problem.get("source_type", "paper_idea"),
                        problem.get("mechanism_type", "mechanism_mismatch"),
                    }
                )
            ),
            "evidence_packet": build_evidence_packet(
                signal_mix=[problem.get("source_type", "paper_idea"), problem.get("mechanism_type", "mechanism_mismatch")],
                evidence_summary=problem.get("source_evidence", ""),
                falsification=method.get("falsification_hook") or {
                    "summary": "See experimental plan for rejection thresholds."
                },
                structural_evidence=[problem.get("formal_statement", "")],
                non_numeric_evidence=problem.get("non_numeric_evidence", []),
            ),
            "novelty_status": "unchecked",
            "generation_tokens": total_tokens,
            "llm_calls": total_calls,
        }

        input_issue = get_evosci_input_issue(deep_insight, mode="verification")
        if input_issue:
            missing = ", ".join(input_issue.get("missing_fields") or [])
            print(
                f"[PAPER_IDEA] Skipped underspecified idea '{title[:60]}' (missing: {missing})",
                flush=True,
            )
            continue

        deep_insights.append(enrich_deep_insight(deep_insight))
        print(f"[PAPER_IDEA] Accepted: {method['name']} — {title[:60]}", flush=True)

    print(f"[PAPER_IDEA] Done: {len(deep_insights)} paper ideas from {len(problems)} problems. "
          f"Tokens: {total_tokens}, LLM calls: {total_calls}", flush=True)
    return deep_insights
