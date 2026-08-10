"""Tier 2 Paper Idea Agent: generate directly executable top-venue paper ideas.

Not brainstorming — concrete paper-ready research with genuine technical novelty.
The bar: a senior researcher reads it and says "this is a real paper, let me implement it."

3-stage LLM pipeline:
  Call 1: Problem Sharpening — formal problem definition + identify what causes failure
  Call 2: Method Invention — design a NEW algorithm/loss/architecture (not "apply A to B")
  Call 3: Experimental Design — complete plan with baselines, datasets, ablations
"""
import json
import re
from collections import Counter
from dataclasses import asdict
from difflib import SequenceMatcher
from agents.compute_profile import detect_compute_profile
from agents.discovery_metadata import build_evidence_packet, enrich_deep_insight
from agents.idea_taste import (
    attach_graph_taste_to_insight,
    format_frontier_block,
    graph_novelty_gate,
    signal_type_weight,
)
from agents.insight_validation import get_evosci_input_issue
from agents.llm_client import (
    call_llm_json_for_role,
    configured_role_prompt_version,
    is_llm_auth_error,
    is_llm_provider_unavailable_error,
)
from agents.problem_first import (
    discover_research_problems,
    match_problem_to_research_problem,
    select_problem_first_candidates,
)
from agents.paper_title_policy import TITLE_NAMING_STANDARD_TEXT, normalize_paper_title
from agents.signal_harvester import get_solution_signals, get_tier2_signals, signal_refs_from_rows
from agents.tier2_review_refine import review_and_refine_tier2_idea
from config import TIER2_EVOSCI_PREINSERT_REVIEW
from db import database as db

RECENT_TIER2_MEMORY_LIMIT = 120


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
      "central_question": "One crisp question the paper will answer",
      "motivation": "Why this question matters now and what prior papers leave unresolved",
      "result_that_would_change_belief": "The smallest concrete empirical result that would convince a skeptical top-conference reviewer",
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

8. **Problem Awareness**: Make the paper spine explicit
   - What exact problem is the paper answering?
   - What motivates the problem relative to the closest real papers?
   - What method mechanism resolves the failure mode?
   - What result would support or falsify the claim?

9. **Title Naming**: Follow this binding title policy.
""" + TITLE_NAMING_STANDARD_TEXT + """

10. **Execution Requirements**: Declare the cheapest falsification run as a
structured capability contract before any execution grant exists.
   - Use concrete public dataset/model repository IDs and an explicit revision
     or tag; never use a display name as a repository ID.
   - Declare task protocol, semantic dataset field roles, model task/framework,
     metric direction, dependency/network/disk/VRAM needs, seeds/sample cap,
     backend preferences, and required raw artifacts.
   - Do not claim availability. A separate metadata preflight verifies every
     repository, revision, schema, dependency, and resource before grant.

Output: one raw JSON object only (no markdown fences; strict JSON).

Return JSON:
{
  "paper_title": "Suggested paper title using SymbolicName: Descriptive Subtitle or ACRONYM: Expansion Subtitle",
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
  "execution_requirements": {
    "schema_version": "experiment_requirements_v1",
    "task_protocol": "generative_qa|sequence_classification",
    "candidate_hook": "candidate_prompt for generative_qa, candidate_text for sequence_classification",
    "dataset": {
      "repository_id": "public repository id",
      "revision": "immutable commit or explicit tag",
      "config": "dataset config or empty string",
      "split": "evaluation split",
      "field_mapping": {"semantic_role": "actual_column_name"}
    },
    "model": {
      "repository_id": "public repository id",
      "revision": "immutable commit or explicit tag",
      "framework": "transformers|sentence_transformers|another explicit framework",
      "task": "causal_lm|sequence_classification|embedding|another explicit task",
      "min_vram_gb": 0,
      "requires_cuda": false,
      "quantization": "none|4bit|8bit"
    },
    "metric": {
      "name": "machine-computable primary metric",
      "direction": "higher|lower",
      "required_prediction_fields": ["prediction", "target"]
    },
    "dependencies": ["package_name"],
    "network_required": true,
    "min_disk_gb": 1,
    "seeds": [0],
    "sample_cap": 32,
    "artifact_contract": ["final_results", "raw_predictions", "environment_manifest", "dataset_manifest", "model_manifest"],
    "preferred_backends": ["cpu|local_gpu|ssh_gpu|colab_gpu"]
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
  "problem_awareness": {
    "central_question": "What problem does the paper answer?",
    "motivation": "Why the problem matters and why prior methods do not settle it",
    "method_answer": "How the proposed mechanism answers the question",
    "result_claim": "What experiment/result would support the answer",
    "falsification_result": "What concrete result would kill the paper claim"
  },
  "submission_keywords": ["keyword 1", "keyword 2"]
}"""


def _json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [value] if value.strip() else []
        return parsed if isinstance(parsed, list) else []
    return []


def _problem_node_ids(problem: dict) -> list[str]:
    nodes = _json_list(problem.get("related_node_ids"))
    return [str(node).strip() for node in nodes if str(node).strip()]


def _problem_source_refs(problem: dict) -> dict:
    refs = problem.get("source_signal_refs")
    if isinstance(refs, dict):
        return refs
    if isinstance(refs, str):
        try:
            parsed = json.loads(refs)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _problem_ruled_out(problem: dict) -> list[dict]:
    rows = _json_list(problem.get("ruled_out_approaches"))
    return [row for row in rows if isinstance(row, dict)]


def _attach_research_problem_context(problem: dict, research_problems: list[dict], fallback_refs: dict) -> dict:
    enriched = dict(problem)
    matched = match_problem_to_research_problem(problem, research_problems)
    if matched:
        source_ref = matched.get("source_signal_ref") or {}
        enriched["research_problem_id"] = matched.get("id")
        enriched["problem_statement"] = matched.get("problem_statement") or enriched.get("formal_statement") or enriched.get("title")
        enriched["ruled_out_approaches"] = matched.get("ruled_out_approaches") or []
        enriched["source_signal_refs"] = {
            "signals": [source_ref] if source_ref else [],
            "node_ids": matched.get("node_ids") or enriched.get("related_node_ids") or [],
            "paper_ids": matched.get("paper_ids") or [],
        }
        enriched["source_paper_ids"] = matched.get("paper_ids") or []
    else:
        enriched["source_signal_refs"] = fallback_refs
        enriched["source_paper_ids"] = fallback_refs.get("paper_ids", [])
    return enriched


def _recent_tier2_memory(limit: int = RECENT_TIER2_MEMORY_LIMIT) -> list[dict]:
    try:
        rows = db.fetchall(
            "SELECT title, mechanism_type, source_node_ids, proposed_method, "
            "problem_statement, created_at "
            "FROM deep_insights "
            "WHERE tier IN (1, 2) "
            "ORDER BY created_at DESC, id DESC "
            "LIMIT ?",
            (int(limit),),
        )
    except Exception:
        return []

    memory: list[dict] = []
    for row in rows:
        method = {}
        try:
            method = json.loads(row.get("proposed_method") or "{}")
        except (json.JSONDecodeError, TypeError):
            method = {}
        memory.append(
            {
                "title": row.get("title") or "",
                "mechanism_type": row.get("mechanism_type") or "",
                "source_node_ids": _json_list(row.get("source_node_ids")),
                "method_name": method.get("name") or "",
                "problem_statement": row.get("problem_statement") or "",
                "created_at": row.get("created_at") or "",
            }
        )
    return memory


def _recent_idea_memory_block(memory: list[dict]) -> str:
    if not memory:
        return ""
    lines = [
        "## RECENT TIER-1/TIER-2 IDEAS TO AVOID REPEATING",
        "The system has already generated these paper ideas. Prefer different source-node families, mechanism types, datasets, and failure mechanisms unless the evidence is materially stronger.",
    ]
    for item in memory[:20]:
        nodes = ", ".join(str(node) for node in item.get("source_node_ids", [])[:4])
        method = item.get("method_name") or "?"
        title = str(item.get("title") or "")[:180]
        mechanism = item.get("mechanism_type") or "?"
        node_text = nodes or "?"
        lines.append(
            f"- {title} | mechanism={mechanism} | "
            f"method={method[:80]} | nodes={node_text}"
        )
    return "\n".join(lines)


def _text_similarity(a: str, b: str) -> float:
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


_DUPLICATE_STOPWORDS = {
    "the", "and", "for", "are", "with", "from", "into", "over", "under",
    "via", "whose", "that", "this", "same", "common", "shared", "unifies",
    "unify", "based", "model", "models", "benchmark", "benchmarks",
    "evaluation", "reasoning", "agent", "agents", "llm", "vlm", "code",
    "proof", "closed", "open", "loop", "policy", "visual", "scene",
    "graph", "relation", "relations", "method", "paper",
    "selective", "audited", "typed", "evidence", "risk", "protocol",
    "offline", "training", "free", "certified", "residual", "routing",
}


def _token_set(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in _DUPLICATE_STOPWORDS
    }


def _token_jaccard(a: str, b: str) -> float:
    left = _token_set(a)
    right = _token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _node_jaccard(a, b) -> float:
    left = {str(x).strip() for x in _json_list(a) if str(x).strip()}
    right = {str(x).strip() for x in _json_list(b) if str(x).strip()}
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _node_family_prefixes(nodes) -> set[str]:
    prefixes: set[str] = set()
    for raw in _json_list(nodes):
        node = str(raw).strip()
        if not node:
            continue
        parts = [part for part in node.split(".") if part]
        for depth in range(2, min(len(parts), 5) + 1):
            prefixes.add(".".join(parts[:depth]))
    return prefixes


def _node_family_jaccard(a, b) -> float:
    left = _node_family_prefixes(a)
    right = _node_family_prefixes(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _find_existing_tier2_duplicate(
    candidate: dict, *, exclude_id: int | None = None
) -> dict | None:
    """Find a prior idea this candidate duplicates.

    ``exclude_id`` is the candidate's own pre-idea identity row. That
    placeholder is seeded from the research problem, so it carries the same
    source_node_ids and mechanism_type as the idea being realized from it and
    matches itself on every threshold below. Comparing a candidate against its
    own placeholder rejected every proposal the grant had already paid for.
    """

    title = str(candidate.get("title") or "")
    nodes = candidate.get("source_node_ids")
    mechanism = str(candidate.get("mechanism_type") or "")
    skip = int(exclude_id or 0)
    own_problem = int(candidate.get("research_problem_id") or 0)
    try:
        rows = db.fetchall(
            "SELECT id, title, source_node_ids, mechanism_type, status, novelty_status, "
            "outcome, research_problem_id "
            "FROM deep_insights WHERE tier IN (1, 2) ORDER BY id DESC LIMIT 400"
        )
    except Exception:
        return None
    for row in rows:
        if skip and int(row.get("id") or 0) == skip:
            continue
        title_score = _token_jaccard(title, row.get("title") or "")
        node_score = _node_jaccard(nodes, row.get("source_node_ids"))
        family_score = _node_family_jaccard(nodes, row.get("source_node_ids"))
        same_mechanism = mechanism and mechanism == str(row.get("mechanism_type") or "")
        # Two ideas raised from the same research problem inherit that
        # problem's source_node_ids verbatim, so their node overlap is 1.0 by
        # construction and carries no information about whether the ideas are
        # the same. Scoring it anyway meant a problem could yield exactly one
        # idea ever: agenda 11's second idea on problem 8 was rejected as a
        # duplicate of its first at node_overlap 1.0 and title_sim 0.095.
        # Content signals still apply between siblings; provenance does not.
        siblings = bool(own_problem) and int(row.get("research_problem_id") or 0) == own_problem
        too_close = (
            title_score >= 0.32
            or (not siblings and node_score >= 0.50 and same_mechanism)
            or (not siblings and node_score >= 0.62 and title_score >= 0.04)
            or (
                not siblings
                and family_score >= 0.42
                and same_mechanism
                and title_score >= 0.02
            )
            or (same_mechanism and title_score >= 0.18)
        )
        if too_close:
            return {
                "id": row.get("id"),
                "title": row.get("title"),
                "title_similarity": round(title_score, 3),
                "node_overlap": round(node_score, 3),
                "node_family_overlap": round(family_score, 3),
            }
    return None


def _diversify_problems(problems: list[dict], budget: int, recent_memory: list[dict]) -> list[dict]:
    # Greedily prefer problems that do not repeat recent nodes/mechanisms.
    budget = min(max(0, int(budget)), len(problems))
    if budget <= 0:
        return []
    if len(problems) <= 1:
        return problems[:budget]

    recent_mechanisms = Counter(str(item.get("mechanism_type") or "") for item in recent_memory)
    recent_nodes = Counter(
        str(node)
        for item in recent_memory
        for node in item.get("source_node_ids", [])
        if str(node).strip()
    )
    recent_titles = [str(item.get("title") or "") for item in recent_memory]

    candidates = [(idx, problem) for idx, problem in enumerate(problems) if isinstance(problem, dict)]
    selected: list[tuple[int, dict]] = []
    selected_mechanisms: Counter[str] = Counter()
    selected_sources: Counter[str] = Counter()
    selected_nodes: Counter[str] = Counter()

    def score(idx: int, problem: dict) -> float:
        mechanism = str(problem.get("mechanism_type") or problem.get("source_type") or "")
        source = str(problem.get("source_type") or "")
        nodes = _problem_node_ids(problem)
        title = str(problem.get("title") or "")

        value = -idx * 0.05
        if mechanism:
            value -= min(1.25, recent_mechanisms.get(mechanism, 0) * 0.25)
            value -= selected_mechanisms.get(mechanism, 0) * 0.8
        if source:
            value -= selected_sources.get(source, 0) * 0.35
        if nodes:
            value += min(0.35, len(nodes) * 0.08)
            value -= min(1.5, sum(recent_nodes.get(node, 0) for node in nodes) * 0.18)
            value -= sum(selected_nodes.get(node, 0) for node in nodes) * 0.9
        if any(_text_similarity(title, recent_title) >= 0.58 for recent_title in recent_titles):
            value -= 2.0
        return value

    while candidates and len(selected) < budget:
        idx, problem = max(candidates, key=lambda item: score(item[0], item[1]))
        selected.append((idx, problem))
        mechanism = str(problem.get("mechanism_type") or problem.get("source_type") or "")
        source = str(problem.get("source_type") or "")
        if mechanism:
            selected_mechanisms[mechanism] += 1
        if source:
            selected_sources[source] += 1
        for node in _problem_node_ids(problem):
            selected_nodes[node] += 1
        candidates = [(cand_idx, cand) for cand_idx, cand in candidates if cand_idx != idx]

    return [problem for _idx, problem in selected]


def _build_problem_prompt(
    signals: dict,
    recent_memory: list[dict] | None = None,
    *,
    agenda_id: int,
) -> str:
    """Build evidence prompt for Call 1 (Problem Sharpening)."""
    sections = ["# EVIDENCE FROM 10,000+ ML PAPERS\n"]
    compute = detect_compute_profile()
    try:
        compute_payload = asdict(compute)
    except TypeError:
        compute_payload = vars(compute)
    sections.append("## LOCAL EXECUTION CONSTRAINTS")
    sections.append(json.dumps(compute_payload, ensure_ascii=False, default=str))

    weighted_signals = []
    for key in (
        "contradiction_clusters",
        "performance_plateaus",
        "limitation_clusters",
        "mechanism_mismatches",
        "protocol_artifacts",
        "negative_space_gaps",
        "hidden_variable_bridges",
        "claim_method_gaps",
    ):
        rows = signals.get(key) or []
        if rows:
            weighted_signals.append(
                (
                    signal_type_weight(key.rstrip("s"), agenda_id=agenda_id),
                    key,
                    len(rows),
                )
            )
    if weighted_signals:
        weighted_signals.sort(reverse=True)
        sections.append("\n## SIGNAL PRIORITY (meta-learned weights)")
        for weight, key, count in weighted_signals[:6]:
            sections.append(f"- {key}: weight={weight:.2f}, rows={count}")
    if not compute.gpu_allowed:
        sections.append(
            "Generate paper ideas that can be executed locally without GPU training: "
            "inference-time evaluation, controlled materialized traces, CPU-only analysis, "
            "evaluation protocols, lightweight ablations, or small public-data studies. "
            "Do not require fine-tuning, embedding model training, large ASR/vision training, "
            "or multi-GPU experiments. Prefer ideas whose evidence can be executed from "
            "existing local artifacts or concrete public datasets with standard loaders. "
            "Do not invent benchmark names or require unavailable datasets."
        )
    memory_block = _recent_idea_memory_block(recent_memory or [])
    if memory_block:
        sections.append("\n" + memory_block)

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

    frontier_nodes = []
    for cluster in signals.get("limitation_clusters") or []:
        if cluster.get("node_id"):
            frontier_nodes.append(cluster["node_id"])
    for plateau in signals.get("performance_plateaus") or []:
        if plateau.get("node_id"):
            frontier_nodes.append(plateau["node_id"])
    frontier_nodes = list(dict.fromkeys(frontier_nodes))[:4]
    if frontier_nodes:
        sections.append("\n" + format_frontier_block(frontier_nodes))

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


def _build_method_prompt(problem: dict, solution_signals: list[dict] | None = None) -> str:
    """Build prompt for Call 2 (Method Invention)."""
    compute = detect_compute_profile()
    compute_constraint = ""
    ruled_out = _problem_ruled_out(problem)
    ruled_out_block = ""
    if ruled_out:
        lines = ["## Ruled-Out Approaches"]
        for item in ruled_out[:6]:
            approach = str(item.get("approach") or "").strip()
            failed = json.dumps(item.get("failed_under") or {}, ensure_ascii=False, default=str)[:240]
            if approach:
                lines.append(f"- {approach} | failed_under={failed}")
        ruled_out_block = "\n" + "\n".join(lines) + "\n"
    solution_block = ""
    if solution_signals:
        lines = ["## Candidate Solution Signals From The Graph"]
        for signal in solution_signals[:8]:
            table = str(signal.get("_signal_table") or signal.get("source") or "solution_signal")
            title = (
                signal.get("title")
                or signal.get("summary")
                or signal.get("theme")
                or signal.get("shared_factor")
                or ""
            )
            nodes = ", ".join(str(node) for node in signal.get("_node_ids") or [] if str(node).strip())
            lines.append(f"- [{table}] {title[:200]} | nodes={nodes or '?'}")
        solution_block = "\n" + "\n".join(lines) + "\n"
    if not compute.gpu_allowed:
        compute_constraint = """
## Local Execution Constraint
This machine has no usable local NVIDIA GPU and no configured remote GPU worker.
Design a method that can be validated without GPU training: inference-time evaluation,
deterministic selection, CPU-only statistical analysis, materialized trace evaluation,
or lightweight public-data experiments. Avoid methods whose core contribution requires
fine-tuning, representation learning, large speech/vision training, or GPU-heavy sweeps.
The validation path must use existing local artifacts or concrete public datasets; do not
depend on a new benchmark recipe that is not already available.
"""

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
{ruled_out_block}
{solution_block}
{compute_constraint}

Design a NEW method that addresses this specific failure mode.
The method must be technically novel — not "apply [existing technique] to [this domain]"."""


def _build_experiment_prompt(problem: dict, method: dict) -> str:
    """Build prompt for Call 3 (Experimental Design)."""
    compute = detect_compute_profile()
    compute_constraint = ""
    if not compute.gpu_allowed:
        compute_constraint = """
## Local Execution Constraint
The experimental plan must be runnable without GPU training. Use CPU-only or API-free
materialized artifacts, small public benchmark subsets, deterministic simulations,
statistical tests, and native matplotlib figures. If a GPU-trained model would be a
future extension, put it in limitations rather than the main validation plan.
Use only concrete executable datasets or local artifacts. Do not name a new benchmark
unless the plan also provides an executable local artifact recipe; otherwise choose a
controlled materialized-trace study or a standard public benchmark with a loader.
"""

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
{compute_constraint}

Design a complete experimental plan for validating this method.
Be specific: exact model names, dataset names, metric names, compute estimates."""


def _extract_method_payload(result: dict) -> dict:
    """Accept common JSON shapes from method-invention models."""
    if not isinstance(result, dict):
        return {}
    candidates = [
        result.get("method"),
        result.get("proposed_method"),
        result.get("method_definition"),
        result.get("algorithm"),
        result,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        normalized = dict(candidate)
        if not normalized.get("name"):
            normalized["name"] = (
                normalized.get("method_name")
                or normalized.get("title")
                or normalized.get("algorithm_name")
            )
        if not normalized.get("one_line"):
            normalized["one_line"] = (
                normalized.get("summary")
                or normalized.get("description")
                or normalized.get("abstract")
                or ""
            )
        if not normalized.get("why_novel"):
            normalized["why_novel"] = (
                normalized.get("novelty")
                or normalized.get("novelty_argument")
                or normalized.get("difference_from_prior_work")
                or ""
            )
        if normalized.get("name"):
            return normalized
    return {}


def _llm_temporarily_unavailable(exc: Exception) -> bool:
    return is_llm_auth_error(exc) or is_llm_provider_unavailable_error(exc)


class ProposalProblemUnavailable(Exception):
    """This research problem cannot yield a proposal candidate right now.

    Raised instead of aborting discovery: the remaining problems in the pass
    are unaffected and must still be attempted.
    """


def _proposal_candidate_and_grant(
    *,
    agenda_id: int,
    problem: dict,
) -> tuple[int, dict | None]:
    """Persist honest pre-idea identity and load an authorized proposal grant."""
    problem_id = int(problem.get("research_problem_id") or problem.get("id") or 0)
    if problem_id <= 0:
        raise ValueError("proposal generation requires a persisted research problem")
    existing = db.fetchone(
        """
        SELECT id, status FROM deep_insights
        WHERE agenda_id=? AND research_problem_id=?
          AND COALESCE(outcome, 'pending')='pending'
          AND COALESCE(status, 'candidate') NOT IN ('archived', 'exists')
        ORDER BY id DESC LIMIT 1
        """,
        (agenda_id, problem_id),
    )
    if existing:
        candidate_id = int(existing["id"])
        if str(existing.get("status") or "") != "proposal_pending":
            return candidate_id, None
    else:
        inserted = db.fetchone(
            """
            INSERT INTO deep_insights
                (agenda_id, tier, status, title, problem_statement,
                 supporting_papers, source_node_ids, source_paper_ids,
                 source_signal_refs, research_problem_id, prompt_version,
                 outcome)
            VALUES (?, 2, 'proposal_pending', ?, ?, ?, ?, ?, ?, ?, ?,
                    'pending')
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (
                agenda_id,
                str(problem.get("title") or f"Proposal for problem {problem_id}"),
                str(
                    problem.get("problem_statement")
                    or problem.get("formal_statement")
                    or ""
                ),
                json.dumps(problem.get("source_paper_ids") or []),
                json.dumps(
                    problem.get("related_node_ids")
                    or problem.get("source_node_ids")
                    or []
                ),
                json.dumps(problem.get("source_paper_ids") or []),
                json.dumps(_problem_source_refs(problem)),
                problem_id,
                configured_role_prompt_version("proposer"),
            ),
        )
        if inserted:
            candidate_id = int(inserted["id"])
            db.commit()
        else:
            # The insert collided with idx_deep_insights_pending_proposal, so
            # look up the row that actually owns that key rather than
            # re-running the usable-candidate query that just missed it. A
            # candidate left at status='proposal_pending' with a terminal
            # outcome owns the key without being usable; that is a spent
            # problem, not a race, and it must not abort the whole pass.
            holder = db.fetchone(
                """
                SELECT id, outcome FROM deep_insights
                WHERE agenda_id=? AND research_problem_id=?
                  AND status='proposal_pending'
                ORDER BY id DESC LIMIT 1
                """,
                (agenda_id, problem_id),
            )
            if not holder:
                db.rollback()
                raise RuntimeError("proposal candidate identity race")
            db.commit()
            if str(holder.get("outcome") or "pending") != "pending":
                raise ProposalProblemUnavailable(
                    f"research problem {problem_id} is held by spent proposal "
                    f"candidate {int(holder['id'])} "
                    f"(outcome={holder.get('outcome')})"
                )
            candidate_id = int(holder["id"])
    grant = db.fetchone(
        """
        SELECT id, token_cap
        FROM resource_grants
        WHERE agenda_id=? AND idea_id=? AND stage='proposal'
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        ORDER BY id DESC LIMIT 1
        """,
        (agenda_id, candidate_id),
    )
    return candidate_id, grant


def discover_paper_ideas(
    max_problems: int = 8,
    max_papers: int | None = None,
    *,
    agenda_id: int,
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
    fallback_problem_refs = signal_refs_from_rows(
        getattr(signals, "payload", signals),
        roles={"problem", "derived"},
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
        print(
            "[PAPER_IDEA] No harvested signals available; continuing from "
            "the agenda direction seed.",
            flush=True,
        )

    recent_memory = _recent_tier2_memory()
    problems = select_problem_first_candidates(
        limit=max(max_problems * 2, max_problems),
        agenda_id=agenda_id,
        refresh=True,
    )
    if problems:
        print(
            f"[PAPER_IDEA] Problem-first pool selected {len(problems)} persisted research problems",
            flush=True,
        )
    else:
        problems = discover_research_problems(
            limit=max(max_problems * 2, max_problems),
            agenda_id=agenda_id,
            persist=True,
        )
        if not problems:
            print(
                "[PAPER_IDEA] No persisted research problems; proposal LLM "
                "generation remains fail-closed.",
                flush=True,
            )
            return []
    problem_budget = min(len(problems), max_problems + max(2, max_papers // 2))
    problems = _diversify_problems(problems, problem_budget, recent_memory)
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
        try:
            proposal_candidate_id, proposal_grant = _proposal_candidate_and_grant(
                agenda_id=agenda_id,
                problem=problem,
            )
        except ProposalProblemUnavailable as exc:
            print(f"[PAPER_IDEA] Skipping problem: {exc}", flush=True)
            continue
        if not proposal_grant:
            print(
                "[PAPER_IDEA] Proposal candidate "
                f"{proposal_candidate_id} awaits Frontier/Portfolio grant.",
                flush=True,
            )
            continue
        proposal_token_cap = max(
            1,
            min(16_000, int(proposal_grant.get("token_cap") or 0) // 2),
        )
        prompt_version = configured_role_prompt_version("proposer")
        # Reaching here proves no earlier attempt delivered an idea: a realized
        # proposal settles its grant to 'consumed', and the grant lookup above
        # accepts only an active one. So a fresh attempt cannot recharge the
        # agenda for work it already has.
        from meta_harness.grant_usage import GrantUsageError, GrantUsageLedger

        attempts = GrantUsageLedger(int(proposal_grant["id"]))
        try:
            method_key = attempts.next_attempt_key(
                f"proposal-method:{agenda_id}:{proposal_candidate_id}"
            )
            experiment_key = attempts.next_attempt_key(
                f"proposal-experiment:{agenda_id}:{proposal_candidate_id}"
            )
        except GrantUsageError as exc:
            print(
                f"[PAPER_IDEA] Proposal candidate {proposal_candidate_id} "
                f"is out of attempts: {exc}",
                flush=True,
            )
            continue

        # Stage 2: Method Invention
        print(f"[PAPER_IDEA] Call 2/3: Inventing method for '{title[:50]}'...", flush=True)
        solution_signals = get_solution_signals(
            {"node_ids": problem.get("related_node_ids") or problem.get("source_node_ids") or []},
            limit=12,
        )
        solution_signal_refs = {
            "signals": [
                signal.get("_source_ref")
                for signal in solution_signals
                if isinstance(signal, dict) and isinstance(signal.get("_source_ref"), dict)
            ],
            "node_ids": list(
                dict.fromkeys(
                    str(node)
                    for signal in solution_signals
                    if isinstance(signal, dict)
                    for node in signal.get("_node_ids") or []
                    if str(node).strip()
                )
            ),
            "paper_ids": list(
                dict.fromkeys(
                    str(pid)
                    for signal in solution_signals
                    if isinstance(signal, dict)
                    for pid in signal.get("_paper_ids") or []
                    if str(pid).strip()
                )
            ),
        }
        method_prompt = _build_method_prompt(problem, solution_signals=solution_signals)
        method_route: dict = {}
        try:
            result2, tokens2, method_route = call_llm_json_for_role(
                METHOD_INVENTION_SYSTEM,
                method_prompt,
                agenda_id=agenda_id,
                idea_id=proposal_candidate_id,
                role="proposer",
                stage="proposal",
                resource_grant_id=int(proposal_grant["id"]),
                operation="proposal_method_invention",
                idempotency_key=method_key,
                prompt_version=prompt_version,
                max_tokens=proposal_token_cap,
            )
            total_tokens += tokens2
            total_calls += 1
        except Exception as e:
            if _llm_temporarily_unavailable(e):
                print(f"[PAPER_IDEA] Method invention paused: LLM unavailable ({e})", flush=True)
                break
            print(f"[PAPER_IDEA] Method invention failed for '{title[:50]}': {e}", flush=True)
            continue

        method = _extract_method_payload(result2)
        if not method.get("name"):
            print(f"[PAPER_IDEA] No method produced for '{title[:50]}'", flush=True)
            continue

        why_novel = method.get("why_novel", "").lower()
        if not why_novel or len(why_novel) < 30:
            print(f"[PAPER_IDEA] Rejected (no novelty argument): {method['name']}", flush=True)
            continue

        precheck = {
            "title": title,
            "problem_statement": problem.get("problem_statement") or problem.get("formal_statement", ""),
            "proposed_method": json.dumps(method),
            "source_node_ids": json.dumps(problem.get("related_node_ids", [])),
            "mechanism_type": problem.get("mechanism_type", "mechanism_mismatch"),
        }
        gate = graph_novelty_gate(precheck)
        if gate:
            print(
                f"[PAPER_IDEA] Rejected by graph novelty gate ({gate['graph_novelty']['score']}): "
                f"{method['name']}",
                flush=True,
            )
            continue

        # Stage 3: Experimental Design
        print(f"[PAPER_IDEA] Call 3/3: Designing experiments for '{method['name']}'...", flush=True)
        exp_prompt = _build_experiment_prompt(problem, method)
        experiment_route: dict = {}
        try:
            result3, tokens3, experiment_route = call_llm_json_for_role(
                EXPERIMENT_DESIGN_SYSTEM,
                exp_prompt,
                agenda_id=agenda_id,
                idea_id=proposal_candidate_id,
                role="proposer",
                stage="proposal",
                resource_grant_id=int(proposal_grant["id"]),
                operation="proposal_experiment_design",
                idempotency_key=experiment_key,
                prompt_version=prompt_version,
                max_tokens=proposal_token_cap,
            )
            total_tokens += tokens3
            total_calls += 1
        except Exception as e:
            if _llm_temporarily_unavailable(e):
                print(f"[PAPER_IDEA] Experiment design skipped: LLM unavailable ({e})", flush=True)
                result3 = {}
            else:
                print(f"[PAPER_IDEA] Experiment design failed: {e}", flush=True)
                result3 = {}

        generated_problem_awareness = result3.get("problem_awareness")
        if not isinstance(generated_problem_awareness, dict):
            generated_problem_awareness = {}
        expected_results = result3.get("expected_results")
        if not isinstance(expected_results, dict):
            expected_results = {}
        problem_awareness = {
            "central_question": generated_problem_awareness.get("central_question")
            or problem.get("central_question")
            or title,
            "motivation": generated_problem_awareness.get("motivation")
            or problem.get("motivation")
            or problem.get("current_failure_mode", ""),
            "method_answer": generated_problem_awareness.get("method_answer")
            or method.get("mechanism_repair")
            or method.get("one_line", ""),
            "result_claim": generated_problem_awareness.get("result_claim")
            or expected_results.get("solid")
            or problem.get("result_that_would_change_belief", ""),
            "falsification_result": generated_problem_awareness.get("falsification_result")
            or method.get("falsification_hook", ""),
        }

        raw_paper_title = result3.get("paper_title") or f"{method['name']}: {title}"
        normalized_paper_title = normalize_paper_title(
            raw_paper_title,
            method_name=method.get("name"),
            claim=problem_awareness.get("central_question") or method.get("one_line") or title,
            context={"full_benchmark_completed": False},
        )

        deep_insight = {
            "proposal_candidate_id": proposal_candidate_id,
            "resource_grant_id": int(proposal_grant["id"]),
            "agenda_id": agenda_id,
            "tier": 2,
            "status": "candidate",
            "title": normalized_paper_title,
            "problem_statement": problem.get("problem_statement") or problem.get("formal_statement", ""),
            "existing_weakness": problem.get("current_failure_mode", ""),
            "proposed_method": json.dumps(method),
            "experimental_plan": json.dumps({
                "baselines": result3.get("baselines", []),
                "datasets": result3.get("datasets", []),
                "metrics": result3.get("metrics", {}),
                "ablations": result3.get("ablations", []),
                "expected_results": result3.get("expected_results", {}),
                "compute_budget": result3.get("compute_budget", {}),
                "execution_requirements": result3.get(
                    "execution_requirements", {}
                ),
                "risks": result3.get("risks", []),
                "paper_title": normalized_paper_title,
                "raw_paper_title": raw_paper_title,
                "title_source": "paper_idea_title_policy",
            }),
            "related_work_positioning": json.dumps(result3.get("paper_outline", {})),
            "supporting_papers": json.dumps(problem.get("source_paper_ids", [])),
            "source_node_ids": json.dumps(problem.get("related_node_ids", [])),
            "source_paper_ids": json.dumps(problem.get("source_paper_ids", [])),
            "source_signal_ids": json.dumps(
                [
                    ref.get("content_hash")
                    for ref in (
                        _problem_source_refs(problem).get("signals", [])
                        + solution_signal_refs.get("signals", [])
                    )
                    if isinstance(ref, dict) and ref.get("content_hash")
                ]
            ),
            "source_signal_refs": json.dumps(
                {
                    "signals": (
                        _problem_source_refs(problem).get("signals", [])
                        + solution_signal_refs.get("signals", [])
                    ),
                    "node_ids": list(
                        dict.fromkeys(
                            (problem.get("related_node_ids") or [])
                            + solution_signal_refs.get("node_ids", [])
                        )
                    ),
                    "paper_ids": list(
                        dict.fromkeys(
                            (problem.get("source_paper_ids") or [])
                            + solution_signal_refs.get("paper_ids", [])
                        )
                    ),
                }
            ),
            "evidence_summary": problem.get("source_evidence", ""),
            "mechanism_type": problem.get("mechanism_type", "mechanism_mismatch"),
            "problem_awareness": json.dumps(problem_awareness),
            "research_problem_id": problem.get("research_problem_id"),
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
            "prompt_version": prompt_version,
            "model_version": str(
                experiment_route.get("model")
                or method_route.get("model")
                or ""
            ),
            "proposer_route": experiment_route or method_route,
        }

        duplicate = _find_existing_tier2_duplicate(
            deep_insight, exclude_id=proposal_candidate_id
        )
        if duplicate:
            print(
                f"[PAPER_IDEA] Rejected duplicate of idea {duplicate['id']} "
                f"(title_sim={duplicate['title_similarity']}, node_overlap={duplicate['node_overlap']}): "
                f"{method['name']}",
                flush=True,
            )
            continue

        input_issue = get_evosci_input_issue(deep_insight, mode="verification")
        if input_issue:
            missing = ", ".join(input_issue.get("missing_fields") or [])
            print(
                f"[PAPER_IDEA] Skipped underspecified idea '{title[:60]}' (missing: {missing})",
                flush=True,
            )
            continue

        refined_insight = deep_insight
        if TIER2_EVOSCI_PREINSERT_REVIEW:
            print(f"[PAPER_IDEA] Pre-insert EvoScientist review + debate refinement for '{method['name']}'...", flush=True)
            review_result = review_and_refine_tier2_idea(deep_insight)
            if not review_result.get("accepted"):
                print(
                    f"[PAPER_IDEA] Rejected after pre-insert review ({review_result.get('reason')}): {method['name']}",
                    flush=True,
                )
                continue
            refined_insight = review_result.get("insight") or deep_insight
            duplicate = _find_existing_tier2_duplicate(
                refined_insight, exclude_id=proposal_candidate_id
            )
            if duplicate:
                print(
                    f"[PAPER_IDEA] Rejected duplicate after review/refine of idea {duplicate['id']} "
                    f"(title_sim={duplicate['title_similarity']}, node_overlap={duplicate['node_overlap']}): "
                    f"{method['name']}",
                    flush=True,
                )
                continue
            print(
                f"[PAPER_IDEA] Accepted after review/refine: {method['name']} — {refined_insight.get('title', title)[:60]}",
                flush=True,
            )
        else:
            print(
                f"[PAPER_IDEA] Accepted without pre-insert EvoScientist review: {method['name']}",
                flush=True,
            )

        deep_insights.append(enrich_deep_insight(attach_graph_taste_to_insight(refined_insight)))

    print(f"[PAPER_IDEA] Done: {len(deep_insights)} paper ideas from {len(problems)} problems. "
          f"Tokens: {total_tokens}, LLM calls: {total_calls}", flush=True)
    return deep_insights
