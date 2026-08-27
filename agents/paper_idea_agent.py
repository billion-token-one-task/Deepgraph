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
import time
from collections import Counter
from pathlib import Path
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
    call_llm_for_role,
    call_llm_json_for_role,
    configured_role_prompt_version,
    is_llm_auth_error,
    is_llm_provider_unavailable_error,
    parse_llm_json_text,
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
   - Use concrete public dataset/model repository IDs; never use a display
     name as a repository ID. The same rule binds "datasets" and "baselines"
     above: the execution contract is checked against the repository ids named
     there, so a plan that says "MOCHEG" in one place and "owner/mocheg" in the
     other is refused as unbound. Write the repository id in both.
   - "field_mapping" is keyed by CONTRACT ROLE (the dataset field roles in the
     capability envelope), and valued by the dataset's own column name. A key
     that is a column name is refused.
   - Set "revision" to "main" unless you are copying a revision hash from
     material provided in this prompt. NEVER invent a commit hash: a
     fabricated revision fails the metadata preflight and strands the idea
     (idea 124 did exactly this on 2026-08-17). The preflight resolves
     "main" to a concrete revision and pins it for reproducibility.
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
      "model": "Hugging Face repository id, owner/name -- NOT a display name. One baseline must carry exactly the string in execution_requirements.model.repository_id",
      "source_paper": "paper ID if known",
      "expected_performance": "Estimated metric value"
    }
  ],
  "datasets": [
    {
      "name": "Hugging Face repository id, owner/name -- NOT a display name, and no trailing parenthetical. One dataset must carry exactly the string in execution_requirements.dataset.repository_id",
      "split": "train/val/test sizes",
      "why": "Why this dataset tests the hypothesis"
    }
  ],
  "metrics": {
    "primary": "begin with the exact execution_requirements.metric.name token, then the explanation",
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
      "field_mapping": {"<contract role from the capability envelope above>": "actual_column_name"}
    },
    "model": {
      "repository_id": "public repository id",
      "revision": "immutable commit or explicit tag",
      "framework": "transformers|sentence_transformers|another explicit framework",
      "task": "causal_lm|sequence_classification|embedding|another explicit task",
      "min_vram_gb": "estimate from the checkpoint you named: fp16 weights plus KV cache, in GB. Never 0 for a model that needs an accelerator",
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
    "sample_cap": 200,
    "artifact_contract": ["final_results", "raw_predictions", "environment_manifest", "dataset_manifest", "model_manifest"],
    "preferred_backends": ["only backends the capability envelope above lists for your task_protocol; a protocol that needs an accelerator does not accept cpu"]
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


# Jaccard over very small node sets is degenerate: between two single-node
# ideas it is 1.0 whenever the node matches, which says "same subfield", not
# "same idea". 27 of the 128 tier-1/2 ideas carry exactly one node, so treating
# that as near-conclusive evidence let one idea per taxonomy node exist across
# the whole system - agenda 11's HOCSU candidate was rejected against agenda
# 10's idea 99 on a single shared node at title similarity 0.143. Below this
# size the overlap scores stay informative enough to report, but they may not
# carry a rejection on their own.
MIN_NODES_FOR_OVERLAP_EVIDENCE = 2


# The mechanism taxonomy the Tier-2 prompt actually asks for. Anything else
# in the column is a provenance label, not a mechanism: discovery_supervisor
# stamps the candidate pool's name there, so every idea raised from the
# research-problem pool carries the literal string "problem". Twelve of
# agenda 10's ideas shared that value on 2026-08-19, which made
# ``same_mechanism`` true for essentially every pair and reduced the
# duplicate gate to "any title sharing 18% of its tokens with any prior
# idea". Agenda 10 could then never invent anything again -- the M2 funnel
# went dry with a funded proposal grant in hand.
MECHANISM_TAXONOMY = frozenset(
    {
        "protocol_artifact",
        "mechanism_mismatch",
        "negative_space_gap",
        "hidden_variable_bridge",
        "claim_method_gap",
        "plateau",
    }
)


def _is_mechanism(value: object) -> bool:
    """True only for a real mechanism, never for a pool-provenance label."""
    return str(value or "").strip().lower() in MECHANISM_TAXONOMY


def _node_overlap_is_decisive(a, b) -> bool:
    """Are both node sets large enough for their overlap to mean anything?"""

    left = {str(x).strip() for x in _json_list(a) if str(x).strip()}
    right = {str(x).strip() for x in _json_list(b) if str(x).strip()}
    return (
        min(len(left), len(right)) >= MIN_NODES_FOR_OVERLAP_EVIDENCE
    )


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
    # Every unrealized proposal shell, not only the candidate's own: two
    # funded shells raised from one problem otherwise dup-kill each other's
    # realization symmetrically, and neither can ever realize (ideas 141/142
    # deadlocked this way on 2026-08-18, burning grant 79 to expiry). A
    # pending shell is a funding stub, not a prior idea. Failing open to an
    # empty set only narrows the exclusion, never disables the gate.
    try:
        pending_shells = {
            int(r["deep_insight_id"])
            for r in db.fetchall(
                "SELECT deep_insight_id FROM auto_research_jobs"
                " WHERE status='deferred' AND stage='proposal_generation_granted'"
            )
        }
    except Exception:
        pending_shells = set()
    for row in rows:
        row_id = int(row.get("id") or 0)
        if skip and row_id == skip:
            continue
        if row_id in pending_shells:
            continue
        title_score = _token_jaccard(title, row.get("title") or "")
        node_score = _node_jaccard(nodes, row.get("source_node_ids"))
        family_score = _node_family_jaccard(nodes, row.get("source_node_ids"))
        # Only a taxonomy mechanism may stand in for identity; see
        # MECHANISM_TAXONOMY on why a provenance label must not.
        same_mechanism = (
            _is_mechanism(mechanism)
            and mechanism == str(row.get("mechanism_type") or "")
        )
        # Two ideas raised from the same research problem inherit that
        # problem's source_node_ids verbatim, so their node overlap is 1.0 by
        # construction and carries no information about whether the ideas are
        # the same. Scoring it anyway meant a problem could yield exactly one
        # idea ever: agenda 11's second idea on problem 8 was rejected as a
        # duplicate of its first at node_overlap 1.0 and title_sim 0.095.
        # Content signals still apply between siblings; provenance does not.
        siblings = bool(own_problem) and int(row.get("research_problem_id") or 0) == own_problem
        # Provenance may decide only when it carries information: not between
        # ideas raised from the same problem (identical by construction), and
        # not between node sets too small for their overlap to distinguish
        # "same idea" from "same subfield".
        provenance_counts = not siblings and _node_overlap_is_decisive(
            nodes, row.get("source_node_ids")
        )
        too_close = (
            title_score >= 0.32
            or (provenance_counts and node_score >= 0.50 and same_mechanism)
            or (provenance_counts and node_score >= 0.62 and title_score >= 0.04)
            or (
                provenance_counts
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
    else:
        # Tell the designer what hardware actually exists. Idea 129 declared
        # Meta-Llama-3-8B on 2026-08-17 while the only live accelerator was a
        # 14.5 GB Colab T4; preflight correctly refused it and the idea
        # stalled. The ceiling below is the measured canary value, not an
        # aspiration; models that cannot run inference within it will fail
        # capability preflight.
        try:
            from meta_harness.preflight_repository import runtime_preflight_environment
            from meta_harness.runner_capability import RunnerRegistry

            environment = runtime_preflight_environment()
            live_vram = [
                environment.backend_vram_gb.get(name, 0.0)
                for name in environment.enabled_backends
            ]
            ceiling = max(live_vram, default=0.0)
            # Idea 114 declared TruthfulQA mc2 on 2026-08-17 and was refused
            # with three structural blockers: the designer was never told what
            # the runners can execute. The envelope below is read from the
            # same registry preflight enforces -- one fact, both ends.
            envelope = "\n".join(
                f"- task_protocol {'/'.join(cap.task_protocols)}: model task "
                f"{'/'.join(cap.model_tasks)}; dataset field roles "
                f"{'/'.join(cap.dataset_roles)}; metrics "
                f"{', '.join(cap.metric_names)}"
                for cap in RunnerRegistry().all()
            )
            compute_constraint = f"""
## Available Execution Hardware (measured, not aspirational)
Enabled backends: {", ".join(environment.enabled_backends) or "cpu"}.
Largest verified GPU right now: {ceiling:.1f} GB VRAM.
Declare a model whose inference fits that VRAM (fp16 weights plus KV cache);
anything larger fails capability preflight and the idea stalls unexecuted.
Put larger-model scaling in limitations or future work, not the primary plan.

## Executable Capability Envelope (the only contracts a runner can execute)
{envelope}
execution_requirements MUST stay strictly inside one line above: its task
protocol, model task, dataset field roles and metric. A metric or protocol
outside this envelope is not creativity, it is an unexecutable plan; the
preflight gate will refuse it and the idea stalls.

## The runners EVALUATE ONLY -- they never train
No runner has a training stage: there is no Trainer, no optimizer and no
backward pass anywhere in meta_harness/runners. Whatever weights you name are
loaded and measured exactly as published.

The declared model must therefore ALREADY carry the head for the task you
declare. For task_protocol sequence_classification that means an
already-fine-tuned classifier checkpoint, NOT a base encoder: roberta-base,
bert-base-uncased and distilbert-base-uncased publish a fill-mask head, so
preflight reads their task as fill-mask and refuses the plan as
model_task_mismatch. Loading one of them for classification would attach a
randomly initialised head and measure noise.

This is the single largest cause of stalled ideas: thirteen of the nineteen
refusals between 2026-08-17 and 2026-08-20 were model_task_mismatch, every one
of them a base encoder declared for a classification protocol, and each retry
re-declared the same pairing. Name a checkpoint whose published task IS the
task you declare, or choose a protocol its published head already serves.
"""
        except Exception:
            compute_constraint = ""

        # A redesign that is not told how the last one died repeats it. Ideas
        # 136, 144, 150 and 159 each declared the same unexecutable pairing
        # twice; the reason codes were recorded on the retired candidate and
        # in its outcome, and then read by nobody who could act on them. The
        # retirement path returns the problem to the pool -- this returns what
        # the pool needs to do better than last time.
        try:
            prior = db.fetchall(
                """
                SELECT DISTINCT p.reason_codes_json
                FROM candidate_preflight_results_v1 p
                JOIN deep_insights d ON d.id = p.idea_id
                WHERE p.status <> 'passed'
                  AND d.research_problem_id = ?
                ORDER BY p.reason_codes_json
                LIMIT 5
                """,
                (problem.get("id") or problem.get("research_problem_id") or 0,),
            )
            codes: list[str] = []
            for row in prior or []:
                raw = dict(row).get("reason_codes_json")
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw or "[]")
                    except json.JSONDecodeError:
                        raw = []
                for code in raw or []:
                    if str(code) not in codes:
                        codes.append(str(code))
            if codes:
                compute_constraint += f"""
## Earlier plans for THIS problem were refused for these reasons
{", ".join(codes)}
Each of those was a plan that could not run, not a plan that ran and failed.
Do not re-declare the pairing that produced them; if you cannot avoid a
refusal reason, change the protocol or the dataset rather than restating it.
"""
        except Exception:
            pass

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


def _load_json_value(value, default):
    if isinstance(value, type(default)):
        return value
    if isinstance(value, (dict, list)):
        return default
    try:
        parsed = json.loads(value or "")
    except (TypeError, ValueError):
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _load_exact_proposal_problem(
    *,
    job_id: int,
    agenda_id: int,
    idea_id: int,
    resource_grant_id: int,
) -> tuple[dict, dict]:
    """Load only the records named by a controlled proposal request.

    This deliberately does not hydrate from signal tables, refresh the problem
    pool, inspect recent ideas, or touch scheduler/EvoScientist state.  The
    persisted research problem and proposal shell are the complete authority
    for this one-shot recovery path.
    """

    row = db.fetchone(
        """
        SELECT arj.id AS job_id, arj.status AS job_status,
               arj.stage AS job_stage, arj.resource_grant_id,
               di.id AS idea_id, di.status AS insight_status,
               di.title AS insight_title,
               di.problem_statement AS insight_problem_statement,
               di.source_node_ids AS insight_node_ids,
               di.source_paper_ids AS insight_paper_ids,
               di.source_signal_refs AS insight_signal_refs,
               di.research_problem_id,
               rp.problem_statement, rp.source_signal_ref,
               rp.node_ids, rp.paper_ids, rp.ruled_out_approaches,
               rp.problem_quality_score, rp.status AS problem_status,
               rp.attempts_count,
               rg.id AS grant_id, rg.stage AS grant_stage,
               rg.status AS grant_status, rg.token_cap, rg.max_gpu_hours,
               rg.backend_allowlist_json,
               CASE WHEN rg.expires_at > CURRENT_TIMESTAMP THEN 1 ELSE 0 END
                   AS grant_live
        FROM auto_research_jobs arj
        JOIN deep_insights di
          ON di.id=arj.deep_insight_id AND di.agenda_id=arj.agenda_id
        JOIN research_problems rp
          ON rp.id=di.research_problem_id AND rp.agenda_id=arj.agenda_id
        JOIN resource_grants rg
          ON rg.id=arj.resource_grant_id AND rg.agenda_id=arj.agenda_id
         AND rg.idea_id=arj.deep_insight_id
        WHERE arj.id=? AND arj.agenda_id=? AND arj.deep_insight_id=?
          AND arj.resource_grant_id=?
        """,
        (job_id, agenda_id, idea_id, resource_grant_id),
    )
    if not row:
        raise ValueError("exact proposal job/problem/grant scope was not found")
    scope = dict(row)
    backends = {
        str(item).strip().lower()
        for item in _load_json_value(scope.get("backend_allowlist_json"), [])
    }
    if (
        str(scope.get("job_status") or "") != "deferred"
        or str(scope.get("job_stage") or "") != "proposal_generation_granted"
        or str(scope.get("insight_status") or "") != "proposal_pending"
        or str(scope.get("problem_status") or "") not in {"open", "exploring"}
        or int(scope.get("attempts_count") or 0) >= 3
        or str(scope.get("grant_stage") or "") != "proposal"
        or str(scope.get("grant_status") or "") != "active"
        or int(scope.get("grant_live") or 0) != 1
        or int(scope.get("token_cap") or 0) <= 0
        or float(scope.get("max_gpu_hours") or 0.0) != 0.0
        or backends != {"llm"}
    ):
        raise ValueError("exact proposal scope is not executable and llm-only")

    statement = str(
        scope.get("problem_statement")
        or scope.get("insight_problem_statement")
        or ""
    ).strip()
    if not statement:
        raise ValueError("exact proposal research problem has no statement")
    node_ids = _load_json_value(scope.get("node_ids"), []) or _load_json_value(
        scope.get("insight_node_ids"), []
    )
    paper_ids = _load_json_value(scope.get("paper_ids"), []) or _load_json_value(
        scope.get("insight_paper_ids"), []
    )
    source_ref = _load_json_value(scope.get("source_signal_ref"), {})
    source_refs = _load_json_value(scope.get("insight_signal_refs"), {})
    if not source_refs:
        source_refs = {
            "signals": [source_ref] if source_ref else [],
            "node_ids": node_ids,
            "paper_ids": paper_ids,
        }
    source_table = str(source_ref.get("table") or "persisted_problem")
    mechanism = {
        "contradiction_clusters": "mechanism_mismatch",
        "performance_plateaus": "plateau",
        "protocol_artifacts": "protocol_artifact",
        "negative_space_gaps": "negative_space_gap",
        "claim_method_gaps": "claim_method_gap",
        "mechanism_mismatches": "mechanism_mismatch",
    }.get(source_table, "claim_method_gap")
    title = str(scope.get("insight_title") or statement).strip()
    problem = {
        "id": int(scope["research_problem_id"]),
        "research_problem_id": int(scope["research_problem_id"]),
        "title": title,
        "source_type": source_table,
        "source_evidence": (
            f"Persisted research problem {int(scope['research_problem_id'])}; "
            f"supporting_papers={len(paper_ids)}"
        ),
        "formal_statement": statement,
        "problem_statement": statement,
        "current_failure_mode": statement,
        "desideratum": (
            "Produce a falsifiable, bounded method and an executable "
            "CPU-first validation plan for this persisted problem."
        ),
        "impact_scope": (
            f"{len(paper_ids)} persisted supporting papers across "
            f"{len(node_ids)} persisted taxonomy nodes"
        ),
        "related_node_ids": [str(item) for item in node_ids],
        "source_paper_ids": [str(item) for item in paper_ids],
        "source_signal_refs": source_refs,
        "source_evidence_ref": source_ref,
        "ruled_out_approaches": _load_json_value(
            scope.get("ruled_out_approaches"), []
        ),
        "mechanism_type": mechanism,
        "central_question": statement,
        "motivation": statement,
        "result_that_would_change_belief": (
            "A preregistered bounded experiment that rejects the named failure "
            "mode against explicit baselines."
        ),
        "non_numeric_evidence": [statement],
        "problem_quality_score": float(
            scope.get("problem_quality_score") or 0.0
        ),
    }
    return problem, {"id": int(scope["grant_id"]), "token_cap": int(scope["token_cap"])}


def _build_exact_method_prompt(problem: dict) -> str:
    """Pure prompt builder for an exact, CPU-first recovery proposal."""

    ruled_out = _problem_ruled_out(problem)
    ruled_out_text = json.dumps(ruled_out[:6], ensure_ascii=False, sort_keys=True)
    return f"""# EXACT PERSISTED RESEARCH PROBLEM

Title: {problem['title']}
Formal statement: {problem['formal_statement']}
Failure mode: {problem['current_failure_mode']}
Desideratum: {problem['desideratum']}
Persisted taxonomy nodes: {json.dumps(problem.get('related_node_ids') or [])}
Persisted ruled-out approaches: {ruled_out_text}

Invent one technically novel method for this exact problem. The first
validation must be CPU-only and bounded; do not require GPU training or a
global discovery refresh."""


def _build_exact_experiment_prompt(problem: dict, method: dict) -> str:
    """Pure experiment prompt; unlike the periodic path it queries no state."""

    return f"""# EXACT BOUNDED PROPOSAL

Problem: {problem['formal_statement']}
Failure mode: {problem['current_failure_mode']}
Method: {method.get('name', 'Unnamed')}
Method summary: {method.get('one_line', '')}
Method definition: {str(method.get('definition') or '')[:800]}
Persisted taxonomy nodes: {json.dumps(problem.get('related_node_ids') or [])}

Design one complete, falsifiable CPU-only pilot. Specify concrete data or
materialized artifacts, baselines, metrics, ablations, rejection thresholds,
expected outcomes, and bounded execution requirements. Do not require GPU
training, an agenda-wide scan, or a new global benchmark service."""


def _call_exact_proposal_llm(
    *,
    job_id: int,
    agenda_id: int,
    idea_id: int,
    grant_id: int,
    operation: str,
    system_prompt: str,
    user_prompt: str,
    prompt_version: str,
    token_cap: int,
) -> tuple[str, int, dict]:
    """Replay one delivered output or buy it once behind a durable checkpoint."""

    from meta_harness.grant_usage import GrantUsageLedger
    from meta_harness.proposal_checkpoint import (
        ProposalCheckpointError,
        ProposalCheckpointRepository,
        ProposalCheckpointScope,
        proposal_input_digest,
    )

    digest = proposal_input_digest(
        job_id=job_id,
        agenda_id=agenda_id,
        idea_id=idea_id,
        resource_grant_id=grant_id,
        operation=operation,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_version=prompt_version,
        token_cap=token_cap,
    )
    checkpoint_scope = ProposalCheckpointScope(
        job_id=job_id,
        agenda_id=agenda_id,
        idea_id=idea_id,
        resource_grant_id=grant_id,
        operation=operation,
        input_digest=digest,
    )
    checkpoints = ProposalCheckpointRepository()
    delivered = checkpoints.recover_or_refuse(checkpoint_scope)
    if delivered is None:
        base_key = checkpoints.idempotency_base(checkpoint_scope)
        attempt_key = GrantUsageLedger(grant_id).next_attempt_key(base_key)
        call_llm_for_role(
            system_prompt,
            user_prompt,
            agenda_id=agenda_id,
            idea_id=idea_id,
            role="proposer",
            stage="proposal",
            resource_grant_id=grant_id,
            operation=operation,
            idempotency_key=attempt_key,
            prompt_version=prompt_version,
            max_tokens=token_cap,
            total_token_cap=token_cap,
            delivery_sink=lambda payload: checkpoints.save_delivery(
                checkpoint_scope, payload
            ),
        )
        delivered = checkpoints.recover_or_refuse(checkpoint_scope)
        if delivered is None:
            raise ProposalCheckpointError(
                "proposal provider returned without a durable delivery checkpoint"
            )
    return (
        str(delivered["output"]),
        int(delivered.get("tokens_used") or 0),
        dict(delivered.get("route") or {}),
    )


def _discover_exact_bounded_proposal(
    *,
    job_id: int,
    agenda_id: int,
    idea_id: int,
    resource_grant_id: int,
) -> list[dict]:
    """Realize one named proposal without any agenda-wide side path."""

    from meta_harness.proposal_checkpoint import ProposalCheckpointError

    def invalid_delivery(reason: str, exc: Exception | None = None) -> None:
        error = ProposalCheckpointError(
            f"checkpointed exact proposal {reason}; automatic retry is "
            "forbidden and operator reconciliation is required"
        )
        if exc is None:
            raise error
        raise error from exc

    problem, grant = _load_exact_proposal_problem(
        job_id=job_id,
        agenda_id=agenda_id,
        idea_id=idea_id,
        resource_grant_id=resource_grant_id,
    )
    prompt_version = configured_role_prompt_version("proposer")
    # Room for the method call plus every contract attempt, not for exactly
    # two calls. Sized at // 2 the grant funded one design call, so the first
    # repair found the budget already spent -- and left the grant exhausted
    # for the candidate behind it (grant 387, agenda 16, 2026-08-27).
    token_cap = _proposal_call_token_cap(int(grant["token_cap"]))

    raw_method, method_tokens, method_route = _call_exact_proposal_llm(
        job_id=job_id,
        agenda_id=agenda_id,
        idea_id=idea_id,
        grant_id=resource_grant_id,
        operation="proposal_method_invention",
        system_prompt=METHOD_INVENTION_SYSTEM,
        user_prompt=_build_exact_method_prompt(problem),
        prompt_version=prompt_version,
        token_cap=token_cap,
    )
    try:
        method_payload, _ = parse_llm_json_text(raw_method)
    except Exception as exc:
        invalid_delivery("method output is not valid JSON", exc)
    method = _extract_method_payload(method_payload)
    if not method.get("name"):
        invalid_delivery("has no method")
    why_novel = str(method.get("why_novel") or "").strip()
    if len(why_novel) < 30:
        invalid_delivery("has no novelty argument")

    # The design call is the contract loop. This is the path a funded proposal
    # actually takes -- realize_funded_proposals -> execute_bounded_proposal ->
    # here -- so it is the path that produced all 35 candidates of the
    # 2026-08-25..27 batch, of which 2 were executable. The durable checkpoint
    # is keyed on the prompt digest, so appending the refusal reasons makes a
    # genuinely new input rather than replaying the answer that was refused.
    base_experiment_prompt = _build_exact_experiment_prompt(problem, method)
    agenda_rule = _agenda_scope_rule(agenda_id)
    # The same text the topic gate will match this candidate's claim against.
    claim_text = " ".join(
        str(value or "")
        for value in (
            problem["title"],
            problem["problem_statement"],
            json.dumps(method, ensure_ascii=False),
        )
    )
    resolver = RepositoryResolver()
    experiment_tokens = 0
    experiment_calls = 0
    experiment_route: dict = {}
    experiment: dict = {}
    review = None
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        user_prompt = base_experiment_prompt
        if review is not None:
            user_prompt = f"{base_experiment_prompt}\n\n{render_violations(review)}"
        # A repair is a new operation, not the same one re-asked. The proposal
        # checkpoint refuses a changed input fingerprint for an operation that
        # already delivered -- rightly, since that is how a crash loop
        # re-bills the same step. A bounded repair is a different thing: it is
        # billed once per attempt against the grant's own ledger and has to be
        # auditable as its own delivery, so it gets its own operation name and
        # its own checkpoint rather than overwriting the one before it.
        operation = (
            "proposal_experiment_design"
            if attempt == 1
            else f"proposal_experiment_design:repair{attempt - 1}"
        )
        raw_experiment, attempt_tokens, experiment_route = _call_exact_proposal_llm(
            job_id=job_id,
            agenda_id=agenda_id,
            idea_id=idea_id,
            grant_id=resource_grant_id,
            operation=operation,
            system_prompt=EXPERIMENT_DESIGN_SYSTEM,
            user_prompt=user_prompt,
            prompt_version=prompt_version,
            token_cap=token_cap,
        )
        experiment_tokens += attempt_tokens
        experiment_calls += 1
        try:
            experiment, _ = parse_llm_json_text(raw_experiment)
        except Exception as exc:
            invalid_delivery("experiment output is not valid JSON", exc)
        if not isinstance(experiment, dict) or not experiment:
            invalid_delivery("has no experiment design")
        review = review_candidate_plan(
            _experimental_plan_payload(experiment),
            agenda=agenda_rule,
            resolver=resolver,
            claim_text=claim_text,
        )
        _fold_resolved_identities(experiment, review)
        if review.ok:
            if attempt > 1:
                print(
                    f"[PAPER_IDEA] Contract satisfied on attempt {attempt}.",
                    flush=True,
                )
            break
        if not review.actionable:
            # Only conditions of the deployment remain. Preflight defers on
            # those and a later pass retries them, so a rewrite buys nothing.
            print(
                "[PAPER_IDEA] Contract blocked only by deployment conditions "
                f"({', '.join(review.codes)}); storing for the deferred retry path.",
                flush=True,
            )
            break
        # Print what is handed back, not only its labels. The loop's whole
        # value is the quality of the instruction it returns, and a log that
        # shows only reason codes cannot tell a loop that is failing to
        # converge from one whose advice was never actionable.
        print(
            f"[PAPER_IDEA] Plan refused on attempt {attempt}/{CONTRACT_ATTEMPTS}:",
            flush=True,
        )
        for item in review.violations:
            print(f"[PAPER_IDEA]   - {item.code}: {item.detail}", flush=True)
    if review is not None and review.actionable:
        # Storing it would spend a candidate slot on a plan no runner can
        # execute and teach the next generation nothing. Refuse, and hand back
        # the grant so the agenda's concurrency slot does not idle out its TTL.
        print(
            f"[PAPER_IDEA] Bounded proposal {idea_id} abandoned after "
            f"{experiment_calls} contract attempts: {', '.join(review.codes)}",
            flush=True,
        )
        # Imported here rather than at module scope: orchestrator.pipeline
        # imports this module's agents, so a top-level import is a cycle.
        from orchestrator.pipeline import log_event

        log_event(
            "warning",
            {
                "step": "proposal_contract_unsatisfied",
                "agenda_id": agenda_id,
                "idea_id": idea_id,
                "resource_grant_id": resource_grant_id,
                "attempts": experiment_calls,
                "reason_codes": list(review.codes),
            },
        )
        _release_abandoned_proposal_grant({"id": resource_grant_id}, agenda_id)
        return []

    generated_awareness = experiment.get("problem_awareness")
    if not isinstance(generated_awareness, dict):
        generated_awareness = {}
    expected_results = experiment.get("expected_results")
    if not isinstance(expected_results, dict):
        expected_results = {}
    awareness = {
        "central_question": generated_awareness.get("central_question")
        or problem["central_question"],
        "motivation": generated_awareness.get("motivation")
        or problem["motivation"],
        "method_answer": generated_awareness.get("method_answer")
        or method.get("mechanism_repair")
        or method.get("one_line", ""),
        "result_claim": generated_awareness.get("result_claim")
        or expected_results.get("solid")
        or problem["result_that_would_change_belief"],
        "falsification_result": generated_awareness.get("falsification_result")
        or method.get("falsification_hook", ""),
    }
    raw_title = experiment.get("paper_title") or f"{method['name']}: {problem['title']}"
    title = normalize_paper_title(
        raw_title,
        method_name=method.get("name"),
        claim=awareness["central_question"],
        context={"full_benchmark_completed": False},
    )
    source_refs = problem.get("source_signal_refs") or {}
    signal_ids = [
        str(ref.get("content_hash"))
        for ref in source_refs.get("signals", [])
        if isinstance(ref, dict) and ref.get("content_hash")
    ]
    plan = _experimental_plan_payload(
        experiment,
        paper_title=title,
        raw_paper_title=raw_title,
        title_source="bounded_proposal_title_policy",
    )
    return [
        {
            "proposal_candidate_id": idea_id,
            "resource_grant_id": resource_grant_id,
            "agenda_id": agenda_id,
            "tier": 2,
            "status": "candidate",
            "title": title,
            "problem_statement": problem["problem_statement"],
            "existing_weakness": problem["current_failure_mode"],
            "proposed_method": json.dumps(method),
            "experimental_plan": json.dumps(plan),
            "related_work_positioning": json.dumps(
                experiment.get("paper_outline", {})
            ),
            "supporting_papers": json.dumps(problem["source_paper_ids"]),
            "source_node_ids": json.dumps(problem["related_node_ids"]),
            "source_paper_ids": json.dumps(problem["source_paper_ids"]),
            "source_signal_ids": json.dumps(signal_ids),
            "source_signal_refs": json.dumps(source_refs),
            "evidence_summary": problem["source_evidence"],
            "mechanism_type": problem["mechanism_type"],
            "problem_awareness": json.dumps(awareness),
            "research_problem_id": problem["research_problem_id"],
            "signal_mix": json.dumps(
                sorted({problem["source_type"], problem["mechanism_type"]})
            ),
            "evidence_packet": build_evidence_packet(
                signal_mix=[problem["source_type"], problem["mechanism_type"]],
                evidence_summary=problem["source_evidence"],
                falsification=method.get("falsification_hook")
                or {"summary": "See the bounded experimental plan."},
                structural_evidence=[problem["formal_statement"]],
                non_numeric_evidence=problem["non_numeric_evidence"],
            ),
            "novelty_status": "unchecked",
            "generation_tokens": method_tokens + experiment_tokens,
            "llm_calls": 1 + experiment_calls,
            "prompt_version": prompt_version,
            "model_version": str(
                experiment_route.get("model") or method_route.get("model") or ""
            ),
            "proposer_route": experiment_route or method_route,
        }
    ]


def _proposal_problem_is_over_budget(agenda_id: int, problem_id: int) -> bool:
    """Thin wrapper so the grant-time ceiling stays the single definition."""

    # Imported lazily: meta_harness.repository imports this module's agents.
    from meta_harness.repository import proposal_problem_is_over_budget

    return proposal_problem_is_over_budget(agenda_id, problem_id)


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
        # Archiving a spent candidate frees its problem, which is the point --
        # but the problem is only worth re-seeding if it can still be funded.
        # Its undelivered spend is charged to the problem, so a problem already
        # over the ceiling would seed a fresh row, get refused at grant time,
        # be archived, and seed again on the next pass: no tokens, but one dead
        # deep_insights row every ten minutes forever. Skip it the same way a
        # spent candidate is skipped.
        if _proposal_problem_is_over_budget(agenda_id, problem_id):
            raise ProposalProblemUnavailable(
                f"research problem {problem_id} has spent its proposal budget "
                f"share without delivering a candidate"
            )
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
                # Retire the spent holder instead of only reporting it. It sits
                # at proposal_pending with a terminal outcome, so it owns the
                # (agenda, problem) key without being usable, and the problem
                # can never seed another candidate for as long as it holds it.
                # Nothing performed this transition: the terminal outcome was
                # written and no one read it. Idea 110 sterilised problem 9 for
                # a full day that way, while the portfolio kept re-funding it,
                # and a funded proposal preempts discovery -- so one dead
                # candidate starved every other agenda's rotation. It cost five
                # manual grant expiries in nine hours (2026-08-19/20).
                #
                # The run and its evidence are untouched; only the pre-idea
                # placeholder is archived, which frees the key and stops the
                # job from holding a discovery preemption slot.
                spent_id = int(holder["id"])
                try:
                    db.execute(
                        "UPDATE deep_insights SET status='archived'"
                        " WHERE id=? AND status='proposal_pending'",
                        (spent_id,),
                    )
                    db.execute(
                        "UPDATE auto_research_jobs"
                        " SET status='failed', stage='proposal_unrealized',"
                        "     last_note=?, updated_at=CURRENT_TIMESTAMP"
                        " WHERE deep_insight_id=? AND status='deferred'",
                        (
                            f"retired: spent proposal candidate holding research "
                            f"problem {problem_id} (outcome={holder.get('outcome')})",
                            spent_id,
                        ),
                    )
                    db.commit()
                except Exception:
                    db.rollback()
                raise ProposalProblemUnavailable(
                    f"research problem {problem_id} was held by spent proposal "
                    f"candidate {spent_id} "
                    f"(outcome={holder.get('outcome')}); retired, "
                    f"the problem is free for the next pass"
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


from agents.candidate_contract import (  # noqa: E402
    RepositoryResolver,
    render_violations,
    review_candidate_plan,
)

CONTRACT_ATTEMPTS = 3


# A bounded call reserves prompt bytes plus framing plus output against this
# ceiling, and the design prompt carries the capability envelope, the prior
# refusals for the problem and the contract violations being repaired. Sized
# below what the prompt needs, the call cannot be made at all: "prompt cannot
# fit inside total_token_cap".
PROPOSAL_CALL_TOKEN_CAP = 16_000

# One method call plus every contract attempt. The reservation is what the
# grant has to be able to hold at once; settlement is on tokens actually used,
# which measured 4-5k per call, so a larger cap buys room rather than spend.
PROPOSAL_GRANT_TOKEN_CAP = PROPOSAL_CALL_TOKEN_CAP * (CONTRACT_ATTEMPTS + 1)


def _proposal_call_token_cap(grant_token_cap: int) -> int:
    """Per-call ceiling: the whole cap unless the grant is smaller than one call.

    Dividing the grant by the number of attempts is the wrong shape. The cap
    is not a share of a budget, it is the room one call needs for its prompt
    and its answer; what has to divide is the grant, and that is
    PROPOSAL_GRANT_TOKEN_CAP's job.
    """
    cap = int(grant_token_cap or 0)
    if cap <= 0:
        return PROPOSAL_CALL_TOKEN_CAP
    return max(1, min(PROPOSAL_CALL_TOKEN_CAP, cap))


def _release_abandoned_proposal_grant(proposal_grant: dict, agenda_id: int) -> bool:
    """Give back the slot a candidate no longer needs.

    Nothing settles a proposal grant except a stored insight. A candidate
    abandoned for an unsatisfiable contract would therefore hold its grant --
    and the agenda's only concurrency slot -- for the rest of a four-hour TTL,
    which is the stall ``expire_grant_now`` was written for (grants 62/67/68,
    2026-08-17). Settled spend stays settled; only the unspent remainder and
    the slot come back, and the candidate parks for the standard requeue.
    """
    try:
        from meta_harness.repository import MetaHarnessRepository

        return MetaHarnessRepository().expire_grant_now(
            int(proposal_grant["id"]),
            agenda_id=int(agenda_id),
            reason="proposal_contract_unsatisfied",
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[PAPER_IDEA] Could not release grant "
            f"{proposal_grant.get('id')}: {exc}",
            flush=True,
        )
        return False


def _experimental_plan_payload(
    result3: dict,
    *,
    paper_title: str = "",
    raw_paper_title: str = "",
    title_source: str = "paper_idea_title_policy",
) -> dict:
    """The plan exactly as it will be stored.

    One definition, so the review that runs before the row is written judges
    the same object preflight reads afterwards. When these were assembled in
    two places they could disagree, and a disagreement here means a candidate
    passes review and is refused by the gate for a reason nobody was told.
    """
    return {
        "baselines": result3.get("baselines", []),
        "datasets": result3.get("datasets", []),
        "metrics": result3.get("metrics", {}),
        "ablations": result3.get("ablations", []),
        "expected_results": result3.get("expected_results", {}),
        "compute_budget": result3.get("compute_budget", {}),
        "execution_requirements": result3.get("execution_requirements", {}),
        "risks": result3.get("risks", []),
        "paper_title": paper_title,
        "raw_paper_title": raw_paper_title,
        "title_source": title_source,
    }


def _agenda_scope_rule(agenda_id: int):
    """The agenda whose keyword rule the topic gate will apply, or None.

    An early warning, not the ruling: the gate still applies it to the stored
    row. Losing the warning costs one regeneration; refusing to propose because
    the agenda row could not be read would cost the whole pass.
    """
    try:
        from agents.agenda_loader import get_agenda

        return get_agenda(int(agenda_id))
    except Exception as exc:  # noqa: BLE001
        print(f"[PAPER_IDEA] Agenda scope rule unavailable ({exc})", flush=True)
        return None


def _fold_resolved_identities(experiment: dict, review) -> None:
    """Keep the repository ids the hub resolved, so nothing re-derives them."""
    requirements = review.plan.get("execution_requirements")
    if isinstance(requirements, dict) and requirements:
        experiment["execution_requirements"] = requirements
    for field_name in ("datasets", "baselines"):
        if field_name in review.plan:
            experiment[field_name] = review.plan[field_name]


def _design_experiment_within_contract(
    *,
    problem: dict,
    method: dict,
    agenda_id: int,
    agenda,
    proposal_candidate_id: int,
    proposal_grant: dict,
    attempts,
    prompt_version: str,
    proposal_token_cap: int,
    resolver,
):
    """Ask for an executable plan, and say what was wrong until it is one.

    Before this loop existed the design call ran once and whatever came back
    was stored. Preflight then judged it -- after the row existed, after it had
    counted against the agenda's candidate quota, after the grant was spent --
    and nothing carried the verdict back to the generator, so the next
    candidate repeated the same defect. Of the 35 candidates generated for
    agendas 16/17/18 between 2026-08-25 and 08-27, two were executable.

    Attempts come from the grant's own ledger, so the number of tries is
    bounded by the same budget everything else is, and an exhausted candidate
    is refused rather than stored unexecutable.
    """
    from meta_harness.grant_usage import GrantUsageError

    base_prompt = _build_experiment_prompt(problem, method)
    claim_text = " ".join(
        str(value or "")
        for value in (
            problem.get("title"),
            problem.get("problem_statement") or problem.get("formal_statement"),
            json.dumps(method, ensure_ascii=False),
        )
    )
    base_key = f"proposal-experiment:{agenda_id}:{proposal_candidate_id}"
    result3: dict = {}
    experiment_route: dict = {}
    review = None
    tokens = 0
    calls = 0

    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        operation = (
            "proposal_experiment_design"
            if attempt == 1
            else f"proposal_experiment_design:repair{attempt - 1}"
        )
        try:
            attempt_key = attempts.next_attempt_key(
                base_key, max_attempts=CONTRACT_ATTEMPTS
            )
        except GrantUsageError as exc:
            print(
                f"[PAPER_IDEA] Contract loop out of attempts after {attempt - 1}: {exc}",
                flush=True,
            )
            break

        prompt = base_prompt
        if review is not None:
            prompt = f"{base_prompt}\n\n{render_violations(review)}"
        print(
            f"[PAPER_IDEA] Call 3/3 attempt {attempt}/{CONTRACT_ATTEMPTS}: "
            f"designing experiments for '{method.get('name', '')}'...",
            flush=True,
        )
        try:
            result3, tokens3, experiment_route = call_llm_json_for_role(
                EXPERIMENT_DESIGN_SYSTEM,
                prompt,
                agenda_id=agenda_id,
                idea_id=proposal_candidate_id,
                role="proposer",
                stage="proposal",
                resource_grant_id=int(proposal_grant["id"]),
                operation=operation,
                idempotency_key=attempt_key,
                prompt_version=prompt_version,
                max_tokens=proposal_token_cap,
            )
            tokens += tokens3
            calls += 1
        except Exception as exc:
            if _llm_temporarily_unavailable(exc):
                print(
                    f"[PAPER_IDEA] Experiment design skipped: LLM unavailable ({exc})",
                    flush=True,
                )
            else:
                print(f"[PAPER_IDEA] Experiment design failed: {exc}", flush=True)
            return {}, experiment_route, tokens, calls, review

        if not isinstance(result3, dict):
            result3 = {}
        review = review_candidate_plan(
            _experimental_plan_payload(result3),
            agenda=agenda,
            resolver=resolver,
            claim_text=claim_text,
        )
        # Carry the identities the hub resolved for us into what gets stored,
        # so the correction is not re-derived (or lost) downstream.
        requirements = review.plan.get("execution_requirements")
        if isinstance(requirements, dict) and requirements:
            result3["execution_requirements"] = requirements
        for field_name in ("datasets", "baselines"):
            if field_name in review.plan:
                result3[field_name] = review.plan[field_name]
        if review.ok:
            if attempt > 1:
                print(
                    f"[PAPER_IDEA] Contract satisfied on attempt {attempt}.",
                    flush=True,
                )
            return result3, experiment_route, tokens, calls, review
        if not review.actionable:
            # Every remaining objection is a condition of the deployment. A
            # rewritten plan cannot change the weather, and preflight defers
            # (not refuses) on these, so store it and let the retry path run.
            print(
                "[PAPER_IDEA] Contract blocked only by deployment conditions "
                f"({', '.join(review.codes)}); storing for the deferred retry path.",
                flush=True,
            )
            return result3, experiment_route, tokens, calls, None
        print(
            f"[PAPER_IDEA] Plan refused ({', '.join(review.codes)}); "
            "returning the reasons to the generator.",
            flush=True,
        )

    return result3, experiment_route, tokens, calls, review


def discover_paper_ideas(
    max_problems: int = 8,
    max_papers: int | None = None,
    *,
    agenda_id: int,
    tier2_plateau_limit: int = 20,
    tier2_limitation_nodes: int = 15,
    proposal_job_id: int | None = None,
    proposal_candidate_id: int | None = None,
    proposal_grant_id: int | None = None,
) -> list[dict]:
    """Run the 3-stage paper idea discovery pipeline.

    Returns list of deep_insight dicts ready for storage.
    If max_papers is None, every sharpened problem (up to max_problems) is expanded.
    """
    if max_papers is None:
        max_papers = max_problems
    exact_values = (proposal_job_id, proposal_candidate_id, proposal_grant_id)
    if any(value is not None for value in exact_values):
        if any(value is None for value in exact_values):
            raise ValueError(
                "exact proposal requires job, candidate, and grant identities"
            )
        return _discover_exact_bounded_proposal(
            job_id=int(proposal_job_id),
            agenda_id=int(agenda_id),
            idea_id=int(proposal_candidate_id),
            resource_grant_id=int(proposal_grant_id),
        )

    print(f"[PAPER_IDEA] Starting Tier 2 discovery...", flush=True)
    total_tokens = 0
    total_calls = 0

    # Read once and shared by every candidate this pass produces: the agenda
    # carries the keyword rule the topic gate will apply, and the resolver
    # caches repository identities so a pass asking about the same dozen
    # repositories pays for each of them once.
    from agents.agenda_loader import get_agenda
    from orchestrator.pipeline import log_event

    try:
        agenda = get_agenda(int(agenda_id))
    except Exception as exc:  # noqa: BLE001
        # The keyword rule is an early warning, not the ruling: the topic gate
        # still applies it to the stored row. Losing the warning costs one
        # regeneration; refusing to propose because the agenda row could not be
        # read would cost the whole pass.
        print(f"[PAPER_IDEA] Agenda scope rule unavailable ({exc})", flush=True)
        agenda = None
    contract_resolver = RepositoryResolver()

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
            _proposal_call_token_cap(int(proposal_grant.get("token_cap") or 0)),
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
            # The experiment key is allocated per attempt inside the contract
            # loop. Taking one here spent an attempt even when the method call
            # returned nothing usable and the design call never happened.
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
            # Text-first so a malformed response can be dumped verbatim: the
            # invention call failed 8+ times on 2026-08-17 with only a two-key
            # fragment surviving the parse, and the raw shape was invisible.
            raw2, tokens2, method_route = call_llm_for_role(
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

        result2, parse_how = parse_llm_json_text(raw2)
        method = _extract_method_payload(result2)
        if not method.get("name"):
            shape = (
                sorted(result2.keys()) if isinstance(result2, dict)
                else type(result2).__name__
            )
            dump_dir = Path.home() / "deepgraph-reports" / "malformed_llm"
            try:
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / f"method_invention_{proposal_candidate_id}_{int(time.time())}.txt"
                dump_path.write_text(raw2, encoding="utf-8")
            except OSError:
                dump_path = None
            print(
                f"[PAPER_IDEA] No method produced for '{title[:50]}'"
                f" (parsed via {parse_how}, shape: {shape}, raw dumped: {dump_path})",
                flush=True,
            )
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

        # Stage 3: Experimental Design, bounded by the runner contract
        (
            result3,
            experiment_route,
            tokens3,
            calls3,
            contract_review,
        ) = _design_experiment_within_contract(
            problem=problem,
            method=method,
            agenda_id=agenda_id,
            agenda=agenda,
            proposal_candidate_id=proposal_candidate_id,
            proposal_grant=proposal_grant,
            attempts=attempts,
            prompt_version=prompt_version,
            proposal_token_cap=proposal_token_cap,
            resolver=contract_resolver,
        )
        total_tokens += tokens3
        total_calls += calls3
        if contract_review is not None and not contract_review.ok:
            # Storing it would spend a candidate slot on something no runner
            # can execute, and leave the next generation no wiser. Refusing
            # here keeps the quota for plans that can be measured.
            print(
                "[PAPER_IDEA] Candidate abandoned after "
                f"{CONTRACT_ATTEMPTS} contract attempts: "
                f"{', '.join(contract_review.codes) or 'no plan returned'}",
                flush=True,
            )
            log_event(
                "warning",
                {
                    "step": "proposal_contract_unsatisfied",
                    "agenda_id": agenda_id,
                    "idea_id": proposal_candidate_id,
                    "attempts": CONTRACT_ATTEMPTS,
                    "reason_codes": list(contract_review.codes),
                },
            )
            _release_abandoned_proposal_grant(proposal_grant, agenda_id)
            continue

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
            "experimental_plan": json.dumps(
                _experimental_plan_payload(
                    result3,
                    paper_title=normalized_paper_title,
                    raw_paper_title=raw_paper_title,
                )
            ),
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
