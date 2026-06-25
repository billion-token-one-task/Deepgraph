"""Problem-first benchmark design contracts for experiment forge.

The generated runner can execute a few generic text/code benchmarks, but that
does not mean those benchmarks are evidence for every scientific claim. This
module keeps benchmark choice tied to the research problem before the forge
creates runnable experiments.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agents.llm_client import call_llm_json
from config import BENCHMARK_DESIGN_LLM_ENABLED, BENCHMARK_DESIGN_LLM_REQUIRED


DESIGN_STATUS_RESOLVED = "resolved"
DESIGN_STATUS_NEEDS_LITERATURE_REVIEW = "literature_review_required"
DESIGN_STATUS_BLOCKED = "blocked"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _canon(value: Any) -> str:
    return "".join(ch for ch in _text(value).lower() if ch.isalnum())


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, "", "unknown"):
        return []
    return [value]


def _named_values(rows: Any, *keys: str) -> list[str]:
    if not isinstance(rows, list):
        rows = [rows] if rows not in (None, "", "unknown") else []
    values: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            for key in keys:
                text = _text(row.get(key))
                if text:
                    values.append(text)
                    break
        else:
            text = _text(row)
            if text:
                values.append(text)
    return _unique(values)


DOMAIN_RULES: list[dict[str, Any]] = [
    {
        "domain": "legal_nlp",
        "task_family": "legal_text_classification",
        "keywords": (
            "foia", "legal", "privilege", "contractnli", "contract nli",
            "casehold", "ledgar", "law", "regulatory", "legalbench",
        ),
        "candidate_benchmarks": [
            {"name": "LegalBench", "task_type": "legal_nlp", "requires_harness": True, "benchmark_axis": "legal_task_suite"},
            {"name": "ContractNLI", "task_type": "legal_nli", "requires_harness": True, "benchmark_axis": "contract_entailment"},
            {"name": "LEDGAR", "task_type": "legal_text_classification", "requires_harness": True, "benchmark_axis": "contract_clause_classification"},
            {"name": "CaseHOLD", "task_type": "legal_multiple_choice", "requires_harness": True, "benchmark_axis": "legal_reasoning"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "majority/frequency baseline",
            "TF-IDF or bag-of-ngrams linear classifier",
            "Legal-BERT or DeBERTa classifier",
            "calibrated abstention/selective classifier",
        ],
        "metrics": ["macro_f1", "selective_risk", "coverage", "calibration_error"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "vision_language_relations",
        "task_family": "visual_relation_reasoning",
        "keywords": (
            "gqa", "visual relation", "scene graph", "visual genome", "vqa",
            "clip", "llava", "image-text", "vision-language", "multimodal",
        ),
        "candidate_benchmarks": [
            {"name": "GQA", "task_type": "visual_question_answering", "requires_harness": True, "benchmark_axis": "compositional_vqa"},
            {"name": "Visual Genome", "task_type": "scene_graph_relation", "requires_harness": True, "benchmark_axis": "scene_graph_relation"},
            {"name": "VQA v2", "task_type": "visual_question_answering", "requires_harness": True, "benchmark_axis": "general_vqa"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "symbolic/scene-graph rule baseline",
            "CLIP/OpenCLIP zero-shot scorer",
            "LLaVA or VLM direct-answer baseline",
            "matched Euclidean relation head",
        ],
        "metrics": ["macro_f1", "accuracy", "depth_stratified_accuracy"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "physical_spatial_reasoning",
        "task_family": "visual_physical_reasoning",
        "keywords": (
            "causal scene", "scene competence", "physical-spatial", "physical spatial",
            "physical reasoning", "spatial reasoning", "support/contact", "support contact",
            "action-substitution", "action substitution", "intervention-identifiability",
            "intervention identifiability", "clevrer", "clevr", "causalphys", "gqa",
        ),
        "candidate_benchmarks": [
            {"name": "CLEVRER", "task_type": "visual_physical_reasoning", "requires_harness": True, "benchmark_axis": "counterfactual_physical_reasoning"},
            {"name": "CLEVR", "task_type": "compositional_visual_reasoning", "requires_harness": True, "benchmark_axis": "spatial_compositional_reasoning"},
            {"name": "GQA", "task_type": "visual_question_answering", "requires_harness": True, "benchmark_axis": "scene_graph_spatial_reasoning"},
            {"name": "CausalPhys-style intervention set", "task_type": "visual_physical_reasoning", "requires_harness": True, "benchmark_axis": "causal_intervention_pairs"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "direct VQA/VLM answer baseline",
            "scene-graph symbolic reasoning baseline",
            "counterfactual physical reasoning baseline",
            "matched intervention-randomization baseline",
        ],
        "metrics": ["accuracy", "counterfactual_consistency", "spatial_relation_accuracy", "calibration_error"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "robotics_policy",
        "task_family": "robotics_control",
        "keywords": (
            "diffusion policy", "robomimic", "d4rl", "adroit", "libero",
            "robot", "manipulation", "sim2real", "carla", "bench2drive",
        ),
        "candidate_benchmarks": [
            {"name": "Robomimic", "task_type": "robot_manipulation", "requires_harness": True, "benchmark_axis": "imitation_manipulation"},
            {"name": "D4RL / Adroit", "task_type": "offline_rl", "requires_harness": True, "benchmark_axis": "offline_rl_control"},
            {"name": "LIBERO", "task_type": "robot_manipulation", "requires_harness": True, "benchmark_axis": "language_conditioned_manipulation"},
            {"name": "Bench2Drive", "task_type": "autonomous_driving", "requires_harness": True, "benchmark_axis": "driving"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "behavior cloning policy",
            "Diffusion Policy official baseline",
            "SAC/TD3 or offline RL baseline",
            "DPPO-style finetuning baseline",
        ],
        "metrics": ["success_rate", "normalized_return", "collision_rate", "trajectory_cost"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "agent_workflow_optimization",
        "task_family": "agent_task_automation",
        "keywords": (
            "self-evolving agent", "self evolving agent", "workflow self-optimization",
            "workflow self optimization", "agent workflow", "executable policy code",
            "typed stochastic program", "tool agent", "agentbench", "webarena",
            "mind2web", "tau-bench", "workarena", "agentdojo",
        ),
        "candidate_benchmarks": [
            {"name": "AgentBench", "task_type": "agent_task_evaluation", "requires_harness": True, "benchmark_axis": "multi_environment_agent_tasks"},
            {"name": "WebArena", "task_type": "web_agent_evaluation", "requires_harness": True, "benchmark_axis": "web_workflow_execution"},
            {"name": "Mind2Web", "task_type": "web_agent_evaluation", "requires_harness": True, "benchmark_axis": "web_task_generalization"},
            {"name": "tau-bench", "task_type": "tool_agent_evaluation", "requires_harness": True, "benchmark_axis": "tool_use_transactional_workflows"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "direct ReAct/tool-use baseline",
            "planner-executor agent baseline",
            "workflow/code-edit self-improvement baseline",
            "budget-matched random or static workflow baseline",
        ],
        "metrics": ["task_success_rate", "tool_error_rate", "cost", "latency", "safety_violation_rate"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "3d_scene_rendering",
        "task_family": "novel_view_synthesis",
        "keywords": (
            "gaussian splatting", "splatting", "nerf", "neural radiance",
            "novel-view", "novel view", "sparse-view", "sparse view",
            "radiance field", "view synthesis", "llff", "dtu", "mip-nerf",
            "tanks and temples", "zip-splat", "zipsplat", "pixelSplat", "mvsplat",
        ),
        "candidate_benchmarks": [
            {"name": "LLFF", "task_type": "novel_view_synthesis", "requires_harness": True, "benchmark_axis": "forward_facing_sparse_view"},
            {"name": "DTU", "task_type": "novel_view_synthesis", "requires_harness": True, "benchmark_axis": "multi_view_reconstruction"},
            {"name": "Mip-NeRF 360", "task_type": "novel_view_synthesis", "requires_harness": True, "benchmark_axis": "unbounded_scene_rendering"},
            {"name": "Tanks and Temples", "task_type": "novel_view_synthesis", "requires_harness": True, "benchmark_axis": "real_scene_geometry"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "3D Gaussian Splatting baseline",
            "pixelSplat/MVSplat feed-forward baseline",
            "NeRF/RegNeRF/FreeNeRF sparse-view baseline",
            "ZipSplat or compression baseline",
        ],
        "metrics": ["psnr", "ssim", "lpips", "render_fps", "model_size"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "molecular_equivariant_dynamics",
        "task_family": "molecular_conformation_or_equivariant_dynamics",
        "keywords": (
            "geom-qm9", "geom drugs", "geom-drugs", "conformation", "conformer",
            "molecular", "molecule", "equivariant message-passing", "equivariant message passing",
            "e(n)-equivariant", "e3 equivariant", "se(3)", "egnn", "enbp",
            "coordinate denoising", "configuration orbit", "orbit space", "physical dynamics",
        ),
        "candidate_benchmarks": [
            {"name": "GEOM-QM9", "task_type": "molecular_conformation", "requires_harness": True, "benchmark_axis": "small_molecule_conformation_generation"},
            {"name": "GEOM-Drugs", "task_type": "molecular_conformation", "requires_harness": True, "benchmark_axis": "drug_like_conformation_generation"},
            {"name": "MD17", "task_type": "molecular_dynamics", "requires_harness": True, "benchmark_axis": "force_field_dynamics"},
            {"name": "N-body / charged particles equivariance benchmark", "task_type": "equivariant_dynamics", "requires_harness": True, "benchmark_axis": "controlled_physical_dynamics"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "EGNN baseline",
            "ENBP or belief-propagation-style equivariant baseline",
            "non-equivariant GCN/GAT with SE(3) augmentation",
            "DeepSets or relative-coordinate MLP baseline",
        ],
        "metrics": ["coverage", "matching_rmsd", "mean_rmsd", "equivariance_error", "energy_or_force_mae"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "text_to_sql",
        "task_family": "semantic_parsing",
        "keywords": (
            "text-to-sql", "text to sql", "spider", "sql", "schema linking",
            "database question answering",
        ),
        "candidate_benchmarks": [
            {"name": "Spider", "task_type": "text_to_sql", "requires_harness": True, "benchmark_axis": "cross_domain_sql"},
            {"name": "BIRD", "task_type": "text_to_sql", "requires_harness": True, "benchmark_axis": "realistic_database_sql"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "direct generation baseline",
            "schema-linking baseline",
            "execution-guided decoding baseline",
            "strong open text-to-SQL model baseline",
        ],
        "metrics": ["exact_match", "execution_accuracy", "test_suite_accuracy"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "formal_code_reasoning",
        "task_family": "formal_verification_or_code_generation",
        "keywords": (
            "lean", "mathlib", "formal proof", "theorem", "proof-state",
            "verifier", "code reasoning", "tool-augmented verification", "program repair",
        ),
        "candidate_benchmarks": [
            {"name": "miniF2F", "task_type": "formal_proving", "requires_harness": True, "benchmark_axis": "formal_math"},
            {"name": "LeanDojo", "task_type": "formal_proving", "requires_harness": True, "benchmark_axis": "lean_environment"},
            {"name": "MBPP", "task_type": "code_generation", "hf_dataset": "google-research-datasets/mbpp", "requires_harness": False, "benchmark_axis": "code_generation"},
            {"name": "HumanEval", "task_type": "code_generation", "requires_harness": True, "benchmark_axis": "code_generation"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "direct generation baseline",
            "self-repair with verifier feedback",
            "best-of-n sampling",
            "known prover/code model baseline",
        ],
        "metrics": ["pass_at_1", "pass_at_k", "proof_success_rate", "repair_success_rate"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "llm_safety_refusal",
        "task_family": "safety_refusal_evaluation",
        "keywords": (
            "refusal", "jailbreak", "instruction pressure", "safety", "harmbench",
            "advbench", "policy manifold", "unsafe",
            "agentdojo", "prompt injection", "tool-use", "tool use",
            "authorization", "access refusal", "tool_execute",
        ),
        "candidate_benchmarks": [
            {"name": "AgentDojo", "task_type": "agent_tool_safety", "requires_harness": True, "benchmark_axis": "tool_use_prompt_injection"},
            {"name": "HarmBench", "task_type": "safety_evaluation", "requires_harness": True, "benchmark_axis": "harmful_behavior_refusal"},
            {"name": "AdvBench", "task_type": "safety_evaluation", "requires_harness": True, "benchmark_axis": "jailbreak_refusal"},
            {"name": "ToxiGen", "task_type": "toxicity_evaluation", "requires_harness": True, "benchmark_axis": "toxicity"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "direct instruction baseline",
            "standard safety/refusal classifier",
            "jailbreak prompt baseline",
            "policy-compliant refusal baseline",
        ],
        "metrics": ["attack_success_rate", "refusal_precision", "refusal_recall", "false_refusal_rate"],
        "allow_generic_gsm8k": False,
    },
    {
        "domain": "math_reasoning_prm",
        "task_family": "math_reasoning",
        "keywords": (
            "gsm8k", "math", "process reward", "prm", "chain-of-thought",
            "cot", "reasoning trajectory", "bellman", "distillation credit",
        ),
        "candidate_benchmarks": [
            {"name": "GSM8K", "task_type": "math_qa", "hf_dataset": "openai/gsm8k", "requires_harness": False, "benchmark_axis": "grade_school_math"},
            {"name": "MATH", "task_type": "math_qa", "requires_harness": True, "benchmark_axis": "competition_math"},
            {"name": "PRM800K", "task_type": "process_reward_evaluation", "requires_harness": True, "benchmark_axis": "process_reward_annotations"},
            {"name": "ProcessBench", "task_type": "process_reward_evaluation", "requires_harness": True, "benchmark_axis": "process_error_localization"},
        ],
        "minimum_benchmark_count": 2,
        "required_benchmark_axis_markers": ("process_reward", "process supervision", "prm800k", "processbench"),
        "baseline_families": [
            "direct answer baseline",
            "chain-of-thought baseline",
            "self-consistency baseline",
            "process reward model baseline",
            "compute-matched routing baseline",
        ],
        "metrics": ["exact_match", "numeric_accuracy", "reward"],
        "allow_generic_gsm8k": True,
    },
    {
        "domain": "long_context_memory",
        "task_family": "long_context_retrieval",
        "keywords": (
            "long-context", "long context", "memory", "retrieval", "budgeted",
            "longbench", "needle", "multihop", "evidence filtering",
        ),
        "candidate_benchmarks": [
            {"name": "LongMemEval", "task_type": "long_context_memory", "requires_harness": True, "benchmark_axis": "long_term_memory_qa"},
            {"name": "LongBench", "task_type": "long_context_qa", "requires_harness": True, "benchmark_axis": "long_context_suite"},
            {"name": "MuSiQue-Ans", "task_type": "multihop_qa", "hf_dataset": "dgslibisey/MuSiQue", "requires_harness": False, "benchmark_axis": "multihop_qa"},
            {"name": "2WikiMultihopQA", "task_type": "multihop_qa", "hf_dataset": "xanhho/2WikiMultihopQA", "requires_harness": False, "benchmark_axis": "multihop_qa"},
        ],
        "minimum_benchmark_count": 2,
        "baseline_families": [
            "full-context baseline",
            "retrieval-only baseline",
            "recency/window baseline",
            "budget-matched evidence selector",
        ],
        "metrics": ["exact_match", "f1", "tokens", "latency", "cost_adjusted_accuracy"],
        "allow_generic_gsm8k": False,
    },
]


GENERIC_GSM8K_NAMES = {"gsm8k", "openaigsm8k"}


def _corpus(parsed: Mapping[str, Any], method: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    return " ".join(
        [
            _text(parsed.get("title")),
            _text(parsed.get("problem_statement")),
            _text(parsed.get("existing_weakness")),
            _text(parsed.get("formal_structure")),
            _text(parsed.get("transformation")),
            _text(parsed.get("mechanism_type")),
            _text(method.get("name")),
            _text(method.get("type")),
            _text(method.get("one_line")),
            _text(method.get("definition")),
            json.dumps(plan.get("datasets") or [], ensure_ascii=False),
            json.dumps(plan.get("benchmark_targets") or [], ensure_ascii=False),
            json.dumps(plan.get("metrics") or {}, ensure_ascii=False),
        ]
    ).lower()


def infer_benchmark_domain(
    parsed: Mapping[str, Any],
    method: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    method = method or {}
    plan = plan or {}
    corpus = _corpus(parsed, method, plan)
    scored: list[tuple[int, dict[str, Any]]] = []
    for rule in DOMAIN_RULES:
        score = sum(1 for keyword in rule["keywords"] if keyword in corpus)
        if score:
            scored.append((score, rule))
    if not scored:
        return {
            "domain": "unknown",
            "task_family": "unknown",
            "candidate_benchmarks": [],
            "baseline_families": [],
            "metrics": [],
            "allow_generic_gsm8k": False,
        }
    scored.sort(key=lambda item: item[0], reverse=True)
    return dict(scored[0][1])


def _existing_benchmark_targets(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = plan.get("benchmark_targets") if isinstance(plan.get("benchmark_targets"), list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            out.append(dict(row))
        elif _text(row):
            out.append({"name": _text(row)})
    if out:
        return out
    for name in _named_values(plan.get("datasets"), "name", "dataset", "hf_dataset"):
        out.append({"name": name})
    return out


def _is_gsm8k_target(target: Mapping[str, Any]) -> bool:
    values = [
        target.get("name"),
        target.get("hf_dataset"),
        target.get("dataset"),
        target.get("dataset_id"),
    ]
    return any(_canon(value) in GENERIC_GSM8K_NAMES for value in values)


def _target_matches_rule(target: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    names = [_canon(target.get("name")), _canon(target.get("hf_dataset")), _canon(target.get("dataset"))]
    for candidate in rule.get("candidate_benchmarks") or []:
        candidate_names = [
            _canon(candidate.get("name")),
            _canon(candidate.get("hf_dataset")),
            _canon(candidate.get("dataset")),
        ]
        if any(name and cand and (name == cand or name in cand or cand in name) for name in names for cand in candidate_names):
            return True
    return False


def _candidate_literature_sources(candidate: Mapping[str, Any]) -> list[Any]:
    sources: list[Any] = []
    for key in (
        "literature_sources",
        "benchmark_sources",
        "paper_sources",
        "source_papers",
        "citations",
    ):
        sources.extend(_as_list(candidate.get(key)))
    for key in (
        "official_url",
        "source_url",
        "paper_url",
        "benchmark_paper_url",
        "dataset_url",
        "url",
    ):
        if _text(candidate.get(key)):
            sources.append({"url": _text(candidate.get(key)), "role": key})
    for key in ("paper", "paper_title", "benchmark_paper", "citation", "official_source"):
        if _text(candidate.get(key)):
            sources.append({"title": _text(candidate.get(key)), "role": key})
    return sources


def _candidate_has_literature_evidence(candidate: Mapping[str, Any]) -> bool:
    for source in _candidate_literature_sources(candidate):
        if isinstance(source, Mapping):
            if _text(source.get("title") or source.get("url") or source.get("doi") or source.get("arxiv")):
                return True
        elif _text(source):
            return True
    return False


def _benchmark_set_rationale(llm_design: Mapping[str, Any] | None) -> str:
    if not isinstance(llm_design, Mapping):
        return ""
    for key in ("benchmark_set_rationale", "dataset_count_rationale", "selection_rationale", "coverage_rationale"):
        text = _text(llm_design.get(key))
        if text:
            return text
    return ""


def _minimum_benchmark_count(rule: Mapping[str, Any], llm_design: Mapping[str, Any] | None) -> int:
    values = []
    if isinstance(llm_design, Mapping):
        values.extend([llm_design.get("min_required_benchmarks"), llm_design.get("minimum_benchmark_count")])
    values.append(rule.get("minimum_benchmark_count"))
    counts: list[int] = []
    for value in values:
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts.append(count)
    return max(counts) if counts else 2


def _has_multi_suite_justification(candidate: Mapping[str, Any], rationale: str) -> bool:
    del rationale  # Text-only claims are too easy to hallucinate; require explicit axes.
    explicit_axes: list[str] = []
    for key in (
        "subsets",
        "subtasks",
        "benchmark_subsets",
        "coverage_axes",
        "suite_tasks",
        "official_subsets",
    ):
        explicit_axes.extend(_text(item) for item in _as_list(candidate.get(key)) if _text(item))
    return bool(
        len(_unique(explicit_axes)) >= 2
        and (candidate.get("benchmark_family_contains_subsets") or candidate.get("multi_subset_protocol"))
    )


def _candidate_axis_text(candidate: Mapping[str, Any]) -> str:
    parts = [
        _text(candidate.get("name")),
        _text(candidate.get("task_type")),
        _text(candidate.get("benchmark_axis")),
        _text(candidate.get("benchmark_family")),
        _text(candidate.get("why")),
        _text(candidate.get("official_url") or candidate.get("source_url")),
    ]
    for key in ("literature_sources", "benchmark_sources", "subsets", "subtasks", "coverage_axes"):
        for item in _as_list(candidate.get(key)):
            if isinstance(item, Mapping):
                parts.extend(_text(item.get(k)) for k in ("title", "role", "url") if _text(item.get(k)))
            else:
                parts.append(_text(item))
    return " ".join(part for part in parts if part).lower()


def _llm_design_enabled() -> bool:
    return bool(BENCHMARK_DESIGN_LLM_ENABLED)


BENCHMARK_DESIGN_SYSTEM = """You are DeepGraph's benchmark design reviewer.

Given a research idea and proposed method, choose the experiment benchmarks,
baselines, metrics, and ablations by reasoning from relevant domain literature.
Do not choose a benchmark merely because it is easy to load. If the correct
benchmark requires a custom harness, say so. Return one JSON object:
{
  "status": "resolved|literature_review_required|blocked",
  "domain": "...",
  "task_family": "...",
  "domain_literature_rationale": ["specific benchmark papers or benchmark families checked"],
  "benchmark_set_rationale": "why this number of datasets and these axes are sufficient for the claim",
  "min_required_benchmarks": 2,
  "candidate_benchmarks": [
    {
      "name": "...",
      "task_type": "...",
      "benchmark_axis": "what claim axis this benchmark tests",
      "hf_dataset": "",
      "official_url": "",
      "requires_harness": true,
      "why": "...",
      "literature_sources": [
        {"title": "paper or benchmark name", "year": 2024, "url": "...", "role": "introduced_or_standardized_benchmark"}
      ]
    }
  ],
  "required_baselines": [{"name": "...", "role": "simple|strong|compute_matched|diagnostic", "why": "..."}],
  "primary_metric": {"name": "...", "direction": "higher|lower", "why": "..."},
  "required_ablations": [{"name": "...", "why": "..."}],
  "blockers": ["..."]
}

Rules:
- GSM8K is valid only for math reasoning / process reward / chain-of-thought claims.
- Legal, robotics, vision-language, safety, agent/tool-use, and formal proof
  claims must use domain benchmarks or a dedicated harness.
- A resolved design must name the literature or official benchmark source for
  every dataset and explain why the dataset count is enough. One dataset is
  acceptable only when the benchmark paper/protocol itself justifies a complete
  multi-suite evaluation for the claim.
- If you are not confident from the given context, set status to
  "literature_review_required" instead of guessing.
"""


def _llm_benchmark_design(
    parsed: Mapping[str, Any],
    method: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not _llm_design_enabled():
        return None
    prompt = {
        "title": parsed.get("title"),
        "problem_statement": parsed.get("problem_statement"),
        "existing_weakness": parsed.get("existing_weakness"),
        "proposed_method": method,
        "current_experimental_plan": plan,
    }
    try:
        result, _tokens = call_llm_json(
            BENCHMARK_DESIGN_SYSTEM,
            json.dumps(prompt, ensure_ascii=False, indent=2)[:16000],
            temperature=0.0,
        )
    except Exception as exc:
        return {
            "status": DESIGN_STATUS_NEEDS_LITERATURE_REVIEW,
            "blockers": [f"benchmark design LLM failed: {exc}"],
        }
    if isinstance(result, dict):
        return result
    return None


def build_benchmark_design_contract(
    parsed: Mapping[str, Any],
    method: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    method = method or {}
    plan = plan or {}
    rule = infer_benchmark_domain(parsed, method, plan)
    llm_design = _llm_benchmark_design(parsed, method, plan)
    llm_status = _text(llm_design.get("status")) if isinstance(llm_design, dict) else ""
    llm_resolved = llm_status == DESIGN_STATUS_RESOLVED

    candidate_benchmarks = list(rule.get("candidate_benchmarks") or [])
    baseline_families = list(rule.get("baseline_families") or [])
    metrics = list(rule.get("metrics") or [])
    domain = _text(rule.get("domain")) or "unknown"
    task_family = _text(rule.get("task_family")) or "unknown"
    blockers: list[str] = []
    warnings: list[str] = []

    if isinstance(llm_design, dict):
        domain = _text(llm_design.get("domain")) or domain
        task_family = _text(llm_design.get("task_family")) or task_family
        if isinstance(llm_design.get("candidate_benchmarks"), list) and llm_design["candidate_benchmarks"]:
            candidate_benchmarks = [dict(row) for row in llm_design["candidate_benchmarks"] if isinstance(row, Mapping)]
        if isinstance(llm_design.get("required_baselines"), list) and llm_design["required_baselines"]:
            baseline_families = _named_values(llm_design["required_baselines"], "name", "method")
        metric = llm_design.get("primary_metric")
        if isinstance(metric, Mapping) and _text(metric.get("name")):
            metrics = [_text(metric.get("name"))] + [item for item in metrics if item != _text(metric.get("name"))]
        blockers.extend(str(item) for item in llm_design.get("blockers") or [] if str(item).strip())

    source = "llm_literature_design" if isinstance(llm_design, dict) else "deterministic_domain_guard"
    annotated_candidates: list[dict[str, Any]] = []
    for row in candidate_benchmarks:
        if not isinstance(row, Mapping):
            continue
        candidate = dict(row)
        candidate.setdefault("dataset_selection_source", source)
        candidate.setdefault("resolver_policy", "validate_only")
        annotated_candidates.append(candidate)
    candidate_benchmarks = annotated_candidates
    benchmark_set_rationale = _benchmark_set_rationale(llm_design if isinstance(llm_design, Mapping) else None)
    minimum_benchmark_count = _minimum_benchmark_count(rule, llm_design if isinstance(llm_design, Mapping) else None)

    matching_rule = dict(rule)
    matching_rule["candidate_benchmarks"] = candidate_benchmarks or list(rule.get("candidate_benchmarks") or [])
    existing_targets = _existing_benchmark_targets(plan)
    matched_existing = [target for target in existing_targets if _target_matches_rule(target, matching_rule)]
    wrong_gsm8k = [
        target for target in existing_targets
        if _is_gsm8k_target(target) and not bool(rule.get("allow_generic_gsm8k"))
    ]

    if wrong_gsm8k:
        msg = "GSM8K is not semantically aligned with this claim; choose a domain benchmark from the literature instead."
        if llm_resolved and candidate_benchmarks:
            warnings.append(msg)
        else:
            blockers.append(msg)

    status = DESIGN_STATUS_RESOLVED
    if BENCHMARK_DESIGN_LLM_REQUIRED:
        if not BENCHMARK_DESIGN_LLM_ENABLED:
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append("Benchmark design LLM is required but disabled; enable it or provide a reviewed benchmark contract.")
        elif not isinstance(llm_design, dict):
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append("Benchmark design must be resolved by LLM literature review before experiment forge.")
        elif llm_status == DESIGN_STATUS_BLOCKED:
            status = DESIGN_STATUS_BLOCKED
        elif not llm_resolved:
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append("Benchmark design LLM did not resolve the domain benchmarks, baselines, and metrics.")
        elif not candidate_benchmarks:
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append("Benchmark design LLM resolved without candidate benchmarks; a literature-backed benchmark set is required.")
    elif domain == "unknown":
        status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
        blockers.append("No benchmark domain could be inferred; a domain literature review is required before experiment forge.")

    malformed_candidates = []
    missing_literature_evidence = []
    selected_count = len(candidate_benchmarks)
    required_axis_markers = [
        _text(item).lower()
        for item in _as_list(rule.get("required_benchmark_axis_markers"))
        if _text(item)
    ]
    single_multi_suite_ok = (
        selected_count == 1
        and minimum_benchmark_count > 1
        and bool(benchmark_set_rationale)
        and _has_multi_suite_justification(candidate_benchmarks[0], benchmark_set_rationale)
    )
    if status == DESIGN_STATUS_RESOLVED:
        for candidate in candidate_benchmarks:
            if not (
                candidate.get("requires_harness")
                or _text(candidate.get("hf_dataset"))
                or candidate.get("direct_files")
                or candidate.get("derive_from_loaded_benchmarks")
            ):
                malformed_candidates.append(_text(candidate.get("name") or candidate.get("hf_dataset") or "benchmark"))
            if not _candidate_has_literature_evidence(candidate):
                missing_literature_evidence.append(_text(candidate.get("name") or candidate.get("hf_dataset") or "benchmark"))
        if malformed_candidates:
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append(
                "LLM benchmark design named benchmarks without an explicit dataset recipe or requires_harness=true: "
                + ", ".join(malformed_candidates[:6])
            )
        if missing_literature_evidence:
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append(
                "Benchmark design lacks per-dataset paper or official benchmark sources for: "
                + ", ".join(missing_literature_evidence[:6])
            )
        if not benchmark_set_rationale:
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append("Benchmark design must explain why the selected dataset count and coverage axes are sufficient for the claim.")
        if selected_count < minimum_benchmark_count and not single_multi_suite_ok:
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append(
                f"Benchmark design selected {selected_count} benchmark(s), but domain protocol requires at least "
                f"{minimum_benchmark_count} unless a benchmark-suite paper justifies single-suite coverage."
            )
        if required_axis_markers and not any(
            any(marker in _candidate_axis_text(candidate) for marker in required_axis_markers)
            for candidate in candidate_benchmarks
        ):
            status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
            blockers.append(
                "Benchmark design is missing a required domain evidence axis: "
                + ", ".join(required_axis_markers[:4])
            )

    if status == DESIGN_STATUS_RESOLVED and wrong_gsm8k and not (llm_resolved and candidate_benchmarks):
        status = DESIGN_STATUS_NEEDS_LITERATURE_REVIEW
    if status == DESIGN_STATUS_RESOLVED and existing_targets and not matched_existing and not bool(rule.get("allow_generic_gsm8k")):
        warnings.append("Existing benchmark targets do not match the inferred domain benchmark family and will be replaced by the design contract.")

    return {
        "schema_version": "benchmark_design_contract_v1",
        "status": status,
        "domain": domain,
        "task_family": task_family,
        "source": source,
        "llm_literature_review_required": status != DESIGN_STATUS_RESOLVED,
        "domain_literature_rationale": (
            llm_design.get("domain_literature_rationale")
            if isinstance(llm_design, dict) and isinstance(llm_design.get("domain_literature_rationale"), list)
            else [
                "Analyze recent domain papers to identify official benchmark protocols, standard baselines, and accepted metrics.",
                "Do not substitute a generic executable QA benchmark for a domain-specific claim.",
            ]
        ),
        "benchmark_set_rationale": benchmark_set_rationale,
        "minimum_benchmark_count": minimum_benchmark_count,
        "candidate_benchmarks": candidate_benchmarks,
        "benchmark_evidence": [
            {
                "name": row.get("name") or row.get("hf_dataset") or "benchmark",
                "benchmark_axis": row.get("benchmark_axis") or row.get("task_type") or "",
                "sources": _candidate_literature_sources(row),
                "requires_harness": bool(row.get("requires_harness")),
                "hf_dataset": row.get("hf_dataset") or "",
                "official_url": row.get("official_url") or row.get("source_url") or row.get("url") or "",
            }
            for row in candidate_benchmarks
        ],
        "required_baseline_families": baseline_families,
        "primary_metric_candidates": metrics,
        "matched_existing_targets": matched_existing,
        "blockers": _unique(blockers),
        "warnings": _unique(warnings),
    }

def apply_benchmark_design_contract(plan: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Attach design contract and block unsafe benchmark substitutions."""

    updated = dict(plan or {})
    updated["benchmark_design_contract"] = dict(contract or {})
    updated["benchmark_design_status"] = _text(contract.get("status")) or DESIGN_STATUS_BLOCKED
    updated["benchmark_evidence"] = list(contract.get("benchmark_evidence") or [])
    blockers = [str(item) for item in contract.get("blockers") or [] if str(item).strip()]
    if updated["benchmark_design_status"] != DESIGN_STATUS_RESOLVED:
        if not blockers:
            blockers = ["Benchmark design requires domain literature review before formal experiment execution."]
        updated["benchmark_design_blockers"] = blockers
        updated["generated_runner_supported"] = False
        updated["benchmark_recipe_blockers"] = [
            {
                "name": "benchmark_literature_review",
                "reason": blocker,
            }
            for blocker in blockers
        ]
        updated["benchmark_harness_deferred"] = False
        return updated

    candidate_targets = [
        dict(row) for row in contract.get("candidate_benchmarks") or [] if isinstance(row, Mapping)
    ]
    if candidate_targets:
        existing = _existing_benchmark_targets(updated)
        selected_targets: list[dict[str, Any]] = []
        if existing:
            for existing_target in existing:
                for candidate in candidate_targets:
                    if _target_matches_rule(existing_target, {"candidate_benchmarks": [candidate]}):
                        merged = {**candidate, **dict(existing_target)}
                        for key in ("requires_harness", "resolver_policy", "dataset_selection_source", "task_type", "hf_dataset", "official_url", "source_url", "literature_sources", "benchmark_axis"):
                            if key in candidate and not merged.get(key):
                                merged[key] = candidate[key]
                        selected_targets.append(merged)
                        break
        if not selected_targets:
            selected_targets = candidate_targets[:4]
        updated["benchmark_targets"] = selected_targets[:4]
        updated["datasets"] = [
            {
                "name": row.get("name") or row.get("hf_dataset"),
                "hf_dataset": row.get("hf_dataset") or "",
                "source": row.get("dataset_selection_source") or row.get("source") or "benchmark_design",
                "benchmark_axis": row.get("benchmark_axis") or row.get("task_type") or "",
                "official_url": row.get("official_url") or row.get("source_url") or row.get("url") or "",
                "literature_sources": row.get("literature_sources") or row.get("benchmark_sources") or [],
                "requires_harness": bool(row.get("requires_harness")),
            }
            for row in selected_targets[:4]
        ]
    updated.pop("benchmark_design_blockers", None)
    return updated

